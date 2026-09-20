"""Phase 2 — reproduce the frozen STG-NF baseline AND dump per-frame scores.

Runs the SAME flow/eval as train_eval.py (no training, frozen checkpoint), then:
  1. reproduces the official Global AUC (and HR AUC) via the repo's own
     `score_dataset`, so the number is provably comparable to the paper, and
  2. dumps, for EVERY frame of EVERY test clip:
        video_id, frame_idx, stgnf_normality, s_pose, n_poses_detected,
        is_nopose, gt_label
     to results/stgnf_frame_scores.parquet.

Canonical sign (see SCORING.md): s_pose = -normality  ->  higher = MORE anomalous.
gt_label: 1 = anomaly, 0 = normal (the raw ShanghaiTech mask, BEFORE the repo flip).

Run from the repo root (the GT path in scoring_utils is hardcoded relative to cwd):
    python scripts/dump_baseline_scores.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# repo root = parent of this script's dir; ensure imports + relative paths work
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader, shanghaitech_hr_skip, SHANGHAITECH_HR_SKIP
from utils.train_utils import init_model_params
from utils.scoring_utils import score_dataset

CKPT = "checkpoints/ShanghaiTech_85_9.tar"
OUT_PARQUET = "results/stgnf_frame_scores.parquet"
OUT_JSON = "results/baseline_auc.json"


def build_args(dataset):
    parser = init_parser()
    args = parser.parse_args([
        "--dataset", dataset,
        "--checkpoint", CKPT,
        "--num_workers", "0",
        "--seed", "999",
    ])
    args, model_args = init_sub_args(args)
    args.ckpt_dir = None
    return args, model_args


def run_inference(args, model_args):
    """Replicate train_eval.py up to normality_scores (frozen, eval-only)."""
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    trainer = Trainer(args, model, loader["train"], loader["test"],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    trainer.load_checkpoint(CKPT)
    normality_scores = trainer.test()
    return normality_scores, dataset["test"].metadata


def per_frame_dump(normality_scores, metadata, args, seg_len):
    """Re-derive per-frame normality + n_poses_detected, mirroring get_clip_score
    but WITHOUT the inf->max fill, and additionally counting detected poses."""
    metadata_np = np.array(metadata)
    gt_root = "data/ShanghaiTech/gt/test_frame_mask/"
    clip_list = sorted(fn for fn in os.listdir(gt_root) if fn.endswith(".npy"))

    rows = []
    for clip in clip_list:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split(".")[0].split("_")]
        clip_gt = np.load(os.path.join(gt_root, clip))          # 1=anomaly,0=normal (raw)
        n_frames = clip_gt.shape[0]

        clip_inds = np.where((metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id))[0]
        person_ids = sorted(set(int(metadata_np[i, 2]) for i in clip_inds))

        # per-person normality buffer, inf = "not scored at this frame"
        if len(person_ids) == 0:
            buffers = {0: np.full(n_frames, np.inf)}
        else:
            buffers = {pid: np.full(n_frames, np.inf) for pid in person_ids}
            for pid in person_ids:
                p_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id) &
                                  (metadata_np[:, 2] == pid))[0]
                frame_starts = np.array([metadata[i][3] for i in p_inds]).astype(int)
                buffers[pid][frame_starts + seg_len // 2] = normality_scores[p_inds]

        stack = np.stack(list(buffers.values()))                # (P, n_frames)
        n_poses = np.isfinite(stack).sum(axis=0)                # poses scored per frame
        normality = np.amin(stack, axis=0)                      # repo's min-normality
        is_nopose = ~np.isfinite(normality)                     # no person scored -> inf

        vid = "{:02d}_{:04d}".format(scene_id, clip_id)
        for t in range(n_frames):
            norm_t = normality[t]
            rows.append({
                "video_id": vid,
                "frame_idx": t,
                "scene_id": scene_id,
                "clip_id": clip_id,
                "stgnf_normality": np.nan if not np.isfinite(norm_t) else float(norm_t),
                "s_pose": np.nan if not np.isfinite(norm_t) else float(-norm_t),
                "n_poses_detected": int(n_poses[t]),
                "is_nopose": bool(is_nopose[t]),
                "gt_label": int(clip_gt[t]),                    # 1=anomaly
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    # ----- official Global AUC (reuse the repo's own scoring path) -----
    args_g, margs_g = build_args("ShanghaiTech")
    normality_scores, metadata = run_inference(args_g, margs_g)
    global_auc, _ = score_dataset(np.copy(normality_scores), metadata, args=args_g)

    # ----- official HR AUC (same scores, HR skip list) -----
    args_hr, _ = build_args("ShanghaiTech-HR")
    hr_auc, _ = score_dataset(np.copy(normality_scores), metadata, args=args_hr)

    print("\n=== Official (repo score_dataset) ===")
    print("Global ROC-AUC : {:.4f}".format(global_auc))
    print("HR     ROC-AUC : {:.4f}".format(hr_auc))

    # ----- per-frame dump (canonical s_pose, n_poses, nopose flags) -----
    df = per_frame_dump(normality_scores, metadata, args_g, seg_len=args_g.seg_len)

    # sanity: anomaly frames should have higher mean s_pose than normal frames
    scored = df.dropna(subset=["s_pose"])
    mean_anom = scored.loc[scored.gt_label == 1, "s_pose"].mean()
    mean_norm = scored.loc[scored.gt_label == 0, "s_pose"].mean()
    assert mean_anom > mean_norm, (
        "Sign convention broken: anomaly mean s_pose {:.3f} !> normal {:.3f}"
        .format(mean_anom, mean_norm))
    print("Sign check OK: mean s_pose anomaly {:.3f} > normal {:.3f}".format(mean_anom, mean_norm))

    df.to_parquet(OUT_PARQUET, index=False)
    print("Wrote {} ({} frames, {} clips)".format(OUT_PARQUET, len(df), df.video_id.nunique()))

    n_nopose = int(df.is_nopose.sum())
    n_nopose_anom = int(df[(df.is_nopose) & (df.gt_label == 1)].shape[0])
    n_anom = int((df.gt_label == 1).sum())
    summary = {
        "global_roc_auc": float(global_auc),
        "hr_roc_auc": float(hr_auc),
        "n_frames": int(len(df)),
        "n_clips": int(df.video_id.nunique()),
        "n_anomaly_frames": n_anom,
        "n_nopose_frames": n_nopose,
        "pct_nopose_frames": round(100 * n_nopose / len(df), 3),
        "n_nopose_anomaly_frames": n_nopose_anom,
        "pct_of_anomalies_that_are_nopose": round(100 * n_nopose_anom / max(n_anom, 1), 3),
        "hr_skip_clips": [list(c) for c in SHANGHAITECH_HR_SKIP],
        "checkpoint": CKPT,
        "seg_len": int(args_g.seg_len),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote {}".format(OUT_JSON))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
