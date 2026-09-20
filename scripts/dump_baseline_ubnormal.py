"""UBnormal transfer (reviewer item #5) — reproduce the frozen STG-NF UBnormal
baseline AND dump per-frame scores, mirroring dump_baseline_scores.py but for the
UBnormal test set (seg_len 16, unsupervised checkpoint, CPU).

Emits results/stgnf_frame_scores_ubnormal.parquet with the SAME schema as the
ShanghaiTech dump so scripts/common.py + the blind-spot analysis work unchanged:
  video_id, frame_idx, scene_id, clip_id, stgnf_normality (NaN at no-pose),
  s_pose, n_poses_detected, is_nopose, gt_label (1=anomaly).

Anchors to the repo's own score_dataset AUC (~0.7178). Run from repo root:
    python scripts/dump_baseline_ubnormal.py
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch  # noqa: E402
from models.STG_NF.model_pose import STG_NF  # noqa: E402
from models.training import Trainer  # noqa: E402
from utils.data_utils import trans_list  # noqa: E402
from utils.optim_init import init_optimizer, init_scheduler  # noqa: E402
from args import init_parser, init_sub_args  # noqa: E402
from dataset import get_dataset_and_loader  # noqa: E402
from utils.train_utils import init_model_params  # noqa: E402
from utils.scoring_utils import score_dataset  # noqa: E402

CKPT = "checkpoints/UBnormal_unsupervised_71_8.tar"
SEG_LEN = 16
GT_ROOT = "data/UBnormal/gt/"
POSE_TEST_ROOT = "data/UBnormal/pose/test"
OUT_PARQUET = "results/stgnf_frame_scores_ubnormal.parquet"
OUT_JSON = "results/baseline_auc_ubnormal.json"


def build_args():
    parser = init_parser()
    args = parser.parse_args([
        "--dataset", "UBnormal",
        "--seg_len", str(SEG_LEN),
        "--device", "cpu",
        "--checkpoint", CKPT,
        "--num_workers", "0",
        "--seed", "999",
    ])
    args, model_args = init_sub_args(args)
    args.ckpt_dir = None
    return args, model_args


def run_inference(args):
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    trainer = Trainer(args, model, loader["train"], loader["test"],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    trainer.load_checkpoint(CKPT)
    normality_scores = trainer.test()
    return normality_scores, dataset["test"].metadata


def per_frame_dump(normality_scores, metadata, seg_len):
    """Replicate utils.scoring_utils.get_clip_score (UBnormal branch) per clip,
    additionally counting detected poses and keeping raw (pre-fill) normality."""
    metadata_np = np.array(metadata, dtype=object)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "tracks.txt")
        for fn in os.listdir(POSE_TEST_ROOT) if fn.endswith(".json"))

    rows = []
    for clip in clip_list:
        typ, scene_id, cid = re.findall(
            r"(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*", clip)[0]
        clip_id = typ + "_" + cid  # string id, exactly as scoring_utils builds it
        clip_gt = np.load(os.path.join(GT_ROOT, clip))  # UBnormal: 1=normal, 0=anomaly
        n_frames = clip_gt.shape[0]

        clip_inds = np.where((metadata_np[:, 1] == clip_id) &
                             (metadata_np[:, 0] == scene_id))[0]
        person_ids = sorted(set(metadata_np[i, 2] for i in clip_inds))

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

        stack = np.stack(list(buffers.values()))
        n_poses = np.isfinite(stack).sum(axis=0)
        normality = np.amin(stack, axis=0)
        is_nopose = ~np.isfinite(normality)

        vid = clip.replace("_tracks.txt", "")
        for t in range(n_frames):
            norm_t = normality[t]
            rows.append({
                "video_id": vid,
                "frame_idx": t,
                "scene_id": int(scene_id),
                "clip_id": clip_id,
                "stgnf_normality": np.nan if not np.isfinite(norm_t) else float(norm_t),
                "s_pose": np.nan if not np.isfinite(norm_t) else float(-norm_t),
                "n_poses_detected": int(n_poses[t]),
                "is_nopose": bool(is_nopose[t]),
                "gt_label": int(1 - clip_gt[t]),   # -> 1=anomaly
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    args, _ = build_args()
    normality_scores, metadata = run_inference(args)

    # anchor: repo's own scoring path
    global_auc, _ = score_dataset(np.copy(normality_scores), metadata, args=args)
    print("\n=== UBnormal (repo score_dataset) Global ROC-AUC: {:.4f} ===".format(global_auc))

    df = per_frame_dump(normality_scores, metadata, seg_len=SEG_LEN)

    scored = df.dropna(subset=["s_pose"])
    mean_anom = scored.loc[scored.gt_label == 1, "s_pose"].mean()
    mean_norm = scored.loc[scored.gt_label == 0, "s_pose"].mean()
    print("Sign check: mean s_pose anomaly {:.3f} vs normal {:.3f}".format(mean_anom, mean_norm))
    assert mean_anom > mean_norm, "sign/label convention broken"

    df.to_parquet(OUT_PARQUET, index=False)
    n_nopose = int(df.is_nopose.sum())
    n_anom = int((df.gt_label == 1).sum())
    n_nopose_anom = int(df[(df.is_nopose) & (df.gt_label == 1)].shape[0])
    summary = {
        "global_roc_auc": float(global_auc),
        "n_frames": int(len(df)),
        "n_clips": int(df.video_id.nunique()),
        "n_anomaly_frames": n_anom,
        "pct_anomaly_frames": round(100 * n_anom / len(df), 3),
        "n_nopose_frames": n_nopose,
        "pct_nopose_frames": round(100 * n_nopose / len(df), 3),
        "n_nopose_anomaly_frames": n_nopose_anom,
        "pct_of_anomalies_that_are_nopose": round(100 * n_nopose_anom / max(n_anom, 1), 3),
        "checkpoint": CKPT,
        "seg_len": SEG_LEN,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote {} ({} frames, {} clips)".format(OUT_PARQUET, len(df), df.video_id.nunique()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
