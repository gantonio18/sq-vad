"""Ablation — fully label-free router.

The deployed router's UNCERTAIN tier centres its score band on Youden's J,
which is computed from test-set ground truth. This script replays the
pipeline with a router that uses NO labels anywhere:

  BLIND  : n_poses_detected <= 1              (label-free by construction)
  UNCERT : |s_official - median(s)| <= 0.5*std (median replaces Youden's J)
  SAMPLE : every 50th frame                    (label-free by construction)

The median-centred band retains ~98% of the originally queried uncertain
frames, so the ablation runs entirely from the existing cache (queries the
label-free band would add are simply absent — this only penalises the
label-free variant, so the result is a lower bound). Scores are
re-propagated from the surviving queries (radius 30, as in the run) and
evaluated with the exact cv_fusion.py protocol (same seed, folds, grid).

Writes results/cv_fusion_labelfree.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fuse
from cv_fusion import best_params, evidence
from common import official_frame_table
from score_vlm import propagate

OUT_JSON = "results/cv_fusion_labelfree.json"
QUERIES = "results/vlm_query_results_sonnet.parquet"
RADIUS = 30
K = 5
N_BOOT = 2000
SEED = 0


def main():
    pose = pd.read_parquet(fuse.STGNF)
    tab = official_frame_table(pose, hr=False)
    mm = pose.merge(tab, on=["video_id", "frame_idx"])
    s = mm["s_official"].to_numpy()
    center, band = float(np.median(s)), 0.5 * float(np.std(s))
    print("Label-free band: |s - {:.4f}| <= {:.4f} (median-centred)".format(center, band))

    q = pd.read_parquet(QUERIES).merge(
        mm[["video_id", "frame_idx", "s_official"]], on=["video_id", "frame_idx"], how="left")
    lf = (q.routed_reason.isin(["blind", "sample"])) | \
         ((q.routed_reason == "uncertain") & ((q.s_official - center).abs() <= band))
    q_lf = q[lf]
    print("Queries kept: {}/{} ({})".format(
        len(q_lf), len(q), q_lf.routed_reason.value_counts().to_dict()))

    # re-propagate the surviving queries to per-frame scores (radius as in the run)
    frames = mm[["video_id", "frame_idx"]].copy()
    rows = q_lf[["video_id", "frame_idx", "s_vlm", "category", "reason",
                 "routed_reason"]].to_dict("records")
    vlm_lf = propagate(frames, rows, radius=RADIUS)
    print("VLM-covered frames: {}".format(int(vlm_lf["s_vlm"].notna().sum())))

    m = fuse.build(pose, vlm_lf, {"vehicle", "bike", "cart"})
    m = m.reset_index(drop=True)
    gt = m["gt_label"].to_numpy()
    pose_sm = m["pose_sm"].to_numpy()
    s_vlm = m["s_vlm"].fillna(0.0).to_numpy()
    cat = m["category"].fillna("none").to_numpy()
    vids = np.array(sorted(m["video_id"].unique()))
    idx_by_vid = {v: m.index[m.video_id == v].to_numpy() for v in vids}

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(vids))
    folds = np.array_split(perm, K)

    cf_score = pose_sm.copy()
    chosen = []
    for f in range(K):
        test_vids = vids[folds[f]]
        train_vids = vids[np.concatenate([folds[g] for g in range(K) if g != f])]
        train_idx = np.concatenate([idx_by_vid[v] for v in train_vids])
        trust, tau, lam = best_params(train_idx, gt, pose_sm, s_vlm, cat)
        chosen.append({"fold": f, "trust": list(trust), "tau": tau, "lambda": lam})
        e = evidence(s_vlm, cat, trust, tau)
        for v in test_vids:
            ti = idx_by_vid[v]
            cf_score[ti] = pose_sm[ti] + lam * e[ti]

    base_auc = roc_auc_score(gt, pose_sm)
    cf_auc = roc_auc_score(gt, cf_score)
    print("Baseline Global AUC          : {:.4f}".format(base_auc))
    print("Label-free cross-fit AUC     : {:.4f}  (delta {:+.4f})".format(cf_auc, cf_auc - base_auc))
    for c in chosen:
        print("   fold {}: trust={} tau={} lambda={}".format(c["fold"], c["trust"], c["tau"], c["lambda"]))

    deltas = []
    for _ in range(N_BOOT):
        pick = rng.choice(vids, size=len(vids), replace=True)
        ii = np.concatenate([idx_by_vid[v] for v in pick])
        y = gt[ii]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, cf_score[ii]) - roc_auc_score(y, pose_sm[ii]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    excl = bool(lo > 0)
    p_gt0 = float((deltas > 0).mean())
    print("Video-bootstrap ({}x) delta: mean {:+.4f}, 95% CI [{:+.4f}, {:+.4f}], excludes 0: {}, P(>0)={:.4f}"
          .format(len(deltas), deltas.mean(), lo, hi, excl, p_gt0))

    wins = losses = 0
    for v in vids:
        ti = idx_by_vid[v]
        if len(np.unique(gt[ti])) < 2:
            continue
        a_b = roc_auc_score(gt[ti], pose_sm[ti]); a_f = roc_auc_score(gt[ti], cf_score[ti])
        wins += a_f > a_b + 1e-6; losses += a_f < a_b - 1e-6
    print("per-scene wins/losses: {}/{}".format(int(wins), int(losses)))

    res = {"router": "label-free (blind + sample + median-centred uncertain band)",
           "band_center_median": center, "band_halfwidth": band,
           "queries_kept": int(len(q_lf)), "queries_full": int(len(q)),
           "baseline_global": base_auc, "crossfit_global": cf_auc,
           "crossfit_delta": cf_auc - base_auc,
           "boot_delta_mean": float(deltas.mean()),
           "boot_delta_ci95": [float(lo), float(hi)],
           "boot_excludes_zero": excl, "boot_p_gt0": p_gt0,
           "per_scene_wins": int(wins), "per_scene_losses": int(losses),
           "fold_params": chosen}
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print("Wrote {}".format(OUT_JSON))


if __name__ == "__main__":
    main()
