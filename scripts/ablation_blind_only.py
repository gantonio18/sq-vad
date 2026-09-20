"""Ablation — label-free router: blind-tier evidence only.

The full router has three tiers; the UNCERTAIN tier centres its band on
Youden's J computed from test-set ground truth, which is a (mild) label
leak in frame selection. The BLIND tier (n_poses_detected <= 1) needs no
labels at all. This script reruns the exact cv_fusion.py protocol (same
seed, same folds, same hyperparameter grid) with the VLM evidence
restricted to frames covered by blind-tier queries, i.e. what a fully
label-free router would have produced (970 of the 1,520 queries).

If the cross-fitted delta survives, the headline result does not depend
on the label-informed uncertain tier. Reads only cached parquets — no
VLM re-query. Writes results/cv_fusion_blindonly.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fuse
from cv_fusion import best_params, evidence

OUT_JSON = "results/cv_fusion_blindonly.json"
K = 5
N_BOOT = 2000
SEED = 0


def main():
    vlm = pd.read_parquet(fuse.VLM)
    n_before = int(vlm["s_vlm"].notna().sum())
    keep = vlm["routed_reason"] == "blind"
    vlm.loc[~keep, "s_vlm"] = np.nan
    vlm.loc[~keep, "category"] = None
    n_after = int(vlm["s_vlm"].notna().sum())
    print("VLM-covered frames: {} -> {} (blind tier only)".format(n_before, n_after))

    m = fuse.build(pd.read_parquet(fuse.STGNF), vlm, {"vehicle", "bike", "cart"})
    m = m.reset_index(drop=True)
    gt = m["gt_label"].to_numpy()
    pose_sm = m["pose_sm"].to_numpy()
    s_vlm = m["s_vlm"].fillna(0.0).to_numpy()
    cat = m["category"].fillna("none").to_numpy()
    vids = np.array(sorted(m["video_id"].unique()))
    idx_by_vid = {v: m.index[m.video_id == v].to_numpy() for v in vids}

    # identical fold construction to cv_fusion.py (same rng seed)
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
    print("Baseline Global AUC        : {:.4f}".format(base_auc))
    print("Blind-only cross-fit AUC   : {:.4f}  (delta {:+.4f})".format(cf_auc, cf_auc - base_auc))
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

    res = {"router": "blind tier only (label-free)",
           "n_covered_frames_full": n_before, "n_covered_frames_blind": n_after,
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
