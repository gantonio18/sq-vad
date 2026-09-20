"""UBnormal fusion transfer eval (reviewer #5) — reuses the exact cross-fitting +
video-bootstrap machinery of cv_fusion.py, pointed at the UBnormal pose/VLM
parquets, writing to separate files so the ShanghaiTech results are untouched.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import fuse
from cv_fusion import evidence, auc_on, best_params

POSE = "results/stgnf_frame_scores_ubnormal.parquet"
VLM = "results/vlm_frame_scores_ubnormal.parquet"
OUT_JSON = "results/cv_fusion_ubnormal.json"
OUT_FIG = "../Article/figures/ubnormal_cv_bootstrap.png"
K, N_BOOT, SEED = 5, 2000, 0


def main():
    m = fuse.build(pd.read_parquet(POSE), pd.read_parquet(VLM),
                   {"vehicle", "bike", "cart"}).reset_index(drop=True)
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
    print("Baseline {:.4f} | cross-fitted {:.4f} (delta {:+.4f})".format(
        base_auc, cf_auc, cf_auc - base_auc))
    for c in chosen:
        print("  fold {}: {}".format(c["fold"], c))

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
    print("bootstrap delta mean {:+.4f}, 95% CI [{:+.4f},{:+.4f}], excl0={}, P(>0)={:.3f}".format(
        deltas.mean(), lo, hi, excl, (deltas > 0).mean()))

    wins = losses = 0
    for v in vids:
        ti = idx_by_vid[v]
        if len(np.unique(gt[ti])) < 2:
            continue
        if roc_auc_score(gt[ti], cf_score[ti]) > roc_auc_score(gt[ti], pose_sm[ti]) + 1e-6:
            wins += 1
        elif roc_auc_score(gt[ti], cf_score[ti]) < roc_auc_score(gt[ti], pose_sm[ti]) - 1e-6:
            losses += 1

    # in-sample F2 with fixed trust, best lambda on a coarse grid (report only)
    best = (base_auc, 0.0)
    for lam in [0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 5.0]:
        e = evidence(s_vlm, cat, ("vehicle", "bike", "cart"), 0.0)
        a = roc_auc_score(gt, pose_sm + lam * e)
        if a > best[0]:
            best = (a, lam)
    insample_auc, insample_lam = best

    res = {"baseline_global": base_auc, "crossfit_global": cf_auc,
           "crossfit_delta": cf_auc - base_auc,
           "insample_global": insample_auc, "insample_lambda": insample_lam,
           "boot_delta_mean": float(deltas.mean()),
           "boot_delta_ci95": [float(lo), float(hi)], "boot_excludes_zero": excl,
           "boot_p_gt0": float((deltas > 0).mean()),
           "per_scene_wins": int(wins), "per_scene_losses": int(losses),
           "n_vlm_covered": int(m["s_vlm"].notna().sum()),
           "trusted_fire_frames": int(((np.isin(cat, ["vehicle", "bike", "cart"])) & (s_vlm > 0)).sum()),
           "fold_params": chosen}
    json.dump(res, open(OUT_JSON, "w"), indent=2)

    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=40, color="tab:purple", alpha=0.75)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.axvline(lo, color="red", ls=":", lw=1); plt.axvline(hi, color="red", ls=":", lw=1)
    plt.xlabel("cross-fitted Global AUC delta (fused - baseline)")
    plt.ylabel("bootstrap count")
    plt.title("UBnormal (Haiku) delta: mean {:+.4f}, 95% CI [{:+.4f},{:+.4f}]\n{}".format(
        deltas.mean(), lo, hi, "excludes 0" if excl else "includes 0"))
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150); plt.close()
    print("\nWrote", OUT_JSON, "and", OUT_FIG)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
