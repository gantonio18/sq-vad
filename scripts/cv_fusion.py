"""Phase 7b — overfitting-free fusion evaluation via K-fold cross-fitting.

The +0.0083 headline tuned (trust set, lambda) on the full test set. Here we
remove that concern: partition the 107 videos into K folds; for each fold, choose
hyperparameters (trust categories, VLM confidence threshold tau, weight lambda)
on the OTHER folds only, then score the held-out fold with them. Every video is
thus scored by params chosen without seeing it. We pool the cross-fitted per-frame
scores and report one honest Global AUC + a video-bootstrap CI for the delta.

Also reports a single 50/50 val->test split for transparency.

Reads only results/*.parquet (no VLM re-query). Writes results/cv_fusion.json
and results/figs/fusion/cv_bootstrap.png.
"""
import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import fuse   # reuse build() for pose_sm + merged table

OUT_JSON = "results/cv_fusion.json"
OUT_FIG = "results/figs/fusion/cv_bootstrap.png"

# hyperparameter grid searched on the tuning split
TRUST_SETS = [("vehicle",), ("vehicle", "bike"), ("vehicle", "bike", "cart"),
              ("vehicle", "bike", "cart", "skateboard")]
TAUS = [0.0, 0.5, 0.7, 0.8, 0.9]
LAMBDAS = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]


def evidence(s_vlm, category, trust, tau):
    mask = np.isin(category, list(trust)) & (s_vlm >= tau)
    return np.where(mask, s_vlm, 0.0)


def auc_on(idx, gt, score):
    y = gt[idx]
    if len(np.unique(y)) < 2:
        return None
    return roc_auc_score(y, score[idx])


def best_params(train_idx, gt, pose_sm, s_vlm, cat):
    base = auc_on(train_idx, gt, pose_sm)
    best = (base, (("vehicle", "bike", "cart"), 0.0, 0.0))  # lam=0 -> baseline fallback
    for trust, tau, lam in itertools.product(TRUST_SETS, TAUS, LAMBDAS):
        e = evidence(s_vlm, cat, trust, tau)
        a = auc_on(train_idx, gt, pose_sm + lam * e)
        if a is not None and a > best[0]:
            best = (a, (trust, tau, lam))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm", default=fuse.VLM)
    ap.add_argument("--pose", default=fuse.STGNF)
    # NB: default writes the headline results/cv_fusion.json + figure. Always pass
    # --out/--fig when scoring an alternative evidence stream (e.g. the detector
    # ablation), or the primary numbers get silently overwritten.
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--fig", default=OUT_FIG)
    args = ap.parse_args()

    print("VLM scores: {}".format(args.vlm))
    m = fuse.build(pd.read_parquet(args.pose), pd.read_parquet(args.vlm),
                   {"vehicle", "bike", "cart"})
    m = m.reset_index(drop=True)
    gt = m["gt_label"].to_numpy()
    pose_sm = m["pose_sm"].to_numpy()
    s_vlm = m["s_vlm"].fillna(0.0).to_numpy()
    cat = m["category"].fillna("none").to_numpy()
    vids = np.array(sorted(m["video_id"].unique()))
    idx_by_vid = {v: m.index[m.video_id == v].to_numpy() for v in vids}

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(vids))
    folds = np.array_split(perm, args.k)

    # ---- K-fold cross-fitting ----
    cf_score = pose_sm.copy()       # cross-fitted fused score per frame
    chosen = []
    for f in range(args.k):
        test_vids = vids[folds[f]]
        train_vids = vids[np.concatenate([folds[g] for g in range(args.k) if g != f])]
        train_idx = np.concatenate([idx_by_vid[v] for v in train_vids])
        trust, tau, lam = best_params(train_idx, gt, pose_sm, s_vlm, cat)
        chosen.append({"fold": f, "trust": list(trust), "tau": tau, "lambda": lam})
        e = evidence(s_vlm, cat, trust, tau)
        for v in test_vids:
            ti = idx_by_vid[v]
            cf_score[ti] = pose_sm[ti] + lam * e[ti]

    base_auc = roc_auc_score(gt, pose_sm)
    cf_auc = roc_auc_score(gt, cf_score)
    print("Baseline Global AUC      : {:.4f}".format(base_auc))
    print("Cross-fitted fused AUC   : {:.4f}  (delta {:+.4f})".format(cf_auc, cf_auc - base_auc))
    print("Per-fold chosen params   :")
    for c in chosen:
        print("   fold {}: trust={} tau={} lambda={}".format(c["fold"], c["trust"], c["tau"], c["lambda"]))

    # ---- video bootstrap on the cross-fitted scores ----
    deltas = []
    for _ in range(args.n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        ii = np.concatenate([idx_by_vid[v] for v in pick])
        y = gt[ii]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, cf_score[ii]) - roc_auc_score(y, pose_sm[ii]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    excl = bool(lo > 0)
    print("\nVideo-bootstrap ({}x) delta: mean {:+.4f}, 95% CI [{:+.4f}, {:+.4f}], excludes 0: {}"
          .format(len(deltas), deltas.mean(), lo, hi, excl))
    print("  P(delta>0) = {:.3f}".format((deltas > 0).mean()))

    # per-scene wins/losses on cross-fitted scores
    wins = losses = 0
    for v in vids:
        ti = idx_by_vid[v]
        if len(np.unique(gt[ti])) < 2:
            continue
        a_b = roc_auc_score(gt[ti], pose_sm[ti]); a_f = roc_auc_score(gt[ti], cf_score[ti])
        wins += a_f > a_b + 1e-6; losses += a_f < a_b - 1e-6
    print("  per-scene wins/losses: {}/{}".format(int(wins), int(losses)))

    # ---- single 50/50 val->test split (transparency) ----
    half = len(vids) // 2
    val_vids, test_vids = vids[perm[:half]], vids[perm[half:]]
    val_idx = np.concatenate([idx_by_vid[v] for v in val_vids])
    test_idx = np.concatenate([idx_by_vid[v] for v in test_vids])
    trust, tau, lam = best_params(val_idx, gt, pose_sm, s_vlm, cat)
    e = evidence(s_vlm, cat, trust, tau)
    split_base = auc_on(test_idx, gt, pose_sm)
    split_fused = auc_on(test_idx, gt, pose_sm + lam * e)
    print("\n50/50 split: tuned on val (trust={} tau={} lam={}) -> TEST baseline {:.4f}, fused {:.4f} ({:+.4f})"
          .format(list(trust), tau, lam, split_base, split_fused, split_fused - split_base))

    res = {"baseline_global": base_auc, "crossfit_global": cf_auc,
           "crossfit_delta": cf_auc - base_auc,
           "boot_delta_mean": float(deltas.mean()),
           "boot_delta_ci95": [float(lo), float(hi)], "boot_excludes_zero": excl,
           "boot_p_gt0": float((deltas > 0).mean()),
           "per_scene_wins": int(wins), "per_scene_losses": int(losses),
           "fold_params": chosen,
           "split_test_baseline": split_base, "split_test_fused": split_fused,
           "split_params": {"trust": list(trust), "tau": tau, "lambda": lam}}
    json.dump(res, open(args.out, "w"), indent=2)

    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=40, color="tab:green", alpha=0.75)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.axvline(lo, color="red", ls=":", lw=1); plt.axvline(hi, color="red", ls=":", lw=1)
    plt.xlabel("cross-fitted Global AUC delta (fused - baseline)")
    plt.ylabel("bootstrap count")
    plt.title("Cross-fitted delta: mean {:+.4f}, 95% CI [{:+.4f},{:+.4f}]\n{}"
              .format(deltas.mean(), lo, hi, "EXCLUDES 0 (significant)" if excl else "includes 0"))
    plt.tight_layout(); plt.savefig(args.fig, dpi=150); plt.close()
    print("Wrote {} and {}".format(args.out, args.fig))


if __name__ == "__main__":
    main()
