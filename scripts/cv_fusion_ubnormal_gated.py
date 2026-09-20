"""Scene-profile-gated fusion on UBnormal.

The trusted-category rule assumes a vehicle/bicycle/cart is out of place. On
UBnormal that is scene-dependent. Here we gate the trusted evidence per scene using
the normality profiles induced from NORMAL TRAINING video only
(scene_profiles_ubnormal.py): a category contributes evidence in a scene ONLY if
that scene's profile says the object is NOT normally present there.

No anomaly labels are used anywhere: profiles come from train-split normal clips,
and the fusion hyperparameters are still chosen by cross-fitting.

Usage: python scripts/cv_fusion_ubnormal_gated.py --vlm results/vlm_frame_scores_ubnormal_haiku_v2.parquet --tag haiku_v2
"""
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fuse
from cv_fusion import evidence, best_params

POSE = "results/stgnf_frame_scores_ubnormal.parquet"
PROFILES = "results/ubnormal_scene_profiles.json"
K, N_BOOT, SEED = 5, 2000, 0
CAT_KEY = {"vehicle": "vehicles_normal", "bike": "bicycles_normal", "cart": "carts_normal"}


def build_gate(profiles, scenes, cats):
    """gate[i]=True if category cats[i] is NOT normal in scene scenes[i] (so trustable)."""
    g = np.ones(len(cats), dtype=bool)
    for i, (sc, c) in enumerate(zip(scenes, cats)):
        key = CAT_KEY.get(c)
        if key is None:
            continue  # non-trusted categories unaffected
        prof = profiles.get(str(int(sc)))
        if prof is None:
            continue
        g[i] = not bool(prof.get(key, False))
    return g


def crossfit(gt, pose_sm, s_vlm, cat, vids, idx_by_vid, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(vids))
    folds = np.array_split(perm, K)
    cf = pose_sm.copy(); chosen = []
    for f in range(K):
        test_v = vids[folds[f]]
        train_v = vids[np.concatenate([folds[g] for g in range(K) if g != f])]
        tr_idx = np.concatenate([idx_by_vid[v] for v in train_v])
        trust, tau, lam = best_params(tr_idx, gt, pose_sm, s_vlm, cat)
        chosen.append({"fold": f, "trust": list(trust), "tau": tau, "lambda": lam})
        e = evidence(s_vlm, cat, trust, tau)
        for v in test_v:
            ti = idx_by_vid[v]; cf[ti] = pose_sm[ti] + lam * e[ti]
    deltas = []
    for _ in range(N_BOOT):
        pick = rng.choice(vids, size=len(vids), replace=True)
        ii = np.concatenate([idx_by_vid[v] for v in pick])
        y = gt[ii]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, cf[ii]) - roc_auc_score(y, pose_sm[ii]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return roc_auc_score(gt, cf), (lo, hi), float((deltas > 0).mean()), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    profiles = json.load(open(PROFILES))
    m = fuse.build(pd.read_parquet(POSE), pd.read_parquet(args.vlm),
                   {"vehicle", "bike", "cart"}).reset_index(drop=True)
    m["scene"] = m.video_id.str.extract(r"scene_(\d+)_").astype(int)
    gt = m.gt_label.to_numpy(); pose_sm = m.pose_sm.to_numpy()
    s_vlm = m["s_vlm"].fillna(0.0).to_numpy()
    cat = m["category"].fillna("none").to_numpy()
    scenes = m["scene"].to_numpy()
    vids = np.array(sorted(m.video_id.unique()))
    idx_by_vid = {v: m.index[m.video_id == v].to_numpy() for v in vids}
    base = roc_auc_score(gt, pose_sm)

    gate = build_gate(profiles, scenes, cat)
    cat_gated = np.where(gate, cat, "none")   # gated-out fires become inert

    n_fire = int(np.isin(cat, list(CAT_KEY)).sum())
    n_kept = int((np.isin(cat, list(CAT_KEY)) & gate).sum())
    tp_all = int(((np.isin(cat, list(CAT_KEY))) & (gt == 1)).sum())
    tp_kept = int(((np.isin(cat, list(CAT_KEY))) & gate & (gt == 1)).sum())
    print("baseline {:.4f}".format(base))
    print("trusted fire-frames: {} -> {} after scene gate ({} suppressed)".format(
        n_fire, n_kept, n_fire - n_kept))
    print("   precision ungated {:.2f} | gated {:.2f}".format(
        tp_all / max(n_fire, 1), tp_kept / max(n_kept, 1)))

    res = {"baseline": base, "tag": args.tag,
           "fires_ungated": n_fire, "fires_gated": n_kept,
           "prec_ungated": tp_all / max(n_fire, 1), "prec_gated": tp_kept / max(n_kept, 1)}
    for name, c in [("ungated", cat), ("gated", cat_gated)]:
        auc, (lo, hi), p, chosen = crossfit(gt, pose_sm, s_vlm, c, vids, idx_by_vid)
        print("{:8s}: cross-fit {:.4f}  delta {:+.4f}  CI [{:+.4f},{:+.4f}]  P(>0)={:.3f}  excl0={}".format(
            name, auc, auc - base, lo, hi, p, lo > 0))
        res[name] = {"crossfit": auc, "delta": auc - base, "ci95": [lo, hi],
                     "p_gt0": p, "excludes_zero": bool(lo > 0), "folds": chosen}
    out = "results/cv_fusion_ubnormal_gated_{}.json".format(args.tag)
    json.dump(res, open(out, "w"), indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
