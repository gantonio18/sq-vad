"""Probe: does a DOMAIN-ADAPTIVE trusted set fix the transfer?

The deployed fusion hard-codes T={vehicle,bike,cart}, chosen on ShanghaiTech where
non-pose anomalies are object intrusions. UBnormal's anomalies are behavioural, and
the VLM is measurably accurate on those categories (fall ~1.00, run ~0.78) -- but
they are never trusted because they are not in the candidate grid.

Here we widen the candidate grid to include event categories and let the SAME
cross-fitting choose per domain. Nothing else changes. Run on cached scores only.
"""
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fuse

TAUS = [0.0, 0.5, 0.7]
LAMBDAS = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]
OBJ = ("vehicle", "bike", "cart")
GRID_OBJ = [("vehicle",), ("vehicle", "bike"), OBJ, OBJ + ("skateboard",)]
GRID_WIDE = GRID_OBJ + [
    ("fall",), ("run",), ("fall", "run"), ("fall", "run", "throw"),
    OBJ + ("fall",), OBJ + ("run",), OBJ + ("fall", "run"),
    OBJ + ("fall", "run", "skateboard", "throw"),
]


def evidence(s, cat, trust, tau):
    return np.where(np.isin(cat, list(trust)) & (s >= tau), s, 0.0)


def auc_on(idx, gt, sc):
    y = gt[idx]
    return None if len(np.unique(y)) < 2 else roc_auc_score(y, sc[idx])


def best_params(idx, gt, pose, s, cat, grid):
    base = auc_on(idx, gt, pose)
    best = (base, (OBJ, 0.0, 0.0))
    for trust, tau, lam in itertools.product(grid, TAUS, LAMBDAS):
        a = auc_on(idx, gt, pose + lam * evidence(s, cat, trust, tau))
        if a is not None and a > best[0]:
            best = (a, (trust, tau, lam))
    return best[1]


def run(pose_p, vlm_p, grid, k=5, n_boot=2000, seed=0):
    m = fuse.build(pd.read_parquet(pose_p), pd.read_parquet(vlm_p), set(OBJ)).reset_index(drop=True)
    gt = m.gt_label.to_numpy(); pose = m.pose_sm.to_numpy()
    s = m["s_vlm"].fillna(0.0).to_numpy(); cat = m["category"].fillna("none").to_numpy()
    vids = np.array(sorted(m.video_id.unique()))
    idx = {v: m.index[m.video_id == v].to_numpy() for v in vids}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids)); folds = np.array_split(perm, k)
    cf = pose.copy(); chosen = []
    for f in range(k):
        tr = np.concatenate([idx[v] for v in vids[np.concatenate([folds[g] for g in range(k) if g != f])]])
        trust, tau, lam = best_params(tr, gt, pose, s, cat, grid)
        chosen.append(list(trust))
        e = evidence(s, cat, trust, tau)
        for v in vids[folds[f]]:
            cf[idx[v]] = pose[idx[v]] + lam * e[idx[v]]
    base = roc_auc_score(gt, pose); cfa = roc_auc_score(gt, cf)
    d = []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        ii = np.concatenate([idx[v] for v in pick]); y = gt[ii]
        if len(np.unique(y)) < 2:
            continue
        d.append(roc_auc_score(y, cf[ii]) - roc_auc_score(y, pose[ii]))
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    return {"base": base, "cf": cfa, "delta": cfa - base, "ci": [lo, hi],
            "excl0": bool(lo > 0), "p": float((d > 0).mean()), "folds": chosen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True); ap.add_argument("--vlm", required=True)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    for name, grid in [("object-only grid (deployed)", GRID_OBJ), ("WIDE grid (+events)", GRID_WIDE)]:
        r = run(a.pose, a.vlm, grid)
        print("[{}] {:28s} base {:.4f} -> {:.4f}  delta {:+.4f}  CI [{:+.4f},{:+.4f}]  P={:.3f}  excl0={}".format(
            a.tag, name, r["base"], r["cf"], r["delta"], r["ci"][0], r["ci"][1], r["p"], r["excl0"]))
        print("      folds chose:", r["folds"])


if __name__ == "__main__":
    main()
