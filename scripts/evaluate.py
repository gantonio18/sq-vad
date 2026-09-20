"""Phase 7 — evaluation protocol, comparison table & significance.

  results/comparison.csv + results/figs/comparison_table.png
        Global & HR AUC for STG-NF / VLM-only / F1 / F2 / F3.
  results/figs/fusion/auc_bootstrap.png + results/bootstrap.json
        Video-resampled bootstrap (1000x) 95% CI for (best fused - baseline);
        reports whether the delta CI excludes 0, plus per-scene wins/losses.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from common import per_video_auc

FUSED = "results/fused_frame_scores.parquet"
METRICS = "results/fusion_metrics.json"
OUT_CSV = "results/comparison.csv"
OUT_TABLE = "results/figs/comparison_table.png"
OUT_BOOT = "results/figs/fusion/auc_bootstrap.png"
OUT_BOOT_JSON = "results/bootstrap.json"


def comparison_table(met):
    rows = [
        ["STG-NF (frozen baseline)", met["baseline_global"], met["baseline_hr"], "reproduced"],
        ["VLM only", met["vlm_only_global"], "-", "standalone"],
        ["F1 late fusion (best w={:.2f})".format(met["f1_best_w"]),
         met["f1_best_global"], met["f1_best_hr"], ""],
        ["F2 gated fusion", met["f2_global"], met["f2_hr"], "PRIMARY"],
        ["F3 max fusion", met["f3_global"], met["f3_hr"], ""],
    ]
    df = pd.DataFrame(rows, columns=["System", "Global AUC", "HR AUC", "Notes"])
    df.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    ax.axis("off")
    disp = df.copy()
    for c in ["Global AUC", "HR AUC"]:
        disp[c] = disp[c].apply(lambda v: "{:.4f}".format(v) if isinstance(v, (int, float)) else v)
    t = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.4)
    base = met["baseline_global"]
    for i in range(len(df)):
        try:
            if float(df.iloc[i]["Global AUC"]) > base:
                t[(i + 1, 1)].set_facecolor("#d6f5d6")
        except (ValueError, TypeError):
            pass
    plt.title("ShanghaiTech — system comparison", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT_TABLE, dpi=150, bbox_inches="tight"); plt.close()
    return df


def bootstrap(m, best_col, n_boot=1000, seed=0):
    """Resample videos with replacement; CI for (best - baseline) Global AUC.
    Uses the already-final per-frame scores (zp = smoothed pose; s_f* = fused)."""
    base = pd.DataFrame({"video_id": m["video_id"], "y": m["gt_label"],
                         "base": m["zp"].to_numpy(), "fused": m[best_col].to_numpy()})
    groups = {v: g for v, g in base.groupby("video_id")}
    vids = list(groups)
    rng = np.random.default_rng(seed)
    d_base, d_fused, deltas = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        cat = pd.concat([groups[v] for v in pick])
        if cat["y"].nunique() < 2:
            continue
        ab = roc_auc_score(cat["y"], cat["base"])
        af = roc_auc_score(cat["y"], cat["fused"])
        d_base.append(ab); d_fused.append(af); deltas.append(af - ab)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    res = {"n_boot": len(deltas), "delta_mean": float(deltas.mean()),
           "delta_ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0),
           "p_delta_gt_0": float((deltas > 0).mean()),
           "best_stream": best_col}
    # per-scene wins/losses (point estimate)
    pv_b = per_video_auc(base["y"].to_numpy(), base["base"].to_numpy(), base["video_id"].to_numpy())
    pv_f = per_video_auc(base["y"].to_numpy(), base["fused"].to_numpy(), base["video_id"].to_numpy())
    wins = sum(1 for v in pv_b if v in pv_f and pv_f[v] > pv_b[v] + 1e-6)
    losses = sum(1 for v in pv_b if v in pv_f and pv_f[v] < pv_b[v] - 1e-6)
    res["per_scene_wins"] = wins; res["per_scene_losses"] = losses
    res["per_scene_total"] = len(pv_b)

    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=40, color="tab:green", alpha=0.7)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.axvline(lo, color="red", ls=":", lw=1); plt.axvline(hi, color="red", ls=":", lw=1)
    plt.xlabel("Global AUC delta  (best fused - baseline)")
    plt.ylabel("bootstrap count")
    plt.title("Video-bootstrap delta: mean {:+.4f}, 95% CI [{:+.4f}, {:+.4f}]\n{}"
              .format(deltas.mean(), lo, hi,
                      "CI EXCLUDES 0 (significant)" if lo > 0 else "CI includes 0"))
    plt.tight_layout(); plt.savefig(OUT_BOOT, dpi=150); plt.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    met = json.load(open(METRICS))
    m = pd.read_parquet(FUSED)

    df = comparison_table(met)
    print(df.to_string(index=False))

    best_col = max([("s_f1", met["f1_best_global"]), ("s_f2", met["f2_global"]),
                    ("s_f3", met["f3_global"])], key=lambda t: t[1])[0]
    res = bootstrap(m, best_col, n_boot=args.n_boot)
    json.dump(res, open(OUT_BOOT_JSON, "w"), indent=2)
    print("\nBootstrap ({} iters), best stream = {}".format(res["n_boot"], best_col))
    print("  delta mean {:+.4f}, 95% CI [{:+.4f}, {:+.4f}], excludes 0: {}".format(
        res["delta_mean"], res["delta_ci95"][0], res["delta_ci95"][1], res["excludes_zero"]))
    print("  per-scene wins/losses: {}/{} of {}".format(
        res["per_scene_wins"], res["per_scene_losses"], res["per_scene_total"]))
    print("Wrote {}, {}, {}".format(OUT_CSV, OUT_TABLE, OUT_BOOT))


if __name__ == "__main__":
    main()
