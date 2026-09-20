"""Phase 6 figures (results/figs/fusion/), regenerable from the parquets alone:
  auc_vs_w.png      F1 convex AUC vs w (naive contrast) + F2 lambda sweep
  roc_overlay.png   ROC for STG-NF / VLM / F1 / F2 / F3 (the money plot)
  score_scatter.png z(pose) vs trusted VLM evidence, coloured by gt
  gate_effect.png   no-pose rule on vs off (F2)

The per-frame fused columns are already final (pose smoothed + VLM evidence), so
they are used directly here (no further smoothing).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from common import auc

FUSED = "results/fused_frame_scores.parquet"
METRICS = "results/fusion_metrics.json"
OUT = "results/figs/fusion"


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_parquet(FUSED)
    met = json.load(open(METRICS))
    y = m["gt_label"].to_numpy()

    streams = [
        ("STG-NF", m["zp"].to_numpy()),
        ("VLM (all cats)", m["e_all"].to_numpy()),
        ("F1 convex (w={:.2f})".format(met["f1_best_w"]), m["s_f1"].to_numpy()),
        ("F2 category-aware (lam={:.2f})".format(met["f2_best_lambda"]), m["s_f2"].to_numpy()),
        ("F3 max", m["s_f3"].to_numpy()),
    ]

    # --- roc_overlay.png ---
    plt.figure(figsize=(6, 6))
    for name, s in streams:
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, lw=1.8, label="{}  AUC={:.4f}".format(name, auc(y, s)))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC overlay — STG-NF vs VLM vs fusion (ShanghaiTech Global)")
    plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_overlay.png"), dpi=150); plt.close()

    # --- auc_vs_w.png (F1 convex sweep) + F2 lambda sweep inset-style ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sw = met["f1_w_sweep"]
    ws = sorted(float(k) for k in sw); a1 = [sw[str(w)] for w in ws]
    axes[0].plot(ws, a1, "o-"); axes[0].axhline(met["baseline_global"], color="k", ls="--", lw=1)
    axes[0].set_xlabel("w (VLM, all cats)"); axes[0].set_ylabel("Global AUC")
    axes[0].set_title("F1 naive convex fusion (best w={:.2f})".format(met["f1_best_w"]))
    sl = met["f2_lambda_sweep"]
    ls = sorted(float(k) for k in sl); a2 = [sl[str(l)] for l in ls]
    axes[1].plot(ls, a2, "o-", color="tab:green")
    axes[1].axhline(met["baseline_global"], color="k", ls="--", lw=1, label="baseline")
    axes[1].scatter([met["f2_best_lambda"]], [met["f2_global"]], color="red", zorder=5,
                    label="best {:.4f}".format(met["f2_global"]))
    axes[1].set_xlabel("lambda (trusted VLM)"); axes[1].set_ylabel("Global AUC")
    axes[1].set_title("F2 category-aware additive (PRIMARY)"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "auc_vs_w.png"), dpi=150); plt.close()

    # --- score_scatter.png ---
    samp = m.sample(min(8000, len(m)), random_state=0)
    plt.figure(figsize=(6, 5))
    for lab, name, c in [(0, "normal", "tab:blue"), (1, "anomaly", "tab:red")]:
        sub = samp[samp.gt_label == lab]
        jit = sub["e_trust"] + np.random.uniform(-0.01, 0.01, len(sub))
        plt.scatter(sub["zp"], jit, s=6, alpha=0.3, color=c, label=name)
    plt.xlabel("z(s_pose)  STG-NF"); plt.ylabel("trusted VLM evidence (vehicle/bike/cart)")
    plt.title("Complementarity: VLM fires on object anomalies at low pose-score")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "score_scatter.png"), dpi=150); plt.close()

    # --- gate_effect.png (no-pose rule) ---
    vals = [met["baseline_global"], met["f2_global"], met.get("f2_global_nopose_on", met["f2_global"])]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["baseline", "F2 (no-pose rule OFF)", "F2 + no-pose rule ON"], vals,
                   color=["tab:gray", "tab:green", "tab:orange"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, "{:.4f}".format(v), ha="center", va="bottom", fontsize=8)
    plt.ylabel("Global ROC-AUC"); plt.ylim(min(vals) - 0.01, max(vals) + 0.005)
    plt.title("No-pose rule ablation\n(replacing max-normal fill with VLM HURTS:\nno-pose frames are mostly normal)")
    plt.xticks(rotation=8, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gate_effect.png"), dpi=150); plt.close()

    print("Wrote 4 fusion figures to {}/".format(OUT))


if __name__ == "__main__":
    main()
