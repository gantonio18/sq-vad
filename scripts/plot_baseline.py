"""Phase 2 figures — regenerable from results/stgnf_frame_scores.parquet alone.

Produces (results/figs/baseline/):
  roc_baseline.png    ROC curve for the reproduced Global AUC (AUC in title)
  score_hist.png      KDE of canonical s_pose split by gt (separability)
  per_scene_auc.png   per-video Global AUC bar chart (which scenes fail)
  nopose_coverage.png % of zero-pose frames overall vs within anomalous frames
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from common import official_frame_scores, auc, per_video_auc

PARQUET = "results/stgnf_frame_scores.parquet"
OUT = "results/figs/baseline"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    # official reconstructed score (reproduces 85.94)
    y, s, vids = official_frame_scores(df, hr=False)
    global_auc = auc(y, s)
    yhr, shr, _ = official_frame_scores(df, hr=True)
    hr_auc = auc(yhr, shr)
    print("Reconstructed Global AUC = {:.4f} | HR AUC = {:.4f}".format(global_auc, hr_auc))

    # --- roc_baseline.png ---
    fpr, tpr, _ = roc_curve(y, s)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label="STG-NF (frozen)")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("STG-NF baseline ROC — Global AUC = {:.4f}".format(global_auc))
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_baseline.png"), dpi=150); plt.close()

    # --- score_hist.png (canonical raw s_pose, scored frames only) ---
    scored = df.dropna(subset=["s_pose"])
    plt.figure(figsize=(6, 4))
    for lab, name, c in [(0, "normal", "tab:blue"), (1, "anomaly", "tab:red")]:
        sns.kdeplot(scored.loc[scored.gt_label == lab, "s_pose"], label=name,
                    fill=True, alpha=0.4, color=c)
    plt.xlabel("s_pose  (= -normality, higher = more anomalous)")
    plt.title("Canonical pose anomaly score by label (scored frames)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "score_hist.png"), dpi=150); plt.close()

    # --- per_scene_auc.png ---
    pv = per_video_auc(y, s, vids)
    pv_s = pd.Series(pv).sort_values()
    plt.figure(figsize=(12, 4))
    colors = ["tab:red" if a < 0.5 else ("tab:orange" if a < 0.7 else "tab:green")
              for a in pv_s.values]
    plt.bar(range(len(pv_s)), pv_s.values, color=colors)
    plt.axhline(global_auc, color="k", ls="--", lw=1, label="global {:.3f}".format(global_auc))
    plt.axhline(0.5, color="gray", ls=":", lw=1)
    plt.xticks(range(len(pv_s)), pv_s.index, rotation=90, fontsize=6)
    plt.ylabel("per-video ROC-AUC"); plt.ylim(0, 1)
    plt.title("Per-video Global AUC ({} videos with both classes)".format(len(pv_s)))
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_scene_auc.png"), dpi=150); plt.close()

    # --- nopose_coverage.png ---
    tot = len(df); tot_anom = int((df.gt_label == 1).sum())
    pct_overall = 100 * df.is_nopose.mean()
    pct_anom = 100 * df.loc[df.gt_label == 1, "is_nopose"].mean()
    pct_norm = 100 * df.loc[df.gt_label == 0, "is_nopose"].mean()
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["all frames", "normal", "anomalous"],
                   [pct_overall, pct_norm, pct_anom],
                   color=["tab:gray", "tab:blue", "tab:red"])
    for b, v in zip(bars, [pct_overall, pct_norm, pct_anom]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3, "{:.1f}%".format(v), ha="center")
    plt.ylabel("% frames with ZERO detected poses")
    plt.title("STG-NF blind spot: no-pose coverage\n(scored as 'max normal')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nopose_coverage.png"), dpi=150); plt.close()

    print("nopose: overall {:.2f}% | normal {:.2f}% | anomalous {:.2f}%"
          .format(pct_overall, pct_norm, pct_anom))
    print("Wrote 4 figures to {}/".format(OUT))


if __name__ == "__main__":
    main()
