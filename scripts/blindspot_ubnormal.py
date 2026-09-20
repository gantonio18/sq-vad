"""UBnormal blind-spot dissection (reviewer item #5), computed with the SAME
methodology as the ShanghaiTech analysis (scripts/error_analysis.py): reconstruct
the official smoothed per-frame score, take Youden's J on the global ROC, and
report per-frame recall by detected-pose coverage. No RGB frames needed.

Reads results/stgnf_frame_scores_ubnormal.parquet; writes
../Article/figures/ubnormal_blindspot.png and results/ubnormal_blindspot.json.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import official_frame_table, youden_threshold, auc

PARQUET = "results/stgnf_frame_scores_ubnormal.parquet"
OUT_FIG = "../Article/figures/ubnormal_blindspot.png"
OUT_JSON = "results/ubnormal_blindspot.json"


def coverage_bucket(n):
    if n == 0:
        return "no-pose (0)"
    if n == 1:
        return "low (1)"
    if n <= 3:
        return "mid (2-3)"
    return "high (4+)"


ORDER = ["no-pose (0)", "low (1)", "mid (2-3)", "high (4+)"]


def main():
    df = pd.read_parquet(PARQUET)
    tab = official_frame_table(df, hr=False)
    m = df.merge(tab, on=["video_id", "frame_idx"])
    m["bucket"] = m["n_poses_detected"].apply(coverage_bucket)

    y = m["gt_label"].to_numpy()
    s = m["s_official"].to_numpy()
    global_auc = auc(y, s)
    thr = youden_threshold(y, s)
    m["pred_anom"] = (m["s_official"] >= thr).astype(int)
    global_recall = float(m.loc[m.gt_label == 1, "pred_anom"].mean())
    print("Global AUC {:.4f} | Youden thr {:.4f} | global recall {:.3f}".format(
        global_auc, thr, global_recall))

    anom = m[m.gt_label == 1]
    rec = (anom.groupby("bucket")
                .agg(n_anom=("gt_label", "size"), recall=("pred_anom", "mean"))
                .reindex(ORDER).fillna(0))
    print(rec.to_string())

    fn = anom[anom.pred_anom == 0]
    n_fn = len(fn)
    n_fn_blind = int(fn[fn.n_poses_detected <= 1].shape[0])
    pct_fn_blind = 100 * n_fn_blind / max(n_fn, 1)

    # ---- figure (same style as ShanghaiTech failure_buckets) ----
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ORDER))
    ax1.bar(x, rec["n_anom"].values, color="tab:red", alpha=0.65)
    ax1.set_ylabel("# anomalous frames", color="tab:red")
    ax1.set_xticks(x); ax1.set_xticklabels(ORDER)
    for xi, n in zip(x, rec["n_anom"].values):
        ax1.text(xi, n, str(int(n)), ha="center", va="bottom", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, rec["recall"].values, "o-", color="tab:blue", lw=2)
    ax2.set_ylabel("STG-NF recall @Youden", color="tab:blue")
    ax2.set_ylim(0, 1)
    for xi, r in zip(x, rec["recall"].values):
        ax2.text(xi, r + 0.03, "{:.2f}".format(r), ha="center", color="tab:blue", fontsize=9)
    plt.title("UBnormal: STG-NF recall vs detected-pose coverage")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200); plt.close()

    summary = {
        "global_roc_auc": global_auc,
        "youden_thr": thr,
        "global_recall": global_recall,
        "n_anomaly_frames": int(len(anom)),
        "n_false_negatives": n_fn,
        "n_fn_pose_blind_le1": n_fn_blind,
        "pct_fn_pose_blind_le1": round(pct_fn_blind, 1),
        "recall_by_bucket": {b: round(float(rec.loc[b, "recall"]), 3) for b in ORDER},
        "n_anom_by_bucket": {b: int(rec.loc[b, "n_anom"]) for b in ORDER},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSPINE: {}/{} missed anomalies ({:.1f}%) are pose-blind (<=1 pose)."
          .format(n_fn_blind, n_fn, pct_fn_blind))
    print("Wrote", OUT_FIG, "and", OUT_JSON)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
