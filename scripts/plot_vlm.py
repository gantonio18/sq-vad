"""Phase 4 figures (results/figs/vlm/), from the VLM parquets + router log:
  vlm_score_hist.png        s_vlm distribution split by gt (queried frames)
  vlm_category_breakdown.png VLM categories, split by true normal/anomaly
  router_coverage.png       candidate vs queried fraction + cost estimate
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUERY = "results/vlm_query_results.parquet"
STGNF = "results/stgnf_frame_scores.parquet"
ROUTER_LOG = "results/vlm_router_log.csv"
OUT = "results/figs/vlm"
PRICE_PER_QUERY = 0.00105


def main():
    os.makedirs(OUT, exist_ok=True)
    q = pd.read_parquet(QUERY)
    gt = pd.read_parquet(STGNF)[["video_id", "frame_idx", "gt_label"]]
    q = q.merge(gt, on=["video_id", "frame_idx"], how="left")
    q = q[q["s_vlm"].notna()]

    # --- vlm_score_hist.png ---
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    for lab, name, c in [(0, "normal", "tab:blue"), (1, "anomaly", "tab:red")]:
        plt.hist(q.loc[q.gt_label == lab, "s_vlm"], bins=bins, alpha=0.55,
                 label=name, color=c, density=True)
    plt.xlabel("s_vlm (VLM anomaly probability)"); plt.ylabel("density")
    plt.title("VLM score by true label (queried frames)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "vlm_score_hist.png"), dpi=150); plt.close()

    # --- vlm_category_breakdown.png ---
    ct = q.groupby(["category", "gt_label"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=[0, 1], fill_value=0)
    ct["tot"] = ct.sum(axis=1); ct = ct.sort_values("tot", ascending=False)
    plt.figure(figsize=(8, 4))
    x = np.arange(len(ct))
    plt.bar(x, ct[0], color="tab:blue", label="normal")
    plt.bar(x, ct[1], bottom=ct[0], color="tab:red", label="anomaly")
    plt.xticks(x, ct.index, rotation=40, ha="right", fontsize=8)
    plt.ylabel("# queried frames"); plt.title("VLM category vs true label")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "vlm_category_breakdown.png"), dpi=150); plt.close()

    # --- router_coverage.png ---
    try:
        log = pd.read_csv(ROUTER_LOG)
        n = len(log); cand = int(log.candidate.sum()); qd = int(log.queried.sum())
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(["frames", "candidates", "queried"], [n, cand, qd],
                      color=["tab:gray", "tab:orange", "tab:green"])
        for b, v in zip(bars, [n, cand, qd]):
            ax.text(b.get_x() + b.get_width() / 2, v, "{}\n{:.1f}%".format(v, 100 * v / n),
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# frames")
        ax.set_title("Router coverage — {} queries (~${:.2f}), {:.1f}% of frames"
                     .format(qd, qd * PRICE_PER_QUERY, 100 * qd / n))
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "router_coverage.png"), dpi=150); plt.close()
    except FileNotFoundError:
        print("(router log not found; skipping router_coverage.png)")

    print("Wrote VLM figures to {}/  | queried {} frames, {} parsed".format(
        OUT, len(q), int(q.s_vlm.notna().sum())))


if __name__ == "__main__":
    main()
