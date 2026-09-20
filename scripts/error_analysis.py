"""Phase 3 — failure-mode / error analysis (offline, the justification engine).

Quantifies WHERE the frozen STG-NF misses anomalies, organised by pose
availability (the structural axis that determines STG-NF's blind spot). Since
ShanghaiTech ships no per-anomaly-type labels, we operationalise the brief's
A/B/C/D buckets by detected-pose coverage, which is exactly what makes the flow
blind. Outputs:
  results/figs/error_analysis/failure_buckets.png
  results/figs/error_analysis/score_vs_npose.png
  results/figs/error_analysis/fn_gallery/montage.png
  error_analysis.md  (the spine sentence: % of missed anomalies that are pose-blind)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from common import official_frame_table, youden_threshold, auc

import paths

PARQUET = "results/stgnf_frame_scores.parquet"
FRAMES_ROOT = paths.SHANGHAITECH_FRAMES
OUT = "results/figs/error_analysis"
MD = "error_analysis.md"

# pose-coverage buckets (the structural proxy for the brief's A/B/C/D)
def coverage_bucket(n):
    if n == 0:
        return "C: no-pose (0)"
    if n == 1:
        return "B: low (1)"
    if n <= 3:
        return "mid (2-3)"
    return "A: high (4+)"

BUCKET_ORDER = ["C: no-pose (0)", "B: low (1)", "mid (2-3)", "A: high (4+)"]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "fn_gallery"), exist_ok=True)
    df = pd.read_parquet(PARQUET)

    tab = official_frame_table(df, hr=False)
    m = df.merge(tab, on=["video_id", "frame_idx"], suffixes=("", "_t"))
    # gt_label identical in both; keep df's
    m["bucket"] = m["n_poses_detected"].apply(coverage_bucket)

    y = m["gt_label"].to_numpy()
    s = m["s_official"].to_numpy()
    thr = youden_threshold(y, s)
    m["pred_anom"] = (m["s_official"] >= thr).astype(int)
    print("Global AUC {:.4f} | Youden thr {:.4f} | global recall {:.3f}".format(
        auc(y, s), thr, m.loc[m.gt_label == 1, "pred_anom"].mean()))

    anom = m[m.gt_label == 1]

    # ---- bucket table: count + recall ----
    rec = (anom.groupby("bucket")
                .agg(n_anom=("gt_label", "size"),
                     recall=("pred_anom", "mean"))
                .reindex(BUCKET_ORDER).fillna(0))
    print("\nAnomaly frames & STG-NF recall by pose-coverage bucket:")
    print(rec.to_string())

    # ---- failure_buckets.png ----
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(BUCKET_ORDER))
    ax1.bar(x, rec["n_anom"].values, color="tab:red", alpha=0.65)
    ax1.set_ylabel("# anomalous frames", color="tab:red")
    ax1.set_xticks(x); ax1.set_xticklabels(BUCKET_ORDER, rotation=15, fontsize=8)
    for xi, n in zip(x, rec["n_anom"].values):
        ax1.text(xi, n, str(int(n)), ha="center", va="bottom", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, rec["recall"].values, "o-", color="tab:blue", lw=2)
    ax2.set_ylabel("STG-NF recall @Youden", color="tab:blue")
    ax2.set_ylim(0, 1)
    for xi, r in zip(x, rec["recall"].values):
        ax2.text(xi, r + 0.03, "{:.2f}".format(r), ha="center", color="tab:blue", fontsize=8)
    plt.title("STG-NF blind spot: anomaly count & recall vs pose coverage")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "failure_buckets.png"), dpi=150); plt.close()

    # ---- score_vs_npose.png ----
    plt.figure(figsize=(7, 4.5))
    samp = m.sample(min(8000, len(m)), random_state=0)
    jit = samp["n_poses_detected"] + np.random.uniform(-0.25, 0.25, len(samp))
    for lab, name, c in [(0, "normal", "tab:blue"), (1, "anomaly", "tab:red")]:
        sub = samp[samp.gt_label == lab]
        plt.scatter(jit[sub.index], sub["s_official"], s=4, alpha=0.25,
                    color=c, label=name)
    plt.axhline(thr, color="k", ls="--", lw=1, label="Youden thr")
    plt.xlabel("n_poses_detected (jittered)")
    plt.ylabel("s_official  (higher = more anomalous)")
    plt.title("Anomaly score collapses where poses are absent/sparse")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "score_vs_npose.png"), dpi=150); plt.close()

    # ---- worst false negatives (pose present, confidently scored normal) ----
    fn = anom[(anom.pred_anom == 0)].copy()
    # most damaging: lowest anomaly score among missed anomalies, prefer pose-present
    fn_pose = fn[fn.n_poses_detected > 0].sort_values("s_official")
    # representative across videos: one lowest per video, then take 16 lowest
    rep = fn_pose.sort_values("s_official").groupby("video_id", as_index=False).first()
    rep = rep.sort_values("s_official").head(16)
    montage_path = build_montage(rep, os.path.join(OUT, "fn_gallery", "montage.png"))

    # ---- spine numbers ----
    n_anom = len(anom)
    n_fn = len(fn)
    n_fn_noposeblind = int(fn[fn.n_poses_detected <= 1].shape[0])   # no/low pose
    pct_fn_blind = 100 * n_fn_noposeblind / max(n_fn, 1)
    rec_high = float(rec.loc["A: high (4+)", "recall"])
    rec_nopose = float(rec.loc["C: no-pose (0)", "recall"])

    write_md(rec, thr, n_anom, n_fn, n_fn_noposeblind, pct_fn_blind,
             rec_high, rec_nopose, montage_path)
    print("\nSPINE: {}/{} missed anomalies ({:.1f}%) are pose-blind (<=1 pose)."
          .format(n_fn_noposeblind, n_fn, pct_fn_blind))
    print("Wrote figures + {}".format(MD))


def build_montage(rep, out_path, cols=4, cell=240):
    imgs = []
    for _, r in rep.iterrows():
        fp = os.path.join(FRAMES_ROOT, r["video_id"], "{:03d}.jpg".format(int(r["frame_idx"])))
        img = cv2.imread(fp)
        if img is None:
            img = np.zeros((cell, cell, 3), np.uint8)
        img = cv2.resize(img, (cell, cell))
        label = "{} f{} np{} s{:.2f}".format(r["video_id"], int(r["frame_idx"]),
                                             int(r["n_poses_detected"]), r["s_official"])
        cv2.rectangle(img, (0, 0), (cell, 16), (0, 0, 0), -1)
        cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        imgs.append(img)
    if not imgs:
        return None
    rows_n = (len(imgs) + cols - 1) // cols
    grid = np.zeros((rows_n * cell, cols * cell, 3), np.uint8)
    for i, im in enumerate(imgs):
        rr, cc = divmod(i, cols)
        grid[rr * cell:(rr + 1) * cell, cc * cell:(cc + 1) * cell] = im
    cv2.imwrite(out_path, grid)
    return out_path


def write_md(rec, thr, n_anom, n_fn, n_fn_blind, pct_blind, rec_high, rec_nopose, montage):
    lines = []
    lines.append("# error_analysis.md — Where the frozen STG-NF fails (Phase 3)\n")
    lines.append("Operating point: Youden's J on the global ROC, threshold `s_official "
                 ">= {:.4f}`.\n".format(thr))
    lines.append("Buckets approximate the brief's A/B/C/D by **detected-pose coverage**, "
                 "the structural axis that makes the flow blind (ShanghaiTech ships no "
                 "per-type labels). The VLM's Phase-4 `category` output will let us "
                 "cross-tabulate by true type later.\n")
    lines.append("## Anomaly frames & STG-NF recall by pose coverage\n")
    lines.append("| Bucket | # anomaly frames | STG-NF recall |")
    lines.append("|---|---|---|")
    for b in rec.index:
        lines.append("| {} | {} | {:.3f} |".format(b, int(rec.loc[b, "n_anom"]), rec.loc[b, "recall"]))
    lines.append("")
    lines.append("## The spine sentence\n")
    lines.append("> Of **{}** missed anomalous frames (false negatives at the Youden point), "
                 "**{} ({:.1f}%)** occur where STG-NF is structurally blind (≤1 detected "
                 "pose) — exactly the frames a VLM appearance/context stream can address.\n"
                 .format(n_fn, n_fn_blind, pct_blind))
    lines.append("Recall is **{:.2f}** on high-pose-coverage anomalies (4+ people) but "
                 "**{:.2f}** on no-pose anomalies — the latter are silently scored as "
                 "*max-normal* by the `+inf`→max fill (see SCORING.md).\n"
                 .format(rec_high, rec_nopose))
    lines.append("## Artifacts\n")
    lines.append("- `results/figs/error_analysis/failure_buckets.png`")
    lines.append("- `results/figs/error_analysis/score_vs_npose.png`")
    if montage:
        lines.append("- `{}` — montage of worst pose-present false negatives".format(montage.replace("\\", "/")))
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
