"""Qualitative case-study figures for one test video, from cached results only.

Produces (in ../Article/figures/):
  timeline_queries_<vid>.png  two panels: baseline vs fused score + VLM query track
  anomaly_sequence_<vid>.png  frame strip with green/red GT markers (optional)

Usage:
  python scripts/case_study.py 01_0054
  python scripts/case_study.py 10_0037 --strip 96 120 192 216
All times are true video seconds (frame / 24 fps).
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FUSED = "results/fused_frame_scores.parquet"
QUERIES = "results/vlm_query_results_sonnet.parquet"
FRAMES = "../../Datasets/shanghaitech/testing/frames"
OUTDIR = "../Article/figures"
FPS = 24.0
TRUST = {"vehicle", "bike", "cart"}


def plain_timeline(vid, marks=None, title_note=""):
    """Single-panel score plot; marks = strip frame indices to dot (green/red by GT)."""
    f = pd.read_parquet(FUSED)
    g = f[f.video_id == vid].sort_values("frame_idx")
    x = g.frame_idx.to_numpy() / FPS
    gt = g.gt_label.to_numpy()
    gmap = g.set_index("frame_idx")["gt_label"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(x, 0, 1, where=gt == 1, transform=ax.get_xaxis_transform(),
                    color="red", alpha=0.10, label="ground-truth anomaly")
    ax.plot(x, g.pose_sm, color="tab:blue", lw=1.6, label="STG-NF (frozen baseline)")
    ax.plot(x, g.s_f2, color="tab:orange", lw=1.6, label="fused (ours)")
    if marks:
        for fr in marks:
            an = int(gmap.loc[fr])
            ax.axvline(fr / FPS, color="0.55", ls=":", lw=1)
            ax.plot(fr / FPS, ax.get_ylim()[1] * 0.97, marker="o", ms=9,
                    color=("tab:red" if an else "tab:green"), clip_on=False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("anomaly score")
    ax.set_title("Test video {}".format(vid))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "timeline_{}.png".format(vid))
    plt.savefig(out, dpi=170)
    plt.close()
    print("saved", out)


def timeline(vid):
    f = pd.read_parquet(FUSED)
    g = f[f.video_id == vid].sort_values("frame_idx")
    q = pd.read_parquet(QUERIES)
    qg = q[q.video_id == vid].sort_values("frame_idx")
    x = g.frame_idx.to_numpy() / FPS
    gt = g.gt_label.to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.1], "hspace": 0.08})
    ax1.fill_between(x, 0, 1, where=gt == 1, transform=ax1.get_xaxis_transform(),
                     color="red", alpha=0.10, label="ground-truth anomaly")
    ax1.plot(x, g.pose_sm, color="tab:blue", lw=1.6, label="STG-NF (frozen baseline)")
    ax1.plot(x, g.s_f2, color="tab:orange", lw=1.6, label="fused (ours)")
    ax1.set_ylabel("anomaly score")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Test video {}".format(vid))

    ax2.fill_between(x, 0, 1, where=gt == 1, transform=ax2.get_xaxis_transform(),
                     color="red", alpha=0.10)
    qx = qg.frame_idx.to_numpy() / FPS
    cat = qg.category.fillna("none").to_numpy()
    trusted = np.isin(cat, list(TRUST))
    sv = qg.s_vlm.to_numpy()
    ax2.vlines(qx, 0, sv, color=np.where(trusted, "tab:red", "0.6"), lw=1.2)
    ax2.scatter(qx[~trusted], sv[~trusted], s=26, color="0.5", zorder=3,
                label="VLM answer: none / normal")
    lbl = "VLM answer: " + "/".join(sorted(set(cat[trusted]))) if trusted.any() else "trusted"
    ax2.scatter(qx[trusted], sv[trusted], s=30, color="tab:red", zorder=3, label=lbl)
    ax2.set_ylim(-0.06, 1.12)
    ax2.set_ylabel("VLM score")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", fontsize=8.5)
    out = os.path.join(OUTDIR, "timeline_queries_{}.png".format(vid))
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", out)


def strip(vid, frames):
    f = pd.read_parquet(FUSED)
    g = f[f.video_id == vid].set_index("frame_idx")
    tiles = []
    for fr in frames:
        an = int(g.loc[fr].gt_label)
        img = cv2.imread(os.path.join(FRAMES, vid, "{:03d}.jpg".format(fr)))
        h, w = img.shape[:2]
        col = (40, 40, 220) if an else (60, 180, 60)
        cv2.circle(img, (46, 46), 30, (255, 255, 255), -1)
        cv2.circle(img, (46, 46), 26, col, -1)
        cv2.rectangle(img, (0, h - 42), (200, h), (0, 0, 0), -1)
        cv2.putText(img, "t = {:.1f} s".format(fr / FPS), (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
        print("frame {} ({:.1f}s): gt={}".format(fr, fr / FPS, an))
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    row = []
    for t in tiles:
        row += [t, gap]
    out = os.path.join(OUTDIR, "anomaly_sequence_{}.png".format(vid))
    cv2.imwrite(out, np.hstack(row[:-1]))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--strip", type=int, nargs="*", default=None,
                    help="frame indices for the strip (omit to skip)")
    ap.add_argument("--marks", type=int, nargs="*", default=None,
                    help="frame indices to dot in the plain timeline")
    args = ap.parse_args()
    timeline(args.video)
    plain_timeline(args.video, marks=args.marks or args.strip)
    if args.strip:
        strip(args.video, args.strip)
