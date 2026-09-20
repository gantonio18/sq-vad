"""Shared helpers so every figure/eval is regenerable from results/*.parquet alone.

The key function `official_frame_scores` reconstructs STG-NF's exact per-frame
anomaly score (inf-fill + iterated Gaussian smoothing) from the raw per-frame
parquet, so downstream scripts never need to re-run the flow. It reproduces the
repo's 85.94% Global AUC bit-for-bit (asserted in dump_baseline_scores.py).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]


def smooth_clip(arr, sigma=7):
    """Replicate utils.scoring_utils.smooth_scores for one clip (sigma 1..6)."""
    out = arr.astype(float).copy()
    for sig in range(1, sigma):
        out = gaussian_filter1d(out, sigma=sig)
    return out


def official_frame_scores(df, hr=False):
    """Return (y, s_anom, video_ids) reproducing the repo's scoring path.

    df: results/stgnf_frame_scores.parquet (one row per frame, sorted-clip order).
        Uses columns: video_id, scene_id, clip_id, frame_idx, stgnf_normality
        (NaN at no-pose frames), gt_label (1=anomaly).
    hr: if True, drop the 6 HR-skip clips (matches --dataset ShanghaiTech-HR).

    s_anom = -normality after inf-fill+smoothing  (higher = more anomalous),
    aligned with y (1 = anomaly).  roc_auc_score(y, s_anom) == repo Global/HR AUC.
    """
    d = df.copy()
    if hr:
        mask = ~d.apply(lambda r: (int(r.scene_id), int(r.clip_id)) in SHANGHAITECH_HR_SKIP, axis=1)
        d = d[mask]

    # global inf-fill value = max finite normality across the (included) set
    finite = d["stgnf_normality"].to_numpy(dtype=float)
    global_max = np.nanmax(finite)

    ys, ss, vids = [], [], []
    for vid, g in d.groupby("video_id", sort=True):
        g = g.sort_values("frame_idx")
        norm = g["stgnf_normality"].to_numpy(dtype=float)
        norm = np.where(np.isfinite(norm), norm, global_max)   # no-pose -> max normal
        norm = smooth_clip(norm)                               # per-clip smoothing
        ys.append(g["gt_label"].to_numpy(dtype=int))
        ss.append(-norm)                                       # anomaly = -normality
        vids.append(np.full(len(g), vid, dtype=object))
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    v = np.concatenate(vids)
    return y, s, v


def official_frame_table(df, hr=False):
    """Like official_frame_scores but returns a tidy DataFrame keyed by
    (video_id, frame_idx) with columns: gt_label, s_official. Mergeable back
    onto the raw parquet (which carries n_poses_detected / is_nopose)."""
    d = df.copy()
    if hr:
        keep = ~d.apply(lambda r: (int(r.scene_id), int(r.clip_id)) in SHANGHAITECH_HR_SKIP, axis=1)
        d = d[keep]
    global_max = np.nanmax(d["stgnf_normality"].to_numpy(dtype=float))
    rows = []
    for vid, g in d.groupby("video_id", sort=True):
        g = g.sort_values("frame_idx")
        norm = g["stgnf_normality"].to_numpy(dtype=float)
        norm = np.where(np.isfinite(norm), norm, global_max)
        s = -smooth_clip(norm)
        rows.append(pd.DataFrame({
            "video_id": vid,
            "frame_idx": g["frame_idx"].to_numpy(),
            "s_official": s,   # gt_label intentionally omitted (parquet already has it)
        }))
    return pd.concat(rows, ignore_index=True)


def infilled_pose_table(df, hr=False):
    """Per-frame UNSMOOTHED pose stream for fusion. Reproduces STG-NF's inf-fill
    (no-pose -> global max finite normality) but does NOT smooth, so callers can
    fuse first and smooth ONCE afterwards (brief §6.1 / compass §E step 5).

    Returns df with: video_id, frame_idx, gt_label, normality_filled, is_nopose.
    Anchor: smooth_by_clip(-normality_filled) then AUC == repo baseline (asserted
    in fuse.py).
    """
    d = df.copy()
    if hr:
        keep = ~d.apply(lambda r: (int(r.scene_id), int(r.clip_id)) in SHANGHAITECH_HR_SKIP, axis=1)
        d = d[keep]
    gmax = np.nanmax(d["stgnf_normality"].to_numpy(dtype=float))
    norm = d["stgnf_normality"].to_numpy(dtype=float)
    d = d.assign(normality_filled=np.where(np.isfinite(norm), norm, gmax))
    return d[["video_id", "frame_idx", "gt_label", "normality_filled", "is_nopose"]]


def smooth_by_clip(frame_df, value_col, out_col=None):
    """Apply STG-NF's iterated Gaussian smoothing (sigma 1..6) per video, in
    sorted (video, frame) order. Returns a copy with the smoothed column."""
    out_col = out_col or (value_col + "_sm")
    d = frame_df.sort_values(["video_id", "frame_idx"]).copy()
    parts = []
    for _, g in d.groupby("video_id", sort=True):
        parts.append(smooth_clip(g[value_col].to_numpy(dtype=float)))
    d[out_col] = np.concatenate(parts)
    return d


def youden_threshold(y, s):
    """Operating point maximizing TPR - FPR (Youden's J) on the global ROC."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def auc(y, s):
    return float(roc_auc_score(y, s))


def per_video_auc(y, s, vids):
    """AUC per video (only videos containing both classes)."""
    out = {}
    df = pd.DataFrame({"y": y, "s": s, "v": vids})
    for vid, g in df.groupby("v"):
        if g.y.nunique() == 2:
            out[vid] = float(roc_auc_score(g.y, g.s))
    return out
