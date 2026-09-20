"""Phase 4 — the router: decides which frames go to the VLM (brief §4.1).

Querying every frame is wasteful and noisy. The router selects frames where the
VLM can actually help, informed by the Phase-3 finding that STG-NF recall
collapses as pose coverage drops (and 25% of misses are <=1-pose):

  Tier 1  BLIND      : n_poses_detected <= 1            (always — the blind spot)
  Tier 2  UNCERTAIN  : |s_official - youden_thr| <= band (near the boundary)
  Tier 4  SAMPLE     : sparse uniform frames            (calibration / continuity)

Temporal dedup: among candidates, keep representatives spaced >= query_stride
apart; each queried score is later propagated by score_vlm to its +/- stride/2
window. Every per-frame decision is logged to results/vlm_router_log.csv.

Usage (module): worklist, log = build_worklist(df, **kw)
Usage (CLI):    python scripts/router.py [--query_stride 12 --sample_every 50 ...]
"""
import os
import argparse
import numpy as np
import pandas as pd

from common import official_frame_table, youden_threshold

PARQUET = "results/stgnf_frame_scores.parquet"
LOG_CSV = "results/vlm_router_log.csv"


def build_worklist(df, blind_stride=12, other_stride=36, band=None,
                   sample_every=50, blind_max_poses=1, max_queries=None,
                   limit_videos=None, seed=0):
    """Per-tier temporal dedup: BLIND frames (the structural blind spot) are
    queried densely (blind_stride); the abundant UNCERTAIN/SAMPLE frames are
    thinned (other_stride). Blind takes priority when a frame is both."""
    tab = official_frame_table(df, hr=False)
    m = df.merge(tab, on=["video_id", "frame_idx"])
    thr = youden_threshold(m["gt_label"].to_numpy(), m["s_official"].to_numpy())
    if band is None:
        band = 0.5 * float(np.std(m["s_official"]))

    m = m.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    m["tier_blind"] = m["n_poses_detected"] <= blind_max_poses
    m["tier_uncert"] = (m["s_official"] - thr).abs() <= band
    m["tier_sample"] = (m["frame_idx"] % sample_every) == 0
    m["candidate"] = m["tier_blind"] | m["tier_uncert"] | m["tier_sample"]

    videos = sorted(m["video_id"].unique())
    if limit_videos:
        videos = videos[:limit_videos]

    queried_idx = []
    routed_reason = {}
    for vid in videos:
        g = m[m.video_id == vid]
        last_blind = -10 ** 9
        last_other = -10 ** 9
        for _, r in g.iterrows():
            if r.tier_blind:
                if (r.frame_idx - last_blind) >= blind_stride:
                    queried_idx.append(r.name)
                    routed_reason[r.name] = "blind"
                    last_blind = r.frame_idx
            elif (r.tier_uncert or r.tier_sample) and (r.frame_idx - last_other) >= other_stride:
                queried_idx.append(r.name)
                routed_reason[r.name] = "uncertain" if r.tier_uncert else "sample"
                last_other = r.frame_idx

    m["queried"] = False
    m.loc[queried_idx, "queried"] = True
    m["routed_reason"] = m.index.map(lambda i: routed_reason.get(i, ""))
    m = m[m.video_id.isin(videos)].copy()

    worklist = m[m.queried][["video_id", "frame_idx", "routed_reason",
                             "n_poses_detected", "s_official", "gt_label"]].copy()

    # optional cap (smoke test) — keep a spread across reasons & videos
    if max_queries and len(worklist) > max_queries:
        worklist = (worklist.sample(frac=1.0, random_state=seed)
                            .sort_values(["routed_reason"])
                            .head(max_queries)
                            .sort_values(["video_id", "frame_idx"]))

    log = m[["video_id", "frame_idx", "n_poses_detected", "s_official", "gt_label",
             "tier_blind", "tier_uncert", "tier_sample", "candidate",
             "queried", "routed_reason"]]
    meta = {"youden_thr": round(thr, 4), "band": round(band, 4),
            "blind_stride": blind_stride, "other_stride": other_stride,
            "sample_every": sample_every, "n_videos": len(videos),
            "propagate_radius": other_stride // 2}
    return worklist, log, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--blind_stride", type=int, default=12)
    ap.add_argument("--other_stride", type=int, default=36)
    ap.add_argument("--band", type=float, default=None)
    ap.add_argument("--sample_every", type=int, default=50)
    ap.add_argument("--blind_max_poses", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--out_log", default=LOG_CSV)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    worklist, log, meta = build_worklist(
        df, blind_stride=args.blind_stride, other_stride=args.other_stride,
        band=args.band, sample_every=args.sample_every,
        blind_max_poses=args.blind_max_poses,
        max_queries=args.max_queries, limit_videos=args.limit_videos)
    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    log.to_csv(args.out_log, index=False)

    n_frames = len(log)
    print("Router meta:", meta)
    print("Frames considered : {}".format(n_frames))
    print("Candidates        : {} ({:.1f}%)".format(int(log.candidate.sum()),
          100 * log.candidate.mean()))
    print("QUERIES (worklist): {} ({:.1f}% of frames)".format(len(worklist),
          100 * len(worklist) / n_frames))
    print("  by reason       :", worklist.routed_reason.value_counts().to_dict())
    print("Wrote log -> {}".format(args.out_log))


if __name__ == "__main__":
    main()
