"""Phase 4 — run router -> VLM -> per-frame VLM score parquet (brief §4.4).

Loads each queried frame (optionally with a motion overlay), calls the cached
VLMClient, then PROPAGATES each queried score to its +/- query_stride/2 window so
every frame gets an s_vlm (NaN where uncovered). Writes:
  results/vlm_query_results.parquet  one row per actual VLM query (audit)
  results/vlm_frame_scores.parquet   per-frame s_vlm/category/reason (for fusion)

Motion overlay (--motion_overlay):
  none  : the raw RGB frame
  stack : temporal RGB stack (gray of t-2g, t-g, t in B,G,R) -> fast motion
          shows as coloured fringing, helping the VLM judge speed (brief §4.3).

Smoke test (tiny, ~$0.01):
  python scripts/score_vlm.py --max_queries 12 --motion_overlay stack
Full run (after the cost gate):
  python scripts/score_vlm.py --motion_overlay stack
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import router
from vlm_client import VLMClient

import paths

FRAMES_ROOT = paths.SHANGHAITECH_FRAMES
OUT_FRAME = "results/vlm_frame_scores.parquet"
OUT_QUERY = "results/vlm_query_results.parquet"


def load_frame(frames_root, video_id, frame_idx):
    fp = os.path.join(frames_root, video_id, "{:03d}.jpg".format(int(frame_idx)))
    return cv2.imread(fp)


def make_image(frames_root, video_id, frame_idx, overlay, gap=3):
    cur = load_frame(frames_root, video_id, frame_idx)
    if cur is None:
        return None
    if overlay == "none":
        return cur
    if overlay == "stack":
        chans = []
        for k in (2 * gap, gap, 0):
            f = load_frame(frames_root, video_id, max(0, int(frame_idx) - k))
            f = cur if f is None else f
            chans.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        chans = [cv2.resize(c, (cur.shape[1], cur.shape[0])) for c in chans]
        return cv2.merge(chans)   # B,G,R = oldest..newest -> motion = colour fringe
    raise ValueError("unknown overlay {}".format(overlay))


def propagate(log_df, query_rows, radius):
    """Per-frame s_vlm by nearest queried frame within `radius`, per video."""
    q = pd.DataFrame(query_rows)
    out = []
    for vid, g in log_df.groupby("video_id", sort=True):
        g = g.sort_values("frame_idx")
        qv = q[q.video_id == vid].sort_values("frame_idx")
        qf = qv["frame_idx"].to_numpy() if len(qv) else np.array([])
        for _, r in g.iterrows():
            row = {"video_id": vid, "frame_idx": int(r.frame_idx),
                   "s_vlm": np.nan, "category": None, "reason": None,
                   "routed_reason": None}
            if len(qf):
                j = int(np.argmin(np.abs(qf - r.frame_idx)))
                if abs(qf[j] - r.frame_idx) <= radius:
                    src = qv.iloc[j]
                    row.update(s_vlm=src.s_vlm, category=src.category,
                               reason=src.reason, routed_reason=src.routed_reason)
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="results/stgnf_frame_scores.parquet")
    ap.add_argument("--frames_root", default=FRAMES_ROOT)
    ap.add_argument("--motion_overlay", choices=["none", "stack"], default="stack")
    ap.add_argument("--blind_stride", type=int, default=12)
    ap.add_argument("--other_stride", type=int, default=36)
    ap.add_argument("--sample_every", type=int, default=50)
    ap.add_argument("--band", type=float, default=None)
    ap.add_argument("--blind_max_poses", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--backend", default="claude_haiku")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--prompt_version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--out_frame", default=OUT_FRAME)
    ap.add_argument("--out_query", default=OUT_QUERY)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    worklist, log, meta = router.build_worklist(
        df, blind_stride=args.blind_stride, other_stride=args.other_stride,
        band=args.band, sample_every=args.sample_every,
        blind_max_poses=args.blind_max_poses,
        max_queries=args.max_queries, limit_videos=args.limit_videos)
    print("Router:", meta, "| queries:", len(worklist))
    os.makedirs("results", exist_ok=True)
    log.to_csv("results/vlm_router_log.csv", index=False)

    client = VLMClient(backend=args.backend, model=args.model,
                       api_key_file=args.api_key_file,
                       prompt_version=args.prompt_version)

    query_rows = []
    for _, r in tqdm(list(worklist.iterrows()), desc="VLM"):
        img = make_image(args.frames_root, r.video_id, r.frame_idx, args.motion_overlay)
        if img is None:
            query_rows.append({"video_id": r.video_id, "frame_idx": int(r.frame_idx),
                               "s_vlm": np.nan, "category": None, "reason": "frame_missing",
                               "routed_reason": r.routed_reason, "error": "no_frame",
                               "from_cache": False})
            continue
        res = client.score(img, r.video_id, int(r.frame_idx), overlay_mode=args.motion_overlay)
        query_rows.append({"video_id": r.video_id, "frame_idx": int(r.frame_idx),
                           "s_vlm": res["s_vlm"], "category": res["category"],
                           "reason": res["reason"], "routed_reason": r.routed_reason,
                           "error": res["error"], "from_cache": res["from_cache"]})

    qdf = pd.DataFrame(query_rows)
    os.makedirs("results", exist_ok=True)
    qdf.to_parquet(args.out_query, index=False)

    # per-frame propagation over the routed video set
    log_set = log[log.video_id.isin(worklist.video_id.unique())] if args.limit_videos else log
    frame_df = propagate(log_set, query_rows, radius=meta["propagate_radius"])
    frame_df.to_parquet(args.out_frame, index=False)

    st = client.stats()
    n_ok = int(qdf["s_vlm"].notna().sum())
    print("\n=== VLM run summary ===")
    print("queries: {} | parsed ok: {} | cache hits: {} | api calls: {}".format(
        len(qdf), n_ok, st["cache_hits"], st["api_calls"]))
    print("tokens in/out: {}/{} | EST COST: ${}".format(
        st["tokens_in"], st["tokens_out"], st["est_cost_usd"]))
    if n_ok:
        print("s_vlm summary:", qdf["s_vlm"].describe().round(3).to_dict())
        print("categories:", qdf["category"].value_counts().to_dict())
    print("Wrote {} (per-query) and {} (per-frame)".format(args.out_query, args.out_frame))


if __name__ == "__main__":
    main()
