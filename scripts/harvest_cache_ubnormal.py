"""Rebuild UBnormal per-frame VLM parquets straight from the on-disk cache.

score_vlm_ubnormal.py re-decodes ~3k mp4 frames just to look up cached answers,
which is slow. The cache path already encodes everything we need:
  cache/vlm/<backend>.jsonl, keyed <video_id>/<frame:06d>_<overlay>_<promptver>_<hash>
so we can harvest the per-query table directly and re-propagate, with no video I/O
and no API calls.

Usage:
  python scripts/harvest_cache_ubnormal.py --backend claude_haiku --prompt_version v2 \
      --out results/vlm_frame_scores_ubnormal_haiku_v2.parquet
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd

import vlm_cache
from score_vlm_ubnormal import build_worklist_labelfree
from score_vlm import propagate

POSE = "results/stgnf_frame_scores_ubnormal.parquet"
FN_RE = re.compile(r"^(\d{6})_(\w+?)_(v\d)_([0-9a-f]+)\.json$")


def harvest(backend, prompt_version, overlay="none"):
    rows = []
    for vid, frame_idx, _ov, _pv, rec in vlm_cache.entries(
            "cache/vlm", backend, prompt_version=prompt_version, overlay=overlay):
        if "scene_" not in vid:          # skip ShanghaiTech-style ids
            continue
        rows.append({"video_id": vid, "frame_idx": frame_idx,
                     "s_vlm": rec.get("s_vlm"), "category": rec.get("category"),
                     "reason": rec.get("reason")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--prompt_version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_query", default=None)
    args = ap.parse_args()

    q = harvest(args.backend, args.prompt_version)
    if q.empty:
        raise SystemExit("no cached entries for {} / {}".format(args.backend, args.prompt_version))
    print("harvested {} cached queries ({} videos)".format(len(q), q.video_id.nunique()))
    print("categories:", q.category.value_counts().head(8).to_dict())

    df = pd.read_parquet(POSE)
    worklist, log, meta = build_worklist_labelfree(df)
    # keep only harvested frames that the router actually selected
    key = set(zip(worklist.video_id, worklist.frame_idx.astype(int)))
    q = q[[ (v, int(f)) in key for v, f in zip(q.video_id, q.frame_idx) ]]
    print("matched to router worklist: {} of {} queries".format(len(q), len(worklist)))

    rr = dict(zip(zip(worklist.video_id, worklist.frame_idx.astype(int)),
                  worklist.routed_reason))
    q["routed_reason"] = [rr.get((v, int(f))) for v, f in zip(q.video_id, q.frame_idx)]
    if args.out_query:
        q.to_parquet(args.out_query, index=False)

    frame_df = propagate(log, q.to_dict("records"), radius=meta["propagate_radius"])
    frame_df.to_parquet(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
