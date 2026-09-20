"""Phase 4b — region-crop augmentation (attacks SCALE misses).

Distant anomalies (cyclists/vehicles far down the walkway) shrink to a few pixels
in the downscaled full frame, so the VLM calls those frames "empty" (verified:
12_0174 f321 had a distant cyclist scored 0.05 full-frame, 0.60 when zoomed).

For each blind-tier routed frame (n_poses<=1, where distant non-pose anomalies
concentrate) we additionally query two UPPER half-width tiles (left, right;
upscaled ~2x by the client) and take s_vlm = max(full, tile_left, tile_right).
Non-blind frames keep their full-frame v2 score. The full-frame scores come from
the v2 cache (free); only the tile queries are new (~$2).

Outputs results/vlm_{query,frame}_scores_v3.parquet. Re-uses the v2 prompt.
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import router
from vlm_client import VLMClient
from score_vlm import make_image, load_frame, propagate, FRAMES_ROOT


def crop(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default=FRAMES_ROOT)
    ap.add_argument("--blind_stride", type=int, default=12)
    ap.add_argument("--other_stride", type=int, default=60)
    ap.add_argument("--prompt_version", default="v2")
    ap.add_argument("--tile_max_poses", type=int, default=1)   # only tile blind-tier
    ap.add_argument("--max_tiles", type=int, default=None)      # smoke cap (frames)
    ap.add_argument("--out_frame", default="results/vlm_frame_scores_v3.parquet")
    ap.add_argument("--out_query", default="results/vlm_query_results_v3.parquet")
    args = ap.parse_args()

    df = pd.read_parquet("results/stgnf_frame_scores.parquet")
    worklist, log, meta = router.build_worklist(
        df, blind_stride=args.blind_stride, other_stride=args.other_stride)
    print("routed:", len(worklist), "| blind-tier to tile:",
          int((worklist.n_poses_detected <= args.tile_max_poses).sum()))

    client = VLMClient(prompt_version=args.prompt_version)
    tiles = [("tile_ul", 0.0, 0.55, 0.0, 0.5), ("tile_ur", 0.45, 1.0, 0.0, 0.5)]

    n_tiled = 0
    rows = []
    for _, r in tqdm(list(worklist.iterrows()), desc="tiles"):
        full = make_image(args.frames_root, r.video_id, r.frame_idx, "none")
        if full is None:
            rows.append({"video_id": r.video_id, "frame_idx": int(r.frame_idx),
                         "s_vlm": np.nan, "category": None, "reason": "no_frame",
                         "routed_reason": r.routed_reason, "source": "none"})
            continue
        best = client.score(full, r.video_id, int(r.frame_idx), overlay_mode="none")
        s_best, cat_best, src = best["s_vlm"], best["category"], "full"
        do_tile = (r.n_poses_detected <= args.tile_max_poses) and \
                  (args.max_tiles is None or n_tiled < args.max_tiles)
        if do_tile:
            n_tiled += 1
            for name, x0, x1, y0, y1 in tiles:
                sub = crop(full, x0, x1, y0, y1)
                rt = client.score(sub, r.video_id, int(r.frame_idx), overlay_mode=name)
                if np.isfinite(rt["s_vlm"]) and (not np.isfinite(s_best) or rt["s_vlm"] > s_best):
                    s_best, cat_best, src = rt["s_vlm"], rt["category"], name
        rows.append({"video_id": r.video_id, "frame_idx": int(r.frame_idx),
                     "s_vlm": s_best, "category": cat_best,
                     "reason": best["reason"], "routed_reason": r.routed_reason,
                     "source": src})

    qdf = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    qdf.to_parquet(args.out_query, index=False)
    frame_df = propagate(log, rows, radius=meta["propagate_radius"])
    frame_df.to_parquet(args.out_frame, index=False)

    st = client.stats()
    flipped = qdf[(qdf.source.str.startswith("tile")) & (qdf.s_vlm >= 0.5)]
    print("\n=== tile augmentation summary ===")
    print("frames tiled: {} | tile queries (new): {} | cache hits: {} | EST COST ${}".format(
        n_tiled, st["api_calls"], st["cache_hits"], st["est_cost_usd"]))
    print("frames where a TILE beat the full frame to >=0.5: {}".format(len(flipped)))
    print("  their categories:", flipped.category.value_counts().to_dict())
    print("Wrote {} and {}".format(args.out_query, args.out_frame))


if __name__ == "__main__":
    main()
