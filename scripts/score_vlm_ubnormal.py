"""UBnormal fusion transfer (reviewer #5) — VLM appearance stream on UBnormal.

Reads the UBnormal pose parquet, builds the SAME label-free router (blind tier +
median-centred uncertain band + sparse sample), pulls each queried frame straight
from the scenario .mp4 (clean RGB), scores it with the cached VLMClient, and
propagates each score to its +/-window. Writes
results/vlm_query_results_ubnormal.parquet and results/vlm_frame_scores_ubnormal.parquet.

Frames live in mp4s under DATASET_ROOT/Scene<N>/<video_id>.mp4 (indices verified
to align 1:1 with the scored frames). Everything is cached, so reruns never re-bill.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from common import official_frame_table
from vlm_client import VLMClient
from score_vlm import propagate

DATASET_ROOT = "C:/Users/GANTONIO/Desktop/Tese/Datasets/UBnormal"
PARQUET = "results/stgnf_frame_scores_ubnormal.parquet"
OUT_QUERY = "results/vlm_query_results_ubnormal.parquet"
OUT_FRAME = "results/vlm_frame_scores_ubnormal.parquet"


def mp4_for(video_id):
    scene = re.findall(r"scene_(\d+)_", video_id)[0]
    return os.path.join(DATASET_ROOT, "Scene" + scene, video_id + ".mp4")


def build_worklist_labelfree(df, blind_stride=12, other_stride=60,
                             sample_every=60, blind_max_poses=1):
    """Router with a LABEL-FREE (median-centred) uncertain band, matching the
    paper's primary system. Blind tier is queried densely; uncertain/sample thinned."""
    tab = official_frame_table(df, hr=False)
    m = df.merge(tab, on=["video_id", "frame_idx"])
    center = float(np.median(m["s_official"]))          # label-free centre
    band = 0.5 * float(np.std(m["s_official"]))
    m = m.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    m["tier_blind"] = m["n_poses_detected"] <= blind_max_poses
    m["tier_uncert"] = (m["s_official"] - center).abs() <= band
    m["tier_sample"] = (m["frame_idx"] % sample_every) == 0

    queried_idx, reason = [], {}
    for vid, g in m.groupby("video_id", sort=True):
        last_blind, last_other = -10 ** 9, -10 ** 9
        for i, r in g.iterrows():
            if r.tier_blind:
                if (r.frame_idx - last_blind) >= blind_stride:
                    queried_idx.append(i); reason[i] = "blind"; last_blind = r.frame_idx
            elif (r.tier_uncert or r.tier_sample) and (r.frame_idx - last_other) >= other_stride:
                queried_idx.append(i)
                reason[i] = "uncertain" if r.tier_uncert else "sample"
                last_other = r.frame_idx
    m["queried"] = False
    m.loc[queried_idx, "queried"] = True
    m["routed_reason"] = m.index.map(lambda i: reason.get(i, ""))
    worklist = m[m.queried][["video_id", "frame_idx", "routed_reason",
                             "n_poses_detected", "s_official", "gt_label"]].copy()
    meta = {"center_median": round(center, 4), "band": round(band, 4),
            "propagate_radius": other_stride // 2}
    return worklist, m, meta


def read_mp4_frames(video_id, frame_idxs):
    """Return {frame_idx: BGR image} for the requested indices of one video."""
    path = mp4_for(video_id)
    out = {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return out
    for fi in sorted(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok:
            out[int(fi)] = frame
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude_haiku")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--prompt_version", default="v2", choices=["v1", "v2", "v3", "v5", "v6"])
    ap.add_argument("--max_queries", type=int, default=None)  # smoke test cap
    ap.add_argument("--limit_videos", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    worklist, log, meta = build_worklist_labelfree(df)
    if args.limit_videos:
        keep = sorted(worklist.video_id.unique())[:args.limit_videos]
        worklist = worklist[worklist.video_id.isin(keep)]
        log = log[log.video_id.isin(keep)]
    if args.max_queries:
        worklist = worklist.sort_values(["video_id", "frame_idx"]).head(args.max_queries)
    print("Router:", meta, "| queries:", len(worklist),
          "| by reason:", worklist.routed_reason.value_counts().to_dict())

    client = VLMClient(backend=args.backend, model=args.model,
                       prompt_version=args.prompt_version)

    query_rows = []
    for vid, g in tqdm(list(worklist.groupby("video_id", sort=True)), desc="videos"):
        frames = read_mp4_frames(vid, g.frame_idx.tolist())
        for _, r in g.iterrows():
            fi = int(r.frame_idx)
            img = frames.get(fi)
            if img is None:
                query_rows.append({"video_id": vid, "frame_idx": fi, "s_vlm": np.nan,
                                   "category": None, "reason": "frame_missing",
                                   "routed_reason": r.routed_reason, "error": "no_frame",
                                   "from_cache": False})
                continue
            res = client.score(img, vid, fi, overlay_mode="none")
            query_rows.append({"video_id": vid, "frame_idx": fi, "s_vlm": res["s_vlm"],
                               "category": res["category"], "reason": res["reason"],
                               "routed_reason": r.routed_reason, "error": res["error"],
                               "from_cache": res["from_cache"]})

    qdf = pd.DataFrame(query_rows)
    os.makedirs("results", exist_ok=True)
    qdf.to_parquet(OUT_QUERY, index=False)

    frame_df = propagate(log, query_rows, radius=meta["propagate_radius"])
    frame_df.to_parquet(OUT_FRAME, index=False)

    st = client.stats()
    n_ok = int(qdf["s_vlm"].notna().sum())
    print("\n=== UBnormal VLM run ({}) ===".format(args.model))
    print("queries: {} | parsed ok: {} | cache hits: {} | api calls: {}".format(
        len(qdf), n_ok, st["cache_hits"], st["api_calls"]))
    print("tokens in/out: {}/{} | EST COST: ${}".format(
        st["tokens_in"], st["tokens_out"], st["est_cost_usd"]))
    if n_ok:
        print("s_vlm:", qdf["s_vlm"].describe().round(3).to_dict())
        print("categories:", qdf["category"].value_counts().to_dict())
    print("Wrote", OUT_QUERY, "and", OUT_FRAME)


if __name__ == "__main__":
    main()
