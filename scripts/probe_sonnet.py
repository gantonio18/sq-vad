"""Cheap Sonnet probe: query Sonnet (v2 prompt, clean RGB) on the SAME
rescue-target frames Haiku struggled with + a sample of normals, and compare
recall/precision head-to-head before committing to a full Sonnet run.
"""
import argparse
import numpy as np
import pandas as pd

from vlm_client import VLMClient
from score_vlm import make_image, FRAMES_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_normals", type=int, default=100)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--backend", default="claude_sonnet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q = pd.read_parquet("results/vlm_query_results.parquet")   # canonical = v2 Haiku
    f = pd.read_parquet("results/stgnf_frame_scores.parquet")[
        ["video_id", "frame_idx", "gt_label", "n_poses_detected"]]
    fsm = pd.read_parquet("results/fused_frame_scores.parquet")[["video_id", "frame_idx", "pose_sm"]]
    q = q.merge(f, on=["video_id", "frame_idx"]).merge(fsm, on=["video_id", "frame_idx"])

    anom = q[q.gt_label == 1]
    rescue = anom[anom.pose_sm < anom.pose_sm.median()].copy()
    normals = q[q.gt_label == 0].sample(min(args.n_normals, (q.gt_label == 0).sum()),
                                        random_state=args.seed).copy()
    probe = pd.concat([rescue, normals])
    print("probe: {} rescue-targets + {} normals = {} frames".format(
        len(rescue), len(normals), len(probe)))

    client = VLMClient(backend=args.backend, model=args.model, prompt_version="v2")
    son = []
    for _, r in probe.iterrows():
        img = make_image(FRAMES_ROOT, r.video_id, int(r.frame_idx), "none")
        s = client.score(img, r.video_id, int(r.frame_idx), overlay_mode="none")["s_vlm"] if img is not None else np.nan
        son.append(s)
    probe["s_sonnet"] = son
    probe["s_haiku"] = probe["s_vlm"]

    res = rescue.merge(probe[["video_id", "frame_idx", "s_sonnet"]], on=["video_id", "frame_idx"])
    norm = normals.merge(probe[["video_id", "frame_idx", "s_sonnet"]], on=["video_id", "frame_idx"])
    print("\n=== RESCUE TARGETS (n={}) — recall @s>=0.5 ===".format(len(res)))
    print("  Haiku : {:.1%}   mean s={:.2f}".format((res.s_vlm >= 0.5).mean(), res.s_vlm.mean()))
    print("  Sonnet: {:.1%}   mean s={:.2f}".format((res.s_sonnet >= 0.5).mean(), res.s_sonnet.mean()))
    print("=== NORMALS (n={}) — false-alarm rate @s>=0.5 (lower better) ===".format(len(norm)))
    print("  Haiku : {:.1%}".format((norm.s_vlm >= 0.5).mean()))
    print("  Sonnet: {:.1%}".format((norm.s_sonnet >= 0.5).mean()))
    print("\nstats:", client.stats())
    probe.to_parquet("results/_sonnet_probe.parquet", index=False)


if __name__ == "__main__":
    main()
