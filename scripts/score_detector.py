"""Reviewer ablation — replace the frozen VLM with a frozen COCO object detector.

The obvious challenge to the paper is: the trusted set is {vehicle, bike, cart},
i.e. COCO classes, so why not run an off-the-shelf detector instead of paying a
VLM? This script answers it under a strictly matched protocol:

  * the SAME routed frames (read from the existing VLM query parquet, so the
    router, its tiers and its de-duplication are byte-identical),
  * the SAME output contract (s in [0,1] + a category from our label set),
  * the SAME +/-30 frame propagation (reuses score_vlm.propagate),
  * the SAME fusion and cross-fitting (feed the output to cv_fusion.py --vlm).

Nothing here is trained: the detector is used off-the-shelf on COCO weights,
exactly as the VLM is used off-the-shelf.

Writes results/det_query_results_<tag>.parquet and results/det_frame_scores_<tag>.parquet.
"""
import os
import argparse
import numpy as np
import pandas as pd

import score_vlm  # reuse propagate() so the windowing is identical

import paths

FRAMES_ROOT = paths.SHANGHAITECH_FRAMES

# COCO class name -> our category vocabulary. "cart" has NO COCO class, which is
# itself part of the finding; skateboard is included so the detector is offered
# the same candidate grid the VLM had.
COCO_MAP = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "bike",
    "skateboard": "skateboard",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="results/vlm_query_results_sonnet.parquet",
                    help="routed frames to score (defines the identical query set)")
    ap.add_argument("--pose", default="results/stgnf_frame_scores.parquet")
    ap.add_argument("--frames_root", default=FRAMES_ROOT)
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="low floor; cv_fusion's tau grid does the real thresholding")
    ap.add_argument("--radius", type=int, default=30)
    ap.add_argument("--tag", default="yolov8x")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names

    q = pd.read_parquet(args.queries)[["video_id", "frame_idx", "routed_reason"]]
    print("scoring {} routed frames with {}".format(len(q), args.weights))

    rows = []
    for i, r in enumerate(q.itertuples(index=False)):
        fp = os.path.join(args.frames_root, r.video_id,
                          "{:03d}.jpg".format(int(r.frame_idx)))
        best_s, best_c, detail = 0.0, "none", ""
        if os.path.exists(fp):
            res = model.predict(fp, imgsz=args.imgsz, conf=args.conf,
                                verbose=False, device="cpu")[0]
            hits = []
            for b in res.boxes:
                cname = names[int(b.cls)]
                if cname in COCO_MAP:
                    hits.append((float(b.conf), COCO_MAP[cname], cname))
            if hits:
                hits.sort(reverse=True)
                best_s, best_c = hits[0][0], hits[0][1]
                detail = ",".join("{}:{:.2f}".format(h[2], h[0]) for h in hits[:5])
        rows.append({"video_id": r.video_id, "frame_idx": int(r.frame_idx),
                     "s_vlm": best_s, "category": best_c, "reason": detail,
                     "routed_reason": r.routed_reason, "error": None,
                     "from_cache": False})
        if (i + 1) % 200 == 0:
            print("  {}/{}".format(i + 1, len(q)), flush=True)

    qdf = pd.DataFrame(rows)
    qout = "results/det_query_results_{}.parquet".format(args.tag)
    qdf.to_parquet(qout, index=False)

    # identical propagation to the VLM stream
    pose = pd.read_parquet(args.pose)
    log_set = pose[pose.video_id.isin(qdf.video_id.unique())][["video_id", "frame_idx"]]
    frame_df = score_vlm.propagate(log_set, rows, radius=args.radius)
    fout = "results/det_frame_scores_{}.parquet".format(args.tag)
    frame_df.to_parquet(fout, index=False)

    print("\nfires by category (query level):")
    print(qdf["category"].value_counts().to_string())
    print("\nwrote {} and {}".format(qout, fout))


if __name__ == "__main__":
    main()
