"""Summarise the frozen-detector vs frozen-VLM ablation into the numbers the
paper's detector table needs. Read-only: touches no headline result file."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import json

T = ["vehicle", "bike", "cart"]
POSE = "results/stgnf_frame_scores.parquet"

STREAMS = [
    ("Frozen VLM (Sonnet 4.5)", "results/vlm_query_results_sonnet.parquet",
     "results/cv_fusion.json"),
    ("YOLOv8n (COCO)", "results/det_query_results_yolov8n.parquet",
     "results/cv_fusion_yolov8n.json"),
    ("YOLOv8x (COCO)", "results/det_query_results_yolov8x.parquet",
     "results/cv_fusion_yolov8x.json"),
]

gt = pd.read_parquet(POSE)[["video_id", "frame_idx", "gt_label"]]

print("%-24s %6s %6s %7s %8s %9s %-22s" % (
    "stream", "fires", "prec", "recall*", "cart", "dCF", "95% CI"))
print("-" * 92)
for name, qf, cvf in STREAMS:
    if not os.path.exists(qf):
        print("%-24s  (not yet run)" % name)
        continue
    q = pd.read_parquet(qf).merge(gt, on=["video_id", "frame_idx"], how="left")
    tr = q[q["category"].isin(T)]
    n_anom_q = int((q["gt_label"] == 1).sum())
    cart = int((q["category"] == "cart").sum())
    d = ci = "-"
    if os.path.exists(cvf):
        j = json.load(open(cvf))
        d = "%+.4f" % j["crossfit_delta"]
        ci = "[%+.4f,%+.4f]%s" % (j["boot_delta_ci95"][0], j["boot_delta_ci95"][1],
                                  " SIG" if j["boot_excludes_zero"] else "")
    print("%-24s %6d %6.3f %7.3f %8d %9s %-22s" % (
        name, len(tr), tr["gt_label"].mean(),
        (tr["gt_label"] == 1).sum() / max(n_anom_q, 1), cart, d, ci))

print("\n* recall = share of the anomalous ROUTED frames that trusted evidence fires on")
print("\nper-category precision (query level):")
for name, qf, _ in STREAMS:
    if not os.path.exists(qf):
        continue
    q = pd.read_parquet(qf).merge(gt, on=["video_id", "frame_idx"], how="left")
    parts = []
    for c, g in q.groupby("category"):
        if c != "none" and len(g) >= 5:
            parts.append("%s %d/%.2f" % (c, len(g), g["gt_label"].mean()))
    print("  %-24s %s" % (name, "  ".join(parts)))
