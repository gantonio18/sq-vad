"""Oracle ceiling (offline, $0): upper-bound on the fusion mechanism if the VLM
verdict at the SAME routed frames were perfect. Replaces the VLM evidence at each
queried frame with its ground-truth label (1 anomaly / 0 normal), propagates over
the same window, fuses s = pose_sm + lambda*e_oracle, and sweeps lambda. The best
global AUC bounds the headroom of routing+fusion and isolates VLM recall as the
binding constraint (cf. the real Sonnet result ~0.8857).
"""
import numpy as np
import pandas as pd
import json

import router
import fuse
from score_vlm import propagate
from common import auc

m = fuse.build(pd.read_parquet(fuse.STGNF), pd.read_parquet(fuse.VLM), {"vehicle", "bike", "cart"})
df = pd.read_parquet(fuse.STGNF)
worklist, log, meta = router.build_worklist(df, blind_stride=12, other_stride=60)

# oracle evidence = gt label at each queried frame
gt_map = df.set_index(["video_id", "frame_idx"])['gt_label']
rows = [{"video_id": r.video_id, "frame_idx": int(r.frame_idx),
         "s_vlm": float(gt_map.loc[(r.video_id, int(r.frame_idx))]),
         "category": "oracle", "reason": "", "routed_reason": r.routed_reason}
        for _, r in worklist.iterrows()]
frame_e = propagate(log, rows, radius=meta["propagate_radius"]).rename(columns={"s_vlm": "e_oracle"})
mm = m.merge(frame_e[["video_id", "frame_idx", "e_oracle"]], on=["video_id", "frame_idx"], how="left")
mm["e_oracle"] = mm["e_oracle"].fillna(0.0)

y = mm["gt_label"].to_numpy(); pose_sm = mm["pose_sm"].to_numpy(); eo = mm["e_oracle"].to_numpy()
base = auc(y, pose_sm)
best = (base, 0.0)
for lam in [0.1, 0.3, 0.5, 0.8, 1.2, 2, 3, 5, 8, 12, 20]:
    a = auc(y, pose_sm + lam * eo)
    if a > best[0]:
        best = (a, lam)
print("baseline           {:.4f}".format(base))
print("ORACLE ceiling     {:.4f}  (delta {:+.4f}) at lambda={}".format(best[0], best[0] - base, best[1]))
print("real Sonnet F2     0.8857 (delta +0.0263)  -> recall gap = headroom")
json.dump({"baseline": base, "oracle_global": best[0], "oracle_delta": best[0] - base,
           "oracle_lambda": best[1]}, open("results/oracle_ceiling.json", "w"), indent=2)
print("wrote results/oracle_ceiling.json")
