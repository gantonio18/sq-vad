"""Phase 6 — score fusion of frozen STG-NF (s_pose) and the VLM (s_vlm).

Empirical finding driving the design (see error_analysis / fusion_metrics):
the VLM is HIGH-PRECISION on the non-pose OBJECT anomalies that STG-NF is
structurally blind to (vehicle 100%, bike 84%, cart) but NOISY on motion
categories (run ~33% precise — the motion overlay over-triggers on brisk
walkers). Its NORMAL verdicts are unreliable (recall ~0.5). Therefore:

  * the VLM is used as POSITIVE-ONLY evidence (it may RAISE suspicion, never
    vouch for normality), and
  * only its TRUSTED categories (default vehicle/bike/cart) contribute.

Streams: pose anomaly = -normality, inf-filled, then STG-NF's iterated Gaussian
smoothing (== the 0.8594 baseline). The trusted VLM evidence (propagated, so it
already has temporal extent) is added on top.

Strategies (brief §6.2):
  F1 weighted  : (1-w)*z(pose) + w*z(s_vlm_ALL)   -- the doc's naive late fusion;
                 reported as the cautionary contrast (blanket fusion fails).
  F2 additive  : pose_smoothed + lam * e_trust      -- PRIMARY (category-aware,
                 positive-only, blind-spot).  lam swept.
  F3 max       : max(z(pose), z(e_trust)).

No-pose rule (brief §6.3): replacing the max-normal fill with the VLM at no-pose
frames is available via --nopose_rule on; empirically it HURTS here (no-pose
frames are mostly normal), which is itself a reported result.

Outputs: results/fused_frame_scores.parquet + results/fusion_metrics.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from common import infilled_pose_table, smooth_by_clip, auc, SHANGHAITECH_HR_SKIP

STGNF = "results/stgnf_frame_scores.parquet"
VLM = "results/vlm_frame_scores.parquet"
OUT_PARQUET = "results/fused_frame_scores.parquet"
OUT_JSON = "results/fusion_metrics.json"
BASELINE = 0.8593719603559267
TRUST_DEFAULT = ["vehicle", "bike", "cart"]


def gz(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 0 else 1.0)


def hr_mask(m):
    return ~m.apply(lambda r: (int(r.video_id.split("_")[0]),
                               int(r.video_id.split("_")[1])) in SHANGHAITECH_HR_SKIP, axis=1).to_numpy()


def build(pose_parq, vlm_parq, trust):
    pose = infilled_pose_table(pose_parq, hr=False)
    npd = pose_parq[["video_id", "frame_idx", "n_poses_detected"]]
    vlm = vlm_parq[["video_id", "frame_idx", "s_vlm", "category"]]
    m = pose.merge(vlm, on=["video_id", "frame_idx"], how="left")
    m = m.merge(npd, on=["video_id", "frame_idx"], how="left")
    m = m.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    m["a_pose"] = -m["normality_filled"]
    # pose smoothed == official baseline score
    sm = smooth_by_clip(m[["video_id", "frame_idx", "a_pose"]], "a_pose")
    m["pose_sm"] = sm["a_pose_sm"].to_numpy()
    m["zp"] = gz(m["pose_sm"])

    v = m["s_vlm"].fillna(0.0).to_numpy()
    cat = m["category"].fillna("none").to_numpy()
    m["e_all"] = v                                   # positive evidence, all cats
    m["e_trust"] = np.where(np.isin(cat, list(trust)), v, 0.0)  # trusted cats only
    m["vlm_covered"] = m["s_vlm"].notna()
    return m


def AUCs(m, s):
    y = m["gt_label"].to_numpy()
    out = {"global": auc(y, s)}
    hm = hr_mask(m)
    out["hr"] = auc(y[hm], s[hm])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=STGNF)
    ap.add_argument("--vlm", default=VLM)
    ap.add_argument("--trust", nargs="*", default=TRUST_DEFAULT)
    ap.add_argument("--nopose_rule", choices=["on", "off"], default="off")
    ap.add_argument("--out", default=OUT_PARQUET)
    args = ap.parse_args()

    m = build(pd.read_parquet(args.pose), pd.read_parquet(args.vlm), set(args.trust))
    zp = m["zp"].to_numpy(); pose_sm = m["pose_sm"].to_numpy()
    e_trust = m["e_trust"].to_numpy(); e_all = m["e_all"].to_numpy()

    # anchor
    base = AUCs(m, pose_sm)["global"]
    assert abs(base - BASELINE) < 1e-6, "anchor broken: {}".format(base)
    metrics = {"baseline_global": base, "baseline_hr": AUCs(m, pose_sm)["hr"],
               "trust_categories": sorted(args.trust)}
    metrics["vlm_only_global"] = AUCs(m, gz(e_all))["global"]
    print("Baseline {:.4f} [anchor OK] | VLM-only {:.4f}".format(base, metrics["vlm_only_global"]))

    # ---- F1: doc convex late fusion on ALL categories (naive contrast) ----
    ze = gz(e_all)
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    f1 = {float(w): AUCs(m, (1 - w) * zp + w * ze)["global"] for w in ws}
    bw = max(f1, key=f1.get)
    metrics.update(f1_w_sweep={str(k): round(v, 4) for k, v in f1.items()},
                   f1_best_w=bw, f1_best_global=f1[bw],
                   f1_best_hr=AUCs(m, (1 - bw) * zp + bw * ze)["hr"])
    s_f1 = (1 - bw) * zp + bw * ze
    print("F1 convex (all cats) best w={:.2f} -> {:.4f} (naive contrast)".format(bw, f1[bw]))

    # ---- F2: category-aware positive additive (PRIMARY); sweep lambda ----
    lams = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]
    f2 = {float(l): AUCs(m, pose_sm + l * e_trust)["global"] for l in lams}
    bl = max(f2, key=f2.get)
    s_f2 = pose_sm + bl * e_trust
    metrics.update(f2_lambda_sweep={str(k): round(v, 4) for k, v in f2.items()},
                   f2_best_lambda=bl, f2_global=f2[bl], f2_hr=AUCs(m, s_f2)["hr"])
    # no-pose-rule ablation (replace the max-normal fill with trusted VLM at
    # no-pose frames): discard the smoothed fill there — set the pose term to
    # the raw most-normal value (same scale as pose_sm) — and keep only the
    # trusted VLM evidence. Elsewhere the score is the F2 rule unchanged.
    npmask = m["is_nopose"].to_numpy()
    s_np = pose_sm + bl * e_trust
    s_np[npmask] = pose_sm.min() + bl * e_trust[npmask]
    metrics["f2_global_nopose_on"] = AUCs(m, s_np)["global"]
    metrics["f2_nopose_rule_reported"] = (
        "on" if args.nopose_rule == "on" else
        "off (F2 keeps the smoothed fill; see f2_global_nopose_on)")
    print("F2 additive (trust={}) best lam={:.2f} -> {:.4f} (HR {:.4f})".format(
        sorted(args.trust), bl, f2[bl], metrics["f2_hr"]))

    # ---- F3 max ----
    s_f3 = np.maximum(zp, gz(e_trust))
    metrics["f3_global"] = AUCs(m, s_f3)["global"]
    metrics["f3_hr"] = AUCs(m, s_f3)["hr"]
    print("F3 max -> {:.4f} (HR {:.4f})".format(metrics["f3_global"], metrics["f3_hr"]))

    # save per-frame fused scores
    m["s_f1"] = s_f1; m["s_f2"] = s_f2; m["s_f3"] = s_f3
    keep = ["video_id", "frame_idx", "gt_label", "is_nopose", "n_poses_detected",
            "vlm_covered", "category", "zp", "e_all", "e_trust", "pose_sm",
            "s_f1", "s_f2", "s_f3"]
    os.makedirs("results", exist_ok=True)
    m[keep].to_parquet(args.out, index=False)
    json.dump(metrics, open(OUT_JSON, "w"), indent=2)

    best = max([("F1", metrics["f1_best_global"]), ("F2", metrics["f2_global"]),
                ("F3", metrics["f3_global"])], key=lambda t: t[1])
    print("\nBEST: {} Global {:.4f}  (baseline {:.4f}, delta {:+.4f})".format(
        best[0], best[1], base, best[1] - base))
    print("Wrote {} and {}".format(args.out, OUT_JSON))


if __name__ == "__main__":
    main()
