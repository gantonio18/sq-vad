"""Cheap dev harness for iterating on VLM prompts/categories — never touches the test split.

Tuning prompts on the test set is exactly the leakage the reviewers flagged, so we
iterate on two pools that are not used for reporting:

  POOL A  normal TRAIN clips   — anomaly-free by construction (the one-class training
                                 data). Any category that fires here is a FALSE FIRE.
                                 Needs no labels at all, so it also implements the
                                 label-free trusted-set induction idea.
  POOL B  abnormal VALIDATION clips — do the target anomalies get DETECTED at all.
                                 UBnormal's validation GT is coarse (whole-clip
                                 intervals, mean 83% anomalous), so we read it as a
                                 detection check, not a precision estimate.

A category is worth trusting when it (almost) never fires in pool A and fires often
in pool B. Everything is cached, so re-scoring a prompt you already ran is free.

  python scripts/devset_ubnormal.py --prompt_version v2 --n_normal 150 --n_abnormal 150
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from vlm_client import VLMClient

import paths

DR = paths.UBNORMAL_ROOT
TRAIN_POSE = "data/UBnormal/pose/train"           # normal-only training clips
VAL_POSE = "data/UBnormal/pose/validation"
GT_ROOT = "data/UBnormal/gt"
OUT = "results/devset_ubnormal_{}_{}.parquet"
OBJ = ["vehicle", "bike", "cart", "skateboard"]
EVT = ["fall", "run", "fight", "throw", "chase", "loiter"]


def clips_in(pose_dir):
    return [f.replace("_alphapose_tracked_person.json", "")
            for f in sorted(os.listdir(pose_dir)) if f.endswith(".json")]


def mp4_for(clip):
    s = re.findall(r"scene_(\d+)_", clip)[0]
    return os.path.join(DR, "Scene" + s, clip + ".mp4")


def n_frames(mp4):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


def load_gt(clip, n):
    """Per-frame 1=anomaly. Handles both UBnormal GT encodings (npy mask / interval csv)."""
    p = os.path.join(GT_ROOT, clip + "_tracks.txt")
    if not os.path.exists(p):
        return np.zeros(n, dtype=int)             # no GT file => normal clip
    with open(p, "rb") as f:
        magic = f.read(6)
    if magic.startswith(b"\x93NUMPY"):
        a = np.load(p)
        return (1 - a).astype(int)                # raw npy: 1=normal
    mask = np.zeros(max(n, 1), dtype=int)
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            parts = [int(float(x)) for x in line.split(",")]
        except ValueError:
            continue
        if len(parts) >= 3:
            mask[max(0, parts[1]):min(len(mask), parts[2] + 1)] = 1
    return mask


def sample_pool(clips, budget, want_anomaly, seed=0):
    """Pick (clip, frame_idx, gt) triples spread across clips within a query budget."""
    rng = np.random.default_rng(seed)
    per = max(1, budget // max(len(clips), 1))
    rows = []
    for c in clips:
        mp4 = mp4_for(c)
        if not os.path.exists(mp4):
            continue
        n = n_frames(mp4)
        if n <= 0:
            continue
        gt = load_gt(c, n)
        pool = np.where(gt == 1)[0] if want_anomaly else np.where(gt == 0)[0]
        if len(pool) == 0:
            continue
        idx = np.linspace(0, len(pool) - 1, min(per, len(pool))).astype(int)
        for fi in pool[idx]:
            rows.append((c, int(fi), int(gt[fi])))
    rng.shuffle(rows)
    return rows[:budget]


def score_rows(client, rows, tag):
    out = []
    by_clip = {}
    for c, fi, g in rows:
        by_clip.setdefault(c, []).append((fi, g))
    for c, items in tqdm(by_clip.items(), desc=tag):
        cap = cv2.VideoCapture(mp4_for(c))
        if not cap.isOpened():
            continue
        for fi, g in sorted(items):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            r = client.score(img, c, fi, overlay_mode="none")
            out.append({"pool": tag, "video_id": c, "frame_idx": fi, "gt_label": g,
                        "s_vlm": r.get("s_vlm"), "category": r.get("category")})
        cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude_haiku")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--prompt_version", default="v2")
    ap.add_argument("--n_normal", type=int, default=150)
    ap.add_argument("--n_abnormal", type=int, default=150)
    a = ap.parse_args()

    train_normal = [c for c in clips_in(TRAIN_POSE) if c.startswith("normal_")]
    val_abnormal = [c for c in clips_in(VAL_POSE) if c.startswith("abnormal_")]
    print("pool A: {} normal TRAIN clips | pool B: {} abnormal VALIDATION clips".format(
        len(train_normal), len(val_abnormal)))

    rows_n = sample_pool(train_normal, a.n_normal, want_anomaly=False)
    rows_a = sample_pool(val_abnormal, a.n_abnormal, want_anomaly=True)
    print("sampled {} normal-train frames, {} anomalous-val frames".format(len(rows_n), len(rows_a)))

    client = VLMClient(backend=a.backend, model=a.model, prompt_version=a.prompt_version)
    recs = score_rows(client, rows_n, "normal_train") + score_rows(client, rows_a, "abnormal_val")
    df = pd.DataFrame(recs).dropna(subset=["s_vlm"])
    out = OUT.format(a.backend, a.prompt_version)
    df.to_parquet(out, index=False)

    st = client.stats()
    print("\n=== dev set [{} / {}] api {} cache {} est ${} ===".format(
        a.backend, a.prompt_version, st["api_calls"], st["cache_hits"], st["est_cost_usd"]))

    A = df[df.pool == "normal_train"]
    B = df[df.pool == "abnormal_val"]
    print("\n{:12s} {:>5s} {:>14s} {:>14s} {:>10s}".format(
        "category", "kind", "falsefire%(A)", "detect%(B)", "verdict"))
    cats = sorted(set(df.category.dropna()) - {"none"})
    for c in cats:
        fa = (A.category == c).mean() * 100 if len(A) else float("nan")
        db = (B.category == c).mean() * 100 if len(B) else float("nan")
        kind = "OBJ" if c in OBJ else ("EVT" if c in EVT else "")
        verdict = "TRUST" if (fa <= 2.0 and db >= 3.0) else ("noisy" if fa > 2.0 else "rare")
        print("{:12s} {:>5s} {:>13.1f}% {:>13.1f}% {:>10s}".format(str(c), kind, fa, db, verdict))
    print("\npool A frames {} (all normal) | pool B frames {} (all anomalous)".format(len(A), len(B)))
    print("A category=none rate: {:.1f}%  (higher is better)".format((A.category == "none").mean() * 100))
    print("B category=none rate: {:.1f}%  (lower is better -> missed anomalies)".format(
        (B.category == "none").mean() * 100))
    print("Wrote", out)


if __name__ == "__main__":
    main()
