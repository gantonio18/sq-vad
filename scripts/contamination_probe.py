"""Reviewer #13 — contamination probe.

The paper argues our per-frame, context-free contract limits memorization pathways.
This turns that argument into a small experiment: show the VLM real ShanghaiTech
test frames and ask it to IDENTIFY the source dataset. If the model cannot name
ShanghaiTech from a frame, it cannot retrieve any dataset-specific memorized labels
keyed to it, so contamination cannot inflate the per-frame anomaly scores through a
recognition-then-recall pathway.

Samples frames across several test videos and asks a single identification question.
Cheap (~16 frames). Writes results/contamination_probe.json.
"""
import os
import re
import sys
import json
import base64
import argparse
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from vlm_client import VLMClient

import paths

FRAMES = paths.SHANGHAITECH_FRAMES

SYSTEM = (
    "You are shown a single frame from a publicly available academic video "
    "dataset used for research. Try to identify EXACTLY which dataset this frame "
    "comes from, based only on what you see (scene, camera, resolution, style).\n"
    "Return ONE JSON object only:\n"
    '{"dataset_guess": "<the specific dataset name, or \'unknown\'>", '
    '"recognized": <true|false>, "confidence": <int 0-100>, '
    '"clues": "<what you based it on, one short clause>"}\n'
    "Set recognized=true only if you can name the specific benchmark with real "
    "confidence; guessing a plausible-sounding name is recognized=false."
)
USER = ("Which specific public video dataset is this exact frame from? Name it if "
        "you genuinely recognize it. Return only the JSON.")


def ask(client, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = client._client_obj().messages.create(
        model=client.model, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": USER}]}])
    txt = "".join(b.text for b in r.content if b.type == "text")
    usage = (r.usage.input_tokens, r.usage.output_tokens)
    mt = re.search(r"\{.*\}", txt, re.S)
    return (json.loads(mt.group(0)) if mt else None), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()

    allv = sorted(os.listdir(FRAMES))
    # stratify: one video per distinct scene (camera) first, then fill evenly
    by_scene = {}
    for v in allv:
        by_scene.setdefault(v.split("_")[0], []).append(v)
    vids = [vs[len(vs) // 2] for vs in by_scene.values()]          # one per camera
    extra = [v for v in allv if v not in vids]
    idx = np.linspace(0, len(extra) - 1, max(a.n - len(vids), 0)).astype(int) if extra else []
    vids += [extra[i] for i in sorted(set(idx))]
    vids = sorted(set(vids))[:a.n]
    client = VLMClient(backend="probe", model=a.model)
    rows, tin, tout = [], 0, 0
    for v in vids:
        fl = sorted(os.listdir(os.path.join(FRAMES, v)))
        if not fl:
            continue
        f = fl[len(fl) // 2]
        img = cv2.imread(os.path.join(FRAMES, v, f))
        if img is None:
            continue
        prof, usage = ask(client, img)
        tin += usage[0]; tout += usage[1]
        if prof is None:
            continue
        g = str(prof.get("dataset_guess", "")).lower()
        correct = ("shanghai" in g)
        rows.append({"video": v, "guess": prof.get("dataset_guess"),
                     "recognized": bool(prof.get("recognized")),
                     "confidence": prof.get("confidence"), "correct": correct,
                     "clues": prof.get("clues")})
        print("{}: guess='{}' recognized={} conf={} correct={}".format(
            v, prof.get("dataset_guess"), prof.get("recognized"),
            prof.get("confidence"), correct))

    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    n_recog = sum(r["recognized"] for r in rows)
    n_correct_conf = sum(r["correct"] and r["recognized"] for r in rows)
    pin, pout = (3.0 / 1e6, 15.0 / 1e6) if "sonnet" in a.model else (1.0 / 1e6, 5.0 / 1e6)
    summary = {"model": a.model, "n_frames": n,
               "named_shanghaitech": n_correct,
               "named_shanghaitech_with_confidence": n_correct_conf,
               "claimed_recognition": n_recog,
               "pct_correct": round(100 * n_correct / max(n, 1), 1),
               "est_cost_usd": round(tin * pin + tout * pout, 4),
               "rows": rows}
    json.dump(summary, open("results/contamination_probe.json", "w"), indent=2)
    print("\n=== {}/{} frames the model named ShanghaiTech ({}%); {} with claimed "
          "confidence. est ${} ===".format(n_correct, n, summary["pct_correct"],
                                           n_correct_conf, summary["est_cost_usd"]))
    print("Wrote results/contamination_probe.json")


if __name__ == "__main__":
    main()
