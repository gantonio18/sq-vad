"""Per-scene normality profiles for UBnormal, induced from NORMAL TRAINING video only.

Motivation: the trusted-category fusion assumes "a vehicle is out of place". On
UBnormal that premise is scene-dependent (roads vs pedestrian areas), which is why
the fusion fails there. Instead of asserting the premise or asking the VLM to guess
the scene type from a single test frame, we give it what the one-class VAD setting
already provides: normal-only footage of the SAME camera.

For each scene we sample frames from a normal clip in the TRAIN split (never test),
montage them, and ask the frozen VLM to state what is normal for that camera --
in particular whether vehicles/bicycles/carts are normally present. The resulting
profile is injected into the per-frame prompt (prompt v4).

No anomaly labels are used: training clips are normal by construction, exactly the
data STG-NF itself was trained on. Writes results/ubnormal_scene_profiles.json.
"""
import os
import re
import sys
import json
import base64
import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from vlm_client import VLMClient  # reuse key resolution + pricing

import paths

DATASET_ROOT = paths.UBNORMAL_ROOT
TRAIN_POSE = "data/UBnormal/pose/train"          # normal-only training clips
OUT = "results/ubnormal_scene_profiles.json"
N_REF_FRAMES = 3

SYSTEM = (
    "You are configuring a fixed CCTV anomaly detector. You are shown a montage of "
    "frames sampled from NORMAL footage of one camera (nothing unusual happens in "
    "them). Describe what ordinary activity looks like for this camera, and state "
    "which object types are a NORMAL part of this scene.\n"
    "Return ONE JSON object only:\n"
    '{"scene_type": "<e.g. pedestrian walkway | street/road | parking area | mixed>", '
    '"vehicles_normal": <true|false>, "bicycles_normal": <true|false>, '
    '"carts_normal": <true|false>, '
    '"normal_description": "<one short sentence>"}\n'
    "Set vehicles_normal=true if cars/vans/trucks legitimately belong here (e.g. a "
    "road or parking area), false if this is a pedestrian-only space where a vehicle "
    "would be out of place. Judge only from the frames shown."
)
USER = ("These frames are all NORMAL for this camera. What is normal here, and are "
        "vehicles/bicycles/carts a normal part of this scene? Return only the JSON.")


def train_clips_by_scene():
    out = {}
    for fn in sorted(os.listdir(TRAIN_POSE)):
        if not fn.endswith(".json"):
            continue
        clip = fn.replace("_alphapose_tracked_person.json", "")
        m = re.findall(r"scene_(\d+)_", clip)
        if not m:
            continue
        out.setdefault(int(m[0]), []).append(clip)
    return out


def mp4_for(clip):
    scene = re.findall(r"scene_(\d+)_", clip)[0]
    return os.path.join(DATASET_ROOT, "Scene" + scene, clip + ".mp4")


def sample_montage(clip, n=N_REF_FRAMES, cell_w=420):
    """Horizontal montage of n frames spread across a normal clip."""
    path = mp4_for(clip)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    idxs = [int(total * f) for f in np.linspace(0.15, 0.85, n)]
    imgs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            imgs.append(cv2.resize(fr, (cell_w, int(h * cell_w / w))))
    cap.release()
    if not imgs:
        return None
    h = min(i.shape[0] for i in imgs)
    return np.hstack([i[:h] for i in imgs])


def ask(client, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    resp = client._client_obj().messages.create(
        model=client.model, max_tokens=400, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": USER}]}])
    txt = "".join(b.text for b in resp.content if b.type == "text")
    usage = (resp.usage.input_tokens, resp.usage.output_tokens)
    mt = re.search(r"\{.*\}", txt, re.S)
    return (json.loads(mt.group(0)) if mt else None), usage, txt


def main():
    client = VLMClient(backend="claude_haiku", model="claude-haiku-4-5")
    by_scene = train_clips_by_scene()
    profiles, tin, tout = {}, 0, 0
    for scene in sorted(by_scene):
        clip = by_scene[scene][0]          # first NORMAL TRAIN clip of this scene
        img = sample_montage(clip)
        if img is None:
            print("scene {}: montage failed ({})".format(scene, clip)); continue
        prof, usage, raw = ask(client, img)
        tin += usage[0]; tout += usage[1]
        if prof is None:
            print("scene {}: parse failed".format(scene)); continue
        prof["ref_clip"] = clip
        profiles[str(scene)] = prof
        print("scene {:2d} [{}] vehicles_normal={} bikes={} carts={} | {}".format(
            scene, prof.get("scene_type"), prof.get("vehicles_normal"),
            prof.get("bicycles_normal"), prof.get("carts_normal"),
            prof.get("normal_description")))
    os.makedirs("results", exist_ok=True)
    json.dump(profiles, open(OUT, "w"), indent=2)
    pin, pout = 1.0 / 1e6, 5.0 / 1e6
    print("\nprofiled {} scenes | tokens {}/{} | est cost ${:.3f}".format(
        len(profiles), tin, tout, tin * pin + tout * pout))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
