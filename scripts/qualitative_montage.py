"""Reviewer #12 — qualitative figure with real frames: two rescues + one miss.

Builds a labelled 1x3 montage from ShanghaiTech test frames:
  (a) a pose-blind vehicle the flow scores max-normal but the trusted VLM catches,
  (b) a cart anomaly (person present, pose normal) the VLM catches,
  (c) a real anomaly the VLM scores "none" -- recall, the binding constraint.
Writes ../Article/figures/qualitative_examples.png.
"""
import os
import cv2
import numpy as np

FRAMES = "C:/Users/GANTONIO/Desktop/Tese/Datasets/shanghaitech/testing/frames"
OUT = "../Article/figures/qualitative_examples.png"

# (video, frame, top-tag, pose-verdict, vlm-verdict, kind)
PANELS = [
    ("06_0144", 124, "RESCUED  -  vehicle",
     "pose: 0 skeletons -> scored max-normal", "VLM: vehicle (0.95)", "good"),
    ("01_0054", 381, "RESCUED  -  cart",
     "pose: 5 skeletons, scored normal", "VLM: cart (0.65)", "good"),
    ("03_0059", 200, "MISSED  -  both streams blind",
     "pose: 4 skeletons, scored normal", "VLM: none (0.08)", "bad"),
]
CELL_W = 480
BAR_TOP = 34
BAR_BOT = 60


def load(video, frame):
    fp = os.path.join(FRAMES, video, "{:03d}.jpg".format(int(frame)))
    return cv2.imread(fp)


def panel(video, frame, toptag, pose_v, vlm_v, kind):
    img = load(video, frame)
    if img is None:
        img = np.full((int(CELL_W * 0.56), CELL_W, 3), 40, np.uint8)
    h, w = img.shape[:2]
    img = cv2.resize(img, (CELL_W, int(h * CELL_W / w)))
    H = img.shape[0]
    canvas = np.full((BAR_TOP + H + BAR_BOT, CELL_W, 3), 255, np.uint8)
    # top bar: colour by good/bad rescue
    col = (60, 140, 60) if kind == "good" else (60, 60, 190)  # BGR green / red
    cv2.rectangle(canvas, (0, 0), (CELL_W, BAR_TOP), col, -1)
    cv2.putText(canvas, toptag, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    canvas[BAR_TOP:BAR_TOP + H] = img
    # bottom bar: two lines of evidence
    y0 = BAR_TOP + H
    cv2.putText(canvas, "{}  f{}   {}".format(video, frame, pose_v),
                (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "{}    ground truth: ANOMALY".format(vlm_v),
                (8, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    return canvas


def main():
    panels = [panel(*p) for p in PANELS]
    H = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((H - p.shape[0], p.shape[1], 3), 255, np.uint8)])
              if p.shape[0] < H else p for p in panels]
    gap = np.full((H, 12, 3), 255, np.uint8)
    grid = panels[0]
    for p in panels[1:]:
        grid = np.hstack([grid, gap, p])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, grid)
    print("wrote", OUT, grid.shape)
    for v, f, *_ in PANELS:
        fp = os.path.join(FRAMES, v, "{:03d}.jpg".format(f))
        print("  {} f{}: {}".format(v, f, "OK" if os.path.exists(fp) else "MISSING"))


if __name__ == "__main__":
    main()
