"""Phase 4 — thin, cached VLM client (Claude Haiku backend).

Contract (identical across any future backend; only this client changes):
  input : one composite RGB image (the frame, optionally with a motion overlay)
  output: {"anomaly_score": 0-100, "category": "...", "reason": "..."}  -> s_vlm = score/100

Robustness rules (from the brief §9):
  - cache every call to disk keyed by (backend, video_id, frame_idx, overlay_mode);
    reruns NEVER re-bill or re-query.
  - retry transient API errors with backoff; on JSON parse failure retry once,
    then return s_vlm = NaN (caller falls back to s_pose). Never crash the run.
  - track token usage + $ cost.

API key: read from $ANTHROPIC_API_KEY, or from --api_key_file if given. The key is
never logged or written to the cache. Verdicts are cached in one JSONL file per
backend (see vlm_cache.py).
"""
import os
import io
import json
import base64
import time
import hashlib

import numpy as np
import cv2

import vlm_cache

DEFAULT_MODEL = "claude-haiku-4-5"

# public pricing ($ per token), selected by model name
PRICE_IN = 1.0 / 1_000_000   # default = Haiku 4.5
PRICE_OUT = 5.0 / 1_000_000


def model_price(model):
    """(input, output) $/token. Sonnet 4.5 = $3/$15 per M; Haiku 4.5 = $1/$5."""
    if "sonnet" in model.lower():
        return 3.0 / 1_000_000, 15.0 / 1_000_000
    if "opus" in model.lower():
        return 15.0 / 1_000_000, 75.0 / 1_000_000
    return 1.0 / 1_000_000, 5.0 / 1_000_000

CATEGORIES = ["vehicle", "bike", "skateboard", "cart", "fight", "fall",
              "run", "throw", "chase", "crowd", "loiter", "none", "other"]

SYSTEM_PROMPT = (
    "You are a surveillance anomaly rater for a fixed CCTV camera over a "
    "university CAMPUS PEDESTRIAN WALKWAY. NORMAL = people walking or standing. "
    "ANOMALOUS = anything that does not belong on a pedestrian walkway, e.g. any "
    "vehicle/car/van/truck, motorbike, bicycle being ridden, skateboard, "
    "scooter, wheeled cart/trolley, running or chasing, fighting, falling, "
    "throwing or brawling, jumping, or a person clearly moving much faster than "
    "walkers around them.\n"
    "If the image is a 3-frame temporal stack, fast motion appears as coloured "
    "ghosting/fringing around a moving object; use that to judge speed.\n"
    "Rate ONLY what is visible. Reply with a SINGLE JSON object and nothing else:\n"
    '{"anomaly_score": <int 0-100>, "category": "<one of: '
    + ", ".join(CATEGORIES) + '>", "reason": "<one short clause>"}\n'
    "anomaly_score: 0 = clearly normal pedestrians, 100 = obvious vehicle/fight/"
    "fast-motion anomaly. Use the full range; reserve >60 for a concrete anomaly "
    "you can name."
)

USER_TEXT = ("Rate this campus walkway frame. What are the people doing, and is "
             "any non-pedestrian object (vehicle, bike, cart, skateboard) present "
             "or is anyone moving abnormally fast? Return only the JSON object.")

# ---- v2: recall-focused, object-scanning, graded (for clean RGB frames) ----
# Motivation: v1+stack overlay ghosted moving vehicles so Haiku called scenes
# "empty"; v2 runs on clean RGB and pushes the model to scan edges/distance and
# to use the full score range instead of defaulting to 0.
SYSTEM_PROMPT_V2 = (
    "You are a surveillance anomaly rater for a fixed CCTV camera over a "
    "university CAMPUS PEDESTRIAN WALKWAY where ONLY people walking or standing "
    "is normal.\n"
    "ANOMALOUS = anything that is not a pedestrian walking/standing: any "
    "car/van/truck/vehicle, motorbike, bicycle, scooter, skateboard, wheeled "
    "cart or trolley, or a person running, fighting, falling, jumping, lying "
    "down, or throwing.\n"
    "Look CAREFULLY for objects that may be SMALL, DISTANT, PARTIALLY VISIBLE, "
    "or at the EDGES/corners of the frame or far down the walkway — anomalous "
    "vehicles and carts are often small or near the image border. Do NOT call "
    "the scene empty/normal without scanning the whole frame, including the far "
    "background and all edges.\n"
    "Return ONE JSON object only:\n"
    '{"objects": [non-pedestrian objects you see, e.g. "white van","bicycle"; '
    '[] if none], "anomaly_score": <int 0-100>, "category": "<one of: '
    + ", ".join(CATEGORIES) + '>", "reason": "<one short clause>"}\n'
    "Scoring (use the FULL range; do NOT default to 0):\n"
    " 0-15  only normal pedestrians, nothing else.\n"
    " 20-45 something ambiguous or slightly unusual you are unsure about.\n"
    " 50-80 a clear non-pedestrian object (bike/cart/vehicle/skateboard) OR "
    "clear running/fighting/falling.\n"
    " 85-100 obvious vehicle on the walkway, fight, fall, or several anomalies.\n"
    "If you see ANY non-pedestrian wheeled object, score at least 50 and set "
    "category to that object. When uncertain, prefer 35-50 over 0 — missing a "
    "real anomaly is worse than a mild false alarm."
)
USER_TEXT_V2 = ("Carefully scan this campus walkway image, including distant "
                "areas and all four edges. List any non-pedestrian objects and "
                "rate the anomaly. Return only the JSON object.")

# ---- v3: CONTEXT-AWARE / scene-conditioned (for multi-scene deployments) ----
# Motivation: v1/v2 hard-assert "pedestrian walkway where any vehicle is anomalous".
# That is false in scenes where vehicles belong (roads/parking), inflating trusted-
# category false positives (UBnormal). v3 makes the model FIRST infer the scene type
# from the image and flag an object only if it is OUT OF PLACE for that scene, so a
# car on a road is normal but a car on a walkway is not. Same JSON contract + category
# set, so routing/fusion are unchanged.
SYSTEM_PROMPT_V3 = (
    "You are a surveillance anomaly rater for a fixed camera. FIRST infer the SCENE "
    "TYPE from the image itself: a PEDESTRIAN-ONLY area (walkway, plaza, campus path "
    "where only people on foot belong) or a ROAD/PARKING/MIXED-TRAFFIC area (street, "
    "driveway, parking lot where vehicles normally belong).\n"
    "An object or event is ANOMALOUS only if it is OUT OF PLACE for the scene you see:\n"
    " - Pedestrian-only area: any car/van/truck/vehicle, motorbike, ridden bicycle, "
    "scooter, skateboard or wheeled cart is anomalous; so is a person running, "
    "fighting, falling, jumping, lying down or throwing.\n"
    " - Road/parking/mixed-traffic area: vehicles, bicycles and carts are NORMAL and "
    "must NOT be flagged; flag only genuinely out-of-place events (a person lying or "
    "falling in the roadway, a fight, a collision, someone running in panic).\n"
    "Do NOT flag an object merely because it is a vehicle/bike/cart -- flag it only if "
    "it does not belong in THIS scene. Scan small, distant, partially visible and "
    "edge-of-frame objects before deciding.\n"
    "Return ONE JSON object only:\n"
    '{"scene_type": "<pedestrian|road|mixed>", "objects": [out-of-place items; [] if '
    'none], "anomaly_score": <int 0-100>, "category": "<one of: ' + ", ".join(CATEGORIES)
    + '>", "reason": "<one short clause>"}\n'
    "Scoring (use the FULL range; do NOT default to 0):\n"
    " 0-15  nothing out of place for this scene.\n"
    " 20-45 something ambiguous or slightly unusual.\n"
    " 50-80 a clear out-of-place object, or clear running/fighting/falling.\n"
    " 85-100 an obvious out-of-place vehicle on a walkway, a fight, a fall, or several "
    "anomalies.\n"
    "If an object is out of place, set category to that object; if the scene is a "
    "road/parking area and vehicles simply belong there, set category 'none'. When "
    "uncertain, prefer 35-50 over 0."
)
USER_TEXT_V3 = ("First decide whether this is a pedestrian-only area or a "
                "road/parking area, then flag only objects or events that are OUT OF "
                "PLACE for that scene. Return only the JSON object.")

# ---- v5: behaviour-first, object-second (for domains whose anomalies are events) ----
# Motivation: on the UBnormal dev harness the v2 object-focused prompt answered
# "none" on ~60% of anomalous frames -- it treats non-pedestrian OBJECTS as the
# anomaly definition and under-detects behavioural anomalies (fall/run/fight), which
# dominate UBnormal. v5 makes abnormal human behaviour the primary target and keeps
# out-of-place objects secondary; "normal" is calm people + vehicles where they belong,
# to keep false fires on normal footage low.
SYSTEM_PROMPT_V5 = (
    "You are a surveillance anomaly rater for a fixed CCTV camera in a public area "
    "(walkway, plaza, station, or street). NORMAL is people walking, standing, "
    "waiting, sitting or queuing calmly, plus any vehicles or bicycles only where "
    "they plainly belong (a road or parking area).\n"
    "Your FIRST priority is anomalous HUMAN BEHAVIOUR -- the main thing to catch "
    "here: a person running or sprinting, fighting or brawling, falling or lying on "
    "the ground, staggering, jumping, crawling, throwing something, or chasing or "
    "fleeing -- anyone moving or acting very differently from the calm people around "
    "them. Such behaviour is anomalous wherever it occurs.\n"
    "SECOND, flag out-of-place OBJECTS: a car/van/truck, motorbike, ridden bicycle, "
    "scooter, skateboard or wheeled cart in a space meant for pedestrians (not one "
    "where such vehicles normally belong).\n"
    "Scan the whole frame -- including small, distant, partially visible and "
    "edge-of-frame people and objects -- before deciding.\n"
    "Return ONE JSON object only:\n"
    '{"anomaly_score": <int 0-100>, "category": "<one of: ' + ", ".join(CATEGORIES)
    + '>", "reason": "<one short clause>"}\n'
    "Pick the category that best names the anomaly: a BEHAVIOUR "
    "(run/fall/fight/throw/chase) when a person acts abnormally; an OBJECT "
    "(vehicle/bike/cart/skateboard) when an object is out of place; 'none' only when "
    "everyone is calm and nothing is out of place.\n"
    "Scoring (use the FULL range; do NOT default to 0):\n"
    " 0-15  everyone calm, nothing out of place.\n"
    " 20-45 someone slightly unusual, or an ambiguous object you are unsure about.\n"
    " 50-80 a clear abnormal behaviour (running/fighting/falling/throwing) or a clear "
    "out-of-place object.\n"
    " 85-100 an obvious fall, fight, panic, or a vehicle among pedestrians.\n"
    "Missing a real anomaly is worse than a mild false alarm: when a person is acting "
    "even somewhat abnormally, prefer 40-60 over 0."
)
USER_TEXT_V5 = ("Watch the people first: is anyone running, fighting, falling, lying "
                "down, throwing, chasing or moving abnormally? Then check for any "
                "out-of-place vehicle/bike/cart. Rate the anomaly and name its "
                "category. Return only the JSON object.")

# ---- v6: v5 with a stricter running bar ----
# On the dev harness v5 lifted recall (missed-anomaly rate 60%->47%) but "run"
# over-fired on normal footage (false-fire 2%->11%): it called brisk/hurried walkers
# "run". v6 keeps everything in v5 but requires clear sprinting/fleeing for the run
# category and explicitly rules out normal walking pace, to cut those false fires
# without sacrificing genuine running anomalies.
SYSTEM_PROMPT_V6 = (
    "You are a surveillance anomaly rater for a fixed CCTV camera in a public area "
    "(walkway, plaza, station, or street). NORMAL is people walking at ANY pace "
    "(including brisk or hurried walking), standing, waiting, sitting or queuing "
    "calmly, plus any vehicles or bicycles only where they plainly belong (a road or "
    "parking area).\n"
    "Your FIRST priority is anomalous HUMAN BEHAVIOUR -- the main thing to catch "
    "here: a person falling or lying on the ground, fighting or brawling, staggering, "
    "jumping, crawling, throwing something, or chasing or fleeing. For RUNNING, use "
    "the 'run' category ONLY when a person is clearly SPRINTING or running in "
    "panic -- moving much faster than everyone around them, legs in a clear run. Do "
    "NOT use 'run' for someone merely walking briskly, hurrying, or jogging casually; "
    "that is normal.\n"
    "SECOND, flag out-of-place OBJECTS: a car/van/truck, motorbike, ridden bicycle, "
    "scooter, skateboard or wheeled cart in a space meant for pedestrians (not one "
    "where such vehicles normally belong).\n"
    "Scan the whole frame -- including small, distant, partially visible and "
    "edge-of-frame people and objects -- before deciding.\n"
    "Return ONE JSON object only:\n"
    '{"anomaly_score": <int 0-100>, "category": "<one of: ' + ", ".join(CATEGORIES)
    + '>", "reason": "<one short clause>"}\n'
    "Pick the category that best names the anomaly: a BEHAVIOUR "
    "(fall/fight/throw/chase, or run only for clear sprinting) when a person acts "
    "abnormally; an OBJECT (vehicle/bike/cart/skateboard) when an object is out of "
    "place; 'none' when everyone is calm (walking at any pace counts as calm) and "
    "nothing is out of place.\n"
    "Scoring (use the FULL range; do NOT default to 0):\n"
    " 0-15  everyone calm or merely walking/hurrying, nothing out of place.\n"
    " 20-45 someone slightly unusual, or an ambiguous object you are unsure about.\n"
    " 50-80 a clear abnormal behaviour (sprinting/fighting/falling/throwing) or a "
    "clear out-of-place object.\n"
    " 85-100 an obvious fall, fight, panic, or a vehicle among pedestrians.\n"
    "Missing a real anomaly is worse than a mild false alarm, but a person walking "
    "briskly is NOT an anomaly: reserve high scores for genuine abnormal behaviour or "
    "out-of-place objects."
)
USER_TEXT_V6 = ("Watch the people first: is anyone falling, fighting, throwing, "
                "chasing, or clearly SPRINTING (not just walking briskly)? Then check "
                "for any out-of-place vehicle/bike/cart. Rate the anomaly and name "
                "its category. Return only the JSON object.")

PROMPTS = {"v1": (SYSTEM_PROMPT, USER_TEXT), "v2": (SYSTEM_PROMPT_V2, USER_TEXT_V2),
           "v3": (SYSTEM_PROMPT_V3, USER_TEXT_V3), "v5": (SYSTEM_PROMPT_V5, USER_TEXT_V5),
           "v6": (SYSTEM_PROMPT_V6, USER_TEXT_V6)}


class VLMClient:
    def __init__(self, backend="claude_haiku", model=DEFAULT_MODEL,
                 cache_root="cache/vlm", api_key_file=None, max_dim=768,
                 jpeg_quality=85, max_retries=4, prompt_version="v1"):
        self.backend = backend
        self.model = model
        self.prompt_version = prompt_version
        self.system, self.user = PROMPTS[prompt_version]
        self.cache_root = cache_root
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality
        self.max_retries = max_retries
        self.tokens_in = 0
        self.tokens_out = 0
        self.n_api_calls = 0
        self.n_cache_hits = 0
        self._client = None
        self._api_key_file = api_key_file
        self._cache_index = None

    # ---- key + lazy client ----
    def _get_key(self):
        k = os.environ.get("ANTHROPIC_API_KEY")
        if k:
            return k.strip()
        if self._api_key_file and os.path.exists(self._api_key_file):
            with open(self._api_key_file) as f:
                return f.read().strip()
        raise RuntimeError(
            "No Anthropic API key. Set the ANTHROPIC_API_KEY environment variable "
            "(preferred) or pass --api_key_file. Never commit a key to the repository.")

    def _client_obj(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._get_key())
        return self._client

    # ---- cache ----
    def _cache_key(self, video_id, frame_idx, overlay_mode, img_b64):
        # key on prompt version + a short hash of the actual image, so a prompt
        # OR overlay change re-queries instead of returning a stale answer
        h = hashlib.sha1(img_b64.encode()).hexdigest()[:8]
        return "{}/{:06d}_{}_{}_{}".format(
            video_id, int(frame_idx), overlay_mode, self.prompt_version, h)

    def _index(self):
        """Lazily load this backend's cache (see scripts/vlm_cache.py)."""
        if self._cache_index is None:
            self._cache_index = vlm_cache.load(self.cache_root, self.backend)
        return self._cache_index

    # ---- image encode ----
    def _encode(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = self.max_dim / max(h, w)
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buf.tobytes()).decode()

    # ---- main entry ----
    def score(self, img_bgr, video_id, frame_idx, overlay_mode="none"):
        """Return dict: {s_vlm, anomaly_score, category, reason, from_cache, error}."""
        img_b64 = self._encode(img_bgr)
        ckey = self._cache_key(video_id, frame_idx, overlay_mode, img_b64)
        hit = self._index().get(ckey)
        if hit is not None:
            rec = dict(hit)
            rec["from_cache"] = True
            rec.setdefault("error", None)
            self.n_cache_hits += 1
            return rec

        parsed, usage, err = self._call_with_retries(img_b64)
        rec = {
            "s_vlm": (parsed["anomaly_score"] / 100.0) if parsed else float("nan"),
            "anomaly_score": parsed["anomaly_score"] if parsed else None,
            "category": parsed.get("category") if parsed else None,
            "reason": parsed.get("reason") if parsed else None,
            "error": err,
            "from_cache": False,
            "model": self.model,
        }
        if usage:
            self.tokens_in += usage[0]
            self.tokens_out += usage[1]
        # only cache successful parses (so a transient failure can be retried later)
        if parsed is not None:
            keep = {k: rec[k] for k in
                    ["s_vlm", "anomaly_score", "category", "reason", "model"]}
            vlm_cache.append(self.cache_root, self.backend, ckey, keep)
            self._index()[ckey] = keep
        return rec

    def _call_with_retries(self, img_b64):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client_obj().messages.create(
                    model=self.model,
                    max_tokens=200,
                    system=self.system,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                            {"type": "text", "text": self.user},
                        ],
                    }, {
                        "role": "assistant", "content": "{"   # prefill -> force JSON
                    }],
                )
                self.n_api_calls += 1
                usage = (msg.usage.input_tokens, msg.usage.output_tokens)
                text = "{" + msg.content[0].text
                parsed = self._parse(text)
                if parsed is not None:
                    return parsed, usage, None
                # one parse-retry is implicit in the loop; record and continue once
                last_err = "parse_fail: {}".format(text[:80])
                if attempt >= 1:
                    return None, usage, last_err
            except Exception as e:  # transient API/network errors
                last_err = "api_err: {}".format(str(e)[:120])
                time.sleep(min(2 ** attempt, 8))
        return None, None, last_err

    @staticmethod
    def _parse(text):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            sc = int(round(float(obj["anomaly_score"])))
            obj["anomaly_score"] = max(0, min(100, sc))
            obj.setdefault("category", "other")
            obj.setdefault("reason", "")
            return obj
        except Exception:
            return None

    def cost(self):
        pin, pout = model_price(self.model)
        return self.tokens_in * pin + self.tokens_out * pout

    def stats(self):
        return {
            "api_calls": self.n_api_calls,
            "cache_hits": self.n_cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "est_cost_usd": round(self.cost(), 4),
        }
