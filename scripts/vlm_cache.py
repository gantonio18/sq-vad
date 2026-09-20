"""On-disk cache of VLM verdicts, stored as one JSONL file per backend.

Layout
------
    cache/vlm/<backend>.jsonl

with one JSON object per line:

    {"key": "<video_id>/<frame:06d>_<overlay>_<prompt_version>_<imghash>",
     "s_vlm": ..., "anomaly_score": ..., "category": ..., "reason": ..., "model": ...}

The key is exactly the old per-file cache path minus the ".json" suffix, so the
two layouts are interchangeable. The original layout (one small file per query,
`cache/vlm/<backend>/<video_id>/<frame>_<overlay>_<prompt>_<hash>.json`) is still
read when no JSONL is present, or for keys missing from it, so existing caches
keep working; `scripts/consolidate_cache.py` converts one to the other.

Why JSONL: the study caches ~19k queries, and as individual files they dominate
the repository and make it slow to clone and impossible to browse. One appendable
text file per backend is the same data, still greppable and diffable.
"""
import io
import json
import os


def jsonl_path(cache_root, backend):
    return os.path.join(cache_root, "{}.jsonl".format(backend))


def legacy_dir(cache_root, backend):
    return os.path.join(cache_root, backend)


def parse_key(key):
    """'01_0014/000012_none_v2_ab12cd34' -> (video_id, frame_idx, overlay, prompt, hash)."""
    video_id, name = key.split("/", 1)
    frame, rest = name.split("_", 1)
    overlay, prompt, imghash = rest.rsplit("_", 2)
    return video_id, int(frame), overlay, prompt, imghash


def load(cache_root, backend):
    """Return {key: record}, preferring the JSONL and falling back to the old tree."""
    index = {}
    p = jsonl_path(cache_root, backend)
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue          # never let one bad line lose the whole cache
                key = rec.pop("key", None)
                if key:
                    index[key] = rec
    d = legacy_dir(cache_root, backend)
    if os.path.isdir(d):
        for video_id in os.listdir(d):
            vdir = os.path.join(d, video_id)
            if not os.path.isdir(vdir):
                continue
            for fn in os.listdir(vdir):
                if not fn.endswith(".json"):
                    continue
                key = "{}/{}".format(video_id, fn[:-5])
                if key in index:
                    continue          # the JSONL wins
                try:
                    with io.open(os.path.join(vdir, fn), encoding="utf-8") as f:
                        index[key] = json.load(f)
                except ValueError:
                    continue
    return index


def append(cache_root, backend, key, rec):
    """Append one verdict. Creates the file on first write."""
    if not os.path.isdir(cache_root):
        os.makedirs(cache_root)
    row = {"key": key}
    row.update(rec)
    with io.open(jsonl_path(cache_root, backend), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def entries(cache_root, backend, prompt_version=None, overlay=None):
    """Yield (video_id, frame_idx, overlay, prompt_version, record) for every entry."""
    for key, rec in load(cache_root, backend).items():
        try:
            video_id, frame_idx, ov, pv, _ = parse_key(key)
        except ValueError:
            continue
        if prompt_version is not None and pv != prompt_version:
            continue
        if overlay is not None and ov != overlay:
            continue
        yield video_id, frame_idx, ov, pv, rec
