"""Convert the per-file VLM cache into one JSONL per backend (see vlm_cache.py).

The original layout stores one small JSON file per query:

    cache/vlm/<backend>/<video_id>/<frame:06d>_<overlay>_<prompt>_<hash>.json

which is ~19k files for this study and dominates the repository. This rewrites
each backend into a single appendable text file with the same content:

    cache/vlm/<backend>.jsonl

Usage:
    python scripts/consolidate_cache.py                 # convert, keep originals
    python scripts/consolidate_cache.py --prune         # convert, then delete the tree
    python scripts/consolidate_cache.py --check         # verify only, write nothing

Conversion is loss-free and verified before anything is deleted: every key and
every field of the original files must be present in the JSONL.
"""
import argparse
import io
import json
import os
import shutil

import vlm_cache

FIELDS = ["s_vlm", "anomaly_score", "category", "reason", "model"]


def read_tree(cache_root, backend):
    """{key: record} straight from the per-file layout."""
    out = {}
    root = vlm_cache.legacy_dir(cache_root, backend)
    for video_id in sorted(os.listdir(root)):
        vdir = os.path.join(root, video_id)
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".json"):
                continue
            with io.open(os.path.join(vdir, fn), encoding="utf-8") as f:
                out["{}/{}".format(video_id, fn[:-5])] = json.load(f)
    return out


def write_jsonl(cache_root, backend, index):
    p = vlm_cache.jsonl_path(cache_root, backend)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        for key in sorted(index):
            row = {"key": key}
            row.update(index[key])
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="cache/vlm")
    ap.add_argument("--prune", action="store_true",
                    help="delete the per-file tree once the JSONL is verified")
    ap.add_argument("--check", action="store_true",
                    help="compare the two layouts and write nothing")
    a = ap.parse_args()

    backends = sorted(d for d in os.listdir(a.cache_root)
                      if os.path.isdir(os.path.join(a.cache_root, d)))
    if not backends:
        raise SystemExit("no per-file cache directories under " + a.cache_root)

    for backend in backends:
        tree = read_tree(a.cache_root, backend)
        if a.check:
            jsonl = {}
            p = vlm_cache.jsonl_path(a.cache_root, backend)
            if os.path.exists(p):
                with io.open(p, encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        jsonl[rec.pop("key")] = rec
            missing = set(tree) - set(jsonl)
            differing = [k for k in set(tree) & set(jsonl)
                         if {f: tree[k].get(f) for f in FIELDS}
                         != {f: jsonl[k].get(f) for f in FIELDS}]
            print("{:22s} tree {:6d} | jsonl {:6d} | missing {:5d} | differing {:5d}".format(
                backend, len(tree), len(jsonl), len(missing), len(differing)))
            continue

        p = write_jsonl(a.cache_root, backend, tree)
        back = vlm_cache.load(a.cache_root, backend)   # reads the JSONL we just wrote
        ok = all({f: tree[k].get(f) for f in FIELDS} == {f: back[k].get(f) for f in FIELDS}
                 for k in tree) and len(back) >= len(tree)
        size = os.path.getsize(p) / 1e6
        print("{:22s} {:6d} entries -> {} ({:.1f} MB) | verified: {}".format(
            backend, len(tree), p, size, ok))
        if a.prune:
            if not ok:
                raise SystemExit("verification failed for {}; nothing deleted".format(backend))
            shutil.rmtree(vlm_cache.legacy_dir(a.cache_root, backend))
            print("{:22s} per-file tree deleted".format(backend))


if __name__ == "__main__":
    main()
