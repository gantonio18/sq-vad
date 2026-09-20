"""Phase 8 — regenerate EVERY figure/table from results/*.parquet alone.

No VLM is re-queried and the flow is not re-run; this reads only the cached
parquet/csv/json artifacts, so all thesis plots are reproducible offline.

  python scripts/make_figs.py
"""
import importlib
import traceback

# (module, friendly name); each has a main() that reads only results/*
STEPS = [
    ("plot_baseline", "Phase 2 baseline figures"),
    ("error_analysis", "Phase 3 error-analysis figures"),
    ("plot_vlm", "Phase 4 VLM figures"),
    ("plot_fusion", "Phase 6 fusion figures"),
    ("evaluate", "Phase 7 comparison table + bootstrap"),
]


def main():
    for mod_name, label in STEPS:
        print("\n=== {} ({}) ===".format(label, mod_name))
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            mod.main()
        except Exception:
            print("!! {} FAILED:".format(mod_name))
            traceback.print_exc()
    print("\nAll figures regenerated from results/*.")


if __name__ == "__main__":
    main()
