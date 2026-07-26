#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 driver : run the whole falsification suite, cheapest first.
# ------------------------------------------------------------------------
# Order follows the note's dependency / cost graph: F2 (semantics) is cheapest
# and most likely to refute, then the tree-shape checks (F1/F1b), then the
# signal checks (F3/F4/N1), then the geometry feasibility (G1).
#
# With no --features it runs every check on the synthetic fixture (smoke test of
# the plumbing).  Point --features at an npz from common.extract_features to run
# the real Stage-0 battery on a trained detector.
#
#   python tools/stage0/run_stage0.py                 # synthetic smoke test
#   python tools/stage0/run_stage0.py --features f.npz
# ------------------------------------------------------------------------
import argparse
import importlib
import sys

import common

CHECKS = [
    ("F2  tree vs semantics", "f2_tree_vs_semantics"),
    ("F1  tree stability",    "f1_tree_stability"),
    ("F1b DAG check",         "f1b_dag_check"),
    ("F3  halt-depth novelty", "f3_halt_depth_novelty"),
    ("F4  depth vs flat",     "f4_depth_vs_flat"),
    ("N1  norm premise",      "n1_norm_premise"),
    ("G1  annulus feasibility", "g1_annulus"),
]


def main():
    ap = argparse.ArgumentParser(description="Run the full Tree-DETR Stage-0 suite")
    common.add_common_args(ap)
    args = ap.parse_args()

    results = []
    for title, mod_name in CHECKS:
        print(f"\n=== {title} ===")
        mod = importlib.import_module(mod_name)
        # each check module exposes main() returning 0 (pass) / 1 (fail)
        argv = sys.argv[1:]
        old = sys.argv
        sys.argv = [mod_name] + argv
        try:
            rc = mod.main()
        finally:
            sys.argv = old
        results.append((title, rc))

    print("\n" + "=" * 60)
    n_pass = sum(1 for _t, rc in results if rc == 0)
    for title, rc in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {title}")
    print("=" * 60)
    print(f"Stage-0: {n_pass}/{len(results)} checks passed")
    # Stage-0 is a battery of independent gates; report but do not hard-fail the
    # process on a single red check (the synthetic fixture should pass them all).
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
