#!/usr/bin/env python
"""Run detector-free feasibility gates for graph-local continual adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.graph_local.preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON report path")
    args = parser.parse_args()
    report = run_preflight()
    for name, result in report["gates"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {name}: {result['value']:.4f}   gate: {result['gate']}")
    print("\n" + "=" * 60)
    print("PASS  graph-local synthetic preflight" if report["all_passed"] else "FAIL  graph-local synthetic preflight")
    print("Scope: implementation feasibility only; real-data result remains gated by base AP50.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
