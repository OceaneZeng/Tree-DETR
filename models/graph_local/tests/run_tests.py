"""Dependency-light runner for graph-local module tests."""

from __future__ import annotations

import importlib
import os
import sys
import traceback


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("models.graph_local.tests.test_preflight")
    failures = []
    test_names = sorted(item for item in dir(module) if item.startswith("test_"))
    for name in test_names:
        try:
            getattr(module, name)()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}  -- {type(exc).__name__}: {exc}")
    print(f"\nran {len(test_names)}  |  failed {len(failures)}")
    for name, trace in failures:
        print(f"\n--- {name} ---\n{trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
