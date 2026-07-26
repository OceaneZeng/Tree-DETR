"""Dependency-free test runner for the Tree-DETR package.

Discovers every ``test_*`` function in this directory's ``test_*.py`` modules and
runs them, printing a per-test PASS/FAIL line and a summary.  No pytest needed
(the target env has only torch + numpy).

Run from the repo root so ``import models.tree...`` resolves:

    python models/tree/tests/run_tests.py
    C:/Users/23642/miniconda3/envs/deep-infor-agent/python.exe models/tree/tests/run_tests.py
"""
import importlib
import os
import sys
import traceback
import types


def _ensure_repo_root_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _install_models_stub(repo_root: str):
    """Register a bare ``models`` package whose ``__path__`` points at the real
    directory but WITHOUT executing ``models/__init__.py``.

    The base package's ``__init__`` imports the full DeformableDETR (and the
    compiled MSDeformAttn op / a matching torchvision), which need not be present
    to unit-test the tree package.  Submodules like ``models.tree.geometry`` and
    even the lazy ``models.deformable_detr`` remain locatable via ``__path__``;
    only the broken top-level body is skipped.  If the real package imports
    cleanly (proper training env), we leave it untouched.
    """
    if "models" in sys.modules:
        return
    try:
        importlib.import_module("models")
        return                                   # real package is fine; use it
    except Exception:
        stub = types.ModuleType("models")
        stub.__path__ = [os.path.join(repo_root, "models")]
        sys.modules["models"] = stub


def main() -> int:
    repo_root = _ensure_repo_root_on_path()
    _install_models_stub(repo_root)
    here = os.path.dirname(os.path.abspath(__file__))
    modules = sorted(
        f[:-3] for f in os.listdir(here)
        if f.startswith("test_") and f.endswith(".py")
    )

    total = passed = failed = 0
    failures = []
    for modname in modules:
        mod = importlib.import_module(f"models.tree.tests.{modname}")
        fns = sorted(n for n in dir(mod) if n.startswith("test_"))
        for fn in fns:
            total += 1
            label = f"{modname}.{fn}"
            try:
                getattr(mod, fn)()
                passed += 1
                print(f"PASS  {label}")
            except Exception as e:                       # noqa: BLE001
                failed += 1
                failures.append((label, e, traceback.format_exc()))
                print(f"FAIL  {label}  --  {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"ran {total}  |  passed {passed}  |  failed {failed}")
    if failures:
        print("=" * 60)
        for label, _e, tb in failures:
            print(f"\n--- {label} ---")
            print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
