#!/usr/bin/env python
"""Audit server-side experiment artifacts and remove only verified redundancy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


CACHE_NAMES = {"__pycache__", ".pytest_cache"}
FIXED_CACHE_PATHS = (Path("build"), Path("models/ops/build"))
DUPLICATE_LOG_NAMES = {
    "console.log", "launcher.log", "preflight.log", "resume_launcher.log"
}
CHECKPOINT_SNAPSHOT = re.compile(r"checkpoint\d{4}\.pth$")
PRUNED_SOURCE_DIRS = {".git", "data", "pretrained", "exps", "log"}


def byte_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def directory_size(path: Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in files:
            total += byte_size(current_path / name)
    return total


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identical(left: Path, right: Path) -> bool:
    return (left.is_file() and right.is_file()
            and byte_size(left) == byte_size(right)
            and digest(left) == digest(right))


def relative_record(path: Path, root: Path, reason: str) -> dict:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": directory_size(path) if path.is_dir() else byte_size(path),
        "reason": reason,
    }


def source_caches(root: Path) -> list[Path]:
    caches: list[Path] = []
    for current, dirs, _files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept = []
        for name in dirs:
            candidate = current_path / name
            if candidate.is_symlink():
                continue
            if name in CACHE_NAMES:
                caches.append(candidate)
            elif current_path == root and name in PRUNED_SOURCE_DIRS:
                continue
            else:
                kept.append(name)
        dirs[:] = kept
    for relative in FIXED_CACHE_PATHS:
        candidate = root / relative
        if candidate.is_dir() and not candidate.is_symlink():
            caches.append(candidate)
    return sorted(set(caches))


def experiment_records(root: Path, large_log_bytes: int) -> tuple[list[dict], list[dict]]:
    experiments: list[dict] = []
    review: list[dict] = []
    experiment_root = root / "exps"
    if not experiment_root.is_dir():
        return experiments, review
    for current, _dirs, files in os.walk(experiment_root, followlinks=False):
        directory = Path(current)
        names = set(files)
        if "training_complete.json" in names:
            experiments.append(relative_record(
                directory, root, "completed_experiment"))
        elif "checkpoint.pth" in names:
            review.append(relative_record(
                directory / "checkpoint.pth", root, "incomplete_experiment_checkpoint"))
        for name in files:
            path = directory / name
            if name.endswith(".log") and byte_size(path) >= large_log_bytes:
                review.append(relative_record(path, root, "large_log"))
            elif CHECKPOINT_SNAPSHOT.fullmatch(name):
                review.append(relative_record(path, root, "periodic_checkpoint"))
    return experiments, review


def duplicate_logs(root: Path) -> tuple[list[Path], list[dict]]:
    safe: set[Path] = set()
    review: list[dict] = []
    experiment_root = root / "exps"
    if experiment_root.is_dir():
        for current, _dirs, files in os.walk(experiment_root, followlinks=False):
            directory = Path(current)
            names = set(files)
            canonical = directory / "train.log"
            for name in sorted(names & DUPLICATE_LOG_NAMES):
                candidate = directory / name
                if identical(candidate, canonical):
                    safe.add(candidate)
                else:
                    review.append(relative_record(
                        candidate, root, "nonidentical_legacy_launcher_log"))

    legacy_root = root / "log"
    if legacy_root.is_dir():
        for current, _dirs, files in os.walk(legacy_root, followlinks=False):
            directory = Path(current)
            for name in files:
                candidate = directory / name
                relative = candidate.relative_to(legacy_root)
                canonical = root / relative
                if identical(candidate, canonical):
                    safe.add(candidate)
                else:
                    review.append(relative_record(
                        candidate, root, "unmatched_legacy_central_log"))
    return sorted(safe), review


def audit(root: Path, large_log_mib: int = 50) -> dict:
    root = root.resolve()
    caches = source_caches(root)
    duplicates, duplicate_review = duplicate_logs(root)
    experiments, experiment_review = experiment_records(
        root, max(1, large_log_mib) * 1024 * 1024)
    return {
        "project_root": str(root),
        "safe_delete": {
            "cache_directories": [relative_record(
                path, root, "rebuildable_cache") for path in caches],
            "exact_duplicate_logs": [relative_record(
                path, root, "sha256_matches_canonical_log") for path in duplicates],
        },
        "experiments": experiments,
        "review_only": sorted(
            duplicate_review + experiment_review,
            key=lambda item: (item["reason"], item["path"])),
    }


def checked_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    target.relative_to(root)
    return target


def apply_safe(report: dict) -> list[str]:
    root = Path(report["project_root"]).resolve()
    removed: list[str] = []
    for record in report["safe_delete"]["exact_duplicate_logs"]:
        target = checked_path(root, record["path"])
        if target.is_file():
            target.unlink()
            removed.append(record["path"])
    for record in report["safe_delete"]["cache_directories"]:
        target = checked_path(root, record["path"])
        allowed = target.name in CACHE_NAMES or target.relative_to(root) in FIXED_CACHE_PATHS
        if target.is_dir() and allowed and not target.is_symlink():
            shutil.rmtree(target)
            removed.append(record["path"])
    return removed


def total(records: list[dict]) -> int:
    return sum(int(record["bytes"]) for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--large-log-mib", type=int, default=50)
    parser.add_argument("--apply-safe", action="store_true",
                        help="delete only rebuildable caches and byte-identical log copies")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "main.py").is_file() or not (root / "tools/owod").is_dir():
        raise SystemExit(f"not a Tree-DETR project root: {root}")
    report = audit(root, args.large_log_mib)
    cache_records = report["safe_delete"]["cache_directories"]
    duplicate_records = report["safe_delete"]["exact_duplicate_logs"]
    print(f"Project: {root}")
    print(f"Safe caches: {len(cache_records)} ({total(cache_records) / 2**20:.1f} MiB)")
    print(f"Exact duplicate logs: {len(duplicate_records)} "
          f"({total(duplicate_records) / 2**20:.1f} MiB)")
    print(f"Completed experiment directories: {len(report['experiments'])}")
    print(f"Review-only artifacts: {len(report['review_only'])}")
    if report["review_only"]:
        print("Review-only examples:")
        for record in report["review_only"][:20]:
            print(f"  {record['reason']}: {record['path']} ({record['bytes'] / 2**20:.1f} MiB)")
    if args.apply_safe:
        removed = apply_safe(report)
        report["removed"] = removed
        print(f"Removed safe artifacts: {len(removed)}")
    else:
        print("Dry run only. Add --apply-safe after reviewing the report.")
    if args.report_json:
        report_path = args.report_json.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
