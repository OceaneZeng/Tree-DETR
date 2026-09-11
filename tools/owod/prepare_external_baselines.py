#!/usr/bin/env python3
"""Prepare auditable external OWOD baseline checkouts and command manifests.

This tool does not port an external method into Tree-DETR. PROB and OWOBJ use
their own Deformable-DETR forks, VOC-style data views, and evaluators. CAT's
public code URL currently returns 404, so it is recorded as pending rather
than replaced by a local approximation.  The generated commands are the
authors' M-OWODB recipes, not Tree-DETR's COCO protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL = ROOT / "exps" / "external_baselines" / "repos"
METHODS = {
    "prob": {
        "repo": "https://github.com/orrzohar/PROB.git",
        "entry": "main_open_world.py",
        "config": "configs/M_OWOD_BENCHMARK.sh",
        "dataset": "TOWOD",
        "note": "official PROB repository; M-OWODB recipe",
    },
    "owobj": {
        "repo": "https://github.com/AI4Math-ShanZhang/OWOBJ.git",
        "entry": "main_open_world.py",
        "config": "configs/M_OWOD_BENCHMARK.sh",
        "dataset": "OWDETR",
        "note": "official OWOBJ repository; M-OWODB recipe",
    },
    "cat": {
        "repo": None,
        "entry": None,
        "config": None,
        "dataset": None,
        "note": "CVPR 2023 CAT source URL currently unavailable (HTTP 404)",
    },
}


def local_source(method, source_root):
    """Return a checked-in/previously cloned source directory when present."""
    if source_root is None:
        return None
    names = {"prob": {"prob", "PROB"}, "owobj": {"owobj", "OWOBJ"},
             "cat": {"cat", "CAT"}}[method]
    if not source_root.is_dir():
        return None
    # Match case-insensitively while returning the directory's actual spelling;
    # generated commands must also work on case-sensitive Linux filesystems.
    for candidate in source_root.iterdir():
        if candidate.is_dir() and candidate.name.lower() in {name.lower() for name in names} \
                and (candidate / "main_open_world.py").is_file():
            return candidate
    return None


def run(command, cwd=None):
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def checkout(method, root, repo_override=None, source_root=None):
    spec = METHODS[method]
    repo = repo_override or spec["repo"]
    override_path = Path(repo_override).expanduser() if repo_override else None
    directory = (override_path if override_path and override_path.is_dir()
                 else local_source(method, source_root) or (root / method))
    if not repo:
        return {"method": method, "status": "pending_source", "note": spec["note"]}
    # A local source is intentionally never mutated. This is useful on a
    # server where the repositories have already been audited or patched.
    if override_path and override_path.is_dir() or local_source(method, source_root):
        pass
    elif not (directory / ".git").is_dir():
        directory.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", repo, str(directory)])
    else:
        run(["git", "fetch", "--depth", "1", "origin", "HEAD"], cwd=directory)
        run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=directory)
    expected = tuple(name for name in (spec["entry"], spec["config"]) if name)
    missing = [name for name in expected if not (directory / name).is_file()]
    try:
        commit = run(["git", "rev-parse", "HEAD"], cwd=directory)
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "method": method,
        "status": ("ready" if not missing else "invalid_checkout") if expected
                  else "source_found_config_pending",
        "repo": repo,
        "commit": commit,
        "path": str(directory),
        "entry": spec["entry"],
        "config": spec["config"],
        "dataset": spec["dataset"],
        "missing": missing,
        "note": spec["note"],
        "source": "local" if local_source(method, source_root) else "cloned",
    }


def command_for(record, gpus):
    if record["status"] != "ready" or not record.get("config"):
        return None
    # The official scripts expect to be run from the external repository and
    # use their own VOC/TOWOD data and evaluator. Keep this explicit in output.
    count = len([item for item in gpus.split(',') if item.strip()])
    config = record["config"]
    shared = Path(record["path"]) / config.replace(".sh", "_SHARED.sh")
    if shared.is_file():
        config = str(Path(config).with_name(Path(config).stem + "_SHARED.sh"))
    return (f"cd {record['path']} && "
            f"CUDA_VISIBLE_DEVICES={gpus} GPUS_PER_NODE={count} "
            f"./tools/run_dist_launch.sh {count} {config} "
            + ("" if config.endswith("_SHARED.sh") else
               "--dataset TOWOD --data_root ./data/OWOD"))


def stage_commands(record, gpus):
    """Emit explicit official training/evaluation commands for reproducibility."""
    if record.get("status") != "ready" or not record.get("config"):
        return []
    root = record["path"]
    count = len([item for item in gpus.split(',') if item.strip()])
    prefix = f"cd {root} && CUDA_VISIBLE_DEVICES={gpus} GPUS_PER_NODE={count}"
    suffix = "--dataset TOWOD --data_root ./data/OWOD"
    train_config = "configs/M_OWOD_BENCHMARK_SHARED.sh" if (Path(root) / "configs/M_OWOD_BENCHMARK_SHARED.sh").is_file() else "configs/M_OWOD_BENCHMARK.sh"
    eval_config = "configs/EVAL_M_OWOD_BENCHMARK_SHARED.sh" if (Path(root) / "configs/EVAL_M_OWOD_BENCHMARK_SHARED.sh").is_file() else "configs/EVAL_M_OWOD_BENCHMARK.sh"
    train = f"{prefix} ./tools/run_dist_launch.sh {count} {train_config} {'' if train_config.endswith('_SHARED.sh') else suffix}"
    evaluate = f"{prefix} ./tools/run_dist_launch.sh {count} {eval_config} {'' if eval_config.endswith('_SHARED.sh') else suffix}"
    return {"train": train, "evaluate": evaluate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS),
                        default=list(METHODS))
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Use existing method directories under this root without updating them")
    parser.add_argument("--cat-repo", help="CAT checkout URL/path if authors provide one")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shared-mowodb", action="store_true",
                        help="Mark commands as using the registered shared M-OWODB VOC view")
    parser.add_argument("--no-clone", action="store_true",
                        help="Only inspect existing checkouts")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "exps" / "external_baselines" / "external_baseline_manifest.json")
    args = parser.parse_args(argv)
    records = []
    for method in args.methods:
        override = args.cat_repo if method == "cat" else None
        if args.no_clone:
            spec = METHODS[method]
            path = args.repo_root / method
            records.append({"method": method,
                            "status": "pending_source" if not spec["repo"] and not override else "not_checked",
                            "path": str(path), "note": spec["note"]})
        else:
            records.append(checkout(method, args.repo_root, override, args.source_root))
    if args.shared_mowodb:
        for record in records:
            if record.get("method") in {"prob", "owobj"} and record.get("status") == "ready":
                root = Path(record["path"])
                missing_shared = [name for name in (
                    "configs/M_OWOD_BENCHMARK_SHARED.sh",
                    "configs/EVAL_M_OWOD_BENCHMARK_SHARED.sh",
                ) if not (root / name).is_file()]
                if missing_shared:
                    record["status"] = "invalid_shared_setup"
                    record["missing_shared"] = missing_shared
                    record["note"] = (
                        "Run prepare_shared_mowodb.py first; shared 20/20/20/20 configs are missing")
    payload = {
        "schema": 1,
        "paper_comparable": bool(args.shared_mowodb and all(
            r["status"] == "ready" for r in records)),
        "protocol": ("shared_m-owodb_v1; external author evaluator"
                     if args.shared_mowodb else
                     "official_external_recipe; not Tree-DETR COCO matched"),
        "records": records,
        "commands": {r["method"]: command_for(r, args.gpus) for r in records},
        "stage_commands": {r["method"]: stage_commands(r, args.gpus) for r in records},
        "cat_source": "https://github.com/xiaomabufei/CAT (currently unavailable)",
        "limitations": [
            "PROB and OWOBJ retain their own repositories and evaluators; only the data view and 20/20/20/20 protocol are shared.",
            "The shared flag is a metadata claim and requires the converter plus OWOBJ adapter patch to have been applied.",
            "CAT is not runnable until a verified source checkout is supplied.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if any(r["status"] == "invalid_checkout" for r in records):
        raise SystemExit("One or more external checkouts are missing expected files")
    if any(r["status"] == "invalid_shared_setup" for r in records):
        raise SystemExit("Run prepare_shared_mowodb.py before requesting shared commands")


if __name__ == "__main__":
    main()
