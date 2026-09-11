#!/usr/bin/env python3
"""Prepare auditable external OWOD baseline checkouts and command manifests.

This tool does not port an external method into Tree-DETR. PROB and OWOBJ use
their own Deformable-DETR forks, VOC-style data views, and evaluators. CAT's
public code URL currently returns 404, so it is recorded as pending rather
than replaced by a local approximation.
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


def run(command, cwd=None):
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def checkout(method, root, repo_override=None):
    spec = METHODS[method]
    repo = repo_override or spec["repo"]
    directory = root / method
    if not repo:
        return {"method": method, "status": "pending_source", "note": spec["note"]}
    if not (directory / ".git").is_dir():
        directory.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", repo, str(directory)])
    else:
        run(["git", "fetch", "--depth", "1", "origin", "HEAD"], cwd=directory)
        run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=directory)
    missing = [name for name in (spec["entry"], spec["config"])
               if not (directory / name).is_file()]
    commit = run(["git", "rev-parse", "HEAD"], cwd=directory)
    return {
        "method": method,
        "status": "ready" if not missing else "invalid_checkout",
        "repo": repo,
        "commit": commit,
        "path": str(directory),
        "entry": spec["entry"],
        "config": spec["config"],
        "dataset": spec["dataset"],
        "missing": missing,
        "note": spec["note"],
    }


def command_for(record, gpus):
    if record["status"] != "ready":
        return None
    # The official scripts expect to be run from the external repository and
    # use their own VOC/TOWOD data and evaluator. Keep this explicit in output.
    return (f"cd {record['path']} && "
            f"CUDA_VISIBLE_DEVICES={gpus} bash {record['config']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS),
                        default=list(METHODS))
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--cat-repo", help="CAT checkout URL/path if authors provide one")
    parser.add_argument("--gpus", default="0,1")
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
            records.append(checkout(method, args.repo_root, override))
    payload = {
        "schema": 1,
        "paper_comparable": False,
        "protocol": "official_external_recipe; not Tree-DETR COCO matched",
        "records": records,
        "commands": {r["method"]: command_for(r, args.gpus) for r in records},
        "cat_source": "https://github.com/xiaomabufei/CAT (currently unavailable)",
        "limitations": [
            "PROB and OWOBJ use their own repositories, data views and evaluators.",
            "Their official recipes are not exact runs on Tree-DETR's internal unverified manifest.",
            "CAT is not runnable until a verified source checkout is supplied.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if any(r["status"] == "invalid_checkout" for r in records):
        raise SystemExit("One or more external checkouts are missing expected files")


if __name__ == "__main__":
    main()
