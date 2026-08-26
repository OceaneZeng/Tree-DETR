"""Run one of the primary OWOD baselines with reproducible file logging.

The runner deliberately keeps the baseline name in the command line and in
``run_config.json``.  This prevents a result directory from being mistaken
for an IOD experiment or for a different OWOD method.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow direct execution as ``python tools/owod/run_baseline.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.owod_baselines import BASELINES, baseline_config_dict, normalize_baseline


def _read_stage_metadata(manifest: Path, stage: int | None) -> dict[str, Any]:
    if not manifest:
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if str(payload.get("protocol", "")).lower() not in {"m-owodb", "s-owodb"}:
        raise ValueError(
            "OWOD baseline runs require an M-OWODB or S-OWODB manifest; "
            f"got {payload.get('protocol')!r}")
    result = {"protocol": payload.get("protocol"), "order": payload.get("order"),
              "seed": payload.get("seed"), "memory_fraction": payload.get("memory_fraction")}
    if stage is not None:
        stages = payload.get("stages", [])
        if stage < 0 or stage >= len(stages):
            raise ValueError(f"stage {stage} is outside manifest stages [0, {len(stages)})")
        result["stage"] = stages[stage]
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(BASELINES),
                        help="primary OWOD baseline to run")
    parser.add_argument("--coco-path", required=True, type=Path)
    parser.add_argument("--train-ann", required=True, type=Path)
    parser.add_argument("--val-ann", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pretrained", default=None, type=Path)
    parser.add_argument("--manifest", default="", type=Path,
                        help="M-OWODB/S-OWODB split_manifest.json")
    parser.add_argument("--stage", default=None, type=int)
    parser.add_argument("--num-classes", default=91, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="", help="CUDA_VISIBLE_DEVICES, optional")
    parser.add_argument("--nproc-per-node", default=1, type=int,
                        help="number of DDP processes; use 2 for the requested dual-card run")
    parser.add_argument("--master-port", default=29521, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-main-arg", action="append", default=[],
                        help="Additional argument passed to main.py; repeatable")
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    method = normalize_baseline(args.method)
    main_command = [str(PROJECT_ROOT / "main.py"),
               "--coco_path", str(args.coco_path),
               "--train-ann", str(args.train_ann), "--val-ann", str(args.val_ann),
               "--output_dir", str(args.output_dir),
               "--num_classes", str(args.num_classes), "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
               "--seed", str(args.seed), "--device", args.device,
               "--owod-baseline", method,
               "--log-file", str(args.output_dir / "train.log")]
    nproc_per_node = int(getattr(args, "nproc_per_node", 1))
    master_port = int(getattr(args, "master_port", 29521))
    if nproc_per_node > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node", str(nproc_per_node),
                   "--master_port", str(master_port)] + main_command
    else:
        command = [sys.executable] + main_command
    if args.pretrained is not None:
        command += ["--pretrained", str(args.pretrained)]
    if getattr(args, "stage", None) == 0:
        # The official COCO checkpoint contains future-class classifier rows.
        # Discard them for the OWOD base stage to avoid label leakage.
        command.append("--reset-classifier")
    if getattr(args, "known_class_ids", None):
        command += ["--owod-known-class-ids"] + [str(value) for value in args.known_class_ids]
    for item in args.extra_main_arg:
        command.append(item)
    return command


def write_run_metadata(args: argparse.Namespace, command: list[str], stage_metadata: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "runner": "tools/owod/run_baseline.py",
        "method": normalize_baseline(args.method),
        "baseline": baseline_config_dict(args.method),
        "protocol": stage_metadata,
        "command": command,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8")


def main() -> int:
    args = get_parser().parse_args()
    args.method = normalize_baseline(args.method)
    for path in (args.coco_path, args.train_ann, args.val_ann):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    if args.pretrained is not None and not args.pretrained.exists():
        raise SystemExit(f"missing pretrained checkpoint: {args.pretrained}")
    if args.manifest and not args.manifest.exists():
        raise SystemExit(f"missing OWOD manifest: {args.manifest}")
    stage_metadata = _read_stage_metadata(args.manifest, args.stage)
    args.known_class_ids = stage_metadata.get("stage", {}).get("active_classes", [])
    command = build_command(args)
    write_run_metadata(args, command, stage_metadata)
    print(f"OWOD baseline: {args.method}")
    print(f"Output directory: {args.output_dir}")
    print(f"Human-readable log: {args.output_dir / 'train.log'}")
    if args.dry_run:
        print(shlex.join(command))
        return 0
    import os
    env = os.environ.copy()
    ops_path = str(PROJECT_ROOT / "models" / "ops")
    env["PYTHONPATH"] = ops_path + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
