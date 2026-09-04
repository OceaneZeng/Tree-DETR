#!/usr/bin/env python
"""Run the local Deformable DETR control on one validated OWOD stage."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.owod_detector import detector_profile_dict
from tools.owod.protocol import stage_files
from util.experiment_log import prepare_log_file


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pretrained", default=None, type=Path)
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--num-classes", default=91, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nproc-per-node", default=2, type=int)
    parser.add_argument("--master-port", default=29521, type=int)
    parser.add_argument("--lr-drop", default=15, type=int)
    parser.add_argument("--eval-interval", default=5, type=int)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--eval-print-freq", default=100, type=int)
    parser.add_argument("--unknown-threshold", default=0.5, type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace, record: dict, files: dict[str, Path]) -> list[str]:
    command = [
        str(PROJECT_ROOT / "main.py"),
        "--coco_path", str(args.coco_path),
        "--train-ann", str(files["train"]),
        "--val-ann", str(files["full_val"]),
        "--output_dir", str(args.output_dir),
        "--num_classes", str(args.num_classes),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed), "--device", "cuda",
        "--owod-manifest", str(args.manifest), "--owod-stage", str(args.stage),
        "--owod-known-class-ids", *[str(value) for value in record["active_classes"]],
        "--owod-current-class-ids", *[str(value) for value in record["classes"]],
        "--unknown-threshold", str(args.unknown_threshold),
        "--lr_drop", str(args.lr_drop),
        "--eval_interval", str(args.eval_interval),
        "--print-freq", str(args.print_freq),
        "--eval-print-freq", str(args.eval_print_freq), "--no-file-log",
    ]
    previous = sorted(set(record["active_classes"]) - set(record["classes"]))
    if previous:
        command += ["--owod-previous-class-ids", *[str(value) for value in previous]]
    if args.resume:
        command += ["--resume", str(args.resume)]
    elif args.pretrained:
        command += ["--pretrained", str(args.pretrained)]
        if args.stage == 0:
            command.append("--reset-classifier")
    if args.nproc_per_node > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(args.nproc_per_node), "--master_port",
                str(args.master_port), *command]
    return [sys.executable, *command]


def main() -> int:
    args = get_parser().parse_args()
    args.coco_path = args.coco_path.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.coco_path.is_dir():
        raise SystemExit(f"missing COCO path: {args.coco_path}")
    manifest, files = stage_files(args.manifest, args.stage)
    record = manifest["stages"][args.stage]
    if args.resume and not args.resume.is_file():
        raise SystemExit(f"missing resume checkpoint: {args.resume}")
    if args.pretrained and not args.pretrained.is_file():
        raise SystemExit(f"missing pretrained checkpoint: {args.pretrained}")
    command = build_command(args, record, files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "runner": "tools/owod/run_detector_control.py",
        "detector_profile": detector_profile_dict(),
        "protocol": manifest["protocol"], "paper_comparable": manifest["paper_comparable"],
        "stage": args.stage, "command": command,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    print("Local control: Deformable DETR (not a published Table 1 baseline)")
    if args.dry_run:
        print(shlex.join(command))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "models" / "ops") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    train_log = prepare_log_file(args.output_dir / "train.log", append=bool(args.resume))
    with train_log.open("a" if args.resume else "w", encoding="utf-8", buffering=1) as handle:
        event = "resume" if args.resume else "start"
        handle.write(f"===== experiment {event} {datetime.now().isoformat(timespec='seconds')} =====\n")
        handle.write(f"Command: {shlex.join(command)}\n")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        code = int(process.wait())
        handle.write(f"===== experiment end return_code={code} =====\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
