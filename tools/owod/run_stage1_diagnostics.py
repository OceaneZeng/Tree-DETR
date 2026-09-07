#!/usr/bin/env python
"""Run matched Stage 1 diagnostics using a completed pilot's exact training data.

This launcher uses only the standard library. Training runs in the invoking
Python environment. It never rebuilds the graph or replay annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "exps/owod/m-owodb/order0/pilot_unverified"
ARM_NAMES = {"d1": "d1_lora_no_extra_losses", "d2": "d2_full_finetune_no_extra_losses",
             "d3": "d3_lora_all_decoder_train_detection_heads",
             "d4": "d4_frozen_backbone_finetune"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_command(command):
    main_index = next(i for i, value in enumerate(command) if Path(value).name == "main.py")
    options = {}
    key = None
    for value in command[main_index + 1:]:
        if value.startswith("--"):
            if value in options:
                raise ValueError(f"Duplicate source option: {value}")
            key = value
            options[key] = []
        elif key is None:
            raise ValueError(f"Unexpected positional argument: {value}")
        else:
            options[key].append(value)
    return list(command[:main_index]), options


def build_command(source_command, arm, output_dir, port, resume=False):
    prefix, options = split_command(source_command)
    prefix[0] = sys.executable
    if "--master_port" in prefix:
        prefix[prefix.index("--master_port") + 1] = str(port)
    for key in ("--resume", "--log-file", "--append-log", "--off-neighborhood-basis"):
        options.pop(key, None)
    options["--output_dir"] = [str(output_dir)]
    options["--local-margin-coef"] = ["0"]
    options["--off-projection-coef"] = ["0"]
    options["--no-file-log"] = []
    if arm in ("d2", "d4"):
        options.pop("--neighbor-scoped-lora", None)
        options.pop("--trainable-class-ids", None)
        if arm == "d4":
            for key in ("--lora-rank", "--lora-last-decoder-layers", "--lora-train-detection-heads"):
                options.pop(key, None)
            # build_backbone sets requires_grad=False for all backbone weights
            # when this is zero; all non-backbone modules keep normal training.
            options["--lr_backbone"] = ["0"]
    elif arm == "d3":
        options["--lora-train-detection-heads"] = []
        options["--lora-last-decoder-layers"] = ["6"]
    elif arm != "d1":
        raise ValueError(f"Unknown diagnostic: {arm}")
    if resume:
        options["--resume"] = [str(output_dir / "checkpoint.pth")]
    return prefix + [str(ROOT / "main.py")] + [
        token for key, values in options.items() for token in [key, *values]]


def load_source(source):
    config = read_json(source / "run_config.json")
    recorded = read_json(source / "graph/run_config.json")
    complete = read_json(source / "graph/training_complete.json")
    _, options = split_command(config["command"])
    if (recorded.get("owod_stage") != 1 or not recorded.get("neighbor_scoped_lora")
            or not recorded.get("teacher_completion") or recorded.get("resume")
            or "--neighbor-scoped-lora" not in options
            or "--teacher-completion" not in options or "--resume" in options):
        raise ValueError("Source must be a fresh Stage 1 LoRA pilot with teacher completion")
    if complete.get("last_epoch") != recorded["epochs"] - 1:
        raise ValueError("Source pilot has not completed its configured training schedule")
    if recorded.get("lora_train_detection_heads", False):
        raise ValueError("Source must use the original new-class-only LoRA update policy")
    for key in ("--eval", "--skip-eval", "--old-class-distillation", "--lora-train-detection-heads"):
        if key in options:
            raise ValueError(f"Unexpected source option: {key}")
    for option, field in (("--train-ann", "train_ann"), ("--val-ann", "val_ann"),
                          ("--pretrained", "pretrained"), ("--epochs", "epochs"),
                          ("--lr", "lr"), ("--seed", "seed")):
        if options.get(option) != [str(recorded[field])]:
            raise ValueError(f"Source command and recorded configuration differ: {option}")
    return config, options


def training_environment(gpus):
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpus,
               NCCL_IB_DISABLE="1", NCCL_P2P_DISABLE="1",
               TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "models/ops"), str(ROOT), env.get("PYTHONPATH", "")])
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or not torch_spec.origin:
        raise ValueError("PyTorch is missing; activate the tree-detr environment")
    torch_lib = Path(torch_spec.origin).parent / "lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib), env.get("LD_LIBRARY_PATH", "")])
    return env


def metric_rows(path):
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(f"Pending partial final metric line: {path}")
                break
            raise
        if "test_owod_current_ap50" in row:
            rows[int(row["epoch"])] = row
    return rows


def summarize(source, destination):
    baseline = metric_rows(source / "graph/metrics.jsonl")
    print("\nAP50 / H in percent; delta is against D0 at the same epoch.")
    print("run epoch previous current known H delta_previous delta_current status")
    for label, directory in [("d0", source / "graph")] + [
            (arm, destination / name / "graph") for arm, name in ARM_NAMES.items()]:
        status = "complete" if (directory / "training_complete.json").is_file() else "incomplete"
        for epoch, row in sorted(metric_rows(directory / "metrics.jsonl").items()):
            keys = ("previous_ap50", "current_ap50", "known_ap50", "h_score")
            values = [f"{100 * row['test_owod_' + key]:.3f}"
                      if 'test_owod_' + key in row else "NA" for key in keys]
            reference = baseline.get(epoch, {})
            changes = [f"{100 * (row[key] - reference[key]):+.3f}"
                       if key in reference and key in row else "NA"
                       for key in ("test_owod_previous_ap50", "test_owod_current_ap50")]
            print(label, epoch + 1, *values, *changes, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PILOT / "full_three_module_stage1_v1")
    parser.add_argument("--output-dir", type=Path, default=PILOT / "stage1_diagnostics_v1")
    parser.add_argument("--experiment", choices=("d1", "d2", "d3", "d4", "both"), default="both",
                        help="both keeps the original D1 then D2 schedule; D3/D4 must be requested explicitly")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--master-port", type=int, default=29567)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    source, destination = args.source_run.resolve(), args.output_dir.resolve()
    if args.summarize:
        summarize(source, destination)
        return 0
    config, options = load_source(source)
    if args.experiment == "d3":
        recorded = read_json(source / "graph/run_config.json")
        if recorded.get("dec_layers", 6) != 6:
            raise ValueError("D3 requires the source six-layer decoder")
    prefix, _ = split_command(config["command"])
    processes = int(prefix[prefix.index("--nproc_per_node") + 1]) if "--nproc_per_node" in prefix else 1
    if len(args.gpus.split(",")) != processes:
        raise ValueError(f"Source uses {processes} processes; supply exactly that many GPUs")
    inputs = {key: Path(options[key][0]) for key in
              ("--train-ann", "--val-ann", "--pretrained", "--owod-manifest")}
    inputs["source_graph"] = source / "graph.json"
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Missing source input: {path}")
    arms = ["d1", "d2"] if args.experiment == "both" else [args.experiment]
    # Hash once per invocation; both experiments use the identical source files.
    fingerprints = {key: {"path": str(path), "sha256": file_hash(path)}
                    for key, path in inputs.items()}
    plans = []
    for arm in arms:
        arm_root = destination / ARM_NAMES[arm]
        output = arm_root / "graph"
        command = build_command(config["command"], arm, output, args.master_port)
        plan = {"schema_version": 1, "experiment": arm, "source_run": str(source),
                "inputs": fingerprints, "command": command, "gpus": args.gpus,
                "selected_replay_classes": config["selected_replay_classes"]}
        plan_file = arm_root / "diagnostic_plan.json"
        if plan_file.is_file():
            if read_json(plan_file) != plan:
                raise ValueError(f"Plan or source inputs changed: {plan_file}; use a new output directory")
            if (output / "training_complete.json").is_file():
                print(f"Already complete: {arm_root}")
                continue
            if not args.resume or not (output / "checkpoint.pth").is_file():
                raise ValueError(f"Existing incomplete run: {arm_root}; use --resume with a checkpoint")
            command = build_command(config["command"], arm, output, args.master_port, resume=True)
        elif arm_root.exists() and any(arm_root.iterdir()):
            raise ValueError(f"Nonempty output without a diagnostic plan: {arm_root}")
        print(f"\n{arm}: exact D0 annotation, graph selection, teacher and schedule")
        if arm == "d3":
            print("D3: LoRA in all six decoder FFNs plus full detection heads; backbone and other base parameters frozen")
        if arm == "d4":
            print("D4: no LoRA; freeze the entire backbone via lr_backbone=0; fine-tune encoder, decoder, projections and detection heads")
        print(shlex.join(command))
        plans.append((arm_root, output, plan, command))
    if args.dry_run:
        print("\nDry run passed. No training or output files were created.")
        return 0
    if plans:
        env = training_environment(args.gpus)
    for arm_root, output, plan, command in plans:
        output.mkdir(parents=True, exist_ok=True)
        (arm_root / "diagnostic_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        # Capture distributed startup errors as well as the rank-zero training log.
        with (output / "train.log").open("a", encoding="utf-8", buffering=1) as log:
            log.write("\n===== diagnostic launch =====\n" + shlex.join(command) + "\n")
            # Torchrun installs its own SIGHUP handler even under nohup.
            # A separate POSIX session avoids terminal-session hangup signals.
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  start_new_session=(os.name == "posix")) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            log.write(f"===== diagnostic exit {code} =====\n")
        if code:
            print(f"Stopped after failed experiment: {arm_root}; the next experiment was not started")
            return code
        if not (output / "training_complete.json").is_file():
            raise ValueError(f"Process exited without completion marker: {output}")
        summarize(source, destination)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError, StopIteration) as error:
        print(f"Diagnostic preflight failed: {error}", file=sys.stderr)
        sys.exit(1)
