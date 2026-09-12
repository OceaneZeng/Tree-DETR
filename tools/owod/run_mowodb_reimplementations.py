#!/usr/bin/env python3
"""Run CAT and EW-DETR paper reimplementations on official M-OWODB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, stage_files
from tools.owod.run_paper_baselines import (
    annotation_subset, baseline_complete, combine_annotations, payload_hash,
    queue_lock, run_child, select_memory, summarize, validate_fingerprints,
    verify_checkpoint, write_json,
)
from tools.owod.run_stage1_diagnostics import training_environment


METHODS = ("cat", "ew-detr")
DEFAULT_MANIFEST = ROOT / "data/coco-owod/m-owodb/split_manifest.json"
DEFAULT_COCO = ROOT / "data/coco"
DEFAULT_PROPOSALS = ROOT / "data/derived/m-owodb-cat/selective_search.sqlite3"
DEFAULT_OUTPUT = ROOT / "exps/owod/m-owodb/order0/official/baselines"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_methods(values):
    values = tuple(values)
    if not values or len(values) != len(set(values)):
        raise ValueError("Require one or more unique baseline methods")
    unknown = set(values) - set(METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")
    return values


def state_directory(output_dir, methods):
    return output_dir if len(methods) > 1 else output_dir / methods[0]


def create_plan(args):
    methods_to_run = selected_methods(args.methods)
    manifest_path = args.manifest.resolve()
    coco_path = args.coco_path.resolve()
    manifest = read_json(manifest_path)
    if (manifest.get("protocol") != "m-owodb"
            or (not args.allow_unverified and not manifest.get("official_annotations"))
            or (not args.allow_unverified and not manifest.get("paper_comparable"))):
        raise ValueError(
            "Require a validated, paper_comparable official M-OWODB manifest, "
            "or pass --allow-unverified for a local consistency run")
    if len(manifest.get("stages", [])) != 4:
        raise ValueError("M-OWODB requires exactly four stages")
    if len(args.gpus.split(",")) != 2 or len(set(args.gpus.split(","))) != 2:
        raise ValueError("This comparison requires two distinct GPU indices")

    fingerprints = {str(manifest_path): file_sha256(manifest_path)}
    proposal_path = args.cat_proposals.resolve()
    if "cat" in methods_to_run:
        if not proposal_path.is_file():
            raise FileNotFoundError(f"CAT proposal database is missing: {proposal_path}")
        with sqlite3.connect(proposal_path.as_uri() + '?mode=ro', uri=True) as connection:
            proposal_metadata = dict(connection.execute('SELECT key, value FROM metadata'))
            proposal_count = connection.execute(
                'SELECT COUNT(*) FROM image_proposals').fetchone()[0]
        if proposal_metadata.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("CAT proposals were generated from a different manifest")
        expected_images = sum(record["increment_image_count"]
                              for record in manifest["stages"])
        if proposal_count != expected_images:
            raise ValueError(
                f"CAT proposal database is incomplete: {proposal_count}/{expected_images} images")
        fingerprints[str(proposal_path)] = file_sha256(proposal_path)

    runs = {}
    previous_classes = []
    previous_samples = 0
    memory = {"images": [], "annotations": []}
    all_images = set()
    stage_records = []
    for stage in range(4):
        checked_manifest, files = stage_files(
            manifest_path, stage, allow_unverified=args.allow_unverified)
        record = checked_manifest["stages"][stage]
        current_classes = list(record["classes"])
        if (len(current_classes) != 20 or set(current_classes) & set(previous_classes)):
            raise ValueError("Expected four disjoint 20-class increments")
        known_classes = previous_classes + current_classes
        increment = read_json(files["increment_train"])
        full_val = read_json(files["full_val"])
        if not {ann["category_id"] for ann in increment["annotations"]}.issubset(current_classes):
            raise ValueError(f"Stage {stage} increment contains non-current supervision")
        for key in ("increment_train", "full_val"):
            fingerprints[str(files[key])] = file_sha256(files[key])
        all_images.update(coco_path / "train2017" / image["file_name"]
                          for image in increment["images"])
        all_images.update(coco_path / "val2017" / image["file_name"]
                          for image in full_val["images"])

        cat_train = combine_annotations(increment, memory)
        replay_count = sum(bool(image.get("owod_replay"))
                           for image in cat_train["images"])
        stage_records.append({
            "stage": stage,
            "current_classes": current_classes,
            "known_classes": known_classes,
            "increment_images": len(increment["images"]),
            "cat_replay_images": replay_count,
            "ew_replay_images": 0,
        })
        for method in methods_to_run:
            directory = args.output_dir / method / f"stage_{stage}"
            annotation = directory / "train.json"
            payload = (cat_train if method == "cat" else annotation_subset(
                increment, [image["id"] for image in increment["images"]],
                current_classes))
            epochs = args.stage0_epochs if stage == 0 else args.incremental_epochs
            lr_drop = args.stage0_lr_drop if stage == 0 else args.incremental_lr_drop
            command = [
                sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                "--nproc_per_node=2", "--master_addr=127.0.0.1",
                f"--master_port={args.master_port}", str(ROOT / "main.py"),
                "--paper-baseline", method, "--coco_path", str(coco_path),
                "--train-ann", str(annotation), "--val-ann", str(files["full_val"]),
                "--output_dir", str(directory), "--owod-manifest", str(manifest_path),
                "--owod-stage", str(stage), "--num_classes", "92", "--lr_backbone", "0",
                "--lr", str(args.lr), "--weight_decay", str(args.weight_decay),
                "--epochs", str(epochs), "--lr_drop", str(lr_drop),
                "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
                "--seed", str(args.seed), "--num_queries", str(args.num_queries),
                "--enc_layers", str(args.enc_layers), "--dec_layers", str(args.dec_layers),
                "--unknown-threshold", str(args.unknown_threshold),
                "--eval_interval", str(args.eval_interval), "--no-file-log",
                "--print-freq", "100", "--eval-print-freq", "100",
                "--ow-pseudo-warmup", str(args.pseudo_warmup),
                "--ow-top-unknown", str(args.top_unknown),
                "--owod-known-class-ids", *map(str, known_classes),
                "--owod-current-class-ids", *map(str, current_classes),
            ]
            if stage:
                previous_name = ("checkpoint_consolidated.pth"
                                 if method == "ew-detr" else "checkpoint.pth")
                command.extend([
                    "--pretrained", str(directory.parent / f"stage_{stage - 1}" / previous_name),
                    "--owod-previous-class-ids", *map(str, previous_classes),
                ])
            if method == "cat":
                command.extend([
                    "--cat-proposals", str(proposal_path),
                    "--cat-loss-memory", str(args.cat_loss_memory),
                    "--cat-recent-window", str(args.cat_recent_window),
                    "--cat-start-iteration", str(args.cat_start_iteration),
                    "--cat-update-interval", str(args.cat_update_interval),
                    "--cat-positive-momentum", str(args.cat_positive_momentum),
                    "--cat-negative-momentum", str(args.cat_negative_momentum),
                ])
                if stage:
                    command.extend(["--replay-sampling-fraction", str(args.replay_fraction)])
            else:
                command.extend([
                    "--ew-rank", str(args.ew_rank),
                    "--ew-current-samples", str(len(increment["images"])),
                    "--ew-previous-samples", str(previous_samples),
                ])
            runs[f"{method}/stage_{stage}"] = {
                "command": command,
                "annotation": payload,
                "annotation_sha256": payload_hash(payload),
                "epochs": epochs,
            }
        memory = select_memory(cat_train, known_classes, args.memory_images,
                               args.seed + stage)
        previous_classes = known_classes
        previous_samples += len(increment["images"])

    missing = [str(path) for path in all_images if not path.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} COCO images missing; first: {missing[:3]}")
    code_paths = [ROOT / "main.py", ROOT / "engine.py", Path(__file__),
                  ROOT / "tools/owod/protocol.py",
                  ROOT / "tools/owod/verify_paper_checkpoint.py"]
    for folder in ("models", "datasets", "util"):
        code_paths.extend((ROOT / folder).rglob("*.py"))
    for path in code_paths:
        fingerprints[str(path.relative_to(ROOT))] = file_sha256(path)
    plan = {
        "schema": 2,
        "protocol": "official_M-OWODB_20x4",
        "paper_comparable_protocol": not args.allow_unverified,
        "validation_mode": "unverified_local" if args.allow_unverified else "official",
        "implementation": "from-paper reimplementation; not author source",
        "methods": list(methods_to_run),
        "gpus": args.gpus,
        "manifest": str(manifest_path),
        "stage_records": stage_records,
        "assumptions": {
            "shared_schedule": "50 epochs stage 0; 20 epochs stages 1-3 by default",
            "initialization": "repository ImageNet ResNet-50; no future-class detector weights",
            "optimizer": "AdamW with explicit shared learning rate and weight decay",
            "cat_controller": "paper omits controller hyperparameters; explicit CLI values recorded",
            "cat_selective_search": "OpenCV cache; configuration recorded in proposal database",
            "cat_replay": "bounded replay images are mixed into training batches",
            "ew_protocol_transfer": "EWOD method transferred to static-domain M-OWODB",
        },
        "fingerprints": fingerprints,
        "runs": {key: {field: value for field, value in run.items()
                       if field != "annotation"} for key, run in runs.items()},
    }
    return plan, runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coco-path", type=Path, default=DEFAULT_COCO)
    parser.add_argument("--cat-proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--master-port", type=int, default=29579)
    parser.add_argument("--memory-images", type=int, default=400)
    parser.add_argument("--replay-fraction", type=float, default=0.1)
    parser.add_argument("--stage0-epochs", type=int, default=50)
    parser.add_argument("--incremental-epochs", type=int, default=20)
    parser.add_argument("--stage0-lr-drop", type=int, default=40)
    parser.add_argument("--incremental-lr-drop", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--enc-layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--unknown-threshold", type=float, default=0.5)
    parser.add_argument("--pseudo-warmup", type=int, default=9)
    parser.add_argument("--top-unknown", type=int, default=5)
    parser.add_argument("--ew-rank", type=int, default=16)
    parser.add_argument("--cat-loss-memory", type=int, default=100)
    parser.add_argument("--cat-recent-window", type=int, default=20)
    parser.add_argument("--cat-start-iteration", type=int, default=100)
    parser.add_argument("--cat-update-interval", type=int, default=100)
    parser.add_argument("--cat-positive-momentum", type=float, default=0.01)
    parser.add_argument("--cat-negative-momentum", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Use the local/pilot manifest and mark the run non-paper-comparable",
    )
    args = parser.parse_args(argv)
    args.methods = selected_methods(args.methods)
    args.manifest = args.manifest.resolve()
    args.coco_path = args.coco_path.resolve()
    args.cat_proposals = args.cat_proposals.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.summarize:
        summarize(args.output_dir, args.methods)
        return 0
    if (args.memory_images < 80 or not 0 < args.replay_fraction < 1
            or min(args.stage0_epochs, args.incremental_epochs,
                   args.batch_size, args.num_queries, args.enc_layers,
                   args.dec_layers, args.lr) <= 0 or args.weight_decay < 0):
        raise ValueError("Invalid schedule, replay, memory, or architecture setting")
    plan, runs = create_plan(args)
    print(json.dumps({key: plan[key] for key in (
        "protocol", "implementation", "stage_records", "assumptions")}, indent=2))
    for key, run in runs.items():
        print(key, shlex.join(run["command"]))
    state_dir = state_directory(args.output_dir, args.methods)
    plan_path = state_dir / "reimplementation_plan.json"
    if args.dry_run:
        print("Dry run passed. No output files or checkpoints were created.")
        return 0
    environment = training_environment(args.gpus)
    with queue_lock(state_dir):
        if plan_path.is_file() and read_json(plan_path) != plan:
            raise ValueError("Existing plan differs; choose a new output directory")
        write_json(plan_path, plan)
        for key, run in runs.items():
            directory = args.output_dir / key
            annotation_path = directory / "train.json"
            if annotation_path.is_file() and read_json(annotation_path) != run["annotation"]:
                raise ValueError(f"Training annotation changed: {annotation_path}")
            write_json(annotation_path, run["annotation"])
        write_json(state_dir / "reimplementation_queue_status.json",
                   {"status": "running", "pid": os.getpid()})
        try:
            validate_fingerprints(plan)
            for method in args.methods:
                for stage in range(4):
                    key = f"{method}/stage_{stage}"
                    run = runs[key]
                    directory = args.output_dir / key
                    if baseline_complete(directory, run["epochs"], method):
                        verify_checkpoint(directory, run["epochs"], method, stage, environment)
                        continue
                    command = list(run["command"])
                    if args.resume and (directory / "checkpoint.pth").is_file():
                        command.extend(["--resume", str(directory / "checkpoint.pth")])
                    write_json(state_dir / "reimplementation_queue_status.json",
                               {"status": "running", "run": key, "pid": os.getpid()})
                    run_child(command, directory / "console.log", environment)
                    if not baseline_complete(directory, run["epochs"], method):
                        raise RuntimeError(f"{key} ended without complete artifacts")
                    verify_checkpoint(directory, run["epochs"], method, stage, environment)
        except BaseException as error:
            write_json(state_dir / "reimplementation_queue_status.json",
                       {"status": "failed", "error": str(error)})
            raise
        write_json(state_dir / "reimplementation_queue_status.json",
                   {"status": "complete"})
        summarize(args.output_dir, args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
