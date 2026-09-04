#!/usr/bin/env python
"""Run graph-local replay for one M-OWODB/S-OWODB OWOD stage.

The graph is a train-time class-level controller around the detector.  The
main method extracts decoder gradient sketches for current and previous
classes, predicts directed new-to-old interference with a GNN calibrated on
an earlier stage, and turns the selected old classes into a replay annotation.
The actual detector training/evaluation is delegated to ``main.py`` so the
standard full-label OWOD metrics are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.graph_local.gnn import compress_gradient_sketches, load_gnn_checkpoint
from models.graph_local.replay import build_increment_annotation
from tools.owod.gnn_calibration import (build_calibration_dataset,
                                        compute_gradient_sketches,
                                        load_detector)
from tools.owod.protocol import stage_files
from util.experiment_log import prepare_log_file


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="previous stage detector checkpoint, or COCO pretrained checkpoint for stage 0")
    parser.add_argument("--resume", default=None, type=Path,
                        help="incomplete current-stage checkpoint; restores optimizer and epoch")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-classes", default=91, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nproc-per-node", default=2, type=int)
    parser.add_argument("--master-port", default=29561, type=int)
    parser.add_argument("--gnn-checkpoint", default=None, type=Path,
                        help="required for the graph arm; random/global are matched controls")
    parser.add_argument("--graph-k", default=5, type=int)
    parser.add_argument("--gnn-min-score", default=0.0, type=float)
    parser.add_argument("--graph-feature-device", default="cuda:0",
                        help="device used to extract detector gradient sketches")
    parser.add_argument("--sketch-max-images", default=12, type=int)
    parser.add_argument("--sketch-batch-size", default=1, type=int)
    parser.add_argument("--last-decoder-layers", default=2, type=int)
    parser.add_argument("--exemplars-per-class", default=None, type=int,
                        help="required for incremental tasks; use the value from the official protocol")
    parser.add_argument("--control", choices=("graph", "random", "global"), default="graph")
    parser.add_argument("--eval-interval", default=5, type=int)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--eval-print-freq", default=100, type=int)
    parser.add_argument("--lr-drop", default=15, type=int)
    parser.add_argument("--unknown-threshold", default=0.5, type=float)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def rank_stage_old_classes(class_ids: list[int], edge_scores: torch.Tensor,
                           new_ids: list[int], old_ids: list[int], k: int,
                           min_score: float = 0.0) -> tuple[list[int], dict]:
    """Select a stage-level top-k over old classes only.

    A target's risk is the maximum directed edge from any current-stage class.
    This treats ``k`` as the total replay neighborhood size, rather than taking
    k neighbors from all classes for each source and filtering afterwards.
    """
    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError("edge_scores must be a square matrix")
    if len(class_ids) != edge_scores.shape[0]:
        raise ValueError("class_ids and edge_scores must have matching lengths")

    index = {int(class_id): position for position, class_id in enumerate(class_ids)}
    missing = sorted((set(new_ids) | set(old_ids)) - set(index))
    if missing:
        raise KeyError(f"Graph scores are missing classes: {missing}")

    target_ids = list(dict.fromkeys(int(class_id) for class_id in old_ids))
    source_scores: dict[str, list[list[float | int]]] = {}
    aggregate = {target: float("-inf") for target in target_ids}
    detached = edge_scores.detach().float().cpu()
    for source in new_ids:
        ranked = []
        for target in target_ids:
            score = float(detached[index[int(source)], index[target]])
            aggregate[target] = max(aggregate[target], score)
            ranked.append([target, score])
        ranked.sort(key=lambda item: (-item[1], item[0]))
        source_scores[str(int(source))] = ranked

    aggregate_ranking = sorted(
        ([target, score] for target, score in aggregate.items() if score > min_score),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [int(target) for target, _score in aggregate_ranking[:max(0, int(k))]]
    return selected, {
        "selection_scope": "stage_top_k_old_classes",
        "aggregation": "max_over_new_classes",
        "requested_k": int(k),
        "selected_k": len(selected),
        "min_score": float(min_score),
        "candidate_old_classes": target_ids,
        "aggregated_scores": aggregate_ranking,
        "scores": source_scores,
    }


def select_neighbors(args, features: dict[int, torch.Tensor], new_ids: list[int],
                     old_ids: list[int]) -> tuple[list[int], dict]:
    if not old_ids or not new_ids or args.control == "global":
        return (list(old_ids) if args.control == "global" else []), {"estimator": "none"}
    gnn, metadata = load_gnn_checkpoint(args.gnn_checkpoint, device="cpu")
    if metadata.get("supervision") != "empirical_train_loss_increase":
        raise ValueError(
            "The GNN checkpoint is not empirically supervised. Run "
            "tools/owod/calibrate_interference_gnn.py on a completed earlier stage."
        )
    if not metadata.get("production_ready", False):
        raise ValueError(
            "The GNN checkpoint came from a partial smoke calibration. "
            "Run calibration without --source-limit before the detector experiment."
        )
    ordered, compressed = compress_gradient_sketches(features, output_dim=gnn.input_dim)
    with torch.no_grad():
        edge_prob = gnn(compressed)["edge_prob"]
    selected, details = rank_stage_old_classes(
        ordered, edge_prob, new_ids, old_ids, args.graph_k, args.gnn_min_score)
    return selected, {"estimator": "trainable_class_interference_gnn",
                      "checkpoint": str(args.gnn_checkpoint.resolve()),
                      "feature_source": "detector_decoder_ffn_gradient_sketch",
                      "score_semantics": "predicted_directed_train_loss_increase",
                      "metadata": metadata, **details}


def detector_gradient_features(args, current_ann: Path, replay_ann: Path,
                               new_ids: list[int], old_ids: list[int]
                               ) -> tuple[dict[int, torch.Tensor], dict]:
    """Extract real detector gradients for the GNN nodes at the current stage."""
    from main import get_args_parser

    detector_args = get_args_parser().parse_args([])
    detector_args.device = args.graph_feature_device
    detector_args.num_classes = args.num_classes
    detector_args.coco_path = str(args.coco_path)
    detector_args.dataset_file = "coco"
    detector_args.masks = False
    detector_args.cache_mode = False
    detector_args.two_stage = False
    detector_args.lightweight = args.lightweight
    device = torch.device(args.graph_feature_device)
    model, criterion = load_detector(detector_args, args.checkpoint, device)
    old_dataset = build_calibration_dataset(args.coco_path, replay_ann, args.lightweight)
    new_dataset = build_calibration_dataset(args.coco_path, current_ann, args.lightweight)
    datasets = {class_id: old_dataset for class_id in old_ids}
    datasets.update({class_id: new_dataset for class_id in new_ids})
    class_ids = list(dict.fromkeys(old_ids + new_ids))
    sketches, counts = compute_gradient_sketches(
        model, criterion, datasets, class_ids, device,
        cache_dir=args.output_dir / "node_sketches",
        cache_identity={"checkpoint": str(args.checkpoint),
                        "old_annotation": str(replay_ann),
                        "new_annotation": str(current_ann)},
        batch_size=args.sketch_batch_size,
        num_workers=args.num_workers,
        max_images=args.sketch_max_images,
        last_decoder_layers=args.last_decoder_layers,
    )
    del model, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sketches, {
        "feature_source": "detector_decoder_ffn_gradient_sketch",
        "old_annotation": str(replay_ann),
        "new_annotation": str(current_ann),
        "matched_annotations": counts,
        "max_images_per_class": args.sketch_max_images,
        "last_decoder_layers": args.last_decoder_layers,
        "uses_validation_labels": False,
    }


def random_neighbors(old_ids: list[int], graph_k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(old_ids, min(len(old_ids), max(0, graph_k))))


def stage_paths(manifest_path: Path, stage: int, coco_path: Path
                ) -> tuple[dict, Path, Path, Path]:
    del coco_path  # image root is unrelated to annotation provenance
    manifest, current_files = stage_files(manifest_path, stage)
    _prior_manifest, replay_files = stage_files(manifest_path, max(0, stage - 1))
    return (manifest, current_files["increment_train"], replay_files["train"],
            current_files["full_val"])


def build_main_command(args, train_ann: Path, val_ann: Path, arm_dir: Path,
                       active_ids: list[int], old_ids: list[int], reset_classifier: bool) -> list[str]:
    main = [str(ROOT / "main.py"), "--coco_path", str(args.coco_path),
            "--train-ann", str(train_ann), "--val-ann", str(val_ann),
            "--output_dir", str(arm_dir), "--num_classes", str(args.num_classes),
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--seed", str(args.seed),
            "--device", "cuda",
            "--owod-manifest", str(args.manifest), "--owod-stage", str(args.stage),
            "--owod-known-class-ids", *[str(value) for value in active_ids],
            "--owod-current-class-ids", *[str(value) for value in active_ids if value not in set(old_ids)],
            "--unknown-threshold", str(args.unknown_threshold),
            "--eval_interval", str(args.eval_interval), "--lr_drop", str(args.lr_drop),
            "--print-freq", str(args.print_freq),
            "--eval-print-freq", str(args.eval_print_freq),
            "--no-file-log"]
    if old_ids:
        main += ["--owod-previous-class-ids", *[str(value) for value in old_ids]]
    if args.resume:
        # Load the completed previous-stage model before restoring the current
        # stage optimizer/scheduler state.
        if args.stage > 0 and old_ids:
            main += ["--pretrained", str(args.checkpoint)]
        main += ["--resume", str(args.resume)]
    else:
        main += ["--pretrained", str(args.checkpoint)]
    if reset_classifier and not args.resume:
        main.append("--reset-classifier")
    if args.lightweight:
        main.append("--lightweight")
    if args.no_random_crop:
        main.append("--no-random-crop")
    if args.skip_eval:
        main.append("--skip-eval")
    if args.nproc_per_node > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(args.nproc_per_node), "--master_port",
                str(args.master_port)] + main
    return [sys.executable] + main


def stream_process(command: list[str], output_dir: Path, env: dict[str, str],
                   dry_run: bool, append: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if dry_run:
        print(shlex.join(command))
        return 0
    train_log = prepare_log_file(output_dir / "train.log", append=append)
    with train_log.open("a" if append else "w", encoding="utf-8", buffering=1) as handle:
        event = "resume" if append else "start"
        handle.write(f"===== experiment {event} {datetime.now().isoformat(timespec='seconds')} =====\n")
        handle.write(f"Command: {shlex.join(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return_code = int(process.wait())
        handle.write(f"===== experiment end return_code={return_code} =====\n")
        return return_code


def main() -> int:
    args = get_parser().parse_args()
    args.coco_path = args.coco_path.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.gnn_checkpoint:
        args.gnn_checkpoint = args.gnn_checkpoint.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest, current_ann, replay_ann, full_val = stage_paths(args.manifest, args.stage,
                                                              args.coco_path)
    record = manifest["stages"][args.stage]
    new_ids = [int(value) for value in record["classes"]]
    active_ids = [int(value) for value in record["active_classes"]]
    old_ids = [int(value) for value in manifest["category_order_source_ids"]
               if int(value) in set(active_ids) - set(new_ids)]
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.resume and not args.resume.is_file():
        raise SystemExit(f"missing resume checkpoint: {args.resume}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = args.output_dir / "annotations"
    graph_path = args.output_dir / "graph.json"
    if args.resume:
        train_ann = annotation_dir / f"train_{args.control}.json"
        if args.stage == 0:
            train_ann = replay_ann
        if not graph_path.is_file() or not train_ann.is_file():
            raise SystemExit(
                "resume requires the original graph.json and training annotation; "
                f"missing one of: {graph_path}, {train_ann}")
        saved_graph = read_json(graph_path)
        selected = [int(value) for value in saved_graph.get("selected_replay_classes", [])]
        graph_info = saved_graph.get("graph", {})
        replay_info = saved_graph.get("replay", {})
        print(f"Resume fast path: reusing {graph_path}", flush=True)
        print(f"Resume fast path: reusing {train_ann}", flush=True)
    else:
        feature_info = {}
        features = None
        if args.control == "global":
            selected = list(old_ids)
            graph_info = {"estimator": "none", "control": "global",
                          "selected_k": len(selected)}
        elif args.control == "random":
            selected = random_neighbors(old_ids, args.graph_k, args.seed)
            graph_info = {"estimator": "none", "control": "random",
                          "requested_k": args.graph_k, "selected_k": len(selected),
                          "seed": args.seed}
        else:
            print(f"Loading graph features from {args.checkpoint}", flush=True)
            if args.gnn_checkpoint is None or not args.gnn_checkpoint.is_file():
                raise SystemExit(f"missing GNN checkpoint: {args.gnn_checkpoint}")
            features, feature_info = detector_gradient_features(
                args, current_ann, replay_ann, new_ids, old_ids)
            selected, graph_info = select_neighbors(args, features, new_ids, old_ids)
        if args.stage == 0 or not selected:
            train_ann = current_ann if args.stage > 0 else replay_ann
            replay_info = {"replay_classes": selected, "replay_images": 0,
                           "total_images": len(read_json(train_ann).get("images", []))}
        else:
            if args.exemplars_per_class is None or args.exemplars_per_class <= 0:
                raise SystemExit(
                    "incremental replay requires --exemplars-per-class from the official "
                    "OWOD recipe; no paper value is assumed")
            train_ann = annotation_dir / f"train_{args.control}.json"
            replay_info = build_increment_annotation(
                current_ann, replay_ann, selected, args.exemplars_per_class,
                train_ann, args.seed)

        if features is not None:
            gnn, _metadata = load_gnn_checkpoint(args.gnn_checkpoint, device="cpu")
            graph_class_ids, graph_features = compress_gradient_sketches(
                features, output_dim=gnn.input_dim)
            torch.save({
                "schema_version": 1,
                "class_ids": graph_class_ids,
                "features": graph_features,
                "source_classes": new_ids,
                "target_classes": old_ids,
                "metadata": feature_info,
            }, args.output_dir / "gnn_node_features.pt")
        write_json(graph_path, {
            "stage": args.stage, "new_classes": new_ids, "old_classes": old_ids,
            "selected_replay_classes": selected, "graph": graph_info,
            "replay": replay_info, **feature_info,
        })

    arm_dir = args.output_dir / args.control
    command = build_main_command(args, train_ann, full_val, arm_dir, active_ids, old_ids,
                                 args.stage == 0)
    metadata = {"runner": "tools/owod/run_graph_local_increment.py", "stage": args.stage,
                "new_classes": new_ids, "old_classes": old_ids,
                "active_classes": active_ids, "selected_replay_classes": selected,
                "graph_estimator": "gnn" if args.control == "graph" else "none",
                "exemplars_per_class": args.exemplars_per_class,
                "checkpoint": str(args.checkpoint),
                "resume": str(args.resume) if args.resume else "",
                "command": command}
    write_json(args.output_dir / "run_config.json", metadata)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "models" / "ops") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    code = stream_process(command, arm_dir, env, args.dry_run, append=bool(args.resume))
    write_json(args.output_dir / "summary.json", {**metadata, "return_code": code,
                "train_log": str((arm_dir / "train.log").resolve()),
                "metrics_log": str((arm_dir / "metrics.jsonl").resolve())})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
