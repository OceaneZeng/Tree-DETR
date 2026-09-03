#!/usr/bin/env python
"""Run graph-local replay for one M-OWODB/S-OWODB OWOD stage.

The graph is a class-level controller around the detector.  It reads the
previous checkpoint's classifier prototypes, scores directed new-to-old
interference edges, and turns the selected old classes into a replay
annotation.  The actual detector training/evaluation is delegated to
``main.py`` so the standard OWOD objectness and full-label metrics are kept.

Use ``--graph-estimator cosine`` for the first stage.  A later stage may use
``--graph-estimator gnn`` with a GNN trained only from earlier ``gnn_stage.pt``
artifacts via ``tools/graph_local/train_gnn.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.graph_local.gnn import compress_gradient_sketches, load_gnn_checkpoint
from models.graph_local.interference import build_conflict_matrix
from models.graph_local.replay import build_increment_annotation


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
    parser.add_argument("--owod-baseline", default="vanilla_d_detr")
    parser.add_argument("--num-classes", default=91, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nproc-per-node", default=2, type=int)
    parser.add_argument("--master-port", default=29561, type=int)
    parser.add_argument("--graph-estimator", choices=("cosine", "gnn"), default="cosine")
    parser.add_argument("--gnn-checkpoint", default="", type=Path)
    parser.add_argument("--graph-k", default=5, type=int)
    parser.add_argument("--gnn-min-score", default=0.0, type=float)
    parser.add_argument("--replay-budget", default=256, type=int)
    parser.add_argument("--control", choices=("graph", "random", "global"), default="graph")
    parser.add_argument("--eval-interval", default=5, type=int)
    parser.add_argument("--lr-drop", default=15, type=int)
    parser.add_argument("--unknown-threshold", default=0.5, type=float)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_state(path: Path) -> Mapping[str, torch.Tensor]:
    # Older project checkpoints contain an argparse.Namespace alongside the
    # tensors; allow that harmless metadata while still using weights_only.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        # PyTorch 2.4 has no safe_globals. This is an explicitly supplied
        # local checkpoint, matching main.load_local_checkpoint behaviour.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with safe_globals([argparse.Namespace]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is not a state dict: {path}")
    return state


def classifier_features(checkpoint: Path, class_ids: list[int]) -> dict[int, torch.Tensor]:
    state = load_state(checkpoint)
    key = next((name for name in state
                if name.endswith("class_embed.0.weight")), None)
    if key is None:
        raise KeyError("checkpoint has no class_embed.0.weight; cannot build class graph")
    weights = state[key].float()
    if max(class_ids) >= weights.shape[0]:
        raise ValueError(f"checkpoint classifier has {weights.shape[0]} rows, needs {max(class_ids) + 1}")
    return {class_id: weights[class_id].reshape(-1).cpu() for class_id in class_ids}


def build_prototype_similarity_matrix(features: dict[int, torch.Tensor]
                                      ) -> tuple[list[int], torch.Tensor]:
    """Return positive cosine similarity for classifier prototypes.

    Classifier weights are class prototypes, not gradient sketches. Similar
    prototype directions indicate greater confusion risk; negative cosine is
    reserved for actual gradient-conflict measurements.
    """
    if len(features) < 2:
        raise ValueError("At least two class prototypes are required")
    class_ids = sorted(int(class_id) for class_id in features)
    rows = torch.stack([features[class_id].float().cpu() for class_id in class_ids])
    rows = F.normalize(rows, dim=1, eps=1e-12)
    similarity = torch.clamp(rows @ rows.t(), min=0.0)
    similarity.fill_diagonal_(0.0)
    return class_ids, similarity


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
    if args.graph_estimator == "cosine":
        class_ids, similarity = build_prototype_similarity_matrix(features)
        selected, details = rank_stage_old_classes(
            class_ids, similarity, new_ids, old_ids, args.graph_k, 0.0)
        return selected, {
            "estimator": "classifier_prototype_cosine_similarity",
            "score_semantics": "max_positive_cosine_similarity",
            **details,
        }
    if not args.gnn_checkpoint:
        raise ValueError("--gnn-checkpoint is required for --graph-estimator gnn")
    gnn, metadata = load_gnn_checkpoint(args.gnn_checkpoint, device="cpu")
    ordered, compressed = compress_gradient_sketches(features, output_dim=gnn.input_dim)
    with torch.no_grad():
        edge_prob = gnn(compressed)["edge_prob"]
    selected, details = rank_stage_old_classes(
        ordered, edge_prob, new_ids, old_ids, args.graph_k, args.gnn_min_score)
    return selected, {"estimator": "trainable_class_interference_gnn",
                      "checkpoint": str(args.gnn_checkpoint.resolve()),
                      "metadata": metadata, **details}


def random_neighbors(old_ids: list[int], graph_k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(old_ids, min(len(old_ids), max(0, graph_k))))


def stage_paths(manifest_path: Path, stage: int, coco_path: Path) -> tuple[dict, Path, Path, Path]:
    manifest = read_json(manifest_path)
    stages = manifest.get("stages", [])
    if stage < 0 or stage >= len(stages):
        raise ValueError(f"stage {stage} is outside manifest [0, {len(stages)})")
    root = manifest_path.parent
    record = stages[stage]
    stage_dir = root / f"stage_{stage}"
    current = stage_dir / "instances_increment_train2017.json"
    if not current.is_file():
        current = stage_dir / "instances_increment_only_train2017.json"
    seen = stage_dir / "instances_train2017.json"
    val = stage_dir / "instances_val2017_full.json"
    if not val.is_file():
        val = stage_dir / "instances_val2017.json"
    if not val.is_file():
        # IOD-style manifests may not materialize a full-label file.  The
        # source COCO validation annotation is the correct OWOD fallback.
        val = coco_path / "annotations" / "instances_val2017.json"
    for path in (current, seen, val):
        if not path.is_file():
            raise FileNotFoundError(f"manifest stage file is missing: {path}")
    return manifest, current, seen, val


def build_main_command(args, train_ann: Path, val_ann: Path, arm_dir: Path,
                       active_ids: list[int], old_ids: list[int], reset_classifier: bool) -> list[str]:
    main = [str(ROOT / "main.py"), "--coco_path", str(args.coco_path),
            "--train-ann", str(train_ann), "--val-ann", str(val_ann),
            "--output_dir", str(arm_dir), "--num_classes", str(args.num_classes),
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--seed", str(args.seed),
            "--device", "cuda", "--owod-baseline", args.owod_baseline,
            "--owod-known-class-ids", *[str(value) for value in active_ids],
            "--owod-current-class-ids", *[str(value) for value in active_ids if value not in set(old_ids)],
            "--unknown-threshold", str(args.unknown_threshold),
            "--eval_interval", str(args.eval_interval), "--lr_drop", str(args.lr_drop),
            "--log-file", str(arm_dir / "train.log")]
    if old_ids:
        main += ["--owod-previous-class-ids", *[str(value) for value in old_ids]]
    if args.resume:
        # main.py loads --pretrained first to construct the frozen teacher,
        # then --resume restores the current-stage detector/optimizer state.
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
    if args.stage > 0 and old_ids:
        main += ["--teacher-completion", "--teacher-old-class-ids",
                 *[str(value) for value in old_ids]]
    if args.nproc_per_node > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(args.nproc_per_node), "--master_port",
                str(args.master_port)] + main
    return [sys.executable] + main


def stream_process(command: list[str], output_dir: Path, env: dict[str, str], dry_run: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    if dry_run:
        print(" ".join(command))
        return 0
    console = output_dir / "console.log"
    with console.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return int(process.wait())


def main() -> int:
    args = get_parser().parse_args()
    args.coco_path = args.coco_path.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest, current_ann, seen_ann, full_val = stage_paths(args.manifest, args.stage,
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
            train_ann = seen_ann
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
        all_ids = [int(value) for value in manifest["category_order_source_ids"]]
        print(f"Loading graph features from {args.checkpoint}", flush=True)
        features = classifier_features(args.checkpoint, all_ids)
        selected, graph_info = select_neighbors(args, features, new_ids, old_ids)
        if args.control == "random":
            selected = random_neighbors(old_ids, len(selected), args.seed)
        if args.stage == 0 or not selected:
            train_ann = current_ann if args.stage > 0 else seen_ann
            replay_info = {"replay_classes": selected, "replay_images": 0,
                           "total_images": len(read_json(train_ann).get("images", []))}
        else:
            train_ann = annotation_dir / f"train_{args.control}.json"
            replay_info = build_increment_annotation(
                current_ann, seen_ann, selected, args.replay_budget, train_ann, args.seed)

        graph_class_ids, graph_features = compress_gradient_sketches(features, output_dim=128)
        _, conflict = build_conflict_matrix(features)
        harm = torch.zeros_like(conflict)
        valid = torch.zeros_like(conflict, dtype=torch.bool)
        index = {class_id: i for i, class_id in enumerate(graph_class_ids)}
        for source in new_ids:
            for target in old_ids:
                harm[index[source], index[target]] = conflict[index[source], index[target]]
                valid[index[source], index[target]] = True
        torch.save({"schema_version": 2, "class_ids": graph_class_ids,
                    "features": graph_features, "harm": harm, "valid_mask": valid,
                    "source_classes": new_ids, "target_classes": old_ids},
                   args.output_dir / "gnn_stage.pt")
        write_json(graph_path, {
            "stage": args.stage, "new_classes": new_ids, "old_classes": old_ids,
            "selected_replay_classes": selected, "graph": graph_info,
            "replay": replay_info, "feature_source": "checkpoint classifier prototypes",
        })

    arm_dir = args.output_dir / args.control
    command = build_main_command(args, train_ann, full_val, arm_dir, active_ids, old_ids,
                                 args.stage == 0)
    metadata = {"runner": "tools/owod/run_graph_local_increment.py", "stage": args.stage,
                "new_classes": new_ids, "old_classes": old_ids,
                "active_classes": active_ids, "selected_replay_classes": selected,
                "graph_estimator": args.graph_estimator, "control": args.control,
                "checkpoint": str(args.checkpoint),
                "resume": str(args.resume) if args.resume else "",
                "command": command}
    write_json(args.output_dir / "run_config.json", metadata)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "models" / "ops") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    code = stream_process(command, arm_dir, env, args.dry_run)
    write_json(args.output_dir / "summary.json", {**metadata, "return_code": code,
                "train_log": str((arm_dir / "train.log").resolve()),
                "console_log": str((arm_dir / "console.log").resolve())})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
