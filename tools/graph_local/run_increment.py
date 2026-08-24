#!/usr/bin/env python
"""Run a graph-local class-increment experiment on a COCO-format split.

The script deliberately keeps the research lifecycle explicit:

1. Check the base detector on known classes.
2. On one expanded, frozen teacher state, measure dense target-weight gradient
   sketches and predict a positive-conflict neighborhood for one new class.
3. Apply a one-step rank-8 probe and compare the predicted neighborhood with
   empirical old-class loss harm.
4. Only after the quality gate passes, compare graph-local, random-neighborhood,
   and global-replay low-rank updates.

It is single-GPU by design. The Oxford Pet run is an engineering preflight;
passing it does not establish an OWOD result on VOC/COCO IOD.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_coco_api_from_dataset
from datasets.coco import CocoDetection, make_coco_transforms
from engine import evaluate
from main import get_args_parser
from models import build_model
from models.graph_local.interference import (
    build_conflict_matrix,
    build_off_neighborhood_basis,
    evaluate_neighborhood,
    flatten_gradients,
    select_positive_neighbors,
)
from models.graph_local.lora import (
    expand_classification_head,
    freeze_for_increment,
    inject_decoder_lora,
    lora_delta_vector,
    merge_decoder_lora,
    target_base_weights,
)
from models.graph_local.losses import (
    local_margin_loss,
    projection_loss,
    weighted_detection_loss,
)
from models.graph_local.pseudo_labels import complete_targets_with_teacher
from models.graph_local.replay import build_increment_annotation
import util.misc as utils
from util.experiment_log import start_file_logging, stop_file_logging


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Graph-local continual Deformable DETR experiment",
        parents=[get_args_parser()],
    )
    parser.add_argument("--baseline", required=True,
                        help="known-class checkpoint produced by main.py")
    parser.add_argument("--metadata", required=True,
                        help="split_metadata.json from prepare_oxford_pet.py")
    parser.add_argument("--new-ann", default="",
                        help="new-class training annotation; inferred from metadata if omitted")
    parser.add_argument("--increment-val-ann", default="",
                        help="known + one new class validation annotation; inferred if omitted")
    parser.add_argument("--known-val-ann", default="instances_val2017.json",
                        help="known-only validation annotation relative to coco_path/annotations")
    parser.add_argument("--new-class", type=int, default=None,
                        help="remapped new class id; defaults to metadata increment id")
    parser.add_argument("--increment-epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--last-decoder-layers", type=int, default=2)
    parser.add_argument("--graph-k", type=int, default=5)
    parser.add_argument("--min-conflict", type=float, default=0.0)
    parser.add_argument("--replay-budget", type=int, default=64)
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--increment-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay-increment", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=1.0)
    parser.add_argument("--lambda-off", type=float, default=1.0)
    parser.add_argument("--off-basis-rank", type=int, default=8)
    parser.add_argument("--pseudo-score", type=float, default=0.5)
    parser.add_argument("--pseudo-iou", type=float, default=0.7)
    parser.add_argument("--pseudo-gt-iou", type=float, default=0.5)
    parser.add_argument("--pseudo-max-per-image", type=int, default=20)
    parser.add_argument("--sketch-max-images", type=int, default=20)
    parser.add_argument("--probe-max-images", type=int, default=20)
    parser.add_argument("--min-matched-per-class", type=int, default=20)
    parser.add_argument("--quality-gate-ap50", type=float, default=0.40)
    parser.add_argument("--harm-random-trials", type=int, default=1000)
    parser.add_argument("--run-despite-quality-gate", action="store_true",
                        help="run adaptation arms as plumbing only after a failed base quality gate")
    parser.add_argument("--module-ablations", action="store_true",
                        help="add graph-local ablations for pseudo labels, margin, projection, and replay")
    parser.add_argument("--arms", nargs="+", choices=("graph", "random", "global"),
                        default=("graph", "random", "global"))
    parser.set_defaults(dataset_file="coco", masks=False, cache_mode=False, two_stage=False)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def resolve_annotation(coco_path: Path, candidate: str) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = coco_path / "annotations" / path
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    return path


def build_dataset(coco_path: Path, annotation: Path, image_set: str, args) -> CocoDetection:
    image_dir = coco_path / ("train2017" if image_set == "train" else "val2017")
    return CocoDetection(
        image_dir,
        annotation,
        transforms=make_coco_transforms(image_set, args.lightweight),
        return_masks=False,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def move_targets(targets: Sequence[Dict], device: torch.device) -> List[Dict]:
    return [{key: value.to(device) if torch.is_tensor(value) else value
             for key, value in target.items()} for target in targets]


def clone_target(target: Dict) -> Dict:
    return {key: value.clone() if torch.is_tensor(value) else value
            for key, value in target.items()}


def target_for_class(target: Dict, class_id: int) -> Dict:
    filtered = clone_target(target)
    keep = target["labels"] == int(class_id)
    length = int(target["labels"].numel())
    for key, value in target.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == length:
            filtered[key] = value[keep].clone()
    return filtered


def dataset_indices_for_class(dataset: CocoDetection, class_id: int) -> List[int]:
    indices = []
    for index, image_id in enumerate(dataset.ids):
        if any(int(annotation["category_id"]) == int(class_id)
               for annotation in dataset.coco.imgToAnns.get(image_id, [])):
            indices.append(index)
    return indices


def class_loader(dataset: CocoDetection, class_id: int, batch_size: int,
                 num_workers: int, max_images: int) -> Tuple[DataLoader, int]:
    indices = dataset_indices_for_class(dataset, class_id)
    if max_images is not None:
        indices = indices[:max_images]
    return make_loader(Subset(dataset, indices), batch_size, False, num_workers), len(indices)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.endswith(("total_params", "total_ops"))]
    if missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint did not match the requested detector configuration. "
            f"Missing={missing}; unexpected={unexpected}"
        )


def build_expanded_model(args, base_num_classes: int, total_num_classes: int,
                         checkpoint: Path, device: torch.device):
    args.num_classes = int(base_num_classes)
    model, criterion, postprocessors = build_model(args)
    load_checkpoint(model, checkpoint)
    model.to(device)
    old_classes, _ = expand_classification_head(
        model, total_num_classes, initialization="mean_old")
    if old_classes != base_num_classes:
        raise ValueError(f"Checkpoint exposes {old_classes} classes, expected {base_num_classes}")
    criterion.num_classes = int(total_num_classes)
    criterion.to(device)
    return model, criterion, postprocessors


def build_teacher(args, base_num_classes: int, checkpoint: Path, device: torch.device):
    args.num_classes = int(base_num_classes)
    teacher, _criterion, _postprocessors = build_model(args)
    load_checkpoint(teacher, checkpoint)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def class_loss(model, criterion, loader: DataLoader, class_id: int,
               device: torch.device) -> Tuple[float, int]:
    """Mean full detector loss for annotations of exactly one class."""
    was_training = model.training
    model.eval()
    values = []
    matched = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = move_targets(targets, device)
            filtered = [target_for_class(target, class_id) for target in targets]
            count = sum(int(target["labels"].numel()) for target in filtered)
            if count == 0:
                continue
            losses = criterion(model(samples), filtered)
            values.append(float(weighted_detection_loss(losses, criterion.weight_dict).item()))
            matched += count
    model.train(was_training)
    return (float(np.mean(values)) if values else math.nan), matched


def compute_gradient_sketches(model, criterion, dataset: CocoDetection,
                              class_ids: Iterable[int], args,
                              device: torch.device) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Average dense target-weight gradients on one frozen expanded teacher state."""
    model.eval()
    weights = target_base_weights(model, args.last_decoder_layers)
    sketches: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for class_id in class_ids:
        loader, image_count = class_loader(
            dataset, class_id, args.batch_size, args.num_workers, args.sketch_max_images)
        total = None
        matched = 0
        for samples, targets in loader:
            samples = samples.to(device)
            targets = move_targets(targets, device)
            filtered = [target_for_class(target, class_id) for target in targets]
            batch_count = sum(int(target["labels"].numel()) for target in filtered)
            if batch_count == 0:
                continue
            outputs = model(samples)
            losses = criterion(outputs, filtered)
            scalar = weighted_detection_loss(losses, criterion.weight_dict)
            gradients = torch.autograd.grad(scalar, weights, allow_unused=True)
            vector = flatten_gradients(gradients, weights).cpu()
            total = vector if total is None else total + vector
            matched += batch_count
        if total is None:
            raise RuntimeError(f"No matched calibration examples for class {class_id} ({image_count} images)")
        sketches[int(class_id)] = total / max(1, matched)
        counts[int(class_id)] = matched
    return sketches, counts


def one_step_probe(model, criterion, new_loader: DataLoader, base_num_classes: int,
                   args, device: torch.device) -> None:
    """Apply a rank-r new-class-only update used solely to define empirical harm."""
    inject_decoder_lora(model, rank=args.rank, last_n=args.last_decoder_layers)
    handles, parameters = freeze_for_increment(model, base_num_classes)
    optimizer = torch.optim.AdamW(parameters, lr=args.probe_lr,
                                  weight_decay=args.weight_decay_increment)
    model.train()
    for samples, targets in new_loader:
        samples = samples.to(device)
        targets = move_targets(targets, device)
        losses = criterion(model(samples), targets)
        objective = weighted_detection_loss(losses, criterion.weight_dict)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.clip_max_norm)
        optimizer.step()
        break
    for handle in handles:
        handle.remove()


def measure_harm(base_model, probe_model, criterion, dataset: CocoDetection,
                 old_class_ids: Sequence[int], args, device: torch.device) -> Tuple[Dict[int, float], Dict[int, int]]:
    harm: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for class_id in old_class_ids:
        loader, _ = class_loader(dataset, class_id, args.batch_size, args.num_workers,
                                 args.probe_max_images)
        before, count = class_loss(base_model, criterion, loader, class_id, device)
        after, _ = class_loss(probe_model, criterion, loader, class_id, device)
        harm[int(class_id)] = max(0.0, after - before) if np.isfinite(before + after) else 0.0
        counts[int(class_id)] = count
    return harm, counts


def per_class_ap(coco_evaluator) -> Dict[int, float]:
    precision = coco_evaluator.coco_eval["bbox"].eval["precision"]
    category_ids = coco_evaluator.coco_eval["bbox"].params.catIds
    values: Dict[int, float] = {}
    for index, category_id in enumerate(category_ids):
        entries = precision[:, :, index, 0, -1]
        entries = entries[entries > -1]
        values[int(category_id)] = float(entries.mean()) if entries.size else math.nan
    return values


def evaluate_detector(model, criterion, postprocessors, dataset, args,
                      device: torch.device, output_dir: Path) -> Tuple[Dict, Dict[int, float]]:
    loader = make_loader(dataset, args.batch_size, False, args.num_workers)
    stats, evaluator = evaluate(
        model, criterion, postprocessors, loader,
        get_coco_api_from_dataset(dataset), device, str(output_dir))
    return stats, per_class_ap(evaluator)


def make_random_neighborhood(old_class_ids: Sequence[int], size: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(list(old_class_ids), min(size, len(old_class_ids))))


def run_increment_arm(name: str, args, base_num_classes: int, total_num_classes: int,
                      checkpoint: Path, teacher, train_annotation: Path, eval_dataset,
                      sketches: Mapping[int, torch.Tensor], neighborhood: Sequence[int],
                      components: Mapping[str, bool],
                      baseline_ap: Mapping[int, float], device: torch.device,
                      output_dir: Path) -> Dict:
    model, criterion, postprocessors = build_expanded_model(
        args, base_num_classes, total_num_classes, checkpoint, device)
    inject_decoder_lora(model, rank=args.rank, last_n=args.last_decoder_layers)
    handles, trainable = freeze_for_increment(model, base_num_classes)
    optimizer = torch.optim.AdamW(trainable, lr=args.increment_lr,
                                  weight_decay=args.weight_decay_increment)
    replay_dataset = build_dataset(Path(args.coco_path), train_annotation, "train", args)
    loader = make_loader(replay_dataset, args.batch_size, True, args.num_workers)

    new_class = int(args.new_class)
    local_classes = [new_class] + list(neighborhood)
    if components["off_projection"]:
        off_basis = build_off_neighborhood_basis(
            sketches, excluded_classes=local_classes, max_rank=args.off_basis_rank)
    else:
        off_basis = None

    history = []
    model.train()
    for epoch in range(args.increment_epochs):
        metric = defaultdict(float)
        batches = 0
        pseudo_total = 0
        for samples, targets in loader:
            samples = samples.to(device)
            targets = move_targets(targets, device)
            if components["pseudo_labels"]:
                completed, pseudo_counts = complete_targets_with_teacher(
                    teacher, samples, targets, old_num_classes=base_num_classes,
                    score_threshold=args.pseudo_score, duplicate_iou=args.pseudo_iou,
                    ground_truth_iou=args.pseudo_gt_iou,
                    max_per_image=args.pseudo_max_per_image)
            else:
                completed = targets
                pseudo_counts = [0] * len(targets)
            outputs = model(samples)
            losses = criterion(outputs, completed)
            detector = weighted_detection_loss(losses, criterion.weight_dict)
            local = local_margin_loss(
                outputs, completed, criterion.matcher, local_classes, margin=args.margin) if components["local_margin"] else detector * 0.0
            off = projection_loss(lora_delta_vector(model), off_basis) if off_basis is not None else detector * 0.0
            objective = detector + args.lambda_local * local + args.lambda_off * off
            if not torch.isfinite(objective):
                raise FloatingPointError(f"{name}: non-finite incremental objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_max_norm)
            optimizer.step()
            metric["detector"] += float(detector.detach().item())
            metric["local"] += float(local.detach().item())
            metric["off"] += float(off.detach().item())
            metric["objective"] += float(objective.detach().item())
            metric["grad_norm"] += float(grad_norm)
            pseudo_total += sum(pseudo_counts)
            batches += 1
        if batches == 0:
            raise RuntimeError(f"{name}: incremental training annotation contains no batches")
        history.append({
            "epoch": epoch,
            **{key: value / batches for key, value in metric.items()},
            "pseudo_labels_per_image": pseudo_total / max(1, len(replay_dataset)),
        })

    verification_loader = make_loader(eval_dataset, 1, False, 0)
    sample_batch, _ = next(iter(verification_loader))
    model.eval()
    with torch.no_grad():
        before_merge = model(sample_batch.to(device))
    merged_layers = merge_decoder_lora(model, args.last_decoder_layers)
    with torch.no_grad():
        after_merge = model(sample_batch.to(device))
    merge_error = max(
        float((before_merge["pred_logits"] - after_merge["pred_logits"]).abs().max().item()),
        float((before_merge["pred_boxes"] - after_merge["pred_boxes"]).abs().max().item()),
    )
    for handle in handles:
        handle.remove()

    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    stats, ap = evaluate_detector(model, criterion, postprocessors, eval_dataset, args, device, arm_dir)
    torch.save({"model": model.state_dict(), "args": vars(args), "arm": name}, arm_dir / "checkpoint_merged.pth")

    old_ids = list(range(base_num_classes))
    neighbor_set = set(neighborhood)
    def mean_delta(class_ids: Iterable[int]) -> float:
        values = [ap.get(class_id, math.nan) - baseline_ap.get(class_id, math.nan)
                  for class_id in class_ids]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else math.nan

    return {
        "arm": name,
        "neighborhood": [int(value) for value in neighborhood],
        "components": {key: bool(value) for key, value in components.items()},
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "history": history,
        "merge_layers": merged_layers,
        "merge_max_abs_error": merge_error,
        "coco_stats": stats.get("coco_eval_bbox", []),
        "per_class_ap": {str(key): value for key, value in ap.items()},
        "new_class_ap": ap.get(new_class, math.nan),
        "neighbor_old_ap_delta": mean_delta(neighbor_set),
        "off_neighborhood_old_ap_delta": mean_delta(
            class_id for class_id in old_ids if class_id not in neighbor_set),
    }


def main() -> int:
    parser = get_parser()
    args = parser.parse_args()
    if not args.output_dir:
        parser.error("--output_dir is required for a reproducible experiment")
    set_seed(args.seed)
    device = torch.device(args.device)
    coco_path = Path(args.coco_path).resolve()
    checkpoint = Path(args.baseline).resolve()
    metadata = read_json(Path(args.metadata))
    base_num_classes = int(metadata["num_known"])
    args.new_class = int(metadata["increment_remapped_id"] if args.new_class is None else args.new_class)
    if args.new_class < base_num_classes:
        parser.error("--new-class must be outside the base known-class range")
    total_num_classes = args.new_class + 1
    new_ann = resolve_annotation(
        coco_path, args.new_ann or f"instances_increment_new_{args.new_class}.json")
    increment_val_ann = resolve_annotation(
        coco_path, args.increment_val_ann or f"instances_increment_val_{args.new_class}.json")
    known_val_ann = resolve_annotation(coco_path, args.known_val_ann)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_log_state = start_file_logging(args, is_main_process=True)

    known_val = build_dataset(coco_path, known_val_ann, "val", args)
    increment_val = build_dataset(coco_path, increment_val_ann, "val", args)
    base_model, base_criterion, base_postprocessors = build_expanded_model(
        args, base_num_classes, total_num_classes, checkpoint, device)
    quality_stats, baseline_ap = evaluate_detector(
        base_model, base_criterion, base_postprocessors, known_val, args, device, output_dir / "quality_gate")
    ap50 = float(quality_stats["coco_eval_bbox"][1])
    quality_pass = ap50 >= args.quality_gate_ap50
    quality = {
        "known_ap": float(quality_stats["coco_eval_bbox"][0]),
        "known_ap50": ap50,
        "required_ap50": args.quality_gate_ap50,
        "passed": quality_pass,
    }
    summary = {
        "schema_version": 1,
        "experiment": "graph_local_increment",
        "seed": args.seed,
        "base_num_classes": base_num_classes,
        "new_class": args.new_class,
        "new_class_name": metadata.get("increment_name"),
        "quality_gate": quality,
        "scientific_verdict": "pending",
        "artifacts": {
            "baseline": str(checkpoint),
            "new_annotation": str(new_ann),
            "increment_validation_annotation": str(increment_val_ann),
        },
    }
    if not quality_pass and not args.run_despite_quality_gate:
        summary["scientific_verdict"] = (
            "inconclusive: base detector did not satisfy the pre-registered AP50 quality gate; "
            "graph locality and continual-learning claims were not tested")
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        stop_file_logging(file_log_state)
        return 0

    if args.module_ablations and "graph" not in args.arms:
        parser.error("--module-ablations requires the graph arm")

    all_ids = list(range(total_num_classes))
    sketches, sketch_counts = compute_gradient_sketches(
        base_model, base_criterion, increment_val, all_ids, args, device)
    if any(count < args.min_matched_per_class for count in sketch_counts.values()):
        summary["scientific_verdict"] = (
            "inconclusive: at least one class has fewer than the required calibration matches")
        summary["sketch_matches"] = {str(key): value for key, value in sketch_counts.items()}
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        stop_file_logging(file_log_state)
        return 0

    class_ids, conflict = build_conflict_matrix(sketches)
    scored_neighbors = select_positive_neighbors(
        class_ids, conflict, args.new_class, k=args.graph_k, min_conflict=args.min_conflict)
    graph_neighbors = [class_id for class_id, _score in scored_neighbors]
    old_class_ids = list(range(base_num_classes))

    probe_model = copy.deepcopy(base_model)
    probe_loader = make_loader(
        build_dataset(coco_path, new_ann, "train", args), args.batch_size, False, args.num_workers)
    one_step_probe(probe_model, base_criterion, probe_loader, base_num_classes, args, device)
    harm, harm_counts = measure_harm(
        base_model, probe_model, base_criterion, increment_val, old_class_ids, args, device)
    neighborhood_eval = evaluate_neighborhood(
        graph_neighbors, harm, k=len(graph_neighbors),
        random_trials=args.harm_random_trials, seed=args.seed)
    del probe_model
    torch.cuda.empty_cache()

    annotations_dir = output_dir / "annotations"
    graph_ann_info = build_increment_annotation(
        new_ann, coco_path / "annotations" / "instances_train2017.json",
        graph_neighbors, args.replay_budget, annotations_dir / "graph.json", seed=args.seed)
    random_neighbors = make_random_neighborhood(old_class_ids, len(graph_neighbors), args.seed + 1)
    random_ann_info = build_increment_annotation(
        new_ann, coco_path / "annotations" / "instances_train2017.json",
        random_neighbors, args.replay_budget, annotations_dir / "random.json", seed=args.seed + 1)
    global_ann_info = build_increment_annotation(
        new_ann, coco_path / "annotations" / "instances_train2017.json",
        old_class_ids, args.replay_budget, annotations_dir / "global.json", seed=args.seed)
    no_replay_ann_info = None
    if args.module_ablations:
        no_replay_ann_info = build_increment_annotation(
            new_ann, coco_path / "annotations" / "instances_train2017.json",
            graph_neighbors, 0, annotations_dir / "graph_no_replay.json", seed=args.seed)

    teacher = build_teacher(args, base_num_classes, checkpoint, device)
    graph_components = {
        "pseudo_labels": True,
        "local_margin": True,
        "off_projection": True,
        "replay": True,
    }
    arms = {}
    arm_inputs = {
        "graph": (Path(graph_ann_info["output"]), graph_neighbors, graph_ann_info, graph_components),
        "random": (Path(random_ann_info["output"]), random_neighbors, random_ann_info, graph_components),
        "global": (Path(global_ann_info["output"]), old_class_ids, global_ann_info, {
            "pseudo_labels": True,
            "local_margin": False,
            "off_projection": False,
            "replay": True,
        }),
        "graph_no_pseudo": (Path(graph_ann_info["output"]), graph_neighbors, graph_ann_info, {
            **graph_components, "pseudo_labels": False,
        }),
        "graph_no_margin": (Path(graph_ann_info["output"]), graph_neighbors, graph_ann_info, {
            **graph_components, "local_margin": False,
        }),
        "graph_no_projection": (Path(graph_ann_info["output"]), graph_neighbors, graph_ann_info, {
            **graph_components, "off_projection": False,
        }),
    }
    if no_replay_ann_info is not None:
        arm_inputs["graph_no_replay"] = (
            Path(no_replay_ann_info["output"]), graph_neighbors, no_replay_ann_info, {
                **graph_components, "replay": False,
            })
    arm_names = list(args.arms)
    if args.module_ablations:
        arm_names.extend(("graph_no_pseudo", "graph_no_margin", "graph_no_projection", "graph_no_replay"))
    for name in arm_names:
        annotation, neighbors, info, components = arm_inputs[name]
        result = run_increment_arm(
            name, args, base_num_classes, total_num_classes, checkpoint, teacher,
            annotation, increment_val, sketches, neighbors, components, baseline_ap, device, output_dir)
        result["annotation"] = info
        arms[name] = result
        write_json(output_dir / name / "result.json", result)
        torch.cuda.empty_cache()

    summary.update({
        "sketch_matches": {str(key): value for key, value in sketch_counts.items()},
        "probe_matches": {str(key): value for key, value in harm_counts.items()},
        "graph_neighbors": [{"class_id": class_id, "conflict": score}
                            for class_id, score in scored_neighbors],
        "neighborhood_diagnostic": neighborhood_eval,
        "arms": arms,
    })
    if not quality_pass:
        summary["scientific_verdict"] = (
            "engineering-only: adaptation arms ran despite a failed base quality gate; "
            "their metrics cannot establish the locality hypothesis")
    elif not graph_neighbors:
        summary["scientific_verdict"] = (
            "negative result: no positive gradient-conflict neighbors were found for the selected new class")
    elif neighborhood_eval["oracle_concentration"] < 0.60:
        summary["scientific_verdict"] = (
            "negative result: oracle top-k harm is not sufficiently concentrated; locality is not supported")
    elif neighborhood_eval["predicted_concentration"] < 0.60 or (
            neighborhood_eval["predicted_concentration"] - neighborhood_eval["random_concentration_mean"] < 0.20):
        summary["scientific_verdict"] = (
            "negative result: the gradient-conflict graph does not predict local harm better than random")
    else:
        summary["scientific_verdict"] = (
            "preliminary support only: the one-step locality diagnostic passed. "
            "Compare graph versus random/global arms across three seeds before any research claim")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    stop_file_logging(file_log_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
