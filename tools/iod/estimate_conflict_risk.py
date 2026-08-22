#!/usr/bin/env python3
"""Estimate class-conditioned forgetting risk without updating the detector.

Loads a frozen base Deformable DETR, computes a gradient sketch per old and
new class, and writes their positive-conflict matrix and old-class risk.  This
diagnostic uses no validation labels and does not update the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

# Running this file directly puts tools/iod on sys.path, not the repository
# root. Add the root explicitly so util, datasets, models, and main resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.misc as utils
from datasets.coco import CocoDetection, make_coco_transforms
from main import get_args_parser
from models import build_model


def target_base_weights(model: torch.nn.Module, last_n: int) -> List[torch.nn.Parameter]:
    """Select stable shared decoder FFN weights for the gradient sketch."""
    detector = model.detr if hasattr(model, "detr") else model
    layers = detector.transformer.decoder.layers
    if last_n <= 0 or last_n > len(layers):
        raise ValueError(f"last-decoder-layers must be in [1, {len(layers)}]")
    weights = []
    for layer in layers[-last_n:]:
        for name in ("linear1", "linear2"):
            module = getattr(layer, name)
            weights.append(module.weight)
    return weights


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [x for x in unexpected if not x.endswith(("total_params", "total_ops"))]
    if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def checkpoint_num_classes(path: Path) -> int:
    """Read the classifier width before constructing the matching model."""
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    for key, value in state.items():
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in {"class_embed.0.weight", "class_embed.weight"} and torch.is_tensor(value):
            return int(value.shape[0])
    raise RuntimeError("could not infer classifier width from checkpoint")


def move_targets(targets: Sequence[Mapping], device: torch.device) -> List[Dict]:
    return [{key: value.to(device) if torch.is_tensor(value) else value
             for key, value in target.items()} for target in targets]


def target_for_class(target: Mapping, class_id: int) -> Dict:
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in target.items()}
    labels = target["labels"]
    keep = labels == int(class_id)
    count = int(labels.numel())
    for key, value in target.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            result[key] = value[keep].clone()
    return result


def weighted_loss(losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    terms = [value * weights[key] for key, value in losses.items() if key in weights]
    if not terms:
        raise RuntimeError("criterion returned no weighted losses")
    return sum(terms)


def class_indices(dataset: CocoDetection, class_id: int) -> List[int]:
    return [index for index, image_id in enumerate(dataset.ids)
            if any(int(ann["category_id"]) == int(class_id)
                   for ann in dataset.coco.imgToAnns.get(image_id, []))]


def compute_sketch(model: torch.nn.Module, criterion: torch.nn.Module,
                   dataset: CocoDetection, class_id: int, args,
                   weights: Sequence[torch.nn.Parameter], device: torch.device) -> Dict:
    indices = class_indices(dataset, class_id)
    if args.max_images > 0:
        indices = indices[:args.max_images]
    if not indices:
        raise RuntimeError(f"no images for class {class_id} in the supplied annotation")
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                        pin_memory=True)
    total = None
    matched = 0
    model.eval()
    for samples, targets in loader:
        samples = samples.to(device)
        targets = move_targets(targets, device)
        filtered = [target_for_class(target, class_id) for target in targets]
        boxes = sum(int(target["labels"].numel()) for target in filtered)
        if boxes == 0:
            continue
        outputs = model(samples)
        losses = criterion(outputs, filtered)
        scalar = weighted_loss(losses, criterion.weight_dict)
        gradients = torch.autograd.grad(scalar, weights, allow_unused=True)
        parts = [gradient.detach().reshape(-1) if gradient is not None
                 else parameter.detach().new_zeros(parameter.numel())
                 for gradient, parameter in zip(gradients, weights)]
        vector = torch.cat(parts).float().cpu()
        total = vector if total is None else total + vector
        matched += boxes
    if total is None:
        raise RuntimeError(f"no matched boxes for class {class_id}")
    sketch = total / float(max(1, matched))
    return {"sketch": sketch, "images": len(indices), "boxes": matched,
            "energy": float(torch.dot(sketch, sketch).item())}


def main() -> None:
    parser = argparse.ArgumentParser("Frozen RCGC conflict-risk estimator", parents=[get_args_parser()])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--old-ann", type=Path, required=True)
    parser.add_argument("--new-ann", type=Path, required=True)
    parser.add_argument("--old-classes", type=int, nargs="+", required=True)
    parser.add_argument("--new-classes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--last-decoder-layers", type=int, default=2)
    args = parser.parse_args()
    args.dataset_file = "coco"
    args.num_classes = max(max(args.old_classes + args.new_classes) + 1,
                           checkpoint_num_classes(args.checkpoint))
    args.masks = False
    device = torch.device(args.device)
    model, criterion, _postprocessors = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    criterion.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weights = target_base_weights(model, args.last_decoder_layers)
    for parameter in weights:
        parameter.requires_grad_(True)
    transform = make_coco_transforms("train", args.lightweight, True, True)
    old_dataset = CocoDetection(args.coco_root / "train2017", args.old_ann, transform,
                                 False, False, 0, 1)
    new_dataset = CocoDetection(args.coco_root / "train2017", args.new_ann, transform,
                                 False, False, 0, 1)
    sketches = {}
    for class_id in args.old_classes:
        sketches[f"old:{class_id}"] = compute_sketch(model, criterion, old_dataset, class_id,
                                                       args, weights, device)
    for class_id in args.new_classes:
        sketches[f"new:{class_id}"] = compute_sketch(model, criterion, new_dataset, class_id,
                                                       args, weights, device)
    old_vectors = torch.stack([sketches[f"old:{x}"]["sketch"] for x in args.old_classes])
    new_vectors = torch.stack([sketches[f"new:{x}"]["sketch"] for x in args.new_classes])
    old_energy = old_vectors.square().sum(dim=1)
    new_energy = new_vectors.square().sum(dim=1)
    dot = new_vectors @ old_vectors.T
    denom = torch.sqrt(new_energy[:, None] * old_energy[None, :]).clamp_min(1e-12)
    conflict = torch.relu(dot / denom)
    risk = conflict.sum(dim=0)
    payload = {"schema_version": 1, "method": "positive_gradient_conflict",
               "checkpoint": str(args.checkpoint), "old_classes": args.old_classes,
               "new_classes": args.new_classes, "max_images": args.max_images,
               "target_parameters": sum(parameter.numel() for parameter in weights),
               "conflict": conflict.tolist(), "risk": risk.tolist(),
               "sketch_stats": {key: {k: value for k, value in val.items() if k != "sketch"}
                                for key, val in sketches.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "risk": dict(zip(args.old_classes, risk.tolist()))},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
