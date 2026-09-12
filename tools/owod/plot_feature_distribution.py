#!/usr/bin/env python
"""Extract D2 decoder-query features and draw publication-ready t-SNE plots."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import random
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



GROUP_ORDER = ("Previous", "Current", "Unknown", "Background")
GROUP_COLORS = {
    "Previous": "#2878B5",
    "Current": "#E6862D",
    "Unknown": "#C33C54",
    "Background": "#777777",
}
GROUP_MARKERS = {
    "Previous": "o",
    "Current": "s",
    "Unknown": "X",
    "Background": "x",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="D2 graph directory containing run_config.json and checkpoint.pth")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 processes the complete validation set")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--background-iou", type=float, default=0.1)
    parser.add_argument("--background-per-image", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--class-confidence", type=float, default=0.3,
                        help="Class plot only: require correct known-class prediction and this confidence")
    parser.add_argument("--max-per-group", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 <= args.class_confidence <= 1:
        parser.error("--class-confidence must lie in [0, 1]")
    if not 0 <= args.background_iou < args.iou_threshold <= 1:
        parser.error("Require 0 <= background-iou < iou-threshold <= 1")
    if min(args.batch_size, args.max_per_class, args.max_per_group) <= 0:
        parser.error("Batch size and sampling limits must be positive")
    if min(args.max_images, args.num_workers, args.background_per_image) < 0:
        parser.error("Image/worker/background counts cannot be negative")
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive")
    return args


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detector_args_from_config(config: Mapping, cli_args):
    from main import get_args_parser
    detector_args = get_args_parser().parse_args([])
    for name, value in config.items():
        if hasattr(detector_args, name):
            setattr(detector_args, name, value)
    detector_args.device = cli_args.device
    detector_args.batch_size = cli_args.batch_size
    detector_args.num_workers = cli_args.num_workers
    detector_args.dataset_file = "coco"
    detector_args.masks = False
    detector_args.cache_mode = False
    detector_args.paper_baseline = None
    detector_args.with_tree = False
    if not detector_args.coco_path:
        raise ValueError("run_config.json does not contain coco_path")
    if not detector_args.val_ann:
        raise ValueError("run_config.json does not contain the full-validation val_ann")
    return detector_args


def load_d2_model(detector_args, checkpoint_path: Path, device: torch.device):
    from main import load_local_checkpoint
    from models import build_model
    model, _criterion, _postprocessors = build_model(detector_args)
    payload = load_local_checkpoint(checkpoint_path)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected
                  if not name.endswith(("total_params", "total_ops"))]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint and run_config.json do not describe the same model. "
            f"Missing={missing}; unexpected={unexpected}")
    model.return_baseline_features = True
    model.to(device).eval()
    return model


def category_group(class_id: int, previous_ids: set[int], current_ids: set[int]) -> str:
    if class_id in previous_ids:
        return "Previous"
    if class_id in current_ids:
        return "Current"
    return "Unknown"


def match_image(features, pred_boxes, pred_logits, target, previous_ids, current_ids,
                iou_threshold, background_iou, background_per_image, rng):
    from scipy.optimize import linear_sum_assignment
    from util import box_ops

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()
    pred_boxes = pred_boxes.detach().cpu()
    image_id = int(target["image_id"].item())
    # Match the actual unmasked D2 classifier, including sparse COCO slots.
    # Restricting argmax to GT/known IDs would hide some classification errors.
    scores, predictions = pred_logits.detach().cpu().sigmoid().max(dim=-1)
    records = []

    if len(gt_boxes):
        ious = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(gt_boxes),
            box_ops.box_cxcywh_to_xyxy(pred_boxes))[0]
        gt_indices, query_indices = linear_sum_assignment(1.0 - ious.numpy())
        matched_queries = set()
        for gt_index, query_index in zip(gt_indices.tolist(), query_indices.tolist()):
            overlap = float(ious[gt_index, query_index])
            if overlap < iou_threshold:
                continue
            class_id = int(gt_labels[gt_index])
            matched_queries.add(query_index)
            records.append({
                "feature": features[query_index].numpy().copy(),
                "group": category_group(class_id, previous_ids, current_ids),
                "class_id": class_id,
                "image_id": image_id,
                "query_id": query_index,
                "iou": overlap,
                "predicted_class_id": int(predictions[query_index]),
                "confidence": float(scores[query_index]),
            })
        max_iou = ious.max(dim=0).values.numpy()
        candidates = [index for index, value in enumerate(max_iou)
                      if index not in matched_queries and value < background_iou]
    else:
        candidates = list(range(len(pred_boxes)))
        max_iou = np.zeros(len(pred_boxes))

    if background_per_image > 0 and candidates:
        chosen = rng.sample(candidates, min(background_per_image, len(candidates)))
        for query_index in chosen:
            records.append({
                "feature": features[query_index].numpy().copy(),
                "group": "Background",
                "class_id": -1,
                "image_id": image_id,
                "query_id": query_index,
                "iou": float(max_iou[query_index]),
                "predicted_class_id": int(predictions[query_index]),
                "confidence": float(scores[query_index]),
            })
    return records


def extract_features(model, dataset, device, args, previous_ids, current_ids):
    from torch.utils.data import DataLoader, SequentialSampler
    from util.misc import collate_fn
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
        drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=device.type == "cuda")
    rng = random.Random(args.seed)
    records = []
    processed = 0
    for samples, targets in loader:
        if args.max_images and processed >= args.max_images:
            break
        samples = samples.to(device)
        outputs = model(samples)
        layer_features = outputs["decoder_features"][-1].detach().cpu()
        pred_boxes = outputs["pred_boxes"].detach().cpu()
        pred_logits = outputs["pred_logits"].detach().cpu()
        for batch_index, target in enumerate(targets):
            if args.max_images and processed >= args.max_images:
                break
            records.extend(match_image(
                layer_features[batch_index], pred_boxes[batch_index], pred_logits[batch_index], target,
                previous_ids, current_ids, args.iou_threshold,
                args.background_iou, args.background_per_image, rng))
            processed += 1
        if processed % 100 == 0 or processed == len(dataset):
            print(f"Extracted {processed}/{min(len(dataset), args.max_images or len(dataset))} images",
                  flush=True)
    return records


def balanced_indices(records, max_per_class: int, max_background: int, seed: int):
    rng = random.Random(seed)
    buckets = {}
    for index, record in enumerate(records):
        key = (record["group"], record["class_id"])
        buckets.setdefault(key, []).append(index)
    selected = []
    for (group, _class_id), indices in sorted(buckets.items()):
        limit = max_background if group == "Background" else max_per_class
        selected.extend(rng.sample(indices, min(limit, len(indices))))
    return sorted(selected)


def group_balanced_indices(groups, max_per_group: int, seed: int):
    rng = random.Random(seed)
    selected = []
    for group in GROUP_ORDER:
        indices = np.flatnonzero(groups == group).tolist()
        selected.extend(rng.sample(indices, min(max_per_group, len(indices))))
    return sorted(selected)


def tsne_embedding(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import normalize
    except ImportError as error:
        raise SystemExit("Install plotting dependencies: pip install scikit-learn matplotlib") from error
    if len(features) < 3:
        raise ValueError("At least three matched features are required for t-SNE")
    normalized = normalize(features)
    components = min(50, normalized.shape[0] - 1, normalized.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(normalized)
    effective_perplexity = min(perplexity, max(2.0, (len(reduced) - 1) / 3.0))
    kwargs = dict(n_components=2, perplexity=effective_perplexity, init="pca",
                  learning_rate="auto", random_state=seed)
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 2000
    else:
        kwargs["n_iter"] = 2000
    return TSNE(**kwargs).fit_transform(reduced)


def configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install plotting dependencies: pip install scikit-learn matplotlib") from error
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_group_plot(embedding, groups, output_dir, max_per_group, seed):
    plt = configure_matplotlib()
    indices = group_balanced_indices(groups, max_per_group=max_per_group, seed=seed)
    figure, axis = plt.subplots(figsize=(6.6, 5.2))
    for group in GROUP_ORDER:
        mask = np.array([index for index in indices if groups[index] == group])
        if not len(mask):
            continue
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=14,
                     c=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                     alpha=0.72, linewidths=0.5 if group == "Background" else 0,
                     label=group, rasterized=True)
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.legend(frameon=False, markerscale=1.5)
    figure.tight_layout()
    figure.savefig(output_dir / "d2_group_distribution.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "d2_group_distribution.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def save_class_plot(embedding, groups, class_ids, category_names, output_dir):
    plt = configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True, sharey=True)
    color_map = plt.get_cmap("tab20")
    for axis, group, title, marker in zip(
            axes, ("Previous", "Current"), ("Task 1: previous classes", "Task 2: current classes"),
            ("o", "s")):
        ids = sorted(set(class_ids[groups == group].tolist()))
        for color_index, class_id in enumerate(ids):
            mask = (groups == group) & (class_ids == class_id)
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=12,
                         color=color_map(color_index % 20), marker=marker,
                         alpha=0.72, linewidths=0, rasterized=True,
                         label=category_names.get(class_id, str(class_id)))
        axis.set_title(title)
        axis.set_xlabel("t-SNE dimension 1")
        axis.legend(frameon=False, fontsize=7, ncol=2, markerscale=1.2,
                    loc="upper center", bbox_to_anchor=(0.5, -0.14))
    axes[0].set_ylabel("t-SNE dimension 2")
    figure.tight_layout()
    figure.savefig(output_dir / "d2_class_distribution.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "d2_class_distribution.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def save_data(output_dir, features, embedding, groups, class_ids, image_ids, query_ids, ious):
    np.savez_compressed(
        output_dir / "features.npz", features=features, embedding=embedding,
        groups=groups, class_ids=class_ids, image_ids=image_ids,
        query_ids=query_ids, ious=ious)
    with (output_dir / "embedding.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "group", "class_id", "image_id", "query_id", "iou"))
        for point, group, class_id, image_id, query_id, overlap in zip(
                embedding, groups, class_ids, image_ids, query_ids, ious):
            writer.writerow((float(point[0]), float(point[1]), group, int(class_id),
                             int(image_id), int(query_id), float(overlap)))


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    config_path = (args.config or run_dir / "run_config.json").resolve()
    checkpoint_path = (args.checkpoint or run_dir / "checkpoint.pth").resolve()
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = read_json(config_path)
    previous_ids = {int(value) for value in config.get("owod_previous_class_ids") or []}
    current_ids = {int(value) for value in config.get("owod_current_class_ids") or []}
    known_ids = {int(value) for value in config.get("owod_known_class_ids") or []}
    if not previous_ids or not current_ids or previous_ids | current_ids != known_ids:
        raise ValueError(
            "D2 run_config.json must contain consistent previous/current/known OWOD class IDs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu for a slow CPU run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    detector_args = detector_args_from_config(config, args)
    model = load_d2_model(detector_args, checkpoint_path, device)
    dataset = build_dataset("val", detector_args)
    records = extract_features(model, dataset, device, args, previous_ids, current_ids)
    if not records:
        raise RuntimeError("No query features passed the matching thresholds")

    selected = balanced_indices(records, args.max_per_class, args.max_per_group, args.seed)
    records = [records[index] for index in selected]
    features = np.stack([record["feature"] for record in records]).astype(np.float32)
    groups = np.asarray([record["group"] for record in records])
    class_ids = np.asarray([record["class_id"] for record in records], dtype=np.int64)
    image_ids = np.asarray([record["image_id"] for record in records], dtype=np.int64)
    query_ids = np.asarray([record["query_id"] for record in records], dtype=np.int64)
    ious = np.asarray([record["iou"] for record in records], dtype=np.float32)
    embedding = tsne_embedding(features, args.perplexity, args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    category_names = {int(key): value["name"] for key, value in dataset.coco.cats.items()}
    save_data(output_dir, features, embedding, groups, class_ids, image_ids, query_ids, ious)
    save_group_plot(embedding, groups, output_dir, args.max_per_group, args.seed)
    save_class_plot(embedding, groups, class_ids, category_names, output_dir)
    counts = {group: int((groups == group).sum()) for group in GROUP_ORDER}
    (output_dir / "summary.json").write_text(json.dumps({
        "run_dir": str(run_dir), "checkpoint": str(checkpoint_path),
        "validation_annotation": str(detector_args.val_ann),
        "decoder_layer": -1, "feature_dimension": int(features.shape[1]),
        "sample_counts": counts, "seed": args.seed,
        "iou_threshold": args.iou_threshold, "background_iou": args.background_iou,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved feature plots to {output_dir}")


if __name__ == "__main__":
    main()
