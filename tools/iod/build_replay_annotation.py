#!/usr/bin/env python3
"""Build a fixed-budget COCO increment annotation.

The output contains all images/annotations from the new-class increment and a
deterministically selected set of old-class images.  Selection is based only
on the supplied training annotations, never on validation annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import random
from collections import defaultdict
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def read_json(path: Path) -> Mapping:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_ids_from_annotation(path: Path) -> list[int]:
    payload = read_json(path)
    return sorted({int(item["category_id"]) for item in payload.get("annotations", [])})


def class_ids_from_risk(path: Path) -> list[int]:
    payload = read_json(path)
    return [int(value) for value in payload["old_classes"]]


def risk_values(path: Path) -> dict[int, float]:
    payload = read_json(path)
    classes = [int(value) for value in payload["old_classes"]]
    scores = [max(0.0, float(value)) for value in payload["risk"]]
    if len(classes) != len(scores):
        raise ValueError("risk JSON old_classes and risk have different lengths")
    return dict(zip(classes, scores))


def class_image_index(coco: Mapping) -> dict[int, list[int]]:
    index: dict[int, set[int]] = defaultdict(set)
    for annotation in coco.get("annotations", []):
        index[int(annotation["category_id"])].add(int(annotation["image_id"]))
    return {class_id: sorted(image_ids) for class_id, image_ids in index.items()}


def allocate_quotas(classes: list[int], budget: int, scores: Mapping[int, float] | None,
                    epsilon: float) -> dict[int, int]:
    if budget <= 0 or not classes:
        return {class_id: 0 for class_id in classes}
    weights = {class_id: (scores.get(class_id, 0.0) if scores is not None else 1.0)
               + epsilon for class_id in classes}
    total = sum(weights.values())
    raw = {class_id: budget * weights[class_id] / total for class_id in classes}
    quotas = {class_id: int(raw[class_id]) for class_id in classes}
    remainder = budget - sum(quotas.values())
    for class_id in sorted(classes, key=lambda value: (raw[value] - quotas[value], -value), reverse=True)[:remainder]:
        quotas[class_id] += 1
    return quotas


def select_images(coco: Mapping, classes: list[int], budget: int, seed: int,
                  scores: Mapping[int, float] | None, epsilon: float) -> tuple[list[int], dict[int, int]]:
    """Select a fixed number of unique images using uniform or risk quotas."""
    rng = random.Random(seed)
    by_class = class_image_index(coco)
    quotas = allocate_quotas(classes, budget, scores, epsilon)
    candidates = {}
    for class_id in classes:
        candidates[class_id] = list(by_class.get(class_id, []))
        rng.shuffle(candidates[class_id])

    selected: list[int] = []
    selected_set: set[int] = set()
    for class_id in sorted(classes, key=lambda value: (-quotas[value], value)):
        taken = 0
        for image_id in candidates[class_id]:
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            taken += 1
            if len(selected) >= budget or taken >= quotas[class_id]:
                break
    if len(selected) < budget:
        all_candidates = sorted({image_id for values in by_class.values() for image_id in values})
        rng.shuffle(all_candidates)
        for image_id in all_candidates:
            if image_id not in selected_set:
                selected.append(image_id)
                selected_set.add(image_id)
                if len(selected) == budget:
                    break
    return selected, quotas


def build_annotation(new_path: Path, base_path: Path, output_path: Path,
                     replay_ids: Iterable[int], replay_classes: list[int], seed: int,
                     selection: str, quotas: Mapping[int, int]) -> dict[str, object]:
    new_coco = read_json(new_path)
    base_coco = read_json(base_path)
    replay_set = {int(value) for value in replay_ids}
    replay_images = [image for image in base_coco.get("images", [])
                     if int(image["id"]) in replay_set]
    replay_annotations = [annotation for annotation in base_coco.get("annotations", [])
                          if int(annotation["image_id"]) in replay_set]

    images = [dict(image) for image in new_coco.get("images", [])]
    used_ids = {int(image["id"]): image.get("file_name") for image in images}
    next_image_id = max(used_ids, default=0) + 1
    replay_id_map = {}
    for image in replay_images:
        old_id = int(image["id"])
        new_id = old_id
        if old_id in used_ids and used_ids[old_id] != image.get("file_name"):
            new_id = next_image_id
            next_image_id += 1
        replay_id_map[old_id] = new_id
        copied = dict(image)
        copied["id"] = new_id
        if new_id not in used_ids:
            images.append(copied)
            used_ids[new_id] = copied.get("file_name")

    annotations = []
    for annotation in list(new_coco.get("annotations", [])) + replay_annotations:
        copied = dict(annotation)
        copied["id"] = len(annotations) + 1
        if int(annotation["image_id"]) in replay_id_map:
            copied["image_id"] = replay_id_map[int(annotation["image_id"])]
        annotations.append(copied)
    active_category_ids = {int(annotation["category_id"]) for annotation in annotations}
    category_by_id = {
        int(category["id"]): dict(category)
        for category in list(base_coco.get("categories", [])) + list(new_coco.get("categories", []))
    }
    missing_categories = active_category_ids - set(category_by_id)
    if missing_categories:
        raise ValueError(f"annotations reference undefined categories: {sorted(missing_categories)}")
    categories = [category_by_id[class_id] for class_id in sorted(active_category_ids)]
    combined = {
        "info": {"description": "COCO continual increment with fixed-budget replay",
                 "new_images": len(new_coco.get("images", [])),
                 "replay_images": len(replay_images), "replay_classes": replay_classes,
                 "selection": selection, "seed": seed},
        "licenses": new_coco.get("licenses", base_coco.get("licenses", [])),
        "images": images, "annotations": annotations,
        "categories": categories,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, ensure_ascii=True), encoding="utf-8")
    return {"schema_version": 1, "output": str(output_path), "selection": selection,
            "new_images": len(new_coco.get("images", [])),
            "replay_images": len(replay_images), "total_images": len(images),
            "replay_classes": replay_classes,
            "replay_class_quota": {str(k): int(v) for k, v in quotas.items()},
            "seed": seed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-ann", type=Path, required=True,
                        help="increment-only training annotation")
    parser.add_argument("--base-ann", type=Path, required=True,
                        help="stage-0 training annotation containing old classes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-budget", type=int, required=True,
                        help="total number of old replay images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="old class IDs; defaults to classes in --base-ann")
    parser.add_argument("--risk", type=Path, default=None,
                        help="optional risk JSON for risk-weighted class quotas")
    parser.add_argument("--selection", choices=("uniform", "risk"), default="uniform")
    parser.add_argument("--risk-epsilon", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replay_budget < 0:
        raise SystemExit("--replay-budget must be non-negative")
    if args.classes is not None and args.risk is not None:
        raise SystemExit("use either --classes or --risk, not both")
    if args.selection == "risk" and args.risk is None:
        raise SystemExit("--selection risk requires --risk")
    if args.risk is not None:
        replay_classes: Iterable[int] = class_ids_from_risk(args.risk)
    elif args.classes is not None:
        replay_classes = args.classes
    else:
        replay_classes = class_ids_from_annotation(args.base_ann)
    replay_classes = sorted({int(value) for value in replay_classes})
    if not replay_classes and args.replay_budget:
        raise SystemExit("no replay classes were found")

    base_coco = read_json(args.base_ann)
    scores = risk_values(args.risk) if args.selection == "risk" else None
    replay_ids, quotas = select_images(
        base_coco, replay_classes, args.replay_budget, args.seed, scores, args.risk_epsilon)
    if len(replay_ids) != args.replay_budget:
        raise SystemExit(
            f"requested {args.replay_budget} unique replay images, but only "
            f"{len(replay_ids)} are available for the selected classes"
        )
    info = build_annotation(args.new_ann, args.base_ann, args.output, replay_ids,
                            replay_classes, args.seed, args.selection, quotas)
    info.update({
        "schema_version": 1,
        "selection": args.selection,
        "new_annotation": str(args.new_ann.resolve()),
        "base_annotation": str(args.base_ann.resolve()),
        "replay_budget": args.replay_budget,
        "seed": args.seed,
        "risk": str(args.risk.resolve()) if args.risk is not None else None,
    })
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
