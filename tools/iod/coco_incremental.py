#!/usr/bin/env python3
"""Build and validate disjoint-image COCO incremental-detection splits.

The generated files are deliberately ordinary COCO JSON files so they can be
used by the existing dataset loader.  Images are assigned to exactly one
training stage.  An image may contain labels from other stages; those labels
are omitted from that stage's annotation rather than leaking the future
classes.  Validation remains a fixed, shared COCO val split and is filtered by
the active class set for each stage.

This is a protocol utility, not a training implementation.  It records the
category order, image assignment, class counts, memory selection, and all
integrity checks in ``split_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROTOCOLS = {
    "40+20x2": (40, 20, 2),
    "40+10x4": (40, 10, 4),
    "70+10": (70, 10, 1),
}


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def protocol_stages(protocol: str) -> List[List[int]]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol {protocol}; choose {sorted(PROTOCOLS)}")
    base, increment, count = PROTOCOLS[protocol]
    return [list(range(base))] + [list(range(base + i * increment, base + (i + 1) * increment))
                                   for i in range(count)]


def category_order(categories: Sequence[Mapping], order: str, seed: int) -> List[int]:
    ids = sorted(int(c["id"]) for c in categories)
    if len(ids) != 80:
        raise ValueError(f"COCO IOD protocols require 80 categories, found {len(ids)}")
    if order == "id":
        return ids
    rng = random.Random(seed)
    if order == "random":
        rng.shuffle(ids)
        return ids
    raise ValueError("order must be id or random")


def remap_categories(categories: Sequence[Mapping], order: Sequence[int]) -> Tuple[List[Dict], Dict[int, int]]:
    """Keep official COCO IDs while recording the experimental order.

    Deformable DETR's official COCO checkpoint has a 91-row head (COCO IDs
    include gaps).  Renumbering categories to 0..79 would silently mismatch
    that head and the COCO evaluator, so the protocol only changes stage
    membership and preserves source IDs in every generated annotation.
    """
    by_id = {int(c["id"]): c for c in categories}
    mapping = {old: old for old in order}
    result = []
    for old_id in order:
        source = dict(by_id[old_id])
        result.append(source)
    return result, mapping


def image_category_sets(coco: Mapping) -> Dict[int, set]:
    result: Dict[int, set] = defaultdict(set)
    for ann in coco.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 0:
            result[int(ann["image_id"])].add(int(ann["category_id"]))
    return result


def assign_images(images: Sequence[Mapping], annotations: Sequence[Mapping],
                  stages: Sequence[Sequence[int]], seed: int) -> Dict[int, int]:
    """Greedily assign each train image to one stage with balanced coverage.

    Assignment uses only train annotations and is deterministic for a seed.
    Images containing categories from multiple stages are assigned to the
    stage with the largest current deficit, then to the stage with fewer
    assigned images.  This keeps the image sets disjoint while avoiding a
    hidden future-label split.
    """
    image_cats = image_category_sets({"annotations": annotations})
    stage_of = {cat: i for i, cats in enumerate(stages) for cat in cats}
    rng = random.Random(seed)
    ids = [int(image["id"]) for image in images]
    rng.shuffle(ids)
    totals = Counter(cat for cats in image_cats.values() for cat in cats)
    target = {i: sum(totals.get(cat, 0) for cat in cats) / max(1, len(cats))
              for i, cats in enumerate(stages)}
    assigned = Counter()
    category_counts = Counter()
    result = {}
    for image_id in ids:
        cats = image_cats.get(image_id, set())
        candidates = sorted({stage_of[cat] for cat in cats if cat in stage_of})
        if not candidates:
            # Background-only images are assigned after labelled images have
            # established stage proportions.
            candidates = list(range(len(stages)))
        def score(stage: int) -> Tuple[float, int, int]:
            deficit = target[stage] - sum(category_counts[c] for c in stages[stage])
            return (deficit, -assigned[stage], -stage)
        chosen = max(candidates, key=score)
        result[image_id] = chosen
        assigned[chosen] += 1
        for cat in cats:
            category_counts[cat] += 1
    return result


def filter_coco(coco: Mapping, image_ids: Iterable[int], active_categories: Iterable[int],
                mapping: Mapping[int, int]) -> Dict:
    image_set = set(int(x) for x in image_ids)
    active = set(int(x) for x in active_categories)
    annotations = []
    for ann in coco.get("annotations", []):
        if int(ann["image_id"]) not in image_set or int(ann["category_id"]) not in active:
            continue
        item = dict(ann)
        item["category_id"] = int(mapping[item["category_id"]])
        annotations.append(item)
    images = [dict(image) for image in coco.get("images", []) if int(image["id"]) in image_set]
    categories = []
    for category_id in active_categories:
        source = next(c for c in coco["categories"] if int(c["id"]) == int(category_id))
        item = dict(source)
        item["id"] = int(mapping[int(category_id)])
        categories.append(item)
    return {"info": dict(coco.get("info", {})), "licenses": list(coco.get("licenses", [])),
            "images": images, "annotations": annotations, "categories": categories}


def choose_memory(coco: Mapping, image_ids: Sequence[int], active_categories: Sequence[int],
                  fraction: float, seed: int) -> List[int]:
    """Select a fixed-size, class-balanced image memory from one stage."""
    if not 0 < fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    by_class: Dict[int, List[int]] = defaultdict(list)
    image_cats = image_category_sets(coco)
    allowed = set(active_categories)
    for image_id in image_ids:
        for cat in sorted(image_cats.get(int(image_id), set()) & allowed):
            by_class[cat].append(int(image_id))
    total = max(1, round(len(set(image_ids)) * fraction))
    rng = random.Random(seed)
    for values in by_class.values():
        rng.shuffle(values)
    selected: List[int] = []
    while len(selected) < total and by_class:
        progressed = False
        for cat in sorted(list(by_class)):
            if by_class[cat]:
                candidate = by_class[cat].pop()
                if candidate not in selected:
                    selected.append(candidate)
                    progressed = True
                    if len(selected) >= total:
                        break
            if not by_class[cat]:
                del by_class[cat]
        if not progressed:
            break
    return sorted(selected)


def validate_manifest(manifest: Mapping) -> List[str]:
    errors: List[str] = []
    if "source_train_image_ids" in manifest and "source_val_image_ids" in manifest:
        overlap = set(manifest["source_train_image_ids"]) & set(manifest["source_val_image_ids"])
        if overlap:
            errors.append(f"source train/val image overlap: {len(overlap)}")
    stages = manifest["stages"]
    seen_images = set()
    seen_classes = set()
    for stage in stages:
        classes = set(stage["classes"])
        overlap = classes & seen_classes
        if overlap:
            errors.append(f"class overlap at stage {stage['index']}: {sorted(overlap)}")
        seen_classes |= classes
        images = set(stage["train_image_ids"])
        image_overlap = images & seen_images
        if image_overlap:
            errors.append(f"train image overlap at stage {stage['index']}: {len(image_overlap)}")
        seen_images |= images
        memory = set(stage["memory_image_ids"])
        if not memory <= images | set().union(*(set(s["train_image_ids"]) for s in stages[:stage["index"]])):
            errors.append(f"memory contains image outside seen training stages at stage {stage['index']}")
        if len(memory) != stage["memory_size"]:
            errors.append(f"memory size mismatch at stage {stage['index']}")
    return errors


def build_split(coco_root: Path, output_root: Path, protocol: str, order_name: str,
                seed: int, memory_fraction: float) -> Dict:
    train = read_json(coco_root / "annotations" / "instances_train2017.json")
    val = read_json(coco_root / "annotations" / "instances_val2017.json")
    order = category_order(train["categories"], order_name, seed)
    original_stages = [[order[i] for i in stage] for stage in protocol_stages(protocol)]
    mapping_categories, mapping = remap_categories(train["categories"], order)
    assignments = assign_images(train["images"], train["annotations"], original_stages, seed)
    stages = []
    prior_images: List[int] = []
    for index, classes in enumerate(original_stages):
        train_ids = sorted(image_id for image_id, stage in assignments.items() if stage == index)
        memory_ids = choose_memory(train, prior_images, list(sum(original_stages[:index], [])),
                                   memory_fraction, seed + index)
        prior_images.extend(train_ids)
        active = list(sum(original_stages[:index + 1], []))
        train_json = filter_coco(train, train_ids + memory_ids, active, mapping)
        increment_json = filter_coco(train, train_ids, classes, mapping)
        val_json = filter_coco(val, [int(x["id"]) for x in val["images"]], active, mapping)
        stage_dir = output_root / f"stage_{index}"
        write_json(stage_dir / "instances_train2017.json", train_json)
        write_json(stage_dir / "instances_increment_only_train2017.json", increment_json)
        write_json(stage_dir / "instances_val2017.json", val_json)
        write_json(stage_dir / "train_image_ids.json", {"image_ids": train_ids})
        write_json(stage_dir / "memory_image_ids.json", {"image_ids": memory_ids})
        stages.append({"index": index, "classes": list(classes),
                       "active_classes": list(active), "source_classes": classes,
                       "train_image_ids": train_ids,
                       "memory_image_ids": memory_ids, "memory_size": len(memory_ids),
                       "train_annotation": str((stage_dir / "instances_train2017.json").relative_to(output_root)),
                       "increment_annotation": str((stage_dir / "instances_increment_only_train2017.json").relative_to(output_root)),
                       "val_annotation": str((stage_dir / "instances_val2017.json").relative_to(output_root))})
    manifest = {"schema_version": 1, "protocol": protocol, "order": order_name,
                "seed": seed, "memory_fraction": memory_fraction,
                "category_order_source_ids": order, "categories": mapping_categories,
                "source_train_image_ids": [int(x["id"]) for x in train["images"]],
                "source_val_image_ids": [int(x["id"]) for x in val["images"]],
                "stages": stages}
    errors = validate_manifest(manifest)
    manifest["integrity"] = {"passed": not errors, "errors": errors,
                              "train_image_count": len(seen_all_images(stages)),
                              "category_count": len(order)}
    if errors:
        raise RuntimeError("generated split failed integrity checks: " + "; ".join(errors))
    write_json(output_root / "split_manifest.json", manifest)
    return manifest


def seen_all_images(stages: Sequence[Mapping]) -> set:
    return set().union(*(set(s["train_image_ids"]) for s in stages)) if stages else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--order", choices=("id", "random"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    args = parser.parse_args()
    manifest = build_split(args.coco_root, args.output_root, args.protocol, args.order,
                           args.seed, args.memory_fraction)
    print(json.dumps(manifest["integrity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
