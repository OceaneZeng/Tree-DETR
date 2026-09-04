"""Shared COCO split helpers used by the OWOD protocol builder."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, Mapping, Sequence


def image_category_sets(coco: Mapping) -> Dict[int, set[int]]:
    result: Dict[int, set[int]] = defaultdict(set)
    for annotation in coco.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) == 0:
            result[int(annotation["image_id"])].add(int(annotation["category_id"]))
    return result


def assign_images(images: Sequence[Mapping], annotations: Sequence[Mapping],
                  stages: Sequence[Sequence[int]], seed: int) -> Dict[int, int]:
    """Assign every image to one stage while balancing category coverage."""
    image_categories = image_category_sets({"annotations": annotations})
    stage_of = {category: index for index, categories in enumerate(stages)
                for category in categories}
    generator = random.Random(seed)
    image_ids = [int(image["id"]) for image in images]
    generator.shuffle(image_ids)
    totals = Counter(category for categories in image_categories.values()
                     for category in categories)
    target = {
        index: sum(totals.get(category, 0) for category in categories) /
        max(1, len(categories))
        for index, categories in enumerate(stages)
    }
    assigned = Counter()
    category_counts = Counter()
    result: Dict[int, int] = {}
    for image_id in image_ids:
        categories = image_categories.get(image_id, set())
        candidates = sorted({stage_of[category] for category in categories
                             if category in stage_of})
        if not candidates:
            candidates = list(range(len(stages)))

        def score(stage: int) -> tuple[float, int, int]:
            deficit = target[stage] - sum(category_counts[category]
                                          for category in stages[stage])
            return deficit, -assigned[stage], -stage

        selected = max(candidates, key=score)
        result[image_id] = selected
        assigned[selected] += 1
        for category in categories:
            category_counts[category] += 1
    return result


def filter_coco(coco: Mapping, image_ids: Iterable[int],
                active_categories: Iterable[int], mapping: Mapping[int, int]) -> dict:
    image_set = {int(value) for value in image_ids}
    active = {int(value) for value in active_categories}
    annotations = []
    for annotation in coco.get("annotations", []):
        if (int(annotation["image_id"]) not in image_set or
                int(annotation["category_id"]) not in active):
            continue
        item = dict(annotation)
        item["category_id"] = int(mapping[item["category_id"]])
        annotations.append(item)
    images = [dict(image) for image in coco.get("images", [])
              if int(image["id"]) in image_set]
    categories_by_id = {int(category["id"]): category
                        for category in coco.get("categories", [])}
    categories = []
    for category_id in active_categories:
        item = dict(categories_by_id[int(category_id)])
        item["id"] = int(mapping[int(category_id)])
        categories.append(item)
    return {
        "info": dict(coco.get("info", {})),
        "licenses": list(coco.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def choose_memory(coco: Mapping, image_ids: Sequence[int],
                  active_categories: Sequence[int], fraction: float,
                  seed: int) -> list[int]:
    """Select a deterministic class-balanced exemplar memory."""
    if not 0 < fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    by_class: Dict[int, list[int]] = defaultdict(list)
    image_categories = image_category_sets(coco)
    allowed = set(active_categories)
    for image_id in image_ids:
        for category in sorted(image_categories.get(int(image_id), set()) & allowed):
            by_class[category].append(int(image_id))
    total = max(1, round(len(set(image_ids)) * fraction))
    generator = random.Random(seed)
    for values in by_class.values():
        generator.shuffle(values)
    selected: list[int] = []
    while len(selected) < total and by_class:
        progressed = False
        for category in sorted(list(by_class)):
            if by_class[category]:
                candidate = by_class[category].pop()
                if candidate not in selected:
                    selected.append(candidate)
                    progressed = True
                    if len(selected) >= total:
                        break
            if not by_class[category]:
                del by_class[category]
        if not progressed:
            break
    return sorted(selected)
