"""Class-scoped exemplar selection and COCO annotation assembly."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def _load_coco(source) -> Dict:
    if isinstance(source, Mapping):
        return dict(source)
    with Path(source).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_image_index(coco: Mapping) -> Dict[int, List[int]]:
    index = defaultdict(set)
    for annotation in coco.get("annotations", []):
        index[int(annotation["category_id"])].add(int(annotation["image_id"]))
    return {class_id: sorted(image_ids) for class_id, image_ids in index.items()}


def select_replay_images(coco: Mapping, class_ids: Iterable[int], exemplars_per_class: int,
                         seed: int = 42) -> List[int]:
    """Select a deterministic exemplar quota independently for each class."""
    if exemplars_per_class <= 0:
        return []
    rng = random.Random(seed)
    by_class = class_image_index(coco)
    classes = sorted({int(class_id) for class_id in class_ids})
    selected = set()
    for class_id in classes:
        candidates = list(by_class.get(class_id, []))
        rng.shuffle(candidates)
        selected.update(candidates[:exemplars_per_class])
    return sorted(selected)


def build_increment_annotation(new_annotations, base_annotations,
                               replay_classes: Sequence[int], exemplars_per_class: int,
                               output_path, seed: int = 42) -> Dict[str, object]:
    """Combine fully labeled new-class images with selected old exemplars."""
    new_coco = _load_coco(new_annotations)
    base_coco = _load_coco(base_annotations)
    replay_ids = set(select_replay_images(
        base_coco, replay_classes, exemplars_per_class, seed=seed,
    ))
    replay_images = [image for image in base_coco.get("images", [])
                     if int(image["id"]) in replay_ids]
    replay_annotations = [annotation for annotation in base_coco.get("annotations", [])
                          if int(annotation["image_id"]) in replay_ids]

    images = [dict(image) for image in new_coco.get("images", [])]
    used_image_ids = {int(image["id"]): image.get("file_name") for image in images}
    next_image_id = max(used_image_ids, default=0) + 1
    replay_id_map = {}
    for image in replay_images:
        old_id = int(image["id"])
        new_id = old_id
        if old_id in used_image_ids and used_image_ids[old_id] != image.get("file_name"):
            new_id = next_image_id
            next_image_id += 1
        replay_id_map[old_id] = new_id
        copied = dict(image)
        copied["id"] = new_id
        if new_id not in used_image_ids:
            images.append(copied)
            used_image_ids[new_id] = copied.get("file_name")

    annotations = []
    # Both files may originate from the same COCO source. Preserve the source
    # annotation identity while merging so an overlapping image cannot acquire
    # duplicate boxes merely because output IDs are regenerated.
    annotation_keys = set()
    for annotation in new_coco.get("annotations", []):
        copied = dict(annotation)
        source_key = (int(copied["image_id"]), int(copied.get("id", -1)))
        annotation_keys.add(source_key)
        copied["id"] = len(annotations) + 1
        annotations.append(copied)
    for annotation in replay_annotations:
        copied = dict(annotation)
        copied["image_id"] = replay_id_map[int(annotation["image_id"])]
        source_key = (int(copied["image_id"]), int(annotation.get("id", -1)))
        if source_key in annotation_keys:
            continue
        annotation_keys.add(source_key)
        copied["id"] = len(annotations) + 1
        annotations.append(copied)

    categories_by_id = {
        int(category["id"]): dict(category)
        for category in [*base_coco.get("categories", []), *new_coco.get("categories", [])]
    }

    combined = {
        "info": {
            "description": "Graph-local continual adaptation increment",
            "new_images": len(new_coco.get("images", [])),
            "replay_images": len(replay_images),
            "replay_classes": [int(class_id) for class_id in replay_classes],
            "exemplars_per_class": int(exemplars_per_class),
            "seed": int(seed),
        },
        "licenses": new_coco.get("licenses", base_coco.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": [categories_by_id[class_id] for class_id in sorted(categories_by_id)],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=True)
    return {
        "output": str(destination),
        "new_images": len(new_coco.get("images", [])),
        "replay_images": len(replay_images),
        "total_images": len(images),
        "replay_classes": [int(class_id) for class_id in replay_classes],
        "exemplars_per_class": int(exemplars_per_class),
        "selected_images_per_class": {
            str(class_id): len(set(class_image_index(base_coco).get(int(class_id), [])) & replay_ids)
            for class_id in replay_classes
        },
        "replay_source": (str(base_annotations)
                          if not isinstance(base_annotations, Mapping) else "<in-memory>"),
    }
