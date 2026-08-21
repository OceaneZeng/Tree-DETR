#!/usr/bin/env python
"""Collapse a prepared Oxford-Pet COCO split from breeds to cat/dog species."""

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_split(source_root: Path, output_root: Path, split: str,
                  category_to_species: dict[int, int]) -> dict:
    ann_path = source_root / "annotations" / f"instances_{split}2017.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    converted = dict(data)
    converted["categories"] = [
        {"id": 0, "name": "cat", "supercategory": "animal"},
        {"id": 1, "name": "dog", "supercategory": "animal"},
    ]
    annotations = []
    for annotation in data["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in category_to_species:
            raise ValueError(f"category {category_id} is missing from source categories")
        item = dict(annotation)
        item["category_id"] = category_to_species[category_id]
        annotations.append(item)
    converted["annotations"] = annotations

    output_images = output_root / f"{split}2017"
    source_images = source_root / f"{split}2017"
    for image in data["images"]:
        link_or_copy(source_images / image["file_name"], output_images / image["file_name"])
    output_ann = output_root / "annotations" / f"instances_{split}2017.json"
    output_ann.parent.mkdir(parents=True, exist_ok=True)
    output_ann.write_text(json.dumps(converted), encoding="utf-8")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="prepared breed-level COCO split")
    parser.add_argument("--output", type=Path, required=True,
                        help="new species-level COCO split")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    categories = json.loads(
        (source / "annotations" / "instances_train2017.json").read_text(encoding="utf-8")
    )["categories"]
    category_to_species = {}
    for category in categories:
        species = str(category.get("supercategory", "")).lower()
        if species not in {"cat", "dog"}:
            raise ValueError(f"unsupported supercategory: {species!r}")
        category_to_species[int(category["id"])] = 0 if species == "cat" else 1

    train = convert_split(source, output, "train", category_to_species)
    val = convert_split(source, output, "val", category_to_species)
    metadata = {
        "source": str(source),
        "num_known": 2,
        "known_names": ["cat", "dog"],
        "train_images": len(train["images"]),
        "val_known_images": len(val["images"]),
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
