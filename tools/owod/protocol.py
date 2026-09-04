"""Validation and path resolution for official OWOD stage annotations."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Mapping


PROTOCOLS = {"m-owodb": "M-OWODB", "s-owodb": "S-OWODB"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def annotation_class_ids(payload: Mapping) -> set[int]:
    return {int(item["category_id"]) for item in payload.get("annotations", [])}


def annotation_image_ids(payload: Mapping) -> set[int]:
    return {int(item["image_id"]) for item in payload.get("annotations", [])}


def _find_stage_file(stage_dir: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = stage_dir / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Missing official annotation in {stage_dir}; expected one of {list(names)}")


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_files(manifest_path: Path, stage: int) -> tuple[dict, dict[str, Path]]:
    manifest = read_json(manifest_path)
    if not manifest.get("official_annotations"):
        raise ValueError(
            "Manifest was not produced from externally supplied official OWOD annotations")
    stages = manifest.get("stages", [])
    if stage < 0 or stage >= len(stages):
        raise ValueError(f"stage {stage} is outside manifest [0, {len(stages)})")
    record = stages[stage]
    files = {
        key: resolve_manifest_path(manifest_path, value)
        for key, value in record["files"].items()
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Manifest annotation is missing: {path}")
    return manifest, files


def build_official_manifest(annotation_root: Path, protocol: str, source_reference: str,
                            allow_image_overlap: bool = False) -> dict:
    """Validate externally supplied stage files without synthesizing a split."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported OWOD protocol: {protocol}")
    annotation_root = annotation_root.resolve()
    if not source_reference.strip():
        raise ValueError("An official release URL, repository commit, or archive ID is required")
    records = []
    previous_classes: set[int] = set()
    prior_increment_images: set[int] = set()
    overlaps: dict[str, list[int]] = {}
    full_validation_classes: list[set[int]] = []

    for index in range(4):
        stage_dir = annotation_root / f"stage_{index}"
        increment_path = _find_stage_file(stage_dir, (
            "instances_increment_train2017.json",
            "instances_increment_only_train2017.json",
        ))
        train_path = _find_stage_file(stage_dir, ("instances_train2017.json",))
        known_val_path = _find_stage_file(stage_dir, ("instances_val2017.json",))
        full_val_path = _find_stage_file(stage_dir, ("instances_val2017_full.json",))

        increment = read_json(increment_path)
        train = read_json(train_path)
        full_val = read_json(full_val_path)
        current_classes = annotation_class_ids(increment)
        active_classes = annotation_class_ids(train)
        full_classes = annotation_class_ids(full_val)
        full_validation_classes.append(full_classes)
        if not current_classes:
            raise ValueError(f"stage_{index} increment has no annotated classes")
        duplicate_classes = current_classes & previous_classes
        if duplicate_classes:
            raise ValueError(
                f"stage_{index} repeats classes from earlier tasks: {sorted(duplicate_classes)}")
        expected_active = previous_classes | current_classes
        if active_classes != expected_active:
            raise ValueError(
                f"stage_{index} train classes do not equal previous + current; "
                f"missing={sorted(expected_active - active_classes)}, "
                f"extra={sorted(active_classes - expected_active)}")
        if not expected_active.issubset(full_classes):
            raise ValueError(f"stage_{index} full validation omits known classes")

        increment_images = annotation_image_ids(increment)
        duplicate_images = sorted(increment_images & prior_increment_images)
        if duplicate_images:
            overlaps[f"stage_{index}"] = duplicate_images
        prior_increment_images.update(increment_images)
        previous_classes = expected_active
        records.append({
            "index": index,
            "classes": sorted(current_classes),
            "active_classes": sorted(active_classes),
            "unknown_classes": sorted(full_classes - active_classes),
            "increment_image_count": len(increment_images),
            "train_image_count": len({int(item["id"]) for item in train.get("images", [])}),
            "files": {
                "increment_train": str(increment_path),
                "train": str(train_path),
                "known_val": str(known_val_path),
                "full_val": str(full_val_path),
            },
            "sha256": {
                "increment_train": file_sha256(increment_path),
                "train": file_sha256(train_path),
                "known_val": file_sha256(known_val_path),
                "full_val": file_sha256(full_val_path),
            },
        })

    if len(previous_classes) != 80:
        raise ValueError(
            f"Official COCO OWOD protocols must cover 80 classes, found {len(previous_classes)}")
    for index, full_classes in enumerate(full_validation_classes):
        if full_classes != previous_classes:
            raise ValueError(
                f"stage_{index} full validation must contain all 80 protocol classes; "
                f"found {len(full_classes)}")
    if overlaps and not allow_image_overlap:
        counts = {stage: len(images) for stage, images in overlaps.items()}
        raise ValueError(
            "Increment annotations reuse images across tasks. This is not comparable to "
            f"the corrected DEUS Table 1 setting: {counts}")
    return {
        "schema_version": 2,
        "benchmark": PROTOCOLS[protocol],
        "protocol": protocol,
        "official_annotations": True,
        "source_reference": source_reference.strip(),
        "annotation_root": str(annotation_root),
        "paper_comparable": not bool(overlaps),
        "annotation_overlap": {key: value for key, value in overlaps.items()},
        "category_order_source_ids": [
            class_id for record in records for class_id in record["classes"]
        ],
        "stages": records,
        "integrity": {
            "category_count": len(previous_classes),
            "stage_count": len(records),
            "increment_images_disjoint": not bool(overlaps),
        },
    }
