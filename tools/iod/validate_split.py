#!/usr/bin/env python3
"""Validate a generated COCO IOD split and its annotation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coco_incremental import read_json, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    errors = validate_manifest(manifest)
    root = args.manifest.parent
    for stage in manifest["stages"]:
        for key in ("train_annotation", "val_annotation"):
            path = root / stage[key]
            if not path.is_file():
                errors.append(f"missing {key}: {path}")
                continue
            data = read_json(path)
            image_ids = {int(x["id"]) for x in data.get("images", [])}
            ann_image_ids = {int(x["image_id"]) for x in data.get("annotations", [])}
            if not ann_image_ids <= image_ids:
                errors.append(f"annotation image leak in {path}")
            category_ids = {int(x["id"]) for x in data.get("categories", [])}
            if any(int(x["category_id"]) not in category_ids for x in data.get("annotations", [])):
                errors.append(f"annotation category leak in {path}")
    result = {"passed": not errors, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
