#!/usr/bin/env python3
"""Precompute CAT's input-driven selective-search proposals for M-OWODB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, stage_files


DEFAULT_MANIFEST = ROOT / "data/coco-owod/m-owodb/split_manifest.json"
DEFAULT_COCO = ROOT / "data/coco"
DEFAULT_OUTPUT = ROOT / "data/derived/m-owodb-cat/selective_search.sqlite3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selective_search(image_path, mode, max_proposals, min_area):
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "CAT proposal generation requires opencv-contrib-python-headless") from error
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError(
            "OpenCV lacks ximgproc.segmentation; install opencv-contrib-python-headless")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    engine = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    engine.setBaseImage(image)
    if mode == "quality":
        engine.switchToSelectiveSearchQuality()
    else:
        engine.switchToSelectiveSearchFast()
    boxes = []
    seen = set()
    for x, y, box_width, box_height in engine.process():
        x, y = max(0, int(x)), max(0, int(y))
        box_width = min(int(box_width), width - x)
        box_height = min(int(box_height), height - y)
        box = (x, y, box_width, box_height)
        if box_width * box_height < min_area or box in seen:
            continue
        seen.add(box)
        boxes.append(list(box))
        if len(boxes) >= max_proposals:
            break
    return boxes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coco-path", type=Path, default=DEFAULT_COCO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("fast", "quality"), default="fast")
    parser.add_argument("--max-proposals", type=int, default=2000)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_proposals <= 0 or args.min_area <= 0:
        raise ValueError("Proposal count and minimum area must be positive")
    manifest_path = args.manifest.resolve()
    coco_path = args.coco_path.resolve()
    images = {}
    annotation_hashes = {}
    for stage in range(4):
        manifest, files = stage_files(manifest_path, stage)
        if manifest.get("protocol") != "m-owodb":
            raise ValueError("CAT comparison requires protocol=m-owodb")
        increment = read_json(files["increment_train"])
        images.update({int(image["id"]): image for image in increment["images"]})
        annotation_hashes[str(files["increment_train"])] = file_sha256(files["increment_train"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Proposal database exists; pass --resume: {args.output}")
    metadata = {
        "schema_version": "1",
        "method": "CAT input-driven pseudo-labelling",
        "algorithm": "OpenCV selective search",
        "coordinate_format": "xywh_absolute",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "annotation_sha256": json.dumps(annotation_hashes, sort_keys=True),
        "mode": args.mode,
        "max_proposals": str(args.max_proposals),
        "min_area": str(args.min_area),
    }
    with sqlite3.connect(args.output) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS image_proposals '
                           '(image_id INTEGER PRIMARY KEY, boxes BLOB NOT NULL)')
        existing = dict(connection.execute('SELECT key, value FROM metadata'))
        if existing and existing != metadata:
            raise ValueError("Existing proposal database uses a different manifest/configuration")
        connection.executemany('INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)',
                               metadata.items())
        completed = {row[0] for row in connection.execute('SELECT image_id FROM image_proposals')}
        for position, (image_id, image) in enumerate(sorted(images.items()), start=1):
            if image_id in completed:
                continue
            image_path = coco_path / "train2017" / image["file_name"]
            boxes = selective_search(
                image_path, args.mode, args.max_proposals, args.min_area)
            blob = zlib.compress(json.dumps(boxes, separators=(',', ':')).encode('utf-8'))
            connection.execute('INSERT INTO image_proposals(image_id, boxes) VALUES (?, ?)',
                               (image_id, blob))
            if position % 100 == 0:
                connection.commit()
                print(f"processed {position}/{len(images)}", flush=True)
        connection.commit()
        count = connection.execute('SELECT COUNT(*) FROM image_proposals').fetchone()[0]
    print(f"wrote {count} image proposal sets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
