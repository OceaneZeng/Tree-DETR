#!/usr/bin/env python
"""Extract real known/unknown query features for the Stage-0 gate suite."""

import argparse
import json
from pathlib import Path

import common
from main import get_args_parser


def main():
    parser = argparse.ArgumentParser(
        description="Extract Tree-DETR Stage-0 features",
        parents=[get_args_parser()])
    parser.add_argument("--stage0_ann", required=True,
                        help="COCO annotation JSON containing known and held-out classes")
    parser.add_argument("--split_metadata", required=True,
                        help="metadata JSON produced by prepare_oxford_pet.py")
    parser.add_argument("--features_output", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    with open(args.split_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if args.num_classes is None:
        args.num_classes = int(metadata["num_known"])
    args.unknown_class_ids = list(metadata["unknown_remapped_ids"])
    if not args.resume:
        parser.error("--resume is required")

    features = common.extract_features(args)
    output = Path(args.features_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.save(str(output))
    print(f"Saved {len(features.labels)} detections to {output}")
    print(f"known={int((features.kind == common.KNOWN).sum())} "
          f"unknown={int((features.kind == common.UNKNOWN).sum())} "
          f"background={int((features.kind == common.BACKGROUND).sum())}")


if __name__ == "__main__":
    main()
