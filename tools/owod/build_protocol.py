"""Build deterministic M-OWODB/S-OWODB-style COCO stage annotations.

For S-OWODB, pass the official super-category grouping with ``--groups-json``
when reproducing published numbers.  Without it, the script creates an
explicit, deterministic grouping from COCO's ``supercategory`` field and
records that fact in the manifest; this is useful for smoke tests, not for a
paper table claiming exact historical comparability.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.owod.protocol_utils import (assign_images, choose_memory, filter_coco,
                                       image_category_sets)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def category_order(categories: Sequence[Mapping], mode: str, seed: int) -> list[int]:
    ids = sorted(int(c["id"]) for c in categories)
    if mode == "id":
        return ids
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def load_groups(path: Path | None, categories: Sequence[Mapping], protocol: str,
                order: Sequence[int], allow_heuristic: bool = False) -> tuple[list[list[int]], str]:
    if protocol == "m-owodb":
        if len(order) != 80:
            raise ValueError(f"M-OWODB expects 80 categories, found {len(order)}")
        return [list(order[i:i + 20]) for i in range(0, 80, 20)], "ordered_20_per_stage"
    if path is not None:
        payload = read_json(path)
        groups = payload.get("stages", payload) if isinstance(payload, dict) else payload
        if isinstance(groups, dict):
            groups = list(groups.values())
        result = [[int(x) for x in group] for group in groups]
        if not result or set(sum(result, [])) != set(order):
            raise ValueError("--groups-json must partition every COCO category exactly once")
        return result, "user_supplied_groups"

    if not allow_heuristic:
        raise ValueError(
            "S-OWODB requires the official class grouping via --groups-json; "
            "use --allow-heuristic-groups only for non-paper smoke tests")
    by_supercategory: dict[str, list[int]] = defaultdict(list)
    for category in categories:
        by_supercategory[str(category.get("supercategory", "unknown"))].append(int(category["id"]))
    groups = [sorted(values) for _name, values in sorted(by_supercategory.items())]
    # Pack groups into four balanced bins without splitting a super-category.
    bins: list[list[int]] = [[], [], [], []]
    for group in sorted(groups, key=lambda values: (-len(values), values[0])):
        target = min(range(4), key=lambda index: (len(bins[index]), index))
        bins[target].extend(group)
    return bins, "coarse_supercategory_greedy"


def build(coco_root: Path, output_root: Path, protocol: str, order_mode: str,
          seed: int, memory_fraction: float, groups_path: Path | None,
          allow_heuristic_groups: bool = False) -> dict:
    train = read_json(coco_root / "annotations" / "instances_train2017.json")
    val = read_json(coco_root / "annotations" / "instances_val2017.json")
    order = category_order(train["categories"], order_mode, seed)
    stages, grouping_source = load_groups(
        groups_path, train["categories"], protocol, order,
        allow_heuristic=allow_heuristic_groups)
    mapping = {category_id: category_id for category_id in order}
    assignments = assign_images(train["images"], train["annotations"], stages, seed)
    prior_images: list[int] = []
    stage_records = []
    all_val_ids = [int(image["id"]) for image in val["images"]]
    for index, classes in enumerate(stages):
        train_ids = sorted(image_id for image_id, stage in assignments.items() if stage == index)
        memory_ids = choose_memory(train, prior_images, list(sum(stages[:index], [])),
                                   memory_fraction, seed + index)
        active = list(sum(stages[:index + 1], []))
        stage_dir = output_root / f"stage_{index}"
        current = filter_coco(train, train_ids, classes, mapping)
        seen = filter_coco(train, train_ids + memory_ids, active, mapping)
        known_val = filter_coco(val, all_val_ids, active, mapping)
        full_val = filter_coco(val, all_val_ids, order, mapping)
        write_json(stage_dir / "instances_increment_train2017.json", current)
        write_json(stage_dir / "instances_train2017.json", seen)
        write_json(stage_dir / "instances_val2017.json", known_val)
        write_json(stage_dir / "instances_val2017_full.json", full_val)
        write_json(stage_dir / "train_image_ids.json", {"image_ids": train_ids})
        write_json(stage_dir / "memory_image_ids.json", {"image_ids": memory_ids})
        write_json(stage_dir / "unknown_category_ids.json", {"category_ids": sorted(set(order) - set(active))})
        prior_images.extend(train_ids)
        stage_records.append({
            "index": index, "classes": classes, "active_classes": active,
            "unknown_classes": sorted(set(order) - set(active)),
            "train_image_ids": train_ids, "memory_image_ids": memory_ids,
            "memory_size": len(memory_ids),
        })
    manifest = {
        "schema_version": 1, "benchmark": protocol.upper(), "protocol": protocol,
        "order": order_mode, "seed": seed, "memory_fraction": memory_fraction,
        "grouping_source": grouping_source, "category_order_source_ids": order,
        "categories": train["categories"], "stages": stage_records,
        "integrity": {"category_count": len(order), "stage_count": len(stages)},
    }
    write_json(output_root / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("m-owodb", "s-owodb"))
    parser.add_argument("--order", default="random", choices=("id", "random"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--memory-fraction", default=0.10, type=float)
    parser.add_argument("--groups-json", default=None, type=Path)
    parser.add_argument("--allow-heuristic-groups", action="store_true",
                        help="allow a non-paper supercategory grouping for smoke tests")
    args = parser.parse_args()
    manifest = build(args.coco_root, args.output_root, args.protocol, args.order,
                     args.seed, args.memory_fraction, args.groups_json,
                     args.allow_heuristic_groups)
    print(json.dumps(manifest["integrity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
