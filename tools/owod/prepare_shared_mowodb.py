#!/usr/bin/env python3
"""Build the shared M-OWODB VOC view consumed by PROB and OWOBJ.

The official Tree-DETR protocol stores COCO JSON files. PROB/OWOBJ consume
VOC XML plus ImageSets files. This converter keeps the source annotations and
image selection unchanged, and only changes the serialization format.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from protocol import stage_files as validated_stage_files
except ModuleNotFoundError:  # Importing this utility as tools.owod.prepare_shared_mowodb.
    from tools.owod.protocol import stage_files as validated_stage_files


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_OUTPUT = ROOT / "data" / "derived" / "m-owodb-voc"
DEFAULT_BASELINE_ROOT = ROOT / "baselines"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_names(payload: dict) -> list[str]:
    return [Path(str(image["file_name"])).stem for image in payload.get("images", [])]


def write_xml(path: Path, image: dict, annotations: list[dict], categories: dict[int, str]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = str(image["file_name"])
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image["width"])
    ET.SubElement(size, "height").text = str(image["height"])
    ET.SubElement(size, "depth").text = "3"
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = categories[int(ann["category_id"])]
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(max(1, int(round(x)) + 1))
        ET.SubElement(box, "ymin").text = str(max(1, int(round(y)) + 1))
        ET.SubElement(box, "xmax").text = str(max(1, int(round(x + w)) + 1))
        ET.SubElement(box, "ymax").text = str(max(1, int(round(y + h)) + 1))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def find_image(source_roots: list[Path], file_name: str) -> Path | None:
    value = Path(file_name)
    candidates = []
    for root in source_roots:
        candidates.extend((root / value, root / value.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def materialize_images(output: Path, images: list[dict], source_roots: list[Path],
                       mode: str) -> None:
    if mode == "none":
        return
    image_dir = output / "JPEGImages"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        file_name = str(image["file_name"])
        source = find_image(source_roots, file_name)
        if source is None:
            raise FileNotFoundError(
                f"Cannot find image {file_name!r} under {[str(root) for root in source_roots]}")
        # Both external loaders append `.jpg` to split names, so normalize the
        # materialized filename even when the source extension is `.jpeg`.
        target = image_dir / f"{Path(file_name).stem}.jpg"
        if target.exists():
            continue
        if mode == "symlink":
            try:
                target.symlink_to(source)
                continue
            except OSError as exc:
                raise OSError(
                    f"Cannot create image symlink {target}; use --image-mode copy on Windows") from exc
        shutil.copy2(source, target)


def prepare_one(output: Path, train: dict, val: dict, stage_payloads: list[dict[str, dict]],
                clean: bool, image_roots: list[Path], image_mode: str) -> None:
    if clean and output.exists():
        for child in (output / "Annotations", output / "ImageSets"):
            if child.exists():
                shutil.rmtree(child)
    annotations_dir = output / "Annotations"
    imagesets_dir = output / "ImageSets"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("TOWOD", "OWDETR"):
        (imagesets_dir / dataset).mkdir(parents=True, exist_ok=True)

    categories = {int(c["id"]): str(c["name"]) for c in train.get("categories", [])}
    categories.update({int(c["id"]): str(c["name"]) for c in val.get("categories", [])})
    train_images = {int(x["id"]): x for x in train.get("images", [])}
    val_images = {int(x["id"]): x for x in val.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in [*train.get("annotations", []), *val.get("annotations", [])]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    selected: set[int] = set()
    for stage in stage_payloads:
        for key in ("train", "full_val"):
            selected.update(int(x["id"]) for x in stage[key].get("images", []))
    all_images = {**val_images, **train_images}
    for image_id in sorted(selected):
        image = all_images[image_id]
        write_xml(annotations_dir / f"{Path(image['file_name']).stem}.xml",
                  image, anns_by_image.get(image_id, []), categories)
    materialize_images(output, [all_images[image_id] for image_id in sorted(selected)],
                       image_roots, image_mode)

    def write_split(name: str, values: list[str], dataset: str) -> None:
        (imagesets_dir / dataset / f"{name}.txt").write_text(
            "\n".join(values) + "\n", encoding="utf-8")

    for index, stage in enumerate(stage_payloads, start=1):
        train_names = image_names(stage["train"])
        test_names = image_names(stage["full_val"])
        # Both naming conventions are emitted in both folders. The shared
        # recipes force dataset=TOWOD, while the aliases keep the authors'
        # original train_set/test_set arguments unchanged.
        for dataset in ("TOWOD", "OWDETR"):
            write_split(f"owod_t{index}_train", train_names, dataset)
            write_split(f"owod_t{index}_ft", train_names, dataset)
            write_split("owod_all_task_test", test_names, dataset)
            write_split(f"t{index}_train", train_names, dataset)
            write_split(f"t{index}_ft", train_names, dataset)
            write_split("test", test_names, dataset)


def write_shared_configs(repo: Path, method: str) -> list[Path]:
    """Create non-destructive 20/20/20/20 variants beside each checkout."""
    configs = repo / "configs"
    generated = []
    for name in ("M_OWOD_BENCHMARK.sh", "EVAL_M_OWOD_BENCHMARK.sh"):
        source = configs / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing baseline config: {source}")
        content = source.read_text(encoding="utf-8")
        original_output = {
            "prob": "EXP_DIR=exps/MOWODB/PROB",
            "owobj": "EXP_DIR=exps/MOWODB/OWOBJ",
        }[method]
        content = content.replace(
            original_output,
            f'EXP_DIR="${{MOWODB_OUTPUT_ROOT:?Set MOWODB_OUTPUT_ROOT}}/{method}"')
        content = content.replace("--dataset OWDETR", "--dataset TOWOD")
        # OWOBJ's published recipe is 19/21/20/20; M-OWODB comparison is 20
        # classes per increment. Only the argument values are changed.
        content = content.replace("--PREV_INTRODUCED_CLS 19", "--PREV_INTRODUCED_CLS 20")
        content = content.replace("--CUR_INTRODUCED_CLS 19", "--CUR_INTRODUCED_CLS 20")
        content = content.replace("--CUR_INTRODUCED_CLS 21", "--CUR_INTRODUCED_CLS 20")
        content = content.replace("--data_root ./data/OWDETR/coco", "--data_root ./data/OWOD")
        # OWOBJ's published t2 line contains an escaped duplicate --lr token;
        # remove that shell typo in the shared runnable copy.
        content = content.replace("--lr 2e-5\\--lr 2e-5", "--lr 2e-5")
        content = content.replace(
            "PY_ARGS=${@:1}",
            'PY_ARGS="${@:1} --dataset TOWOD '
            '--data_root ${MOWODB_DATA_ROOT:?Set MOWODB_DATA_ROOT}"')
        target = configs / name.replace(".sh", "_SHARED.sh")
        target.write_text(content, encoding="utf-8")
        generated.append(target)
    return generated

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-coco", type=Path, required=True,
                        help="Full instances_train2017.json, not a stage subset")
    parser.add_argument("--val-coco", type=Path, required=True,
                        help="Full instances_val2017.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_OUTPUT,
                        help="Single shared VOC view used by every external baseline")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT,
                        help="Directory containing prob/ and owobj/ checkouts")
    parser.add_argument("--image-root", type=Path, action="append", default=[],
                        help="COCO image directory; repeat for train2017 and val2017")
    parser.add_argument("--image-mode", choices=("none", "copy", "symlink"), default="none",
                        help="Materialize JPEGImages (default: leave existing image store untouched)")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Use a local/pilot manifest and mark the resulting run as non-paper-comparable",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    if manifest.get("protocol") != "m-owodb":
        raise ValueError("The shared adapter only accepts protocol=m-owodb")
    if len(manifest.get("stages", [])) != 4:
        raise ValueError("M-OWODB requires exactly four stages")
    if not manifest.get("official_annotations") and not args.allow_unverified:
        raise ValueError(
            "Refusing an unverified/pilot manifest; pass --allow-unverified for a local consistency run")
    train, val = read(args.train_coco), read(args.val_coco)
    payloads = []
    for index in range(4):
        _, files = validated_stage_files(
            manifest_path, index, allow_unverified=args.allow_unverified)
        payloads.append({key: read(path) for key, path in files.items()})
    image_roots = [path.resolve() for path in args.image_root]
    if args.image_mode != "none" and not image_roots:
        raise ValueError("--image-root is required when --image-mode is copy or symlink")
    output = args.output.resolve()
    prepare_one(output, train, val, payloads, args.clean, image_roots, args.image_mode)
    for method in ("prob", "owobj"):
        generated = write_shared_configs(args.baseline_root / method, method)
        for path in generated:
            print(f"generated {path}")
    print(f"prepared shared data {output}")
    if args.allow_unverified:
        print("WARNING: using an unverified local manifest; results are not paper-comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
