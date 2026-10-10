"""Reconstruct the COCO component of ORE from its published conversion rules.

Run prepare_voc.py first. This preserves pinned ImageSets and existing VOC XML.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from tools.owod.manage_data import check_source, required_stems
from tools.owod.prepare_voc import publish, save
from tools.owod.protocol import file_sha256

AUTHOR_COMMIT = '23890f188cd1a6801c6ac0e3dacd78b8572b8c29'
AUTHOR_CONVERTER = ('https://github.com/JosephKJ/OWOD/blob/' + AUTHOR_COMMIT
                    + '/datasets/coco_utils/coco_annotation_to_voc_style.py')


def annotation_bytes(image, annotations, categories):
    """Match ORE's XML fields, annotation order, integer conversion and crowd policy."""
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = image['file_name']
    size = ET.SubElement(root, 'size')
    for name, value in [('width', image['width']), ('height', image['height']), ('depth', 3)]:
        ET.SubElement(size, name).text = str(value)
    for ann in annotations:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = categories[ann['category_id']]
        ET.SubElement(obj, 'difficult').text = '0'
        box = ET.SubElement(obj, 'bndbox')
        x, y, w, h = ann['bbox']
        for name, value in zip(('xmin', 'ymin', 'xmax', 'ymax'),
                               (x + 1.0, y + 1.0, x + w + 1.0, y + h + 1.0)):
            ET.SubElement(box, name).text = str(int(value))
    return ET.tostring(root, encoding='us-ascii')


def prepare(coco_root, ore_root):
    coco_root, ore_root = Path(coco_root).resolve(), Path(ore_root).resolve()
    ore_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ore_root / '.prepare_coco.lock'), timeout=1):
        return _prepare(coco_root, ore_root)


def _prepare(coco_root, ore_root):
    required = set(required_stems('m-owodb'))
    coco_ids = {stem for stem in required if len(stem) == 12 and stem.isdigit()}
    for stem in required - coco_ids:
        for folder, suffix in [('Annotations', '.xml'), ('JPEGImages', '.jpg')]:
            if not (ore_root / folder / (stem + suffix)).is_file():
                raise ValueError('Run prepare_voc.py first; missing VOC input: ' + stem)
    inputs = {split: coco_root / 'annotations' / ('instances_' + split + '.json')
              for split in ('train2017', 'val2017')}
    hashes = {split: file_sha256(path) for split, path in inputs.items()}
    provenance = {'author_commit': AUTHOR_COMMIT, 'author_converter': AUTHOR_CONVERTER,
                  'implementation_sha256': file_sha256(Path(__file__)),
                  'input_json_sha256': hashes, 'coco_root': str(coco_root)}
    record_path = ore_root / 'coco_source_manifest.json'
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding='utf-8'))
        if previous['provenance'] != provenance:
            raise ValueError('COCO provenance changed; use a fresh ore-root')
    images, annotations = {}, defaultdict(list)
    for split, path in inputs.items():
        payload = json.loads(path.read_text(encoding='utf-8'))
        categories = {c['id']: c['name'] for c in payload['categories']}
        for image in payload['images']:
            stem = Path(image['file_name']).stem
            if stem not in coco_ids:
                continue
            if stem in images:
                raise ValueError('Duplicate COCO image: ' + stem)
            source = coco_root / split / image['file_name']
            if not source.is_relative_to(coco_root) or not source.is_file():
                raise ValueError('Missing or invalid COCO image: ' + str(source))
            images[stem] = (image, source, categories)
        for ann in payload['annotations']:
            stem = f"{ann['image_id']:012d}"
            if stem in coco_ids:
                annotations[stem].append(ann)
        del payload
    if set(images) != coco_ids or any(not annotations[stem] for stem in coco_ids):
        raise ValueError('Pinned M ImageSets require missing COCO images/annotations')
    output_hashes = {}
    for index, stem in enumerate(sorted(coco_ids), 1):
        image, source, categories = images[stem]
        xml = annotation_bytes(image, annotations[stem], categories)
        destination = ore_root / 'Annotations' / (stem + '.xml')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != xml:
                raise FileExistsError('Refusing to replace different XML: ' + str(destination))
        else:
            temporary = destination.with_suffix('.xml.preparing')
            temporary.write_bytes(xml)
            temporary.replace(destination)
        image_hash = file_sha256(source)
        publish(source, ore_root / 'JPEGImages' / (stem + '.jpg'), image_hash)
        output_hashes[stem] = {'xml_sha256': hashlib.sha256(xml).hexdigest(),
                               'image_sha256': image_hash}
        if index % 5000 == 0:
            print(f'ORE COCO pairs prepared: {index}/{len(coco_ids)}', flush=True)
    if any(file_sha256(inputs[split]) != value for split, value in hashes.items()):
        raise ValueError('Input JSON changed during preparation')
    result = {'status': 'author_code_reconstruction', 'provenance': provenance,
              'coco_pairs': len(coco_ids), 'files': output_hashes,
              'source_check': check_source('m-owodb', ore_root)}
    if not result['source_check']['ready']:
        raise ValueError('ORE source coverage check failed')
    save(record_path, result)
    return {k: v for k, v in result.items() if k != 'files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, default=ROOT / 'data/raw/coco')
    parser.add_argument('--ore-root', type=Path, default=ROOT / 'data/raw/ore')
    args = parser.parse_args()
    print(json.dumps(prepare(args.coco_root, args.ore_root), indent=2))


if __name__ == '__main__':
    main()
