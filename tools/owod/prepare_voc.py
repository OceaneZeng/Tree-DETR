"""Preserve VOC releases and publish their exact M-OWODB image/XML component.

This does not manufacture the missing author COCO XML release or a full M
manifest. Original archives, annotations and author split identities are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import uuid
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod.benchmark_spec import official_split, specification
from tools.owod.manage_data import check_source, required_stems
from tools.owod.protocol import file_sha256

RELEASES = (
    ('voc2007-trainval', 'VOCtrainval_06-Nov-2007.tar', 'VOC2007', 'trainval', 5011),
    ('voc2007-test', 'VOCtest_06-Nov-2007.tar', 'VOC2007', 'test', 4952),
    ('voc2012-trainval', 'VOCtrainval_11-May-2012.tar', 'VOC2012', 'trainval', 11540),
)


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def extract_release(archive, directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive_hash = file_sha256(archive)
    manifest_path = directory / 'source_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['archive_sha256'] != archive_hash:
            raise ValueError(f'Another archive is already registered: {directory}')
    hashes = {}
    with tarfile.open(archive, 'r:') as source:
        members = source.getmembers()
        # Validate all entries before making any extraction writes.
        for member in members:
            relative = PurePosixPath(member.name.replace('\\', '/'))
            destination = (directory / Path(*relative.parts)).resolve()
            if (relative.is_absolute() or '..' in relative.parts
                    or not destination.is_relative_to(directory)
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f'Unsafe/nonregular TAR entry: {member.name}')
        for number, member in enumerate(members, 1):
            relative = Path(*PurePosixPath(member.name.replace('\\', '/')).parts)
            destination = directory / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.extracting')
            digest = hashlib.sha256()
            size = 0
            try:
                with source.extractfile(member) as incoming, temporary.open('wb') as output:
                    while block := incoming.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                        output.write(block)
                if size != member.size:
                    raise ValueError(f'Truncated TAR entry: {member.name}')
                expected = digest.hexdigest()
                if destination.exists():
                    if not destination.is_file() or file_sha256(destination) != expected:
                        raise FileExistsError(f'Refusing to replace different existing data: {destination}')
                else:
                    temporary.replace(destination)
                hashes[relative.as_posix()] = expected
            finally:
                temporary.unlink(missing_ok=True)
            if number % 5000 == 0:
                print(f'{archive.name}: extracted/verified {number}/{len(members)} entries', flush=True)
    record = {'archive': str(archive), 'archive_sha256': archive_hash,
              'archive_bytes': archive.stat().st_size, 'file_sha256': hashes}
    save(manifest_path, record)
    return record


def publish(source, destination, expected_hash):
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != expected_hash:
            raise FileExistsError(f'Refusing to replace different existing data: {destination}')
        return 'reused'
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)  # same-volume raw files remain byte-identical
        return 'hardlinked'
    except OSError:
        temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.copying')
        try:
            shutil.copyfile(source, temporary)
            if file_sha256(temporary) != expected_hash:
                raise ValueError(f'Copy verification failed: {source}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return 'copied'


def prepare(downloads, voc_root, ore_root):
    records, sources, ids_by_release = [], {}, {}
    for tag, filename, year, split, expected in RELEASES:
        archive = (downloads / filename).resolve()
        directory = voc_root / tag
        print(f'Preserving {tag}: {archive}', flush=True)
        record = extract_release(archive, directory)
        original = directory / 'VOCdevkit' / year
        image_set = original / 'ImageSets/Main' / (split + '.txt')
        ids = [line.split()[0] for line in image_set.read_text().splitlines() if line.strip()]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f'Unexpected original {year}/{split} image identities')
        ids_by_release[tag] = set(ids)
        raw_images = sorted((original / 'JPEGImages').glob('*.jpg'))
        raw_xml = sorted((original / 'Annotations').glob('*.xml'))
        if {p.stem for p in raw_images} != {p.stem for p in raw_xml}:
            raise ValueError(f'Image/XML coverage differs: {original}')
        for stem in ids:
            if stem in sources:
                raise ValueError(f'Original VOC image identity collision: {stem}')
            sources[stem] = (original, directory, record)
        records.append({'release': tag, 'archive': str(archive),
                        'archive_sha256': record['archive_sha256'], 'root': str(original),
                        'original_split_sha256': file_sha256(image_set),
                        'original_split_images': len(ids), 'all_archive_images': len(raw_images),
                        'all_archive_xml': len(raw_xml)})
    # Assert original VOC lists match the pinned ORE lists exactly.
    train = set(official_split('m-owodb', 't1_train'))
    if train != ids_by_release['voc2007-trainval'] | ids_by_release['voc2012-trainval']:
        raise ValueError('Original VOC trainval differs from the pinned ORE Task 1 list')
    test = set(official_split('m-owodb', specification('m-owodb')['test_split']))
    voc_test = {stem for stem in test if len(stem) == 6 and stem.isdigit()}
    if voc_test != ids_by_release['voc2007-test']:
        raise ValueError('Original VOC2007 test differs from the pinned ORE test list')
    selected = [stem for stem in required_stems('m-owodb')
                if (len(stem) == 6 and stem.isdigit()) or '_' in stem]
    actions = {'hardlinked': 0, 'copied': 0, 'reused': 0}
    for number, stem in enumerate(selected, 1):
        original, directory, record = sources[stem]
        image_path = original / 'JPEGImages' / (stem + '.jpg')
        xml_path = original / 'Annotations' / (stem + '.xml')
        annotation = ET.parse(xml_path).getroot()
        with Image.open(image_path) as image:
            if image.format != 'JPEG' or image.size != (
                    int(annotation.findtext('size/width')), int(annotation.findtext('size/height'))):
                raise ValueError(f'Image/XML dimensions differ: {stem}')
            image.verify()
        for folder, path in [('JPEGImages', image_path), ('Annotations', xml_path)]:
            relative = path.relative_to(directory).as_posix()
            actions[publish(path, ore_root / folder / path.name, record['file_sha256'][relative])] += 1
        if number % 5000 == 0:
            print(f'Published and verified VOC pairs: {number}/{len(selected)}', flush=True)
    result = {'status': 'voc_component_ready', 'voc_root': str(voc_root), 'ore_root': str(ore_root),
              'releases': records, 'official_voc_image_xml_pairs': len(selected),
              'jpeg_headers_and_xml_dimensions_verified': True, 'publish_actions': actions,
              'm_benchmark_source_check': check_source('m-owodb', ore_root),
              'full_m_manifest_created': False,
              'remaining': 'COCO images and the corresponding original author VOC-style XML component'}
    save(ore_root / 'voc_source_manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--voc-root', type=Path, default=ROOT / 'data/raw/voc')
    parser.add_argument('--ore-root', type=Path, default=ROOT / 'data/raw/ore')
    args = parser.parse_args(argv)
    result = prepare(args.downloads.resolve(), args.voc_root.resolve(), args.ore_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
