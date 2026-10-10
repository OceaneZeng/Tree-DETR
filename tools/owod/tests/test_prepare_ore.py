import json
import xml.etree.ElementTree as ET

import pytest

from tools.owod import prepare_ore as module


def test_author_coordinates_crowd_and_no_clipping():
    image = dict(file_name='000000000001.jpg', width=100, height=80)
    anns = [dict(category_id=1, bbox=[.5, 2.9, 10.7, 15.2], iscrowd=1),
            dict(category_id=1, bbox=[-1.5, 0, 103, 80], iscrowd=0)]
    root = ET.fromstring(module.annotation_bytes(image, anns, {1: 'person'}))
    objects = root.findall('object')
    assert len(objects) == 2
    assert [o.findtext('difficult') for o in objects] == ['0', '0']
    assert [[int(e.text) for e in o.find('bndbox')] for o in objects] == [
        [1, 3, 12, 19], [0, 1, 102, 81]]


def test_prepare_reuses_data_and_rejects_changed_inputs(tmp_path, monkeypatch):
    coco, ore = tmp_path / 'coco', tmp_path / 'ore'
    monkeypatch.setattr(module, 'required_stems', lambda _: ['000000000001'])
    monkeypatch.setattr(module, 'check_source', lambda *_: {'ready': True})
    (coco / 'annotations').mkdir(parents=True)
    (coco / 'train2017').mkdir()
    (coco / 'train2017/000000000001.jpg').write_bytes(b'image fixture')
    payload = dict(images=[dict(id=1, file_name='000000000001.jpg', width=100, height=80)],
                   annotations=[dict(id=1, image_id=1, category_id=1, bbox=[1, 2, 3, 4])],
                   categories=[dict(id=1, name='person')])
    path = coco / 'annotations/instances_train2017.json'
    path.write_text(json.dumps(payload))
    (coco / 'annotations/instances_val2017.json').write_text(
        json.dumps(dict(images=[], annotations=[], categories=payload['categories'])))
    assert module.prepare(coco, ore)['coco_pairs'] == 1
    assert module.prepare(coco, ore)['source_check']['ready']
    xml = ore / 'Annotations/000000000001.xml'
    xml.write_bytes(b'different XML')
    with pytest.raises(FileExistsError, match='different XML'):
        module.prepare(coco, ore)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='provenance changed'):
        module.prepare(coco, ore)


def test_missing_voc_stops_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'required_stems', lambda _: ['000001'])
    with pytest.raises(ValueError, match='prepare_voc.py'):
        module.prepare(tmp_path / 'coco', tmp_path / 'ore')
