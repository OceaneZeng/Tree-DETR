import copy

import pytest
import torch
from pycocotools.coco import COCO

from datasets.coco_eval import CocoEvaluator


@pytest.mark.parametrize('info', [None, {'description': 'original metadata'}])
def test_evaluate_detections_preserves_source_with_optional_info(info):
    gt = COCO()
    gt.dataset = {
        'images': [{'id': 1, 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'object'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}],
    }
    if info is not None:
        gt.dataset['info'] = info
    gt.createIndex()
    original = copy.deepcopy(gt.dataset)
    evaluator = CocoEvaluator(gt, ['bbox'])
    evaluator.update({1: {'boxes': torch.tensor([[10., 10., 30., 30.]]),
                          'scores': torch.tensor([0.9]), 'labels': torch.tensor([1])}})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    assert gt.dataset == original
    assert evaluator.coco_gt.dataset['info'] == (info if info is not None else {})
    assert evaluator.coco_eval['bbox'].eval['recall'][0, 0, 0, -1] == 1.0
