"""Opt-in actual CUDA training/evaluation/checkpoint transition on synthetic COCO."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

from main import get_args_parser, main, load_local_checkpoint
from datasets import transforms as T
from tools.owod.verify_paper_checkpoint import verify


@pytest.mark.skipif(os.environ.get('TREE_BASELINE_INTEGRATION') != '1', reason='opt-in CUDA integration')
def test_actual_training_stage_transition_resume_and_ew_consolidation(tmp_path):
    coco = tmp_path / 'coco'
    for split in ('train2017', 'val2017'):
        (coco / split).mkdir(parents=True)
        for index in range(2):
            Image.new('RGB', (96, 96), (80 + index * 50, 100, 150)).save(coco / split / f'{index}.jpg')
    categories = [{'id': label, 'name': str(label)} for label in (2, 9, 77)]
    images = [{'id': i, 'file_name': f'{i}.jpg', 'height': 96, 'width': 96} for i in range(2)]
    annotations = [{'id': i * 3 + j, 'image_id': i, 'category_id': label, 'bbox': [10 + j * 20, 10, 15, 20],
                    'area': 300, 'iscrowd': 0} for i in range(2) for j, label in enumerate((2, 9, 77))]
    full = {'info': {}, 'licenses': [], 'images': images, 'annotations': annotations, 'categories': categories}
    val = tmp_path / 'val.json'
    val.write_text(json.dumps(full))
    train_paths = []
    for label in (2, 9):
        train = tmp_path / f'train_{label}.json'
        train.write_text(json.dumps({**full, 'annotations': [ann for ann in annotations if ann['category_id'] == label]}))
        train_paths.append(train)
    transforms = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    for method in ('ow-detr', 'ew-detr'):
        previous = None
        for stage in (0, 1):
            output = tmp_path / method / f'stage_{stage}'
            command = ['--paper-baseline', method, '--num_classes', '92', '--lr_backbone', '0',
                       '--owod-stage', str(stage), '--coco_path', str(coco), '--train-ann', str(train_paths[stage]),
                       '--val-ann', str(val), '--output_dir', str(output), '--epochs', '1', '--lr_drop', '1',
                       '--batch_size', '2', '--num_workers', '0', '--no-file-log', '--ow-pseudo-warmup', '0',
                       '--owod-current-class-ids', '2' if stage == 0 else '9',
                       '--ew-current-samples', '2', '--ew-previous-samples', str(stage * 2)]
            command += ['--owod-known-class-ids', '2'] + (['9'] if stage else [])
            if previous:
                command += ['--pretrained', str(previous), '--owod-previous-class-ids', '2']
            args = get_args_parser().parse_args(command)
            with mock.patch('datasets.coco.make_coco_transforms', return_value=transforms):
                main(args)
            verify(output, 1, method, stage)
            checkpoint = load_local_checkpoint(output / 'checkpoint.pth')
            assert checkpoint['epoch'] == 0
            if stage == 0:
                base_backbone = {key: tensor.clone() for key, tensor in checkpoint['model'].items()
                                 if key.startswith('detr.backbone.')}
                # An interruption after checkpoint/evaluation but before the marker
                # must recover with no extra optimizer step or duplicate merge.
                (output / 'training_complete.json').unlink()
                resumed_args = get_args_parser().parse_args(command + ['--resume', str(output / 'checkpoint.pth')])
                with mock.patch('datasets.coco.make_coco_transforms', return_value=transforms):
                    main(resumed_args)
                verify(output, 1, method, stage)
            else:
                for key, tensor in base_backbone.items():
                    torch.testing.assert_close(checkpoint['model'][key], tensor, rtol=0, atol=0)
            previous = output / ('checkpoint_consolidated.pth' if method == 'ew-detr' else 'checkpoint.pth')
            if method == 'ew-detr':
                merged = load_local_checkpoint(previous)
                assert merged['ew_consolidated']
                assert all(value.count_nonzero() == 0 for key, value in merged['model'].items() if key.endswith('.task_b'))
            del checkpoint
            torch.cuda.empty_cache()
