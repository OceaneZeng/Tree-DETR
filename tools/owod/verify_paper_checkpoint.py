"""Validate trusted local training artifacts before releasing a dependent run."""

import argparse
import json
from pathlib import Path

import torch


def verify(directory, epochs, method, stage):
    marker = json.loads((directory / 'training_complete.json').read_text())
    if marker.get('last_epoch') != epochs - 1 or marker.get('epochs') != epochs:
        raise ValueError('Completion marker does not match the requested schedule')
    names = ['checkpoint.pth']
    if method == 'ew-detr':
        names.append('checkpoint_consolidated.pth')
    for name in names:
        safe_globals = getattr(torch.serialization, 'safe_globals', None)
        if safe_globals is None:
            checkpoint = torch.load(directory / name, map_location='cpu', weights_only=False)
        else:
            with safe_globals([argparse.Namespace]):
                checkpoint = torch.load(directory / name, map_location='cpu', weights_only=True)
        config = checkpoint.get('args')
        if checkpoint.get('epoch') != epochs - 1 or getattr(config, 'owod_stage', None) != stage:
            raise ValueError(f'Wrong checkpoint epoch/stage: {directory / name}')
        if getattr(config, 'lr_backbone', None) != 0:
            raise ValueError('Expected fully frozen backbone')
        if method == 'd4':
            if getattr(config, 'neighbor_scoped_lora', False) or getattr(config, 'paper_baseline', None):
                raise ValueError('Wait target is not D4')
        elif getattr(config, 'paper_baseline', None) != method:
            raise ValueError('Checkpoint method mismatch')
        if name == 'checkpoint_consolidated.pth' and not checkpoint.get('ew_consolidated'):
            raise ValueError('EW aggregate checkpoint has not been consolidated')
        tensors = checkpoint.get('model', {})
        if not tensors or not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError('Checkpoint has missing/nonfinite model tensors')
    print(f'Validated {method} stage {stage}, epoch {epochs}: {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--method', choices=('d4', 'ow-detr', 'cat', 'ew-detr'), required=True)
    parser.add_argument('--stage', type=int, required=True)
    args = parser.parse_args()
    verify(args.directory, args.epochs, args.method, args.stage)
