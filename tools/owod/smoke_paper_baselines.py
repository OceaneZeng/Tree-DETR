"""Small real CUDA forward/backward checks; no COCO data or training outputs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from main import get_args_parser
from models.paper_baselines import build
from models.paper_baselines.ew_modules import DualLoRA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    options = parser.parse_args()
    for method in ('ow-detr', 'cat', 'ew-detr'):
        torch.manual_seed(42)
        arguments = [
            '--paper-baseline', method, '--num_classes', '92', '--lr_backbone', '0',
            '--owod-known-class-ids', '2', '9', '--device', options.device]
        if method == 'cat':
            arguments.extend(['--cat-proposals', 'synthetic-smoke-proposals.json'])
        args = get_args_parser().parse_args(arguments)
        model, criterion, post = build(args)
        model.to(options.device).train()
        criterion.to(options.device)
        criterion.epoch = 9
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        samples = torch.randn(2, 3, 128, 128, device=options.device)
        targets = [{'labels': torch.tensor([2], device=options.device),
                    'boxes': torch.tensor([[.5, .5, .3, .3]], device=options.device),
                    **({'proposal_boxes': torch.tensor(
                        [[.2, .2, .2, .2], [.7, .7, .2, .2]], device=options.device)}
                       if method == 'cat' else {})} for _ in range(2)]
        for _ in range(2):
            optimizer.zero_grad()
            output = model(samples)
            losses = criterion(output, targets)
            total = sum(value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict)
            assert torch.isfinite(total)
            total.backward()
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            assert not missing, missing
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                assert torch.equal(parameter, before[name]), name
        model.eval()
        with torch.no_grad():
            if method == 'ew-detr':
                model.consolidate(100, 0)
                assert all(module.task_b.count_nonzero() == 0 for module in model.modules() if isinstance(module, DualLoRA))
            results = post['bbox'](model(samples), torch.tensor([[128, 128], [128, 128]], device=options.device))
        assert all(torch.isfinite(result['scores']).all() for result in results)
        print(f'{method}: CUDA forward/backward, frozen parameters, gradients, postprocess passed; '
              f'trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}', flush=True)
        del model, optimizer, before, output, losses, total
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
