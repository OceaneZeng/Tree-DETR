"""Protect teacher state and optimizer semantics in the runtime fast paths."""
import copy

import pytest
import torch
from torch import nn

import engine
from models.graph_local.discovery import make_ema
from models.graph_local.routed_adapter import RiskRoutedStage, mask_inactive_adapter_gradients
from util.misc import NestedTensor


class TinyDiscovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer('observations', torch.zeros((), dtype=torch.long))
        self.calls = 0

    def forward(self, samples):
        self.calls += 1
        return {'prediction': self.weight * samples.tensors.mean()}

    def observe_training(self, evidence):
        assert evidence == 'current-batch'
        self.observations.add_(1)


class TinyCriterion(nn.Module):
    weight_dict = {'loss_ce': 1.}
    warmup = 2

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch
        self.evidence = None
        self.had_teacher = None

    def forward(self, outputs, targets):
        self.had_teacher = 'discovery_teacher' in outputs
        self.evidence = 'current-batch'
        return {'loss_ce': outputs['prediction'].square().sum(),
                'class_error': outputs['prediction'].detach().sum() * 0}


class FixedPrefetcher:
    def __init__(self, loader, *args, **kwargs):
        self.loader = iter(loader)

    def next(self):
        return next(self.loader, (None, None))


@pytest.mark.parametrize('epoch,teacher_calls', [(0, 0), (1, 0), (2, 1), (3, 1)])
def test_warmup_skips_only_forward_and_keeps_ema_and_memory(monkeypatch, epoch, teacher_calls):
    monkeypatch.setattr(engine, 'data_prefetcher', FixedPrefetcher)
    model = TinyDiscovery()
    teacher = make_ema(model)
    criterion = TinyCriterion(epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    batch = (NestedTensor(torch.ones(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.bool)),
             [{'image_id': torch.tensor([1])}])
    engine.train_one_epoch(model, criterion, [batch], optimizer, torch.device('cpu'),
                           epoch, discovery_teacher=teacher)
    assert teacher.calls == teacher_calls
    assert criterion.had_teacher == bool(teacher_calls)
    assert model.calls == 1
    assert model.observations.item() == teacher.observations.item() == 1
    assert teacher.ema_update_count.item() == 1
    # First EMA update has decay=0 and must copy the updated student.
    torch.testing.assert_close(teacher.weight, model.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.weight, torch.tensor([.8]))


@pytest.mark.parametrize('activity', [[False, False, False], [True, True, True], [True, False, True]])
def test_inactive_factors_skip_existing_adamw_momentum_and_decay(activity):
    torch.manual_seed(8)
    stage = RiskRoutedStage(nn.Linear(4, 5), dict(
        class_ids=[1, 2, 3], old_ids=[0], rank=2, risk_beta=2.,
        risk=[[.2], [.3], [.4]], routing='hard'))
    optimizer = torch.optim.AdamW(stage.parameters(), lr=.01, weight_decay=.1)
    for parameter in stage.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # Populate momentum before any branch is inactive.
    parameters_before = copy.deepcopy(stage.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    reference = copy.deepcopy(stage)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.1)
    reference_optimizer.load_state_dict(optimizer_before)
    for model in (stage, reference):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    stage.last_gates = torch.tensor(activity, dtype=torch.float32).reshape(1, 1, -1)
    mask_inactive_adapter_gradients(stage)
    for enabled, branch in zip(activity, reference.adapters):
        if not enabled:
            for parameter in branch.parameters():
                parameter.grad = None
    assert [p.grad is None for p in stage.parameters()] == [p.grad is None for p in reference.parameters()]
    optimizer.step()
    reference_optimizer.step()
    for name, parameter in stage.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)
    for index, enabled in enumerate(activity):
        if not enabled:
            for factor in ('down', 'up'):
                key = f'adapters.{index}.{factor}.weight'
                torch.testing.assert_close(stage.state_dict()[key], parameters_before[key], rtol=0, atol=0)


def test_global_activity_is_reduced_before_host_read(monkeypatch):
    stage = RiskRoutedStage(nn.Linear(4, 5), dict(
        class_ids=[1, 2, 3], old_ids=[0], rank=2, risk_beta=2.,
        risk=[[.2], [.3], [.4]], routing='hard'))
    stage.last_gates = torch.tensor([1., 0., 0.]).reshape(1, 1, -1)
    for parameter in stage.parameters():
        parameter.grad = torch.ones_like(parameter)
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    def reduce(flags, op):
        assert op == torch.distributed.ReduceOp.MAX
        flags.copy_(torch.tensor([1, 1, 0], dtype=flags.dtype))
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    mask_inactive_adapter_gradients(stage)
    for enabled, branch in zip([True, True, False], stage.adapters):
        assert all((parameter.grad is not None) == enabled for parameter in branch.parameters())
