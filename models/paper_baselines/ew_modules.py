"""EW-DETR equations (2)-(11) and supplementary Appendix A.

Unspecified choices are recorded in docs/paper-baselines.md, not attributed to
the authors. Parametrizations also cover functional MultiheadAttention weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


class DualLoRA(nn.Module):
    def __init__(self, weight, rank=16):
        super().__init__()
        rank = min(rank, *weight.shape)
        self.task_a = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.task_b = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.task_a, a=math.sqrt(5))
        self.register_buffer('aggregate_a', torch.zeros_like(self.task_a))
        self.register_buffer('aggregate_b', torch.zeros_like(self.task_b))

    def forward(self, original):
        return original + self.aggregate_b @ self.aggregate_a + self.task_b @ self.task_a

    @torch.no_grad()
    def consolidate(self, beta):
        update = ((1 - beta) * (self.aggregate_b @ self.aggregate_a)
                  + beta * (self.task_b @ self.task_a)).float()
        options = {'driver': 'gesvd'} if update.is_cuda else {}
        u, s, vh = torch.linalg.svd(update, full_matrices=False, **options)
        rank = self.task_a.shape[0]
        self.aggregate_b.copy_(u[:, :rank] * s[:rank])
        self.aggregate_a.copy_(vh[:rank])
        # A must remain nonzero: zeroing both factors would kill their gradients.
        nn.init.kaiming_uniform_(self.task_a, a=math.sqrt(5))
        self.task_b.zero_()


def inject_dual_lora(detector, rank):
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    names = []
    for prefix in ('encoder', 'decoder'):
        block = getattr(detector.transformer, prefix)
        for name, module in list(block.named_modules()):
            attributes = ['weight'] if isinstance(module, nn.Linear) else []
            if isinstance(module, nn.MultiheadAttention):
                attributes += ['in_proj_weight']
            for attribute in attributes:
                weight = getattr(module, attribute)
                if weight is None:
                    raise ValueError('Separate Q/K/V attention weights are not supported')
                parametrize.register_parametrization(module, attribute, DualLoRA(weight, rank))
                names.append(f'transformer.{prefix}.{name}.{attribute}')
    for head in (detector.class_embed, detector.bbox_embed):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    return names


def merge_beta(current_samples, previous_samples, minimum=0.2, maximum=0.8):
    if current_samples <= 0 or previous_samples < 0 or not 0 <= minimum <= maximum <= 1:
        raise ValueError('Invalid sample counts or merging bounds')
    if previous_samples == 0:
        return 1.0
    return max(minimum, min(maximum, maximum - (maximum - minimum)
                            * current_samples / previous_samples))


class QueryNormEUMix(nn.Module):
    def __init__(self, hidden_dim, known_ids, unknown_id=91):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.objectness = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.theta_mix = nn.Parameter(torch.tensor(0.0))
        self.theta_alpha = nn.Parameter(torch.tensor(math.log(9.0)))
        self.theta_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.theta_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.objectness_bias = nn.Parameter(torch.tensor(0.0))
        self.known_ids = list(known_ids)
        self.unknown_id = unknown_id
        self.temperature = 1.0

    def forward(self, features, classifier):
        normalized = F.normalize(self.norm(features), dim=-1, eps=1e-6)
        mixing = self.theta_mix.sigmoid()
        logits = classifier((1 - mixing) * features + mixing * normalized)
        objectness = self.objectness(features.norm(dim=-1, keepdim=True)).squeeze(-1)
        objectness = objectness / (self.temperature + 1e-6)
        known = logits[..., self.known_ids]
        gap = (1 - known.sigmoid().amax(-1)).clamp(min=1e-6).pow(F.softplus(self.theta_gamma))
        unknown_objectness = objectness.sigmoid() * gap
        unknown_classifier = (logits[..., self.unknown_id] + self.objectness_bias).sigmoid()
        alpha = self.theta_alpha.sigmoid()
        unknown = alpha * unknown_classifier + (1 - alpha) * unknown_objectness
        calibrated = torch.full_like(logits, -1e8)
        calibrated[..., self.known_ids] = known - F.softplus(self.theta_lambda) * unknown_objectness[..., None]
        calibrated[..., self.unknown_id] = torch.logit(unknown.clamp(1e-6, 1 - 1e-6))
        return calibrated
