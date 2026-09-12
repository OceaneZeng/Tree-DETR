"""From-paper CAT pseudo-labelling components.

The implementation follows CAT equations (3)-(8) and Algorithm 1.  The paper
does not publish the controller window sizes, update period, or momentum
amplitudes; those values are therefore explicit command-line parameters and
are stored in every run configuration.
"""

from __future__ import annotations

import torch
from torch import nn

from util import box_ops


def _linear_average(values: torch.Tensor) -> torch.Tensor:
    """Weighted average with linearly decreasing weight from new to old."""
    if values.numel() == 0:
        raise ValueError("Cannot average an empty loss window")
    weights = torch.arange(values.numel(), 0, -1, device=values.device,
                           dtype=values.dtype)
    return (values * (weights / weights.sum())).sum()


class AdaptivePseudoLabeler(nn.Module):
    """Stateful implementation of CAT's Measurer, Sensor, and Adjuster."""

    def __init__(self, memory_size=100, recent_size=20, start_iteration=100,
                 update_interval=100, positive_momentum=0.01,
                 negative_momentum=0.01):
        super().__init__()
        if not 1 <= recent_size < memory_size:
            raise ValueError("CAT requires 1 <= recent_size < memory_size")
        if start_iteration < memory_size or update_interval <= 0:
            raise ValueError("CAT start must fill the loss memory; interval must be positive")
        if positive_momentum < 0 or negative_momentum < 0:
            raise ValueError("CAT momentum amplitudes must be nonnegative")
        self.memory_size = int(memory_size)
        self.recent_size = int(recent_size)
        self.start_iteration = int(start_iteration)
        self.update_interval = int(update_interval)
        self.positive_momentum = float(positive_momentum)
        self.negative_momentum = float(negative_momentum)
        self.register_buffer("model_weight", torch.tensor(0.8))
        self.register_buffer("input_weight", torch.tensor(0.2))
        self.register_buffer("loss_memory", torch.zeros(memory_size))
        self.register_buffer("loss_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("iteration", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe(self, loss: torch.Tensor) -> None:
        """Record one loss and apply CAT equations (5)-(8) when scheduled."""
        value = loss.detach().float().reshape(())
        index = int(self.iteration.item()) % self.memory_size
        self.loss_memory[index] = value
        self.iteration.add_(1)
        self.loss_count.copy_(torch.minimum(
            self.loss_count + 1, self.loss_count.new_tensor(self.memory_size)))
        step = int(self.iteration.item())
        if (step <= self.start_iteration or step % self.update_interval
                or int(self.loss_count.item()) < self.memory_size):
            return

        # Reconstruct newest-to-oldest order from the circular buffer.
        newest = (step - 1) % self.memory_size
        order = (newest - torch.arange(self.memory_size,
                                       device=self.loss_memory.device)) % self.memory_size
        history = self.loss_memory[order]
        recent = _linear_average(history[:self.recent_size])
        older = _linear_average(history[self.recent_size:])
        trend = recent / older.clamp_min(torch.finfo(history.dtype).eps)
        if trend > 1:
            delta = self.negative_momentum * torch.sigmoid(trend - 1)
        else:
            delta = -self.positive_momentum * trend
        model = (self.model_weight * (1 + delta)).clamp_min(0)
        input_driven = (self.input_weight * (1 - delta)).clamp_min(0)
        normalizer = (model + input_driven).clamp_min(torch.finfo(model.dtype).eps)
        self.model_weight.copy_(model / normalizer)
        self.input_weight.copy_(input_driven / normalizer)

    def extra_repr(self) -> str:
        return (f"memory_size={self.memory_size}, recent_size={self.recent_size}, "
                f"start_iteration={self.start_iteration}, "
                f"update_interval={self.update_interval}")


@torch.no_grad()
def cat_pseudo_queries(attention_feature, boxes, matched, proposal_boxes,
                       padded_size, model_weight, input_weight, top_k=5):
    """Select CAT pseudo unknowns using equation (3).

    ``boxes`` and ``proposal_boxes`` are normalized cxcywh boxes in the same
    augmented image coordinate system.  Attention box means provide the
    model-driven score; maximum proposal IoU provides the input-driven prior.
    """
    height, width = padded_size
    attention = torch.nn.functional.interpolate(
        attention_feature.mean(1, keepdim=True), size=(height, width),
        mode="bilinear", align_corners=False)[:, 0]
    integral = torch.nn.functional.pad(attention.cumsum(1).cumsum(2), (1, 0, 1, 0))
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    coords = (xyxy * boxes.new_tensor([width, height, width, height])).long()
    coords[..., 0::2].clamp_(0, width)
    coords[..., 1::2].clamp_(0, height)
    selected = []
    for batch, ((src, _), proposals) in enumerate(zip(matched, proposal_boxes)):
        x1, y1, x2, y2 = coords[batch].unbind(-1)
        area = (x2 - x1) * (y2 - y1)
        sums = (integral[batch, y2, x2] - integral[batch, y1, x2]
                - integral[batch, y2, x1] + integral[batch, y1, x1])
        valid = area > 0
        valid[src.to(valid.device)] = False
        candidates = torch.where(valid)[0]
        if candidates.numel() == 0 or proposals.numel() == 0:
            selected.append(candidates[:0])
            continue
        model_score = sums[candidates] / area[candidates]
        low, high = model_score.min(), model_score.max()
        model_score = ((model_score - low) / (high - low).clamp_min(1e-6)).clamp_min(1e-6)
        proposal_xyxy = box_ops.box_cxcywh_to_xyxy(proposals.to(boxes))
        prior_score = box_ops.box_iou(xyxy[batch, candidates], proposal_xyxy)[0].amax(1)
        fused = (model_score.pow(model_weight.to(model_score))
                 * prior_score.clamp_min(1e-6).pow(input_weight.to(model_score)))
        chosen = candidates[fused.topk(min(top_k, candidates.numel())).indices]
        selected.append(chosen)
    return selected
