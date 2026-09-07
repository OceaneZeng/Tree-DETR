"""Ports OW-DETR's three components and implements EW-DETR from its paper.

OW reference: akshitac8/OW-DETR@3515ff8c36687a4f582ef6a2c83866966b56ce23.
Dataset, schedules, masking and geometry compatibility changes are documented.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.deformable_detr import build as build_base, SetCriterion
from models.segmentation import sigmoid_focal_loss
from util import box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized
from .ew_modules import DualLoRA, QueryNormEUMix, inject_dual_lora, merge_beta


class PaperDetector(nn.Module):
    def __init__(self, detector, args):
        super().__init__()
        self.detr = detector
        self.method = args.paper_baseline
        self.known_ids = list(args.owod_known_class_ids)
        self.unknown_id = 91
        detector.return_baseline_features = True
        if self.method == 'ow-detr':
            self.objectness = nn.Linear(args.hidden_dim, 1)
            nn.init.constant_(self.objectness.bias, -math.log(99))
        else:
            self.adapter_layers = inject_dual_lora(detector, args.ew_rank)
            self.calibration = QueryNormEUMix(args.hidden_dim, self.known_ids)

    def forward(self, samples):
        output = self.detr(samples)
        features = output.pop('decoder_features')
        layers = [*output.get('aux_outputs', []), output]
        layer_features = features if len(layers) > 1 else features[-1:]
        heads = self.detr.class_embed if len(layers) > 1 else self.detr.class_embed[-1:]
        for layer, hidden, head in zip(layers, layer_features, heads):
            if self.method == 'ow-detr':
                logits = layer['pred_logits']
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[self.known_ids + [self.unknown_id]] = False
                layer['pred_logits'] = logits.masked_fill(mask, -1e8)
                layer['pred_objectness'] = self.objectness(hidden)
            else:
                layer['pred_logits'] = self.calibration(hidden, head)
        if self.method != 'ow-detr':
            output.pop('attention_feature')
            output.pop('padded_size')
        return output

    @torch.no_grad()
    def consolidate(self, current_samples, previous_samples):
        beta = merge_beta(current_samples, previous_samples)
        for module in self.modules():
            if isinstance(module, DualLoRA):
                module.consolidate(beta)
        return beta


@torch.no_grad()
def attention_pseudo_queries(feature, boxes, matched, padded_size, top_k=5):
    """Integral-image box means, with clipped valid boxes and GT queries excluded."""
    height, width = padded_size
    attention = F.interpolate(feature.mean(1, keepdim=True), size=(height, width),
                              mode='bilinear', align_corners=False)[:, 0]
    integral = F.pad(attention.cumsum(1).cumsum(2), (1, 0, 1, 0))
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    coords = (xyxy * boxes.new_tensor([width, height, width, height])).long()
    coords[..., 0::2].clamp_(0, width)
    coords[..., 1::2].clamp_(0, height)
    selected = []
    for batch, (src, _) in enumerate(matched):
        x1, y1, x2, y2 = coords[batch].unbind(-1)
        area = (x2 - x1) * (y2 - y1)
        sums = (integral[batch, y2, x2] - integral[batch, y1, x2]
                - integral[batch, y2, x1] + integral[batch, y1, x1])
        valid = area > 0
        valid[src] = False
        candidates = torch.where(valid)[0]
        means = sums[candidates] / area[candidates]
        chosen = candidates[means.topk(min(top_k, candidates.numel())).indices]
        selected.append(chosen)
    return selected


class PaperCriterion(SetCriterion):
    def __init__(self, original, args):
        super().__init__(92, original.matcher, dict(original.weight_dict), original.losses,
                         original.focal_alpha)
        self.method = args.paper_baseline
        self.known_ids = list(args.owod_known_class_ids)
        self.epoch = 0
        self.warmup = args.ow_pseudo_warmup
        self.top_k = args.ow_top_unknown
        if self.method == 'ow-detr':
            self.weight_dict['loss_NC'] = 0.1
            for layer in range(args.dec_layers - 1):
                self.weight_dict[f'loss_NC_{layer}'] = 0.1

    def known_targets(self, targets):
        filtered = []
        for target in targets:
            mask = torch.zeros_like(target['labels'], dtype=torch.bool)
            for category in self.known_ids:
                mask |= target['labels'] == category
            filtered.append({**target, 'labels': target['labels'][mask], 'boxes': target['boxes'][mask]})
        return filtered

    def forward(self, outputs, targets):
        # Full validation labels remain available to the evaluator, never matcher/loss.
        targets = self.known_targets(targets)
        if self.method != 'ow-detr':
            return super().forward(outputs, targets)
        count = outputs['pred_logits'].new_tensor([sum(len(t['labels']) for t in targets)])
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(count)
        count = (count / get_world_size()).clamp(min=1).item()
        losses = {}
        for index, layer in enumerate([outputs, *outputs.get('aux_outputs', [])]):
            matched = self.matcher(layer, targets)
            classification_targets = copy.deepcopy(targets)
            classification_indices = matched
            if self.epoch >= self.warmup:
                selected = attention_pseudo_queries(outputs['attention_feature'], layer['pred_boxes'],
                                                    matched, outputs['padded_size'], self.top_k)
                classification_indices = []
                for target, (src, dst), queries in zip(classification_targets, matched, selected):
                    offset = len(target['labels'])
                    target['labels'] = torch.cat([target['labels'], target['labels'].new_full((len(queries),), 91)])
                    classification_indices.append((torch.cat([src.to(queries.device), queries]),
                        torch.cat([dst.to(queries.device), torch.arange(offset, offset + len(queries), device=queries.device)])))
            layer_losses = self.loss_labels(layer, classification_targets, classification_indices, count, log=index == 0)
            layer_losses.update(self.loss_boxes(layer, targets, matched, count))
            foreground = torch.zeros_like(layer['pred_objectness'])
            foreground[self._get_src_permutation_idx(classification_indices)] = 1
            layer_losses['loss_NC'] = sigmoid_focal_loss(layer['pred_objectness'], foreground, count,
                                                        alpha=self.focal_alpha, gamma=2) * foreground.shape[1]
            suffix = '' if index == 0 else f'_{index - 1}'
            losses.update({key + suffix: value for key, value in layer_losses.items()})
        return losses


class PaperPostProcess(nn.Module):
    def __init__(self, known_ids, threshold):
        super().__init__()
        self.known_ids = list(known_ids)
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        logits = outputs['pred_logits']
        # Sparse COCO slots stay intact; 91 is a dedicated unknown category.
        categories = torch.tensor(self.known_ids + [91], device=logits.device)
        probabilities = logits[..., categories].sigmoid()
        scores, indices = probabilities.flatten(1).topk(min(100, probabilities[0].numel()), dim=1)
        queries = indices // len(categories)
        labels = categories[indices % len(categories)]
        boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
        boxes = boxes.gather(1, queries[..., None].expand(-1, -1, 4))
        height, width = target_sizes.unbind(1)
        boxes *= torch.stack([width, height, width, height], -1)[:, None]
        unknown = logits[..., 91].sigmoid().gather(1, queries)
        return [dict(scores=s, labels=l, boxes=b, unknown_scores=u, unknown_mask=u >= self.threshold)
                for s, l, b, u in zip(scores, labels, boxes, unknown)]


def build(args):
    if args.paper_baseline == 'ew-detr' and args.replay_sampling_fraction:
        raise ValueError('EW-DETR is exemplar-free; replay is not allowed')
    if (args.lr_backbone != 0 or args.with_tree or args.neighbor_scoped_lora
            or args.teacher_completion or args.old_class_distillation or args.masks
            or args.two_stage or args.with_box_refine or args.num_feature_levels < 3
            or args.local_margin_coef or args.off_projection_coef):
        raise ValueError('Paper baselines require the isolated frozen-backbone, one-stage detector configuration')
    if args.num_classes != 92 or not args.owod_known_class_ids:
        raise ValueError('Expected sparse COCO labels with 92 slots (unknown=91)')
    detector, criterion, _ = build_base(args)
    model = PaperDetector(detector, args)
    return model, PaperCriterion(criterion, args), {
        'bbox': PaperPostProcess(args.owod_known_class_ids, args.unknown_threshold)}
