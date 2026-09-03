# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import numpy as np
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from models.graph_local.pseudo_labels import complete_targets_with_teacher
from models.owod_metrics import compute_owod_metrics


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    teacher: torch.nn.Module | None = None,
                    teacher_old_class_ids: Iterable[int] | None = None,
                    teacher_score_threshold: float = 0.5,
                    teacher_duplicate_iou: float = 0.7,
                    teacher_ground_truth_iou: float = 0.5,
                    teacher_max_per_image: int = 20):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    pseudo_total = 0

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if teacher is not None:
            targets, pseudo_counts = complete_targets_with_teacher(
                teacher,
                samples,
                targets,
                old_class_ids=teacher_old_class_ids,
                score_threshold=teacher_score_threshold,
                duplicate_iou=teacher_duplicate_iou,
                ground_truth_iou=teacher_ground_truth_iou,
                max_per_image=teacher_max_per_image,
            )
            pseudo_total += sum(pseudo_counts)
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if teacher is not None:
            metric_logger.update(pseudo_labels=pseudo_total / max(1, _ + 1))

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _coco_ap50_for_categories(coco_eval, category_ids) -> float:
    """Return COCO-style AP50 for a category subset, in the usual [0, 1] scale."""
    requested = {int(class_id) for class_id in category_ids or []}
    if not requested or not coco_eval.eval:
        return float("nan")
    params = coco_eval.params
    category_indices = [index for index, class_id in enumerate(params.catIds)
                        if int(class_id) in requested]
    iou_indices = np.flatnonzero(np.isclose(params.iouThrs, 0.5))
    if not category_indices or not iou_indices.size:
        return float("nan")
    area_index = list(params.areaRngLbl).index("all")
    max_dets_index = list(params.maxDets).index(100)
    precision = coco_eval.eval["precision"][
        int(iou_indices[0]), :, category_indices, area_index, max_dets_index]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else float("nan")


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             owod_known_class_ids=None, owod_unknown_threshold=0.5,
             owod_previous_class_ids=None, owod_current_class_ids=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    owod_predictions = []
    owod_targets = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if owod_known_class_ids is not None:
            owod_predictions.extend([
                {key: value.detach().cpu() for key, value in result.items()
                 if torch.is_tensor(value)} for result in results])
            owod_targets.extend([
                {key: value.detach().cpu() for key, value in target.items()
                 if torch.is_tensor(value)} for target in targets])
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            bbox_eval = coco_evaluator.coco_eval['bbox']
            class_ap50 = {
                'Previous': _coco_ap50_for_categories(bbox_eval, owod_previous_class_ids),
                'Current': _coco_ap50_for_categories(bbox_eval, owod_current_class_ids),
                'Known': _coco_ap50_for_categories(bbox_eval, owod_known_class_ids),
            }
            class_ap50 = {key: value for key, value in class_ap50.items()
                          if not math.isnan(value)}
            if class_ap50:
                stats.update({f'owod_{key.lower()}_ap50': value
                              for key, value in class_ap50.items()})
                print("OWOD class AP50:", class_ap50)
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    if owod_known_class_ids is not None:
        if utils.is_dist_avail_and_initialized():
            gathered_predictions = [None for _ in range(utils.get_world_size())]
            gathered_targets = [None for _ in range(utils.get_world_size())]
            torch.distributed.all_gather_object(gathered_predictions, owod_predictions)
            torch.distributed.all_gather_object(gathered_targets, owod_targets)
            owod_predictions = [item for batch in gathered_predictions for item in batch]
            owod_targets = [item for batch in gathered_targets for item in batch]
        owod_stats = compute_owod_metrics(
            owod_predictions, owod_targets, owod_known_class_ids,
            threshold=owod_unknown_threshold)
        stats.update({f'owod_{key.lower().replace("-", "_")}': value
                      for key, value in owod_stats.items()})
        print("OWOD metrics:", owod_stats)
    return stats, coco_evaluator
