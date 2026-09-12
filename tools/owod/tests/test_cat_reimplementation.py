import json
import os
import sqlite3
import zlib
from types import SimpleNamespace

import torch

from datasets.coco import CocoDetection
from datasets import transforms as T
from main import get_args_parser
from models.paper_baselines.cat_modules import AdaptivePseudoLabeler, cat_pseudo_queries
from models.paper_baselines.detector import PaperCriterion
from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from tools.owod.run_mowodb_reimplementations import state_directory


def test_adaptive_controller_updates_normalized_weights_and_round_trips_state():
    controller = AdaptivePseudoLabeler(
        memory_size=4, recent_size=2, start_iteration=4, update_interval=1,
        positive_momentum=0.1, negative_momentum=0.1)
    initial = controller.model_weight.item()
    for loss in (4.0, 4.0, 1.0, 1.0):
        controller.observe(torch.tensor(loss))
    assert controller.model_weight.item() < initial
    torch.testing.assert_close(controller.model_weight + controller.input_weight,
                               torch.tensor(1.0))
    restored = AdaptivePseudoLabeler(
        memory_size=4, recent_size=2, start_iteration=4, update_interval=1)
    restored.load_state_dict(controller.state_dict())
    torch.testing.assert_close(restored.model_weight, controller.model_weight)
    torch.testing.assert_close(restored.loss_memory, controller.loss_memory)


def test_cat_fusion_excludes_matched_query_and_uses_selective_search_iou():
    feature = torch.ones(1, 4, 2, 2)
    boxes = torch.tensor([[[.2, .2, .1, .1], [.4, .4, .2, .2], [.75, .75, .2, .2]]])
    matched = [(torch.tensor([0]), torch.tensor([0]))]
    proposals = [torch.tensor([[.75, .75, .2, .2]])]
    selected = cat_pseudo_queries(
        feature, boxes, matched, proposals, (20, 20),
        torch.tensor(.8), torch.tensor(.2), top_k=1)
    assert selected[0].tolist() == [2]


def test_cat_proposals_follow_geometric_augmentation_and_normalization():
    image = torch.zeros(3, 100, 200)
    target = {
        "boxes": torch.tensor([[20., 10., 60., 50.]]),
        "proposal_boxes": torch.tensor([[100., 20., 180., 80.]]),
        "labels": torch.tensor([1]),
        "area": torch.tensor([1600.]),
        "iscrowd": torch.tensor([0]),
        "size": torch.tensor([100, 200]),
    }
    image, target = T.hflip(image, target)
    torch.testing.assert_close(target["proposal_boxes"],
                               torch.tensor([[20., 20., 100., 80.]]))
    image, target = T.resize(image, target, 50)
    image, target = T.Normalize([0, 0, 0], [1, 1, 1])(image, target)
    torch.testing.assert_close(target["proposal_boxes"],
                               torch.tensor([[.3, .5, .4, .6]]))


def test_sqlite_proposals_are_loaded_lazily_and_missing_ids_are_empty(tmp_path):
    database = tmp_path / "selective_search.sqlite3"
    boxes = [[1, 2, 30, 40], [5, 6, 7, 8]]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE image_proposals "
                           "(image_id INTEGER PRIMARY KEY, boxes BLOB NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?, ?)",
                           ("coordinate_format", "xywh_absolute"))
        connection.execute("INSERT INTO image_proposals VALUES (?, ?)",
                           (17, zlib.compress(json.dumps(boxes).encode("utf-8"))))

    dataset = CocoDetection.__new__(CocoDetection)
    dataset.proposal_file = str(database)
    dataset._proposal_connection = None
    dataset._proposal_pid = None
    dataset.proposals = {}
    assert dataset._proposal_boxes(17) == boxes
    assert dataset._proposal_boxes(18) == []
    assert dataset._proposal_pid == os.getpid()
    dataset._proposal_connection.close()


def test_main_parser_accepts_cat_database_and_controller_parameters():
    args = get_args_parser().parse_args([
        "--paper-baseline", "cat", "--cat-proposals", "proposals.sqlite3",
        "--cat-loss-memory", "120", "--cat-recent-window", "30",
    ])
    assert args.paper_baseline == "cat"
    assert args.cat_proposals == "proposals.sqlite3"
    assert (args.cat_loss_memory, args.cat_recent_window) == (120, 30)


def test_single_method_plans_do_not_overwrite_each_other(tmp_path):
    assert state_directory(tmp_path, ("cat",)) == tmp_path / "cat"
    assert state_directory(tmp_path, ("ew-detr",)) == tmp_path / "ew-detr"
    assert state_directory(tmp_path, ("cat", "ew-detr")) == tmp_path


def test_cat_validation_neither_requires_proposals_nor_updates_controller():
    base = SetCriterion(
        92, HungarianMatcher(),
        {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2},
        ["labels", "boxes", "cardinality"])
    controller = AdaptivePseudoLabeler(
        memory_size=4, recent_size=2, start_iteration=4, update_interval=1)
    args = SimpleNamespace(
        paper_baseline="cat", owod_known_class_ids=[2], ow_pseudo_warmup=0,
        ow_top_unknown=5, dec_layers=1)
    criterion = PaperCriterion(base, args, SimpleNamespace(cat_adaptive=controller))
    outputs = {
        "pred_logits": torch.randn(1, 3, 92),
        "pred_boxes": torch.rand(1, 3, 4),
        "pred_objectness": torch.randn(1, 3, 1),
        "attention_feature": torch.rand(1, 2, 4, 4),
        "padded_size": (16, 16),
    }
    targets = [{"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)}]
    criterion.eval()
    losses = criterion(outputs, targets)
    assert torch.isfinite(sum(losses.values()))
    assert controller.iteration.item() == 0
