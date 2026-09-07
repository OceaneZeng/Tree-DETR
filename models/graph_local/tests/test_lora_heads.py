"""Optimizer checks for the frozen-feature-extractor LoRA diagnostics."""

import copy
import unittest

import torch
from torch import nn

from models.graph_local.lora import (freeze_for_class_ids, inject_decoder_lora,
                                    merge_decoder_lora)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(4, 4)
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 4)

    def forward(self, features):
        features = features + self.attention(features)
        return features + self.linear2(torch.relu(self.linear1(features)))


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Linear(4, 4)
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList([TinyLayer() for _ in range(6)])
        classifier = nn.Linear(4, 5)
        box_head = nn.Linear(4, 4)
        self.class_embed = nn.ModuleList([classifier] * 6)
        self.bbox_embed = nn.ModuleList([box_head] * 6)

    def forward(self, features):
        features = self.transformer.encoder(self.backbone(features))
        for layer in self.transformer.decoder.layers:
            features = layer(features)
        return self.class_embed[-1](features), self.bbox_embed[-1](features)


class LoRAHeadTests(unittest.TestCase):
    def test_d3_optimizer_changes_heads_and_lora_but_no_base_features(self):
        torch.manual_seed(42)
        model = TinyDetector()
        wrappers = inject_decoder_lora(model, rank=2, last_n=6)
        self.assertEqual(len(wrappers), 12)
        handles, trainable = freeze_for_class_ids(model, [1, 3], train_detection_heads=True)
        self.assertFalse(handles)
        self.assertEqual(len(trainable), len({id(parameter) for parameter in trainable}))
        for name, parameter in model.named_parameters():
            expected = ("lora_" in name or name.startswith(("class_embed.", "bbox_embed.")))
            self.assertEqual(parameter.requires_grad, expected, name)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            classes, boxes = model(torch.ones(3, 4))
            (classes.square().sum() + boxes.square().sum()).backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertFalse(torch.equal(model.class_embed[0].weight[0], before["class_embed.0.weight"][0]))
        self.assertFalse(torch.equal(model.class_embed[0].weight[1], before["class_embed.0.weight"][1]))
        self.assertFalse(torch.equal(model.bbox_embed[0].weight, before["bbox_embed.0.weight"]))
        for wrapper in wrappers:
            self.assertGreater(float(wrapper.lora_b.abs().sum()), 0.0)
        inputs = torch.randn(3, 4)
        expected_outputs = model(inputs)
        restored = copy.deepcopy(model)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(merge_decoder_lora(restored, last_n=6), 12)
        for expected, actual in zip(expected_outputs, restored(inputs)):
            torch.testing.assert_close(expected, actual)

    def test_d1_still_masks_old_rows_and_freezes_box_head(self):
        torch.manual_seed(42)
        model = TinyDetector()
        inject_decoder_lora(model, rank=2, last_n=2)
        handles, trainable = freeze_for_class_ids(model, [1, 3])
        before = model.class_embed[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0.0)
        classes, _ = model(torch.ones(3, 4))
        classes.square().sum().backward()
        optimizer.step()
        torch.testing.assert_close(model.class_embed[0].weight[[0, 2, 4]], before[[0, 2, 4]], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.class_embed[0].weight[1], before[1]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.bbox_embed.parameters()))
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
