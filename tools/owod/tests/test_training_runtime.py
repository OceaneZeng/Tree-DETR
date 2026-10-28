"""Regression checks for resume-safe runtime changes; no detector build needed."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from util.training_runtime import (atomic_torch_save, copy_model_buffers,
                                   foreach_ema_parameters, gather_evidence,
                                   loader_worker_options)


def load_memory():
    path = Path(__file__).resolve().parents[3] / 'models/graph_local/discovery_memory.py'
    spec = importlib.util.spec_from_file_location('runtime_memory_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrototypeMemory


def reference_observe(memory, features, labels, weights, threshold=.8):
    """Original v5 implementation, kept to detect assignment/order changes."""
    assignments = []
    for feature, label, weight in zip(features, labels, weights):
        if not torch.isfinite(feature).all() or weight <= 0:
            continue
        if label >= 0:
            if label >= memory.known_slots:
                raise ValueError('Future/unknown semantic IDs are not prototype labels')
            index = int(label)
        else:
            counts = memory.counts[memory.known_slots:]
            similarity = nn.functional.cosine_similarity(feature[None], memory.centers[memory.known_slots:])
            similarity[counts == 0] = -2
            best = int(similarity.argmax())
            if similarity[best] >= threshold:
                index = memory.known_slots + best
            elif (counts == 0).any():
                index = memory.known_slots + int(torch.where(counts == 0)[0][0])
            else:
                continue
            assignments.append(index - memory.known_slots)
        previous = memory.counts[index].clone()
        total = previous + weight
        difference = feature - memory.centers[index]
        memory.centers[index].add_(difference * weight / total)
        memory.variance[index].add_(weight * difference * (feature - memory.centers[index]))
        memory.counts[index] = total
    return sorted(set(assignments))


class TrainingRuntimeTests(unittest.TestCase):
    def test_atomic_checkpoint_survives_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            atomic_torch_save({'epoch': 16}, path)

            def fail(value, handle, **kwargs):
                handle.write(b'partial checkpoint')
                raise OSError('simulated disk failure')

            with patch('util.training_runtime.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_torch_save({'epoch': 17}, path)
            self.assertEqual(torch.load(path, weights_only=True)['epoch'], 16)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_torch_save({'epoch': 17}, path)
            self.assertEqual(torch.load(path, weights_only=True)['epoch'], 17)

    def test_ema_matches_original_arithmetic_and_restored_state(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            torch.manual_seed(4)
            model = nn.Sequential(nn.Linear(7, 11), nn.BatchNorm1d(11), nn.Linear(11, 3)).to(device)
            reference = copy.deepcopy(model)
            optimized = copy.deepcopy(model)
            for decay in (0., .5, .999):
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(.031)
                    model[1].num_batches_tracked.add_(1)
                    for target, source in zip(reference.parameters(), model.parameters()):
                        target.mul_(decay).add_(source, alpha=1 - decay)
                    for target, source in zip(reference.buffers(), model.buffers()):
                        target.copy_(source)
                    foreach_ema_parameters(optimized, model, decay)
                    copy_model_buffers(optimized, model)
                for name, tensor in reference.state_dict().items():
                    torch.testing.assert_close(tensor, optimized.state_dict()[name], rtol=0, atol=0)
                # A restored teacher must not depend on any unsaved runtime cache.
                restored = copy.deepcopy(model)
                restored.load_state_dict(optimized.state_dict(), strict=True)
                optimized = restored

    def test_memory_preserves_sequential_assignments_and_running_state(self):
        Memory = load_memory()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            torch.manual_seed(7)
            reference = Memory(8, capacity=3, known_slots=4).to(device)
            optimized = copy.deepcopy(reference)
            for step in range(5):
                features = torch.randn(20, 8, device=device)
                features[5] = features[4]  # repeated anonymous evidence in a batch
                features[0, 0] = float('nan')
                labels = torch.tensor([0, 1, 2, 3] + [-1] * 16, device=device)
                weights = torch.linspace(0, 1, 20, device=device)
                weights[2] = -1
                expected = reference_observe(reference, features, labels, weights)
                actual = optimized.observe(features, labels, weights)
                self.assertEqual(expected, actual)
                for name, tensor in reference.state_dict().items():
                    torch.testing.assert_close(tensor, optimized.state_dict()[name], rtol=0, atol=0)
            empty = torch.empty(0, 8, device=device)
            self.assertEqual(optimized.observe(empty, labels[:0], weights[:0]), [])

    def test_single_rank_evidence_keeps_dtype_and_order(self):
        evidence = (torch.randn(3, 8), torch.tensor([-1, 2, 1]), torch.rand(3))
        for expected, actual in zip(evidence, gather_evidence(evidence)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(expected.data_ptr(), actual.data_ptr())

    def test_worker_options_handle_zero_workers(self):
        self.assertEqual(loader_worker_options(0), {})
        self.assertEqual(loader_worker_options(4), {'multiprocessing_context': 'spawn'})


if __name__ == '__main__':
    unittest.main()
