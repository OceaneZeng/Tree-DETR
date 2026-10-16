"""Verify immutable inputs for a diagnostic Task 2 fork, without rewriting checkpoints."""
from pathlib import Path

from tools.owod.protocol import file_sha256
from tools.owod.screening_data import screening_files


def validate_inputs(manifest, screening, checkpoint, gnn, weights, seed):
    import torch

    manifest, screening, checkpoint, gnn, weights = map(Path, (manifest, screening, checkpoint, gnn, weights))
    parent, files = screening_files(manifest, screening, 0)
    identity = {'checkpoint_sha256': file_sha256(checkpoint), 'gnn_sha256': file_sha256(gnn),
                'manifest_sha256': file_sha256(manifest), 'screening_sha256': file_sha256(screening),
                'weights_sha256': file_sha256(weights)}
    # Trusted local training artifact: args is an argparse.Namespace.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = payload.get('args')
    expected = {'owod_stage': 0, 'owod_phase': 'increment', 'seed': seed,
                'owod_discovery': True, 'graph_routed_adapter': True,
                'owod_manifest_sha256': identity['manifest_sha256'],
                'owod_screening_sha256': identity['screening_sha256'],
                'backbone_weights_sha256': identity['weights_sha256'],
                'paper_comparable_protocol': False}
    if any(getattr(saved, key, None) != value for key, value in expected.items()):
        raise ValueError('Common parent must be a matching diagnostic Task 1 discovery checkpoint')
    if (getattr(saved, 'graph_adapter_specs', []) or getattr(saved, 'screen_branch_checkpoint_sha256', None)
            or payload.get('epoch', -2) != getattr(saved, 'epochs', 0) - 1
            or not payload.get('model')
            or any('.graph_adapter.stages.' in key for key in payload['model'])):
        raise ValueError('Common Task 1 parent must be complete and have no class-adapter stages')
    del payload
    calibration = torch.load(gnn, map_location='cpu', weights_only=True)
    metadata = calibration.get('extra', {})
    expected_gnn = {'checkpoint_sha256': identity['checkpoint_sha256'],
                    'manifest_sha256': identity['manifest_sha256'],
                    'screening_sha256': identity['screening_sha256'],
                    'annotation_sha256': file_sha256(files['train']), 'stage': 0,
                    'production_ready': True, 'uses_validation_labels': False,
                    'supervision': 'empirical_train_loss_increase',
                    'last_decoder_layers': saved.graph_adapter_last_n}
    if (not {'config', 'state_dict'} <= calibration.keys()
            or any(metadata.get(key) != value for key, value in expected_gnn.items())
            or set(metadata.get('training_classes', [])) != set(parent['stages'][0]['active_classes'])):
        raise ValueError('Common GNN must be calibrated from exactly the common Task 1 checkpoint and data')
    return identity
