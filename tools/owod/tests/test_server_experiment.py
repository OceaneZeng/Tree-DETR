from argparse import Namespace
from pathlib import Path

from tools.owod.run_server_experiment import commands


def test_two_gpu_suite_changes_only_routing_and_localization(tmp_path):
    args = Namespace(variant='suite', manifest=tmp_path / 'manifest.json', weights=tmp_path / 'dino.pth',
                     output=tmp_path, gpus='0,1', batch_size=2, workers=4, seed=42,
                     resume=False, execute=False)
    runs = commands(args)
    assert [name for name, _ in runs] == ['hard', 'soft', 'soft-localization']
    normalized = []
    for name, command in runs:
        for flag, expected in [('--gpus', '0,1'), ('--nproc-per-node', '2'), ('--batch-size', '2'),
                               ('--discovery-scoring', 'hierarchical')]:
            assert command[command.index(flag) + 1] == expected
        assert '--dry-run' in command
        copy = command.copy()
        for flag in ('--graph-routing', '--graph-localization', '--output-dir'):
            copy[copy.index(flag) + 1] = 'VARIANT'
        normalized.append(copy)
    assert normalized[0] == normalized[1] == normalized[2]
    assert runs[-1][1][runs[-1][1].index('--graph-localization') + 1] == 'bbox-last'
    args.execute = args.resume = True
    assert all('--resume' in cmd and '--dry-run' not in cmd for _, cmd in commands(args))
