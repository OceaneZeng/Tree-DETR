"""Real two-process regression for variable-length training evidence."""
import os
from pathlib import Path
import socket
import subprocess
import sys


WORKER = r'''
import torch
import torch.distributed as dist
from util.training_runtime import gather_evidence
dist.init_process_group('gloo')
rank = dist.get_rank()
for sizes in ((0, 3), (2, 5), (0, 0)):
    count = sizes[rank]
    evidence = (torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) + 100 * rank,
                torch.full((count,), rank - 1, dtype=torch.long),
                torch.arange(count, dtype=torch.float32) / 10)
    reference = [None, None]
    dist.all_gather_object(reference, evidence)
    actual = gather_evidence(evidence)
    for field in range(3):
        torch.testing.assert_close(actual[field], torch.cat([part[field] for part in reference]),
                                   rtol=0, atol=0)
dist.destroy_process_group()
'''


def test_two_rank_gather_preserves_ragged_empty_and_rank_order():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    processes = []
    try:
        for rank in range(2):
            env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       WORLD_SIZE='2', RANK=str(rank), OMP_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', USE_LIBUV='0')
            processes.append(subprocess.Popen(
                [sys.executable, '-c', WORKER], cwd=Path(__file__).resolve().parents[3],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        for process in processes:
            output, _ = process.communicate(timeout=45)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
