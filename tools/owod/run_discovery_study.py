"""Resume official downloads, prepare exact author splits, train and compare AP.

Run this as a persistent local process. Progress and subprocess logs live under
the output directory. A failure never becomes an AP result or a substitute split.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod.manage_data import prepare_dataset, verify_dataset
from tools.owod.validate_discovery import compare_runs


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def execute(command, logfile):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'models/ops'), str(ROOT),
                                               environment.get('PYTHONPATH', '')])
    environment['PYTHONUNBUFFERED'] = '1'
    with logfile.open('a', encoding='utf-8') as stream:
        stream.write('\n' + datetime.now(timezone.utc).isoformat() + '\n' + repr(command) + '\n')
        stream.flush()
        return subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def study(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocols = ['m-owodb', 's-owodb'] if args.benchmark == 'both' else [args.benchmark]
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.pop('download_attempts')
    configuration['study_version'] = 3
    config_path = output / 'study_config.json'
    if config_path.is_file() and json.loads(config_path.read_text(encoding='utf-8')) != configuration:
        raise ValueError('Study already exists with different settings; use another output directory')
    save(config_path, configuration)
    state = {'pid': os.getpid(), 'status': 'running', 'ap_measured': False, 'benchmarks': {}}
    def update(**values):
        state.update(values)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save(output / 'study_status.json', state)
    update(phase='starting')
    try:
        for protocol in protocols:
            record = state['benchmarks'][protocol] = {'status': 'downloading', 'ap_measured': False}
            update(phase='download', benchmark=protocol)
            command = [sys.executable, str(ROOT / 'tools/owod/download_data.py'),
                       '--benchmark', protocol, '--weights', str(args.weights.resolve())]
            if args.proxy:
                command += ['--proxy', args.proxy]
            for attempt in range(args.download_attempts):
                record['download_attempt'] = attempt + 1
                update()
                code = execute(command, output / (protocol + '-download.log'))
                if code == 0 or protocol == 'm-owodb':
                    break  # don't repeatedly poll an empty author Drive folder
                if attempt + 1 < args.download_attempts:
                    time.sleep(min(60, 5 * (attempt + 1)))
            if code:
                record.update(status='source_unavailable', download_exit_code=code,
                              reason='See download log; official source incomplete. No substitute split used.')
                update()
                continue
            record['status'] = 'preparing'
            update(phase='prepare')
            source = ROOT / 'data/raw' / ('ore' if protocol == 'm-owodb' else 'coco')
            dataset = ROOT / 'data/owod' / protocol
            record['data'] = prepare_dataset(protocol, source, dataset, decode_images=True)
            record['verification'] = verify_dataset(dataset, protocol, source)
            update(phase='training')
            for method in ('discovery', 'discovery-no-message', 'discovery-scratch'):
                record.update(status='training', method=method)
                update()
                command = [sys.executable, str(ROOT / 'tools/owod/run_benchmark.py'),
                           '--manifest', str(dataset / 'split_manifest.json'),
                           '--output-dir', str(output / method), '--recipe', 'ow-detr-paper',
                           '--backbone-weights', str(args.weights.resolve()),
                           '--method', method, '--seed', str(args.seed), '--nproc-per-node', '1',
                           '--batch-size', str(args.batch_size), '--num-workers', str(args.num_workers),
                           '--resume']
                if args.activation_checkpointing:
                    command += ['--activation-checkpointing']
                code = execute(command, output / (protocol + '-' + method + '-training.log'))
                if code:
                    record.update(status='training_failed', training_exit_code=code)
                    update()
                    break
            if record['status'] == 'training_failed':
                continue
            update(phase='comparison')
            seed_root = Path(protocol) / f'seed_{args.seed}'
            for baseline, label in [('discovery-no-message', 'graph-vs-self-loop'),
                                    ('discovery-scratch', 'transfer-vs-scratch')]:
                result = compare_runs(output / baseline / seed_root, output / 'discovery' / seed_root)
                save(output / (protocol + '-' + label + '.json'), result)
            record.update(status='measured_single_seed', ap_measured=True)
            update(ap_measured=True)
        failures = any(r['status'] != 'measured_single_seed' for r in state['benchmarks'].values())
        update(status='incomplete' if failures else 'completed', phase='finished')
        return int(failures)
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('m-owodb', 's-owodb', 'both'), default='both')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/discovery/full-study-v3')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--proxy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--download-attempts', type=int, default=12)
    args = parser.parse_args(argv)
    if args.download_attempts < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid positive run/download count or worker count')
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output.resolve() / 'study.lock'), timeout=1):
        return study(args)


if __name__ == '__main__':
    raise SystemExit(main())
