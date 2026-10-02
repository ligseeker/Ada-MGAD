#!/usr/bin/env python3
"""Manual ten-seed TCN retraining and frozen 68D XGB transfer orchestration.

predict seals all ten prediction sets; evaluate requires committed locks.
No Test-driven selection or RCA training entry point is provided.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.p6.repeat_protocol import (ROOT, SEEDS, PROTOCOL, read, sha, write, git,
    path, validate, verify_scope, verify_seal, family_bindings, require_committed)


def preflight(config):
    validate(config)
    observed = {}
    for kind, modules in [('detector', ['torch', 'numpy', 'pandas', 'sklearn', 'adabelief_pytorch']),
                          ('rca', ['xgboost', 'numpy', 'pandas', 'sklearn', 'scipy'])]:
        interpreter = config[kind + '_python']
        code = ('import json,importlib,platform; data={m:getattr(importlib.import_module(m),"__version__","available") for m in '
                + repr(modules) + '}; data["python"]=platform.python_version(); print(json.dumps(data))')
        record = subprocess.check_output([interpreter, '-c', code], text=True)
        print(kind.upper() + '_ENV', record.strip(), flush=True)
        environment = json.loads(record.strip().splitlines()[-1])
        if environment != config['expected_environment'][kind]:
            raise ValueError('frozen ' + kind + ' environment differs: ' + record)
        observed[kind] = environment
    for name in ('data_root', 'artifact_root', 'registry', 'raw_manifest', 'base_config'):
        if not path(config[name]).exists():
            raise FileNotFoundError(config[name])
    for split in ('train', 'test'):
        for name in ('timestamps', 'metric', 'log', 'trace'):
            if not (path(config['data_root']) / split / (name + '.npy')).is_file():
                raise FileNotFoundError(split + '/' + name)
    print('PREFLIGHT_OK; metadata and environments only; no telemetry arrays or labels opened', flush=True)
    return observed


def prepare(config, config_path, output, args):
    observed = preflight(config)
    if git('status', '--porcelain'):
        raise ValueError('commit integrated code/config before execution')
    output.resolve().relative_to(ROOT / 'experiments/p6/two_stage_repeats')
    if output.exists():
        raise FileExistsError(output)
    sources = {config_path.resolve(), path(config['detector_config']), path(config['base_config']),
               ROOT / 'docs/P6_TWO_STAGE_REPEATS_PROTOCOL.md'}
    sources.add(path(read(path(config['detector_config']))['source_config']['path']))
    for directory in ('src', 'util', 'scripts/p6'):
        sources.update((ROOT / directory).rglob('*.py'))
    inputs = {path(config['registry']), path(config['raw_manifest']),
              path(config['artifact_root']) / 'ad_data_manifest.json', path(config['data_root']) / 'graph.npy'}
    for key in ('model', 'train_manifest', 'run_lock'):
        inputs.add(path(config['frozen_rca'][key]))
    for split in ('train', 'test'):
        inputs.update(path(config['data_root']) / split / (name + '.npy') for name in ('timestamps', 'metric', 'log', 'trace'))
    ad_manifest = read(path(config['artifact_root']) / 'ad_data_manifest.json')
    expected_ad = {path(config['data_root']) / 'graph.npy': ad_manifest['graph']['sha256']}
    for split in ('train', 'test'):
        for name in ('timestamps', 'metric', 'log', 'trace'):
            expected_ad[path(config['data_root']) / split / (name + '.npy')] = ad_manifest['split_files'][split][name]['sha256']
    if (ad_manifest['test_used_for_selection'] or ad_manifest['gt_labels_used_for_schema']
            or ad_manifest['decision_inputs'] != ['train']):
        raise ValueError('shared preprocessing violates the frozen Train-only contract')
    raw = read(path(config['raw_manifest'])); raw_root = path(config['raw_manifest']).parent
    expected_raw = {}
    for record in raw['metric_series']:
        for key in ('timestamps', 'values'):
            expected_raw[(raw_root / record[key]).resolve()] = record[key + '_sha256']
    for record in raw['logs'].values():
        for key in ('timestamps', 'levels'):
            expected_raw[(raw_root / record[key]).resolve()] = record[key + '_sha256']
    for record in raw['traces']['parts']:
        for key in ('timestamps', 'errors', 'latencies'):
            expected_raw[(raw_root / record[key]).resolve()] = record[key + '_sha256']
    inputs.update(expected_raw)
    bindings = {}
    for file in sorted(inputs):
        if file in expected_raw:
            file.relative_to(raw_root)
        digest = sha(file); stat = file.stat()
        if file in expected_raw and digest != expected_raw[file]:
            raise ValueError('raw-index checksum mismatch: ' + str(file))
        if file in expected_ad and digest != expected_ad[file]:
            raise ValueError('frozen detector input checksum mismatch: ' + str(file))
        bindings[str(file)] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    base = read(path(config['base_config']))
    if bindings[str(path(config['registry']))]['sha256'] != base['event_registry']['sha256']:
        raise ValueError('frozen event registry mismatch')
    output.mkdir(parents=True)
    write(output / 'scope_lock.json', dict(protocol_id=PROTOCOL, execution_commit=git('rev-parse', 'HEAD'),
        config=config, source_sha256={str(p): sha(p) for p in sorted(sources)}, inputs=bindings,
        runtime=dict(gpu=args.gpu, threads=args.threads, threshold_workers=args.threshold_workers,
                     feature_workers=args.feature_workers), seeds=SEEDS,
        observed_environment=observed, hash_seed_policy=dict(detector='registered seed', rca=20260826),
        evidence_grade='frozen-scorer transfer; reused-Test descriptive',
        test_labels_used_for_selection=False, rca_retrained=False))


def predict(config, config_path, output, args):
    if args.resume:
        scope = verify_scope(output)
        if preflight(config) != scope['observed_environment']:
            raise ValueError('resume environment changed')
        if scope['config'] != config or scope['runtime'] != dict(gpu=args.gpu, threads=args.threads,
                threshold_workers=args.threshold_workers, feature_workers=args.feature_workers):
            raise ValueError('resume changes scope or runtime')
    else:
        prepare(config, config_path, output, args)
    if (output / 'global_prediction_lock.json').exists():
        raise FileExistsError('prediction family already locked; proceed to evaluate')
    for seed in SEEDS:
        for stage, kind, script in [('detector', 'detector', 'repeat_detector.py'), ('rca', 'rca', 'repeat_rca.py')]:
            folder = output / ('seed-%02d' % seed) / stage
            if folder.exists():
                verify_seal(folder)
                print('RESUME_COMPLETE', seed, stage, flush=True)
                continue
            subprocess.run([config[kind + '_python'], str(ROOT / 'scripts/p6' / script),
                            'predict', '--output-dir', str(output), '--seed', str(seed)], check=True,
                env=dict(os.environ, PYTHONHASHSEED=str(seed if kind == 'detector' else 20260826),
                         PYTHONDONTWRITEBYTECODE='1'))
    verify_scope(output, deep=True)
    write(output / 'global_prediction_lock.json', dict(status='ALL_TEN_PREDICTION_SETS_LOCKED',
        seeds=SEEDS, scope_sha256=sha(output / 'scope_lock.json'), files=family_bindings(output),
        test_gt_or_matching_read=False, test_used_for_selection=False))
    print('ALL_TEN_LOCKED; commit scope_lock.json and global_prediction_lock.json before evaluate', flush=True)


def check_family(output, committed=True):
    scope = verify_scope(output, deep=True)
    if preflight(scope['config']) != scope['observed_environment']:
        raise ValueError('prediction/evaluation environment changed')
    lock = read(output / 'global_prediction_lock.json')
    if (lock['status'] != 'ALL_TEN_PREDICTION_SETS_LOCKED' or lock['test_gt_or_matching_read']
            or lock['test_used_for_selection']):
        raise ValueError('prediction family is not a valid label-free lock')
    if lock['seeds'] != SEEDS or lock['scope_sha256'] != sha(output / 'scope_lock.json'):
        raise ValueError('incomplete or different prediction family')
    if lock['files'] != family_bindings(output):
        raise ValueError('prediction bytes changed since global lock')
    if committed:
        require_committed(output / 'scope_lock.json')
        require_committed(output / 'global_prediction_lock.json')
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'predict', 'evaluate', 'summarize'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/e2e/gaia_p6_two_stage_repeats_v1.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--threshold-workers', type=int, default=8)
    parser.add_argument('--feature-workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if min(args.threads, args.threshold_workers, args.feature_workers) < 1:
        parser.error('worker/thread counts must be positive')
    if args.action == 'preflight':
        preflight(config); return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    output = args.output_dir.resolve()
    if args.action == 'predict':
        predict(config, args.config, output, args)
    else:
        # The RCA child performs the committed all-ten barrier and deep input
        # verification before its first semantic GT read.
        scope = verify_scope(output)
        if scope['config'] != config:
            raise ValueError('requested config differs from the frozen run scope')
        subprocess.run([scope['config']['rca_python'], str(ROOT / 'scripts/p6/repeat_rca.py'), args.action,
                        '--output-dir', str(output)], check=True,
                       env=dict(os.environ, PYTHONHASHSEED='20260826', PYTHONDONTWRITEBYTECODE='1'))


if __name__ == '__main__':
    main()
