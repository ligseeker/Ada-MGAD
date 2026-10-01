#!/usr/bin/env python3
"""Frozen cross-worktree detector observation: inputs -> lock -> GT evaluation.

No fit, checkpoint selection or threshold search is available in this driver.
The registered candidate family is immutable before any new Test prediction.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.e2e.protocol import sha256_file, write_json
from src.e2e.system_trigger import (assign_legal_events, trigger_temporal_blocks,
    window_split_assignment, system_score_frame, to_builtin)
from src.e2e.system_trigger_data import build_trigger_loader
from scripts.p6.run_c0_trigger import infer_trigger_logits
from src.e2e.event_detection import construct_predicted_episodes, match_events, event_metrics
from src.e2e.bin_trigger_decoder import construct_bin_candidates
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def read_json(path):
    return json.loads(Path(path).read_text())


def seal_files(directory, name, extra):
    files = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*')
             if p.is_file() and p.name != name}
    write_json(directory / name, dict(extra, files=files))


def check_hashes(mapping):
    bad = [p for p, digest in mapping.items() if sha256_file(Path(p)) != digest]
    if bad:
        raise ValueError('immutable source/input drift: ' + str(bad))


class InputOnlyWindows(Dataset):
    """Exactly the frozen telemetry slicing; no label/registry argument exists."""
    def __init__(self, directory, indices=None):
        self.timestamps = np.load(Path(directory) / 'timestamps.npy', mmap_mode='r')
        self.metric = np.load(Path(directory) / 'metric.npy', mmap_mode='r')
        self.log = np.load(Path(directory) / 'log.npy', mmap_mode='r')
        self.trace = np.load(Path(directory) / 'trace.npy', mmap_mode='r')
        if not len(self.timestamps) == len(self.metric) == len(self.log) == len(self.trace):
            raise ValueError('modality length mismatch')
        if (self.metric.shape[1:] != (10, 48) or self.log.shape[1:] != (10, 32)
                or self.trace.shape[1:] != (10, 10, 8)
                or not np.all(np.diff(self.timestamps) == 30000)):
            raise ValueError('frozen telemetry schema/grid mismatch')
        self.sample_indices = np.arange(len(self.timestamps) - 9) if indices is None else np.asarray(indices)
        if len(self.sample_indices) == 0 or np.any(np.diff(self.sample_indices) <= 0):
            raise ValueError('nonempty chronological windows required')
        if self.sample_indices.min() < 0 or self.sample_indices.max() + 10 > len(self.timestamps):
            raise ValueError('window leaves input block')

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, position):
        index = int(self.sample_indices[position])
        return {'data_node': np.array(self.metric[index:index + 10], dtype=np.float32),
                'data_log': np.array(self.log[index:index + 10], dtype=np.float32),
                'data_edge': np.array(self.trace[index:index + 10], dtype=np.float32),
                'sample_index': np.asarray(index, dtype=np.int64)}


def load_nn(config, model_record):
    selection = read_json(model_record['selection'])
    arguments = dict(selection['model_args'])
    if arguments['batch_size'] != 32 or arguments['window'] != 10:
        raise ValueError('archived model geometry mismatch')
    arguments['gpu'] = True
    if not torch.cuda.is_available():
        raise RuntimeError('frozen GPU comparison requires CUDA')
    # Metric residual has its own archived class and checkpoint buffers. Other
    # models use byte-checked local numeric dependencies and the original args.
    if model_record['kind'] == 'metric':
        path = Path(model_record['source_tree']) / 'src/e2e/system_trigger_model.py'
        spec = importlib.util.spec_from_file_location('src.e2e.frozen_metric_model', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_class = module.SystemEventTrigger
    elif model_record['kind'] == 'tcn':
        from src.e2e.system_trigger_tcn import WindowCausalTCNTrigger
        model_class = WindowCausalTCNTrigger
    else:
        from src.e2e.system_trigger_model import SystemEventTrigger
        model_class = SystemEventTrigger
    graph = np.load(Path(config['data_root']) / 'graph.npy', allow_pickle=False)
    model = model_class(graph, **arguments).to('cuda')
    model.load_state_dict(torch.load(model_record['checkpoint'], map_location='cuda'), strict=True)
    model.eval()
    return model


def probability(logits):
    # Preserve the exact NumPy float32 sigmoid used by archived C0 selection.
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))


def validation_replay(model, config, record):
    archived = pd.read_csv(Path(record['run']) / 'validation_predictions.csv', float_precision='round_trip',
        usecols=['sample_index', 'system_trigger_score', 'system_trigger_logit'])
    errors = []
    for start in sorted({0, (len(archived) // 64) * 32, ((len(archived) - 1) // 32) * 32}):
        reference = archived.iloc[start:start + 32]
        dataset = InputOnlyWindows(Path(config['data_root']) / 'train', reference.sample_index.to_numpy())
        indices, logits = infer_trigger_logits(model, build_trigger_loader(dataset, batch_size=32,
            num_workers=0), torch.device('cuda'))
        if not np.array_equal(indices, reference.sample_index):
            raise ValueError('Validation replay window identity mismatch')
        error = float(np.max(np.abs(probability(logits).astype(float) - reference.system_trigger_score.to_numpy())))
        logit_error = float(np.max(np.abs(logits.astype(float) - reference.system_trigger_logit.to_numpy())))
        if error > 1e-6 or logit_error > 1e-4:
            raise ValueError('archived Validation numeric replay failed: ' + record['id'])
        errors.append({'first_row': start, 'n': len(reference), 'probability_error': error, 'logit_error': logit_error})
    return {'passed': True, 'batches': errors, 'labels_consumed': False}


def prepare(config, config_path, output):
    if output.exists():
        raise FileExistsError(output)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('commit this evaluation source/config before starting')
    if len(config['models']) != 10 or len(config['arms']) != 19:
        raise ValueError('registered candidate family is incomplete')
    if len({m['id'] for m in config['models']}) != 10 or len({a['id'] for a in config['arms']}) != 19:
        raise ValueError('duplicate candidate identity')
    bindings = {str(config_path.resolve()): sha256_file(config_path),
                str((ROOT / 'docs/P6_FROZEN_TEST_REVIEW_PROTOCOL.md').resolve()):
                    sha256_file(ROOT / 'docs/P6_FROZEN_TEST_REVIEW_PROTOCOL.md')}
    for directory in (ROOT / 'src', ROOT / 'util'):
        for path in directory.rglob('*.py'):
            bindings[str(path.resolve())] = sha256_file(path)
    for name in ('run_frozen_test_review.py', 'score_frozen_flat_xgb.py', 'run_c0_trigger.py'):
        path = ROOT / 'scripts/p6' / name
        bindings[str(path.resolve())] = sha256_file(path)
    for path in (Path(config['registry']), Path(config['base_config']), Path(config['ad_manifest']),
                 Path(config['data_root']) / 'graph.npy'):
        bindings[str(path.resolve())] = sha256_file(path)
    for split in ('train', 'test'):
        for name in ('metric', 'log', 'trace', 'timestamps'):
            path = Path(config['data_root']) / split / (name + '.npy')
            bindings[str(path.resolve())] = sha256_file(path)
    for record in config['models']:
        for name in ('selection', 'checkpoint', 'score_reference'):
            if record.get(name):
                path = Path(record[name]); bindings[str(path.resolve())] = sha256_file(path)
        if sha256_file(Path(record['checkpoint'])) != record['checkpoint_sha256']:
            raise ValueError('registered checkpoint digest mismatch')
        if record.get('selection') and record['kind'] not in ('historical', 'flat'):
            selection = read_json(record['selection'])
            if sha256_file(Path(record['checkpoint'])) != selection['checkpoint']['sha256']:
                raise ValueError('selected checkpoint digest mismatch')
        if record.get('completion'):
            path = Path(record['completion']); manifest = read_json(path)
            for leaf, value in manifest.get('files', {}).items():
                digest = value if isinstance(value, str) else value['sha256']
                if sha256_file(path.parent / leaf) != digest:
                    raise ValueError('candidate completion mismatch: ' + str(path.parent / leaf))
            bindings[str(path.resolve())] = sha256_file(path)
            for value in manifest.get('source_artifacts', {}).values():
                original = Path(value['path'])
                digest = bindings.get(str(original.resolve()))
                if digest is None:
                    digest = sha256_file(original)
                if digest != value['sha256']:
                    raise ValueError('archived input digest mismatch: ' + str(original))
                bindings[str(original.resolve())] = digest
        if record.get('source_tree'):
            for name in ('src/e2e/system_trigger_model.py', 'src/model_util.py', 'src/e2e/system_trigger_tcn.py'):
                path = Path(record['source_tree']) / name
                if path.exists(): bindings[str(path.resolve())] = sha256_file(path)
    for arm in config['arms']:
        if arm.get('threshold_source'):
            path = Path(arm['threshold_source']); bindings[str(path.resolve())] = sha256_file(path)
            value = read_json(path)
            for key in arm['threshold_key'].split('.'):
                value = value[key]
            if float(value) != arm['threshold']:
                raise ValueError('frozen threshold identity mismatch: ' + arm['id'])
    path = Path(config['decoder_recipe_source'])
    bindings[str(path.resolve())] = sha256_file(path)
    output.mkdir(parents=True)
    write_json(output / 'scope_lock.json', {'execution_commit': head(), 'config': config,
        'bindings': bindings, 'test_metrics_used_for_candidate_selection': False,
        'test_labels_read_before_global_prediction_lock': False,
        'test_telemetry_bytes_hashed_for_provenance': True,
        'candidate_rule': config['candidate_rule']})
    print('SCOPE_LOCKED', len(config['models']), 'models', len(config['arms']), 'arms', flush=True)


def verify_scope(output):
    scope = read_json(output / 'scope_lock.json')
    if scope['execution_commit'] != head():
        raise ValueError('execution HEAD drift')
    check_hashes(scope['bindings'])
    return scope['config']


def infer(config, output, model_id):
    records = [m for m in config['models'] if m['id'] == model_id]
    if len(records) != 1: raise ValueError('unknown model')
    record = records[0]; directory = output / 'models' / model_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if record['kind'] == 'historical':
        source = pd.read_csv(record['score_reference'], float_precision='round_trip',
            usecols=['sample_index', 'prediction_available_time', 'system_trigger_score', 'system_trigger_logit'])
        source = source.rename(columns={'system_trigger_score': 'system_score', 'system_trigger_logit': 'logit'})
        replay = {'mode': 'immutable historical scores; no fresh inference', 'graph_scope': 'batch'}
    elif record['kind'] == 'flat':
        subprocess.run([record['python'], str(ROOT / 'scripts/p6/score_frozen_flat_xgb.py'),
            '--scope', str(output / 'scope_lock.json'), '--model-id', model_id,
            '--output-dir', str(directory)], check=True)
        source = pd.read_csv(directory / 'scores.csv', float_precision='round_trip')
        replay = read_json(directory / 'replay.json')
    else:
        torch.set_num_threads(8); torch.manual_seed(record['seed']); torch.cuda.manual_seed_all(record['seed'])
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model = load_nn(config, record)
        replay = validation_replay(model, config, record)
        dataset = InputOnlyWindows(Path(config['data_root']) / 'test')
        loader = build_trigger_loader(dataset, batch_size=32, num_workers=2, pin_memory=True)
        indices, logits = infer_trigger_logits(model, loader, torch.device('cuda'))
        if not np.array_equal(indices, dataset.sample_indices):
            raise ValueError('Test window coverage mismatch')
        times = np.asarray(dataset.timestamps[indices + 9], dtype=np.int64) + 30000
        source = pd.DataFrame({'sample_index': indices, 'prediction_available_time': times,
                               'system_score': probability(logits), 'logit': logits})
    if source.sample_index.duplicated().any() or not np.isfinite(source[['system_score', 'logit']]).all().all():
        raise ValueError('invalid Test scores')
    if not np.all(np.diff(source.prediction_available_time) == 30000):
        raise ValueError('nonchronological Test scores')
    blocks = trigger_temporal_blocks(read_json(config['base_config']))
    if (int(source.prediction_available_time.iloc[0]) != blocks[2].start_ms + 300000
            or int(source.prediction_available_time.iloc[-1]) != blocks[2].end_ms):
        raise ValueError('frozen Test input bounds mismatch')
    source.to_csv(directory / 'scores.csv', index=False)
    write_json(directory / 'replay.json', to_builtin(replay))
    seal_files(directory, 'completion_manifest.json', {'status': 'PREDICTIONS_COMPLETE',
        'model_id': model_id, 'rows': len(source), 'execution_commit': head(),
        'elapsed_seconds': time.monotonic() - started, 'test_labels_read': False})
    print('PREDICTIONS_COMPLETE', model_id, len(source), flush=True)


def make_episodes(scores, arm):
    frame = system_score_frame('test', scores.prediction_available_time, scores.system_score)
    threshold = float(arm['threshold'])
    if arm['decoder'] == 'merged':
        return construct_predicted_episodes(frame, threshold)
    if arm['decoder'] == 'bin':
        return construct_bin_candidates(frame, threshold)
    if arm['decoder'] != 'rise': raise ValueError('unknown registered decoder')
    positive = scores.system_score.to_numpy() >= threshold
    starts = positive.copy(); starts[1:] &= ~positive[:-1]
    rises = np.zeros(len(scores), dtype=bool)
    rises[1:] = positive[1:] & positive[:-1] & (np.diff(scores.logit.to_numpy()) > 0)
    selected = np.flatnonzero(starts | rises)
    times = scores.prediction_available_time.to_numpy(dtype=np.int64)[selected]
    return pd.DataFrame({'prediction_id': ['test-rise-{:06d}'.format(i) for i in range(len(selected))],
        'split': 'test', 't_hat': times, 'episode_end_time': times + 30000,
        'positive_bins': 1, 'system_score': scores.system_score.to_numpy()[selected]})


def lock_predictions(config, output):
    if (output / 'global_prediction_lock.json').exists(): raise FileExistsError('prediction lock exists')
    bindings = {str((output / 'scope_lock.json').resolve()): sha256_file(output / 'scope_lock.json')}
    for model in config['models']:
        directory = output / 'models' / model['id']; manifest = read_json(directory / 'completion_manifest.json')
        for name, digest in manifest['files'].items():
            if sha256_file(directory / name) != digest: raise ValueError('prediction drift')
            bindings[str((directory / name).resolve())] = digest
    for arm in config['arms']:
        directory = output / 'arms' / arm['id']; directory.mkdir(parents=True, exist_ok=False)
        scores = pd.read_csv(output / 'models' / arm['model_id'] / 'scores.csv', float_precision='round_trip')
        make_episodes(scores, arm).to_csv(directory / 'episodes.csv', index=False)
        bindings[str((directory / 'episodes.csv').resolve())] = sha256_file(directory / 'episodes.csv')
    write_json(output / 'global_prediction_lock.json', {'status': 'PREDICTION_LOCKED',
        'execution_commit': head(), 'bindings': bindings, 'arms': config['arms'],
        'test_gt_read': False, 'test_metric_join_run': False, 'threshold_search': False})
    print('GLOBAL_PREDICTION_LOCKED', len(config['arms']), flush=True)


def evaluate(config, output):
    lock = read_json(output / 'global_prediction_lock.json'); check_hashes(lock['bindings'])
    if (output / 'evaluation').exists(): raise FileExistsError('evaluation exists')
    destination = output / 'evaluation'; destination.mkdir()
    registry = pd.read_csv(config['registry'])
    base = read_json(config['base_config']); blocks = trigger_temporal_blocks(base)
    legal, assigned, purged = assign_legal_events(registry, blocks)
    gt = assigned.loc[assigned.split.eq('test')].sort_values(['start_ms', 'source_index', 'case_id']).reset_index(drop=True)
    if len(gt) != 5787 or not gt.case_id.is_unique: raise ValueError('complete frozen GT identity mismatch')
    gt.to_csv(destination / 'test_gt.csv', index=False); purged.to_csv(destination / 'purged_gt.csv', index=False)
    results = {}; hit_sets = {}
    for arm in config['arms']:
        directory = destination / arm['id']; directory.mkdir()
        scores = pd.read_csv(output / 'models' / arm['model_id'] / 'scores.csv', float_precision='round_trip')
        episodes = pd.read_csv(output / 'arms' / arm['id'] / 'episodes.csv', float_precision='round_trip')
        matching = match_events(episodes, gt, tolerance_seconds=60); metrics = event_metrics(matching)
        ledger, invariants = build_failure_ledger(gt, split='test',
            slot_times_ms=scores.prediction_available_time.to_numpy(dtype=np.int64),
            slot_scores=scores.system_score.to_numpy(), threshold=arm['threshold'],
            episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64),
            matching=matching, origin_ms=blocks[0].start_ms, split_end_ms=blocks[2].end_ms,
            context_events=legal)
        if not all(invariants[k] for k in ('categories_sum_to_total', 'matched_equals_tp',
            'unmatched_events_equal_fn', 'unmatched_episodes_equal_fp')):
            raise ValueError('failure denominator mismatch')
        if metrics['ground_truth_event_count'] != len(gt): raise ValueError('GT denominator drift')
        matching.to_csv(directory / 'matching.csv', index=False); ledger.to_csv(directory / 'failure_ledger.csv', index=False)
        hit_sets[arm['id']] = set(matching.loc[matching.match_status.eq('matched'), 'case_id'])
        results[arm['id']] = {'arm': arm, 'metrics': metrics, 'failure': invariants,
            'stratified': stratified_summary(ledger, split='test')}
        write_json(directory / 'result.json', to_builtin(results[arm['id']]))
    baseline = hit_sets['causal_merged']
    for arm_id, hits in hit_sets.items():
        results[arm_id]['paired_vs_causal'] = {'both': len(hits & baseline),
            'candidate_only': sorted(hits - baseline), 'baseline_only': sorted(baseline - hits),
            'both_miss': len(gt) - len(hits | baseline), 'delta_recall': (len(hits) - len(baseline)) / len(gt)}
    write_json(destination / 'results.json', to_builtin({'status': 'FROZEN_TEST_OBSERVATION_COMPLETE',
        'complete_gt': len(gt), 'prediction_lock_sha256': sha256_file(output / 'global_prediction_lock.json'),
        'test_used_for_selection': False, 'reused_test': True, 'results': results}))
    table = [{'arm_id': k, 'model_id': v['arm']['model_id'], 'decoder': v['arm']['decoder'],
        'threshold': v['arm']['threshold'], 'graph_scope': v['arm']['graph_scope'],
        'role': v['arm']['role'], **{x: v['metrics'][x] for x in ('true_positive_events',
            'false_positive_events', 'false_negative_events', 'event_precision', 'event_recall', 'event_f1')}}
        for k, v in results.items()]
    pd.DataFrame(table).to_csv(destination / 'detector_results.csv', index=False)
    seal_files(destination, 'completion_manifest.json', {'status': 'COMPLETE', 'execution_commit': head(),
        'source_scope_sha256': sha256_file(output / 'scope_lock.json'), 'test_used_for_selection': False})
    print('EVALUATION_COMPLETE', len(table), 'arms', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'infer', 'lock', 'evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id')
    args = parser.parse_args(); config = read_json(args.config)
    if args.action == 'prepare': prepare(config, args.config, args.output_dir); return
    frozen = verify_scope(args.output_dir)
    if frozen != config: raise ValueError('registered config drift')
    if args.action == 'infer': infer(config, args.output_dir, args.model_id)
    elif args.action == 'lock': lock_predictions(config, args.output_dir)
    elif args.action == 'evaluate': evaluate(config, args.output_dir)
    verify_scope(args.output_dir)


if __name__ == '__main__':
    main()
