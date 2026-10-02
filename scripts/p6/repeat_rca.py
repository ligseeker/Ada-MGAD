#!/usr/bin/env python3
"""Frozen 68D XGB ranking and post-lock detector/matched RCA/full E2E reports."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.p6.repeat_protocol import (SEEDS, read, sha, write, path, seal, verify_scope, verify_seal)
from scripts.p6.run_frozen_rca_test_review import initialize_index, extract, score, metrics_for
from src.e2e.protocol import GAIA_SERVICES, SUPPORTED_FAULT_TYPES
from src.e2e.system_trigger import assign_legal_events, trigger_temporal_blocks, to_builtin
from src.e2e.event_detection import match_events, event_metrics
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary


def predict(output, seed):
    if seed not in SEEDS or xgb.__version__ != '2.1.4':
        raise ValueError('unregistered seed or frozen XGB environment drift')
    if os.environ.get('PYTHONHASHSEED') != '20260826':
        raise ValueError('frozen RCA extraction uses a constant startup hash seed')
    scope = verify_scope(output); config = scope['config']
    detector = output / ('seed-%02d' % seed) / 'detector'
    completed = verify_seal(detector)
    if completed['seed'] != seed or completed['scope_sha256'] != sha(output / 'scope_lock.json'):
        raise ValueError('detector seed/scope mismatch')
    directory = output / ('seed-%02d' % seed) / 'rca'
    directory.mkdir(parents=True, exist_ok=False)
    episodes = pd.read_csv(detector / 'episodes.csv', float_precision='round_trip')
    anchors = episodes.t_hat.to_numpy(dtype=np.int64) - config['frozen_rca']['anchor_backdate_ms']
    blocks = trigger_temporal_blocks(read(path(config['base_config'])))
    valid = (anchors - 300000 >= blocks[2].start_ms) & (anchors + 300000 <= blocks[2].end_ms)
    values = np.zeros((len(episodes), 10, 68), dtype=np.float32)
    positions = np.flatnonzero(valid)
    workers = scope['runtime']['feature_workers']
    # Only predicted timestamps enter extraction; no GT identities/root labels.
    if len(positions):
        with mp.get_context('spawn').Pool(workers, initializer=initialize_index,
                initargs=(path(config['raw_manifest']),)) as pool:
            for done, (position, item) in enumerate(zip(positions, pool.imap(extract,
                    anchors[positions].tolist(), chunksize=4)), 1):
                anchor, features = item
                if anchor != anchors[position]:
                    raise ValueError('anchor-to-feature mapping changed')
                values[position] = features
                if done % 256 == 0:
                    print('RCA_FEATURE_PROGRESS', seed, done, '/', len(positions), flush=True)
    frame = episodes[['prediction_id', 't_hat']].copy()
    frame['anchor_ms'] = anchors; frame['legal'] = valid
    frame['ranking'] = score(path(config['frozen_rca']['model']), values, valid)
    if not frame.prediction_id.is_unique:
        raise ValueError('duplicate RCA prediction IDs')
    frame.to_csv(directory / 'rankings.csv', index=False)
    np.save(directory / 'features.npy', values, allow_pickle=False)
    np.save(directory / 'valid.npy', valid, allow_pickle=False)
    write(directory / 'method_record.json', dict(seed=seed, hash_seed=os.environ['PYTHONHASHSEED'],
        scorer_sha256=config['frozen_rca']['model_sha256'], scorer_retrained=False, scaler=None,
        anchor_backdate_ms=config['frozen_rca']['anchor_backdate_ms'], feature_shape=list(values.shape),
        illegal_context_predictions=int((~valid).sum()), feature_window='W300-B15; +/-300s',
        test_gt_or_matching_read=False, evidence_grade=scope['evidence_grade']))
    seal(directory, status='COMPLETE', seed=seed, scope_sha256=sha(output / 'scope_lock.json'),
         test_gt_or_matching_read=False)
    print('RCA_COMPLETE', seed, 'predictions', len(frame), 'legal', int(valid.sum()), flush=True)


def evaluate_seed(episodes, scores, rankings, gt, legal, blocks, tau):
    matching = match_events(episodes, gt, tolerance_seconds=60)
    detector = event_metrics(matching)
    if detector['ground_truth_event_count'] != len(gt):
        raise ValueError('complete detector denominator changed')
    ledger, invariants = build_failure_ledger(gt, split='test',
        slot_times_ms=scores.prediction_available_time.to_numpy(dtype=np.int64),
        slot_scores=scores.system_score.to_numpy(), threshold=tau,
        episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
        episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64),
        matching=matching, origin_ms=blocks[0].start_ms, split_end_ms=blocks[2].end_ms,
        context_events=legal)
    if not all(invariants[key] for key in ('categories_sum_to_total', 'matched_equals_tp',
            'unmatched_events_equal_fn', 'unmatched_episodes_equal_fp')):
        raise ValueError('detector failure ledger does not close')
    diagnosis, joined = metrics_for(rankings, matching, len(gt))
    diagnosis['macro_e2e_recall'] = {}
    for dimension, labels in [('gt_service', GAIA_SERVICES), ('fault_type', SUPPORTED_FAULT_TYPES)]:
        groups = []
        for label in labels:
            group = joined.loc[joined.match_status.ne('false_alarm') & joined[dimension].eq(label)]
            groups.append(dict(group=label, n=len(group),
                recall={str(k): float(group.root_rank.le(k).mean()) if len(group) else None for k in (1, 3, 5)}))
        diagnosis['macro_e2e_recall'][dimension] = dict(groups=groups,
            mean={str(k): float(np.mean([g['recall'][str(k)] for g in groups if g['n']])) for k in (1, 3, 5)},
            defined_groups=sum(g['n'] > 0 for g in groups), total_groups=len(labels))
    return dict(detector=detector, detector_failure=invariants,
                detector_stratified=stratified_summary(ledger, split='test'), diagnosis=diagnosis), matching, ledger, joined


def evaluate(output):
    # Direct invocation is held to the same committed all-ten barrier.
    from scripts.p6.run_two_stage_repeats import check_family
    scope = check_family(output); config = scope['config']
    destination = output / 'evaluation'
    if destination.exists():
        verify_seal(destination)
        raise FileExistsError('evaluation already complete; use summarize')
    registry = pd.read_csv(path(config['registry']))  # first Test semantic label read
    blocks = trigger_temporal_blocks(read(path(config['base_config'])))
    legal, assigned, purged = assign_legal_events(registry, blocks)
    gt = assigned.loc[assigned.split.eq('test')].sort_values(['start_ms', 'source_index', 'case_id']).reset_index(drop=True)
    if len(gt) != config['expected_test_gt'] or not gt.case_id.is_unique:
        raise ValueError('complete frozen Test GT identity/count mismatch')
    destination.mkdir()
    gt.to_csv(destination / 'test_gt.csv', index=False)
    purged.to_csv(destination / 'purged_gt.csv', index=False)
    for seed in SEEDS:
        folder = output / ('seed-%02d' % seed)
        episodes = pd.read_csv(folder / 'detector/episodes.csv', float_precision='round_trip')
        scores = pd.read_csv(folder / 'detector/scores.csv', float_precision='round_trip')
        rankings = pd.read_csv(folder / 'rca/rankings.csv', keep_default_na=False, float_precision='round_trip')
        selection = read(folder / 'detector/validation_selection.json')
        result, matching, ledger, diagnosis = evaluate_seed(episodes, scores, rankings, gt, legal, blocks,
            float(selection['selected_validation_threshold']))
        result.update(seed=seed, evidence_grade=scope['evidence_grade'], selection=selection,
            validation_bin=read(folder / 'detector/validation_bin_metrics.json'),
            prediction_lock_sha256=sha(output / 'global_prediction_lock.json'), test_used_for_selection=False)
        sub = destination / ('seed-%02d' % seed); sub.mkdir()
        matching.to_csv(sub / 'matching.csv', index=False)
        ledger.to_csv(sub / 'detector_failure_ledger.csv', index=False)
        diagnosis.to_csv(sub / 'diagnosis_failure_ledger.csv', index=False)
        write(sub / 'result.json', to_builtin(result))
    seal(destination, status='COMPLETE', seeds=SEEDS, complete_gt=len(gt),
         prediction_lock_sha256=sha(output / 'global_prediction_lock.json'), test_used_for_selection=False)
    summarize(output)


def summarize(output):
    verify_seal(output / 'evaluation')
    destination = output / 'summary'; destination.mkdir(exist_ok=False)
    rows = []
    for seed in SEEDS:
        result = read(output / 'evaluation' / ('seed-%02d' % seed) / 'result.json')
        if result['seed'] != seed:
            raise ValueError('seed identity mismatch in evaluation')
        selection = result['selection']; diagnosis = result['diagnosis']
        row = dict(seed=seed, selected_epoch=selection['selected_epoch'], epochs_completed=selection['epochs_completed'],
                   threshold=selection['selected_validation_threshold'], rca_n=diagnosis['legal_matched'])
        for prefix, metrics in [('validation_bin', result['validation_bin']), ('test_detector', result['detector'])]:
            for key in ('true_positive_events', 'false_positive_events', 'false_negative_events',
                        'event_precision', 'event_recall', 'event_f1'):
                row[prefix + '_' + key] = metrics[key]
        for key in ('AC@1', 'AC@3', 'AC@5', 'MRR'):
            row['matched_rca_' + key] = diagnosis['matched_rca'][key]
        for k in (1, 3, 5):
            for key, value in diagnosis['e2e']['@' + str(k)].items():
                row['e2e_' + key + '@' + str(k)] = value
        for category, count in diagnosis['failure'].items():
            row['failure_' + category] = count
        rows.append(row)
    table = pd.DataFrame(rows)
    for column in table.columns:
        if column.startswith('failure_'):
            table[column] = table[column].fillna(0).astype(int)
    table.to_csv(destination / 'per_seed.csv', index=False)
    statistics = {}
    for column in table.columns:
        if column == 'seed':
            continue
        values = table[column].dropna().to_numpy(dtype=float)
        statistics[column] = dict(defined_seeds=len(values), total_seeds=10,
            mean=float(values.mean()) if len(values) else None,
            sample_std=float(values.std(ddof=1)) if len(values) > 1 else None,
            minimum=float(values.min()) if len(values) else None, maximum=float(values.max()) if len(values) else None)
    write(destination / 'statistics.json', dict(seeds=SEEDS, statistics=statistics,
        selected_best_seed=None, intervals='no independent-data confidence interval claimed',
        evidence_grade='initialization variability conditional on fixed data/scorer; reused-Test descriptive',
        rca_retrained=False, test_used_for_selection=False))
    seal(destination, status='COMPLETE', seeds=SEEDS)
    print('ALL_TEN_EVALUATED', destination / 'per_seed.csv', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('predict', 'evaluate', 'summarize'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'predict':
        if args.seed is None:
            parser.error('--seed is required for predict')
        predict(args.output_dir.resolve(), args.seed)
    elif args.action == 'evaluate':
        evaluate(args.output_dir.resolve())
    else:
        from scripts.p6.run_two_stage_repeats import check_family
        check_family(args.output_dir.resolve())
        summarize(args.output_dir.resolve())
