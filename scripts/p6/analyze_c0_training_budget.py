#!/usr/bin/env python3
"""Audit fresh paired trajectories, then evaluate budget and one selector on Val."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.p6.analyze_c0_causal_development import paired_table, verify_completion, verify_gt_identity
from scripts.p6.analyze_c0_tcn_replication import temporal_pairs, matching_identity
from src.e2e.bin_trigger_decoder import evaluate_bin_threshold
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config, sha256_file, write_json
from src.e2e.system_trigger import evaluate_system_threshold, onset_density, system_score_frame, to_builtin
from src.e2e.system_trigger_failure_audit import build_failure_ledger, stratified_summary
from src.e2e.budget_prefix_control import prefix_control


def paired_trajectory_gate(control_dir, budget_dir):
    control = json.loads((control_dir / 'epoch_progress.json').read_text())['history']
    budget = json.loads((budget_dir / 'epoch_progress.json').read_text())['history']
    if len(budget) != 30 or len(control) > len(budget):
        raise ValueError('trajectory length drift')
    reports = []
    for index, previous in enumerate(control):
        current = budget[index]
        keys = ('epoch', 'train_loss', 'train_bce', 'train_graph_regularization', 'validation_threshold',
                'validation_candidate_count', 'validation_metrics', 'selection_key', 'learning_rates_used',
                'learning_rates_next_epoch', 'fit_consumed_labels', 'diagnostics')
        same_log = all(previous[key] == current[key] for key in keys)
        same_rng = previous['observer_rng_unchanged'] and current['observer_rng_unchanged']
        peer_files = [directory / 'epochs' / ('epoch-{:02d}'.format(index)) for directory in (control_dir, budget_dir)]
        arrays = [np.load(path / 'validation_outputs.npz', allow_pickle=False) for path in peer_files]
        same_arrays = (set(arrays[0].files) == set(arrays[1].files)
                       and all(np.array_equal(arrays[0][key], arrays[1][key]) for key in arrays[0].files))
        weights = [torch.load(path / 'model_state.pt', map_location='cpu') for path in peer_files]
        same_weights = (weights[0].keys() == weights[1].keys()
                        and all(torch.equal(value, weights[1][key]) for key, value in weights[0].items()))
        record = {'epoch': index, 'logs_exact': same_log, 'arrays_exact': same_arrays,
                  'weights_exact': same_weights, 'rng_unchanged': bool(same_rng)}
        if not all(record[key] for key in ('logs_exact', 'arrays_exact', 'weights_exact', 'rng_unchanged')):
            raise ValueError('fresh peer trajectory mismatch: ' + json.dumps(record))
        reports.append(record)
    return reports


def choose_bin_epoch(history):
    """Only checkpoint key changes; every epoch keeps its merged-derived tau."""
    def key(row):
        metrics = row['diagnostics']['bin_metrics_at_merged_threshold']
        return (metrics['event_f1'], metrics['event_recall'], row['validation_threshold'])
    return max(history, key=key)['epoch']  # max preserves the earliest exact tie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--training-root', required=True, type=Path)
    parser.add_argument('--expected-training-commit', required=True)
    parser.add_argument('--run-suffix', default='deterministic-v1-20261002')
    parser.add_argument('--design', choices=('prefix', 'separate'), default='prefix')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config.read_text())
    baseline_dir = Path(config['baseline_run'])
    verify_completion(baseline_dir)
    base_path = ROOT / config['base_config']
    registry = Path(config['budget']['registry'])
    state = OnsetDevelopmentState(load_config(base_path), Path(config['budget']['data_root']), registry)
    gt = state.gt_events('validation')
    dataset = state.build_dataset('validation')
    if len(gt) != 2901:
        raise ValueError('complete Val denominator drift')
    verify_gt_identity(pd.read_csv(baseline_dir / 'validation_gt.csv'), gt)
    baseline_selection = json.loads((baseline_dir / 'validation_selection.json').read_text())
    baseline_scores = pd.read_csv(baseline_dir / 'validation_predictions.csv', float_precision='round_trip')
    if (not np.array_equal(baseline_scores.sample_index, dataset.sample_indices)
            or not np.array_equal(baseline_scores.prediction_available_time, dataset.prediction_times())):
        raise ValueError('baseline score cohort drift')
    baseline_frame = system_score_frame('validation', baseline_scores.prediction_available_time,
                                        baseline_scores.system_trigger_score)
    _, baseline_matching, baseline_metrics = evaluate_system_threshold(
        baseline_frame, gt, baseline_selection['selected_validation_threshold'])
    for key in ('true_positive_events', 'false_positive_events', 'false_negative_events', 'event_f1'):
        if baseline_metrics[key] != baseline_selection['selected_validation_metrics'][key]:
            raise ValueError('baseline arithmetic replay failed')
    sources = [Path(__file__), args.config, base_path, registry,
               ROOT / 'docs/P6_TCN_BUDGET_SELECTOR_ANALYSIS_PROTOCOL.md',
               ROOT / 'docs/P6_DETERMINISTIC_PAIRED_BUDGET_PROTOCOL.md',
               ROOT / 'src/e2e/event_detection.py', ROOT / 'src/e2e/bin_trigger_decoder.py',
               ROOT / 'src/e2e/system_trigger_failure_audit.py',
               baseline_dir / 'completion_manifest.json', baseline_dir / 'validation_predictions.csv',
               baseline_dir / 'validation_selection.json']
    runs = {}
    peers = {}
    choices = {}
    for seed in (42, 17, 2026):
        runs[seed] = {}
        for arm in ('control', 'budget'):
            directory = args.training_root / ('seed{}-prefix-budget-v1-20261002'.format(seed) if args.design == 'prefix'
                                              else 'seed{}-{}-{}'.format(seed, arm, args.run_suffix))
            completion = verify_completion(directory)
            selection_file = ('prefix_control_selection.json' if args.design == 'prefix' and arm == 'control'
                              else 'validation_selection.json')
            selection = json.loads((directory / selection_file).read_text())
            lock = json.loads((directory / 'development_input_lock.json').read_text())
            if (len(args.expected_training_commit) != 40
                    or completion['execution_commit'] != args.expected_training_commit):
                raise ValueError('unregistered training source')
            declared_arm = ('prefix_control' if arm == 'control' else 'prefix_budget') if args.design == 'prefix' else arm
            if (selection['arm'] != declared_arm or selection['effective_seed'] != seed
                    or selection['protocol_id'] != config['protocol_id']
                    or selection['training'] != dict(config['training'], patience=8 if arm == 'control' else 30)
                    or lock['environment']['deterministic_algorithms'] != (args.design == 'separate')
                    or not json.loads((directory / 'real_window_causality_gate.json').read_text())['passed']):
                raise ValueError('model/recipe/environment/gate drift')
            verify_gt_identity(pd.read_csv(directory / 'validation_gt.csv'), gt)
            runs[seed][arm] = {'directory': directory, 'selection': selection,
                               'history': selection['history'] if arm == 'control' and args.design == 'prefix'
                                          else json.loads((directory / 'epoch_progress.json').read_text())['history']}
            sources += [directory / name for name in ('completion_manifest.json', selection_file,
                                                      'development_input_lock.json', 'epoch_progress.json')]
        if runs[seed]['control']['selection']['model_args'] != runs[seed]['budget']['selection']['model_args']:
            raise ValueError('fresh peer model argument drift')
        if args.design == 'separate':
            peers[str(seed)] = paired_trajectory_gate(runs[seed]['control']['directory'], runs[seed]['budget']['directory'])
        else:
            expected = prefix_control(runs[seed]['budget']['history'])
            selection = runs[seed]['control']['selection']
            if any(expected[key] != selection[key] for key in expected):
                raise ValueError('prefix early-stopping rule did not replay exactly')
            for key in ('checkpoint', 'validation_outputs'):
                if sha256_file(Path(selection[key]['path'])) != selection[key]['sha256']:
                    raise ValueError('prefix selected artifact binding drift')
            peers[str(seed)] = {'design': 'same realised stochastic trajectory prefix',
                                'prefix_rule_exact': True, 'selected_artifacts_bound': True,
                                'independent_control_run': False}
        choices[str(seed)] = {'control': runs[seed]['control']['selection']['selected_epoch'],
                             'budget': runs[seed]['budget']['selection']['selected_epoch'],
                             'selector_candidate': choose_bin_epoch(runs[seed]['budget']['history'])}
        for role, epoch in choices[str(seed)].items():
            arm = 'control' if role == 'control' else 'budget'
            directory = runs[seed][arm]['directory'] / 'epochs' / 'epoch-{:02d}'.format(epoch)
            sources += [directory / 'model_state.pt', directory / 'validation_outputs.npz', directory / 'epoch_audit.json']
    bindings = {str(path.resolve()): sha256_file(path) for path in sources}
    execution_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'analysis_scope_prediction_lock.json', {
        'execution_commit': execution_commit, 'source_sha256': bindings, 'choices': choices,
        'test_read': False, 'trajectory_control_gate': peers, 'validation_reused_for_selection': True})
    baseline_hits = set(baseline_matching.loc[baseline_matching.match_status.eq('matched'), 'case_id'])
    density = onset_density(state.legal_events, origin_ms=state.blocks[0].start_ms).set_index('onset_bin')
    multiplicity = ((gt.start_ms - state.blocks[0].start_ms) // 30000).map(density.onset_count)
    single_ids = gt.loc[multiplicity.eq(1), 'case_id']
    margins = config['development_gate']
    results = {}

    def evaluate(seed, role):
        arm = 'control' if role == 'control' else 'budget'
        epoch = choices[str(seed)][role]
        run = runs[seed][arm]
        directory = run['directory'] / 'epochs' / 'epoch-{:02d}'.format(epoch)
        data = np.load(directory / 'validation_outputs.npz', allow_pickle=False)
        if (not np.array_equal(data['sample_index'], dataset.sample_indices)
                or not np.array_equal(data['prediction_available_time'], dataset.prediction_times())
                or not np.array_equal(data['trigger_label'], dataset.labels_at())):
            raise ValueError('selected output window/time/target mismatch')
        frame = system_score_frame('validation', data['prediction_available_time'], data['system_score'])
        threshold = run['history'][epoch]['validation_threshold']
        episodes, matching, metrics = evaluate_bin_threshold(frame, gt, threshold)
        matching_identity(matching, gt, metrics)
        recorded = run['history'][epoch]['diagnostics']['bin_metrics_at_merged_threshold']
        if any(metrics[key] != recorded[key] for key in ('true_positive_events', 'false_positive_events',
                'false_negative_events', 'event_precision', 'event_recall', 'event_f1')):
            raise ValueError('captured bin integer/metric replay failed')
        hits = set(matching.loc[matching.match_status.eq('matched'), 'case_id'])
        single = paired_table(single_ids, baseline_hits, hits)
        halves = temporal_pairs(gt, baseline_matching, matching, state.blocks[1].start_ms, state.blocks[1].end_ms)
        gate = {'precision_floor': metrics['event_precision'] >= margins['precision_floor'],
                'recall_gain': metrics['event_recall'] - baseline_metrics['event_recall'] >= margins['recall_gain'],
                'f1_gain': metrics['event_f1'] - baseline_metrics['event_f1'] >= margins['f1_gain'],
                'single_onset': single['delta_recall'] >= -margins['single_onset_recall_decline_limit'],
                'first_half': halves['first_half']['delta_recall'] >= .03,
                'second_half': halves['second_half']['delta_recall'] >= .03}
        ledger, invariant = build_failure_ledger(
            gt, split='validation', slot_times_ms=frame.prediction_available_time.to_numpy(),
            slot_scores=frame.system_score.to_numpy(), threshold=threshold,
            episode_anchors_ms=episodes.t_hat.to_numpy(dtype=np.int64),
            episode_end_times_ms=episodes.episode_end_time.to_numpy(dtype=np.int64), matching=matching,
            origin_ms=state.blocks[0].start_ms, split_end_ms=state.blocks[1].end_ms, context_events=state.legal_events)
        if not all(invariant[key] for key in ('categories_sum_to_total', 'matched_equals_tp',
                                             'unmatched_events_equal_fn', 'unmatched_episodes_equal_fp')):
            raise ValueError('failure ledger does not close')
        name = 'seed{}-{}'.format(seed, role)
        episodes.to_csv(args.output_dir / (name + '-episodes.csv'), index=False)
        matching.to_csv(args.output_dir / (name + '-matching.csv'), index=False)
        ledger.to_csv(args.output_dir / (name + '-failure-ledger.csv'), index=False)
        return {'epoch': epoch, 'threshold': threshold, 'metrics': metrics, 'gates': gate,
                'passed': all(gate.values()), 'paired_vs_baseline': paired_table(gt.case_id, baseline_hits, hits),
                'single_onset': single, 'halves': halves, 'failure_invariants': invariant,
                'stratified': stratified_summary(ledger, split='validation')}

    for seed in runs:
        results[str(seed)] = {role: evaluate(seed, role) for role in ('control', 'budget')}
    budget_gate = all(item['budget']['passed'] for item in results.values())
    late_recovery = {str(seed): (choices[str(seed)]['budget'] >= len(runs[seed]['control']['history'])
                     and runs[seed]['budget']['selection']['history'][choices[str(seed)]['budget']]['selection_key']
                     > runs[seed]['control']['selection']['history'][choices[str(seed)]['control']]['selection_key'])
                     for seed in runs}
    if not budget_gate:
        for seed in runs:
            results[str(seed)]['selector_candidate'] = evaluate(seed, 'selector_candidate')
    report = {'execution_commit': execution_commit, 'test_read': False, 'gt_denominator': len(gt),
              'baseline_metrics': baseline_metrics, 'trajectory_control_gate': peers, 'design': args.design, 'choices': choices,
              'results': results, 'budget_all_seed_gate': budget_gate, 'late_recovery': late_recovery,
              'budget_hypothesis_supported': budget_gate and all(late_recovery.values()),
              'selector_activated': not budget_gate,
              'selector_all_seed_gate': all(item.get('selector_candidate', {}).get('passed', False)
                                            for item in results.values()) if not budget_gate else None,
              'full_e2e_run': False, 'validation_reused_for_selection': True}
    write_json(args.output_dir / 'budget_selector_comparison.json', to_builtin(report))
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != execution_commit
            or any(sha256_file(Path(path)) != sha for path, sha in bindings.items())):
        raise ValueError('analysis source/input drift')
    write_json(args.output_dir / 'completion_manifest.json', {'status': 'COMPLETE_DEVELOPMENT_ONLY',
               'execution_commit': execution_commit, 'test_inference_run': False,
               'files': {str(path.relative_to(args.output_dir)): sha256_file(path)
                         for path in args.output_dir.rglob('*') if path.is_file()}})
    print(json.dumps({'budget_gate': budget_gate, 'late_recovery': late_recovery,
                      'selector_gate': report['selector_all_seed_gate']}))


if __name__ == '__main__':
    main()
