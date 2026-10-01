#!/usr/bin/env python3
"""Render completed Train/Validation budget records; no model or data loading."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', required=True, type=Path)
    parser.add_argument('--training-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    complete = json.loads((args.analysis_dir / 'completion_manifest.json').read_text())
    assert complete['status'] == 'COMPLETE_DEVELOPMENT_ONLY'
    for name, digest in complete['files'].items():
        if sha(args.analysis_dir / name) != digest:
            raise ValueError('sealed analysis artifact changed: ' + name)
    comparison = json.loads((args.analysis_dir / 'budget_selector_comparison.json').read_text())
    args.output_dir.mkdir(parents=True)
    rows, metric_rows, bindings = [], [], {}
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    for row_index, seed in enumerate((42, 17, 2026)):
        run = args.training_root / ('seed{}-prefix-budget-v1-20261002'.format(seed))
        history_path = run / 'epoch_progress.json'
        control_path = run / 'prefix_control_selection.json'
        training_completion_path = run / 'completion_manifest.json'
        training_completion = json.loads(training_completion_path.read_text())
        if training_completion['status'] != 'COMPLETE_DEVELOPMENT_ONLY':
            raise ValueError('training has not completed')
        for path in (history_path, control_path):
            if sha(path) != training_completion['files'][path.name]:
                raise ValueError('sealed training record changed')
        history = json.loads(history_path.read_text())['history']
        control = json.loads(control_path.read_text())
        if len(history) != 30:
            raise ValueError('incomplete trajectory')
        bindings[str(history_path.resolve())] = sha(history_path)
        bindings[str(control_path.resolve())] = sha(control_path)
        bindings[str(training_completion_path.resolve())] = sha(training_completion_path)
        for item in history:
            merged = item['validation_metrics']
            bins = item['diagnostics']['bin_metrics_at_merged_threshold']
            rows.append(dict(seed=seed, epoch_index=item['epoch'], train_loss=item['train_loss'],
                             learning_rate=item['learning_rates_used'][0], threshold=item['validation_threshold'],
                             merged_precision=merged['event_precision'], merged_recall=merged['event_recall'],
                             merged_f1=merged['event_f1'], bin_precision=bins['event_precision'],
                             bin_recall=bins['event_recall'], bin_f1=bins['event_f1'],
                             bin_tp=bins['true_positive_events'], bin_fp=bins['false_positive_events'],
                             bin_fn=bins['false_negative_events'],
                             neg_high_fraction=item['diagnostics']['high_score_by_target']['NEG']['high_fraction']))
        x = [item['epoch'] for item in history]
        axes[row_index, 0].plot(x, [item['train_loss'] for item in history], color='tab:blue')
        axes[row_index, 1].plot(x, [item['validation_metrics']['event_f1'] for item in history], color='tab:orange')
        for key, color, label in (('event_precision', 'tab:green', 'P'), ('event_recall', 'tab:purple', 'R'),
                                  ('event_f1', 'tab:red', 'F1')):
            axes[row_index, 2].plot(x, [item['diagnostics']['bin_metrics_at_merged_threshold'][key]
                                      for item in history], color=color, label=label)
        axes[row_index, 3].plot(x, [item['validation_threshold'] for item in history], color='tab:brown')
        for axis in axes[row_index]:
            axis.axvspan(control['epochs_completed'] - .5, 29.5, color='grey', alpha=.12)
            axis.axvline(10, linestyle=':', color='grey', linewidth=1)
            axis.axvline(20, linestyle=':', color='grey', linewidth=1)
            axis.grid(alpha=.2)
            axis.set_xlim(-.5, 29.5)
        for column in (1, 2, 3):
            axes[row_index, column].set_ylim(0, 1.03)
        axes[row_index, 0].set_ylabel('seed {}\nTrain loss'.format(seed))
        for role, result in comparison['results'][str(seed)].items():
            metrics = result['metrics']
            metric_rows.append(dict(seed=seed, role=role, selected_epoch_index=result['epoch'],
                                    threshold=result['threshold'], tp=metrics['true_positive_events'],
                                    fp=metrics['false_positive_events'], fn=metrics['false_negative_events'],
                                    precision=metrics['event_precision'], recall=metrics['event_recall'],
                                    f1=metrics['event_f1'], passed=result['passed'],
                                    failed_gates=','.join(key for key, passed in result['gates'].items() if not passed)))
        selected = comparison['choices'][str(seed)]['budget']
        axes[row_index, 1].scatter([selected], [history[selected]['validation_metrics']['event_f1']],
                                  color='black', marker='x', s=50, zorder=5)
        if comparison['selector_activated']:
            selected = comparison['choices'][str(seed)]['selector_candidate']
            axes[row_index, 2].scatter([selected],
                [history[selected]['diagnostics']['bin_metrics_at_merged_threshold']['event_f1']],
                color='black', marker='x', s=50, zorder=5)
    for axis, title in zip(axes[0], ('Training loss', 'Validation merged F1', 'Validation bin metrics', 'Merged-derived threshold')):
        axis.set_title(title)
    axes[0, 2].legend(loc='lower right', fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Epoch index (0-based)')
    fig.suptitle('Fixed 30-epoch trajectories: grey = beyond patience-8 stop; dotted = lower LR; x = selected epoch')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output_dir / 'training_budget_curves.png', dpi=160)
    fig.savefig(args.output_dir / 'training_budget_curves.svg')
    plt.close(fig)
    for name, values in (('epoch_history.csv', rows), ('selected_bin_results.csv', metric_rows)):
        with (args.output_dir / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    bindings[str((args.analysis_dir / 'completion_manifest.json').resolve())] = sha(args.analysis_dir / 'completion_manifest.json')
    bindings[str(Path(__file__).resolve())] = sha(Path(__file__))
    (args.output_dir / 'render_manifest.json').write_text(json.dumps(dict(
        status='COMPLETE_REPORT_RENDER', test_read=False, inputs_sha256=bindings,
        files_sha256={p.name: sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}), indent=2) + '\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'epoch_rows': len(rows), 'result_rows': len(metric_rows)}))


if __name__ == '__main__':
    main()
