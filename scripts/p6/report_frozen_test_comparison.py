#!/usr/bin/env python3
"""Audit completed frozen outputs and export descriptive cross-worktree tables."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

BASE = Path('/home/zhangll24/RCA_project')
MAIN = BASE / 'Ada-MGAD-e2e-v2'
DET = BASE / 'Ada-MGAD-e2e-v2-testreview'
RCA = BASE / 'Ada-MGAD-e2e-v2-rcatestreview'
D = DET / 'experiments/p6/frozen_test_review/frozen-candidates-v1-20261001'
R = RCA / 'experiments/p6/frozen_rca_test_review/frozen-candidates-v1-20261002'
C1 = MAIN / 'experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42'
SERVICES = ('dbservice1', 'dbservice2', 'logservice1', 'logservice2', 'mobservice1',
            'mobservice2', 'redisservice1', 'redisservice2', 'webservice1', 'webservice2')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(root, manifest):
    for name, digest in manifest['files'].items():
        if sha(root / name) != digest:
            raise ValueError('completed output drift: ' + str(root / name))


def same(a, b):
    if not np.isclose(a, b, rtol=0., atol=1e-12):
        raise ValueError('independent arithmetic differs: {} {}'.format(a, b))


def training_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = {
        42: BASE / 'Ada-MGAD-e2e-v2-c0onsettcn/experiments/p6/c0_onset_tcn/tcn-v1-seed42',
        17: BASE / 'Ada-MGAD-e2e-v2-c0replica/experiments/p6/c0_onset_tcn_replication/seed17-v1',
        2026: BASE / 'Ada-MGAD-e2e-v2-c0replica/experiments/p6/c0_onset_tcn_replication/seed2026-v1',
    }
    rows, summary, receipts = [], [], {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for seed, run in runs.items():
        history = read(run / 'training_log.json')['history']
        selection = read(run / 'validation_selection.json')
        for filename in ('training_log.json', 'validation_selection.json'):
            receipts[str(run / filename)] = sha(run / filename)
        selected = selection['selected_epoch']
        epochs = [v['epoch'] for v in history]
        losses = [v['train_loss'] for v in history]
        f1 = [v['validation_metrics']['event_f1'] for v in history]
        summary.append(dict(seed=seed, selected_epoch=selected, epochs_completed=len(history),
            stop_reason=selection['stop_reason'], selected_train_loss=losses[selected],
            last_train_loss=losses[-1], selected_validation_f1=f1[selected], last_validation_f1=f1[-1]))
        for entry in history:
            rows.append(dict(seed=seed, epoch=entry['epoch'], train_loss=entry['train_loss'],
                train_bce=entry['train_bce'], validation_merged_f1=entry['validation_metrics']['event_f1'],
                threshold=entry['validation_threshold'], selected=entry['epoch']==selected))
        axes[0].plot(epochs, losses, marker='.', label='seed '+str(seed))
        axes[1].plot(epochs, f1, marker='.', label='seed '+str(seed))
        axes[1].scatter([selected], [f1[selected]], marker='*', s=120)
    for ax in axes:
        ax.set_xlabel('Epoch (zero based)'); ax.set_xticks(range(11)); ax.grid(alpha=.2)
        ax.axvline(10, color='grey', linestyle='--', alpha=.6)
    axes[0].set_ylabel('Train batch-averaged loss (BCE + graph)')
    axes[1].set_ylabel('Validation merged event F1')
    axes[0].legend(); axes[1].set_title('Stars: selected checkpoint; lower LR starts at 10', fontsize=9)
    fig.savefig(output / 'tcn_training_curves.png', dpi=180)
    fig.savefig(output / 'tcn_training_curves.svg')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'tcn_epoch_history.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'tcn_training_summary.csv', index=False)
    return receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    # Both complete evaluations must exist before this report reads labels.
    dm = read(D / 'evaluation/completion_manifest.json')
    rm = read(R / 'evaluation/completion_manifest.json')
    if dm['status'] != 'COMPLETE' or rm['status'] != 'COMPLETE':
        raise ValueError('the full comparison is not complete')
    config = read(RCA / 'configs/e2e/gaia_p6_frozen_rca_test_review_v1.json')
    ds, rs = read(D / 'scope_lock.json'), read(R / 'scope_lock.json')
    for tree, scope in [(DET, ds), (RCA, rs)]:
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
        if actual != scope['execution_commit']:
            raise ValueError('execution HEAD drift')
    dl, rl = read(D / 'global_prediction_lock.json'), read(R / 'predictions/global_prediction_lock.json')
    if len(dl['arms']) != 19 or len(rl['arms']) != 40 or len(ds['config']['models']) != 10:
        raise ValueError('family scope incomplete')
    if sha(D / 'global_prediction_lock.json') != config['detector_prediction_lock_sha256']:
        raise ValueError('independent detector prediction receipt drift')
    for path, digest in config['detector_model_manifest_sha256'].items():
        if sha(path) != digest:
            raise ValueError('detector execution receipt drift')
    verify_files(D / 'evaluation', dm)
    verify_files(R / 'evaluation', rm)
    verify_files(R / 'predictions', rl)
    dr, rr = read(D / 'evaluation/results.json'), read(R / 'evaluation/results.json')
    audit = read(Path(config['detector_audit_dir']) / 'audit.json')
    if audit['status'] != 'PASS' or audit['evaluation_manifest_sha256'] != sha(D / 'evaluation/completion_manifest.json'):
        raise ValueError('detector independent audit drift')
    if rr['prediction_lock_sha256'] != sha(R / 'predictions/global_prediction_lock.json'):
        raise ValueError('RCA prediction receipt drift')
    gt = pd.read_csv(D / 'evaluation/test_gt.csv')
    if len(gt) != 5787 or not gt.case_id.is_unique:
        raise ValueError('GT denominator drift')
    a.output_dir.mkdir(parents=True)
    detector_rows, family_rows, groups, failures, decompositions = [], [], [], [], []
    ledgers = {}
    for arm in dl['arms']:
        m = dr['results'][arm['id']]['metrics']
        matching = pd.read_csv(D / 'evaluation' / arm['id'] / 'matching.csv')
        tp = int(matching.match_status.eq('matched').sum())
        fp = int(matching.match_status.eq('false_alarm').sum())
        fn = int(matching.match_status.eq('miss').sum())
        if tp + fn != 5787 or tp != m['true_positive_events'] or fp != m['false_positive_events'] or fn != m['false_negative_events']:
            raise ValueError('detector integer mismatch')
        same(2 * tp / (5787 + tp + fp), m['event_f1'])
        row = dict(arm_id=arm['id'], graph_scope=arm['graph_scope'], decoder=arm['decoder'],
            threshold=arm['threshold'], TP=tp, FP=fp, FN=fn, P=m['event_precision'],
            R=m['event_recall'], F1=m['event_f1'], Val_F1=arm['validation_metrics']['event_f1'])
        detector_rows.append(row)
        combined = dict(row)
        for scorer in ('original', 'backdate'):
            value = rr['results'][arm['id'] + '__' + scorer]
            combined[scorer + '_legal_n'] = value['legal_matched']
            combined[scorer + '_AC1'] = value['matched_rca']['AC@1']
            for k in (1, 3, 5):
                combined[scorer + '_E2E_F1_' + str(k)] = value['e2e']['@' + str(k)]['f1']
        family_rows.append(combined)
    for arm in rl['arms']:
        aid = arm['id']; value = rr['results'][aid]
        path = Path(config['inputs']['native_matching']) if arm['native'] else D / 'evaluation' / arm['detector_arm'] / 'matching.csv'
        matching = pd.read_csv(path, float_precision='round_trip')
        frame = pd.read_csv(R / 'predictions' / (aid + '.csv'), keep_default_na=False)
        observed = matching.loc[matching.match_status.ne('false_alarm')]
        if len(observed) != 5787 or set(observed.case_id) != set(gt.case_id):
            raise ValueError('RCA GT identity mismatch')
        nonmiss = matching.loc[matching.match_status.ne('miss')]
        if not frame.prediction_id.is_unique or nonmiss.prediction_id.duplicated().any() or set(frame.prediction_id) != set(nonmiss.prediction_id):
            raise ValueError('prediction population mismatch')
        joined = matching.merge(frame, on='prediction_id', how='left', validate='many_to_one', suffixes=('', '_rank'))
        ranks = []
        for record in joined.itertuples():
            rank = 11
            if record.match_status == 'matched' and record.legal and record.ranking:
                candidates = json.loads(record.ranking)
                if len(candidates) != 10 or len(set(candidates)) != 10 or set(candidates) != set(SERVICES):
                    raise ValueError('ranking contract mismatch')
                rank = candidates.index(record.gt_service) + 1
            ranks.append(rank)
        joined['rank'] = ranks
        legal = joined.match_status.eq('matched') & joined.legal.eq(True)
        n = int(legal.sum())
        if n != value['legal_matched']:
            raise ValueError('legal matched denominator mismatch')
        for k in (1, 3, 5):
            tp = int(joined['rank'].le(k).sum()); v = value['e2e']['@' + str(k)]
            if (tp, len(frame)-tp, 5787-tp) != (v['tp'], v['fp'], v['fn']):
                raise ValueError('E2E integer mismatch')
            same(v['f1'], 2 * tp / (len(frame) + 5787))
            same(value['matched_rca']['AC@' + str(k)], joined.loc[legal, 'rank'].le(k).sum()/n)
        same(value['matched_rca']['MRR'], np.where(joined.loc[legal, 'rank'] <= 10, 1./joined.loc[legal, 'rank'], 0).mean())
        disk = pd.read_csv(R / 'evaluation' / (aid + '_failure.csv'))
        if not np.array_equal(disk.root_rank.to_numpy(), joined['rank'].to_numpy()):
            raise ValueError('failure ledger root-rank replay mismatch')
        counts = disk.failure_category.value_counts().to_dict()
        if counts != value['failure'] or len(disk) != 5787 + int(matching.match_status.eq('false_alarm').sum()):
            raise ValueError('failure ledger closure mismatch')
        failures.append(dict(arm_id=aid, **counts))
        ledgers[aid] = joined.loc[joined.match_status.ne('false_alarm')].set_index('case_id').reindex(gt.case_id)
        for dimension in ('gt_service', 'fault_type'):
            for label, group in joined.loc[joined.match_status.ne('false_alarm')].groupby(dimension):
                matched_group = group.loc[group.match_status.eq('matched') & group.legal.eq(True)]
                groups.append(dict(arm_id=aid, dimension=dimension, group=label, complete_gt=len(group),
                    event_matched=int(group.match_status.eq('matched').sum()), legal_matched=len(matched_group),
                    E2E_TP1=int(group['rank'].eq(1).sum()), E2E_R1=float(group['rank'].eq(1).mean()),
                    AC1=float(matched_group['rank'].eq(1).mean()) if len(matched_group) else None))
    # Decompose all-event Top-1 change versus causal C0; cohorts and RCA anchors may change.
    for scorer in ('original', 'backdate'):
        baseline = ledgers['causal_merged__' + scorer]
        for arm in dl['arms']:
            candidate = ledgers[arm['id'] + '__' + scorer]
            old_match = baseline.match_status.eq('matched'); new_match = candidate.match_status.eq('matched')
            old_ok = baseline['rank'].eq(1); new_ok = candidate['rank'].eq(1)
            common = old_match & new_match
            record = dict(arm_id=arm['id'], scorer=scorer, common_matched=int(common.sum()),
                newly_matched=int((new_match & ~old_match).sum()), lost_matches=int((old_match & ~new_match).sum()),
                new_match_top1=int((new_ok & ~old_match).sum()), lost_match_top1=int((old_ok & ~new_match).sum()),
                common_top1_gain=int((common & new_ok & ~old_ok).sum()),
                common_top1_loss=int((common & old_ok & ~new_ok).sum()),
                total_top1_change=int(new_ok.sum()-old_ok.sum()))
            if record['total_top1_change'] != record['new_match_top1']-record['lost_match_top1']+record['common_top1_gain']-record['common_top1_loss']:
                raise ValueError('stage attribution does not close')
            decompositions.append(record)
    pd.DataFrame(detector_rows).to_csv(a.output_dir / 'detector_19.csv', index=False)
    pd.DataFrame(family_rows).to_csv(a.output_dir / 'combined_19.csv', index=False)
    pd.DataFrame(failures).fillna(0).to_csv(a.output_dir / 'failure_40.csv', index=False)
    pd.DataFrame(groups).to_csv(a.output_dir / 'strata_40.csv', index=False)
    pd.DataFrame(decompositions).to_csv(a.output_dir / 'stage_attribution_38.csv', index=False)
    for name, source in [('detector_results.json', D / 'evaluation/results.json'),
                         ('rca_results.json', R / 'evaluation/results.json'),
                         ('rca_e2e_40.csv', R / 'evaluation/rca_e2e_results.csv')]:
        (a.output_dir / name).write_bytes(source.read_bytes())
    receipts = [D/'scope_lock.json', D/'global_prediction_lock.json', D/'evaluation/completion_manifest.json',
        R/'scope_lock.json', R/'predictions/global_prediction_lock.json', R/'evaluation/completion_manifest.json',
        Path(config['detector_audit_dir'])/'audit.json', RCA/'configs/e2e/gaia_p6_frozen_rca_test_review_v1.json']
    train_receipts = training_curves(a.output_dir)
    result = dict(status='PASS', independent_detector_integer_replays=19,
        independent_rca_ranking_metric_ledger_replays=40, full_gt=5787,
        source_receipts={str(path):sha(path) for path in receipts}, train_log_receipts=train_receipts,
        detector_execution_commit=ds['execution_commit'], rca_execution_commit=rs['execution_commit'],
        report_generator_sha256=sha(Path(__file__)), reused_test=True, test_selection_performed=False)
    (a.output_dir/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(family_rows).to_string(index=False))
    print('ABLATIONS', json.dumps({k:v for k,v in rr['results'].items() if k.startswith('native__')}, indent=2))


if __name__ == '__main__':
    main()
