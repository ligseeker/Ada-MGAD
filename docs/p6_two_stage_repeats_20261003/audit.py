"""Read-only checks of sealed repeat artifacts; no training or model inference.

Run after check_family and verify_seal have passed. Output goes to a NEW
analysis directory outside the immutable experiment collection.
"""
from pathlib import Path
import argparse
from collections import Counter, deque
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a, b):
    assert np.isclose(a, b, rtol=0, atol=1e-12), (a, b)


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(mean=float(x.mean()), sample_std=float(x.std(ddof=1)),
                minimum=float(x.min()), maximum=float(x.max()))


def capacity(starts, slots):
    """Maximum cardinality for causal equal-width intervals, WITHOUT FP cost.

    This GT-derived capacity diagnostic is not a feasible detector. Each slot
    can serve one GT onset; oldest eligible onset has the earliest deadline.
    """
    starts = sorted(int(x) for x in starts)
    pending = deque()
    pos = hits = 0
    for t in slots:
        while pos < len(starts) and starts[pos] <= t:
            pending.append(starts[pos])
            pos += 1
        while pending and pending[0] + 60000 < t:
            pending.popleft()
        if pending:
            pending.popleft()
            hits += 1
    return hits


def audit(run, out):
    assert not out.exists(), 'Never overwrite an analysis directory'
    scope = read(run / 'scope_lock.json')
    sys.path.insert(0,str(run.parents[3]))
    from src.e2e.system_trigger import trigger_temporal_blocks
    base = read(run.parents[3] / scope['config']['base_config'])
    blocks = trigger_temporal_blocks(base)
    family = read(run / 'global_prediction_lock.json')
    summary = pd.read_csv(run / 'summary/per_seed.csv').set_index('seed')
    statistics = read(run / 'summary/statistics.json')['statistics']
    gt = pd.read_csv(run / 'evaluation/test_gt.csv', keep_default_na=False).set_index('case_id')
    assert len(gt) == 5787 and gt.index.is_unique
    ids = set(gt.index)
    source_root = Path(scope['config']['registry']).parents[0]
    registry = pd.read_csv(source_root / 'gt_event_registry.csv', keep_default_na=False)
    assert registry.case_id.is_unique
    canonical = registry.set_index('case_id').loc[gt.index]
    for k in ('start_ms', 'end_ms', 'service', 'fault_type'):
        assert np.array_equal(gt[k].values, canonical[k].values)
    model_sha = scope['config']['frozen_rca']['model_sha256']
    per_seed = []
    strata = []
    ranks_by_seed = []
    anchors_by_seed = []
    seen_anchors = {}
    comparisons = 0
    known_slots = None
    training_spec = None
    model_spec = None
    validation_ids = None
    runtime_spec = None
    for seed in range(1, 11):
        folder = run / f'seed-{seed:02d}'
        ev = run / f'evaluation/seed-{seed:02d}'
        result = read(ev / 'result.json')
        selection = read(folder / 'detector/validation_selection.json')
        method = read(folder / 'detector/method_record.json')
        rca_method = read(folder / 'rca/method_record.json')
        log = read(folder / 'detector/training_log.json')
        history = selection['history']
        assert history == log['history']
        assert selection['effective_seed'] == selection['model_args']['random_seed'] == seed
        assert selection['training']['max_epochs'] == 30 and selection['training']['patience'] == 8
        current_spec = {k: v for k, v in selection['model_args'].items() if k != 'random_seed'}
        if training_spec is None:
            training_spec = selection['training']; model_spec = current_spec
        assert training_spec == selection['training'] and model_spec == current_spec
        assert method['causality_gate']['passed'] and read(folder / 'detector/validation_replay.json')['passed']
        assert method['cohort'] == {'fit': {'events':7443,'windows':43550},'validation':{'events':2901,'windows':17414}}
        if runtime_spec is None:
            runtime_spec = method['cuda_runtime']
        assert runtime_spec == method['cuda_runtime']
        assert not method['test_label_columns_consumed']
        assert rca_method['scorer_sha256'] == model_sha and not rca_method['scorer_retrained']
        assert rca_method['scaler'] is None and rca_method['anchor_backdate_ms'] == 25621
        assert not result['test_used_for_selection']
        assert result['prediction_lock_sha256'] == sha(run / 'global_prediction_lock.json')
        best_key = None
        best_index = -1
        stagnant = 0
        for i, h in enumerate(history):
            assert h['epoch'] == i
            key = tuple(h['selection_key'])
            assert key == (h['validation_metrics']['event_f1'], h['validation_metrics']['event_recall'], h['validation_threshold'])
            if best_key is None or key > best_key:
                best_key = key; best_index = i; stagnant = 0
            else:
                stagnant += 1
            assert stagnant < 8 or i == len(history)-1
        assert selection['selected_epoch'] == best_index
        assert selection['epochs_completed'] == len(history)
        assert stagnant == 8 and selection['stop_reason'] == 'validation_event_f1_patience'
        tau = selection['selected_validation_threshold']
        same(tau, history[best_index]['validation_threshold'])
        scores = pd.read_csv(folder / 'detector/scores.csv', float_precision='round_trip')
        episodes = pd.read_csv(folder / 'detector/episodes.csv', float_precision='round_trip')
        rankings = pd.read_csv(folder / 'rca/rankings.csv', keep_default_na=False, float_precision='round_trip')
        matching = pd.read_csv(ev / 'matching.csv', keep_default_na=False, float_precision='round_trip')
        ledger = pd.read_csv(ev / 'detector_failure_ledger.csv', keep_default_na=False)
        diagnosis = pd.read_csv(ev / 'diagnosis_failure_ledger.csv', keep_default_na=False)
        features = np.load(folder / 'rca/features.npy', allow_pickle=False)
        valid = np.load(folder / 'rca/valid.npy', allow_pickle=False)
        assert features.shape == (len(episodes), 10, 68)
        assert np.isfinite(features).all() and np.array_equal(valid, rankings.legal.to_numpy())
        assert not np.any(features[~valid])
        assert len(scores) == 26127 and np.array_equal(scores.sample_index, np.arange(len(scores)))
        assert np.isfinite(scores[['system_score','logit']].values).all()
        slots = scores.prediction_available_time.to_numpy(dtype=np.int64)
        if known_slots is None:
            known_slots = slots
        assert np.array_equal(known_slots, slots) and np.all(np.diff(slots) == 30000)
        assert np.array_equal(episodes.t_hat.to_numpy(), slots[scores.system_score.to_numpy() >= tau])
        assert np.array_equal(rankings.prediction_id, episodes.prediction_id)
        assert np.array_equal(rankings.t_hat, episodes.t_hat)
        assert np.array_equal(rankings.anchor_ms, rankings.t_hat - 25621)
        expected_valid = (rankings.anchor_ms-300000 >= blocks[2].start_ms) & (rankings.anchor_ms+300000 <= blocks[2].end_ms)
        assert np.array_equal(valid,expected_valid)
        assert slots[0] == blocks[2].start_ms+300000 and slots[-1] == blocks[2].end_ms
        assert episodes.prediction_id.is_unique
        assert set(matching.loc[matching.match_status != 'miss', 'prediction_id']) == set(episodes.prediction_id)
        truth_rows = matching.loc[matching.match_status != 'false_alarm'].set_index('case_id')
        assert truth_rows.index.is_unique and set(truth_rows.index) == ids
        for k, g in [('gt_service','service'),('fault_type','fault_type')]:
            assert np.array_equal(truth_rows.loc[gt.index,k], gt[g])
        matched = matching.match_status.eq('matched')
        positive = int(matched.sum())
        fp = int(matching.match_status.eq('false_alarm').sum())
        fn = int(matching.match_status.eq('miss').sum())
        delays = matching.loc[matched,'detection_delay_seconds'].astype(float)
        assert delays.between(0,60).all()
        actual_delay = (matching.loc[matched,'t_hat'].astype(float) - matching.loc[matched,'gt_start_ms'].astype(float)) / 1000
        assert np.allclose(delays,actual_delay,rtol=0,atol=1e-9)
        assert (positive,fp,fn) == (result['detector']['true_positive_events'],result['detector']['false_positive_events'],result['detector']['false_negative_events'])
        assert positive+fn == 5787 and positive+fp == len(episodes)
        for key, value in [('true_positive_events',positive),('false_positive_events',fp),('false_negative_events',fn)]:
            same(summary.loc[seed,'test_detector_'+key],value)
        assert capacity(gt.start_ms, slots[scores.system_score >= tau]) == positive
        assert set(ledger.case_id) == ids and ledger.case_id.is_unique
        assert ledger.failure_category.value_counts().to_dict() == {k:v for k,v in result['detector_failure']['category_counts'].items() if v}
        assert int(ledger.failure_category.eq('MATCHED').sum()) == positive
        for i, row in enumerate(rankings.itertuples()):
            if row.legal:
                ranking = json.loads(row.ranking)
                assert len(ranking) == len(set(ranking)) == 10 and set(ranking) == set(gt.service.unique())
                binding = (hashlib.sha256(features[i].tobytes()).hexdigest(), row.ranking)
                if row.anchor_ms in seen_anchors:
                    assert binding == seen_anchors[row.anchor_ms], 'Same anchor has inconsistent features/ranking'
                    comparisons += 1
                else:
                    seen_anchors[row.anchor_ms] = binding
            else:
                assert row.ranking == ''
        prediction_rows = rankings.set_index('prediction_id')
        reconstructed = []
        for row in matching.itertuples():
            rank = 11
            if row.match_status == 'matched':
                pred = prediction_rows.loc[row.prediction_id]
                if pred.legal:
                    rank = json.loads(pred.ranking).index(row.gt_service)+1
            reconstructed.append(rank)
        assert reconstructed == diagnosis.root_rank.tolist()
        assert result['diagnosis']['failure'] == diagnosis.failure_category.value_counts().to_dict()
        assert len(diagnosis) == 5787+fp
        complete = diagnosis.loc[diagnosis.match_status != 'false_alarm'].set_index('case_id').loc[gt.index]
        legal = complete.match_status.eq('matched') & complete.legal.isin([True,'True'])
        n = int(legal.sum())
        legal_ranks = complete.loc[legal,'root_rank'].to_numpy()
        same(np.mean(1/legal_ranks), result['diagnosis']['matched_rca']['MRR'])
        same(np.mean(1/legal_ranks), summary.loc[seed,'matched_rca_MRR'])
        same(summary.loc[seed,'rca_n'],n)
        for k in (1,3,5):
            tp = int(complete.root_rank.le(k).sum())
            e = result['diagnosis']['e2e'][f'@{k}']
            assert (e['tp'],e['fp'],e['fn']) == (tp,len(episodes)-tp,5787-tp)
            for key in ('tp','fp','fn'):
                same(e[key],summary.loc[seed,f'e2e_{key}@{k}'])
            for key,value in [('precision',tp/len(episodes)),('recall',tp/5787),('f1',2*tp/(len(episodes)+5787))]:
                same(e[key],value); same(summary.loc[seed,f'e2e_{key}@{k}'],value)
            same(result['diagnosis']['matched_rca'][f'AC@{k}'],np.mean(legal_ranks <= k))
            same(summary.loc[seed,f'matched_rca_AC@{k}'],np.mean(legal_ranks <= k))
        same(result['detector']['event_precision'],positive/len(episodes))
        same(result['detector']['event_recall'],positive/5787)
        same(result['detector']['event_f1'],2*positive/(len(episodes)+5787))
        for dimension in ('fault_type','service'):
            for label, cases in gt.groupby(dimension):
                group = complete.loc[cases.index]
                det_n = int(group.match_status.eq('matched').sum())
                legal_n = int((group.match_status.eq('matched') & group.legal.isin([True,'True'])).sum())
                tp1 = int(group.root_rank.eq(1).sum())
                strata.append(dict(seed=seed,dimension=dimension,group=label,n=len(group),
                    detected=det_n,legal_matched=legal_n,top1=tp1,top3=int(group.root_rank.le(3).sum()),
                    top5=int(group.root_rank.le(5).sum()),detector_recall=det_n/len(group),e2e_recall_at_1=tp1/len(group),
                    matched_ac_at_1=tp1/legal_n if legal_n else None))
        val = pd.read_csv(folder / 'detector/validation_predictions.csv',float_precision='round_trip')
        alarms = val.system_trigger_score >= tau
        assert int(alarms.sum()) == result['validation_bin']['predicted_episode_count']
        vm = pd.read_csv(folder / 'detector/validation_matching.csv',keep_default_na=False)
        current_validation_ids = set(vm.loc[vm.match_status != 'false_alarm','case_id'])
        if validation_ids is None:
            validation_ids = current_validation_ids
        assert current_validation_ids == validation_ids and len(validation_ids) == 2901
        counts = vm.match_status.value_counts()
        assert (counts.get('matched',0),counts.get('false_alarm',0),counts.get('miss',0)) == tuple(selection['selected_validation_metrics'][k] for k in ('true_positive_events','false_positive_events','false_negative_events'))
        categories = ledger.failure_category.value_counts().to_dict()
        for key in summary:
            if key.startswith('failure_'):
                same(summary.loc[seed,key],result['diagnosis']['failure'].get(key[len('failure_'):],0))
        rank_errors = int((complete.match_status.eq('matched') & complete.root_rank.between(2,10)).sum())
        anchor_err = (complete.loc[legal,'anchor_ms'].astype(float)-complete.loc[legal,'gt_start_ms'].astype(float))/1000
        per_seed.append(dict(seed=seed,selected_epoch=best_index,epochs_completed=len(history),
            train_loss_first=history[0]['train_loss'],train_loss_last=history[-1]['train_loss'],
            val_merged_f1=selection['selected_validation_metrics']['event_f1'],
            val_bin_alarms=int(alarms.sum()),val_merged_alarms=len(vm.loc[vm.match_status != 'miss']),
            val_positive_bins_by_label={str(k):int(v) for k,v in val.loc[alarms,'trigger_label'].value_counts().items()},
            val_bin_alarms_first_half=int(alarms.iloc[:len(val)//2].sum()),val_bin_alarms_second_half=int(alarms.iloc[len(val)//2:].sum()),
            detector_categories=categories,top1_ranking_errors=rank_errors,
            invalid_matched=positive-n,invalid_predictions=int((~valid).sum()),legal_matched=n,
            root_macro_recall_at_1=result['diagnosis']['macro_e2e_recall']['gt_service']['mean']['1'],
            type_macro_recall_at_1=result['diagnosis']['macro_e2e_recall']['fault_type']['mean']['1'],
            detected_mean_delay_seconds=float(delays.mean()),anchor_error_mean_seconds=float(anchor_err.mean()),
            anchor_error_mean_absolute_seconds=float(anchor_err.abs().mean()),
            anchor_error_abs_gt30s=int(anchor_err.abs().gt(30).sum()),
            cuda_runtime=method['cuda_runtime']))
        ranks_by_seed.append(complete.root_rank.rename(seed))
        anchors_by_seed.append(pd.to_numeric(complete.t_hat,errors='coerce').rename(seed))
    for column in summary:
        values = summary[column].dropna().to_numpy(float)
        assert statistics[column]['defined_seeds'] == len(values) == 10
        for field,value in describe(values).items():
            same(statistics[column][field],value)
    rmat = pd.concat(ranks_by_seed,axis=1)
    amat = pd.concat(anchors_by_seed,axis=1)
    common = rmat.le(10).all(axis=1)
    common_anchors = amat.loc[common].nunique(axis=1)
    capacity_n = capacity(gt.start_ms,known_slots)
    union_detected = amat.notna().any(axis=1)
    all_detected = amat.notna().all(axis=1)
    all_top1 = rmat.eq(1).all(axis=1)
    never_top1 = rmat.ne(1).all(axis=1)
    stratum_table = pd.DataFrame(strata)
    means = stratum_table.groupby(['dimension','group'],sort=True).agg(n=('n','first'),
        detected_mean=('detected','mean'),detected_min=('detected','min'),detected_max=('detected','max'),
        top1_mean=('top1','mean'),top1_min=('top1','min'),top1_max=('top1','max'),
        top3_mean=('top3','mean'),top5_mean=('top5','mean'),detector_recall_mean=('detector_recall','mean'),
        e2e_recall_at_1_mean=('e2e_recall_at_1','mean')).reset_index()
    lock_commit = subprocess.check_output(['git','log','-1','--format=%H','--',str(run/'global_prediction_lock.json')],text=True).strip()
    lock_time = int(subprocess.check_output(['git','show','-s','--format=%ct',lock_commit],text=True).strip())
    prediction_paths = [run/p for p in family['files']]
    assert max(p.stat().st_mtime for p in prediction_paths) <= lock_time
    assert (run/'global_prediction_lock.json').stat().st_mtime <= lock_time < (run/'evaluation/test_gt.csv').stat().st_mtime
    aggregate = dict(status='PASS_ARTIFACT_INTEGRITY_AND_ARITHMETIC',experiment_reproduction=False,
        analysis_script_sha256=sha(Path(__file__)),
        run=str(run),execution_commit=scope['execution_commit'],prediction_lock_commit=lock_commit,
        scope_lock_sha256=sha(run/'scope_lock.json'),prediction_lock_sha256=sha(run/'global_prediction_lock.json'),
        evaluation_manifest_sha256=sha(run/'evaluation/completion_manifest.json'),summary_manifest_sha256=sha(run/'summary/completion_manifest.json'),
        source_files=len(scope['source_sha256']),input_files=len(scope['inputs']),sealed_stages=20,seeds=list(range(1,11)),
        complete_gt=5787,test_windows=26127,gt_identity_matches_registry=True,
        same_anchor_unique_count=len(seen_anchors),same_anchor_duplicate_comparisons=comparisons,
        same_anchor_features_and_rankings_identical=True,per_seed=per_seed,
        detector_categories={k:describe([r['detector_categories'].get(k,0) for r in per_seed]) for k in ('MATCHED','NO_LEGAL_PREDICTION','BELOW_THRESHOLD','MATCHING_COMPETITION','NO_NEW_EPISODE')},
        failure_mean={k:describe([r[k] for r in per_seed]) for k in ('top1_ranking_errors','invalid_matched','invalid_predictions')},
        macro_root_recall_at_1=describe([r['root_macro_recall_at_1'] for r in per_seed]),
        macro_type_recall_at_1=describe([r['type_macro_recall_at_1'] for r in per_seed]),
        common_legal_matched=dict(n=int(common.sum()),same_anchor_all_ten=int(common_anchors.eq(1).sum()),
            shifted_anchor_across_seeds=int(common_anchors.gt(1).sum()),
            ac_at_1_by_seed={str(i):float(rmat.loc[common,i].eq(1).mean()) for i in range(1,11)}),
        case_stability=dict(always_detected=int(all_detected.sum()),detected_by_any_seed=int(union_detected.sum()),
            never_detected=int((~union_detected).sum()),always_top1=int(all_top1.sum()),never_top1=int(never_top1.sum())),
        structural_capacity=dict(maximum_detectable_with_all_grid_slots=capacity_n,
            minimum_FN_grid_and_one_to_one=5787-capacity_n,maximum_recall=capacity_n/5787,
            note='GT-derived optimistic capacity, unlimited FP; not achievable model performance; competition is not automatically structural'),
        data_skew=dict(root_counts=gt.service.value_counts().to_dict(),fault_counts=gt.fault_type.value_counts().to_dict()),
        observed_cuda=per_seed[0]['cuda_runtime'])
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n')
    means.to_csv(out/'strata_mean.csv',index=False)
    stratum_table.to_csv(out/'strata_per_seed.csv',index=False)
    print(json.dumps({k:v for k,v in aggregate.items() if k not in ('per_seed','data_skew')},ensure_ascii=False,indent=2))
    print('STRATA_MEAN\n'+means.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audit(args.run.resolve(),args.output.resolve())
