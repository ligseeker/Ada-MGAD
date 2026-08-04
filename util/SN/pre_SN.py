import argparse
import glob
import json
import logging
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.SN.constant import (
    SN_SERVICES, SN_SERVICE2NID, SN_NUM_NODES, SN_METRICS,
    FAULT_CONTAINER_TO_SERVICE, SN_EDGES,
)

DEFAULT_RAW_FAULT_DIR = "/home/zhangll24/RCA_project/Eadro/SN_Dataset/SN Dataset/data"
DEFAULT_RAW_NOFAULT_DIR = "/home/zhangll24/RCA_project/Eadro/SN_Dataset/SN Dataset/no fault"
DEFAULT_SAVE_PATH = "./data/SN-pre"

LOG_TS_PATTERN = re.compile(r'\[(\d{4})-(\w{3})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)\]')
MONTH_MAP = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def parse_log_timestamp(line):
    match = LOG_TS_PATTERN.match(line)
    if not match:
        return None
    year, mon, day, h, m, s, _ = match.groups()
    dt_str = f"{year}-{MONTH_MAP[mon]:02d}-{day} {h}:{m}:{s}"
    return time.mktime(time.strptime(dt_str, '%Y-%m-%d %H:%M:%S'))


def find_fault_json(exp_dir):
    exp_name = os.path.basename(exp_dir)
    fault_name = exp_name.replace('SN.', 'SN.fault-') + '.json'
    for base in (exp_dir, os.path.dirname(exp_dir)):
        candidate = os.path.join(base, fault_name)
        if os.path.exists(candidate):
            return candidate
    return None


def load_metric_frame(exp_dir):
    frames = []
    for service in SN_SERVICES:
        csv_path = os.path.join(exp_dir, 'metrics', f'{service}.csv')
        df = pd.read_csv(csv_path)
        rename_map = {kpi: f'{service}_{kpi}' for kpi in SN_METRICS if kpi in df.columns}
        assert len(rename_map) == len(SN_METRICS), f"missing KPI columns in {csv_path}"
        frames.append(df.rename(columns=rename_map)[['timestamp'] + list(rename_map.values())].set_index('timestamp'))
    metric = pd.concat(frames, axis=1).sort_index()
    t_start, t_end = int(metric.index.min()), int(metric.index.max())
    metric = metric.reindex(range(t_start, t_end + 1)).ffill().fillna(0.0)
    return metric, t_start, t_end


def robust_zscore(frame):
    medians = frame.median()
    mads = frame.sub(medians).abs().median()
    flat = mads < 1e-9
    normalized = frame.sub(medians).div(1.4826 * mads.where(~flat, 1.0))
    normalized.loc[:, flat] = 0.0
    return normalized.fillna(0.0)


def build_labels(fault_data, t_start, T):
    label = np.zeros((T, SN_NUM_NODES))
    for fault in fault_data.get('faults', []):
        service = FAULT_CONTAINER_TO_SERVICE.get(fault['name'])
        if service is None or service not in SN_SERVICE2NID:
            logging.warning("label skipped for unknown container: %s", fault['name'])
            continue
        rel_start = max(0, int(round(fault['start'])) - t_start)
        rel_end = min(T, rel_start + int(fault['duration']))
        label[rel_start:rel_end, SN_SERVICE2NID[service]] = 1
    return label


def fault_windows(fault_data, t_start, T):
    windows = []
    for fault in fault_data.get('faults', []):
        rel_start = int(round(fault['start'])) - t_start
        windows.append((max(0, rel_start), min(T, rel_start + int(fault['duration'])), fault))
    return windows


def load_spans(exp_dir):
    spans_path = os.path.join(exp_dir, 'spans.json')
    if not os.path.exists(spans_path):
        logging.warning("spans.json not found in %s", exp_dir)
        return None
    try:
        with open(spans_path) as fh:
            traces_raw = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logging.warning("corrupted spans.json in %s (%s), trace features zeroed", exp_dir, exc)
        return None

    records = []
    for trace in traces_raw:
        processes = trace.get('processes', {})
        span_service = {}
        for span in trace['spans']:
            pid = span.get('processID', '')
            if pid in processes:
                span_service[span['spanID']] = processes[pid].get('serviceName', 'unknown')
        for span in trace['spans']:
            service = span_service.get(span['spanID'], 'unknown')
            if service not in SN_SERVICE2NID:
                continue
            parent_service = 'unknown'
            for ref in span.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    parent_service = span_service.get(ref.get('spanID', ''), 'unknown')
                    break
            start_sec = span['startTime'] / 1e6
            duration_sec = span['duration'] / 1e6
            records.append({
                'cmbd_id': service,
                'fatherpod': parent_service,
                'stats': span.get('operationName', 'unknown'),
                'start_sec': start_sec,
                'end_sec': start_sec + duration_sec,
                'duration': duration_sec,
            })
    return records


def _peak_offset(A, B, trace_min, t_start):
    peak = int(np.argmax(np.correlate(A, B, 'full')))
    k = peak - (len(B) - 1)
    n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
    if n_hi - n_lo < 30:
        return None
    a_seg, b_seg = A[k + n_lo:k + n_hi], B[n_lo:n_hi]
    if a_seg.std() < 1e-12 or b_seg.std() < 1e-12:
        return None
    peak_corr = float(np.corrcoef(a_seg, b_seg)[0, 1])
    if peak_corr <= 0.1:
        return None
    return k + int(np.floor(trace_min)) - t_start


def estimate_trace_offset(span_df, windows, T, trace_min, t_start, log_secs):
    """Estimate offset = trace_clock - metric_clock for this experiment.

    Stage 1: cross-correlate per-second trace duration mass against the fault
    indicator. Stage 2 (no usable fault signal): cross-correlate per-second
    span counts against the log rate, needed for no-fault experiments whose
    Jaeger collection windows extend far beyond the experiment itself.
    """
    start_secs = span_df['start_sec'].values
    bins = np.floor(start_secs).astype(np.int64) - int(np.floor(trace_min))
    span_len = int(np.floor(start_secs.max())) - int(np.floor(trace_min)) + 1

    if windows:
        A = np.zeros(span_len)
        np.add.at(A, bins, span_df['duration'].values)
        B = np.zeros(T)
        for s, e, _ in windows:
            B[s:e] = 1.0
        offset = _peak_offset(A, B, trace_min, t_start)
        if offset is not None:
            return offset, 'corr'

    if log_secs is not None and len(log_secs):
        A = np.zeros(span_len)
        np.add.at(A, bins, 1)
        L = np.zeros(T)
        np.add.at(L, log_secs, 1)
        if L.sum() > 0:
            offset = _peak_offset(A, L, trace_min, t_start)
            if offset is not None:
                return offset, 'corr(log)'

    return None, 'fallback'


def process_experiment(exp_dir, miner, concat_offset):
    exp_name = os.path.basename(exp_dir)
    logging.info("processing experiment %s", exp_name)

    metric, t_start, t_end = load_metric_frame(exp_dir)
    T = t_end - t_start + 1
    metric = robust_zscore(metric)

    fault_path = find_fault_json(exp_dir)
    fault_data = json.load(open(fault_path)) if fault_path else {'start': t_start, 'faults': []}
    windows = fault_windows(fault_data, t_start, T)
    label = build_labels(fault_data, t_start, T)
    has_fault = len(fault_data.get('faults', [])) > 0

    logs_path = os.path.join(exp_dir, 'logs.json')
    log_rows, log_secs = [], []
    with open(logs_path) as fh:
        logs_raw = json.load(fh)
    for service in SN_SERVICES:
        for line in logs_raw.get(service, []):
            ts = parse_log_timestamp(line)
            if ts is None:
                continue
            rel = int(ts) - t_start
            if rel < 0 or rel >= T:
                continue
            log_secs.append(rel)
            cluster_id = miner.add_log_message(line)['cluster_id']
            log_rows.append((cluster_id, service, rel + concat_offset))
    log_secs = np.asarray(log_secs, dtype=np.int64)

    span_records = load_spans(exp_dir)
    diag = {'name': exp_name, 'T': T, 'offset': None, 'method': 'none',
            'in_range_ratio': 0.0, 'window_spans': [], 'num_spans': 0}
    trace_rows = []
    if span_records:
        span_df = pd.DataFrame(span_records)
        trace_min = span_df['start_sec'].min()
        offset, method = estimate_trace_offset(span_df, windows, T, trace_min, t_start, log_secs)
        if offset is None:
            method = 'fallback'
            offset = trace_min - t_start
        diag['offset'], diag['method'] = float(offset), method

        end_rel = np.floor(span_df['end_sec'].values - offset - t_start).astype(np.int64)
        diag['num_spans'] = len(span_df)
        diag['in_range_ratio'] = float(((end_rel >= 0) & (end_rel < T)).mean())
        for s, e, fault in windows:
            count = int(((end_rel >= s) & (end_rel < e)).sum())
            diag['window_spans'].append((fault['name'], fault['fault'], count))

        # drop spans outside this experiment's axis so they cannot leak into neighbors
        keep = (end_rel >= 0) & (end_rel < T)
        span_df = span_df[keep].assign(end_time=end_rel[keep] + concat_offset)
        trace_rows = span_df[['cmbd_id', 'fatherpod', 'stats', 'end_time', 'duration']]

    return {
        'name': exp_name,
        'metric': metric,
        'label': label,
        'log_rows': log_rows,
        'trace': trace_rows,
        'has_fault': has_fault,
        'diag': diag,
        'T': T,
    }


def preprocess(raw_fault_dir, raw_nofault_dir, save_path):
    os.makedirs(save_path, exist_ok=True)

    fault_dirs = sorted(d for d in glob.glob(os.path.join(raw_fault_dir, 'SN.20*')) if os.path.isdir(d))
    nofault_dirs = sorted(d for d in glob.glob(os.path.join(raw_nofault_dir, 'SN.20*')) if os.path.isdir(d))
    experiment_dirs = fault_dirs + nofault_dirs
    logging.info("found %d fault + %d no-fault experiments", len(fault_dirs), len(nofault_dirs))

    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sn.ini'))
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    metric_parts, log_parts, trace_parts, label_parts, bounds, diags = [], [], [], [], [], []
    stats_vocab = set()
    concat_offset = 0
    for exp_dir in experiment_dirs:
        result = process_experiment(exp_dir, miner, concat_offset)
        metric_parts.append(result['metric'].values.astype(np.float32))
        label_parts.append(result['label'])
        log_parts.extend(result['log_rows'])
        if len(result['trace']):
            trace_parts.append(result['trace'])
            stats_vocab.update(result['trace']['stats'].unique().tolist())
        bounds.append({'name': result['name'], 'start': concat_offset,
                       'end': concat_offset + result['T'], 'has_fault': result['has_fault']})
        diags.append(result['diag'])
        concat_offset += result['T']
    T_total = concat_offset

    columns = [f'{service}_{kpi}' for service in SN_SERVICES for kpi in SN_METRICS]
    metric_all = np.concatenate(metric_parts, axis=0)
    metric_df = pd.DataFrame(metric_all, columns=columns)
    metric_df.insert(0, 'now', np.arange(T_total))
    metric_df.to_csv(os.path.join(save_path, 'metric.csv'), index=False)
    logging.info("saved metric.csv: %s", metric_df.shape)

    log_df = pd.DataFrame(log_parts, columns=['templateid', 'Hostname', '@timestamp'])
    log_df = log_df.sort_values(by='@timestamp')
    log_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)
    logging.info("saved log.csv: %d rows, %d drain templates", len(log_df), len(miner.drain.clusters))

    if trace_parts:
        trace_df = pd.concat(trace_parts, ignore_index=True).sort_values(by='end_time')
    else:
        trace_df = pd.DataFrame(columns=['cmbd_id', 'fatherpod', 'stats', 'end_time', 'duration'])
    trace_df.to_csv(os.path.join(save_path, 'trace.csv'), index=False)
    logging.info("saved trace.csv: %d spans", len(trace_df))

    label_all = np.concatenate(label_parts, axis=0)
    pickle.dump(label_all, open(os.path.join(save_path, 'label.pkl'), 'wb'))
    logging.info("saved label.pkl: %s, nonzero=%d", label_all.shape, int(label_all.sum()))

    adjacency = np.zeros((SN_NUM_NODES, SN_NUM_NODES))
    for src, dst in SN_EDGES:
        adjacency[SN_SERVICE2NID[src], SN_SERVICE2NID[dst]] = 1
        adjacency[SN_SERVICE2NID[dst], SN_SERVICE2NID[src]] = 1
    pickle.dump(adjacency, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))

    pickle.dump(bounds, open(os.path.join(save_path, 'bounds.pkl'), 'wb'))
    stats_vocab = sorted(stats_vocab)
    pickle.dump(stats_vocab, open(os.path.join(save_path, 'stats_vocab.pkl'), 'wb'))
    logging.info("saved bounds.pkl (%d experiments), stats_vocab.pkl (%d stats)", len(bounds), len(stats_vocab))

    logging.info("=" * 80)
    logging.info("per-experiment trace alignment diagnostics (T_total=%d):", T_total)
    for diag, bound in zip(diags, bounds):
        logging.info("  %s | T=%d | offset=%s | method=%s | spans=%d | in-range=%.2f%%",
                     diag['name'], diag['T'],
                     f"{diag['offset']:.0f}" if diag['offset'] is not None else 'n/a',
                     diag['method'], diag['num_spans'], diag['in_range_ratio'] * 100)
        for container, fault_type, count in diag['window_spans']:
            logging.info("      %-42s %-14s spans_in_window=%d", container, fault_type, count)
    logging.info("all data saved to %s", save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess the Eadro SN dataset.')
    parser.add_argument('--raw_fault_dir', type=str, default=DEFAULT_RAW_FAULT_DIR)
    parser.add_argument('--raw_nofault_dir', type=str, default=DEFAULT_RAW_NOFAULT_DIR)
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()
    preprocess(args.raw_fault_dir, args.raw_nofault_dir, args.save_path)
