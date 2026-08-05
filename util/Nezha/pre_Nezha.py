import argparse
import glob
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.Nezha.constant import (
    NEZHA_DAYS, NEZHA_SERVICES, NEZHA_SERVICE2NID, NEZHA_NUM_NODES,
    NEZHA_KPIS, FAULT_DURATION_SEC, BUCKET_SEC, pod_to_service,
)

DEFAULT_RAW_DIR = "/home/zhangll24/RCA_project/Nezha"
DEFAULT_SAVE_PATH = "./data/Nezha-pre"


def robust_zscore(frame):
    medians = frame.median()
    mads = frame.sub(medians).abs().median()
    flat = mads < 1e-9
    normalized = frame.sub(medians).div(1.4826 * mads.where(~flat, 1.0))
    normalized.loc[:, flat] = 0.0
    return normalized.fillna(0.0)


def epoch_from_time_column(time_str):
    return pd.to_datetime(time_str.str[:19], format='%Y-%m-%d %H:%M:%S', utc=True).astype(np.int64) // 10**9


def repair_timestamps(df, pod):
    time_epoch = epoch_from_time_column(df['Time']).values
    delta = df['TimeStamp'].values - time_epoch
    offset = int(np.median(delta))
    if abs(offset) <= 1:
        return df['TimeStamp'].values.astype(np.int64)
    repaired = df['TimeStamp'].values - offset
    residual = int(np.abs(repaired - time_epoch).max())
    if residual <= 30:
        logging.warning("pod %s: TimeStamp shifted by %d seconds, repaired with constant offset", pod, offset)
        return repaired.astype(np.int64)
    logging.warning("pod %s: TimeStamp corrupted (offset %d, residual %d s), rebuilt from Time column",
                    pod, offset, residual)
    return time_epoch.astype(np.int64)


def load_day_metric(raw_dir, day):
    metric_dir = os.path.join(raw_dir, 'rca_data', day, 'metric')
    frames = {}
    for pod in sorted(os.listdir(metric_dir)):
        if not pod.endswith('_metric.csv'):
            continue
        service = pod_to_service(pod[:-len('_metric.csv')])
        assert service in NEZHA_SERVICE2NID, f"unknown service {service} from {pod}"
        df = pd.read_csv(os.path.join(metric_dir, pod))
        missing = [kpi for kpi in NEZHA_KPIS if kpi not in df.columns]
        assert not missing, f"missing KPI columns {missing} in {pod}"
        ts = repair_timestamps(df, pod)
        frames[service] = pd.DataFrame(
            {f'{service}_{kpi}': df[kpi].values for kpi in NEZHA_KPIS}, index=ts)
    assert set(frames) == set(NEZHA_SERVICES), f"day {day}: pods {sorted(frames)} != expected services"

    t0 = int(min(frame.index.min() for frame in frames.values()))
    t_end = int(max(frame.index.max() for frame in frames.values()))
    T = (t_end - t0) // BUCKET_SEC + 1
    parts = []
    for service in NEZHA_SERVICES:
        frame = frames[service]
        buckets = (frame.index.values - t0) // BUCKET_SEC
        grouped = frame.groupby(buckets).mean()
        grouped = grouped.reindex(range(T)).ffill().fillna(0.0)
        parts.append(grouped)
    metric = pd.concat(parts, axis=1).sort_index()
    assert len(metric) == T, f"day {day}: metric rows {len(metric)} != {T}"
    logging.info("day %s metric: t0=%d t_end=%d rows=%d, per-pod rows=%s",
                 day, t0, t_end, T,
                 {svc: len(frames[svc]) for svc in NEZHA_SERVICES})
    return robust_zscore(metric), t0, T


def extract_log_message(raw):
    try:
        outer = json.loads(raw)
        inner = json.loads(outer['log'])
        message = inner.get('message')
        if message:
            return message
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return raw


def day_window_files(raw_dir, day, kind):
    files = sorted(glob.glob(os.path.join(raw_dir, 'rca_data', day, kind, '*.csv')))
    files += sorted(glob.glob(os.path.join(raw_dir, 'construct_data', day, kind, '*.csv')))
    return files


def load_day_logs(raw_dir, day, miner, t0, T, concat_offset):
    seen, rows, skipped = set(), [], 0
    for path in day_window_files(raw_dir, day, 'log'):
        df = pd.read_csv(path, usecols=['TimeUnixNano', 'PodName', 'SpanID', 'Log'])
        for ts_ns, pod, span_id, raw in df.values.tolist():
            key = (ts_ns, pod, span_id)
            if key in seen:
                continue
            seen.add(key)
            bucket = int(ts_ns // 10**9 - t0) // BUCKET_SEC
            if bucket < 0 or bucket >= T:
                skipped += 1
                continue
            service = pod_to_service(pod)
            if service not in NEZHA_SERVICE2NID:
                skipped += 1
                continue
            cluster_id = miner.add_log_message(extract_log_message(raw))['cluster_id']
            rows.append((cluster_id, service, bucket + concat_offset))
    logging.info("day %s logs: %d rows kept (%d out-of-range/unknown skipped)", day, len(rows), skipped)
    return rows


def load_day_traces(raw_dir, day, t0, T, concat_offset):
    seen, records, raw_spans = set(), [], 0
    for path in day_window_files(raw_dir, day, 'trace'):
        df = pd.read_csv(path)
        raw_spans += len(df)
        span_pod = dict(zip(df['SpanID'], df['PodName']))
        fresh = df[~df['SpanID'].isin(seen)]
        seen.update(df['SpanID'].values.tolist())
        fresh = fresh[fresh['ParentID'] != 'root']
        parent_pod = fresh['ParentID'].map(span_pod)
        fresh = fresh[parent_pod.notna()]
        src = fresh['PodName'].map(pod_to_service)
        dst = parent_pod.map(pod_to_service)
        mask = (src != dst) & src.isin(NEZHA_SERVICE2NID) & dst.isin(NEZHA_SERVICE2NID)
        fresh = fresh[mask]
        src, dst = src[mask], dst[mask]
        if not len(fresh):
            continue
        end_bucket = ((fresh['EndTimeUnixNano'].values // 10**9) - t0) // BUCKET_SEC
        keep = (end_bucket >= 0) & (end_bucket < T)
        duration = (fresh['EndTimeUnixNano'].values - fresh['StartTimeUnixNano'].values) / 1e9
        records.append(pd.DataFrame({
            'cmbd_id': src.values[keep],
            'fatherpod': dst.values[keep],
            'stats': fresh['OperationName'].values[keep],
            'end_time': end_bucket[keep] + concat_offset,
            'duration': duration[keep],
        }))
    trace_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=['cmbd_id', 'fatherpod', 'stats', 'end_time', 'duration'])
    logging.info("day %s traces: %d raw spans, %d unique SpanIDs, %d edge records kept",
                 day, raw_spans, len(seen), len(trace_df))
    return trace_df, raw_spans


def load_day_labels(raw_dir, day, t0, T, concat_offset):
    label = np.zeros((T, NEZHA_NUM_NODES))
    fault_path = os.path.join(raw_dir, 'rca_data', day, f'{day}-fault_list.json')
    fault_data = json.load(open(fault_path))
    faults = [fault for faults in fault_data.values() for fault in faults]
    for fault in sorted(faults, key=lambda item: int(item['inject_timestamp'])):
        service = pod_to_service(fault['inject_pod'])
        assert service in NEZHA_SERVICE2NID, f"unknown fault pod {fault['inject_pod']}"
        start = int(fault['inject_timestamp'])
        lo = (start - t0) // BUCKET_SEC
        hi = (start + FAULT_DURATION_SEC - 1 - t0) // BUCKET_SEC
        lo, hi = max(0, lo), min(T - 1, hi)
        label[lo:hi + 1, NEZHA_SERVICE2NID[service]] = 1
        nows = [concat_offset + i for i in range(lo, hi + 1)]
        logging.info("day %s fault %-19s %-14s %-38s +%ds -> buckets %s",
                     day, fault['inject_time'], fault['inject_type'], service,
                     (hi - lo + 1) * BUCKET_SEC, nows)
    return label, len(faults)


def load_adjacency(raw_dir):
    adjacency = np.zeros((NEZHA_NUM_NODES, NEZHA_NUM_NODES))
    for day in NEZHA_DAYS:
        dep = pd.read_csv(os.path.join(raw_dir, 'rca_data', day, 'metric', 'dependency.csv'))
        for src, dst in dep[['Source', 'Target']].values.tolist():
            if src in NEZHA_SERVICE2NID and dst in NEZHA_SERVICE2NID:
                adjacency[NEZHA_SERVICE2NID[src], NEZHA_SERVICE2NID[dst]] = 1
                adjacency[NEZHA_SERVICE2NID[dst], NEZHA_SERVICE2NID[src]] = 1
    return adjacency


def preprocess(raw_dir, save_path):
    os.makedirs(save_path, exist_ok=True)

    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nezha.ini'))
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    metric_parts, log_parts, trace_parts, label_parts, bounds = [], [], [], [], []
    stats_vocab, diag = set(), []
    concat_offset = 0
    for day in NEZHA_DAYS:
        logging.info("processing day %s", day)
        metric, t0, T = load_day_metric(raw_dir, day)
        metric_parts.append(metric.values.astype(np.float32))
        log_parts.extend(load_day_logs(raw_dir, day, miner, t0, T, concat_offset))
        trace_df, raw_spans = load_day_traces(raw_dir, day, t0, T, concat_offset)
        if len(trace_df):
            trace_parts.append(trace_df)
            stats_vocab.update(trace_df['stats'].unique().tolist())
        label, num_faults = load_day_labels(raw_dir, day, t0, T, concat_offset)
        label_parts.append(label)
        bounds.append({'name': day, 'start': concat_offset, 'end': concat_offset + T,
                       'has_fault': num_faults > 0})
        diag.append({'name': day, 'T': T, 'raw_spans': raw_spans, 'kept_spans': len(trace_df),
                     'num_faults': num_faults, 'label_nonzero': int(label.sum())})
        concat_offset += T
    T_total = concat_offset

    columns = [f'{service}_{kpi}' for service in NEZHA_SERVICES for kpi in NEZHA_KPIS]
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

    adjacency = load_adjacency(raw_dir)
    pickle.dump(adjacency, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))
    logging.info("saved trace_path.pkl: %dx%d adjacency, %d edges",
                 NEZHA_NUM_NODES, NEZHA_NUM_NODES, int(adjacency.sum() // 2))

    pickle.dump(bounds, open(os.path.join(save_path, 'bounds.pkl'), 'wb'))
    stats_vocab = sorted(stats_vocab)
    pickle.dump(stats_vocab, open(os.path.join(save_path, 'stats_vocab.pkl'), 'wb'))
    logging.info("saved bounds.pkl (%d days), stats_vocab.pkl (%d stats)", len(bounds), len(stats_vocab))

    logging.info("=" * 80)
    logging.info("per-day diagnostics (T_total=%d):", T_total)
    for entry in diag:
        logging.info("  %s | T=%d | faults=%d | label_nonzero=%d | raw_spans=%d -> kept_spans=%d",
                     entry['name'], entry['T'], entry['num_faults'],
                     entry['label_nonzero'], entry['raw_spans'], entry['kept_spans'])
    logging.info("all data saved to %s", save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess the Nezha OnlineBoutique dataset.')
    parser.add_argument('--raw_dir', type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.save_path)
