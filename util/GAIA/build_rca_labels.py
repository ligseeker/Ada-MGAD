"""Build root-cause-analysis labels for GAIA from the fault-injection event table.

Reads ``label_events.csv`` (parsed from ``run_table_2021-07.csv`` by
``pre_GAIA.deal_label``) and produces ``label_rca.csv`` on the canonical 30s
timestamp grid of ``label.csv``. Only the four RCA fault types are kept:

    login failure, memory_anomalies, file moving program,
    access permission denied exception

cpu_anomalies are excluded (3s injections are below the 30s observability
floor) and normal / normal memory freed label / unknown / error_event are not
faults. The detection labels in ``label.csv`` are left untouched.

Output columns:
    timestamp       epoch ms, same grid as label.csv
    root_cause_id   node index (GAIA_SERVICES order) when exactly one node is
                    the root cause, else -1
    root_cause_ids  '|'-joined node indices of all active root-cause nodes
                    ('' when no fault is active)
    fault_type      '|'-joined active fault types, '[none]' when no fault
    n_events        number of injection events covering the window
    multi_root      1 when >= 2 distinct nodes are root causes, else 0
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.GAIA.constant import GAIA_SERVICES

logger = logging.getLogger(__name__)

RCA_FAULT_TYPES = [
    'login failure',
    'memory_anomalies',
    'file moving program',
    'access permission denied exception',
]

INTERVAL_MS = 30 * 1000


def load_timestamp_grid(data_path):
    label_file = os.path.join(data_path, 'label.csv')
    grid = pd.read_csv(label_file, usecols=['timestamp'])['timestamp'].values.astype(np.int64)
    grid.sort()
    return grid


def load_rca_events(data_path):
    events_file = os.path.join(data_path, 'label_events.csv')
    events = pd.read_csv(events_file)
    events['anomaly_type'] = events['anomaly_type'].str.strip('[]')
    events = events[events['anomaly_type'].isin(RCA_FAULT_TYPES)].copy()
    events = events[events['instance'].isin(GAIA_SERVICES)]
    events['node_id'] = events['instance'].map({s: i for i, s in enumerate(GAIA_SERVICES)})
    events['st_ms'] = (pd.to_datetime(events['st_time'], utc=True).astype(np.int64) // 10 ** 6)
    events['ed_ms'] = (pd.to_datetime(events['ed_time'], utc=True).astype(np.int64) // 10 ** 6)
    # floor-align to the 30s grid, identical to pre_GAIA.deal_label
    events['st_al'] = (events['st_ms'] // INTERVAL_MS) * INTERVAL_MS
    events['ed_al'] = (events['ed_ms'] // INTERVAL_MS) * INTERVAL_MS
    return events


def map_events_to_grid(events, grid):
    node_sets = [set() for _ in range(len(grid))]
    type_sets = [set() for _ in range(len(grid))]
    n_events = np.zeros(len(grid), dtype=np.int64)
    unmatched = 0

    for event in events.itertuples(index=False):
        left = np.searchsorted(grid, event.st_al, side='left')
        right = np.searchsorted(grid, event.ed_al, side='right')
        if left >= right:
            unmatched += 1
            continue
        for pos in range(left, right):
            node_sets[pos].add(event.node_id)
            type_sets[pos].add(event.anomaly_type)
            n_events[pos] += 1

    return node_sets, type_sets, n_events, unmatched


def build_rca_frame(grid, node_sets, type_sets, n_events):
    root_cause_id = np.full(len(grid), -1, dtype=np.int64)
    root_cause_ids = []
    fault_type = []
    multi_root = np.zeros(len(grid), dtype=np.int64)

    for i in range(len(grid)):
        nodes = sorted(node_sets[i])
        if len(nodes) == 1:
            root_cause_id[i] = nodes[0]
        elif len(nodes) > 1:
            multi_root[i] = 1
        root_cause_ids.append('|'.join(str(n) for n in nodes))
        types = sorted(type_sets[i])
        fault_type.append('|'.join(f'[{t}]' for t in types) if types else '[none]')

    return pd.DataFrame({
        'timestamp': grid,
        'root_cause_id': root_cause_id,
        'root_cause_ids': root_cause_ids,
        'fault_type': fault_type,
        'n_events': n_events,
        'multi_root': multi_root,
    })


def log_statistics(events, frame, unmatched):
    logger.info('=== RCA label statistics ===')
    logger.info('Events kept: %d (unmatched to grid: %d)', len(events), unmatched)
    for fault_type in RCA_FAULT_TYPES:
        n_ev = int((events['anomaly_type'] == fault_type).sum())
        n_win = int(frame['fault_type'].str.contains(f'[{fault_type}]', regex=False).sum())
        logger.info('  %-40s events=%6d  windows=%6d', f'[{fault_type}]', n_ev, n_win)

    n_any = int((frame['n_events'] > 0).sum())
    n_single = int((frame['root_cause_id'] >= 0).sum())
    n_multi = int(frame['multi_root'].sum())
    logger.info('Windows with a root cause: %d / %d', n_any, len(frame))
    logger.info('  single-root: %d (%.2f%%)', n_single, 100 * n_single / max(n_any, 1))
    logger.info('  multi-root:  %d (%.2f%%)', n_multi, 100 * n_multi / max(n_any, 1))

    logger.info('Root-cause windows per node:')
    for node_id, service in enumerate(GAIA_SERVICES):
        n = int((frame['root_cause_id'] == node_id).sum())
        logger.info('  %-15s %6d', service, n)


def build_rca_labels(data_path, out_name='label_rca.csv'):
    grid = load_timestamp_grid(data_path)
    logger.info('Timestamp grid: %d windows (%s -> %s)', len(grid),
                pd.to_datetime(grid[0], unit='ms', utc=True),
                pd.to_datetime(grid[-1], unit='ms', utc=True))

    events = load_rca_events(data_path)
    node_sets, type_sets, n_events, unmatched = map_events_to_grid(events, grid)
    frame = build_rca_frame(grid, node_sets, type_sets, n_events)

    out_file = os.path.join(data_path, out_name)
    frame.to_csv(out_file, index=False)
    logger.info('Saved %s', out_file)

    log_statistics(events, frame, unmatched)
    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Build GAIA root-cause-analysis labels.')
    parser.add_argument('--data_path', default='./data/GAIA-pre',
                        help='Directory containing label.csv and label_events.csv.')
    parser.add_argument('--out_name', default='label_rca.csv',
                        help='Output CSV file name inside data_path.')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_cli_args()
    build_rca_labels(args.data_path, args.out_name)
