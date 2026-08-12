import os
import pickle

import numpy as np
import pandas as pd

base = '/home/zhangll24/RCA_project/MSTGAD/data/TT-pre'


def spear(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


for exp in ['TT.2022-04-17T212101D2022-04-17T230842',
            'TT.2022-04-19T001753D2022-04-19T020534']:
    d = os.path.join(base, exp)
    tr = pd.read_csv(os.path.join(d, 'trace.csv'), usecols=['start_sec', 'end_sec', 'duration', 'span_id', 'base_trace'])
    n_raw = len(tr)
    tr = tr.drop_duplicates()
    log = pd.read_csv(os.path.join(d, 'log.csv'), usecols=['@timestamp'])
    lab = pickle.load(open(os.path.join(d, 'label.pkl'), 'rb'))
    T = lab.shape[0]
    t_start = int(pd.read_csv(os.path.join(d, 'metric.csv'), usecols=['now'])['now'].min())

    starts = tr['start_sec'].values
    trace_min = int(np.floor(starts.min()))
    bins = np.floor(starts).astype(np.int64) - trace_min
    span_len = int(np.floor(starts.max())) - trace_min + 1
    B = lab.sum(axis=1)

    A_dur = np.zeros(span_len)
    np.add.at(A_dur, bins, tr['duration'].values)
    A_cnt = np.zeros(span_len)
    np.add.at(A_cnt, bins, 1)

    L = np.zeros(T)
    np.add.at(L, np.clip((log['@timestamp'].values - t_start).astype(int), 0, T - 1), 1)

    def report(A, X, name):
        peak = int(np.argmax(np.correlate(A, X, 'full')))
        k = peak - (len(X) - 1)
        n_lo, n_hi = max(0, -k), min(len(X), len(A) - k)
        a, b = A[k + n_lo:k + n_hi], X[n_lo:n_hi]
        print('  %-10s argmax_off=%d spear=%.4f pear=%.4f' %
              (name, k + trace_min - t_start, spear(a, b), np.corrcoef(a, b)[0, 1]))

    print(exp[2:20], 'dedup rows %d -> %d' % (n_raw, len(tr)))
    report(A_dur, B, 'dur~label')
    report(A_cnt, L, 'cnt~log')
    report(A_dur, L, 'dur~log')
    for off in (26637, 28800, 28801):
        end_rel = np.floor(tr['end_sec'].values - off - t_start).astype(np.int64)
        keep = (end_rel >= 0) & (end_rel < T)
        inw = keep & (B[np.clip(end_rel, 0, T - 1)] > 0)
        print('   off=%d kept=%d in-fault-window=%d' % (off, int(keep.sum()), int(inw.sum())))
