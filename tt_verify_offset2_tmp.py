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
    tr = pd.read_csv(os.path.join(d, 'trace.csv'), usecols=['start_sec', 'end_sec', 'duration'])
    tr = tr.drop_duplicates()
    lab = pickle.load(open(os.path.join(d, 'label.pkl'), 'rb'))
    T = lab.shape[0]
    t_start = int(pd.read_csv(os.path.join(d, 'metric.csv'), usecols=['now'])['now'].min())
    starts = tr['start_sec'].values
    trace_min = int(np.floor(starts.min()))
    bins = np.floor(starts).astype(np.int64) - trace_min
    span_len = int(np.floor(starts.max())) - trace_min + 1
    A = np.zeros(span_len)
    np.add.at(A, bins, tr['duration'].values)
    B = lab.sum(axis=1)

    full = np.correlate(A, B, 'full')
    top_lags = np.argsort(full)[::-1][:200]
    scored = []
    for peak in top_lags:
        k = int(peak) - (len(B) - 1)
        n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
        if n_hi - n_lo < 30:
            continue
        a, b = A[k + n_lo:k + n_hi], B[n_lo:n_hi]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        scored.append((spear(a, b), k))
    scored.sort(reverse=True)
    print(exp[2:20])
    for s, k in scored[:5]:
        print('   spear=%.4f off=%d' % (s, k + trace_min - t_start))
    for off in (26637, 28800, 28801):
        k = off - trace_min + t_start
        n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
        a, b = A[k + n_lo:k + n_hi], B[n_lo:n_hi]
        print('   fixed off=%d spear=%.4f inner=%.1f' % (off, spear(a, b), full[k + len(B) - 1]))
