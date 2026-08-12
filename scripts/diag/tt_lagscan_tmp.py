import os
import pickle

import numpy as np
import pandas as pd

SRC = '/home/zhangll24/RCA_project/MSTGAD/data/TT-pre'


def spear(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


exps = sorted(d for d in os.listdir(SRC) if d.startswith('TT.'))
for exp in exps:
    d = os.path.join(SRC, exp)
    tr = pd.read_csv(os.path.join(d, 'trace.csv'),
                     usecols=['start_sec', 'end_sec', 'duration'])
    tr = tr.drop_duplicates()
    lab = pickle.load(open(os.path.join(d, 'label.pkl'), 'rb'))
    now = pd.read_csv(os.path.join(d, 'metric.csv'), usecols=['now'])['now']
    t0, T = int(now.min()), len(now)

    starts = tr['start_sec'].values
    a_origin = int(np.floor(starts.min()))
    bins = np.floor(starts).astype(np.int64) - a_origin
    span_len = int(np.floor(starts.max())) - a_origin + 1
    A = np.zeros(span_len)
    np.add.at(A, bins, tr['duration'].values)
    B = (lab[:T].sum(axis=1) > 0).astype(np.float64)

    corr = np.correlate(A, B, 'full')
    order = np.argsort(corr)[::-1]
    max_end = tr['end_sec'].max()
    n_dedup = len(tr)
    print('==== %s  T=%d dedup=%d max_end=%.0f' % (exp, T, n_dedup, max_end))
    shown = 0
    for rank, peak in enumerate(order[:200]):
        k = int(peak) - (len(B) - 1)
        n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
        if n_hi - n_lo < 30:
            continue
        a_seg, b_seg = A[k + n_lo:k + n_hi], B[n_lo:n_hi]
        if a_seg.std() < 1e-12 or b_seg.std() < 1e-12:
            continue
        s = spear(a_seg, b_seg)
        if s <= 0.1:
            continue
        offset = a_origin + k - t0
        end_rel = np.floor(tr['end_sec'].values - offset - t0).astype(np.int64)
        kept = int(((end_rel >= 0) & (end_rel < T)).sum())
        tail = int(np.floor(max_end - offset - t0)) - (T - 1)
        print('  rank=%3d offset=%7d spear=%.4f kept=%5d tail=%7d corr=%.3e'
              % (rank, offset, s, kept, tail, corr[peak]))
        shown += 1
        if shown >= 6:
            break
    off_true = int(np.floor(max_end)) - t0 - (T - 1)
    k = off_true - a_origin + t0
    n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
    if n_hi - n_lo >= 30:
        s = spear(A[k + n_lo:k + n_hi], B[n_lo:n_hi])
        end_rel = np.floor(tr['end_sec'].values - off_true - t0).astype(np.int64)
        kept = int(((end_rel >= 0) & (end_rel < T)).sum())
        print('  TAIL-HEURISTIC offset=%d spear=%.4f kept=%d' % (off_true, s, kept))
