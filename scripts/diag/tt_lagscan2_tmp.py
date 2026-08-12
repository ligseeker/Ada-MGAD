import os
import pickle

import numpy as np
import pandas as pd

SRC = '/home/zhangll24/RCA_project/MSTGAD/data/TT-pre'


def spear(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


CASES = {
    'TT.2022-04-17T212101D2022-04-17T230842': [26637, 28746, 28790],
    'TT.2022-04-18T102520D2022-04-18T121301': [-18253, 28793],
    'TT.2022-04-18T121515D2022-04-18T140256': [-116088, 28769],
    'TT.2022-04-18T150729D2022-04-18T165511': [26418, 26064, 28790],
    'TT.2022-04-18T203805D2022-04-18T222547': [28042, 28755, 28790],
    'TT.2022-04-21T105158D2022-04-21T130705': [28788],
    'TT.2022-04-21T153246D2022-04-21T174753': [28777],
}

for exp, offsets in CASES.items():
    d = os.path.join(SRC, exp)
    tr = pd.read_csv(os.path.join(d, 'trace.csv'),
                     usecols=['start_sec', 'end_sec', 'duration', 'stats'])
    tr = tr.drop_duplicates()
    lg = pd.read_csv(os.path.join(d, 'log.csv'), usecols=['@timestamp'])
    lab = pickle.load(open(os.path.join(d, 'label.pkl'), 'rb'))
    now = pd.read_csv(os.path.join(d, 'metric.csv'), usecols=['now'])['now']
    t0, T = int(now.min()), len(now)

    starts = tr['start_sec'].values
    a_origin = int(np.floor(starts.min()))
    bins = np.floor(starts).astype(np.int64) - a_origin
    span_len = int(np.floor(starts.max())) - a_origin + 1
    A_dur = np.zeros(span_len)
    np.add.at(A_dur, bins, tr['duration'].values)
    A_cnt = np.zeros(span_len)
    np.add.at(A_cnt, bins, 1)
    has_fault = lab.sum() > 0

    L = np.zeros(T)
    rel_log = (lg['@timestamp'].values - t0).astype(np.int64)
    rel_log = rel_log[(rel_log >= 0) & (rel_log < T)]
    np.add.at(L, rel_log, 1)

    metric = np.load(os.path.join('./data/TT-pre', exp, 'metric.npy'))
    metric = metric.reshape(T, 27, 7)
    M = (metric.astype(np.float64) ** 2).sum(axis=(1, 2))

    print('==== %s (fault=%s)' % (exp, has_fault))
    for offset in offsets:
        k = offset - a_origin + t0
        n_lo, n_hi = max(0, -k), min(T, span_len - k)
        if has_fault:
            B = (lab[:T].sum(axis=1) > 0).astype(np.float64)
            s_main = spear(A_dur[k + n_lo:k + n_hi], B[n_lo:n_hi])
        else:
            s_main = spear(A_cnt[k + n_lo:k + n_hi], M[n_lo:n_hi])
        end_rel = np.floor(tr['end_sec'].values - offset - t0).astype(np.int64)
        keep = (end_rel >= 0) & (end_rel < T)
        kept = int(keep.sum())
        C = np.zeros(T)
        np.add.at(C, end_rel[keep], 1)
        s_log = spear(C, L)
        s_met = spear(C, M)
        first3 = []
        active = lab[:T].sum(axis=1) > 0
        win, st = [], None
        for i, f in enumerate(active):
            if f and st is None:
                st = i
            elif not f and st is not None:
                win.append((st, i)); st = None
        for s_, e_ in win[:3]:
            first3.append(int(((end_rel >= s_) & (end_rel < e_)).sum()))
        print('  offset=%7d spear_main=%.4f kept=%5d spear_log=%.4f spear_metric=%.4f win3=%s'
              % (offset, s_main, kept, s_log, s_met, first3))
