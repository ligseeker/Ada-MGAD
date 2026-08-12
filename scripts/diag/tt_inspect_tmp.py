import os
import pickle

import pandas as pd

MSTGAD_TT = '/home/zhangll24/RCA_project/MSTGAD/data/TT-pre'
RAW = '/home/zhangll24/RCA_project/Eadro/TT_Dataset/TT Dataset'

print('=== MSTGAD TT-pre dirs ===')
for name in sorted(os.listdir(MSTGAD_TT)):
    path = os.path.join(MSTGAD_TT, name)
    if os.path.isdir(path):
        print(name, sorted(os.listdir(path)))

exps = sorted(d for d in os.listdir(MSTGAD_TT) if os.path.isdir(os.path.join(MSTGAD_TT, d)))
exp = exps[0]
print('\n=== sample experiment:', exp, '===')
m = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'metric.csv'), nrows=3)
print('metric columns[:8]:', list(m.columns[:8]))
print('metric columns[-3:]:', list(m.columns[-3:]))
print('metric now head:', m['now'].head(3).tolist())

full_now = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'metric.csv'), usecols=['now'])
print('metric rows:', len(full_now), 'now min/max:', full_now['now'].min(), full_now['now'].max())

log = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'log.csv'), nrows=5)
print('\nlog.csv head:')
print(log.to_string())
print('log.csv row count:', sum(1 for _ in open(os.path.join(MSTGAD_TT, exp, 'log.csv'))) - 1)

tr = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'trace.csv'), nrows=5)
print('\ntrace.csv head:')
print(tr.to_string())
print('trace.csv row count:', sum(1 for _ in open(os.path.join(MSTGAD_TT, exp, 'trace.csv'))) - 1)

lab = pickle.load(open(os.path.join(MSTGAD_TT, exp, 'label.pkl'), 'rb'))
print('\nlabel.pkl:', lab.shape, lab.dtype, 'nonzero:', int(lab.sum()))
tp = pickle.load(open(os.path.join(MSTGAD_TT, exp, 'trace_path.pkl'), 'rb'))
print('trace_path.pkl:', tp.shape, tp.dtype, 'edges:', int(tp.sum()))

print('\n=== Eadro raw dirs ===')
for sub in ('data', 'no fault'):
    base = os.path.join(RAW, sub)
    names = sorted(os.listdir(base))
    print(sub, len(names))
    for n in names:
        if os.path.isdir(os.path.join(base, n)):
            subdirs = sorted(os.listdir(os.path.join(base, n)))
            print('  ', n, subdirs[:8])

fault_exp = exps[0]
svc_dir = os.path.join(RAW, 'data', fault_exp, 'metrics')
svcs = sorted(os.listdir(svc_dir))
print('\nraw metric services (%d):' % len(svcs), svcs[:5], '...')
head = pd.read_csv(os.path.join(svc_dir, svcs[0]), nrows=3)
print('raw metric head:', head.to_string())
raw_ts = pd.read_csv(os.path.join(svc_dir, svcs[0]), usecols=['timestamp'])
print('raw metric rows:', len(raw_ts), 'min/max:', raw_ts['timestamp'].min(), raw_ts['timestamp'].max())

print('\n=== all experiments: label nonzero / T ===')
for e in exps:
    l = pickle.load(open(os.path.join(MSTGAD_TT, e, 'label.pkl'), 'rb'))
    now = pd.read_csv(os.path.join(MSTGAD_TT, e, 'metric.csv'), usecols=['now'])
    print(e, 'T=', l.shape[0], 'now_min=', int(now['now'].min()), 'label_nonzero=', int(l.sum()))
