import json
import os
import pickle

import numpy as np
import pandas as pd

MSTGAD_TT = '/home/zhangll24/RCA_project/MSTGAD/data/TT-pre'
RAW = '/home/zhangll24/RCA_project/Eadro/TT_Dataset/TT Dataset'

exp = 'TT.2022-04-17T212101D2022-04-17T230842'
fault_json = os.path.join(RAW, 'data', exp.replace('TT.', 'TT.fault-') + '.json')
print('fault json exists:', os.path.exists(fault_json))
fd = json.load(open(fault_json))
print('fault json start/end:', fd['start'], fd['end'], 'num faults:', len(fd.get('faults', [])))
for f in fd.get('faults', [])[:3]:
    print('  fault:', f['name'], f.get('fault'), f['start'], f['duration'])

now = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'metric.csv'), usecols=['now'])['now']
print('metric now min/max/rows:', now.min(), now.max(), len(now))
lab = pickle.load(open(os.path.join(MSTGAD_TT, exp, 'label.pkl'), 'rb'))
nz = np.argwhere(lab.sum(axis=1) > 0).ravel()
print('label shape:', lab.shape, 'active seconds range:', nz.min(), nz.max(), 'count:', len(nz))
rel_start = int(fd['start']) - int(now.min())
print('fault json start relative to metric t0:', rel_start)

print('\n=== no-fault experiment metric rows ===')
for e in ['TT.2022-04-21T105158D2022-04-21T130705', 'TT.2022-04-21T153246D2022-04-21T174753']:
    now = pd.read_csv(os.path.join(MSTGAD_TT, e, 'metric.csv'), usecols=['now'])['now']
    lab = pickle.load(open(os.path.join(MSTGAD_TT, e, 'label.pkl'), 'rb'))
    raw = pd.read_csv(os.path.join(RAW, 'no fault', e, 'metrics', 'ts-auth-service.csv'), usecols=['timestamp'])
    print(e, 'metric_rows:', len(now), 'label_T:', lab.shape[0], 'raw_rows:', len(raw),
          'now_min:', now.min(), 'raw_min:', raw['timestamp'].min(), 'raw_max:', raw['timestamp'].max())

print('\n=== trace.csv details (fault exp) ===')
tr = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'trace.csv'))
print('rows:', len(tr), 'unique span_id:', tr['span_id'].nunique())
print('start_sec min/max:', tr['start_sec'].min(), tr['start_sec'].max())
print('duration stats:', tr['duration'].min(), tr['duration'].max(), tr['duration'].mean())
print('stats values:', sorted(tr['stats'].unique()))
print('cmbd_id unknown count:', (tr['cmbd_id'] == 'unknown').sum(), 'fatherpod unknown:', (tr['fatherpod'] == 'unknown').sum())
print('self loops:', (tr['cmbd_id'] == tr['fatherpod']).sum())
print('dtypes:', dict(tr.dtypes))
print('start_time vs start_sec equal:', (tr['start_time'] == tr['start_sec']).all())

print('\n=== trace.csv min start per experiment ===')
for e in sorted(os.listdir(MSTGAD_TT)):
    p = os.path.join(MSTGAD_TT, e, 'trace.csv')
    t = pd.read_csv(p, usecols=['start_sec', 'end_sec'])
    print(e, 'rows:', len(t), 'min_start:', t['start_sec'].min(), 'max_end:', t['end_sec'].max())

print('\n=== log.csv details ===')
lg = pd.read_csv(os.path.join(MSTGAD_TT, exp, 'log.csv'))
print('rows:', len(lg), 'hostnames:', sorted(lg['Hostname'].unique())[:5], '...')
print('ts min/max:', lg['@timestamp'].min(), lg['@timestamp'].max(), 'dtype:', lg['@timestamp'].dtype)
print('payload sample:', lg['Payload'].iloc[0][:120])
print('null payloads:', lg['Payload'].isna().sum())

print('\n=== log rows across all experiments ===')
tot = 0
for e in sorted(os.listdir(MSTGAD_TT)):
    lg = pd.read_csv(os.path.join(MSTGAD_TT, e, 'log.csv'))
    tot += len(lg)
    print(e, len(lg))
print('total log rows:', tot)
