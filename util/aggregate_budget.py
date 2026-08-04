"""Aggregate label-efficiency sweep results into a per-budget table.

Reads the JSON produced by util/eval_rca.py (--localizer_path for every
trained localizer) plus each localizer dir's config.json, groups the runs
by evidence config (saA / saAP / saAPT) and label budget, and reports
macro HR@1/3/5, MRR and NDCG@3 as mean +- std over seeds, together with
per-fault-type HR@1 (login / memory) for the paper's analysis.

Example:
  python util/aggregate_budget.py \
      --eval_json result/rca-eval-budget.json \
      --dirs result/rca-saA-seed42 result/rca-saA-b0.1-seed42 ...
"""

import argparse
import json
import os
import re
import sys

import numpy as np

METRICS = ['HR@1', 'HR@3', 'HR@5', 'MRR', 'NDCG@3']
FAULT_TYPES = ['access permission denied exception', 'file moving program',
               'memory_anomalies', 'login failure']


def method_name(loc_dir):
    """Mirror util/eval_rca.py's result-table naming for a localizer dir."""
    base = os.path.basename(loc_dir.rstrip('/')).replace('rca-', '')
    variant, _, seed = base.rpartition('-seed')
    return f'loc[{variant}]@{seed}'


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_json', required=True)
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Localizer dirs (same set passed to eval_rca.py).')
    parser.add_argument('--out', default=None, help='Optional aggregated JSON output.')
    return parser.parse_args()


def main():
    cli = parse_cli_args()
    with open(cli.eval_json) as f:
        ev = json.load(f)
    results = ev['results']

    rows = []  # (tag, budget, seed, macro_dict, per_type)
    for loc_dir in cli.dirs:
        cfg = json.load(open(os.path.join(loc_dir, 'config.json')))
        tag = 'saA' if cfg['groups'] == ['anomaly'] else \
              'saAP' if cfg['groups'] == ['anomaly', 'propagation'] else 'saAPT'
        name = method_name(loc_dir)
        if name not in results:
            print(f'WARN: {name} not found in {cli.eval_json} (dir {loc_dir})',
                  file=sys.stderr)
            continue
        res = results[name]
        rows.append((tag, cfg['label_budget'], cfg['seed'],
                     res['macro'], res['per_type']))

    tags = ['saA', 'saAP', 'saAPT']
    budgets = sorted({b for _, b, *_ in rows})
    aggregated = {'detector_macro': results.get('detector', {}).get('macro'),
                  'tags': tags, 'budgets': budgets, 'cells': {}}

    for tag in tags:
        for budget in budgets:
            cell = [r for r in rows if r[0] == tag and r[1] == budget]
            if not cell:
                continue
            key = f'{tag}-b{budget}'
            cell_out = {'seeds': len(cell)}
            for metric in METRICS:
                vals = np.asarray([c[3][metric] for c in cell], dtype=float)
                cell_out[metric] = [float(vals.mean()), float(vals.std())]
            cell_out['HR@1_per_type'] = {
                'login failure': [float(np.mean([c[4]['login failure']['HR@1'] for c in cell])),
                                  float(np.std([c[4]['login failure']['HR@1'] for c in cell]))],
                'memory_anomalies': [float(np.mean([c[4]['memory_anomalies']['HR@1'] for c in cell])),
                                     float(np.std([c[4]['memory_anomalies']['HR@1'] for c in cell]))],
            }
            aggregated['cells'][key] = cell_out

    # printable table
    header = f"{'config':<14}{'budget':>7}{'n':>4}" + ''.join(f'{m:>18}' for m in METRICS)
    print('macro metrics (mean +/- std over seeds)')
    print(header)
    for tag in tags:
        for budget in budgets:
            key = f'{tag}-b{budget}'
            if key not in aggregated['cells']:
                continue
            c = aggregated['cells'][key]
            cells = ''.join(f'{(f"{c[m][0]:.4f}+/-{c[m][1]:.4f}"):>18}' for m in METRICS)
            print(f'{tag:<14}{budget:>7}{c["seeds"]:>4}' + cells)
    print()
    print('per-fault-type HR@1 (mean +/- std over seeds)')
    for tag in tags:
        for budget in budgets:
            key = f'{tag}-b{budget}'
            if key not in aggregated['cells']:
                continue
            c = aggregated['cells'][key]
            login = c['HR@1_per_type']['login failure']
            mem = c['HR@1_per_type']['memory_anomalies']
            print(f'{tag:<10} b{budget:<5} login {login[0]:.4f}+/-{login[1]:.4f}'
                  f'   memory {mem[0]:.4f}+/-{mem[1]:.4f}')

    if cli.out:
        with open(cli.out, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f'\nSaved {cli.out}')


if __name__ == '__main__':
    main()
