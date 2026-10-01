#!/usr/bin/env python3
"""Independent receipt guard and integer audit for the frozen detector family."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.e2e.protocol import sha256_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());run=Path(c['detector_run'])
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    prediction=run/'global_prediction_lock.json';scope=json.loads((run/'scope_lock.json').read_text())
    lock=json.loads(prediction.read_text())
    if sha256_file(prediction)!=c['detector_prediction_lock_sha256']:
        raise ValueError('independently committed detector prediction receipt mismatch')
    if lock['status']!='PREDICTION_LOCKED' or lock['arms']!=scope['config']['arms']:
        raise ValueError('prediction scope identity mismatch')
    for path,digest in {**scope['bindings'],**lock['bindings'],**c['detector_model_manifest_sha256']}.items():
        if sha256_file(Path(path))!=digest:raise ValueError('detector receipt/source/input drift: '+path)
    source=Path(c['detector_tree']);command=[c['detector_python'],str(source/'scripts/p6/run_frozen_test_review.py'),
        'evaluate','--config',str(source/'configs/e2e/gaia_p6_frozen_test_review_v1.json'),'--output-dir',str(run)]
    subprocess.run(command,cwd=source,check=True)
    report=json.loads((run/'evaluation/results.json').read_text());records={}
    for arm in lock['arms']:
        matching=pd.read_csv(run/'evaluation'/arm['id']/'matching.csv')
        counts={key:int(matching.match_status.eq(status).sum()) for key,status in
            [('tp','matched'),('fp','false_alarm'),('fn','miss')]}
        n=len(pd.read_csv(run/'arms'/arm['id']/'episodes.csv'));m=report['results'][arm['id']]['metrics']
        if counts['tp']+counts['fn']!=5787 or counts['tp']+counts['fp']!=n:
            raise ValueError('independent denominator closure failed')
        for key,name in [('tp','true_positive_events'),('fp','false_positive_events'),('fn','false_negative_events')]:
            if counts[key]!=m[name]:raise ValueError('independent detector integer mismatch')
        expected=2.*counts['tp']/(5787+n)
        if abs(expected-m['event_f1'])>1e-15:raise ValueError('detector F1 mismatch')
        records[arm['id']]=counts
    if len(records)!=19:raise ValueError('incomplete detector family')
    a.output_dir.mkdir(parents=True)
    write_json(a.output_dir/'audit.json',dict(status='PASS',prediction_lock_sha256=c['detector_prediction_lock_sha256'],
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        detector_execution_commit=scope['execution_commit'],models_metadata_receipts=c['detector_model_manifest_sha256'],
        complete_gt=5787,arms=records,evaluation_manifest_sha256=sha256_file(run/'evaluation/completion_manifest.json')))
    print('INDEPENDENT_DETECTOR_AUDIT_PASS',len(records),flush=True)


if __name__=='__main__':main()
