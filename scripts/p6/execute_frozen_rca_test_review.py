#!/usr/bin/env python3
"""Wait for the live detector job, seal receipts, and execute the fixed RCA scope."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--detector-pid',type=int,required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();c=json.loads(a.config.read_text());det=Path(c['detector_run'])
    prediction=det/'global_prediction_lock.json'
    while not prediction.exists():
        try:os.kill(a.detector_pid,0)
        except ProcessLookupError:
            raise RuntimeError('detector process ended without global prediction lock; do not restart automatically')
        done=len(list((det/'models').glob('*/completion_manifest.json')))
        print('WAIT_VERIFIED_DETECTOR_PID',a.detector_pid,done,'/10',flush=True)
        time.sleep(30)
    metadata=list((det/'models').glob('*/completion_manifest.json'))
    if len(metadata)!=10:raise ValueError('detector global lock missing model completion family')
    c['detector_prediction_lock_sha256']=digest(prediction)
    c['detector_model_manifest_sha256']={str(path.resolve()):digest(path) for path in metadata}
    a.config.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n')
    files=['docs/GAIA_P5_CURRENT_CONTEXT.md','docs/P6_FROZEN_RCA_TEST_REVIEW_PROTOCOL.md',
        'scripts/p6/run_frozen_rca_test_review.py','scripts/p6/audit_frozen_detector_test_review.py',
        'scripts/p6/execute_frozen_rca_test_review.py','tests/test_frozen_rca_test_review.py',
        str(a.config.relative_to(ROOT)) if a.config.is_absolute() else str(a.config)]
    subprocess.run(['git','add',*files],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Freeze all detector receipts and RCA Test comparison scope'],cwd=ROOT,check=True)
    runner=ROOT/'scripts/p6/run_frozen_rca_test_review.py'
    base=[sys.executable,str(runner)]
    options=['--config',str(a.config),'--output-dir',str(a.output_dir)]
    for action in ('prepare','train','features','rank'):
        print('START_RCA_STAGE',action,flush=True)
        command=base+[action]+options
        if action=='features':command+=['--workers',str(a.workers)]
        subprocess.run(command,cwd=ROOT,check=True)
    rca_lock=digest(a.output_dir/'predictions/global_prediction_lock.json')
    subprocess.run([sys.executable,str(ROOT/'scripts/p6/audit_frozen_detector_test_review.py'),
        '--config',str(a.config),'--output-dir',c['detector_audit_dir']],cwd=ROOT,check=True)
    audit_lock=digest(Path(c['detector_audit_dir'])/'audit.json')
    subprocess.run(base+['evaluate']+options+['--prediction-lock-sha256',rca_lock,
        '--detector-audit-sha256',audit_lock],cwd=ROOT,check=True)
    print('FULL_FROZEN_TEST_COMPARISON_COMPLETE_19_DETECTOR_38_TRANSFER_2_ABLATION',flush=True)


if __name__=='__main__':main()
