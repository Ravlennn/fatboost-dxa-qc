"""Sequential low-load queue; waits for existing CNN pair, then external ROI and matched comparison."""
import json
import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cgmh_pipeline_v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continue-roi',action='store_true')
    args_cli=parser.parse_args()
    os.nice(10)
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    OUT.mkdir(parents=True,exist_ok=True)
    state=dict(pid=os.getpid(),status='WAITING_FOR_CURRENT_PAIR',steps=[])
    if args_cli.continue_roi:
        previous=json.loads((OUT/'status.json').read_text())
        if previous['status']!='STOPPED_BY_USER': raise ValueError('Only explicitly stopped queue can continue')
        interrupted=ROOT/'outputs/hip_cnn/cgmh_full_plus_roi_s0_v1'
        if list(interrupted.glob('fold*.pt')): raise ValueError('Saved folds exist; use fold recovery instead')
        provenance=json.loads((interrupted/'provenance.json').read_text())
        control=ROOT/'outputs/hip_cnn/cgmh_control_full_s0_v1'
        if not (control/'metrics.md').exists(): raise ValueError('Control incomplete')
        if provenance['tables']!=json.loads((control/'provenance.json').read_text())['tables']:
            raise ValueError('Control cohort mismatch')
        for name,digest in provenance['tables'].items():
            if hashlib.sha256((ROOT/'data/interim'/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Dataset changed')
        if hashlib.sha256((ROOT/'outputs/cgmh_roi_v1/dxa_rois.json').read_bytes()).hexdigest()!=provenance['roi_sha256']:
            raise ValueError('ROI changed')
        stamp=str(time.time_ns())
        (OUT/f'status_stopped_{stamp}.json').write_text(json.dumps(previous,indent=2))
        interrupted.rename(interrupted.with_name(interrupted.name+'_stopped_'+stamp))
        log=OUT/'cgmh_full_plus_roi_s0_v1.log'
        if log.exists(): log.rename(log.with_name(log.stem+'_stopped_'+stamp+'.log'))
        state['steps']=[s for s in previous['steps'] if s['status']=='COMPLETE']
        state['restart_reason']='User requested continuation; interrupted ROI run had no complete folds'
    def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2))
    save()
    while True:
        pair=json.loads((ROOT/'outputs/hip_cnn/local_pair_lowload_v2/status.json').read_text())
        runs=pair['runs']
        if any(r['status']=='FAILED' for r in runs): raise RuntimeError('Previous pair failed; inspect before continuing')
        if len(runs)==2 and all(r['status']=='COMPLETE' for r in runs): break
        # Do not start a second GPU job if the prior runner is paused or still active.
        os.kill(pair['pid'],0)
        time.sleep(30)
    stages=[('segmenter',['scripts/train_cgmh_roi.py'])]
    for tag,roi in [('cgmh_control_full_s0_v1',False),('cgmh_full_plus_roi_s0_v1',True)]:
        args=['scripts/train_hip_cnn.py','--tag',tag,'--arch','convnext_tiny','--res','384','--epochs','30',
              '--seed','0','--cpu-threads','2','--batch-pause','1']
        if roi: args+=['--roi-file',str(ROOT/'outputs/cgmh_roi_v1/dxa_rois.json')]
        stages.append((tag,args))
    stages.append(('comparison',['scripts/compare_cgmh_roi.py']))
    if args_cli.continue_roi:
        stages=[s for s in stages if s[0] in ('cgmh_full_plus_roi_s0_v1','comparison')]
    for name,args in stages:
        state['status']='RUNNING'; step=dict(name=name,status='RUNNING',started=time.time()); state['steps'].append(step); save()
        with (OUT/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        step.update(status='COMPLETE' if result.returncode==0 else 'FAILED',returncode=result.returncode,finished=time.time()); save()
        if result.returncode: raise RuntimeError(f'Stage failed: {name}')
        if name not in ('segmenter','comparison'):
            report=(ROOT/'outputs/hip_cnn'/name/'metrics.md').read_text()
            with (ROOT/'docs/EXPERIMENTS.md').open('a') as f: f.write('\n'+report+'\n')
    state['status']='COMPLETE'; save()

if __name__=='__main__':
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/'status.json'; state=json.loads(p.read_text()) if p.exists() else {}
        state.update(status='FAILED',error=str(e)); p.write_text(json.dumps(state,indent=2)); raise
