"""Scan24 exploration, freeze one adaptive rule, then 14 unchanged validations."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SCANS=[24,37,40,55,63,65,69,83,97,105,106,110,114,118,122]
DTU='/root/autodl-tmp/DTU_MVS_Data'

def result(scan,method):
    p=(ROOT/'outtest'/f'scan{scan}'/'vis'/'results.json' if method=='expected'
       else HERE/'结果'/f'scan{scan}'/method/'results_layered.json')
    return json.loads(p.read_text())

def table():
    rows=[]
    for scan in SCANS:
        for method in ['expected','median','gap_fixed','gap_adaptive','contribution_valley']:
            try:r=result(scan,method)
            except FileNotFoundError:continue
            rows.append(dict(scan=scan,method=method,**r))
    for method in ['expected','median','gap_fixed','gap_adaptive','contribution_valley']:
        subset=[r for r in rows if r['method']==method and isinstance(r['scan'],int)]
        if subset:
            rows.append(dict(scan=f'平均({len(subset)})',method=method,
                 **{key:sum(r[key] for r in subset)/len(subset)
                    for key in ['mean_d2s','mean_s2d','overall']}))
    with (HERE/'结果表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scan','method','mean_d2s','mean_s2d','overall'])
        w.writeheader();w.writerows(rows)

def process(scan,methods):
    model=ROOT/'outtest'/f'scan{scan}'
    dest=HERE/'结果'/f'scan{scan}';dest.mkdir(parents=True,exist_ok=True)
    missing=[m for m in methods if not (dest/m/'recon_post_layered.ply').exists()]
    if missing:
        log_name='mesh_extract_adaptive.log' if missing==['gap_adaptive'] else 'mesh_extract_layered.log'
        with (dest/log_name).open('w') as log:
            subprocess.run([sys.executable,str(HERE/'mesh_extract_layered.py'),'-m',str(model),
                             '--methods',*missing],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for method in methods:
        if not (dest/method/'results_layered.json').exists():
            with (dest/method/'evaluate_dtu_mesh_layered.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate_dtu_mesh_layered.py'),'-m',str(model),
                    '--DTU',DTU,'--method',method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('RESULT',scan,method,result(scan,method),flush=True)
        table()

if __name__=='__main__':
    os.environ.setdefault('OMP_NUM_THREADS','16');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    started=time.time();failures=[]
    if '--complete_adaptive' in sys.argv:
        for scan in SCANS:
            try:process(scan,['gap_adaptive'])
            except Exception as e:
                failures.append(dict(scan=scan,error=repr(e)))
                print('SCAN_FAILED',scan,repr(e),flush=True)
        table()
        (HERE/'自适应补充状态.json').write_text(json.dumps(dict(finished=True,
            seconds=time.time()-started,method='gap_adaptive',failures=failures),indent=2))
        print('ADAPTIVE_DONE',failures,flush=True)
        sys.exit(1 if failures else 0)
    process(24,['median','gap_fixed','gap_adaptive','contribution_valley'])
    candidates=['gap_adaptive','contribution_valley']
    if '--scan24_only' in sys.argv:
        print('SCAN24_DONE_REVIEW_BEFORE_FREEZING',flush=True)
        sys.exit(0)
    if '--selected' not in sys.argv:
        raise SystemExit('Review Scan24 completeness/details, then pass --selected gap_adaptive or contribution_valley')
    selected=sys.argv[sys.argv.index('--selected')+1]
    assert selected in candidates
    (HERE/'固定规则.json').write_text(json.dumps(dict(selected=selected,selection_scan=24,
        criterion='Scan24 overall plus completeness and mesh detail review; exploratory selection; no per-scan tuning',
        candidates={m:result(24,m) for m in candidates}),ensure_ascii=False,indent=2))
    print('FROZEN_RULE',selected,flush=True)
    for scan in [105]+[s for s in SCANS if s not in [24,105]]:
        try:process(scan,['median','gap_fixed',selected])
        except Exception as e:
            failures.append(dict(scan=scan,error=repr(e)))
            print('SCAN_FAILED',scan,repr(e),flush=True)
    table()
    (HERE/'批次状态.json').write_text(json.dumps(dict(finished=True,seconds=time.time()-started,
                   selected=selected,failures=failures),ensure_ascii=False,indent=2))
    print('BATCH_DONE',failures,flush=True)
