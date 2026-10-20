"""Four prespecified exploration scans, one frozen lambda, then all 15 scans."""
import sys,os,json,csv,time,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=HERE.parent/'01_冻结模型分层'
SCANS=[24,37,40,55,63,65,69,83,97,105,106,110,114,118,122]
EXPLORE=[24,65,105,110];HELDOUT=[s for s in SCANS if s not in EXPLORE]
KEYS=['mean_d2s','mean_s2d','overall'];DTU='/root/autodl-tmp/DTU_MVS_Data'

def metric(scan,method):
    if method=='expected':p=ROOT/'outtest'/f'scan{scan}'/'vis'/'results.json'
    elif method in ['median','gap_fixed','gap_adaptive','contribution_valley']:p=OLD/'结果'/f'scan{scan}'/method/'results_layered.json'
    else:p=HERE/'结果'/f'scan{scan}'/method/'results_surface_support.json'
    return json.loads(p.read_text())

def suffix(lam):return 'l001' if lam==.01 else 'l003'

def tables(selected=None):
    exploration=[]
    for scan in EXPLORE:
        for method in ['expected','reference_depth','reference_fixed_mean','reference_support_mean_l001','reference_support_mean_l003']:
            try:r=metric(scan,method)
            except FileNotFoundError:continue
            exploration.append(dict(scope='探索',scan=scan,method=method,**r))
    for method in ['expected','reference_depth','reference_fixed_mean','reference_support_mean_l001','reference_support_mean_l003']:
        subset=[r for r in exploration if r['method']==method and isinstance(r['scan'],int)]
        if subset:exploration.append(dict(scope='探索',scan=f'平均({len(subset)})',method=method,**{k:sum(r[k] for r in subset)/len(subset) for k in KEYS}))
    write('探索结果.csv',exploration,['scope','scan','method',*KEYS])
    if selected is None:return
    names=['expected','median','gap_fixed','gap_adaptive','contribution_valley','reference_depth','reference_fixed_mean',
        'reference_support_mean_'+suffix(selected),'reference_support_wmedian_'+suffix(selected)]
    rows=[]
    for scan in SCANS:
        for method in names:
            try:r=metric(scan,method)
            except FileNotFoundError:continue
            rows.append(dict(scope='探索' if scan in EXPLORE else '验证',scan=scan,method=method,**r))
    original=list(rows)
    for scope,scans in [('探索4场景',EXPLORE),('验证11场景',HELDOUT),('全部15场景',SCANS)]:
        for method in names:
            subset=[r for r in original if r['method']==method and r['scan'] in scans]
            if subset:rows.append(dict(scope=scope,scan=f'平均({len(subset)})',method=method,**{k:sum(r[k] for r in subset)/len(subset) for k in KEYS}))
    write('结果表.csv',rows,['scope','scan','method',*KEYS])

def write(name,rows,fields):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def process(scan,methods,lambdas):
    if shutil.disk_usage(HERE).free<1.5*1024**3:raise RuntimeError('Less than 1.5 GiB free; preserve originals and stop before fusion/evaluation')
    dest=HERE/'结果'/f'scan{scan}';dest.mkdir(parents=True,exist_ok=True)
    missing=[m for m in methods if not (dest/m/'recon_post_surface_support.ply').exists()]
    if missing:
        with (dest/('mesh_extract_'+('_'.join(missing))+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(HERE/'mesh_extract_surface_support.py'),'-m',str(ROOT/'outtest'/f'scan{scan}'),
                '--methods',*missing,'--lambdas',*[str(l) for l in lambdas]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for method in methods:
        p=dest/method
        if not (p/'results_surface_support.json').exists():
            with (p/'evaluate_surface_support.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'evaluate_dtu_mesh_surface_support.py'),'-m',str(ROOT/'outtest'/f'scan{scan}'),
                    '--method',method,'--DTU',DTU],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('RESULT',scan,method,metric(scan,method),flush=True)

if __name__=='__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser();parser.add_argument('--phase',choices=['explore','validate'],required=True)
    parser.add_argument('--lambda',dest='lam',type=float,choices=[.01,.03]);args=parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','16');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    start=time.time();failures=[]
    if args.phase=='explore':
        for scan in EXPLORE:
            try:process(scan,['reference_depth','reference_fixed_mean','reference_support_mean_l001','reference_support_mean_l003'],[.01,.03])
            except Exception as e:failures.append(dict(scan=scan,error=repr(e)));print('SCAN_FAILED',scan,repr(e),flush=True)
            tables()
        (HERE/'探索状态.json').write_text(json.dumps(dict(finished=True,seconds=time.time()-start,failures=failures),indent=2))
        print('EXPLORATION_DONE',failures,flush=True)
    else:
        assert args.lam in [.01,.03]
        candidates={suffix(l):{k:sum(metric(s,'reference_support_mean_'+suffix(l))[k] for s in EXPLORE)/4 for k in KEYS} for l in [.01,.03]}
        frozen=HERE/'固定参数.json'
        if frozen.exists():assert json.loads(frozen.read_text())['lambda']==args.lam,'Cannot change frozen parameter'
        else:frozen.write_text(json.dumps(dict(**{'lambda':args.lam},exploration_scans=EXPLORE,heldout_scans=HELDOUT,
            candidates=candidates,criterion='One global lambda after reviewing four exploration averages and directional tradeoffs; no per-scan tuning'),indent=2))
        for scan in SCANS:
            try:process(scan,['reference_depth','reference_fixed_mean','reference_support_mean_'+suffix(args.lam),
                'reference_support_wmedian_'+suffix(args.lam)],[args.lam])
            except Exception as e:failures.append(dict(scan=scan,error=repr(e)));print('SCAN_FAILED',scan,repr(e),flush=True)
            tables(args.lam)
        (HERE/'批次状态.json').write_text(json.dumps(dict(finished=True,seconds=time.time()-start,selected_lambda=args.lam,failures=failures),indent=2))
        print('VALIDATION_DONE',failures,flush=True)
