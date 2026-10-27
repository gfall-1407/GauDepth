"""G0-only user-authorized cleanup. Run inside the verified independent guard.

Preserve original data/models and accepted ray records before removing outputs.
No training, rendering, CUDA import/build, platform shutdown or Trash operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import zipfile

HOST='autodl-container-5d5147b55f-67549ee6'
HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
STATE=Path('/root/autodl-tmp/GauDepth_Cleanup_20261027')
DATA=Path('/root/autodl-tmp/GauDepth_RetainedData_20261027')
ORIGINAL=Path('/root/autodl-tmp/GauDepth')
MODEL=Path('/root/GauDepth_HuberRepair_20261026/inputs_G1/outtest')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def original_status():
    return hashlib.sha256(subprocess.check_output(['git','-C',str(ORIGINAL),'status','--porcelain'])).hexdigest()


def targets():
    huber=Path('/root/GauDepth_HuberRepair_20261026')
    protected={'inputs_G1','G1_frozen_inputs.tar','build_deps','photo_observations_v1',
               'photo_selected40_v1.tar','photo_selected40_v1.json','all_inputs_ready.json',
               'protected_original_before.json'}
    items=[p for p in huber.iterdir() if p.name not in protected]
    items += [Path(p) for p in ['/root/GauDepth_Gap_20261027',
              '/root/autodl-tmp/GauDepth_Training_20261027','/root/autodl-tmp/GauDepth_Gap_20261027']]
    for repo in [ORIGINAL,Path('/root/autodl-tmp/GauDepth_surface'),REPO]:
        for name in ['01_冻结模型分层','02_参考表面分层与定位','03_高斯场层归属探索']:
            items.append(repo/'实验'/name/'结果')
    result=[]
    for p in items:
        if not p.exists():continue
        if p.is_symlink() or p.resolve()!=p:raise ValueError('Symlink cleanup target rejected: '+str(p))
        if p==STATE or p==DATA or MODEL==p or p in MODEL.parents:raise ValueError('Protected target')
        result.append(p)
    return result


def run():
    if socket.gethostname()!=HOST or not STATE.is_dir():raise ValueError('G0 and prepared state required')
    if (STATE/'cleanup.json').exists():raise ValueError('Existing cleanup run preserved')
    git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a]).decode().strip()
    if git('remote','get-url','origin')!='git@github.com:gfall-1407/GauDepth.git' or git('status','--porcelain'):
        raise ValueError('Committed approved source required')
    refs=json.loads((HERE/'G1_Huber全场景输入参考.json').read_text(encoding='utf-8'))['scans']
    protected=[]
    for scan,ref in refs.items():
        for p,digest in [(MODEL/f'scan{scan}/point_cloud/iteration_30000/point_cloud.ply',ref['model_sha256']),
                         (MODEL/f'scan{scan}/cfg_args',ref['cfg_sha256'])]:
            if sha(p)!=digest:raise ValueError('Baseline model differs: '+str(p))
            protected.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size))
    if original_status()!='e400d2a9aadfb67c1ddfbd312c08e7e239dbb7ec128249ba493c6d5cb1c193c5':
        raise ValueError('Original source modifications changed')
    started=time.monotonic();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    def check():
        if stopped or time.monotonic()-started>840:raise InterruptedError('Cleanup stop/deadline')
    state=dict(status='planning',source=dict(commit=git('rev-parse','HEAD'),branch=git('branch','--show-current')),
        hostname=HOST,protected=protected,targets=[],preserved=[],removed=[],disk_before={p:shutil.disk_usage(p).free for p in ['/root','/root/autodl-tmp']})
    save(STATE/'cleanup.json',state)
    try:
        chosen=targets();preserve={}
        for target in chosen:
            check();files=[]
            if target.is_dir():
                for d,dirs,names in os.walk(target,followlinks=False):
                    for n in dirs:
                        if (Path(d)/n).is_symlink():raise ValueError('Symlink in result tree')
                    files.extend(Path(d)/n for n in names)
            else:files=[target]
            for p in files:
                if p.is_symlink():raise ValueError('Symlink result file')
                keep=any(part in ('inputs','inputs_G1','G2_inputs','score_inputs','observations_v1') for part in p.relative_to(target).parts[:-1])
                if p.name=='depth_anything_v2_vits.pth':keep=True
                if p.suffix=='.npz':
                    with zipfile.ZipFile(p) as z:keys={Path(n).stem for n in z.namelist()}
                    keep |= {'offsets','depth','weight','gaussian_id'}<=keys or {'offsets','records','gaussian_id'}<=keys
                if keep:
                    preserve[str(p)]=p
                    parent=p.parent
                    while parent!=target and target in parent.parents:
                        if (parent/'metadata.json').is_file():preserve[str(parent/'metadata.json')]=parent/'metadata.json'
                        parent=parent.parent
            state['targets'].append(dict(path=str(target),files=len(files),bytes=sum(p.stat().st_size for p in files)))
        save(STATE/'plan.json',dict(targets=state['targets'],preserve_paths=list(preserve),protected=protected))
        state['status']='planned_waiting_execute_marker';save(STATE/'cleanup.json',state)
        while not (STATE/'execute_cleanup').exists():check();time.sleep(1)
        DATA.mkdir(exist_ok=False)
        for p in preserve.values():
            check();destination=DATA/p.relative_to('/root');destination.parent.mkdir(parents=True,exist_ok=True)
            digest=sha(p);size=p.stat().st_size
            shutil.move(str(p),str(destination))
            if sha(destination)!=digest:raise ValueError('Preserved record changed')
            state['preserved'].append(dict(original_path=str(p),path=str(destination),bytes=size,sha256=digest))
            save(STATE/'cleanup.json',state)
        save(DATA/'sources.json',dict(original_models=protected,raw_files=state['preserved']))
        for target in chosen:
            check()
            # Targets are the immutable in-memory allowlist built above, never read from external JSON.
            if target.is_dir():shutil.rmtree(target)
            else:target.unlink()
            state['removed'].append(str(target));save(STATE/'cleanup.json',state)
        for item in protected:
            if sha(item['path'])!=item['sha256']:raise ValueError('Protected baseline changed after cleanup')
        if original_status()!='e400d2a9aadfb67c1ddfbd312c08e7e239dbb7ec128249ba493c6d5cb1c193c5':raise ValueError('Original source changed')
        state.update(status='complete_waiting_transfer',protected_models_unchanged=True,original_dirty_unchanged=True,
            disk_after={p:shutil.disk_usage(p).free for p in ['/root','/root/autodl-tmp']},seconds=time.monotonic()-started)
        save(STATE/'cleanup.json',state)
        while not (STATE/'transfer_verified').exists():check();time.sleep(1)
        state['status']='complete';save(STATE/'cleanup.json',state)
    except BaseException as error:
        state.update(status='partial_failed',error=repr(error));save(STATE/'cleanup.json',state);raise


if __name__=='__main__':run()
