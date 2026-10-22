"""Frozen, full-mask contribution influence diagnostics; no training or refitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
from depth_accuracy import SCANS,load_scene,coordinate_check,points_from_depth,in_observation
from sample import sha256,atomic_json,git
from night_guard import ALLOWED_IDENTITIES

METHODS=['all','top10pct_count','top50pct_count','cover90pct_weight','without_far_high_group','without_far_high_single']


def moments(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(n=int(len(a)),sum=float(a.sum()),sum_square=float(np.square(a).sum()),
        min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)


def comparison(errors,mean,stats,scale,ray_norm,obs):
    result={}
    common=np.isfinite(errors[:,:4]).all(1)
    for scope,mask in [('all_mask',common),('common_observation',common&obs[:,:4].all(1))]:
        result[scope]={}
        for j,method in enumerate(METHODS):
            take=mask&np.isfinite(errors[:,j])
            delta=errors[take,j].astype(float)-errors[take,0]
            shift=(mean[take,j].astype(float)-mean[take,0])*scale
            result[scope][method]=dict(error_mm=moments(errors[take,j]),delta_error_mm=moments(delta),
                axial_shift_mm=moments(shift),euclidean_shift_mm=moments(np.abs(shift)*ray_norm[take]),
                improved=int((delta < -1e-4).sum()),worsened=int((delta > 1e-4).sum()),
                tied=int((np.abs(delta)<=1e-4).sum()),over20mm=int((errors[take,j]>=20).sum()),
                within1mm=int((errors[take,j]<=1).sum()),missing=int((mask&~np.isfinite(errors[:,j])).sum()))
    # Fixed bins, chosen before results: mass and separation in all-member weighted SD units.
    result['mechanism_bins']=[]
    for label,j,mass,depth,exists in [
        ('tail',3,stats['tail_mass'],stats['tail_mean'],stats['tail_mass']>0),
        ('far_group',4,stats['far_high_mass'],stats['far_high_mean'],stats['has_far_high']>0),
        ('far_single',5,stats['far_single_p'],stats['far_single_depth'],stats['has_far_single']>0)]:
        sigma=stats['all_weighted_sd']
        sep=np.divide(np.abs(depth-mean[:,0]),sigma,out=np.full(len(mean),np.nan),where=sigma>0)
        mass_bin=np.digitize(mass,[.05,.1,.25,.5],right=False)
        sep_bin=np.digitize(sep,[.5,1,2,4],right=False)
        for a in range(5):
            for b in range(5):
                take=common&exists&np.isfinite(errors[:,j])&np.isfinite(sep)&(mass_bin==a)&(sep_bin==b)
                delta=errors[take,j].astype(float)-errors[take,0]
                result['mechanism_bins'].append(dict(group=label,mass_bin=a,separation_bin=b,
                    delta_error_mm=moments(delta),improved=int((delta < -1e-4).sum()),
                    worsened=int((delta > 1e-4).sum()),tied=int((np.abs(delta)<=1e-4).sum())))
    return result


def run_scan(a,scan,out):
    import torch
    from torch.utils.cpp_extension import load
    from diff_gaussian_rasterization import _C
    from scipy.spatial import cKDTree
    from scipy.io import loadmat
    from plyfile import PlyData
    from sample import forward
    from pixel_contribution import retained_mask,ensure_ninja,concentration_torch
    from ray_influence import diagnostics_torch,_numpy_reference,_STAT_NAMES
    assert ALLOWED_IDENTITIES[a.instance_id]==os.uname().nodename
    assert not git('status','--short')
    commit=git('rev-parse','HEAD');out.mkdir(parents=True,exist_ok=True)
    dataset,pipe,g,cameras,ply,digest=load_scene(scan,out,a.model_root,a.source_root)
    transform,alignment=coordinate_check(dataset,cameras)
    statepath=out/'metadata.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(views=[])
    assert state.get('git_commit',commit)==commit and state.get('model_sha256',digest)==digest
    names=[c.image_name for c in cameras]
    assert state.get('camera_names',names)==names
    gtpath=a.gt_root/'Points/stl'/f'stl{scan:03d}_total.ply'
    gt=PlyData.read(gtpath)['vertex'];xyz=np.stack([gt[k] for k in ('x','y','z')],1).astype(float)
    assert np.isfinite(xyz).all();tree=cKDTree(xyz,leafsize=32);del xyz,gt
    obsfile=a.gt_root/'ObsMask'/f'ObsMask{scan}_10.mat';ob=loadmat(obsfile)
    state.update(status='running',git_commit=commit,git_remote=git('remote','get-url','origin'),scan=scan,
        model=str(ply),model_sha256=digest,cfg_sha256=sha256(ply.parents[2]/'cfg_args'),
        camera_names=names,cameras_npz_sha256=sha256(Path(dataset.source_path)/'cameras.npz'),
        alignment=alignment,gt_sha256=sha256(gtpath),obs_sha256=sha256(obsfile),methods=METHODS,
        host_binary_sha256=sha256(_C.__file__),host_build_provenance=a.host_build_provenance,
        host_binary=str(_C.__file__),hostname=os.uname().nodename,model_group=a.model_group,
        protocol='camera.gt_mask>=.5 all actual cameras; accepted original w; no recomposition. Far-high=top ceil(.1N) by w AND z>original weighted depth median. Single=max positive leave-one-out influence in far-high, ties acceptance order. Counterfactual geometry is exact one-way nearest GT point distance, not pixelwise GT depth.',
        aggregation='Per-view sufficient moments pooled by finite pixel count, no averaging view quantiles. Fixed mass edges .05/.1/.25/.5; separation abs(group mean-all mean)/all weighted SD edges .5/1/2/4; zero SD unknown. Samples only for audit, all-mask moments determine conclusions.')
    atomic_json(statepath,state)
    ensure_ninja();os.environ['MAX_JOBS']='1'
    # Both currently use same architecture; ask Torch for actual device instead of assuming it.
    major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    build=HERE/'构建_pixel_stats';build.mkdir(exist_ok=True)
    ext=load(name='gaudepth_pixel_stats_reader_v2',sources=[str(HERE/'reader.cu')],build_directory=str(build),extra_cuda_cflags=['-O3','--use_fast_math'],verbose=True)
    state.update(reader_source_sha256=sha256(HERE/'reader.cu'),reader_binary_sha256=sha256(ext.__file__))
    bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    previous={v['name']:v for v in state['views']}
    for vi,cam in enumerate(cameras[:a.max_views] if a.max_views else cameras):
        if cam.image_name in previous:
            v=previous[cam.image_name]
            for f in v['files']:assert sha256(out/f['name'])==f['sha256']
            continue
        t=time.time();summary_file=out/f'view_{vi:03d}.json';sample_file=out/f'view_{vi:03d}_sample.npz'
        assert not summary_file.exists() and not sample_file.exists(),'Inspect orphan outputs before resuming'
        with torch.no_grad():
            r,fx,fy=forward(cam,g,pipe,bg,dataset.kernel_size)
            _,rgb,alpha,normal,expected,median,_,geom,binning,image=r
            h,w=cam.image_height,cam.image_width
            if cam.gt_mask is None:raise ValueError('Explicit DTU mask required')
            mask=retained_mask(cam.gt_mask.cpu().numpy(),(h,w));index=np.flatnonzero(mask.ravel()).astype(np.int32)
            dense,dw=ext.read_pixel_stats(geom,binning,image,len(g.get_xyz),w,h)
            torch.testing.assert_close(dw[:,:,2],alpha.reshape(h,w),rtol=2e-5,atol=2e-6)
            assert torch.equal(dense[:,:,6],dense[:,:,7])
            assert torch.equal(dense[:,:,1],dense[:,:,2:6].sum(2).int())
            count=dense.reshape(-1,8)[torch.from_numpy(index).cuda().long()].cpu().numpy()
            weights=dw.reshape(-1,4)[torch.from_numpy(index).cuda().long()].cpu().numpy()
            del dense,dw
            mean=np.full((len(index),6),np.nan,np.float64);sizes=np.zeros((len(index),4),np.int32)
            stats={k:np.full(len(index),np.nan) for k in _STAT_NAMES};concentration=np.full((len(index),11),np.nan)
            audited=0;replay_error=0.
            for first in range(0,len(index),512):
                last=min(first+512,len(index));local=torch.from_numpy(index[first:last]).cuda()
                cap=max(1,int(count[first:last,5].max()))
                rec,ids,n,replay=ext.read_rays(geom,binning,image,len(g.get_xyz),w,h,fx,fy,local,cap)
                np.testing.assert_array_equal(n.cpu().numpy(),count[first:last,5])
                target=torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],1)[local.long()]
                torch.testing.assert_close(replay,target,rtol=2e-5,atol=2e-6)
                diag=diagnostics_torch(rec[:,:,0],rec[:,:,1],n)
                keep=torch.isfinite(diag['mean'][:,0])&(n>0)
                torch.testing.assert_close(diag['mean'][keep,0],expected.flatten()[local.long()][keep].double(),rtol=2e-5,atol=2e-6)
                mean[first:last]=diag['mean'].cpu().numpy();sizes[first:last]=diag['sizes'].cpu().numpy()
                for k,v in diag['stats'].items():
                    if k not in stats:stats[k]=np.full(len(index),np.nan)
                    stats[k][first:last]=v.cpu().numpy()
                concentration[first:last]=concentration_torch(rec[:,:,1],n)[0].cpu().numpy()
                replay_error=max(replay_error,float((replay-target).abs().max()))
                # Independent scalar NumPy replay on first 16 rays of every actual view.
                if first==0:
                    ref=_numpy_reference(rec[:16,:,0].cpu().numpy(),rec[:16,:,1].cpu().numpy(),n[:16].cpu().numpy())
                    for k,rr in enumerate(ref):
                        np.testing.assert_allclose(mean[k],rr['mean'],rtol=1e-10,atol=1e-10,equal_nan=True)
                        for key,value in rr['stats'].items():
                            np.testing.assert_allclose(stats[key][k],value,rtol=1e-9,atol=1e-10,equal_nan=True)
                    audited=len(ref)
            del r,geom,binning,image,rgb,alpha,normal,expected,median
        if a.reference_means_root:
            refmeta=json.loads((a.reference_means_root/f'scan{scan}/metadata.json').read_text())
            assert refmeta['model_sha256']==digest and refmeta['camera_names']==names
            view=next(v for v in refmeta['views'] if v['name']==cam.image_name)
            refpath=a.reference_means_root/f'scan{scan}'/view['file'];assert sha256(refpath)==view['sha256']
            with np.load(refpath) as old:
                np.testing.assert_array_equal(index,old['pixel_index']);np.testing.assert_array_equal(sizes,old['selected_counts'])
                np.testing.assert_allclose(mean[:,:4],old['weighted_mean_depth'],rtol=1e-6,atol=3e-7,equal_nan=True)
        errors=np.full(mean.shape,np.nan,np.float64);obs=np.zeros(mean.shape,bool)
        for j in range(6):
            good=np.flatnonzero(np.isfinite(mean[:,j])&(mean[:,j]>0))
            points=points_from_depth(cam,index[good],mean[good,j],transform)
            for first in range(0,len(points),32768):
                last=min(first+32768,len(points));errors[good[first:last],j]=tree.query(points[first:last],k=1,eps=0,workers=8)[0]
            obs[good,j]=in_observation(points,ob['ObsMask'],ob['BB'],ob['Res'])
        y,x=np.divmod(index,w);ray_norm=np.sqrt(1+((x-cam.Cx)/cam.Fx)**2+((y-cam.Cy)/cam.Fy)**2)
        extra={**stats,**{f'count_{j}':count[:,j] for j in range(8)},**{f'weight_{j}':weights[:,j] for j in range(4)},
            **{f'concentration_{j}':concentration[:,j] for j in range(11)}}
        for key in ('tail','far_high','far_single'):
            shift=stats[f'{key}_shift']
            extra[f'{key}_abs_shift_mm']=np.abs(shift)*alignment['scale_to_dtu_mm']
            extra[f'{key}_shift_over_weighted_sd']=np.divide(np.abs(shift),stats['all_weighted_sd'],out=np.full(len(index),np.nan),where=stats['all_weighted_sd']>0)
        result=dict(name=cam.image_name,index=vi,pixels=len(index),empty_accepted=int((count[:,5]==0).sum()),
            replay_max_abs=replay_error,independent_numpy_rays=audited,seconds=time.time()-t,
            moments={k:moments(v) for k,v in extra.items()},
            comparison=comparison(errors,mean,stats,alignment['scale_to_dtu_mm'],ray_norm,obs))
        atomic_json(summary_file,result)
        take=np.unique(np.linspace(0,max(0,len(index)-1),min(256,len(index)),dtype=int))
        with sample_file.open('wb') as f:
            np.savez_compressed(f,pixel_index=index[take],mean=mean[take],error_mm=errors[take],
                selected_counts=sizes[take],**{k:v[take] for k,v in extra.items()})
            f.flush();os.fsync(f.fileno())
        files=[summary_file,sample_file]
        if not a.reference_means_root:
            # Compatible four-map archive for subsequent fixed TSDF comparison.
            mapfile=out/f'view_{vi:03d}.npz';assert not mapfile.exists()
            with mapfile.open('wb') as f:
                np.savez_compressed(f,pixel_index=index,weighted_mean_depth=mean[:,:4].astype(np.float32),selected_counts=sizes,
                    gt_point_distance_mm=errors[:,:4].astype(np.float32),in_observation=obs[:,:4],image_size=np.array([h,w]))
                f.flush();os.fsync(f.fileno())
            files.append(mapfile)
        state['views'].append(dict(name=cam.image_name,index=vi,pixels=len(index),summary=summary_file.name,
            file=f'view_{vi:03d}.npz' if not a.reference_means_root else None,
            sha256=sha256(mapfile) if not a.reference_means_root else None,
            files=[dict(name=f.name,sha256=sha256(f)) for f in files]))
        atomic_json(statepath,state);print('VIEW',a.model_group,scan,vi+1,len(cameras),time.time()-t,flush=True)
        if (out.parent/'stop_after_stage').exists():break
    assert sha256(ply)==digest
    state.update(status='complete' if len(state['views'])==len(cameras) else 'partial',original_model_unchanged=True)
    atomic_json(statepath,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=int,choices=SCANS);p.add_argument('--all-scans',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--gt-root',type=Path,required=True)
    p.add_argument('--instance-id',choices=list(ALLOWED_IDENTITIES),required=True)
    p.add_argument('--model-group',choices=['G1_MV','G2_noMV'],required=True)
    p.add_argument('--host-build-provenance',required=True);p.add_argument('--reference-means-root',type=Path)
    p.add_argument('--max-views',type=int,default=0);a=p.parse_args()
    if bool(a.scan)==bool(a.all_scans):p.error('choose scan or all-scans')
    assert ALLOWED_IDENTITIES[a.instance_id]==os.uname().nodename
    stop_root=a.output if a.all_scans else a.output.parent
    def request_save(signum,frame):
        stop_root.mkdir(parents=True,exist_ok=True)
        (stop_root/'stop_after_stage').touch()
    signal.signal(signal.SIGTERM,request_save)
    if a.all_scans:
        a.output.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',git_commit=git('rev-parse','HEAD'),completed=[],scans=SCANS,started_at=time.time())
        for scan in SCANS:
            out=a.output/f'scan{scan}';out.mkdir(exist_ok=True)
            mp=out/'metadata.json';prior=json.loads(mp.read_text()) if mp.exists() else {}
            if prior.get('status')=='complete':
                assert prior['git_commit']==state['git_commit']
                for v in prior['views']:
                    for f in v['files']:assert sha256(out/f['name'])==f['sha256']
            else:
                args=[sys.executable,str(Path(__file__).resolve()),'--scan',str(scan),'--output',str(out)]
                for name in ('model_root','source_root','gt_root','instance_id','model_group','host_build_provenance','reference_means_root'):
                    if getattr(a,name) is not None:args.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
                with (out/'run.log').open('a') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True,timeout=2400)
            state['completed'].append(scan);current=json.loads(mp.read_text())
            atomic_json(a.output/'batch.json',state)
            if current['status']!='complete' or (a.output/'stop_after_stage').exists():break
        state['status']='complete' if len(state['completed'])==15 and current['status']=='complete' else 'partial'
        atomic_json(a.output/'batch.json',state)
    else:run_scan(a,a.scan,a.output)


if __name__=='__main__':main()
