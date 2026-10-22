"""Frozen contribution-prefix means versus independent DTU scan geometry."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
SCANS=[24,37,40,55,63,65,69,83,97,105,106,110,114,118,122]
METHODS=['all','top10pct_count','top50pct_count','cover90pct_weight']


def means_torch(depth,weight,counts):
    import torch
    axis=torch.arange(depth.shape[1],device=depth.device)[None,:]
    valid=axis<counts[:,None]
    w=torch.where(valid,weight.double(),0.)
    assert bool(torch.isfinite(w).all()) and bool((w[valid]>0).all())
    order=torch.argsort(w,dim=1,descending=True,stable=True)
    w=w.gather(1,order);d=depth.double().gather(1,order)
    n90=(w.cumsum(1)<((.9-1e-12)*w.sum(1))[:,None]).sum(1)+1
    n90=torch.where(counts>0,n90,0)
    sizes=torch.stack([counts,(counts+9)//10,(counts+1)//2,n90],1)
    selected=axis[:,None,:]<sizes[:,:,None]
    measure=torch.where(selected,w[:,None,:],0.)
    origin=torch.where(torch.isfinite(d[:,0]),d[:,0],0.)
    x=torch.where(torch.isfinite(d),d-origin[:,None],0.)
    mu=(measure*x[:,None,:]).sum(2)/measure.sum(2).clamp(min=torch.finfo(torch.float64).tiny)+origin[:,None]
    bad=(selected&~torch.isfinite(d[:,None,:])).any(2)
    return torch.where((sizes>0)&~bad,mu,float('nan')),sizes


def check(device):
    import torch
    d=np.array([[1,3,9,99],[4,5,6,7],[4,np.nan,np.nan,np.nan],[np.nan,2,5,np.nan],[0,0,0,0]],float)
    w=np.array([[.6,.3,.1,0],[.25]*4,[1,0,0,0],[.01,.8,.19,0],[0]*4],float)
    n=np.array([3,4,1,3,0],np.int32)
    mu,size=means_torch(torch.tensor(d,device=device),torch.tensor(w,device=device),torch.tensor(n,device=device))
    for i,k in enumerate(n):
        order=np.argsort(-w[i,:k],kind='stable');z=d[i,:k][order];weight=w[i,:k][order]
        for j,l in enumerate(size[i].cpu().numpy()):
            ref=np.average(z[:l],weights=weight[:l]) if l and np.isfinite(z[:l]).all() else np.nan
            np.testing.assert_allclose(mu[i,j].item(),ref,rtol=1e-12,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(mu[0].cpu().numpy(),[2.4,1,5/3,5/3],atol=1e-12)
    print('weighted means independent NumPy/count/tie/empty/nonfinite checks passed',flush=True)


def load_scene(scan,out):
    import torch
    from arguments import ModelParams,PipelineParams
    from scene import Scene,GaussianModel
    from sample import protect_scene,sha256
    model=Path('/root/autodl-tmp/GauDepth/outtest')/f'scan{scan}'
    node=ast.parse((model/'cfg_args').read_text(),mode='eval').body
    assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Namespace' and not node.args
    cfg={k.arg:ast.literal_eval(k.value) for k in node.keywords}
    p=argparse.ArgumentParser();mp=ModelParams(p);pp=PipelineParams(p)
    values=vars(p.parse_args([]));values.update(cfg)
    values.update(model_path=str(model),source_path=f'/root/autodl-tmp/DTU/scan{scan}')
    args=argparse.Namespace(**values);dataset=mp.extract(args);pipe=pp.extract(args)
    ply=model/'point_cloud/iteration_30000/point_cloud.ply';digest=sha256(ply)
    g=GaussianModel(dataset.sh_degree)
    with protect_scene(out):scene=Scene(dataset,g,load_iteration=30000,shuffle=False)
    cameras=scene.getTrainCameras()+scene.getTestCameras()
    assert len({c.image_name for c in cameras})==len(cameras)
    return dataset,pipe,g,cameras,ply,digest


def coordinate_check(dataset,cameras):
    import cv2
    residual=[]
    with np.load(Path(dataset.source_path)/'cameras.npz') as a:
        transform=a['scale_mat_0'].astype(float);scale=float(transform[0,0])
        assert scale>0 and np.allclose(transform[:3,:3],np.eye(3)*scale,rtol=0,atol=1e-6)
        np.testing.assert_allclose(transform[3],[0,0,0,1],atol=1e-8)
        for cam in cameras:
            index=int(Path(cam.image_name).stem)
            np.testing.assert_allclose(a[f'scale_mat_{index}'],transform,atol=1e-6,rtol=0)
            projection=a[f'world_mat_{index}']@transform
            _,rot,center=cv2.decomposeProjectionMatrix(projection[:3,:4])[:3]
            center=(center[:3]/center[3]).ravel();translation=-rot@center
            actual=cam.world_view_transform.T.cpu().numpy()
            re=float(np.max(np.abs(rot-actual[:3,:3])));te=float(np.max(np.abs(translation-actual[:3,3])))
            assert re<1e-4 and te<1e-4,(cam.image_name,re,te)
            residual.append(dict(name=cam.image_name,rotation_max_abs=re,translation_max_abs_scene_units=te))
    return transform,dict(transform=transform.tolist(),scale_to_dtu_mm=scale,camera_checks=residual,
        definition='Shared dataset scale_mat, verified per named image against world_mat@scale_mat and actual camera R/t; no fitting or method-specific ICP.')


def points_from_depth(cam,index,depth,transform):
    y,x=np.divmod(index,cam.image_width)
    points=np.stack([(x-cam.Cx)/cam.Fx*depth,(y-cam.Cy)/cam.Fy*depth,depth],1)
    pose=np.linalg.inv(cam.world_view_transform.T.cpu().numpy().astype(float))
    world=points@pose[:3,:3].T+pose[:3,3]
    return world@transform[:3,:3].T+transform[:3,3]


def in_observation(points,obs,bb,res):
    grid=np.rint((points-bb[:1])/res).astype(np.int64)
    inside=((grid>=0)&(grid<np.array(obs.shape))).all(1)
    answer=np.zeros(len(points),bool);g=grid[inside]
    answer[inside]=obs[g[:,0],g[:,1],g[:,2]].astype(bool)
    return answer


def metric_rows(scan,errors,common_obs):
    valid=np.isfinite(errors).all(1)
    rows=[]
    for scope,take in [('all_mask_common_finite',valid),('common_observation',valid&common_obs)]:
        e=errors[take].astype(float)
        for j,m in enumerate(METHODS):
            v=e[:,j];delta=v-e[:,0]
            rows.append(dict(scan=scan,method=m,scope=scope,mask_pixels=len(errors),valid_pixels=len(v),
                missing_pixels=int((~valid).sum()),common_obs_pixels=int((valid&common_obs).sum()),
                mean_mm=float(v.mean()) if len(v) else None,rms_mm=float(np.sqrt(np.mean(v*v))) if len(v) else None,
                median_mm=float(np.median(v)) if len(v) else None,p95_mm=float(np.quantile(v,.95)) if len(v) else None,
                within_1mm=float(np.mean(v<=1)) if len(v) else None,within_2mm=float(np.mean(v<=2)) if len(v) else None,
                over_20mm=float(np.mean(v>=20)) if len(v) else None,
                mean_delta_mm=float(delta.mean()) if len(v) else None,
                improved_fraction=float(np.mean(delta < -1e-4)) if len(v) else None,
                worsened_fraction=float(np.mean(delta > 1e-4)) if len(v) else None,
                tied_fraction=float(np.mean(np.abs(delta)<=1e-4)) if len(v) else None))
    return rows


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_scan(scan,out,provenance,max_views):
    import torch
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C
    from torch.utils.cpp_extension import load
    from scipy.io import loadmat
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from sample import forward,sha256,atomic_json,git
    from pixel_contribution import ensure_ninja,retained_mask
    assert os.uname().nodename=='autodl-container-d6a7459179-693ab6ee'
    out.mkdir(parents=True,exist_ok=True)
    started=time.time();dataset,pipe,g,cameras,ply,digest=load_scene(scan,out)
    transform,alignment=coordinate_check(dataset,cameras)
    commit=git('rev-parse','HEAD');status=git('status','--short');assert not status
    path=out/'metadata.json'
    metadata=json.loads(path.read_text()) if path.exists() else dict(git_commit=commit,model_sha256=digest,
        camera_names=[c.image_name for c in cameras],views=[])
    assert metadata['git_commit']==commit and metadata['model_sha256']==digest
    assert metadata['camera_names']==[c.image_name for c in cameras]
    gtpath=Path('/root/autodl-tmp/DTU_MVS_Data/Points/stl')/f'stl{scan:03d}_total.ply'
    gt=PlyData.read(gtpath)['vertex'];xyz=np.stack([gt[k] for k in ('x','y','z')],1).astype(float)
    assert np.isfinite(xyz).all();tree=cKDTree(xyz,leafsize=32);del gt,xyz
    obsfile=Path(f'/root/autodl-tmp/DTU_MVS_Data/ObsMask/ObsMask{scan}_10.mat');ob=loadmat(obsfile)
    metadata.update(status='running',scan=scan,methods=METHODS,alignment=alignment,model=str(ply),
        cfg_sha256=sha256(ply.parents[2]/'cfg_args'),cameras_npz_sha256=sha256(Path(dataset.source_path)/'cameras.npz'),
        gt_path=str(gtpath),gt_sha256=sha256(gtpath),obs_sha256=sha256(obsfile),
        gt_semantics='Independent scanned point samples, no faces; exact one-way nearest-neighbor distance, not pixelwise GT depth MAE, surface distance, Chamfer or completeness.',
        mask='camera.gt_mask>=0.5, no opacity filter',host_binary=_C.__file__,host_binary_sha256=sha256(_C.__file__),
        host_build_provenance=provenance,git_remote=git('remote','get-url','origin'),hostname=os.uname().nodename,
        coordinate_units='DTU mm after shared dataset scale_mat; representative ray-plane camera-z, not Gaussian-center depth',
        prediction='sum(original accepted w*z)/sum(original accepted w) within each stable descending-w prefix; no recomposition',
        saved_precision='float64 weighted arithmetic and exact cKDTree query; means/distances archived float32; tie tolerance 1e-4 mm',
        observation='Supplementary common intersection: all four predictions in DTU ObsMask; prediction-dependent selection, all-mask result is primary.')
    atomic_json(path,metadata)
    ensure_ninja();os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
    build=HERE/'构建_pixel_stats';build.mkdir(exist_ok=True)
    extension=load(name='gaudepth_pixel_stats_reader_v2',sources=[str(HERE/'reader.cu')],build_directory=str(build),extra_cuda_cflags=['-O3','--use_fast_math'],verbose=True)
    metadata.update(reader_binary=extension.__file__,reader_binary_sha256=sha256(extension.__file__),reader_source_sha256=sha256(HERE/'reader.cu'))
    check('cuda');bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    existing={v['name']:v for v in metadata['views']}
    for vi,cam in enumerate(cameras[:max_views] if max_views else cameras):
        if cam.image_name in existing:
            assert sha256(out/existing[cam.image_name]['file'])==existing[cam.image_name]['sha256'];continue
        t=time.time()
        with torch.no_grad():
            r,fx,fy=forward(cam,g,pipe,bg,dataset.kernel_size)
            _,rgb,alpha,normal,expected,median,_,geom,binning,image=r
            h,w=cam.image_height,cam.image_width
            if cam.gt_mask is None:raise ValueError('Explicit DTU mask required')
            mask=retained_mask(cam.gt_mask.cpu().numpy(),(h,w));index=np.flatnonzero(mask.ravel()).astype(np.int32)
            counts,_=extension.read_pixel_stats(geom,binning,image,len(g.get_xyz),w,h)
            n=counts.reshape(-1,8)[torch.from_numpy(index).cuda().long(),5];del counts
            mean=np.empty((len(index),4),np.float32);sizes=np.empty((len(index),4),np.int32)
            replay_error=0.
            for first in range(0,len(index),512):
                last=min(first+512,len(index));local=torch.from_numpy(index[first:last]).cuda()
                cap=max(1,int(n[first:last].max()))
                rec,ids,number,replay=extension.read_rays(geom,binning,image,len(g.get_xyz),w,h,fx,fy,local,cap)
                assert torch.equal(number,n[first:last])
                target=torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],1)[local.long()]
                torch.testing.assert_close(replay,target,rtol=2e-5,atol=2e-6)
                mu,selected=means_torch(rec[:,:,0],rec[:,:,1],number)
                finite=(number>0)&torch.isfinite(mu[:,0])
                torch.testing.assert_close(mu[finite,0],expected.flatten()[local.long()][finite].double(),rtol=2e-5,atol=2e-6)
                replay_error=max(replay_error,float((replay-target).abs().max()))
                mean[first:last]=mu.cpu().numpy();sizes[first:last]=selected.cpu().numpy()
            del r,geom,binning,image,rec,ids,replay,mu,selected,n,rgb,alpha,normal,expected,median
        finite=np.isfinite(mean).all(1)&(mean>0).all(1)
        errors=np.full(mean.shape,np.nan,np.float32);obs=np.zeros(mean.shape,bool)
        good=np.flatnonzero(finite)
        for j in range(4):
            points=points_from_depth(cam,index[good],mean[good,j].astype(float),transform)
            for first in range(0,len(points),32768):
                last=min(first+32768,len(points))
                errors[good[first:last],j]=tree.query(points[first:last],k=1,eps=0,workers=8)[0]
            obs[good,j]=in_observation(points,ob['ObsMask'],ob['BB'],ob['Res'])
        name=f'view_{vi:03d}.npz';temporary=out/(name+'.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,pixel_index=index,weighted_mean_depth=mean,selected_counts=sizes,
                gt_point_distance_mm=errors,in_observation=obs,image_size=np.array([h,w]))
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,out/name)
        view=dict(name=cam.image_name,index=vi,file=name,sha256=sha256(out/name),pixels=len(index),
            common_finite=int(finite.sum()),outside_observation_per_method=(~obs).sum(0).tolist(),
            replay_max_abs=replay_error,seconds=time.time()-t,
            metrics=metric_rows(scan,errors,obs.all(1)))
        metadata['views'].append(view);atomic_json(path,metadata)
        print('VIEW',scan,vi+1,len(cameras),view['seconds'],flush=True)
        if (out.parent/'stop_after_stage').exists():break
    assert sha256(ply)==digest
    metadata.update(status='complete' if len(metadata['views'])==len(cameras) else 'partial',original_model_unchanged=True,seconds=time.time()-started)
    if metadata['status']=='complete':
        values=[];obs=[]
        for view in metadata['views']:
            with np.load(out/view['file']) as a:values.append(a['gt_point_distance_mm']);obs.append(a['in_observation'].all(1))
        metadata['metrics']=metric_rows(scan,np.concatenate(values),np.concatenate(obs))
        write_csv(out/'几何精度.csv',metadata['metrics'])
    atomic_json(path,metadata);print('SCAN',scan,metadata['status'],metadata['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scan',type=int,choices=SCANS);p.add_argument('--all-scans',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--host-build-provenance',required=True)
    p.add_argument('--max-views',type=int,default=0);a=p.parse_args()
    if a.max_views<0:p.error('max-views must be nonnegative')
    if bool(a.scan)==bool(a.all_scans):p.error('choose exactly one of scan/all-scans')
    if a.all_scans:
        from sample import atomic_json,git
        a.output.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',git_commit=git('rev-parse','HEAD'),scans=SCANS,completed=[],started_at=time.time())
        for scan in SCANS:
            out=a.output/f'scan{scan}';mp=out/'metadata.json'
            previous=json.loads(mp.read_text()) if mp.exists() else {}
            if previous.get('status')=='complete':
                assert previous['git_commit']==state['git_commit']
                from sample import sha256
                for view in previous['views']:assert sha256(out/view['file'])==view['sha256']
            else:
                out.mkdir(exist_ok=True)
                with (out/'run.log').open('a',encoding='utf-8') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--scan',str(scan),'--output',str(out),
                        '--host-build-provenance',a.host_build_provenance],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            current=json.loads(mp.read_text());assert current['status']=='complete'
            state['completed'].append(scan);atomic_json(a.output/'batch.json',state)
            rows=[]
            for done in state['completed']:rows+=json.loads((a.output/f'scan{done}/metadata.json').read_text())['metrics']
            write_csv(a.output/'全部scan几何精度.csv',rows)
            print('SCAN_COMPLETED',scan,len(state['completed']),flush=True)
            if (a.output/'stop_after_stage').exists():break
        state.update(status='complete' if len(state['completed'])==len(SCANS) else 'partial',seconds=time.time()-state['started_at'])
        atomic_json(a.output/'batch.json',state)
    else:run_scan(a.scan,a.output.resolve(),a.host_build_provenance,a.max_views)
