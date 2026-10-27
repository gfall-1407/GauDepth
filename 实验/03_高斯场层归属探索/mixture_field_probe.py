"""Frozen accepted-distribution fields at fixed shared sparse world voxels.

No GT query, fitting, CUDA build, dense cache or mesh extraction. A validated
independent guard must surround this executable, including transfer time.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import numpy as np
from photo_surface_probe import HERE, REPO, save, sha256, source_info
sys.path.insert(0, str(REPO))


def native_observation(z,w,t,h):
    """Native float32 validity/SDF, float64 normalized accepted-mass sum."""
    z,w,t=np.asarray(z,np.float32),np.asarray(w,float),np.asarray(t,np.float32)
    if not len(z) or z.shape!=w.shape or not np.isfinite(z).all() or not np.isfinite(w).all() or (z<=0).any() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid complete accepted ray')
    sdf=z[None,:]-t[:,None];h=np.float32(h)
    accepted=(z[None,:]<=np.float32(8))&(t[:,None]>0)&(sdf>=-h)
    mass=accepted*(w/w.sum())[None,:]
    observation=np.minimum(sdf,h)/h
    return (mass*observation).sum(1),mass.sum(1)


def save_npz(path,**arrays):
    temporary=path.with_suffix('.npz.part')
    with temporary.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def run(args):
    if socket.gethostname() != 'autodl-container-5d5147b55f-67549ee6':
        raise ValueError('Only explicitly authorized G0')
    source=source_info()
    if args.output.exists() or not args.output.is_absolute():raise ValueError('New absolute output required')
    if shutil.disk_usage(args.output.parent).free < 1536*1024**2:raise ValueError('Need 1.5GiB free before bounded probe')
    args.output.mkdir();signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop_after_view').touch())
    import torch
    from diff_gaussian_rasterization import _C
    from dense_huber_mesh import HOST_SHA,source_provenance
    from depth_accuracy import load_scene
    from sample import forward
    source_provenance()
    if sha256(_C.__file__)!=HOST_SHA:raise ValueError('Original rasterizer changed')
    config_path=HERE/'分布场固定探针.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    if sha256(HERE/'reader.cu')!='8bc2dcaa970a55d86d7deb94e9d22f927eca25c263e0d3243c6764ad6ef13613':raise ValueError('Reader source differs')
    if sha256(args.reader_binary)!='8987b98cfdca7ec5e39ed42cc899c00cc345da64b79679a55d84e460c07a3357':raise ValueError('Reader binary differs')
    spec=importlib.util.spec_from_file_location(args.reader_binary.stem,args.reader_binary)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    started=time.monotonic();budget=512*1024**2
    state=dict(status='partial',source=source,hostname=socket.gethostname(),instance_id='5d5147b55f-67549ee6',
        query_config_sha256=sha256(config_path),reader_source_sha256=sha256(HERE/'reader.cu'),
        reader_binary=str(args.reader_binary),reader_binary_sha256=sha256(args.reader_binary),
        original_C_sha256=sha256(_C.__file__),torch=torch.__version__,cuda=torch.version.cuda,scans=[],
        arithmetic='float32 native validity/SDF/clip; float64 normalized accepted-mass sums',
        policy=config['policy'],limitation='Sparse common voxels only; original E cache and accepted ray distributions, no GT, mesh, training or efficiency claim.')
    save(args.output/'metadata.json',state)
    def boundary():
        if time.monotonic()-started>args.max_seconds or (args.output/'stop_after_view').exists():raise InterruptedError('Finite deadline/stop')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>budget:raise InterruptedError('Fixed output budget reached')
        if shutil.disk_usage(args.output).free<768*1024**2:raise InterruptedError('Preserve free disk for save and transfer')
    try:
        for scan in (40,63):
            boundary();cfg=config['scans'][str(scan)];out=args.output/f'scan{scan}';out.mkdir()
            cache=args.cache40 if scan==40 else args.cache63;mp=cache/'metadata.json'
            if sha256(mp)!=cfg['cache_metadata_sha256']:raise ValueError('Sealed native cache metadata differs')
            cm=json.loads(mp.read_text(encoding='utf-8'))
            model=args.model_root/f'scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';cf=model/'cfg_args'
            if sha256(ply)!=cfg['model_sha256'] or sha256(cf)!=cfg['cfg_sha256']:raise ValueError('Protected model/cfg differs')
            for relative,digest in cfg['source_inputs'].items():
                if sha256(args.source_root/relative)!=digest:raise ValueError('Camera/data input differs')
            dataset,pipe,gaussians,cameras,_,digest=load_scene(scan,out,args.model_root,args.source_root)
            if digest!=cfg['model_sha256'] or [c.image_name for c in cameras]!=cm['camera_names']:raise ValueError('Model/camera inventory differs')
            points=np.array(cfg['voxel_keys'],np.float32)*np.float32(config['voxel_size']);n=len(points)
            if n>4096:raise ValueError('Fixed query budget exceeded')
            sums=np.zeros((n,4),np.float64);views=[]
            scene=dict(scan=scan,views=views,voxels=n,model_sha256=digest,cfg_sha256=sha256(cf),cache_metadata_sha256=sha256(mp))
            state['scans'].append(scene);bg=torch.ones(3,device='cuda',dtype=torch.float32)
            for camera,vm in zip(cameras,cm['views']):
                boundary();tick=time.monotonic();w,h=camera.image_width,camera.image_height
                # Reserve the full uncompressed worst case plus ZIP/metadata margin
                # before starting this view; compressed output cannot cross budget.
                used=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
                reserve=n*(1024*12+80)+2*1024**2
                if used+reserve>budget:raise InterruptedError('Insufficient reserved output budget for a full view')
                with torch.no_grad():
                    raw,fx,fy=forward(camera,gaussians,pipe,bg,dataset.kernel_size)
                    _,rgb,alpha,normal,expected,median,_,geom,binning,image=raw
                    fp=cache/vm['file']
                    if sha256(fp)!=vm['sha256']:raise ValueError('Sealed native frame differs')
                    with np.load(fp) as f:
                        actual=expected.cpu().numpy();reference=f['expected']
                        if actual.dtype!=reference.dtype or actual.shape!=reference.shape or actual.tobytes()!=reference.tobytes():
                            raise ValueError('Native E cache bytes differ')
                        mask=f['gt_mask'].astype(bool).copy()
                    np.testing.assert_array_equal(camera.gt_mask.cpu().numpy().squeeze()>=.5,mask)
                    em=camera.world_view_transform.T.cpu().numpy().astype(np.float32)
                    local=np.column_stack([((points[:,0]*em[j,0]+points[:,1]*em[j,1])+points[:,2]*em[j,2])+em[j,3] for j in range(3)])
                    z=local[:,2]
                    with np.errstate(divide='ignore',invalid='ignore'):
                        inv=np.float32(1)/z;u=np.float32(camera.Fx)*local[:,0]*inv+np.float32(camera.Cx);v=np.float32(camera.Fy)*local[:,1]*inv+np.float32(camera.Cy)
                    inside=np.isfinite(u)&np.isfinite(v)&(u>=0)&(v>=0)&(u<=w-1)&(v<=h-1)&(z>0)
                    point_pixels=np.full(n,-1,np.int32);point_pixels[inside]=v[inside].astype(np.int32)*w+u[inside].astype(np.int32)
                    inside[inside]&=mask.ravel()[point_pixels[inside]]
                    pixels,inverse=np.unique(point_pixels[inside],return_inverse=True)
                    point_to_ray=np.full(n,-1,np.int32);point_to_ray[inside]=inverse
                    fields=np.zeros((n,4),np.float64);capacity=256
                    if len(pixels):
                        pixel_tensor=torch.tensor(pixels,device='cuda',dtype=torch.int32)
                        while True:
                            records,ids,counts,replay=extension.read_rays(geom,binning,image,len(gaussians.get_xyz),w,h,fx,fy,pixel_tensor,capacity)
                            if int(counts.max())<=capacity:break
                            del records,ids,counts,replay;capacity*=2
                            if capacity>1024:raise ValueError('Complete accepted ray capacity exceeded')
                        target=torch.stack((expected.reshape(-1)[pixel_tensor.long()],median.reshape(-1)[pixel_tensor.long()],alpha.reshape(-1)[pixel_tensor.long()]),1)
                        torch.testing.assert_close(replay,target,rtol=2e-5,atol=2e-6,equal_nan=True)
                        counts_np=counts.cpu().numpy();offsets=np.r_[0,np.cumsum(counts_np)]
                        rr=records.cpu().numpy();ii=ids.cpu().numpy()
                        depths=np.concatenate([rr[j,:c,0] for j,c in enumerate(counts_np)])
                        weights=np.concatenate([rr[j,:c,1] for j,c in enumerate(counts_np)])
                        gids=np.concatenate([ii[j,:c] for j,c in enumerate(counts_np)])
                        native=target[:,0].cpu().numpy()
                        for ray,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
                            q=point_to_ray==ray;t=z[q].astype(float)
                            if b==a:
                                if native[ray]>0:raise ValueError('Positive native E with empty record')
                                continue
                            if not np.isfinite(native[ray]) or native[ray]<=0:raise ValueError('Nonpositive native E')
                            fields[q,0],fields[q,1]=native_observation([native[ray]],[1.],t,np.float32(config['voxel_size'])*np.float32(8))
                            fields[q,2],fields[q,3]=native_observation(depths[a:b],weights[a:b],t,np.float32(config['voxel_size'])*np.float32(8))
                        save_npz(out/f'{camera.image_name}_rays.npz',pixels=pixels,offsets=offsets,depth=depths,weight=weights,gaussian_id=gids,point_to_ray=point_to_ray,native_expected=native)
                        del records,ids,counts,replay,target,pixel_tensor
                    save_npz(out/f'{camera.image_name}_fields.npz',fields=fields,camera_z=z,world_view_transform=em)
                    sums+=fields
                    files=[out/f'{camera.image_name}_fields.npz']+([out/f'{camera.image_name}_rays.npz'] if len(pixels) else [])
                    views.append(dict(name=camera.image_name,queried_pixels=len(pixels),capacity=capacity,cache_expected_bitwise=True,
                        frame_sha256=vm['sha256'],files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in files],seconds=time.monotonic()-tick))
                    save(args.output/'metadata.json',state)
                    del raw,rgb,alpha,normal,expected,median,geom,binning,image
            save_npz(out/'pooled_fields.npz',voxel_keys=np.array(cfg['voxel_keys'],np.int32),points=points,fields=sums)
            scene.update(status='complete',pooled_fields_sha256=sha256(out/'pooled_fields.npz'),original_model_unchanged=sha256(ply)==cfg['model_sha256'],original_cfg_unchanged=sha256(cf)==cfg['cfg_sha256'])
            if not scene['original_model_unchanged'] or not scene['original_cfg_unchanged']:raise ValueError('Protected model changed')
            del gaussians,cameras;torch.cuda.empty_cache()
        state.update(status='complete_waiting_transfer',seconds=time.monotonic()-started)
        save(args.output/'metadata.json',state)
        while not (args.output/'transfer_verified').exists():boundary();time.sleep(2)
        state['status']='complete';save(args.output/'metadata.json',state)
    except BaseException as error:
        if state['status']!='complete_waiting_transfer':state['status']='partial_failed'
        state['error']=repr(error);save(args.output/'metadata.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reader-binary',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path('/root/GauDepth_HuberRepair_20261026/inputs_G1/outtest'))
    p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/DTU'))
    p.add_argument('--cache40',type=Path,default=Path('/root/GauDepth_Gap_20261027/scan40_v1/cache40'))
    p.add_argument('--cache63',type=Path,default=Path('/root/autodl-tmp/GauDepth_Gap_20261027/scan63_v1/cache63'))
    p.add_argument('--max-seconds',type=float,default=720)
    run(p.parse_args())
