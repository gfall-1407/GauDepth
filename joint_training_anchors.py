"""Read-only native cameras and fixed COLMAP track pairs for a matched pilot.

No Scene construction, dataset conversion, mesh GT, Gaussian normal or plane.
All camera metadata is JSON-safe and preserves original COLMAP intrinsics.
"""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from joint_tv_inputs import sha256


def raw_intrinsics(intr):
    if intr.model == 'PINHOLE':
        fx,fy,cx,cy=intr.params
    elif intr.model == 'SIMPLE_PINHOLE':
        fx,cx,cy=intr.params;fy=fx
    else:
        raise ValueError('only undistorted PINHOLE/SIMPLE_PINHOLE cameras supported')
    return tuple(float(x) for x in (fx,fy,cx,cy))


def load_cameras(source_path,resolution=2,data_device='cpu',*,train_only=False,eval_split=False):
    """Return (native Camera list, JSON-safe metadata), reading inputs only."""
    if resolution not in (1,2,4,8): raise ValueError('resolution must be 1,2,4 or8')
    source=Path(source_path).resolve();sparse=source/'sparse'/'0'
    intr=read_intrinsics_binary(str(sparse/'cameras.bin'))
    extr=read_extrinsics_binary(str(sparse/'images.bin'))
    ordered=sorted(extr.values(),key=lambda x:Path(x.name).stem)
    if len({Path(x.name).stem for x in ordered})!=len(ordered):
        raise ValueError('duplicate image basenames')
    if train_only and eval_split:
        ordered=[x for i,x in enumerate(ordered) if i%8!=0]
    cameras=[];records=[];centers=[]
    args=SimpleNamespace(resolution=resolution,data_device=data_device)
    for index,ext in enumerate(ordered):
        k=intr[ext.camera_id];fx,fy,cx,cy=raw_intrinsics(k)
        path=(source/'images'/Path(ext.name).name).resolve()
        if not path.is_relative_to(source): raise ValueError('image outside source')
        with Image.open(path) as im:
            if im.size!=(k.width,k.height): raise ValueError('COLMAP/image size mismatch')
            image=im.copy()
        # Same centered-intrinsic camera and resize code as native training.
        R=qvec2rotmat(ext.qvec).T;T=np.asarray(ext.tvec)
        info=SimpleNamespace(uid=int(ext.id),R=R,T=T,FovX=focal2fov(fx,k.width),
            FovY=focal2fov(fy,k.height),image=image,image_name=Path(ext.name).stem,
            depth=None,normal=None)
        camera=loadCam(args,index,info,1.0);image.close()
        cameras.append(camera);centers.append((-R@T).tolist())
        records.append(dict(index=index,image_id=int(ext.id),name=camera.image_name,
            raw_width=int(k.width),raw_height=int(k.height),raw_fx=fx,raw_fy=fy,
            raw_cx=cx,raw_cy=cy,width=camera.image_width,height=camera.image_height,
            fx=camera.Fx,fy=camera.Fy,cx=camera.Cx,cy=camera.Cy,
            image_path=str(path),image_sha256=sha256(path)))
    cc=np.asarray(centers,dtype=np.float64)
    radius=float(np.linalg.norm(cc-cc.mean(0),axis=1).max()*1.1)
    if not math.isfinite(radius) or radius<=0: raise ValueError('invalid camera extent')
    return cameras,dict(source_path=str(source),resolution=resolution,
        source_sha256={n:sha256(sparse/n) for n in ('cameras.bin','images.bin')},
        camera_records=records,camera_centers=centers,scene_radius=radius,
        split='native_train_excluding_every8' if train_only and eval_split else 'all',
        camera_convention='native centered principal point; COLMAP rays remapped explicitly')


def colmap_to_native(xy,record):
    xy=np.asarray(xy,dtype=np.float64)
    return np.stack(((xy[...,0]-record['raw_cx'])/record['raw_fx']*record['fx']+record['cx'],
                     (xy[...,1]-record['raw_cy'])/record['raw_fy']*record['fy']+record['cy']),-1)


def candidate_track_groups(source_path,metadata,*,neighbors=2,max_groups=32,
                           candidate_limit=128,seed=713,border=2.):
    """Deterministic nearest-camera groups, using actual shared COLMAP track IDs.

    Candidate ordering/limits are fixed before any depth evaluation. Selection
    bias and group coverage are subsequently recorded in the experiment file.
    """
    if neighbors<1 or max_groups<1 or candidate_limit<1 or border<1:
        raise ValueError('invalid track-pool settings')
    source=Path(source_path);ext=read_extrinsics_binary(str(source/'sparse'/'0'/'images.bin'))
    records=metadata['camera_records'];tracks=[]
    for record in records:
        im=ext[record['image_id']];native=colmap_to_native(im.xys,record)
        valid=(im.point3D_ids>=0)&np.isfinite(native).all(-1)
        valid&=(native[:,0]>=border)&(native[:,0]<=record['width']-1-border)
        valid&=(native[:,1]>=border)&(native[:,1]<=record['height']-1-border)
        # Duplicate observations of one track in the same image are ambiguous.
        ids=im.point3D_ids[valid];xy=native[valid];raw=im.xys[valid]
        unique,counts=np.unique(ids,return_counts=True);single=set(unique[counts==1].tolist())
        tracks.append({int(t):(p.tolist(),r.tolist()) for t,p,r in zip(ids,xy,raw) if int(t) in single})
    centers=np.asarray(metadata['camera_centers']);dist=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    edges=set()
    for i in range(len(records)):
        for j in np.argsort(dist[i])[1:neighbors+1]:
            if dist[i,j]>1e-8:edges.add(tuple(sorted((i,int(j)))))
    rng=np.random.default_rng(seed);edges=sorted(edges);rng.shuffle(edges)
    groups=[]
    for i,j in edges:
        ids=sorted(tracks[i].keys()&tracks[j].keys())
        if not ids:continue
        rng.shuffle(ids);ids=ids[:candidate_limit]
        groups.append(dict(camera_indices=[i,j],camera_names=[records[i]['name'],records[j]['name']],
            tracks=[dict(point3D_id=t,xy=[tracks[i][t][0],tracks[j][t][0]],
                         raw_colmap_xy=[tracks[i][t][1],tracks[j][t][1]]) for t in ids]))
        if len(groups)>=max_groups:break
    if not groups:raise ValueError('no shared COLMAP tracks in the declared camera pairs')
    return groups


def sample_map(image,xy):
    """Bilinear sample a native [1,H,W]/[H,W] map at native pixel coordinates."""
    if image.ndim==2:image=image[None]
    if image.ndim!=3 or image.shape[0]!=1:raise ValueError('one-channel map required')
    xy=torch.as_tensor(xy,device=image.device,dtype=image.dtype)
    h,w=image.shape[-2:]
    grid=torch.stack((2*xy[:,0]/(w-1)-1,2*xy[:,1]/(h-1)-1),-1)
    return F.grid_sample(image[None],grid[None,None],mode='bilinear',
                         padding_mode='zeros',align_corners=True).reshape(-1)


def make_schedule(groups,steps,pairs_per_step,seed):
    if steps<1 or pairs_per_step<1:raise ValueError('positive schedule size required')
    if any(len(g['tracks'])<pairs_per_step for g in groups):raise ValueError('undersized fixed pool')
    rng=np.random.default_rng(seed);schedule=[];order=[]
    for step in range(steps):
        if not order:order=rng.permutation(len(groups)).tolist()
        group_index=order.pop();group=groups[group_index]
        anchors=rng.choice(len(group['tracks']),size=pairs_per_step,replace=False).tolist()
        schedule.append(dict(group_index=group_index,anchor_indices=anchors,rgb_side=step%2))
    return schedule
