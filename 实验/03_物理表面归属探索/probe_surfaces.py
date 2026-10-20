"""Small physical-reference audit; original host forward and original weights.

This is a diagnostic, not a trainable estimator. Reference-based membership
is provisional and must not be reported as independently verified truth.
"""
import os
os.environ.setdefault('MAX_JOBS','2')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
import sys,json,math,time,argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2,getProjectionMatrix,focal2fov
from utils.general_utils import build_rotation
from diff_gaussian_rasterization import _C


def camera_from_json(entry,resolution):
    W,H=round(entry['width']/resolution),round(entry['height']/resolution)
    R=np.asarray(entry['rotation'],np.float64)
    center=np.asarray(entry['position'],np.float64)
    T=-R.T@center
    fovx=focal2fov(entry['fx'],entry['width'])
    fovy=focal2fov(entry['fy'],entry['height'])
    view=torch.tensor(getWorld2View2(R,T)).T.cuda()
    projection=getProjectionMatrix(.01,100.,fovx,fovy).T.cuda()
    return MiniCam(W,H,fovy,fovx,.01,100.,view,view@projection)


def forward(cam,pc,kernel):
    scales,opacity=pc.get_scaling_n_opacity_with_3D_filter
    empty=torch.empty(0,device='cuda')
    tx,ty=math.tan(cam.FoVx/2),math.tan(cam.FoVy/2)
    r=_C.rasterize_gaussians(torch.zeros(3,device='cuda'),pc.get_xyz,empty,
        opacity,scales,pc.get_rotation,1.,empty,cam.world_view_transform,
        cam.full_proj_transform,tx,ty,kernel,cam.image_height,cam.image_width,
        pc.get_features,pc.active_sh_degree,cam.camera_center,False,True,False)
    fx=float(np.float32(cam.image_width)/(np.float32(2)*np.float32(tx)))
    fy=float(np.float32(cam.image_height)/(np.float32(2)*np.float32(ty)))
    return r,fx,fy


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    build=HERE/'构建';build.mkdir(exist_ok=True)
    reader=load(name='gaudepth_physical_members',sources=[str(HERE/'ray_members.cu')],
        build_directory=str(build),extra_cuda_cflags=['-O3','--use_fast_math'],verbose=True)
    model=Path(args.model)
    # cfg_args is trusted local host configuration, parse AST without eval.
    import ast
    tree=ast.parse((model/'cfg_args').read_text()).body[0].value
    cfg={kw.arg:ast.literal_eval(kw.value) for kw in tree.keywords}
    pc=GaussianModel(cfg['sh_degree'])
    pc.load_ply(str(model/'point_cloud'/f'iteration_{args.iteration}'/'point_cloud.ply'))
    entries=json.loads((model/'cameras.json').read_text())
    mesh=o3d.t.io.read_triangle_mesh(str(model/'recon_post.ply'))
    ray_scene=o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh)
    verts=mesh.vertex['positions'].numpy();faces=mesh.triangle['indices'].numpy()
    tri_n=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
    tri_n/=np.maximum(np.linalg.norm(tri_n,axis=1,keepdims=True),1e-20)
    manifest=dict(model=str(model),iteration=args.iteration,config=cfg,
        reference='baseline recon_post.ply triangle surface, not dense TSDF field',
        protocol='fixed regular grid plus largest reference residuals; diagnostic biased sample',
        surface_labels='nearest reference intersection inside 1/2/4/8 normal-voxel bands, provisional',
        voxel_size=.002,grid=args.grid,extremes=args.extremes,
        gpu=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),
        views=[],limitations=['reference is not independent GT',
            'no ground-truth Gaussian membership','hidden early-stopped contributors are unobserved'])
    for name in args.views:
        tic=time.time()
        entry=next(c for c in entries if c['img_name']==name)
        cam=camera_from_json(entry,cfg['resolution'])
        r,fx,fy=forward(cam,pc,cfg.get('kernel_size',0.))
        _,rgb,alpha,normal,expected,median,_,geom,binning,img=r
        W,H=cam.image_width,cam.image_height
        yy,xx=np.mgrid[:H,:W]
        rc=np.stack([(xx-(W-1)/2)/fx,(yy-(H-1)/2)/fy,np.ones((H,W))],axis=-1)
        c2w=np.linalg.inv(cam.world_view_transform.T.cpu().numpy().astype(np.float64))
        direction=rc@c2w[:3,:3].T
        rays=np.concatenate([np.broadcast_to(c2w[:3,3],direction.shape),direction],axis=-1).astype(np.float32)
        hit=ray_scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)['t_hit'].numpy()
        valid=np.isfinite(hit)&(hit>0)
        ex=expected[0].cpu().numpy()
        # Uniform positions are fixed before looking at scores; extra extremes
        # are intentionally diagnostic and are never pooled as unbiased rates.
        uniform={(int(x),int(y)) for y in np.linspace(0,H-1,args.grid,dtype=int)
                for x in np.linspace(0,W-1,args.grid,dtype=int)}
        points=set(uniform)
        error=np.where(valid,np.abs(ex-hit),-1).ravel()
        for p in (np.argsort(error)[-args.extremes:] if args.extremes>0 else []):
            if error[p]>=0:points.add((int(p%W),int(p//W)))
        points=sorted(points,key=lambda p:p[1]*W+p[0])
        pixels=torch.tensor([y*W+x for x,y in points],device='cuda',dtype=torch.int32)
        K=1024;retries=0
        while True:
            records,ids,counts,replay=reader.read_rays(geom,binning,img,pixels,len(pc.get_xyz),W,H,fx,fy,K)
            maximum=int(counts.max())
            if maximum<=K:break
            K=maximum;retries+=1
        target=torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],dim=1)[pixels.long()]
        torch.testing.assert_close(replay,target,rtol=2e-5,atol=2e-6)
        replay_error=float((replay-target).abs().max())
        selected_rays=rays.reshape(-1,6)[pixels.cpu().numpy()]
        intersections={k:v.numpy() for k,v in ray_scene.list_intersections(o3d.core.Tensor(selected_rays),nthreads=8).items()}
        records=records.cpu().numpy();ids=ids.cpu().numpy();counts=counts.cpu().numpy()
        scale=pc.get_scaling_with_3D_filter
        R=build_rotation(pc.get_rotation)
        cases=[];packed={}
        for j,(x,y) in enumerate(points):
            dw=records[j,:counts[j]];mid=ids[j,:counts[j]]
            mask=intersections['ray_ids']==j
            zz=intersections['t_hit'][mask].astype(float)
            tids=intersections['primitive_ids'][mask].astype(int)
            order=np.argsort(zz,kind='stable');zz,tids=zz[order],tids[order]
            keep=np.r_[True,np.diff(zz)>1e-5] if len(zz) else np.zeros(0,bool)
            zz,tids=zz[keep],tids[keep]
            zz,tids=zz[zz>0],tids[zz>0]
            case=dict(x=x,y=y,sample_source='uniform' if (x,y) in uniform else 'extreme',
                members=int(counts[j]),expected=float(ex[y,x]),
                median=float(median[0,y,x]),reference_depths=zz.tolist(),
                reference_triangles=tids.tolist(),reference_normals=tri_n[tids].tolist(),
                ray_origin=c2w[:3,3].tolist(),ray_direction=direction[y,x].tolist(),bands={})
            if len(zz) and len(dw):
                d,w=dw[:,0].astype(float),dw[:,1].astype(float)
                nearest=np.argmin(np.abs(d[:,None]-zz[None,:]),axis=1)
                nr_world=tri_n[tids[nearest]]
                nr_cam=nr_world@c2w[:3,:3]
                factor=np.abs(nr_cam@rc[y,x])
                en=np.abs(d-zz[nearest])*factor
                xyz=pc.get_xyz[mid.astype(np.int64)].cpu().numpy()
                xref=c2w[:3,3]+zz[nearest,None]*direction[y,x]
                centre_res=np.abs(np.sum(nr_world*(xyz-xref),axis=1))
                q=R[mid.astype(np.int64)].cpu().numpy()
                ss=scale[mid.astype(np.int64)].cpu().numpy()
                sigma_n=np.linalg.norm(np.einsum('ni,nij->nj',nr_world,q)*ss,axis=1)
                align=np.abs(np.sum(dw[:,2:5]*nr_cam,axis=1))
                for width in [1,2,4,8]:
                    ok=en<=width*.002
                    # Prevent overlapping bands across neighbouring intersections.
                    if len(zz)>1:
                        gap=np.abs(zz[:,None]-zz[None,:]);np.fill_diagonal(gap,np.inf)
                        ok&=(np.abs(d-zz[nearest])<.5*gap.min(axis=1)[nearest])
                    groups=[]
                    for k in np.unique(nearest[ok]):
                        sel=ok&(nearest==k);mass=w[sel].sum();mu=float(w[sel]@d[sel]/mass)
                        groups.append(dict(k=int(k),reference=float(zz[k]),count=int(sel.sum()),
                            mass_share=float(mass/w.sum()),mean=mu,
                            radial_bias_voxel=float((mu-zz[k])*np.linalg.norm(rc[y,x])/.002),
                            normal_bias_voxel=float((mu-zz[k])*abs((tri_n[tids[k]]@c2w[:3,:3])@rc[y,x])/.002),
                            std_z=float(np.sqrt(w[sel]@((d[sel]-mu)**2)/mass)),
                            normal_agreement_median=float(np.median(align[sel])),
                            centre_normal_distance_median=float(np.median(centre_res[sel])),
                            normal_scale_median=float(np.median(sigma_n[sel]))))
                    case['bands'][str(width)]=groups
                packed[f'ray_{j}_audit']=np.stack([nearest,en,centre_res,sigma_n,align],axis=1)
            cases.append(case)
            packed[f'ray_{j}_records']=dw
            packed[f'ray_{j}_ids']=mid
        dest=out/name;dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'射线支持.npz',pixels=np.array(points),**packed)
        (dest/'表面归属.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
        info=dict(view=name,width=W,height=H,points=len(points),replay_max_abs=replay_error,
                  capacity_retries=retries,seconds=time.time()-tic,
                  reference_hit_samples=sum(bool(c['reference_depths']) for c in cases),
                  multi_intersection_samples=sum(len(c['reference_depths'])>1 for c in cases))
        manifest['views'].append(info)
        (out/'运行记录.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        print('VIEW_DONE',info,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--iteration',type=int,default=30000)
    parser.add_argument('--views',nargs='+',default=['0000'])
    parser.add_argument('--grid',type=int,default=16)
    parser.add_argument('--extremes',type=int,default=16)
    args=parser.parse_args()
    with torch.no_grad():run(args)
