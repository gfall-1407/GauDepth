"""Host forward/ray readout plus one reference-anchored support interval."""
import os,sys,json,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'01_冻结模型分层'
sys.path.insert(0,str(OLD))
from layered_depth import forward,extension as reader,protect_scene_output
LAMBDAS=[.01,.03];SUFFIXES=['l001','l003']
METHODS=['reference_depth','reference_fixed_mean']+[
    f'reference_support_{kind}_{suffix}' for suffix in SUFFIXES for kind in ['mean','wmedian']]
def method_names(lambdas):
    return ['reference_depth','reference_fixed_mean']+[
        f'reference_support_{kind}_{SUFFIXES[LAMBDAS.index(lam)]}' for lam in lambdas for kind in ['mean','wmedian']]
build=HERE/'构建';build.mkdir(exist_ok=True)
support=load(name='gaudepth_surface_support',sources=[str(HERE/'support_aggregate.cpp')],
    build_directory=str(build),extra_cflags=['-O3'],verbose=True)

def reference_mesh_scene(model_path):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.io.read_triangle_mesh(str(Path(model_path)/'recon_post.ply')))
    return scene

def reference_depth(scene,camera,fx,fy,validate=False):
    import open3d as o3d
    W,H=camera.image_width,camera.image_height;y,x=np.mgrid[:H,:W]
    rc=np.stack([(x-(W-1)/2)/fx,(y-(H-1)/2)/fy,np.ones((H,W))],-1)
    wc=camera.world_view_transform.T.cpu().numpy().astype(np.float64);cw=np.linalg.inv(wc)
    direction=rc@cw[:3,:3].T;origin=cw[:3,3]
    rays=np.concatenate([np.broadcast_to(origin,direction.shape),direction],-1).astype(np.float32)
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=16)['t_hit'].numpy()
    valid=np.isfinite(hit)&(hit>0);points=rays[valid,:3]+hit[valid,None]*rays[valid,3:]
    pc=points@wc[:3,:3].T+wc[:3,3]
    depth=np.full((H,W),np.nan,np.float32);depth[valid]=pc[:,2]
    valid&=depth>0;depth[~valid]=np.nan
    checks={}
    if validate:
        xy=np.stack([x,y],-1)[np.isfinite(hit)&(hit>0)]
        projected=np.stack([pc[:,0]/pc[:,2]*fx+(W-1)/2,pc[:,1]/pc[:,2]*fy+(H-1)/2],-1)
        error=float(np.max(np.abs(projected-xy)))
        assert error<.002,(error,'ray projection mismatch')
        checks=dict(reprojection_max_px=error,z_vs_t_max=float(np.max(np.abs(pc[:,2]-hit[np.isfinite(hit)&(hit>0)]))),
            world_to_camera=wc.tolist(),fx=fx,fy=fy,cx=(W-1)/2,cy=(H-1)/2)
    scales=(.002/np.linalg.norm(rc,axis=-1)).astype(np.float32)
    return depth,scales,valid,checks

def exhaustive_check():
    """Necessary small independent interval check, including duplicates/ties/fallbacks."""
    rng=np.random.default_rng(2718)
    cases=[([1.,1.,1.004],[.2,.3,.5],1.,.002),([.99,1.01],[.5,.5],1.,.002),
           ([1.,1.002],[.5,.5],1.,.002),([.5,1.5],[.5,.5],1.,.002),([],[],1.,.002)]
    for _ in range(35):
        n=int(rng.integers(1,16));d=np.round(rng.uniform(.98,1.02,n),3);w=rng.uniform(.001,.4,n)
        cases.append((d.tolist(),w.tolist(),float(rng.uniform(.98,1.02)),.002))
    for d,w,z,s in cases:
        n=len(d);rec=torch.zeros((1,max(1,n),2),dtype=torch.float32)
        if n:rec[0,:n]=torch.tensor(list(zip(d,w)),dtype=torch.float32)
        expected=torch.tensor([1.],dtype=torch.float32);ref=torch.tensor([z],dtype=torch.float32);scale=torch.tensor([s],dtype=torch.float32)
        actual,st=support.aggregate_support(rec,torch.tensor([n],dtype=torch.int32),expected,ref,scale,LAMBDAS)
        d=rec[0,:n,0].numpy().astype(float);w=rec[0,:n,1].numpy().astype(float);z=float(ref[0]);s=float(scale[0])
        np.testing.assert_equal(float(actual[0,0]),z)
        fixed=(d>=z-4*s)&(d<=z+4*s)
        fixed_mean=float(d[fixed]@w[fixed]/w[fixed].sum()) if fixed.any() else 1.
        np.testing.assert_allclose(float(actual[0,1]),fixed_mean,rtol=2e-5,atol=2e-6)
        if not n:assert all(actual[0,2:]==1);continue
        ends=np.unique(np.r_[d,z]);A=w.sum()
        for k,lam in enumerate(LAMBDAS):
            possibilities=[]
            for l in ends[ends<=z]:
                for u in ends[ends>=z]:
                    member=(d>=l)&(d<=u);W=w[member].sum();score=W/A-lam*(u-l)/s
                    possibilities.append((float(score),float(u-l),float(l),float(u),member))
            best=sorted(possibilities,key=lambda p:(-p[0],p[1],p[2]))[0]
            q=st[0,6+9*k:6+9*(k+1)].numpy()
            if best[0]<=0 or not best[4].any():assert q[0]==1;continue
            assert q[0]==0
            np.testing.assert_allclose(q[[4,5,6]],[best[0],best[2],best[3]],rtol=1e-10,atol=1e-10)
            ds=d[best[4]];ws=w[best[4]];order=np.argsort(ds,kind='stable')
            mu=ds@ws/ws.sum();md=ds[order][np.searchsorted(np.cumsum(ws[order]),ws.sum()/2)]
            np.testing.assert_allclose(actual[0,2+2*k:4+2*k].numpy(),[mu,md],rtol=2e-5,atol=2e-6)
    rec=torch.tensor([[[1.,.5],[1.001,.5]]],dtype=torch.float32)
    for z in [float('nan'),0.]:
        a,st=support.aggregate_support(rec,torch.tensor([2],dtype=torch.int32),torch.tensor([1.0005]),
            torch.tensor([z]),torch.tensor([.002]),LAMBDAS)
        assert st[0,0]==1 and torch.all(a==torch.tensor(1.0005,dtype=torch.float32))
    print('INTERVAL_CHECK_PASS',len(cases),'and missing-reference checks',flush=True)

def depth_maps(camera,gaussians,pipe,kernel,mesh_scene,lambdas,validate=False,capture=False):
    r,fx,fy=forward(camera,gaussians,pipe,torch.ones(3,device='cuda'),kernel)
    _,rgb,alpha,normal,expected,median,_,geom,binning,image=r
    H,W=camera.image_height,camera.image_width
    ref,scale,hit,checks=reference_depth(mesh_scene,camera,fx,fy,validate)
    ref_t=torch.from_numpy(ref.reshape(-1));scale_t=torch.from_numpy(scale.reshape(-1))
    expected_t=expected.flatten().cpu().contiguous();target=torch.stack([expected.flatten(),median.flatten(),alpha.flatten()],1)
    methods=method_names(lambdas)
    values=np.empty((H*W,len(methods)),np.float32);stat=np.zeros((H*W,6+9*len(lambdas)),np.float64)
    retries=0;replay_error=np.zeros(3)
    for start in range(0,H*W,8192):
        N=min(8192,H*W-start);K=1024
        while True:
            rec,count,replay=reader.read_rays(geom,binning,image,gaussians.get_xyz.shape[0],W,H,fx,fy,start,N,K)
            maximum=int(count.max())
            if maximum<=K:break
            K=maximum;retries+=1
        if validate:
            torch.testing.assert_close(replay,target[start:start+N],rtol=2e-5,atol=2e-6)
            replay_error=np.maximum(replay_error,(replay-target[start:start+N]).abs().amax(0).cpu().numpy())
        result,st=support.aggregate_support(rec.cpu().contiguous(),count.cpu().contiguous(),expected_t[start:start+N],
            ref_t[start:start+N].contiguous(),scale_t[start:start+N].contiguous(),lambdas)
        values[start:start+N]=result.numpy();stat[start:start+N]=st.numpy()
    maps={method:torch.from_numpy(values[:,i].copy().reshape(1,H,W)) for i,method in enumerate(methods)}
    maps['expected']=expected.cpu();maps['host_median']=median.cpu()
    mask=np.ones((H,W),bool) if camera.gt_mask is None else camera.gt_mask.squeeze().cpu().numpy()>=.5
    summary=dict(view=camera.image_name,pixels=H*W,original_mask_pixels=int(mask.sum()),
        missing_reference=int(stat[:,0].sum()),missing_reference_in_mask=int(((~hit)&mask).sum()),
        bad_records=int(stat[:,1].sum()),fixed_empty=int(stat[:,2].sum()),overflow_retries=retries,
        replay_max_abs=replay_error.tolist() if validate else None,coordinate_check=checks,lambdas={})
    fixed_used=hit.reshape(-1)&(stat[:,2]==0)&(stat[:,5]>0)
    summary['fixed_support']=dict(used=int(fixed_used.sum()),
        retained_contribution_sum=float(stat[fixed_used,3].sum()),span_voxel_sum=float(stat[fixed_used,4].sum()),
        member_count_sum=float(stat[fixed_used,5].sum()))
    for k,lam in enumerate(lambdas):
        suffix=SUFFIXES[LAMBDAS.index(lam)]
        q=stat[:,6+9*k:6+9*(k+1)];used=hit.reshape(-1)&(q[:,0]==0)&(q[:,3]>0)
        summary['lambdas'][suffix]=dict(insufficient_support=int(q[:,0].sum()),used=int(used.sum()),
            span_voxel_sum=float(q[used,1].sum()),retained_contribution_sum=float(q[used,2].sum()),
            member_count_sum=float(q[used,3].sum()),mean_median_distance_voxel_sum=float(q[used,7].sum()),
            span_voxel_quantiles=np.quantile(q[used,1],[.1,.5,.9]).tolist() if used.any() else None,
            retained_quantiles=np.quantile(q[used,2],[.1,.5,.9]).tolist() if used.any() else None)
    samples=[]
    if capture:
        # Few actual-image pixels: a normal interior, both directions of readout change,
        # maximal same-support mean/median difference, and a no-reference control.
        valid=hit&mask&(alpha[0].cpu().numpy()>.95)
        e=expected[0].cpu().numpy();fixed=values[:,1].reshape(H,W);mean=values[:,2].reshape(H,W);med=values[:,3].reshape(H,W)
        ys,xs=np.where(valid)
        if len(xs):
            metric=np.abs(mean-ref)-np.abs(fixed-ref);dist=(xs-W*.5)**2+(ys-H*.5)**2
            indices=[('interior',int(np.argmin(dist))),('toward_reference',int(np.argmin(metric[ys,xs]))),
                ('away_from_reference',int(np.argmax(metric[ys,xs]))),('location_difference',int(np.argmax(np.abs(mean[ys,xs]-med[ys,xs]))))]
            chosen=[(label,int(xs[j]),int(ys[j])) for label,j in indices]
            yy,xx=np.where(~hit&mask)
            if len(xx):chosen.append(('missing_reference',int(xx[len(xx)//2]),int(yy[len(yy)//2])))
            for label,x,y in chosen:
                p=y*W+x;K=1024
                while True:
                    rec,count,replay=reader.read_rays(geom,binning,image,gaussians.get_xyz.shape[0],W,H,fx,fy,p,1,K)
                    n=int(count[0])
                    if n<=K:break
                    K=n
                samples.append(dict(label=label,x=x,y=y,reference=float(ref[y,x]) if hit[y,x] else None,scale_z=float(scale[y,x]),
                    expected=float(e[y,x]),fixed=float(fixed[y,x]),depth_weight=rec[0,:n].cpu().numpy().tolist(),
                    values={m:float(values[p,i]) for i,m in enumerate(methods)},statistics=stat[p].tolist(),lambdas=lambdas))
    return rgb.cpu(),maps,summary,hit,samples
