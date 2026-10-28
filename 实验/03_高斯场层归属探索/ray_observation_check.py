"""Fixed logged rays: independent RGB tracking and conditional member explanations."""
import json
import subprocess
import time
import cv2
import numpy as np
import directional_support_probe as base
import observed_plane_check as obs


def project(P, X):
    q=np.c_[X,np.ones(len(X))]@P.T
    return q[:,:2]/q[:,2,None],q[:,2]


def valid_pixel(im, xy):
    x,y=xy
    return bool(np.isfinite(xy).all() and 0<=x<im.shape[1]-1 and 0<=y<im.shape[0]-1 and im[int(round(y)),int(round(x)),3]>127)


def weighted_quantiles(value,weight):
    order=np.argsort(value); v=value[order]; cumulative=np.cumsum(weight[order])/np.sum(weight)
    return {'median':float(v[np.searchsorted(cumulative,.5)]),'p90':float(v[np.searchsorted(cumulative,.9)])}


def process(scan,index,grid):
    pre=f'20261026_观测约束表面/observations_v1/scan{scan}/'
    paths={k:base.indexed(pre+p,index) for k,p in {'cameras_bin':'sparse/0/cameras.bin','images_bin':'sparse/0/images.bin',**{f'image{i}':f'images/{i:04d}.png' for i in range(3)}}.items()}
    for k,p in {'model':f'inputs/scan{scan}/point_cloud.ply','metadata':f'confirm_v1/scan{scan}/metadata.json','trace':f'confirm_v1/scan{scan}/view_000.npz'}.items():
        paths[k]=base.indexed('20261026_夜间深度方法/'+p,index)
    md=json.loads(paths['metadata'].read_text(encoding='utf8')); view=next(v for v in md['views'] if v['view_index']==0)
    assert base.sha(paths['model'])==md['model_sha256'] and base.sha(paths['trace'])==view['npz_sha256']
    assert all(base.sha(paths[k])==grid['inputs_sha256'][k] for k in ('cameras_bin','images_bin','image0','image1','image2'))
    cams=obs.col.read_intrinsics_binary(str(paths['cameras_bin'])); ex=obs.col.read_extrinsics_binary(str(paths['images_bin']))
    records=[next(v for v in ex.values() if v.name==f'{i:04}.png') for i in range(3)]
    R=[obs.col.qvec2rotmat(r.qvec) for r in records]; t=[r.tvec for r in records]; W=np.array(view['world_view_transform'])
    assert np.max(abs(R[0]-W[:3,:3].T))<1e-4 and np.max(abs(t[0]-W[3,:3]))<1e-4
    Ks=[]
    for rec in records:
        c=cams[rec.camera_id]; assert c.model=='PINHOLE'; fx,fy,cx,cy=c.params; Ks.append(np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]))
    Ps=[k@np.column_stack([r,z]) for k,r,z in zip(Ks,R,t)]; Fs=[]
    for j in (1,2):
        r=R[j]@R[0].T; trans=t[j]-r@t[0]; cross=np.array([[0,-trans[2],trans[1]],[trans[2],0,-trans[0]],[-trans[1],trans[0],0]])
        Fs.append(np.linalg.inv(Ks[j]).T@cross@r@np.linalg.inv(Ks[0]))
    imgs=[cv2.imdecode(np.fromfile(paths[f'image{i}'],dtype=np.uint8),cv2.IMREAD_UNCHANGED) for i in range(3)]
    assert all(im is not None and im.shape==(1162,1554,4) for im in imgs)
    gray=[cv2.cvtColor(im[:,:,:3],cv2.COLOR_BGR2GRAY) for im in imgs]
    features=grid['all_three_view_matches']; fxy=np.array([v['pixels'][0] for v in features]); z=np.load(paths['trace']); ids=z['gaussian_id']; vertices=base.ply(paths['model'])
    assert ids.min()>=0 and ids.max()<len(vertices)
    h,w=view['image_size']; fx,fy=view['focal']; center=np.array(view['camera_center']); chosen=np.linspace(0,len(z['offsets'])-2,16,dtype=int); rows=[]
    options=dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for i in chosen:
        x,y=z['pixel_xy'][i]; camray=np.array([(x-(w-1)/2)/fx,(y-(h-1)/2)/fy,1.]); direction=W[:3,:3]@camray
        uv,depth=project(Ps[0],(center+direction)[None]); ref=uv[0]; expected_ref=np.array([2*x+1.,2*y+1.]); mapping_error=float(np.linalg.norm(ref-expected_ref))
        row={'ray_index':int(i),'trace_pixel':[int(x),int(y)],'reference_native_pixel':ref.tolist(),'mapping_error_vs_2xy_plus_1':mapping_error,'status':'pending'}; rows.append(row)
        if depth[0]<=0 or not valid_pixel(imgs[0],ref): row['status']='reference_invalid'; continue
        distances=np.linalg.norm(fxy-ref,axis=1); closest=int(np.argmin(distances)); row['nearest_feature_distance_px']=float(distances[closest])
        if distances[closest]>64: row['status']='no_common_feature_within_64px'; continue
        feat=features[closest]; pixels=[ref]; tracks=[]; failure=None
        for j in (1,2):
            reference=np.array(ref,dtype='float32').reshape(1,1,2); init=np.array(ref+np.array(feat['pixels'][j])-np.array(feat['pixels'][0]),dtype='float32').reshape(1,1,2)
            dst,st,err=cv2.calcOpticalFlowPyrLK(gray[0],gray[j],reference,init,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
            if not st[0,0] or not valid_pixel(imgs[j],dst[0,0]): failure=f'view{j}_forward_or_alpha'; break
            back,bst,berr=cv2.calcOpticalFlowPyrLK(gray[j],gray[0],dst,reference.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
            if not bst[0,0]: failure=f'view{j}_backward'; break
            pos=dst[0,0].astype(float); fb=float(np.linalg.norm(back[0,0]-ref)); epi=list(map(float,obs.epiline_residuals(Fs[j-1],np.r_[ref,1.],np.r_[pos,1.])))
            tracks.append({'view':j,'pixel':pos.tolist(),'fb_px':fb,'epipolar_px':epi,'lk_error':float(err[0,0]),'back_lk_error':float(berr[0,0])})
            if fb>1 or max(epi)>2: failure=f'view{j}_fb_or_epipolar'; break
            pixels.append(pos)
        row['tracks']=tracks
        if failure: row['status']=failure; continue
        X=obs.triangulate(np.array(pixels[:2]),Ps[:2]); pred,dep=project(Ps[2],X[None]); z0=(R[0]@X+t[0])[2];z1=(R[1]@X+t[1])[2]
        if not np.isfinite(X).all() or min(z0,z1,dep[0])<=0: row['status']='triangulation_invalid'; continue
        row['anchor_xyz']=X.tolist(); row['anchor_third_view_residual_px']=float(np.linalg.norm(pred[0]-pixels[2])); row['status']='matched'
        a,b=z['offsets'][i:i+2]; p=vertices[ids[a:b]]; mu=np.stack([p[n] for n in ('x','y','z')],axis=1).astype(float)
        scales=np.exp(np.stack([p[f'scale_{j}'] for j in range(3)],axis=1).astype(float)); scales=np.sqrt(scales**2+p['filter_3D'][:,None].astype(float)**2)
        order=np.argsort(scales,axis=1); short=scales[np.arange(len(p)),order[:,0]]; unique=scales[np.arange(len(p)),order[:,1]]>short*1.001
        rot=base.rotation(np.stack([p[f'rot_{j}'] for j in range(4)],axis=1).astype(float)); normals=rot[np.arange(len(p)),:,order[:,0]]
        denom=normals@direction; plane_depth=np.sum(normals*(mu-center),axis=1)/np.where(abs(denom)>1e-8,denom,np.nan)
        original=z['depth'][a:b].astype(float); weights=z['weight'][a:b].astype(float)
        original_xy,original_z=project(Ps[2],center+original[:,None]*direction); plane_xy,plane_z=project(Ps[2],center+plane_depth[:,None]*direction)
        valid=unique&np.isfinite(plane_depth)&(plane_depth>0)&(original>0)&(original_z>0)&(plane_z>0)&np.isfinite(original_xy).all(axis=1)&np.isfinite(plane_xy).all(axis=1)
        total=float(weights.sum()); kept=float(weights[valid].sum()); row['members_total']=len(p); row['members_common_valid']=int(valid.sum()); row['accepted_weight_total']=total;row['weight_retained_fraction']=kept/total if total else None;row['invalid_plane_fraction']=float(1-np.mean(valid)) if len(p) else None
        if kept>0:
            orig=np.linalg.norm(original_xy[valid]-pixels[2],axis=1); pl=np.linalg.norm(plane_xy[valid]-pixels[2],axis=1); wt=weights[valid]
            row['member_original_residual_px']=obs.summary(orig);row['member_plane_residual_px']=obs.summary(pl)
            row['member_original_weighted_mean_px']=float(np.sum(wt*orig)/kept);row['member_plane_weighted_mean_px']=float(np.sum(wt*pl)/kept)
            row['member_original_weighted_quantiles_px']=weighted_quantiles(orig,wt);row['member_plane_weighted_quantiles_px']=weighted_quantiles(pl,wt)
        for name,j in [('expected',0),('native_median',1)]:
            val=float(z['host_target'][i,j]); pred,zp=project(Ps[2],(center+val*direction)[None]);row[name+'_third_residual_px']=float(np.linalg.norm(pred[0]-pixels[2])) if val>0 and zp[0]>0 else None
    return {'scan':scan,'inputs_sha256':{k:base.sha(p) for k,p in paths.items()},'ray_indices':chosen.tolist(),'rows':rows}


def main():
    start=time.perf_counter(); index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8')); grid_path=base.OUT/'observed_grid_plane_check.json'; grid=json.loads(grid_path.read_text(encoding='utf8'))
    results=[process(s,index,next(x for x in grid['scans'] if x['scan']==s)) for s in (37,40)]
    out={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),'independent_sift_source_sha256':base.sha(grid_path),'parameters':'16 fixed linspace rays per scan; original pixels; nearest existing three-view SIFT within 64px; LK win21 level4 iter30 eps .01, initialized and FB <=1px, alpha >.5, epipolar <=2px; filtered unique short axis relative 1e-3, plane denominator >1e-8','scans':results,'elapsed_seconds':time.perf_counter()-start,'interpretation':'Third view independent tracking is not ground truth. No training or new estimator. Observed features are engineering samples, not independent test set. Accepted IDs bounded and model SHA verified but reader/host binary origin remains unproven; no reordered or recomputed occlusion weights.'}
    dest=base.OUT/'ray_observation_check.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    for scan in results:
        print(scan['scan'],{s:sum(r['status']==s for r in scan['rows']) for s in sorted(set(r['status'] for r in scan['rows']))},'max_mapping_error',max(r['mapping_error_vs_2xy_plus_1'] for r in scan['rows']))


if __name__=='__main__': main()
