"""Predeclared three-view observation check; fixed patch, no GT or estimator fitting."""
import importlib.util
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import directional_support_probe as base

spec=importlib.util.spec_from_file_location('colmap_loader',base.REPO/'scene/colmap_loader.py')
col=importlib.util.module_from_spec(spec); spec.loader.exec_module(col)

def epiline_residuals(F,x,y):
    l=F@x; l0=F.T@y
    return abs(y@l)/np.linalg.norm(l[:2]),abs(x@l0)/np.linalg.norm(l0[:2])

def epiline_self_check():
    K=np.array([[800.,0,320],[0,800.,240],[0,0,1.]])
    t=np.array([.3,.1,.05]); tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    angle=.2; R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    F=np.linalg.inv(K).T@tx@R@np.linalg.inv(K)
    def project(v):
        q=K@v; return np.r_[q[:2]/q[2],1.]
    errors=[]; wrong=[]
    for X in (np.array([.2,.3,3.]),np.array([-.5,.4,4.]),np.array([.7,-.2,2.])):
        x=project(X); y=project(R@X+t); base_error=epiline_residuals(F,x,y); errors.append(max(base_error))
        wrong.append(abs(y@(F.T@y))/np.linalg.norm((F.T@y)[:2]))
        assert max(base_error)<1e-10
        for dx,dy in ((np.array([0.,5.,0.]),np.zeros(3)),(np.zeros(3),np.array([0.,5.,0.]))):
            assert min(epiline_residuals(F,x+dx,y+dy))>1.
    assert min(wrong)>1.
    return {'camera_rotation_y_radians':angle,'translation':t.tolist(),'max_perfect_residual_px':float(max(errors)),'old_wrong_residuals_px':wrong,'both_perturbations_detected':True}

def triangulate(points, projections):
    A=np.concatenate([np.stack([x*P[2]-P[0],y*P[2]-P[1]]) for (x,y),P in zip(points,projections)])
    _,_,vh=np.linalg.svd(A); return vh[-1,:3]/vh[-1,3]

def plane(points, ids):
    a,b,c=points[list(ids)]; n=np.cross(b-a,c-a); length=np.linalg.norm(n)
    if length<1e-10: return None
    return a,n/length

def summary(a):
    a=np.asarray(a); return {'count':len(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(np.max(a))} if len(a) else {'count':0}

def process(scan, index, selection='maximum_response'):
    pre=f'20261026_观测约束表面/observations_v1/scan{scan}/'
    inputs={k:base.indexed(pre+p,index) for k,p in {'camera_npz':'cameras.npz','cameras_bin':'sparse/0/cameras.bin','images_bin':'sparse/0/images.bin',**{f'image{i}':f'images/{i:04d}.png' for i in range(3)}}.items()}
    inputs['metadata']=base.indexed(f'20261026_夜间深度方法/confirm_v1/scan{scan}/metadata.json',index)
    meta=json.loads(inputs['metadata'].read_text(encoding='utf8')); view=next(v for v in meta['views'] if v['view_index']==0)
    cams=col.read_intrinsics_binary(str(inputs['cameras_bin'])); extr=col.read_extrinsics_binary(str(inputs['images_bin']))
    images=[next(im for im in extr.values() if im.name==f'{i:04d}.png') for i in range(3)]
    W=np.array(view['world_view_transform']); rotations=[col.qvec2rotmat(im.qvec) for im in images]; ts=[im.tvec for im in images]
    re=float(np.max(np.abs(rotations[0]-W[:3,:3].T))); te=float(np.max(np.abs(ts[0]-W[3,:3])))
    out={'scan':scan,'inputs_sha256':{k:base.sha(p) for k,p in inputs.items()},'pose_rotation_max_abs':re,'pose_translation_max_abs':te,'status':'coordinate_mismatch'}
    if re>1e-4 or te>1e-4: return out
    cam=cams[images[0].camera_id]; assert cam.model=='PINHOLE' and (cam.width,cam.height)==(1554,1162)
    Ks=[]
    for im in images:
        c=cams[im.camera_id]; assert c.model=='PINHOLE' and (c.width,c.height)==(1554,1162)
        fx,fy,cx,cy=c.params; Ks.append(np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]))
    Ps=[K@np.column_stack([R,t]) for K,R,t in zip(Ks,rotations,ts)]
    out['native_intrinsics']=[K.tolist() for K in Ks]; out['pixel_note']='native K coordinates; trace width 777 uses center 388 vs native cx/2 388.5'
    sift=cv2.SIFT_create(); features=[]; counts=[]
    for i in range(3):
        raw=cv2.imdecode(np.fromfile(inputs[f'image{i}'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        assert raw is not None and raw.shape==(1162,1554,4)
        gray=cv2.cvtColor(raw[:,:,:3],cv2.COLOR_BGR2GRAY); mask=np.where(raw[:,:,3]>127,255,0).astype('uint8')
        kp,des=sift.detectAndCompute(gray,mask); features.append((kp,des)); counts.append(len(kp))
    out['sift_keypoint_counts']=counts
    matcher=cv2.BFMatcher(); pairmaps=[]; paircounts=[]
    for j in (1,2):
        forward=matcher.knnMatch(features[0][1],features[j][1],k=2); back=matcher.knnMatch(features[j][1],features[0][1],k=2)
        fwd={a.queryIdx:a.trainIdx for pair in forward if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance}
        rev={a.queryIdx:a.trainIdx for pair in back if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance}
        mutual={a:b for a,b in fwd.items() if rev.get(b)==a}
        R=rotations[j]@rotations[0].T; t=ts[j]-R@ts[0]; tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
        F=np.linalg.inv(Ks[j]).T@tx@R@np.linalg.inv(Ks[0]); accepted={}
        for a,b in mutual.items():
            x=np.r_[features[0][0][a].pt,1.]; y=np.r_[features[j][0][b].pt,1.]
            if max(epiline_residuals(F,x,y))<=2: accepted[a]=b
        stored={int(k):xy for k,xy in zip(images[j].point3D_ids,images[j].xys) if k>=0}
        checks=[epiline_residuals(F,np.r_[xy,1.],np.r_[stored[int(k)],1.]) for k,xy in zip(images[0].point3D_ids,images[0].xys) if k>=0 and int(k) in stored]
        checks=np.asarray(checks)
        pairmaps.append(accepted); paircounts.append({'lowe_forward':len(fwd),'lowe_reverse':len(rev),'mutual':len(mutual),'epipolar_2px':len(accepted)})
        paircounts[-1]['colmap_shared_id_residual_check']={'count':len(checks),'forward':summary(checks[:,0]),'reverse':summary(checks[:,1]),'both_within_2px':int(np.sum(np.max(checks,axis=1)<=2))} if len(checks) else {'count':0}
    out['pairs']=paircounts; common=sorted(set(pairmaps[0])&set(pairmaps[1])); out['three_view_before_triangulation']=len(common)
    centers=[-R.T@t for R,t in zip(rotations,ts)]; good=[]; positive=0
    for i in common:
        pixel=np.array([features[0][0][i].pt,features[1][0][pairmaps[0][i]].pt,features[2][0][pairmaps[1][i]].pt]); X=triangulate(pixel,Ps)
        if not np.isfinite(X).all() or not all((R@X+t)[2]>0 for R,t in zip(rotations,ts)): continue
        positive+=1; repro=[]
        for P,uv in zip(Ps,pixel):
            q=P@np.r_[X,1]; repro.append(float(np.linalg.norm(q[:2]/q[2]-uv)))
        if max(repro)>2: continue
        dirs=[(X-C)/np.linalg.norm(X-C) for C in centers]
        angles=[np.rad2deg(np.arccos(np.clip(dirs[a]@dirs[b],-1,1))) for a,b in itertools.combinations(range(3),2)]
        good.append({'reference_id':i,'pixels':pixel.tolist(),'xyz':X.tolist(),'reprojection_px':repro,'min_triangulation_angle_degrees':float(min(angles)),'response':float(features[0][0][i].response)})
    out.update(positive_depth_count=positive,three_view_reprojection_2px_count=len(good),angle_summary_degrees=summary([v['min_triangulation_angle_degrees'] for v in good]))
    if not good: out['status']='no_three_view_matches'; return out
    if selection=='fixed_grid_density':
        out['all_three_view_matches']=good
        unique={}
        for v in good:
            key=tuple(round(float(x),3) for x in v['pixels'][0])
            if key not in unique or v['response']>unique[key]['response']: unique[key]=v
        windows={}
        for v in unique.values():
            x,y=v['pixels'][0]; key=(int(np.floor(y/64)),int(np.floor(x/64)))
            windows.setdefault(key,[]).append(v)
        key=min(windows,key=lambda k:(-len(windows[k]),k[0],k[1])); row,column=key
        x0,y0=column*64,row*64; x1,y1=min(x0+64,cam.width),min(y0+64,cam.height)
        patch=windows[key]; cx,cy=(x0+x1)/2,(y0+y1)/2
        out.update(selection=selection,selection_definition='round reference xy to 0.001 px; retain maximum response; select max-count 64px grid window, ties row then col; bounds half-open; no model/GT/plane selection',unique_common_points=len(unique),window_counts=[{'row':k[0],'col':k[1],'count':len(v)} for k,v in sorted(windows.items())],selected_window={'row':row,'col':column,'bounds':[x0,y0,x1,y1]})
        anchor=max(patch,key=lambda x:x['response'])
    else:
        anchor=max(good,key=lambda x:x['response']); cx,cy=anchor['pixels'][0]; patch=[v for v in good if abs(v['pixels'][0][0]-cx)<=32 and abs(v['pixels'][0][1]-cy)<=32]
        out['selection']=selection
    patch=sorted(patch,key=lambda x:-x['response'])[:32]; out.update(patch_center=[cx,cy],patch_reference_id=anchor['reference_id'],patch_match_count=len(patch),patch_matches=patch)
    if len(patch)<4: out['status']='fixed_patch_fewer_than_four'; return out
    xyz=np.array([r['xyz'] for r in patch]); triples=list(itertools.combinations(range(len(xyz)),3)); ids=max(triples,key=lambda ids:np.linalg.norm(np.cross(xyz[ids[1]]-xyz[ids[0]],xyz[ids[2]]-xyz[ids[0]])))
    pl=plane(xyz,ids); out['triangle_indices']=list(ids)
    singular=np.linalg.svd(np.stack([xyz[ids[1]]-xyz[ids[0]],xyz[ids[2]]-xyz[ids[0]]]),compute_uv=False)
    out['triangle_singular_values']=singular.tolist(); out['triangle_condition_number']=float(singular[0]/singular[1]) if singular[1]>0 else None
    if pl is None: out['status']='fixed_patch_collinear'; return out
    if min(r['min_triangulation_angle_degrees'] for r in patch)<1.: out['status']='fixed_patch_low_parallax_below_1_degree'; return out
    p,n=pl; hold=[i for i in range(len(xyz)) if i not in ids]; residual=np.abs((xyz[hold]-p)@n)
    out['held_out_plane_distance_scene_units']=summary(residual); out['plane_point']=p.tolist(); out['plane_normal']=n.tolist()
    perturb=[]
    for sign in (-1,1):
        shifted=[]
        for r in patch:
            pixels=np.array(r['pixels']); pixels[0]+=sign*.5; shifted.append(triangulate(pixels,Ps))
        shifted=np.array(shifted); pl2=plane(shifted,ids)
        if pl2:
            dot=np.clip(abs(n@pl2[1]),0,1); perturb.append({'reference_pixel_shift':sign*.5,'max_point_shift':float(np.max(np.linalg.norm(shifted-xyz,axis=1))),'plane_normal_angle_deg':float(np.rad2deg(np.arccos(dot)))})
    out['pixel_perturbation_sensitivity']=perturb; out['pixel_perturbation_note']='Only simultaneous +/-0.5 reference pixel x and y; finite sensitivity examples, not a total error bound.'
    model=base.indexed(f'20261026_夜间深度方法/inputs/scan{scan}/point_cloud.ply',index); assert base.sha(model)==meta['model_sha256']; out['inputs_sha256']['model']=base.sha(model)
    verts=base.ply(model); mu=np.stack([verts[k] for k in ('x','y','z')],axis=1).astype(float); projected=mu@Ps[0][:,:3].T+Ps[0][:,3]; positive=projected[:,2]>0; xy=projected[:,:2]/projected[:,2,None]
    if selection=='fixed_grid_density': chosen=positive&(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
    else: chosen=positive&(np.abs(xy[:,0]-cx)<=32)&(np.abs(xy[:,1]-cy)<=32)
    v=verts[chosen]; points=mu[chosen]
    scales=np.exp(np.stack([v[f'scale_{j}'] for j in range(3)],axis=1).astype(float)); scales=np.sqrt(scales**2+v['filter_3D'][:,None].astype(float)**2)
    order=np.argsort(scales,axis=1); short=scales[np.arange(len(v)),order[:,0]]; second=scales[np.arange(len(v)),order[:,1]]; unique=second>short*(1+1e-3)
    rot=base.rotation(np.stack([v[f'rot_{j}'] for j in range(4)],axis=1).astype(float)); axes=rot[np.arange(len(v)),:,order[:,0]]
    angles=np.rad2deg(np.arccos(np.clip(np.abs(axes[unique]@n),0,1)))
    out['gaussian_projected_patch']={'all_projected_count':len(v),'unique_axis_count':int(unique.sum()),'nonunique_axis_count':int((~unique).sum()),'nonunique_relative_tolerance':1e-3,'center_abs_distance_scene_units':summary(np.abs((points-p)@n)),'normal_angle_deg':summary(angles),'short_scale_scene_units':summary(short),'interpretation':'all projected centers including possible unrelated overlapping surfaces; sigma is geometric scale, not error standard deviation'}
    out['status']='completed_observation_diagnostic'; return out


def main(selection='maximum_response'):
    started=time.perf_counter(); index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8')); scans=[]
    for s in (37,40): scans.append(process(s,index,selection))
    result={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),'opencv':cv2.__version__,'epiline_synthetic_check':epiline_self_check(),'supersedes':'observed_plane_check.json had incorrect reverse epiline residual and its no-match conclusion is invalid','scans':scans,'elapsed_seconds':time.perf_counter()-started,'thresholds':'Lowe .75 mutual, alpha > .5, 2 original-pixel epipolar and reprojection, fixed 64px patch, max 32 response points; 1 degree minimum angle is an engineering reliability flag, not calibrated. No GT, no ICP, no fit selection by residual.'}
    result['selection']=selection
    path=base.OUT/('observed_grid_plane_check.json' if selection=='fixed_grid_density' else 'observed_plane_check_v2.json')
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    for x in scans: print(x['scan'],x['status'],x.get('sift_keypoint_counts'),x.get('pairs'),x.get('three_view_reprojection_2px_count'),x.get('patch_match_count'))


if __name__=='__main__':
    selection=sys.argv[1] if len(sys.argv)>1 else 'maximum_response'
    assert selection in ('maximum_response','fixed_grid_density')
    main(selection)
