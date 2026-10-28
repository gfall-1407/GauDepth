"""Conditional three-view comparison for one reference-selected 3D Gaussian."""
import itertools
import json
import subprocess
import time
import numpy as np
import directional_support_probe as base
import observed_plane_check as obs

OUT=base.OUT.parent/'20261029_单基元观测'


def plane_intersection(center, normal, origin, direction):
    denom=float(normal@direction)
    if abs(denom)<=1e-8*np.linalg.norm(direction): return None,'parallel'
    z=float(normal@(center-origin)/denom)
    return (z,'ok') if np.isfinite(z) and z>0 else (None,'nonpositive_or_nonfinite')


def test_geometry():
    angle=np.pi/4; e=np.array([np.cos(angle),0.,-np.sin(angle)]); n=np.array([np.sin(angle),0.,np.cos(angle)]); y=np.array([0.,1.,0.]); R=np.column_stack([e,y,n]); s=np.array([.1,.1,.005]); origin=np.zeros(3); v=np.array([0.,0.,1.]); p=np.array([0.,0.,2.]); mu=p+.1*e
    # Independent closed form using local-axis components, not a matrix inverse.
    z,sigma=base.section(mu[None],s[None],R[None],origin,v); closed=2-(.1*np.sin(angle)/s[0]**2)/((np.sin(angle)/s[0])**2+(np.cos(angle)/s[2])**2)
    np.testing.assert_allclose(z,[closed],rtol=1e-12,atol=1e-12); plane,status=plane_intersection(mu,n,origin,v); assert status=='ok' and abs(plane-2)<1e-12 and abs(z[0]-plane)>1e-5
    assert plane_intersection(mu,-n,origin,v)[0]==plane
    assert plane_intersection(p,np.array([1.,0,0]),origin,v)[1]=='parallel'
    assert plane_intersection(-p,v,origin,v)[1]=='nonpositive_or_nonfinite'
    sphere=np.array([[0.,0.,3.]]); z0,s0=base.section(sphere,np.array([[.5,.5,.5]]),np.eye(3)[None],origin,v); assert z0[0]==3 and s0[0]==.5
    u=.41; A=np.array([[np.cos(u),-np.sin(u),0],[np.sin(u),np.cos(u),0],[0,0,1.]]); shift=np.array([3.,-1.,2.]); scale=7.
    z2,s2=base.section((scale*mu@A.T+shift)[None],(scale*s)[None],(A@R)[None],shift,A@v); np.testing.assert_allclose([z2[0],s2[0]],scale*np.array([z[0],sigma[0]]),rtol=1e-12)
    zplane,_=plane_intersection(scale*mu@A.T+shift,A@n,shift,A@v); np.testing.assert_allclose(zplane,scale*plane,rtol=1e-12)
    return 'passed independent oblique closed form, positive sphere, coplanar offset, plane sign, parallel/negative, rigid transform and unit scaling'


def projected(P,X):
    q=P@np.r_[X,1.]
    if not np.isfinite(q).all() or q[2]<=0: return None
    return q[:2]/q[2]


def main():
    start=time.perf_counter(); synthetic=test_geometry(); index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8'))
    pre='20261026_观测约束表面/observations_v1/scan37/'
    paths={k:base.indexed(pre+p,index) for k,p in {'camera_npz':'cameras.npz','cameras_bin':'sparse/0/cameras.bin','images_bin':'sparse/0/images.bin',**{f'image{i}':f'images/{i:04d}.png' for i in range(3)}}.items()}
    paths['model']=base.indexed('20261026_夜间深度方法/inputs/scan37/point_cloud.ply',index)
    paths['metadata']=base.indexed('20261026_夜间深度方法/confirm_v1/scan37/metadata.json',index)
    gridpath=base.OUT/'observed_grid_plane_check.json'; grid=json.loads(gridpath.read_text(encoding='utf8')); recorded=next(v for v in grid['scans'] if v['scan']==37)
    assert recorded['selected_window']['bounds']==[1216,256,1280,320] and len(recorded['patch_matches'])==23
    for k,path in paths.items(): assert base.sha(path)==recorded['inputs_sha256'][k],k
    meta=json.loads(paths['metadata'].read_text(encoding='utf8')); assert base.sha(paths['model'])==meta['model_sha256']
    cameras=obs.col.read_intrinsics_binary(str(paths['cameras_bin'])); ex=obs.col.read_extrinsics_binary(str(paths['images_bin'])); images=[next(v for v in ex.values() if v.name==f'{i:04}.png') for i in range(3)]
    R=[obs.col.qvec2rotmat(v.qvec) for v in images]; t=[v.tvec for v in images]; K=[]
    for v in images:
        cam=cameras[v.camera_id]; assert cam.model=='PINHOLE'; fx,fy,cx,cy=cam.params; K.append(np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]))
    view=next(v for v in meta['views'] if v['view_index']==0); W=np.array(view['world_view_transform']); rotation_error=float(np.max(np.abs(R[0]-W[:3,:3].T))); translation_error=float(np.max(np.abs(t[0]-W[3,:3]))); assert rotation_error<1e-4 and translation_error<1e-4
    P=[k@np.column_stack([r,x]) for k,r,x in zip(K,R,t)]; C=-R[0].T@t[0]
    ply=base.ply(paths['model']); N=len(ply); mu=np.stack([ply[n] for n in ('x','y','z')],1).astype(float)
    raw=np.exp(np.stack([ply[f'scale_{j}'] for j in range(3)],1).astype(float)); scales=np.sqrt(raw*raw+ply['filter_3D'][:,None].astype(float)**2)
    rot=base.rotation(np.stack([ply[f'rot_{j}'] for j in range(4)],1).astype(float)); offsets=np.einsum('nji,nj->ni',rot,C-mu)/scales
    logopacity=-np.logaddexp(0,-ply['opacity'].astype(float))+np.log(raw/scales).sum(axis=1)
    rows=[]
    for sample in recorded['patch_matches']:
        native=np.array(sample['pixels'][0]); raycam=np.linalg.solve(K[0],np.r_[native,1.]); ray=R[0].T@raycam
        local=np.einsum('nji,j->ni',rot,ray)/scales; denominator=np.sum(local*local,axis=1); z=-np.sum(local*offsets,axis=1)/denominator
        # Direct residual at the peak avoids subtracting large quadratic terms.
        residual=offsets+z[:,None]*local; score=logopacity-.5*np.sum(residual*residual,axis=1)
        eligible=np.isfinite(score)&np.isfinite(z)&(z>0); scores=np.where(eligible,score,-np.inf); row={'reference_id':sample['reference_id'],'native_pixels':sample['pixels'],'status':'no_positive_finite_peak'};rows.append(row)
        if not eligible.any(): continue
        winner=int(np.argmax(scores)); best=float(scores[winner]); scores[winner]=-np.inf; runner=float(np.max(scores))
        order=np.argsort(scales[winner]); unique=bool(scales[winner,order[1]]>scales[winner,order[0]]*1.001); normal=rot[winner,:,order[0]]
        row.update(gaussian_id=winner,peak_log_response=best,peak_response=float(np.exp(best)),runner_up_log_score_gap=float(best-runner) if np.isfinite(runner) else None,unique_short_axis=unique,peak_z=float(z[winner]))
        if not unique: row['status']='nonunique_short_axis'; continue
        plane_z,status=plane_intersection(mu[winner],normal,C,ray); row['plane_status']=status
        if status!='ok': row['status']=status;continue
        row['plane_z']=plane_z; Xpeak=C+z[winner]*ray; Xplane=C+plane_z*ray; predicted_peak=[];predicted_plane=[]
        for j in (1,2): predicted_peak.append(projected(P[j],Xpeak)); predicted_plane.append(projected(P[j],Xplane))
        if any(x is None for x in predicted_peak+predicted_plane): row['status']='neighbor_nonpositive_or_nonfinite';continue
        target=np.array(sample['pixels'][1:]); e_peak=np.linalg.norm(np.array(predicted_peak)-target,axis=1);e_plane=np.linalg.norm(np.array(predicted_plane)-target,axis=1)
        row.update(status='ok',peak_world=Xpeak.tolist(),plane_world=Xplane.tolist(),peak_predicted=np.asarray(predicted_peak).tolist(),plane_predicted=np.asarray(predicted_plane).tolist(),peak_residual_px=e_peak.tolist(),plane_residual_px=e_plane.tolist(),plane_minus_peak_px=(e_plane-e_peak).tolist())
    valid=[r for r in rows if r['status']=='ok']; comparisons={}
    for j in range(2):
        delta=np.array([r['plane_minus_peak_px'][j] for r in valid]); comparisons[f'neighbor{j+1}']={'n':len(delta),'improved':int(np.sum(delta<0)),'worsened':int(np.sum(delta>0)),'equal':int(np.sum(delta==0)),'mean':float(delta.mean()) if len(delta) else None,'median':float(np.median(delta)) if len(delta) else None,'p90':float(np.quantile(delta,.9)) if len(delta) else None}
    statuses={k:sum(r['status']==k for r in rows) for k in sorted({r['status'] for r in rows})}
    result={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),'input_sha256':{k:base.sha(p) for k,p in paths.items()},'frozen_grid_sha256':base.sha(gridpath),'selected_window':recorded['selected_window'],'model_vertex_count':N,'pose_rotation_error':rotation_error,'pose_translation_error':translation_error,'synthetic':synthetic,'predeclared':'23 fixed patch tracks; peak positive; select highest exact filtered 3D log response from full PLY only, ties lowest ID; no replacement for invalid plane; short axis unique if next scale > min scale * 1.001; compare same Gaussian in both views','statuses':statuses,'total':len(rows),'pairwise_plane_minus_peak_px':comparisons,'rows':rows,'elapsed_seconds':time.perf_counter()-start,'limitations':'Conditional association is not original renderer accepted contributors, nor GT or true surface. No 2D filter or occlusion; image tracks are preselected engineering samples, not independent blinded test. No training or new depth estimator.'}
    OUT.mkdir(parents=True,exist_ok=True); dest=OUT/'single_primitive.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'statuses':statuses,'pairwise':comparisons,'elapsed':result['elapsed_seconds'],'synthetic':synthetic},indent=2))


if __name__=='__main__': main()
