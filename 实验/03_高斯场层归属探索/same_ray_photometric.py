"""Fixed-ray conditional photometric comparison, no re-fitting or masks."""
import importlib.util,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import directional_support_probe as base
spec=importlib.util.spec_from_file_location('colmap_loader',base.REPO/'scene/colmap_loader.py');col=importlib.util.module_from_spec(spec);spec.loader.exec_module(col)
HERE=Path(__file__).parent; OUT=HERE/'结果/20261029_同射线光度/photometric.json'

def zncc(a,b):
    a=a-a.mean();b=b-b.mean(); va=float(np.mean(a*a));vb=float(np.mean(b*b))
    if va<=1e-8:return None,'reference_variance'
    if vb<=1e-8:return None,'target_variance'
    return float(1-np.mean(a*b)/np.sqrt(va*vb)),None

def project_plane(pixels,K0,R0,t0,K1,R1,t1,n,plane):
    c=-R0.T@t0; dirs=(np.linalg.inv(K0)@np.column_stack([pixels,np.ones(len(pixels))]).T).T@R0
    den=dirs@n
    if np.any(np.abs(den)<=128*np.finfo(float).eps*np.linalg.norm(dirs,axis=1)):return None,'parallel'
    distance=((plane-c)@n)/den
    if np.any(distance<=0):return None,'reference_negative_depth'
    X=c+distance[:,None]*dirs; cam=X@R1.T+t1
    if np.any(cam[:,2]<=0):return None,'target_negative_depth'
    p=cam@K1.T; return p[:,:2]/p[:,2,None],None

def synthetic():
    K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]]) ;R=np.eye(3);t0=np.zeros(3);t1=np.array([-.1,0,0]);n=np.array([0,0,1.])
    xy=np.array([(50+x,50+y) for y in range(-4,5) for x in range(-4,5)],float)
    p2,e=project_plane(xy,K,R,t0,K,R,t1,n,np.array([0,0,2.]));p4,e4=project_plane(xy,K,R,t0,K,R,t1,n,np.array([0,0,4.]))
    err=float(np.max(np.abs(p2-(xy+[-5,0]))));
    def text(p):return np.sin(.21*p[:,0])+np.cos(.17*p[:,1])+.003*p[:,0]*p[:,1]
    good,_=zncc(text(xy),text(p2+[5,0]));wrong,_=zncc(text(xy),text(p4+[5,0]))
    _,flat=zncc(np.ones(81),np.ones(81))
    def periodic(p):return np.sin(2*np.pi*p[:,0]/2.5)+.3*np.cos(.17*p[:,1])
    repeat_good,_=zncc(periodic(xy),periodic(p2+[5,0]));repeat_wrong,_=zncc(periodic(xy),periodic(p4+[5,0]))
    ok=bool(e is None and e4 is None and err<=1e-10 and abs(good)<=1e-12 and wrong>good and flat=='reference_variance' and abs(repeat_good)<=1e-12 and abs(repeat_wrong)<=1e-12)
    return dict(passed=ok,homography_max_error_px=err,known_depth_cost=good,wrong_depth_cost=wrong,flat_reason=flat,periodic_known_cost=repeat_good,periodic_alternate_cost=repeat_wrong,description='Analytic continuous texture; not real-image bilinear error.')

def sample(im,xy):
    h,w=im.shape[:2]
    if np.any(~np.isfinite(xy)) or np.any(xy[:,0]<0) or np.any(xy[:,1]<0) or np.any(xy[:,0]>=w-1) or np.any(xy[:,1]>=h-1):return None,'out_of_bounds'
    x=np.floor(xy[:,0]).astype(int);y=np.floor(xy[:,1]).astype(int);u=xy[:,0]-x;v=xy[:,1]-y
    q=(im[y,x]*(1-u)[:,None]*(1-v)[:,None]+im[y,x+1]*u[:,None]*(1-v)[:,None]+im[y+1,x]*(1-u)[:,None]*v[:,None]+im[y+1,x+1]*u[:,None]*v[:,None])
    if np.any(q[:,3]<=.5):return None,'alpha'
    return q[:,:3]@np.array([.299,.587,.114]),None

def main():
    start=time.monotonic();OUT.parent.mkdir(parents=True,exist_ok=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),unverified_mapping=True,synthetic=synthetic(),rows=[])
    if not result['synthetic']['passed']:
        result['status']='synthetic_failed';OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');raise SystemExit(1)
    idx=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8')); pre='20261026_观测约束表面/observations_v1/scan37/'
    paths={k:base.indexed(pre+s,idx) for k,s in {'cameras':'sparse/0/cameras.bin','images':'sparse/0/images.bin',**{f'image{i}':f'images/{i:04d}.png' for i in range(3)}}.items()}
    prefix='20261026_夜间深度方法/';paths.update({k:base.indexed(prefix+s,idx) for k,s in {'trace':'confirm_v1/scan37/view_000.npz','metadata':'confirm_v1/scan37/metadata.json'}.items()})
    meta=json.loads(paths['metadata'].read_text(encoding='utf8'));view=next(v for v in meta['views'] if v['view_index']==0);assert base.sha(paths['trace'])==view['npz_sha256']
    f=np.load(paths['trace']); prior=json.loads((HERE/'结果/20261029_接受中心几何/accepted_geometry.json').read_text(encoding='utf8')); assert prior['unverified_mapping'] and prior['input_sha256']['trace']==base.sha(paths['trace'])
    cams=col.read_intrinsics_binary(str(paths['cameras'])); extr=col.read_extrinsics_binary(str(paths['images'])); ims=[next(v for v in extr.values() if v.name==f'{i:04d}.png') for i in range(3)]
    Rs=[col.qvec2rotmat(v.qvec) for v in ims];ts=[v.tvec for v in ims];Ks=[]
    for im in ims:
        cam=cams[im.camera_id];assert cam.model=='PINHOLE' and (cam.width,cam.height)==(1554,1162);fx,fy,cx,cy=cam.params;Ks.append(np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]))
    W=np.array(view['world_view_transform']);assert np.max(np.abs(Rs[0]-W[:3,:3].T))<1e-4 and np.max(np.abs(ts[0]-W[3,:3]))<1e-4
    images=[np.asarray(Image.open(paths[f'image{i}']).convert('RGBA'),dtype=float)/255 for i in range(3)]
    c=np.asarray(view['camera_center']);h,w=view['image_size'];fx,fy=view['focal'];off=np.array([(x,y) for y in range(-4,5) for x in range(-4,5)])
    assert np.array_equal(np.array([v['index'] for v in prior['rows']]),np.linspace(0,len(f['offsets'])-2,32,dtype=int))
    for prev in prior['rows']:
        i=prev['index']; x,y=prev['pixel'];n=np.array(prev['normal']); dj=prev['z_depth'];de=float(f['host_target'][i,0]);r=W[:3,:3]@np.array([(x-(w-1)/2)/fx,(y-(h-1)/2)/fy,1.]);anchor=np.array([2*x+1,2*y+1],float); refxy=anchor+off
        p=Ks[0]@(Rs[0]@(c+dj*r)+ts[0]);native=p[:2]/p[2];ref,referr=sample(images[0],refxy);a,b=map(int,f['offsets'][i:i+2]);depth=f['depth'][a:b].astype(float);vals=np.r_[depth,dj,de];assert np.isfinite(vals).all();grid=np.linspace(vals.min(),vals.max(),129)
        row=dict(index=i,pixel=[x,y],anchor=anchor.tolist(),native_projection_offset=(native-anchor).tolist(),DJ=dj,DE=de,depth_range=[float(grid[0]),float(grid[-1])],views=[])
        for j in (1,2):
            def evaluate(D):
                if D<=0:return dict(valid=False,cost=None,reason='nonpositive_candidate')
                if referr:return dict(valid=False,cost=None,reason='reference_'+referr)
                uv,err=project_plane(refxy,Ks[0],Rs[0],ts[0],Ks[j],Rs[j],ts[j],n,c+D*r)
                if err:return dict(valid=False,cost=None,reason=err)
                target,err=sample(images[j],uv)
                if err:return dict(valid=False,cost=None,reason='target_'+err)
                cost,err=zncc(ref,target)
                return dict(valid=err is None,cost=cost,reason=err)
            hj=evaluate(dj);he=evaluate(de);curve=[dict(depth=float(D),**evaluate(D)) for D in grid]
            minima=[k for k in range(1,128) if all(curve[t]['valid'] for t in (k-1,k,k+1)) and curve[k]['cost']<curve[k-1]['cost'] and curve[k]['cost']<curve[k+1]['cost']]
            uvj,ej=project_plane(anchor[None],Ks[0],Rs[0],ts[0],Ks[j],Rs[j],ts[j],n,c+dj*r);uve,ee=project_plane(anchor[None],Ks[0],Rs[0],ts[0],Ks[j],Rs[j],ts[j],n,c+de*r)
            motion=float(np.linalg.norm(uvj-uve)) if ej is None and ee is None and dj>0 and de>0 else None
            row['views'].append(dict(view=j,HJ=hj,HE=he,delta_HJ_minus_HE=hj['cost']-he['cost'] if hj['valid'] and he['valid'] else None,center_motion_px=motion,local_minima_indices=minima,curve=curve))
        result['rows'].append(row)
    result.update(status='complete',input_sha256={k:base.sha(v) for k,v in paths.items()},prior_commit=prior['code_commit'],metadata_source_commit=meta['git_commit'],host_binary_sha256=meta['host_binary_sha256'],host_build_provenance=meta['host_build_provenance'],elapsed_seconds=time.monotonic()-start,limitations='ID mapping unverified; prior CPU alpha/w strict comparison failed. Local plane, visibility, texture and brightness assumptions; alpha does not certify visibility; training images are not blind.')
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('complete',len(result['rows']),'seconds',result['elapsed_seconds'])
if __name__=='__main__':main()
