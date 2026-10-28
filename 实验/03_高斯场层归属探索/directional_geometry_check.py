"""Fixed-ray, unmasked single-direction nearest GT point diagnostic; not DTU metric."""
import json
import subprocess
import time
import sys
import numpy as np
from scipy.spatial import cKDTree
import directional_support_probe as base
import directional_mechanism as mech

METHODS=('expected','native_median','directional','isotropic','gggs_proxy')

def read_stl(path):
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline().decode('ascii').strip(); lines.append(line)
            if line=='end_header': break
        assert 'format binary_little_endian 1.0' in lines and not any(s.startswith('element face ') for s in lines)
        n=int(next(s.split()[-1] for s in lines if s.startswith('element vertex ')))
        typ={'float':'<f4','uchar':'u1'}
        prop=[s.split() for s in lines if s.startswith('property ')]
        a=np.fromfile(f,dtype=np.dtype([(s[2],typ[s[1]]) for s in prop]),count=n)
    pts=np.stack([a[k] for k in ('x','y','z')],axis=1).astype(float)
    assert len(pts)==n and np.isfinite(pts).all()
    return pts,lines


def camera_check(path, view):
    c=np.load(path); i=int(view['image_name']); S=c[f'scale_mat_{i}'].astype(float)
    scale=float(S[0,0]); assert scale>0 and np.allclose(S[:3,:3],np.eye(3)*scale,rtol=0,atol=1e-6)
    assert np.allclose(S[3],[0,0,0,1],atol=1e-8)
    P=(c[f'world_mat_{i}']@S)[:3,:4]; W=np.array(view['world_view_transform'],float)
    R=W[:3,:3].T; center=np.array(view['camera_center'],float)
    assert np.allclose(R@R.T,np.eye(3),atol=1e-5) and np.linalg.det(R)>0
    _,_,vh=np.linalg.svd(P); C=vh[-1,:3]/vh[-1,3]
    center_err=float(np.max(np.abs(C-center)))
    K=P[:,:3]@R.T; K=K/K[2,2]
    lower=float(np.max(np.abs(K[np.tril_indices(3,-1)])))
    assert center_err<1e-4 and lower<1e-4 and K[0,0]>0 and K[1,1]>0,(center_err,lower,K)
    return S,{'center_max_abs':center_err,'intrinsics_lower_max_abs':lower,'K_from_projection':K.tolist(),'scale':scale}


def main(scan):
    start=time.perf_counter(); index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf-8'))
    prefix='20261026_夜间深度方法/'; source='samples_v2' if scan==106 else 'confirm_v1'
    paths={name:base.indexed(prefix+suffix,index) for name,suffix in {
        'model':f'inputs/scan{scan}/point_cloud.ply','camera':f'inputs/scan{scan}/cameras.npz',
        'stl':f'inputs/scan{scan}/stl{scan:03d}_total.ply','metadata':f'{source}/scan{scan}/metadata.json',
        'trace':f'{source}/scan{scan}/view_032.npz'}.items()}
    meta=json.loads(paths['metadata'].read_text(encoding='utf-8')); view=next(v for v in meta['views'] if v['view_index']==32)
    assert base.sha(paths['model'])==meta['model_sha256'] and base.sha(paths['trace'])==view['npz_sha256']
    S,camera=camera_check(paths['camera'],view); f=np.load(paths['trace']); chosen=np.linspace(0,len(f['offsets'])-2,32,dtype=int)
    if scan==83:
        old=json.loads((base.OUT/'mechanism_v2.json').read_text(encoding='utf-8'))
        assert old['input_model_sha256']==base.sha(paths['model']) and old['input_trace_sha256']==base.sha(paths['trace'])
        rows=old['rows']; assert [r['ray_index'] for r in rows]==chosen.tolist()
    else:
        vertices=base.ply(paths['model']); ids=f['gaussian_id']; assert ids.min()>=0 and ids.max()<len(vertices)
        rows=[]
        for i in chosen:
            a,b=f['offsets'][i:i+2]; d=f['depth'][a:b].astype(float); w=f['weight'][a:b].astype(float); alpha=f['alpha'][a:b].astype(float)
            sigma,iso=mech.ray_bandwidths(vertices[ids[a:b]],view,f['pixel_xy'][i]); mid=np.flatnonzero(f['transmittance'][a:b]>.5)
            row={'ray_index':int(i),'expected':float(np.sum(d*w)/np.sum(w)) if len(d) and np.sum(w)>0 else None,'native_median':float(d[mid[-1]]) if len(mid) else None}
            for name,s in [('directional',sigma),('isotropic',iso)]:
                row[name],_,row[name+'_status']=mech.peaks(d,w,2*s)
            row['gggs_proxy'],row['gggs_proxy_status'],_=mech.gg(d,alpha,sigma); rows.append(row)
    pts,header=read_stl(paths['stl']); tree=cKDTree(pts); del pts
    W=np.array(view['world_view_transform'],float); h,width=view['image_size']; fx,fy=view['focal']
    for row in rows:
        i=row['ray_index']; x,y=f['pixel_xy'][i]; camera_ray=np.array([(x-(width-1)/2)/fx,(y-(h-1)/2)/fy,1.])
        row['pixel_xy']=[int(x),int(y)]; row['valid']={m:bool(row[m] is not None and np.isfinite(row[m]) and row[m]>0) for m in METHODS}
        row['common_valid']=all(row['valid'].values())
        if row['common_valid']:
            xyz=np.array([camera_ray*row[m] for m in METHODS]); world=(xyz-W[3,:3])@W[:3,:3].T
            dtu=world@S[:3,:3].T+S[:3,3]; distances,_=tree.query(dtu,k=1)
            row['distance_mm']={m:float(distances[j]) for j,m in enumerate(METHODS)}
            row['difference_vs_expected_mm']={m:float(distances[j]-distances[0]) for j,m in enumerate(METHODS)}
    valid=[r for r in rows if r['common_valid']]
    summary={}
    for m in METHODS:
        v=np.array([r['distance_mm'][m] for r in valid]); delta=np.array([r['difference_vs_expected_mm'][m] for r in valid])
        summary[m]={'mean_mm':float(v.mean()) if len(v) else None,'median_mm':float(np.median(v)) if len(v) else None,'p90_mm':float(np.quantile(v,.9)) if len(v) else None,'paired_mean_vs_expected_mm':float(delta.mean()) if len(v) else None,'failures':sum(not r['valid'][m] for r in rows)}
    out={'scan':scan,'view':32,'code_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(), 'inputs_sha256':{k:base.sha(p) for k,p in paths.items()},'kappa':2,'source':source,'camera_check':camera,'stl_header':header,'common_valid':len(valid),'requested':32,'summary':summary,'rows':rows,'elapsed_seconds':time.perf_counter()-start,'definition':'one-way nearest independent GT point (mm), no ObsMask/Plane; not pixel depth GT, surface distance, Chamfer or formal DTU; no ICP. Engineering samples, not independent generalization. Host binary provenance and ID mapping gaps remain.'}
    dest=base.OUT/f'geometry_scan{scan}.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'scan':scan,'camera':camera,'common':len(valid),'summary':summary,'elapsed':out['elapsed_seconds']},indent=2))


def combine():
    scans=[json.loads((base.OUT/f'geometry_scan{s}.json').read_text(encoding='utf-8')) for s in (83,118,106)]
    valid=[r for scan in scans for r in scan['rows'] if r['common_valid']]
    summary={}
    for m in METHODS:
        v=np.array([r['distance_mm'][m] for r in valid]); delta=np.array([r['difference_vs_expected_mm'][m] for r in valid])
        summary[m]={'mean_mm':float(v.mean()),'median_mm':float(np.median(v)),'p90_mm':float(np.quantile(v,.9)),'paired_mean_vs_expected_mm':float(delta.mean()),'failure_count':sum(not r['valid'][m] for scan in scans for r in scan['rows'])}
    result={'scans':[83,118,106],'common_valid':len(valid),'requested':96,'pooled_summary':summary,'per_scan':[{'scan':s['scan'],'common_valid':s['common_valid'],'summary':s['summary'],'inputs_sha256':s['inputs_sha256'],'camera_check':s['camera_check'],'code_head':s['code_head']} for s in scans],'definition':scans[0]['definition'],'principal_point_note':'Original camera projection principal point differs from centered renderer; backprojection follows logged renderer pixel-center/focal convention, not original K principal point.'}
    dest=base.OUT/'geometry_diagnostic.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='combine': combine()
    else: main(int(sys.argv[1]))
