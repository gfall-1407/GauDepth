"""Conditional geometry diagnosis for the fixed recorded accepted rays."""
import json, subprocess
from pathlib import Path
import numpy as np
import directional_support_probe as base

eps=np.finfo(np.float64).eps

def hull2(points, L):
    tol=128*eps*L
    area_tol=128*eps*L*L
    pts=sorted(set(map(tuple,points.tolist())))
    if len(pts)<3: return None
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=area_tol: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=area_tol: upper.pop()
        upper.append(p)
    poly=np.asarray(lower[:-1]+upper[:-1])
    if len(poly)<3: return None
    area=abs(sum(cross((0,0),poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly))))/2
    if area<=area_tol: return None
    return poly

def location(poly,point,L):
    edges=np.roll(poly,-1,axis=0)-poly
    rel=point-poly
    cross=edges[:,0]*rel[:,1]-edges[:,1]*rel[:,0]
    inside=bool(np.all(cross>=-128*eps*L*L))
    frac=np.clip(np.sum(rel*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
    distance=float(np.min(np.linalg.norm(rel-frac[:,None]*edges,axis=1)))
    return inside,distance

def main():
    index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8'))
    prefix='20261026_夜间深度方法/'
    paths={k:base.indexed(prefix+s,index) for k,s in {'model':'inputs/scan37/point_cloud.ply','cfg':'inputs/scan37/cfg_args','metadata':'confirm_v1/scan37/metadata.json','trace':'confirm_v1/scan37/view_000.npz'}.items()}
    meta=json.loads(paths['metadata'].read_text(encoding='utf8')); view=next(v for v in meta['views'] if v['view_index']==0)
    assert base.sha(paths['model'])==meta['model_sha256'] and base.sha(paths['cfg'])==meta['cfg_sha256'] and base.sha(paths['trace'])==view['npz_sha256']
    f=np.load(paths['trace']); vertices=base.ply(paths['model']); ids=f['gaussian_id']; offsets=f['offsets']
    assert ids.min()>=0 and ids.max()<len(vertices) and offsets[-1]==len(ids)==len(f['weight'])
    W=np.asarray(view['world_view_transform'],float); c=np.asarray(view['camera_center'],float)
    assert np.allclose(np.r_[c,1]@W,[0,0,0,1],rtol=0,atol=1e-5)
    assert np.allclose(W[:3,:3].T@W[:3,:3],np.eye(3),rtol=0,atol=1e-5)
    h,w=view['image_size']; fx,fy=view['focal']; indices=np.linspace(0,len(offsets)-2,32,dtype=int); rows=[]
    for i in indices:
        a,b=map(int,offsets[i:i+2]); v=vertices[ids[a:b]]; weights=f['weight'][a:b].astype(float)
        x,y=map(int,f['pixel_xy'][i]); camera_ray=np.array([(x-(w-1)/2)/fx,(y-(h-1)/2)/fy,1.]); ray=W[:3,:3]@camera_ray
        assert np.allclose(ray@W[:3,:3],camera_ray,rtol=0,atol=1e-5)
        row=dict(index=int(i),pixel=[x,y],count=int(b-a),weight_sum=float(weights.sum()),eigenvalues=None,residual=None,normal_unique=False,normal=None,incidence=None,z_depth=None,positive_intersection=False,hull='unavailable',boundary_distance=None,jacobian_norm=None)
        assert len(weights)>0 and np.isfinite(weights).all() and np.all(weights>0)
        p=weights/weights.sum(); mu=np.stack([v[k] for k in ('x','y','z')],1).astype(float); assert np.isfinite(mu).all()
        mean=p@mu; d=mu-mean; C=(p[:,None]*d).T@d; eig,U=np.linalg.eigh(C); n=U[:,0]
        unique=bool(eig[1]-eig[0]>128*eps*max(float(np.max(np.abs(eig))),np.finfo(float).tiny)); denom=float(n@ray); incidence=abs(denom)/float(np.linalg.norm(ray))
        row.update(eigenvalues=eig.tolist(),residual=float(p@((d@n)**2)),normal_unique=unique,normal=n.tolist() if unique else None,incidence=incidence if unique else None)
        if unique and incidence>128*eps:
            D=float(n@(mean-c)/denom)
            row['z_depth']=D if np.isfinite(D) else None
            if np.isfinite(D) and D>0:
                row['positive_intersection']=True
                L=max(1.,float(np.max(np.abs(d)))); projected=d@U[:,1:]; poly=hull2(projected,L)
                if poly is None: row['hull']='degenerate'
                else:
                    inside,dist=location(poly,(c+D*ray-mean)@U[:,1:],L)
                    row['hull']='inside' if inside else 'outside'; row['boundary_distance']=dist
                q=mean-c-D*ray; grad=np.broadcast_to(n,(len(mu),3)).copy()
                for k in (1,2):
                    u=U[:,k]; coef=float(q@u)/(eig[0]-eig[k]); grad += coef*((d@n)[:,None]*u+(d@u)[:,None]*n)
                grad*=p[:,None]/denom
                row['jacobian_norm']=float(np.linalg.norm(grad))
        rows.append(row)
    counts={k:sum(row['hull']==k for row in rows) for k in ('inside','outside','degenerate','unavailable')}
    counts.update(normal_unique=sum(r['normal_unique'] for r in rows),positive_intersection=sum(r['positive_intersection'] for r in rows))
    report=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),unverified_mapping=True,source_commit=meta['git_commit'],host_binary_sha256=meta['host_binary_sha256'],host_build_provenance=meta['host_build_provenance'],input_sha256={k:base.sha(v) for k,v in paths.items()},counts=counts,rows=rows,limitations='Conditional on unverified source ID to PLY row mapping; CPU alpha/weight strict compatibility failed. No surface membership certification; synthetic mixture can interpolate a false intermediate plane. z_depth uses camera z parameter; Jacobian is local and not a real error bound.')
    out=Path(__file__).parent/'结果/20261029_接受中心几何/accepted_geometry.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__': main()
