"""Experimental CPU source-formula readout, gated by strict saved-ray comparison."""
import json
import subprocess
import sys
import time
import numpy as np
import directional_support_probe as base

F=np.float32
OUT=base.OUT.parent/'20261029_CPU读出'


def prepare(p,view,kernel):
    # CUDA uses float32; numpy does not promise CUDA FMA, norm3df, or expf bit identity.
    W=np.asarray(view['world_view_transform'],dtype=F); h,w=view['image_size']; fx,fy=map(F,view['focal']); tx,ty=map(lambda x:F(np.tan(x/2)),view['fov'])
    mu=np.stack([p[k] for k in ('x','y','z')],1).astype(F); cam=mu@W[:3,:3]+W[3,:3]; z=cam[:,2]; alive=z>F(.2)
    # Camera constructs full_proj_transform as float32 W @ P.T, then CUDA
    # transforms the original point; do not substitute sequential view projection.
    projection=np.zeros((4,4),dtype=F); projection[0,0]=F(1/np.tan(view['fov'][0]/2)); projection[1,1]=F(1/np.tan(view['fov'][1]/2))
    projection[2,2]=F(100/99.99); projection[2,3]=F(-1/99.99); projection[3,2]=F(1)
    full=W@projection.T; homogeneous=np.c_[mu,np.ones(len(mu),dtype=F)]@full
    pw=F(1)/(homogeneous[:,3]+F(1e-7))
    px=((homogeneous[:,0]*pw+F(1))*F(w)-F(1))*F(.5)
    py=((homogeneous[:,1]*pw+F(1))*F(h)-F(1))*F(.5)
    ux=np.clip(cam[:,0]/z,-F(1.3)*tx,F(1.3)*tx); uy=np.clip(cam[:,1]/z,-F(1.3)*ty,F(1.3)*ty)
    raw=np.exp(np.stack([p[f'scale_{j}'] for j in range(3)],1).astype(F)); raw2=raw*raw
    filtered2=raw2+p['filter_3D'][:,None]**2; scales=np.sqrt(filtered2)
    opacity=F(1)/(F(1)+np.exp(-p['opacity']))*np.sqrt(np.prod(raw2,axis=1)/np.prod(filtered2,axis=1))
    q=np.stack([p[f'rot_{j}'] for j in range(4)],1).astype(F)
    rotation=base.rotation(q).astype(F); Rcam=np.einsum('ij,njk->nik',W[:3,:3].T,rotation)
    C=np.einsum('nik,nk,njk->nij',Rcam,filtered2,Rcam,optimize=True)
    J=np.zeros((len(p),2,3),dtype=F); J[:,0,0]=fx/z; J[:,1,1]=fy/z; J[:,0,2]=-fx*ux/z; J[:,1,2]=-fy*uy/z
    cov=np.einsum('nki,nij,nlj->nkl',J,C,J,optimize=True); A=cov[:,0,0]; B=cov[:,0,1]; Cc=cov[:,1,1]
    k=F(kernel); det0=np.maximum(F(1e-6),A*Cc-B*B); det1=np.maximum(F(1e-6),(A+k)*(Cc+k)-B*B)
    opacity=opacity*np.sqrt(det0/det1); A=A+k; Cc=Cc+k; det=A*Cc-B*B
    mid=F(.5)*(A+Cc); lam=mid+np.sqrt(np.maximum(F(.1),mid*mid-det)); radius=np.ceil(F(3)*np.sqrt(lam)).astype(np.int32)
    valid=alive&(det!=0)&np.isfinite(det)&np.isfinite(px)&np.isfinite(py)&(radius>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        inv=F(1)/det; conic=np.stack([Cc*inv,-B*inv,A*inv,opacity],1)
        # center-direction precision, includes renderer's ill-conditioned minimum-axis fallback.
        uv=np.stack([ux,uy,np.ones(len(p),dtype=F)],1)
        local=np.einsum('nji,nj->ni',Rcam,uv); conditioned=np.min(scales,axis=1)>F(1e-7)
        vec=np.einsum('nij,nj->ni',Rcam,local/filtered2)
        mi=np.argmin(scales,axis=1); axis=Rcam[np.arange(len(p)),:,mi]; fallback=axis*np.sum(axis*uv,axis=1)[:,None]
        vec=np.where(conditioned[:,None],vec,fallback); n=vec/np.sqrt(np.sum(vec*vec,axis=1))[:,None]
        dot=np.maximum(np.sum(n*uv,axis=1),F(1e-7)); n=n/dot[:,None]
        pp0=(uy*uy+F(1))*n[:,0]-ux*uy*n[:,1]-ux*n[:,2]
        pp1=-ux*uy*n[:,0]+(ux*ux+F(1))*n[:,1]-uy*n[:,2]
        factor=np.sqrt((ux*z)**2+(uy*z)**2+z*z)/(ux*ux+uy*uy+F(1))
        tc=np.sqrt(np.sum(cam*cam,axis=1)); plane=np.stack([pp0*factor/fx,pp1*factor/fy,tc],1)
        plane=np.where(np.isnan(n[:,0])[:,None],F(0),plane)
    # CUDA int conversion truncates toward zero before tile-grid clipping.
    gx=(w+15)//16; gy=(h+15)//16
    rminx=np.clip(((px-radius)/F(16)).astype(np.int32),0,gx); rmaxx=np.clip(((px+radius+15)/F(16)).astype(np.int32),0,gx)
    rminy=np.clip(((py-radius)/F(16)).astype(np.int32),0,gy); rmaxy=np.clip(((py+radius+15)/F(16)).astype(np.int32),0,gy)
    valid &= (rmaxx>rminx)&(rmaxy>rminy)
    return dict(px=px,py=py,conic=conic,plane=plane,sort_depth=tc,valid=valid,rect=(rminx,rmaxx,rminy,rmaxy),h=h,w=w,fx=fx,fy=fy)


def trace(pre,x,y):
    a,b,c,d=pre['rect']; mask=pre['valid']&(a<=x//16)&(b>x//16)&(c<=y//16)&(d>y//16)
    ids=np.flatnonzero(mask); ids=ids[np.argsort(pre['sort_depth'][ids].view(np.uint32),kind='stable')]
    T=F(1); D=F(0); M=F(0); out=[]; nx=F((F(x)-F(pre['w']-1)/F(2))/pre['fx']); ny=F((F(y)-F(pre['h']-1)/F(2))/pre['fy']); rln=F(1)/np.sqrt(F(nx*nx+ny*ny+F(1)))
    for i in ids:
        dx=F(pre['px'][i]-F(x));dy=F(pre['py'][i]-F(y)); c=pre['conic'][i]
        power=F(-F(.5)*F(c[0]*dx*dx+c[2]*dy*dy)-c[1]*dx*dy)
        if power>0: continue
        alpha=F(min(F(.99),F(c[3]*np.exp(power))))
        if alpha<F(1/255): continue
        nxt=F(T*F(F(1)-alpha))
        if nxt<F(.0001): break
        pl=pre['plane'][i]; t=F(F(pl[0]*dx+pl[1]*dy)+pl[2]); weight=F(T*alpha)
        out.append((int(i),F(t*rln),alpha,T,weight)); D=F(D+F(t*weight))
        if T>F(.5): M=t
        T=nxt
    return out,[float(F(D*rln/F(F(1)-T))) if out else 0.,float(F(M*rln)),float(F(1)-T)]


def main(scan,label):
    start=time.perf_counter(); idx=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf8')); prefix='20261026_夜间深度方法/'
    paths={k:base.indexed(prefix+p,idx) for k,p in {'model':f'inputs/scan{scan}/point_cloud.ply','cfg':f'inputs/scan{scan}/cfg_args','metadata':f'confirm_v1/scan{scan}/metadata.json','trace':f'confirm_v1/scan{scan}/view_000.npz'}.items()}
    meta=json.loads(paths['metadata'].read_text(encoding='utf8')); view=next(v for v in meta['views'] if v['view_index']==0); assert base.sha(paths['model'])==meta['model_sha256'] and base.sha(paths['cfg'])==meta['cfg_sha256'] and base.sha(paths['trace'])==view['npz_sha256']
    f=np.load(paths['trace']); ply=base.ply(paths['model']); pre=prepare(ply,view,meta['effective_model_args']['kernel_size']); chosen=np.linspace(0,len(f['offsets'])-2,32,dtype=int); rows=[]
    fields=('depth','alpha','transmittance','weight'); all_errors={n:[] for n in fields}; host_errors=[]; first=None; first_field={n:None for n in fields}; first_host=None
    for i in chosen:
        x,y=f['pixel_xy'][i]; calc,host=trace(pre,int(x),int(y)); starti,endi=f['offsets'][i:i+2]; actual=f['gaussian_id'][starti:endi]; got=np.array([v[0] for v in calc],dtype=np.int64)
        exact=np.array_equal(actual,got); row={'ray_index':int(i),'pixel':[int(x),int(y)],'expected_count':len(actual),'cpu_count':len(got),'id_order_equal':bool(exact)}
        row['first_id_mismatch']=next(({'position':j,'logged':int(actual[j]),'cpu':int(got[j])} for j in range(min(len(actual),len(got))) if actual[j]!=got[j]),None)
        if first is None and not exact: first={'ray_index':int(i),'counts':[len(actual),len(got)],'first_id_mismatch':row['first_id_mismatch']}
        row['field_errors']={}
        # Compare shared ordered prefix only when IDs agree; never pair unrelated members.
        prefix_count=next((j for j in range(min(len(actual),len(got))) if actual[j]!=got[j]),min(len(actual),len(got)))
        row['matching_id_prefix']=prefix_count
        for j,n in enumerate(fields,1):
            x1=np.asarray([v[j] for v in calc[:prefix_count]],float); x2=f[n][starti:starti+prefix_count].astype(float); e=np.abs(x1-x2); all_errors[n].extend(e.tolist())
            row['field_errors'][n]={'max':float(e.max()) if len(e) else None,'within_tolerance':bool(np.allclose(x1,x2,rtol=2e-5,atol=2e-6))}
            fail=np.flatnonzero(e>2e-6+2e-5*np.abs(x2))
            if len(fail) and first_field[n] is None:
                pos=int(fail[0]); first_field[n]={'ray_index':int(i),'position':pos,'id':int(actual[pos]),'logged':float(x2[pos]),'cpu':float(x1[pos])}
        err=np.abs(np.array(host)-f['host_target'][i]);host_errors.append(err.tolist()); row['host_abs_errors']=err.tolist();row['host_within_tolerance']=bool(np.allclose(host,f['host_target'][i],rtol=2e-5,atol=2e-6)); rows.append(row)
        if first_host is None and not row['host_within_tolerance']: first_host={'ray_index':int(i),'cpu':host,'host':f['host_target'][i].tolist()}
    def stats(v): return {'n':len(v),'max':float(np.max(v)),'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95))} if len(v) else {'n':0}
    report={'scan':scan,'code_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),'input_sha256':{k:base.sha(v) for k,v in paths.items()},'metadata_source_sha256':meta['source_sha256'],'host_source_sha256':meta['host_source_sha256'],'host_binary_sha256':meta['host_binary_sha256'],'projection':'derived symmetric getProjectionMatrix from metadata fov; w=z+1e-7, ndc2Pix; znear=.01 zfar=100 do not affect x/y','kernel_size':meta['effective_model_args']['kernel_size'],'indices':chosen.tolist(),'first_id_mismatch':first,'first_field_mismatch':first_field,'first_host_mismatch':first_host,'matched_id_rays':sum(r['id_order_equal'] for r in rows),'field_error_matching_prefix':{n:stats(v) for n,v in all_errors.items()},'host_error':{n:stats(np.asarray(host_errors)[:,j]) for j,n in enumerate(('expected','native_median','opacity'))},'host_max_abs':np.max(host_errors,axis=0).tolist(),'passed':all(r['id_order_equal'] and r['host_within_tolerance'] and all(v['within_tolerance'] for v in r['field_errors'].values()) for r in rows),'rows':rows,'elapsed_seconds':time.perf_counter()-start,'limitations':'CPU source-formula experiment; GPU FMA/fast math and host binary build provenance unresolved. Unmatched IDs compared only through common prefix; no new rays, no CUDA call, no claim of general equivalence.'}
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/f'scan{scan}_{label}.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps({k:report[k] for k in ('scan','passed','matched_id_rays','first_id_mismatch','field_error_matching_prefix','host_max_abs','elapsed_seconds')},indent=2))


if __name__=='__main__': main(int(sys.argv[1]),sys.argv[2])
