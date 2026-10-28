"""Fixed 32-ray CPU mechanism check; no ground-truth accuracy claim."""
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial as P
import directional_support_probe as base


def score(s, d, w, h):
    q = np.maximum(1-((s-d)/h)**2, 0)
    return float(np.sum(w*q**3))


def peaks(d, w, h):
    if len(d)==0 or np.any(~np.isfinite(h)) or np.any(h<=0) or np.any(~np.isfinite(d)) or np.any(w<0) or not np.any(w>0):
        return None, [], 'invalid_or_empty'
    breaks = np.unique(np.r_[0., np.maximum(0., d-h), np.maximum(0., d+h)])
    cand = []
    for left,right in zip(breaks[:-1], breaks[1:]):
        if right<=left: continue
        center=(left+right)/2; scale=(right-left)/2
        active=np.abs(center-d)<h
        poly=P([0.])
        for di,wi,hi in zip(d[active],w[active],h[active]):
            u=P([(center-di)/hi, scale/hi])
            poly += wi/hi * u * (1-u*u)**2
        if not np.any(poly.coef): continue
        for root in poly.roots():
            if abs(root.imag)<1e-7 and -1-1e-8<=root.real<=1+1e-8:
                val=center+scale*float(root.real)
                if val>0 and left-1e-12<=val<=right+1e-12: cand.append(val)
    cand=sorted(cand)
    unique=[]
    for x in cand:
        if not unique or abs(x-unique[-1])>1e-10*max(1,abs(x)): unique.append(x)
    local=[]; flat=[]
    for x in unique:
        r=x-d; q=np.maximum(1-(r/h)**2,0); val=score(x,d,w,h)
        if val <= np.finfo(float).eps*np.sum(w): continue
        F=np.sum(w*r*q*q/h**2)
        Fs=np.sum(w*(q*q-4*q*(r/h)**2)/h**2)
        scale_f=np.sum(w/h); scale_fs=np.sum(w/h**2)
        if abs(F)>1e-8*scale_f: continue
        if Fs>1e-9*scale_fs: local.append((x,val))
        elif abs(Fs)<=1e-9*scale_fs: flat.append((x,val))
    if not local: return None, flat, 'flat_ambiguous' if flat else 'no_positive_maximum'
    best=max(local,key=lambda x:x[1])
    if score(0.,d,w,h)>best[1]+1e-12*max(1,best[1]):
        return None, local, 'supremum_at_excluded_zero'
    if flat:
        return None, local+flat, 'flat_ambiguous'
    return best[0], local, 'ok'


def gg(d, alpha, sigma):
    if not len(d) or np.any(sigma<=0) or np.any(~np.isfinite(sigma)) or np.any((alpha<0)|(alpha>=1)):
        return None,'invalid_or_empty',None
    def trans(x):
        g=alpha*np.exp(-.5*((x-d)/sigma)**2)
        lv=.5*np.log1p(-g)
        return float(np.exp(np.sum(np.where(x<=d,lv,np.log1p(-alpha)-lv))))
    if trans(0)<=.5: return None,'already_below_at_zero',None
    if np.prod(1-alpha)>=.5: return None,'no_finite_root',None
    lo=0.; hi=max(float(np.max(d+12*sigma)),1e-10)
    if trans(hi)>.5: return None,'bracket_failure',None
    for _ in range(100):
        mid=(lo+hi)/2
        if trans(mid)>.5: lo=mid
        else: hi=mid
    root=(lo+hi)/2
    return root,'ok',abs(trans(root)-.5)


def tests():
    def calc(d,w,h): return peaks(np.array(d,float),np.array(w,float),np.array(h,float))[0]
    assert abs(calc([3],[1],[2])-3)<1e-10
    assert abs(calc([2.9,3.1],[1,1],[1,1])-3)<1e-9
    assert calc([2,4],[1.01,1],[.3,.3])<3 and calc([2,4],[1,1.01],[.3,.3])>3
    assert calc([2,4],[.2,.9],[.3,.3])>3
    assert abs(calc([7],[1],[.4])-7)<1e-10 # shifted evidence stays shifted
    assert peaks(np.array([-.1]),np.array([1.]),np.array([1.]))[2]=='no_positive_maximum'
    x=calc([2.,2.1],[.5,.8],[.4,.7]); assert abs(calc([20,21],[.5,.8],[4,7])-10*x)<1e-8
    d=np.array([2.,2.1]); w=np.array([.5,.8]); h=np.array([.4,.7]); x=calc(d,w,h)
    eps=1e-6; grad=(score(x+eps,d,w,h)-score(x-eps,d,w,h))/(2*eps)
    q=np.maximum(1-((x-d)/h)**2,0); mm=np.sum(w*q*q*d/h**2)/np.sum(w*q*q/h**2)
    assert abs(grad)<1e-7 and abs(mm-x)<1e-8 and abs(calc(d,w+[1e-5,0],h)-x)<1e-4
    r=x-d; Fs=np.sum(w*(q*q-4*q*(r/h)**2)/h**2)
    for j in range(2):
        dw=np.zeros(2); dw[j]=1e-5
        fd=(calc(d,w+dw,h)-calc(d,w-dw,h))/(2*dw[j])
        assert np.isclose(fd,-r[j]*q[j]**2/(h[j]**2*Fs),rtol=1e-5,atol=1e-7)
        dd=np.zeros(2); dd[j]=1e-5
        fd=(calc(d+dd,w,h)-calc(d-dd,w,h))/(2*dd[j])
        analytic=w[j]*(q[j]**2-4*q[j]*(r[j]/h[j])**2)/(h[j]**2*Fs)
        assert np.isclose(fd,analytic,rtol=1e-5,atol=1e-7)
    # Zero-score support endpoints and nonstationary interior points are not peaks.
    _, pp, status=peaks(np.array([2.]),np.array([1.]),np.array([.5])); assert status=='ok' and len(pp)==1 and abs(pp[0][0]-2)<1e-9
    root,status,_=gg(np.array([2.]),np.array([.6]),np.array([.2])); assert status=='ok' and root>2
    assert gg(np.array([2.]),np.array([.5]),np.array([.2]))[1]=='no_finite_root'
    return 'passed: single, middle, peak switching, weak foreground, common shift failure, scale, MM, implicit weight/center gradient central FD, GGGS finite-root cases'


def main():
    test=tests(); index=json.loads((base.DATA/'来源索引.json').read_text(encoding='utf-8'))
    model=base.indexed('20261026_夜间深度方法/inputs/scan83/point_cloud.ply',index)
    trace=base.indexed('20261026_夜间深度方法/confirm_v1/scan83/view_032.npz',index)
    meta=base.indexed('20261026_夜间深度方法/confirm_v1/scan83/metadata.json',index)
    stl=base.indexed('20261026_夜间深度方法/inputs/scan83/stl083_total.ply',index)
    with stl.open('rb') as f:
        header=[]
        while True:
            line=f.readline().decode('ascii').strip(); header.append(line)
            if line=='end_header': break
    view=next(v for v in json.loads(meta.read_text(encoding='utf-8'))['views'] if v['view_index']==32)
    f=np.load(trace); vertices=base.ply(model); W=np.array(view['world_view_transform'],float)
    H,Wid=view['image_size']; fx,fy=view['focal']; tanx,tany=np.tan(np.array(view['fov'])/2)
    chosen=np.linspace(0,len(f['offsets'])-2,32,dtype=int); rows=[]; times={k:0. for k in ('expected','native_median','directional','isotropic','gggs_proxy')}
    ref_errors=[]
    for i in chosen:
        a,b=f['offsets'][i:i+2]; p=vertices[f['gaussian_id'][a:b]]
        d=f['depth'][a:b].astype(float); w=f['weight'][a:b].astype(float); alpha=f['alpha'][a:b].astype(float)
        r=base.rotation(np.stack([p[f'rot_{j}'] for j in range(4)],axis=1).astype(float))
        scales=np.exp(np.stack([p[f'scale_{j}'] for j in range(3)],axis=1).astype(float)); scales=np.sqrt(scales**2+p['filter_3D'][:,None].astype(float)**2)
        mu=np.stack([p[n] for n in ('x','y','z')],axis=1).astype(float); cam=mu@W[:3,:3]+W[3,:3]
        uv=np.column_stack((np.clip(cam[:,0]/cam[:,2],-1.3*tanx,1.3*tanx),np.clip(cam[:,1]/cam[:,2],-1.3*tany,1.3*tany),np.ones(len(p))))
        uv/=np.linalg.norm(uv,axis=1)[:,None]; world=uv@W[:3,:3].T
        local=np.einsum('nji,nj->ni',r,world); inv_sigma=np.sqrt(np.sum((local/scales)**2,axis=1))
        x,y=f['pixel_xy'][i]; raynorm=np.sqrt(1+((x-(Wid-1)/2)/fx)**2+((y-(H-1)/2)/fy)**2)
        sigma=1/inv_sigma/raynorm; iso=np.prod(scales,axis=1)**(1/3)/raynorm
        row={'ray_index':int(i),'members':int(b-a)}
        t=time.perf_counter(); row['expected']=float(np.sum(d*w)/np.sum(w)) if len(d) and sum(w)>0 else None; times['expected']+=time.perf_counter()-t
        t=time.perf_counter(); mid=np.flatnonzero(f['transmittance'][a:b]>.5); row['native_median']=float(d[mid[-1]]) if len(mid) else None; times['native_median']+=time.perf_counter()-t
        for name,s in [('directional',sigma),('isotropic',iso)]:
            t=time.perf_counter(); value,local_peaks,status=peaks(d,w,2*s); times[name]+=time.perf_counter()-t
            row[name]=value; row[name+'_status']=status; row[name+'_peaks']=[{'depth':z,'score':v} for z,v in local_peaks]
            if value is not None:
                best=score(value,d,w,2*s); others=[(z,v) for z,v in local_peaks if abs(z-value)>1e-9]
                row[name+'_near_tie_1pct']=any(v>=.99*best for z,v in others)
                row[name+'_nearer_competing_peak']=any(z<value and v>0 for z,v in others)
                q=np.maximum(1-((value-d)/(2*s))**2,0); denom=np.sum(w*q*q/(2*s)**2)
                row[name+'_mm_residual']=abs(value-np.sum(w*q*q*d/(2*s)**2)/denom) if denom else None
                if name=='directional' and len(ref_errors)<3:
                    grid=np.linspace(max(0,float(np.min(d-2*s))),float(np.max(d+2*s)),20001)
                    vals=np.array([score(v,d,w,2*s) for v in grid]); gi=int(np.argmax(vals))
                    lo=grid[max(0,gi-1)]; hi=grid[min(len(grid)-1,gi+1)]
                    for _ in range(80):
                        x1=lo+(hi-lo)*.3819660112501051; x2=hi-(hi-lo)*.3819660112501051
                        if score(x1,d,w,2*s)<score(x2,d,w,2*s): lo=x1
                        else: hi=x2
                    ref_errors.append(float(score((lo+hi)/2,d,w,2*s)-best))
        t=time.perf_counter(); root,status,res=gg(d,alpha,sigma); times['gggs_proxy']+=time.perf_counter()-t
        row.update(gggs_proxy=root,gggs_proxy_status=status,gggs_residual=res); rows.append(row)
    paired=[abs(r['directional']-r['isotropic']) for r in rows if r['directional'] is not None and r['isotropic'] is not None]
    report={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(), 'input_model_sha256':base.sha(model),'input_trace_sha256':base.sha(trace),'stl_header':header,'stl_has_faces':any(x.startswith('element face ') for x in header),'synthetic':test,'sigma_definition':'filtered covariance, clamped Gaussian-center camera direction, 1/sqrt(v Q v) / current pixel ray norm; float64 CPU reconstruction, host binary origin unverified', 'isotropic':'filtered geometric-mean scale / current pixel ray norm; identical Tukey kernel, h=2 sigma', 'ambiguity':'stationary roots classified by scaled residual and analytic curvature; near tie is heuristic >=99% of best, nearer peak is any separate nearer nondegenerate maximum; flat roots flagged ambiguous if competitive', 'reference_grid_max_score_minus_peak':ref_errors,'runtime_seconds':times,'direction_isotropic_abs_difference':{'median':float(np.median(paired)), 'max':float(max(paired))} if paired else None,'rows':rows,'limitations':'No GT or accuracy claim. Host source/binary provenance and ID identity not independently established. GGGS proxy is not official GGGS; exact globally continuous selection is not claimed. Previous mechanism.json competition peak counts are invalid.'}
    base.OUT.mkdir(parents=True,exist_ok=True); out=base.OUT/'mechanism_v2.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'git_head':report['git_head'],'synthetic':test,'runtime_seconds':times,'stl_has_faces':report['stl_has_faces'],'paired_difference':report['direction_isotropic_abs_difference'],'rows':len(rows),'directional_near_tie':sum(r.get('directional_near_tie_1pct',False) for r in rows),'directional_nearer_peak':sum(r.get('directional_nearer_competing_peak',False) for r in rows),'gggs_status':{s:sum(r['gggs_proxy_status']==s for r in rows) for s in sorted(set(r['gggs_proxy_status'] for r in rows))}},indent=2))


if __name__=='__main__': main()
