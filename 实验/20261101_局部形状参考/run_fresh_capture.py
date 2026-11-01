"""Fixed CPU audit of all 32 scan110 centers; no GPU, training, or root selection."""
import argparse
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time
import zipfile

import local_shape_reference as ref

EXPECTED_SHA='74fbd05c3a2c10549adca00f885cdb39c525d003ce120d1a3d593c7feb78d6ef'
CONFIG=dict(grids=[256,512],bisections=55,ellipsoid_radius_sigma=3.,
            residual_max=1e-8,relative_gap_min=.05,transversality_min=.05,
            effective_support_min=2.,grid_match_sigma_fraction=.01,
            max_seconds=600,center_count=32,neighborhood_count=9)


def read_npz(path):
    names=('seed_flat','xy','ids','count','overflow','weight','di','native_depth','native_alpha',
           'native_median','native_normal_image','unique_id','center','quaternion','scale_unaa',
           'opacity_unaa','filter_3D','scale_render','opacity_render','opacity_logit',
           'world_view_transform','intrinsics','image_size')
    arrays={}; headers={}
    with zipfile.ZipFile(path) as z:
        for name in names:
            with z.open(name+'.npy') as f:
                if f.read(8)!=b'\x93NUMPY\x01\x00': raise ValueError('expected NPY v1')
                size=struct.unpack('<H',f.read(2))[0]; h=ast.literal_eval(f.read(size).decode())
                a=array.array({'<f4':'f','<f8':'d','<i4':'i','<i8':'q','|b1':'b'}[h['descr']])
                a.frombytes(f.read()); arrays[name]=a; headers[name]=h
    return arrays,headers


class RayField:
    """Cached algebra for grid/bisection values; full reference at returned roots."""
    def __init__(self,gs,origin,ray):
        self.gs,self.origin,self.ray=gs,origin,ray; self.items=[]
        for g in gs:
            p=ref.prepare(g); d=ref.sub(origin,g['mu']); qr=ref.mv(p['Q'],ray); qd=ref.mv(p['Q'],d)
            self.items.append(dict(p=p,qd=qd,qr=qr,a=ref.dot(ray,qr),b=ref.dot(ray,qd),
                                   c=ref.dot(d,qd),logamp=math.log(g['amplitude'])))

    def bounds(self):
        intervals=[]
        for it in self.items:
            a,b,c=it['a'],it['b'],it['c']; mode=-b/a
            distance2=max(0.,c-b*b/a)
            if distance2<=9.:
                half=math.sqrt((9.-distance2)/a)
                if mode+half>0: intervals.append((max(0.,mode-half),mode+half))
        return (min(v[0] for v in intervals),max(v[1] for v in intervals)) if intervals else None

    def value(self,z,anchor):
        # Complete the square to avoid cancellation of c+2bz+az^2 near the peak.
        logs=[it['logamp']-.5*(it['a']*(z+it['b']/it['a'])**2+max(0.,it['c']-it['b']**2/it['a'])) for it in self.items]
        peak=max(logs); mass=[math.exp(v-peak) for v in logs]; total=sum(mass); pi=[v/total for v in mass]
        score=[0.]*3; matrix=ref.zeros(); trace_q=0.
        for w,it in zip(pi,self.items):
            b=[-it['qd'][k]-z*it['qr'][k] for k in range(3)]
            for k in range(3):
                score[k]+=w*b[k]
                trace_q+=w*it['p']['Q'][k][k]
                for l in range(3): matrix[k][l]+=w*it['p']['A'][k][l]
        vals,vecs=ref.eig3(matrix)
        if vals[0]-vals[1]<=1e-10: raise ValueError('degenerate_consensus_normal')
        normal=vecs[0]
        if ref.dot(normal,anchor)<0: normal=ref.mul(-1,normal)
        f=ref.dot(normal,score)
        if not all(math.isfinite(v) for v in [f,trace_q,*normal]): raise ValueError('nonfinite_field')
        return dict(F=f,normal=normal,residual_scale=math.sqrt(trace_q),pi=pi)

    def root(self,left,right,anchor):
        fl=self.value(left,anchor)['F']; fr=self.value(right,anchor)['F']
        if fl*fr>0: raise ValueError('unbracketed')
        if fl==0: z=left
        elif fr==0: z=right
        else:
            for _ in range(CONFIG['bisections']):
                z=(left+right)/2; fm=self.value(z,anchor)['F']
                if fm==0: break
                if fl*fm<0: right=z
                else: left,fl=z,fm
            z=(left+right)/2 if fm!=0 else z
        if abs(self.value(z,anchor)['F'])/self.value(z,anchor)['residual_scale']>CONFIG['residual_max']:
            raise ValueError('root_residual_failed')
        return z

    def metrics(self,z):
        st=ref.field(self.gs,ref.add(self.origin,ref.mul(z,self.ray)),self.ray)
        fz=ref.dot(st['gradF'],self.ray); gap=st['eigenvalues'][0]-st['eigenvalues'][1]
        relative_gap=gap/sum(st['eigenvalues']); tau=abs(fz)/(ref.norm(st['gradF'])*ref.norm(self.ray)) if ref.norm(st['gradF']) else 0.
        sigma=1/math.sqrt(sum(w*it['a'] for w,it in zip(st['pi'],self.items)))
        ess=1/sum(w*w for w in st['pi']); residual=abs(st['F'])/st['residual_scale']
        stable=z>0 and st['curvature']<0 and abs(fz)>1e-10
        normal=st['surface_normal']
        if normal is not None and ref.dot(normal,self.ray)>0: normal=ref.mul(-1,normal)
        return dict(z=z,normalized_residual=residual,curvature=st['curvature'],Fz=fz,
                    relative_eigen_gap=relative_gap,transversality=tau,sigma_z=sigma,effective_support=ess,
                    guide_surface_angle_degrees=ref.angle(st['normal'],st['gradF']) if normal else None,
                    surface_normal_world=normal,stable=stable,
                    admissible=stable and residual<=CONFIG['residual_max'] and relative_gap>=CONFIG['relative_gap_min']
                    and tau>=CONFIG['transversality_min'] and ess>=CONFIG['effective_support_min'])

    def search(self,bounds,cells,deadline):
        zs=[bounds[0]+(bounds[1]-bounds[0])*k/cells for k in range(cells+1)]; states=[]; failures={}; roots=[]
        def failed(reason): failures[reason]=failures.get(reason,0)+1
        for z in zs:
            if time.monotonic()>deadline: raise TimeoutError('CPU budget')
            try: states.append(self.value(z,self.ray))
            except ValueError as e: states.append(None); failed(str(e))
        for k in range(cells):
            if time.monotonic()>deadline: raise TimeoutError('CPU budget')
            left,right=states[k:k+2]
            if left is None or right is None: continue
            fl,fr=left['F'],right['F']
            if ref.dot(left['normal'],right['normal'])<0: fr=-fr
            if fl==0 or fr==0 or fl*fr<0:
                try:
                    z=self.root(zs[k],zs[k+1],left['normal'])
                    if not roots or abs(z-roots[-1]['z'])>1e-7:
                        metrics=self.metrics(z)
                        if metrics['normalized_residual']>CONFIG['residual_max']: raise ValueError('reference_root_residual_failed')
                        roots.append(metrics)
                except ValueError as e: failed(str(e))
        return dict(cells=cells,roots=roots,invalid_grid_samples=sum(v is None for v in states),failures=failures)


def quantiles(values):
    if not values: return None
    values=sorted(values)
    def q(t):
        p=(len(values)-1)*t; lo=int(p)
        return values[lo]*(1-p+lo)+values[min(lo+1,len(values)-1)]*(p-lo)
    return dict(count=len(values),min=values[0],median=q(.5),q90=q(.9),max=values[-1])


def summarize(rows):
    done=[r for r in rows if r['status']=='ok']; roots=[v for r in done for v in r['grid_512']['roots']]
    matched=[v for v in roots if v.get('matched_admissible',False)]
    return dict(denominator=32,processed=sum(r['status']!='pending' for r in rows),completed=len(done),
                statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
                rays_with_any_stable_root=sum(any(v['stable'] for v in r['grid_512']['roots']) for r in done),
                rays_with_multiple_stable_roots=sum(sum(v['stable'] for v in r['grid_512']['roots'])>=2 for r in done),
                rays_with_admissible_root=sum(any(v['admissible'] for v in r['grid_512']['roots']) for r in done),
                rays_with_grid_matched_admissible_root=sum(any(v.get('matched_admissible',False) for v in r['grid_512']['roots']) for r in done),
                total_roots_256=sum(len(r['grid_256']['roots']) for r in done),total_roots_512=len(roots),
                unmatched_fine_roots=sum(v.get('coarse_match') is None for v in roots),
                native_alpha=quantiles([r['native_alpha'] for r in done]),
                matched_candidate_metrics={k:quantiles([v[k] for v in matched]) for k in
                    ('effective_support','relative_eigen_gap','transversality','guide_surface_angle_degrees','abs_delta_expected_sigma','abs_delta_median_sigma')},
                original_scan37_64ray_gate='not evaluated: new scene and new fixed 32-ray population')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--capture',type=Path,required=True); ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); start=time.monotonic(); deadline=start+CONFIG['max_seconds']
    sha=hashlib.sha256(args.capture.read_bytes()).hexdigest(); meta=json.loads(args.metadata.read_text(encoding='utf8'))
    if sha!=EXPECTED_SHA or sha!=meta['result_sha256']: raise ValueError('capture SHA mismatch')
    d,h=read_npz(args.capture); seeds=d['seed_flat']; assert len(seeds)==32 and len(d['count'])==288
    for key in ('center','quaternion','scale_render','opacity_render','world_view_transform','intrinsics'):
        if not all(math.isfinite(v) for v in d[key]): raise ValueError('nonfinite effective field '+key)
    assert all(v>0 for v in d['scale_render']) and all(0<v<=1.000001 for v in d['opacity_render'])
    raw=d['world_view_transform']; ff=h['world_view_transform']['fortran_order']
    matrix=[[raw[i+4*j if ff else 4*i+j] for j in range(4)] for i in range(4)]
    rotation=[row[:3] for row in matrix[:3]]; origin=ref.mul(-1,ref.mv(rotation,matrix[3][:3])); fx,fy,cx,cy=d['intrinsics']
    height,width=d['image_size']; capacity=h['ids']['shape'][1]; normal_fortran=h['native_normal_image']['fortran_order']; geom={}
    for j,uid in enumerate(d['unique_id']):
        w,x,y,z=ref.unit(list(d['quaternion'][4*j:4*j+4])); rot=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
        geom[uid]=ref.gaussian(d['center'][3*j:3*j+3],d['scale_render'][3*j:3*j+3],d['opacity_render'][j],rot)
    result=dict(config=CONFIG,complete=False,capture_sha256=sha,capture_metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_capture_commit=meta['git_commit'],ply_sha256=meta['ply_sha256'],view=meta['view'],
                filter_provenance='exported scale_render/opacity_render use saved PLY filter; 64-camera recomputation was comparison only',
                opacity_logit_posinf=sum(v==math.inf for v in d['opacity_logit']),
                limitations=['single view; fixed native accepted support only','no GT accuracy','no normal-output parameter backward','no optimizer or training',
                             'finite grid may miss roots','new scan110 cohort does not complete the old scan37 64-ray protocol'],
                rays=[dict(center_index=i,flat_pixel=int(p),status='pending') for i,p in enumerate(seeds)])
    def save():
        result['seconds']=time.monotonic()-start; result['summary']=summarize(result['rays'])
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for i,pixel in enumerate(seeds):
        row=result['rays'][i]; q=i*9+4
        assert d['xy'][2*q]==pixel%width and d['xy'][2*q+1]==pixel//width
        count=d['count'][q]; row.update(query_index=q,count=int(count),native_alpha=d['native_alpha'][q],
                                      native_expected=d['native_depth'][q],native_median=d['native_median'][q])
        try:
            if time.monotonic()>deadline: raise TimeoutError('CPU budget')
            if d['overflow'][q]: raise ValueError('capture_overflow')
            if count==0: raise ValueError('empty_support')
            ids=list(d['ids'][q*capacity:q*capacity+count]); assert len(ids)==len(set(ids))
            gs=[geom[uid] for uid in ids]; ray=ref.mv(rotation,[(d['xy'][2*q]-cx)/fx,(d['xy'][2*q+1]-cy)/fy,1.])
            evaluator=RayField(gs,origin,ray); bounds=evaluator.bounds()
            if bounds is None: raise ValueError('no_positive_3sigma_ray_interval')
            row['bounds']=bounds; row['member_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            # Validate cached formula against the reviewed general 3D evaluator.
            parity=[]
            for z in (bounds[0],(bounds[0]+bounds[1])/2,bounds[1],row['native_expected'],row['native_median']):
                fast=evaluator.value(z,ray); full=ref.field(gs,ref.add(origin,ref.mul(z,ray)),ray)
                parity.append(abs(fast['F']-full['F'])/full['residual_scale'])
            row['cached_reference_parity_max']=max(parity)
            if max(parity)>1e-7: raise ValueError('cached_reference_parity_failed')
            coarse=evaluator.search(bounds,256,deadline); fine=evaluator.search(bounds,512,deadline)
            row.update(grid_256=coarse,grid_512=fine)
            pairs=sorted((abs(f['z']-c['z'])/f['sigma_z'],fi,ci) for fi,f in enumerate(fine['roots']) for ci,c in enumerate(coarse['roots']))
            used_f=set(); used_c=set()
            for distance,fi,ci in pairs:
                if distance<=CONFIG['grid_match_sigma_fraction'] and fi not in used_f and ci not in used_c:
                    f=fine['roots'][fi]; f.update(coarse_match=ci,grid_shift_sigma=distance,
                        matched_admissible=f['admissible'] and coarse['roots'][ci]['admissible']); used_f.add(fi); used_c.add(ci)
            native_cam=[d['native_normal_image'][k+3*q if normal_fortran else k*288+q] for k in range(3)]
            native_world=ref.mv(rotation,native_cam)
            for root in fine['roots']:
                root.setdefault('coarse_match',None); root.setdefault('matched_admissible',False)
                root['delta_expected']=root['z']-row['native_expected']; root['delta_median']=root['z']-row['native_median']
                root['abs_delta_expected_sigma']=abs(root['delta_expected'])/root['sigma_z']
                root['abs_delta_median_sigma']=abs(root['delta_median'])/root['sigma_z']
                root['native_surface_normal_angle_degrees']=ref.angle(root['surface_normal_world'],native_world) if root['surface_normal_world'] and ref.norm(native_world)>1e-12 else None
            row['status']='ok'
        except TimeoutError:
            row['status']='budget_exceeded'; save(); break
        except (ValueError,KeyError,AssertionError,ZeroDivisionError) as e:
            row['status']='invalid'; row['reason']=str(e) or type(e).__name__
        save()
        print(json.dumps(dict(center=i,status=row['status'],roots_512=len(row.get('grid_512',{}).get('roots',[])),seconds=round(time.monotonic()-start,2))),flush=True)
    result['complete']=all(r['status'] not in ('pending','budget_exceeded') for r in result['rays']); save()
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__': main()
