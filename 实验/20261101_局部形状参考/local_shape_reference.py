"""CPU-only normal-guided Gaussian density ridge reference; Python stdlib only.

Returns local root branches, NOT a visibility decision or a trained depth plugin.
All derivatives include the spatial/parameter derivative of the consensus normal.
The finite sign-change grid does not certify that every root has been found.
"""
import argparse
import array
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile


def dot(a, b): return sum(x*y for x, y in zip(a, b))
def add(a, b): return [x+y for x, y in zip(a, b)]
def sub(a, b): return [x-y for x, y in zip(a, b)]
def mul(s, a): return [s*x for x in a]
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(1/norm(a), a)
def outer(a, b): return [[x*y for y in b] for x in a]
def zeros(): return [[0.0]*3 for _ in range(3)]
def eye(): return [[float(i == j) for j in range(3)] for i in range(3)]
def mv(a, b): return [dot(row, b) for row in a]
def madd(a, b): return [add(x, y) for x, y in zip(a, b)]
def mscale(s, a): return [mul(s, row) for row in a]
def transpose(a): return [list(x) for x in zip(*a)]
def mm(a, b): return [[dot(row, col) for col in zip(*b)] for row in a]
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def eig3(a):
    """Jacobi eigensystem, descending eigenvalues, column eigenvectors."""
    a, v = copy.deepcopy(a), eye()
    for _ in range(40):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-14: break
        theta = .5*math.atan2(2*a[p][q], a[q][q]-a[p][p])
        c, s = math.cos(theta), math.sin(theta)
        j = eye(); j[p][p] = j[q][q] = c; j[p][q] = s; j[q][p] = -s
        a = mm(transpose(j), mm(a, j)); v = mm(v, j)
    order = sorted(range(3), key=lambda i: a[i][i], reverse=True)
    return [a[i][i] for i in order], [[v[k][i] for k in range(3)] for i in order]


def gaussian(mu, scale, amplitude=1.0, rotation=None):
    return dict(mu=list(mu), scale=list(scale), amplitude=amplitude, rotation=rotation or eye())


def prepare(g):
    s, r = g['scale'], g['rotation']
    order = sorted(range(3), key=lambda k: s[k])
    lo, mid = order[:2]
    n = [row[lo] for row in r]
    gamma = 1-(s[lo]/s[mid])**2
    q = zeros()
    for k in range(3):
        axis = [row[k] for row in r]
        q = madd(q, mscale(1/s[k]**2, outer(axis, axis)))
    return dict(Q=q, n=n, gamma=gamma, A=mscale(gamma, outer(n, n)), lo=lo, mid=mid)


def eigen_jvp(vals, vecs, normal, dnmat):
    out = [0.0]*3
    for j in (1, 2):
        out = add(out, mul(dot(vecs[j], mv(dnmat, normal))/(vals[0]-vals[j]), vecs[j]))
    return out


def field(gs, x, orient=(0., 0., 1.)):
    items, logs = [], []
    for g in gs:
        p = prepare(g); delta = sub(x, g['mu']); b = mul(-1, mv(p['Q'], delta))
        items.append(dict(g=g, p=p, delta=delta, b=b))
        logs.append(math.log(g['amplitude'])-.5*dot(delta, mv(p['Q'], delta)))
    peak = max(logs); exp = [math.exp(v-peak) for v in logs]; total = sum(exp)
    pi = [v/total for v in exp]; score = [0.0]*3; nmat = zeros(); second = zeros()
    for w, item in zip(pi, items):
        score = add(score, mul(w, item['b']))
        nmat = madd(nmat, mscale(w, item['p']['A']))
        second = madd(second, mscale(w, madd(outer(item['b'], item['b']), mscale(-1, item['p']['Q']))))
    hess = madd(second, mscale(-1, outer(score, score)))
    vals, vecs = eig3(nmat)
    if vals[0]-vals[1] <= 1e-10:
        raise ValueError('undefined consensus normal: eigenvalue gap <= 1e-10')
    normal = vecs[0]
    if dot(normal, orient) < 0: normal = mul(-1, normal)
    grad_f = mv(hess, normal)
    for k in range(3):
        dnmat = zeros()
        for w, item in zip(pi, items):
            dnmat = madd(dnmat, mscale(w*(item['b'][k]-score[k]), item['p']['A']))
        grad_f[k] += dot(eigen_jvp(vals, vecs, normal, dnmat), score)
    return dict(F=dot(normal, score), gradF=grad_f, curvature=dot(normal, mv(hess, normal)),
                normal=normal, surface_normal=unit(grad_f) if norm(grad_f)>1e-12 else None,
                eigenvalues=vals, log_density=peak+math.log(total), pi=pi, score=score,
                items=items, vecs=vecs)


def parameter_jvp(state, index, kind, axis):
    """Exact F_theta for one center/log-scale/log-amplitude/world-rotation parameter."""
    item = state['items'][index]; g, p, delta = item['g'], item['p'], item['delta']
    dmu = [0.0]*3; dq = zeros(); da = zeros(); dloga = 0.0
    if kind == 'center': dmu[axis] = 1.0
    elif kind == 'log_amplitude': dloga = 1.0
    elif kind == 'log_scale':
        raxis = [row[axis] for row in g['rotation']]
        dq = mscale(-2/g['scale'][axis]**2, outer(raxis, raxis))
        ratio2 = (g['scale'][p['lo']]/g['scale'][p['mid']])**2
        dgamma = -2*ratio2 if axis == p['lo'] else 2*ratio2 if axis == p['mid'] else 0.0
        da = mscale(dgamma, outer(p['n'], p['n']))
    elif kind == 'rotation':
        e = [float(k == axis) for k in range(3)]
        omega = transpose([cross(e, col) for col in eye()])
        dq = madd(mm(omega, p['Q']), mscale(-1, mm(p['Q'], omega)))
        dn = cross(e, p['n']); da = mscale(p['gamma'], madd(outer(dn, p['n']), outer(p['n'], dn)))
    else: raise ValueError(kind)
    dlog = dloga + dot(dmu, mv(p['Q'], delta))-.5*dot(delta, mv(dq, delta))
    db = sub(mv(p['Q'], dmu), mv(dq, delta))
    mean_dlog = state['pi'][index]*dlog; dg = [0.0]*3; dnmat = zeros()
    for i, (w, it) in enumerate(zip(state['pi'], state['items'])):
        dw = w*((dlog if i == index else 0.0)-mean_dlog)
        dg = add(dg, mul(dw, it['b'])); dnmat = madd(dnmat, mscale(dw, it['p']['A']))
        if i == index:
            dg = add(dg, mul(w, db)); dnmat = madd(dnmat, mscale(w, da))
    dn = eigen_jvp(state['eigenvalues'], state['vecs'], state['normal'], dnmat)
    return dot(dn, state['score'])+dot(state['normal'], dg)


def shifted(gs, i, kind, axis, step):
    out = copy.deepcopy(gs); g = out[i]
    if kind == 'center': g['mu'][axis] += step
    elif kind == 'log_amplitude': g['amplitude'] *= math.exp(step)
    elif kind == 'log_scale': g['scale'][axis] *= math.exp(step)
    elif kind == 'rotation':
        e = [float(k == axis) for k in range(3)]
        cols = transpose(g['rotation']); c, s = math.cos(step), math.sin(step)
        cols = [add(add(mul(c, v), mul(s, cross(e, v))), mul((1-c)*dot(e, v), e)) for v in cols]
        g['rotation'] = transpose(cols)
    return out


def root_in_bracket(gs, ray, left, right, origin=(0., 0., 0.)):
    def f(z): return field(gs, add(origin, mul(z, ray)), ray)['F']
    fl, fr = f(left), f(right)
    if fl == 0: return left
    if fr == 0: return right
    if fl*fr > 0: raise ValueError('root is not bracketed')
    for _ in range(55):
        mid = (left+right)/2; fm = f(mid)
        if fm == 0: return mid
        if fl*fm < 0: right = mid
        else: left, fl = mid, fm
    return (left+right)/2


def root_branches(gs, ray, bounds, cells=256, origin=(0., 0., 0.)):
    """Finite diagnostic only: sign-changing roots, no first-hit selection."""
    zs = [bounds[0]+(bounds[1]-bounds[0])*i/cells for i in range(cells+1)]
    fs = [field(gs, add(origin, mul(z, ray)), ray)['F'] for z in zs]; roots = []
    for i in range(cells):
        if fs[i] == 0 or fs[i]*fs[i+1] < 0 or fs[i+1] == 0:
            z = root_in_bracket(gs, ray, zs[i], zs[i+1], origin=origin)
            if not roots or abs(z-roots[-1]['z'])>1e-7:
                st = field(gs, add(origin, mul(z, ray)), ray); fz = dot(st['gradF'], ray)
                surface_normal=st['surface_normal']
                if surface_normal is not None and dot(surface_normal,ray)>0:
                    surface_normal=mul(-1,surface_normal)
                roots.append(dict(z=z, curvature=st['curvature'], Fz=fz,
                                  stable=st['curvature']<0 and abs(fz)>1e-10,
                                  log_density=st['log_density'],guide_normal=st['normal'],
                                  surface_normal=surface_normal,eigen_gap=st['eigenvalues'][0]-st['eigenvalues'][1],
                                  effective_support=1/sum(w*w for w in st['pi'])))
    return roots


def angle(a, b): return math.degrees(math.acos(min(1., max(-1., abs(dot(unit(a), unit(b)))))))


def controlled_cases():
    n, tangent, other = [.6, 0., .8], [.8, 0., -.6], [0., 1., 0.]
    rotation = transpose([tangent, other, n]); anchor = [0., 0., 2.]; ray = [.3, .1, 1.]
    plane = [gaussian(add(anchor, add(mul(t, tangent), mul(v, other))), [.55+.04*i, .35+.03*i, .06+.006*i],
                      amplitude=.2+.11*i, rotation=rotation) for i,(t,v) in enumerate([(-.5,-.2),(-.3,.3),(.1,-.1),(.4,.25),(.6,-.35)])]
    roots = root_branches(plane, ray, (1.1, 2.2)); expected = dot(n, anchor)/dot(n, ray)
    assert len(roots)==1 and abs(roots[0]['z']-expected)<1e-9 and roots[0]['stable']
    def layers(a): return [gaussian([0., 0., 2.+sign*a], [.5,.4,.1], .5) for sign in (-1.,1.)]
    resolved = root_branches(layers(.15), [0.,0.,1.], (1.6,2.4))
    unresolved = root_branches(layers(.05), [0.,0.,1.], (1.6,2.4))
    assert len(resolved)==3 and [v['stable'] for v in resolved]==[True,False,True]
    assert len(unresolved)==1 and unresolved[0]['stable'] and abs(unresolved[0]['z']-2.)<1e-10
    unequal_near=layers(.15); unequal_near[0]['amplitude']=.005
    unequal_far=layers(.25); unequal_far[0]['amplitude']=.005
    unequal_near_roots=root_branches(unequal_near,[0.,0.,1.],(1.5,2.5),512)
    unequal_far_roots=root_branches(unequal_far,[0.,0.,1.],(1.5,2.5),512)
    assert sum(r['stable'] for r in unequal_near_roots)==1
    assert sum(r['stable'] for r in unequal_far_roots)==2
    low = copy.deepcopy(plane)
    for g in low: g['amplitude'] *= 1e-9
    low_roots = root_branches(low, ray, (1.1,2.2))
    assert abs(low_roots[0]['z']-roots[0]['z'])<1e-10
    floaters = [gaussian([0,0,1.],[.2,.15,.02],.005), gaussian([0,0,2.],[.5,.4,.1],.9)]
    floater_roots = root_branches(floaters, [0.,0.,1.], (.8,2.3), 512)
    assert len([r for r in floater_roots if r['stable']])==2
    try: field([gaussian([0,0,2.],[.1,.1,.1])], [0,0,2.])
    except ValueError: isotropic_rejected = True
    else: isotropic_rejected = False
    assert isotropic_rejected
    curved = [gaussian([-.25,0.,1.96],[.5,.35,.1],.6), gaussian([.3,.05,2.08],[.55,.4,.13],.8)]
    curved = shifted(curved,0,'rotation',1,.25); curved = shifted(curved,1,'rotation',1,-.3)
    ray2 = [.08,.02,1.]; branch = (1.8,2.25); z = root_in_bracket(curved,ray2,*branch)
    state = field(curved,mul(z,ray2),ray2); fz = dot(state['gradF'],ray2); derivatives=[]; eps=1e-5
    for i,kind,axis in [(0,'center',0),(0,'center',2),(1,'log_amplitude',0),(0,'log_scale',0),(0,'log_scale',2),(1,'rotation',1)]:
        implicit = -parameter_jvp(state,i,kind,axis)/fz
        plus=root_in_bracket(shifted(curved,i,kind,axis,eps),ray2,*branch)
        minus=root_in_bracket(shifted(curved,i,kind,axis,-eps),ray2,*branch)
        fd=(plus-minus)/(2*eps); error=abs(implicit-fd)/max(abs(implicit),abs(fd),1e-5)
        derivatives.append(dict(index=i,parameter=kind,axis=axis,implicit=implicit,root_finite_difference=fd,relative_error=error))
        assert error<1e-4, derivatives[-1]
    rp=add(ray2,[eps,0.,0.]); rm=sub(ray2,[eps,0.,0.])
    qfd=(root_in_bracket(curved,rp,*branch)-root_in_bracket(curved,rm,*branch))/(2*eps)
    qanalytic=-z*state['gradF'][0]/fz
    assert abs(qanalytic-qfd)<1e-6
    # Independent spatial derivative check catches omission of normal-field variation.
    grad_fd=[]
    for k in range(3):
        delta=[float(j==k)*eps for j in range(3)]; point=mul(z,ray2)
        grad_fd.append((field(curved,add(point,delta),ray2)['F']-field(curved,sub(point,delta),ray2)['F'])/(2*eps))
    assert norm(sub(grad_fd,state['gradF']))/norm(grad_fd)<1e-5
    c,s=math.cos(.7),math.sin(.7); transform=[[c,-s,0.],[s,c,0.],[0.,0.,1.]]; translation=[.3,-.2,.5]
    transformed=copy.deepcopy(curved)
    for g in transformed:
        g['mu']=add(mv(transform,g['mu']),translation); g['rotation']=mm(transform,g['rotation'])
    transformed_root=root_in_bracket(transformed,mv(transform,ray2),*branch,origin=translation)
    assert abs(transformed_root-z)<1e-10
    return dict(single_plane=dict(true_camera_z=expected,roots=roots),
                resolved_double_layer=dict(sigma=.1,separation=.3,roots=resolved),
                unresolved_double_layer=dict(sigma=.1,separation=.1,roots=unresolved,verdict='stable root lies on neither input plane'),
                unequal_double_layer=dict(amplitude_ratio=.01,sigma=.1,
                    separation_3sigma_roots=unequal_near_roots,separation_5sigma_roots=unequal_far_roots,
                    verdict='weak branch lost at 3sigma, present at 5sigma; 2sigma threshold only applies to symmetric case'),
                low_amplitude=dict(multiplier=1e-9,roots=low_roots,verdict='root has no absolute opacity confidence'),
                weak_front_floater=dict(amplitude=.005,roots=floater_roots,verdict='local geometry alone cannot select visible branch'),
                isotropic_rejected=isotropic_rejected,
                varying_normal=dict(root=z,guide_surface_normal_angle_degrees=angle(state['normal'],state['gradF']),
                                    rigid_transform_root_error=abs(transformed_root-z),
                                    spatial_derivative_relative_error=norm(sub(grad_fd,state['gradF']))/norm(grad_fd),
                                    parameter_derivatives=derivatives,query_ray_x_derivative=dict(implicit=qanalytic,finite_difference=qfd)))


def read_capture(path):
    """Read only existing shape/alpha capture; does NOT invent opacity/filter."""
    z=zipfile.ZipFile(path); out={}
    for k in ('ids','alpha','count','unique_id','center','quaternion','scale_unaa',
              'focal','xy','world_view_transform','flat_pixel','native_expected','native_median'):
        f=z.open(k+'.npy'); magic=f.read(8)
        if magic != b'\x93NUMPY\x01\x00': raise ValueError('only NPY v1 supported')
        size=struct.unpack('<H',f.read(2))[0]; header=ast.literal_eval(f.read(size).decode())
        a=array.array({'<f4':'f','<f8':'d','<i4':'i','<i8':'q'}[header['descr']]); a.frombytes(f.read()); out[k]=a
        if k=='world_view_transform': out['matrix_fortran_order']=header['fortran_order']
    return out


def capture_audit(path):
    d=read_capture(path); geom={}
    for j,uid in enumerate(d['unique_id']):
        w,x,y,z=unit(list(d['quaternion'][4*j:4*j+4]))
        r=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
           [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
           [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
        g=gaussian(d['center'][3*j:3*j+3],d['scale_unaa'][3*j:3*j+3],rotation=r)
        geom[uid]=(g,prepare(g))
    rows=[]; eligible=[]; capacity=len(d['ids'])//len(d['count'])
    for j,count in enumerate(d['count']):
        t=1.; members=[]; nmat=zeros()
        for k in range(count):
            idx=j*capacity+k; a=d['alpha'][idx]; w=t*a; t*=1-a
            g,p=geom[d['ids'][idx]]; members.append((w,g,p)); nmat=madd(nmat,mscale(w,p['A']))
        opacity=1-t
        if opacity<.9: continue
        eligible.append(j)
        vals,vecs=eig3(nmat); n=vecs[0]; coherence=vals[0]/sum(vals); cover=[]
        for bandwidth in (1.,2.):
            events=[]
            for w,g,p in members:
                h=dot(n,g['mu']); projections=mv(transpose(g['rotation']),n)
                sigma=norm([v*s for v,s in zip(projections,g['scale'])])
                events.extend([(h-bandwidth*sigma,0,w/opacity),(h+bandwidth*sigma,1,-w/opacity)])
            current=best=0.
            for _,_,change in sorted(events): current+=change; best=max(best,current)
            cover.append(best)
        mass_flat=sum(w/opacity for w,g,p in members if g['scale'][p['lo']]/g['scale'][p['mid']]<=.5)
        rows.append(dict(coherence=coherence,cover1=cover[0],cover2=cover[1],mass_flat=mass_flat,
                         effective_count=1/sum((w/opacity)**2 for w,_,_ in members)))
    def quantile(values,f):
        values=sorted(values); p=(len(values)-1)*f; lo=int(p)
        return values[lo]*(1-p+lo)+values[min(lo+1,len(values)-1)]*(p-lo)
    def summary(rs):
        return dict(count=len(rs),fields={k:dict(mean=sum(r[k] for r in rs)/len(rs),q10=quantile([r[k] for r in rs],.1),
                    median=quantile([r[k] for r in rs],.5),q90=quantile([r[k] for r in rs],.9)) for k in rows[0]})
    # A second, finite real-shape diagnostic: points where the actual ray crosses
    # each primitive's shortest-axis center plane. No density amplitudes invented.
    # Maximum coverage among these finite candidates is not a global optimum.
    raw=d['world_view_transform']; ff=d['matrix_fortran_order']
    matrix=[[raw[i+4*j if ff else 4*i+j] for j in range(4)] for i in range(4)]
    camera_rotation=[row[:3] for row in matrix[:3]]; origin=mul(-1,mv(camera_rotation,matrix[3][:3]))
    fx,fy,cx,cy=d['focal']; ray_rows=[]
    for j in [eligible[len(eligible)*k//min(64,len(eligible))] for k in range(min(64,len(eligible)))]:
        ray=mv(camera_rotation,[(d['xy'][2*j]-cx)/fx,(d['xy'][2*j+1]-cy)/fy,1.])
        t=1.; members=[]; candidates=[]
        for k in range(d['count'][j]):
            idx=j*capacity+k; a=d['alpha'][idx]; w=t*a; t*=1-a; g,p=geom[d['ids'][idx]]
            members.append((w,g,p)); denom=dot(p['n'],ray)
            if abs(denom)>1e-8:
                z=dot(p['n'],sub(g['mu'],origin))/denom
                if z>0: candidates.append(z)
        opacity=1-t
        def coverage(z):
            point=add(origin,mul(z,ray)); covered=[[],[]]
            for w,g,p in members:
                delta=sub(point,g['mu']); q=dot(delta,mv(p['Q'],delta))
                for slot,threshold in enumerate((1.,4.)):
                    if q<=threshold: covered[slot].append(w/opacity)
            return [dict(mass=sum(ws),members=len(ws),effective_members=sum(ws)**2/sum(v*v for v in ws) if ws else 0.) for ws in covered]
        best=[dict(z=None,mass=0.,members=0,effective_members=0.) for _ in range(2)]
        for z in candidates:
            for slot,cov in enumerate(coverage(z)):
                if cov['mass']>best[slot]['mass']: best[slot]=dict(z=z,**cov)
        ray_rows.append(dict(flat_pixel=int(d['flat_pixel'][j]),accepted=int(d['count'][j]),candidate_count=len(candidates),
                             best_1sigma=best[0],best_2sigma=best[1],
                             native_expected_coverage=coverage(d['native_expected'][j]),
                             native_median_coverage=coverage(d['native_median'][j])))
    finite_summary={key:dict(median_mass=quantile([r[key]['mass'] for r in ray_rows],.5),
        median_effective_members=quantile([r[key]['effective_members'] for r in ray_rows],.5),
        rays_with_at_least_two_members=sum(r[key]['members']>=2 for r in ray_rows)) for key in ('best_1sigma','best_2sigma')}
    return dict(capture_path=str(path),capture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                interpretation='shape/accepted-contribution diagnostic only; unfiltered scales; no real density evaluation',
                projection_bound='maximum overlapping projected ellipsoid intervals; necessary, not sufficient, for common 3D support',
                high_opacity=summary(rows),high_opacity_coherent=summary([r for r in rows if r['coherence']>=.8]),
                finite_ray_candidates=dict(selection='64 evenly spaced capture-order rays among opacity>=0.9; all positive shortest-axis plane intersections',
                    interpretation='geometry-only finite candidate coverage, not density-ridge roots or depth accuracy',summary=finite_summary,rays=ray_rows))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--capture',type=Path); ap.add_argument('--output',type=Path)
    args=ap.parse_args(); result=dict(definition='normal-guided Gaussian log-density stationary branches',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python_version=sys.version,
        host_baseline_commit='8474727dd1c1f4fee77a3dc02a6cbc0b38075f21',
        visibility_selection='not implemented; separate research interface',
        root_search='finite sign-change grid plus 55 bisection steps; not exhaustive',controlled=controlled_cases())
    if args.capture: result['real_shape_diagnostic']=capture_audit(args.capture)
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.output: args.output.write_text(text+'\n',encoding='utf-8')
    else: print(text)


if __name__=='__main__': main()
