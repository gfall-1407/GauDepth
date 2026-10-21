"""CPU-only descriptive pair-conflict diagnostic for sampled Gaussian rays."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


def valid_members(d, w, n, nv):
    nl = np.linalg.norm(n, axis=1)
    normal = np.asarray(nv, bool) & np.isfinite(n).all(axis=1) & (nl > 1e-8)
    depth = np.isfinite(d) & (d > 0)
    positive = np.isfinite(w) & (w > 0)
    return depth & positive & normal, depth, positive, normal, nl


def pair_row(d, w, n, nv, scope, refz, band_z, angle):
    # Match analyze.group_ray's float64 pair calculations, including tight tolerances.
    d, w, n = (np.asarray(x, dtype=np.float64) for x in (d, w, n))
    nv = np.asarray(nv, dtype=bool)
    ok, depth, pos, normal, nl = valid_members(d, w, n, nv)
    depth_normal = depth & normal
    if scope == 'all':
        in_scope = np.ones(len(d), bool)
    elif refz is None:
        in_scope = None
    else:
        in_scope = np.isfinite(d) & (d > 0) & (np.abs(d-refz) <= band_z)
    if in_scope is None:
        return dict(status='reference_unavailable', members_total=None, positive_weight=None,
                    valid_depth_normal=None, mass_total=None, mass_positive_weight=None,
                    mass_valid_depth_normal=None, coverage_positive_weight=None,
                    coverage_valid_depth_normal=None, pairs_count=None, pair_mass=None,
                    **{f'{k}_{s}': None for k in ('depth_fail','depth_pass_normal_fail_signed',
                       'depth_pass_normal_fail_unsigned','sign_only')
                       for s in ('count','mass','ratio_count','ratio_mass')})
    ix = np.flatnonzero(in_scope)
    v = ix[ok[ix]]
    wt = np.where(np.isfinite(w[ix]) & (w[ix] > 0), w[ix], 0.)
    mass_all = float(wt.sum()); mass_pos = float(wt[np.isfinite(w[ix]) & (w[ix] > 0)].sum())
    mass_valid = float(np.where(np.isfinite(w[ix]) & (w[ix] > 0) & depth_normal[ix], w[ix], 0.).sum())
    # Temporary O(n^2) pair vectors, never retained or written; no dense matrices.
    ii, jj = np.triu_indices(len(d), 1)
    use = ok[ii] & ok[jj] & in_scope[ii] & in_scope[jj]
    ii, jj = ii[use], jj[use]
    denom_count = len(ii)
    q = w[ii] * w[jj]
    denom_mass = float(q.sum())
    depth_bad = np.abs(np.log(d[ii]) - np.log(d[jj])) / np.log1p(.01) > 1 + 1e-12
    dots = np.clip(np.einsum('ij,ij->i', n[ii], n[jj])/(nl[ii]*nl[jj]), -1., 1.)
    signed_bad = np.arccos(dots) / np.deg2rad(angle) > 1 + 1e-12
    unsigned_bad = np.arccos(np.abs(dots)) / np.deg2rad(angle) > 1 + 1e-12
    masks = {'depth_fail': depth_bad,
             'depth_pass_normal_fail_signed': ~depth_bad & signed_bad,
             'depth_pass_normal_fail_unsigned': ~depth_bad & unsigned_bad,
             'sign_only': ~depth_bad & signed_bad & ~unsigned_bad}
    out = dict(status='ok', members_total=int(len(ix)), positive_weight=int(np.sum(pos[ix])),
               valid_depth_normal=int(np.sum(depth_normal[ix])), mass_total=mass_all, mass_positive_weight=mass_pos,
               mass_valid_depth_normal=mass_valid,
               coverage_positive_weight=mass_pos/mass_all if mass_all else None,
               coverage_valid_depth_normal=mass_valid/mass_all if mass_all else None,
               pairs_count=denom_count, pair_mass=denom_mass)
    for k, mask in masks.items():
        c, m = int(mask.sum()), float(q[mask].sum())
        out[f'{k}_count'] = c; out[f'{k}_mass'] = m
        out[f'{k}_ratio_count'] = c/denom_count if denom_count else None
        out[f'{k}_ratio_mass'] = m/denom_mass if denom_mass else None
    return out


def self_check():
    d=np.array([1.,1.]); w=np.array([.5,.5]); v=np.ones(2,bool)
    same=np.tile([0.,0.,1.],(2,1))
    assert pair_row(d,w,same,v,'all',None,None,15)['depth_pass_normal_fail_signed_count']==0
    rev=np.array([[0.,0.,1.],[0.,0.,-1.]])
    r=pair_row(d,w,rev,v,'all',None,None,15)
    assert r['depth_pass_normal_fail_signed_count']==1 and r['depth_pass_normal_fail_unsigned_count']==0 and r['sign_only_count']==1
    r=pair_row(np.array([1.,1.2]),w,same,v,'all',None,None,15)
    assert r['depth_fail_count']==1 and r['depth_pass_normal_fail_signed_count']==0
    # One weak member creates conflicts for 2/3 pairs, but almost no pair mass.
    d=np.ones(3); n=np.array([[0.,0,1],[0,0,1],[0,0,-1.]])
    r=pair_row(d,np.array([.5,.5,1e-8]),n,np.ones(3,bool),'all',None,None,15)
    assert np.isclose(r['depth_pass_normal_fail_signed_ratio_count'], 2/3)
    assert r['depth_pass_normal_fail_signed_ratio_mass'] < 1e-6
    for count in (0,1):
        r=pair_row(np.ones(count),np.ones(count),np.empty((count,3)),np.ones(count,bool),'all',None,None,15)
        assert r['pairs_count']==0 and r['depth_fail_ratio_count'] is None
    miss=pair_row(np.ones(1),np.ones(1),same[:1],v[:1],'band_2',None,None,15.)
    normal_keys=set(pair_row(np.ones(1),np.ones(1),same[:1],v[:1],'all',None,None,15.))
    assert set(miss)==normal_keys
    # Match analyze.group_ray: normalized log depth and 1e-12 tolerance at 90°.
    r=pair_row(np.ones(2),w,np.array([[1.,0,0],[0,1.,0]]),v,'all',None,None,90.)
    assert r['depth_pass_normal_fail_signed_count']==0


def run(inputs, reference_csv, output):
    files=[]
    for p in inputs:
        p=Path(p); files.extend(sorted(p.glob('view_*.npz')) if p.is_dir() else [p])
    files=sorted(set(p.resolve() for p in files))
    if not files: raise ValueError('No NPZ inputs')
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Output must be new/empty: {output}')
    refs={}
    with Path(reference_csv).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=(row['scan'].lower(),row['view'],int(row['ray']),int(row['half_band_voxels']))
            val=(row['reference_z'],row['half_band_camera_z'],row['status'])
            if key in refs and refs[key]!=val: raise ValueError(f'Inconsistent reference rows: {key}')
            refs[key]=val
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; manifests=[]
    for path in files:
        mp=path.parent/'metadata.json'; meta=json.loads(mp.read_text(encoding='utf-8'))
        scan=path.parent.name.lower()
        m=re.match(r'(scan\d+)',scan)
        if not m: raise ValueError(f'Unrecognized input directory: {path.parent}')
        scan=m.group(1)
        view=path.stem
        vi=int(view.split('_')[-1]); vm=next((x for x in meta.get('views',[]) if int(x['view_index'])==vi),None)
        if vm is None or vm['npz_sha256']!=sha(path): raise ValueError(f'Manifest SHA/view mismatch: {path}')
        with np.load(path,allow_pickle=False) as z:
            off,d,w,n,nv=z['offsets'],z['depth'],z['weight'],z['normal'],z['normal_valid']
            nr=len(off)-1
            if off[0]!=0 or off[-1]!=len(d) or np.any(np.diff(off)<0) or any(len(a)!=len(d) for a in (w,n,nv)):
                raise ValueError(f'Invalid ragged schema: {path}')
            if len(z['qualitative'])!=nr: raise ValueError(f'Invalid qualitative schema: {path}')
            for ray in range(nr):
                sl=slice(int(off[ray]),int(off[ray+1])); qual=bool(z['qualitative'][ray])
                base=dict(scan=scan,view=view,ray=ray,qualitative=qual)
                for band in (None,2,4,8):
                    ref=None; hz=None; status='not_applicable'
                    if band is not None:
                        value=refs.get((scan,view,ray,band))
                        if value is None: status='reference_row_missing'
                        else:
                            rz,bz,status=value
                            ref=float(rz) if status=='hit' and rz else None
                            hz=float(bz) if status=='hit' and bz else None
                    for angle in (15.,30.,60.):
                        scope='all' if band is None else f'band_{band}'
                        metric=pair_row(d[sl],w[sl],n[sl],nv[sl],scope,ref,hz,angle)
                        rows.append(dict(**base,scope=scope,angle_degrees=angle,
                                         reference_status=status,reference_z=ref,half_band_camera_z=hz,**metric))
        manifests.append(dict(npz=str(path),npz_sha256=sha(path),metadata=str(mp),metadata_sha256=sha(mp),scan=scan,view=view))
    write_csv(output/'rays.csv',rows)
    keys=[k for k in rows[0] if k.endswith('_ratio_count') or k.endswith('_ratio_mass')]
    summaries=[]
    for scope in ('all','band_2','band_4','band_8'):
        for angle in (15.,30.,60.):
            sub=[r for r in rows if r['scope']==scope and r['angle_degrees']==angle and not r['qualitative']]
            groups=[('all',None,None)] + [(s,v, [r for r in sub if r['scan']==s and r['view']==v])
                      for s,v in sorted(set((r['scan'],r['view']) for r in sub))]
            for scan,view,part in groups:
                part=sub if scan=='all' else part
                summaries.append(dict(scan=scan,view=view,scope=scope,angle_degrees=angle,
                    nonqualitative_rays=len(part),
                    **{k+'_rays':sum(r[k] is not None for r in part) for k in keys},
                    denominator_count_rays=sum(r['pairs_count'] not in (None,0) for r in part),
                    denominator_mass_rays=sum(r['pair_mass'] is not None and r['pair_mass']>0 for r in part),
                    status_ok_rays=sum(r['status']=='ok' for r in part),
                    reference_unavailable_rays=sum(r['status']=='reference_unavailable' for r in part),
                    **{f'reference_{name}_rays':sum(r['reference_status']==name for r in part)
                       for name in ('not_applicable','hit','miss','hit_replay_mismatch','reference_row_missing')},
                    mean_coverage_positive_weight=float(np.mean([r['coverage_positive_weight'] for r in part if r['coverage_positive_weight'] is not None])) if any(r['coverage_positive_weight'] is not None for r in part) else None,
                    mean_coverage_valid_depth_normal=float(np.mean([r['coverage_valid_depth_normal'] for r in part if r['coverage_valid_depth_normal'] is not None])) if any(r['coverage_valid_depth_normal'] is not None for r in part) else None,
                    **{'mean_'+k:float(np.mean([r[k] for r in part if r[k] is not None])) if any(r[k] is not None for r in part) else None for k in keys}))
    write_csv(output/'view_summary.csv',summaries)
    (output/'metadata.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),inputs=manifests,
        reference_csv=str(Path(reference_csv).resolve()),reference_sha256=sha(reference_csv),
        interpretation='Original contribution pair mass is descriptive, not probability/support; signed constraints are directed, unsigned checks are unoriented diagnostics; reference mesh is a work reference.',
        qualitative_excluded_from_summary=True),indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',help='NPZ file or directory; repeatable')
    p.add_argument('--reference-csv'); p.add_argument('--output')
    p.add_argument('--self-check',action='store_true'); a=p.parse_args()
    if a.self_check: self_check()
    elif not a.input or not a.reference_csv or not a.output: p.error('Provide --input, --reference-csv and --output')
    else: run(a.input,Path(a.reference_csv),Path(a.output))
