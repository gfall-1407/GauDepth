"""固定成员组的三维短轴平面交点合成控制；不自动分组或声称真实表面精度。

先执行 prepare --manifest <路径> 冻结生成输入，再以该文件 SHA 执行
run --manifest <路径> --manifest-sha256 <SHA> --output <空目录>。
所有深度为单位射线射程；合成场景中 1 坐标单位定义为 1 米。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar


MODES = ('precision_gap', 'uniform', 'inverse_short')
MIN_RATIO = 1e-12
MIN_GAP = 1e-6
MIN_INCIDENCE = .1
RHO2 = 9.
FOOT_MARGIN = 1e-10 * max(1., RHO2)
SEED = 20261101


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def covariance(sx=.04, sy=.04, h=.01):
    return np.diag([sx*sx, sy*sy, h*h]).tolist()


def member(x, y, z, cov=None, opacity=.5, label=''):
    return dict(center=[x, y, z], covariance=cov if cov is not None else covariance(),
                opacity=opacity, label=label)


def case(name, members, origin=(0., 0., 0.), ray=(0., 0., 1.), interval=(.5, 3.),
         expected='accept', truth_t=None, allow_single=False, signs=None):
    return dict(name=name, members=members, origin=list(origin), ray=list(ray),
                interval=list(interval), expected=expected, truth_t=truth_t,
                allow_single=allow_single, normal_signs=signs)


def generate_manifest():
    rng = np.random.default_rng(SEED)
    base = [member(.02, 0, 1, covariance(.04, .06, .01), .2, 'a'),
            member(-.03, .01, 1, covariance(.04, .06, .01), .7, 'b'),
            member(.01, -.02, 1, covariance(.04, .06, .01), .9, 'c')]
    seeded = [member(float(x), float(y), 1, covariance(.05, .07, .012), label=str(i))
              for i, (x, y) in enumerate(rng.uniform(-.025, .025, (3, 2)))]
    thickness = [member(x, 0, 1, covariance(.05, .07, h), label=str(i))
                 for i, (x, h) in enumerate(zip((-.02, .03, .01), (.006, .015, .025)))]
    tangent = [member(x, y, 1, covariance(s, s*1.3, .01), label=str(i))
               for i, (x, y, s) in enumerate(((.01, 0, .03), (-.03, .01, .06), (.02, -.01, .09)))]
    near = [member(-.02, 0, 1, label='near_a'), member(.02, 0, 1, label='near_b')]
    far = [member(-.02, 0, 1.02, label='far_a'), member(.02, 0, 1.02, label='far_b')]
    noisy = [member(-.02, 0, 1, label='true_center'), member(.02, 0, 1.015, label='shifted_center')]
    thin_correct = [member(-.02, 0, 1, covariance(.04, .04, .006), label='thin_true'),
                    member(.02, 0, 1.015, covariance(.04, .04, .025), label='thick_shifted')]
    thin_wrong = [member(-.02, 0, 1, covariance(.04, .04, .025), label='thick_true'),
                  member(.02, 0, 1.015, covariance(.04, .04, .006), label='thin_shifted')]
    # Deterministic proper rotation and translation; no fitted geometry.
    a = .37
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    translation = np.array([.3, -.2, .4])
    rotated = [dict(m, center=(rotation @ np.asarray(m['center']) + translation).tolist(),
                    covariance=(rotation @ np.asarray(m['covariance']) @ rotation.T).tolist()) for m in base]
    scaled = [dict(m, center=(2*np.asarray(m['center'])).tolist(),
                   covariance=(4*np.asarray(m['covariance'])).tolist()) for m in base]
    a0, s, h = .02, .04, .01
    cases = [
        case('same_layout', base, truth_t=1.),
        case('same_seeded_layout', seeded, truth_t=1.),
        case('same_thickness', thickness, truth_t=1.),
        case('same_tangent_scale', tangent, truth_t=1.),
        case('same_opacity_change', [dict(m, opacity=o) for m, o in zip(base, (.01, .5, .99))], truth_t=1.),
        case('same_duplicate_density', base + [dict(base[0], label=f'copy_{i}') for i in range(8)], truth_t=1.),
        case('single_member_control', base[:1], truth_t=1., allow_single=True),
        case('duplicate_only_reject', [base[0], dict(base[0], label='duplicate')], expected='insufficient_unique_members'),
        case('product_peak_shift', [member(a0, 0, 1, covariance(s, s, h)),
                                    member(2*a0, 0, 1, covariance(s, s, h))],
             origin=(-1, 0, 0), ray=(2**-.5, 0, 2**-.5), truth_t=2**.5),
        case('two_layer_mixed', near + far, truth_t=1.),
        case('two_layer_oracle_near', near, truth_t=1.),
        case('two_layer_oracle_far', far, truth_t=1.02),
        case('grazing_reject', [member(2, 0, 1), member(2.02, 0, 1)],
             origin=(0, 0, .9), ray=((1-.05**2)**.5, 0, .05), expected='grazing_incidence'),
        case('isotropic_reject', [member(0, 0, 1, covariance(.01, .01, .01)),
                                  member(.02, 0, 1, covariance(.01, .01, .01))], expected='short_axis_gap'),
        case('flat_ratio_reject', [member(0, 0, 1, covariance(.01, .01, 1e-9)),
                                   member(.02, 0, 1, covariance(.01, .01, 1e-9))], expected='flat_ratio'),
        case('footprint_reject', [member(.3, 0, 1), member(.35, 0, 1)], expected='outside_footprint'),
        case('nan_center_reject', [member(None, 0, 1), member(.02, 0, 1)], expected='nonfinite_member'),
        case('singular_cov_reject', [member(0, 0, 1, covariance(.04, .04, 0)), member(.02, 0, 1)], expected='non_spd'),
        case('nonunit_ray_reject', near, ray=(0, 0, 2), expected='non_unit_ray'),
        case('common_center_shift', [member(-.02, 0, 1.015), member(.02, 0, 1.015)], truth_t=1.),
        case('asymmetric_center_noise', [member(-.02, 0, 1), member(0, .01, 1.01),
                                         member(.02, 0, 1.015)], truth_t=1.),
        case('noise_two', noisy, truth_t=1.),
        case('noise_duplicate_shifted', noisy + [dict(noisy[1], label=f'copy_{i}') for i in range(8)], truth_t=1.),
        case('heterogeneous_thin_correct', thin_correct, truth_t=1.),
        case('heterogeneous_thin_wrong', thin_wrong, truth_t=1.),
        case('rigid_rotation', rotated, origin=translation, ray=rotation @ np.array([0., 0., 1.]), truth_t=1.),
        case('uniform_scale_x2', scaled, interval=(1., 6.), truth_t=2.),
        case('normal_sign_flip', base, truth_t=1., signs=[-1, 1, -1]),
    ]
    return dict(seed=SEED, stochastic_draw='Only same_seeded_layout uses a fixed RNG draw; all other parameters are explicit.',
                units='Synthetic coordinate unit is one meter.', fixed_interval='[0.5,3.0] meter ray range except x2 scale case [1,6].',
                thresholds=dict(min_ratio_exclusive=MIN_RATIO, min_gap_exclusive=MIN_GAP,
                                min_abs_incidence_inclusive=MIN_INCIDENCE, rho=3, footprint_margin=FOOT_MARGIN),
                modes=list(MODES), cases=cases)


def solve(item, mode):
    result = dict(case=item['name'], mode=mode, status='rejected', reason='',
                  members=len(item['members']), unique_members=0, support_kind='', t_hat=np.nan,
                  point_x=np.nan, point_y=np.nan, point_z=np.nan, truth_t=item['truth_t'])
    details = [dict(case=item['name'], mode=mode, member=i, label=m.get('label',''),
                    opacity=m.get('opacity'), member_status='unprocessed', lambda0=np.nan,
                    lambda1=np.nan, lambda2=np.nan, beta_scaled=np.nan, incidence=np.nan,
                    signed_offset=np.nan, residual=np.nan, footprint_q=np.nan)
               for i, m in enumerate(item['members'])]
    origin = np.asarray(item['origin'], float)
    ray = np.asarray(item['ray'], float)
    interval = np.asarray(item['interval'], float)
    if not (origin.shape==(3,) and ray.shape==(3,) and interval.shape==(2,) and
            np.isfinite(origin).all() and np.isfinite(ray).all() and np.isfinite(interval).all() and
            0 < interval[0] < interval[1]):
        result['reason']='invalid_ray_or_interval';return result, details
    if abs(float(np.linalg.norm(ray))-1) > 1e-12:
        result['reason']='non_unit_ray';return result, details
    geometry = []
    logweights = []
    signatures = set()
    reasons = []
    if not item['members']:
        result['reason']='empty_group';return result, details
    for i, m in enumerate(item['members']):
        d = details[i]
        try:
            mu = np.asarray(m['center'], float)
            cov = np.asarray(m['covariance'], float)
            opacity = float(m['opacity'])
        except (TypeError, ValueError):
            d['member_status']='nonfinite_member';reasons.append('nonfinite_member');continue
        if not (mu.shape==(3,) and cov.shape==(3,3) and np.isfinite(mu).all() and
                np.isfinite(cov).all() and np.isfinite(opacity) and 0 <= opacity <= 1):
            d['member_status']='nonfinite_member';reasons.append('nonfinite_member');continue
        if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14*max(1., float(np.max(np.abs(cov))))):
            d['member_status']='nonsymmetric_cov';reasons.append('nonsymmetric_cov');continue
        signatures.add((mu.tobytes(), cov.tobytes()))
        lam, axes = np.linalg.eigh((cov+cov.T)/2)
        d.update(lambda0=float(lam[0]), lambda1=float(lam[1]), lambda2=float(lam[2]))
        if not np.isfinite(lam).all() or lam[0] <= 0:
            d['member_status']='non_spd';reasons.append('non_spd');continue
        if lam[0]/lam[2] <= MIN_RATIO:
            d['member_status']='flat_ratio';reasons.append('flat_ratio');continue
        gap = 1-lam[0]/lam[1]
        if gap <= MIN_GAP:
            d['member_status']='short_axis_gap';reasons.append('short_axis_gap');continue
        sign = item['normal_signs'][i] if item['normal_signs'] is not None else 1
        n = axes[:,0]*sign
        incidence = float(n @ ray)
        offset = float(n @ (mu-origin))
        d.update(incidence=incidence, signed_offset=offset)
        if abs(incidence) < MIN_INCIDENCE:
            d['member_status']='grazing_incidence';reasons.append('grazing_incidence');continue
        logweight = 0. if mode=='uniform' else -np.log(lam[0]) if mode=='inverse_short' else np.log(gap)-np.log(lam[0])
        B = axes[:,1:] @ np.diag(1/lam[1:]) @ axes[:,1:].T
        if not np.isfinite(B).all() or not np.isfinite(logweight):
            d['member_status']='invalid_precision';reasons.append('invalid_precision');continue
        geometry.append((i, mu, n, incidence, offset, B))
        logweights.append(logweight)
        d['member_status']='valid'
    result['unique_members']=len(signatures)
    if reasons:
        result['reason']=reasons[0];return result, details
    if len(signatures)<2 and not (len(item['members'])==1 and item['allow_single']):
        result['reason']='insufficient_unique_members';return result, details
    result['support_kind']='single_control' if len(item['members'])==1 else 'joint_candidate'
    logs=np.asarray(logweights, float)
    weights=np.exp(logs-np.max(logs))
    denominator=float(np.sum(weights*np.square([g[3] for g in geometry])))
    numerator=float(np.sum(weights*np.asarray([g[3]*g[4] for g in geometry])))
    if not (np.isfinite(denominator) and denominator>0 and np.isfinite(numerator)):
        result['reason']='invalid_denominator';return result, details
    t=numerator/denominator
    if not np.isfinite(t):
        result['reason']='nonfinite_solution';return result, details
    result['t_hat']=float(t)
    point=origin+t*ray
    result.update(point_x=float(point[0]), point_y=float(point[1]), point_z=float(point[2]))
    for weight, (i, mu, n, incidence, offset, B) in zip(weights, geometry):
        delta=point-mu
        details[i].update(beta_scaled=float(weight), residual=float(n @ delta),
                          footprint_q=float(delta @ B @ delta))
    if not interval[0] <= t <= interval[1]:
        result['reason']='outside_range';return result, details
    if any(d['footprint_q'] > RHO2+FOOT_MARGIN or not np.isfinite(d['footprint_q']) for d in details):
        result['reason']='outside_footprint';return result, details
    result.update(status='accepted', reason='')
    return result, details


def check_results(manifest, rows, member_rows):
    by={(r['case'],r['mode']):r for r in rows}
    checks=[]
    def check(name, passed, observed):
        def safe(value):
            if isinstance(value, dict):return {k:safe(v) for k,v in value.items()}
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):return None
            return value
        checks.append(dict(name=name, passed=bool(passed), observed=safe(observed)))
    for item in manifest['cases']:
        for mode in MODES:
            row=by[item['name'],mode]
            expected=item['expected']
            check(f"{item['name']}/{mode}/status", row['status']=='accepted' if expected=='accept' else row['reason']==expected,
                  dict(status=row['status'], reason=row['reason']))
            if expected=='accept' and item['name'] in ('same_layout','same_seeded_layout','same_thickness',
                    'same_tangent_scale','same_opacity_change','same_duplicate_density','single_member_control',
                    'product_peak_shift','two_layer_oracle_near','two_layer_oracle_far',
                    'rigid_rotation','uniform_scale_x2','normal_sign_flip'):
                check(f"{item['name']}/{mode}/exact_plane", abs(row['t_hat']-item['truth_t'])<1e-10,
                      row['t_hat']-item['truth_t'])
    a=.02;s=.04;h=.01
    peak_offset=3*a*h*h/(2*(h*h+s*s))
    product=by['product_peak_shift','precision_gap']
    product_case=next(x for x in manifest['cases'] if x['name']=='product_peak_shift')
    def negative_log_product(z):
        x=np.array([z-1, 0, z])
        return .5*sum(float((x-np.asarray(m['center'])) @
                              np.linalg.solve(np.asarray(m['covariance']), x-np.asarray(m['center'])))
                      for m in product_case['members'])
    numerical_peak=minimize_scalar(negative_log_product,bounds=(.98,1.05),method='bounded',
                                   options={'xatol':1e-15})
    check('full_product_peak_offset_mm',numerical_peak.success and
          abs(numerical_peak.x-(1+peak_offset))<1e-9 and abs(product['point_z']-1)<1e-10 and
          abs(peak_offset*1000-1.7647058823529411)<1e-12,
          dict(numerical_peak_z=float(numerical_peak.x), analytic_peak_z=1+peak_offset,
               shift_mm=(numerical_peak.x-1)*1000, plane_ls_z=product['point_z']))
    mixed=by['two_layer_mixed','precision_gap']['t_hat']
    check('mixed_layers_false_middle',abs(mixed-1.01)<1e-10 and min(abs(mixed-1),abs(mixed-1.02))>.005,mixed)
    shifted=by['common_center_shift','precision_gap']['t_hat']
    check('common_center_offset_undetectable',abs(shifted-1.015)<1e-10,shifted)
    noise=by['asymmetric_center_noise','precision_gap']['t_hat']
    check('asymmetric_noise_bias',abs(noise-1)>.001,noise)
    original=by['noise_two','precision_gap']['t_hat']
    repeated=by['noise_duplicate_shifted','precision_gap']['t_hat']
    check('copied_shifted_member_bias',repeated-original>.004,
          dict(original=original, copied=repeated, unique_members=by['noise_duplicate_shifted','precision_gap']['unique_members']))
    for name, better in [('heterogeneous_thin_correct',True),('heterogeneous_thin_wrong',False)]:
        gap=abs(by[name,'precision_gap']['t_hat']-1)
        uniform=abs(by[name,'uniform']['t_hat']-1)
        check(f'{name}/weight_not_guarantee',gap<uniform if better else gap>uniform,
              dict(precision_gap_error=gap, uniform_error=uniform,
                   inverse_short_error=abs(by[name,'inverse_short']['t_hat']-1)))
    for mode in MODES:
        base=by['same_layout',mode]['t_hat']
        check(f'rigid_invariance/{mode}',abs(by['rigid_rotation',mode]['t_hat']-base)<1e-10,
              by['rigid_rotation',mode]['t_hat']-base)
        check(f'scale_invariance/{mode}',abs(by['uniform_scale_x2',mode]['t_hat']-2*base)<1e-10,
              by['uniform_scale_x2',mode]['t_hat']-2*base)
        check(f'normal_flip_invariance/{mode}',abs(by['normal_sign_flip',mode]['t_hat']-base)<1e-10,
              by['normal_sign_flip',mode]['t_hat']-base)
        for transform in ('rigid_rotation','uniform_scale_x2','normal_sign_flip'):
            base_q=[x['footprint_q'] for x in member_rows if x['case']=='same_layout' and x['mode']==mode]
            transformed_q=[x['footprint_q'] for x in member_rows if x['case']==transform and x['mode']==mode]
            check(f'{transform}/{mode}/footprint_invariance',
                  len(base_q)==len(transformed_q) and np.allclose(base_q,transformed_q,rtol=1e-11,atol=1e-11),
                  float(np.max(np.abs(np.asarray(base_q)-np.asarray(transformed_q)))))
    return checks


def write_csv(path, rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        if args.manifest.exists():
            raise ValueError('Manifest already exists')
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(generate_manifest(),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(manifest_sha256=sha(args.manifest),cases=len(generate_manifest()['cases']))))
        return
    if not args.manifest_sha256 or not args.output:
        parser.error('run needs --manifest-sha256 and --output')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    if sha(args.manifest)!=args.manifest_sha256:
        raise ValueError('Frozen manifest SHA mismatch')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest!=generate_manifest():
        raise ValueError('Manifest differs from committed deterministic generator')
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--short'],cwd=repo,text=True):
        raise ValueError('Commit source before running')
    start=time.monotonic()
    rows, members=[],[]
    for item in manifest['cases']:
        for mode in MODES:
            row, detail=solve(item,mode)
            for member_row in detail:
                member_row.update(group_status=row['status'], group_reason=row['reason'])
            rows.append(row);members.extend(detail)
            print(json.dumps(dict(case=item['name'],mode=mode,status=row['status'],reason=row['reason'],t_hat=row['t_hat'])))
    checks=check_results(manifest,rows,members)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'cases.csv',rows)
    write_csv(args.output/'members.csv',members)
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (args.output/'metadata.json').write_text(json.dumps(dict(commit=commit,script_sha256=sha(Path(__file__)),
        manifest_sha256=sha(args.manifest),command_argv=[sys.executable,*sys.argv],
        case_count=len(manifest['cases']),method_count=len(MODES),checks_passed=sum(x['passed'] for x in checks),
        checks_total=len(checks),elapsed_seconds=time.monotonic()-start,
        limitation='Fixed synthetic member groups and local-plane least squares only; no automatic grouping or real DTU surface claim.'),indent=2)+'\n',encoding='utf-8')
    if not all(x['passed'] for x in checks):
        raise AssertionError('Recorded synthetic checks failed; see checks.json')


if __name__=='__main__':
    main()
