"""Fit-only continuous shared graph curvature versus equally optimized planes; oracle K2."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize

from analyze import pair_metrics
from joint_plane_probe import curve_candidates, curve_energy, curve_labels, csv_write, sha

LENGTH = .01
SCALE = .002
LAMBDA = 1.


def physical(parameters, zref, curved):
    coefficients = np.asarray(parameters[:6]).reshape(2, 3).copy()
    coefficients[:, 0] = zref + SCALE * coefficients[:, 0]
    coefficients[:, 1:] *= SCALE / LENGTH
    d = SCALE / LENGTH**2 * parameters[6] if curved else 0.
    return coefficients, float(d)


def normalized(coefficients, zref):
    result = np.asarray(coefficients).copy()
    result[:, 0] = (result[:, 0] - zref) / SCALE
    result[:, 1:] *= LENGTH / SCALE
    return result.ravel()


def pooled_start(points, zref, curved):
    uv = points[:, :2] / LENGTH
    columns = [np.ones(len(points)), uv[:, 0], uv[:, 1]]
    if curved:
        columns.append((uv**2).sum(axis=1))
    design = np.column_stack(columns)
    target = (points[:, 2] - zref) / SCALE
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    # Two source-free residual-height quantiles, not truth gap or curvature.
    offsets = np.quantile(target - design[:, 1:] @ solution[1:], [.25, .75])
    result = np.column_stack((offsets, np.tile(solution[1:3], (2, 1)))).ravel()
    return np.r_[result, solution[3]] if curved else result


def optimize_model(points, normal, starts, zref, curved):
    invalid_calls = 0

    def objective(parameters):
        nonlocal invalid_calls
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            coefficients, d = physical(parameters, zref, curved)
            costs = curve_energy(points, normal, coefficients, d, SCALE, LAMBDA)
            value = float(costs.min(axis=1).mean())
        if not np.isfinite(value):
            invalid_calls += 1
            return 1e100
        return value

    trials = []
    candidates = []
    for name, initial in starts:
        initial_value = objective(initial)
        if initial_value < 1e100:
            candidates.append((initial_value, np.asarray(initial).copy(), name + ':initial'))
        result = minimize(objective, initial, method='L-BFGS-B',
                          options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
        value = objective(result.x)
        trials.append(dict(start=name, initial_objective=initial_value, objective=value,
                           success=bool(result.success), status=int(result.status), message=str(result.message),
                           iterations=int(result.nit), evaluations=int(result.nfev)))
        if value < 1e100:
            candidates.append((value, result.x.copy(), name + ':optimized'))
    if not candidates:
        return dict(status='nonfinite_failure', trials=trials, invalid_calls=invalid_calls)
    value, parameters, selected = min(candidates, key=lambda item: item[0])
    coefficients, d = physical(parameters, zref, curved)
    return dict(status='finite_candidate', objective=value, parameters=parameters.tolist(),
                coefficients=coefficients.tolist(), d=d, selected=selected,
                trials=trials, invalid_calls=invalid_calls)


def fit_models(points, normal, finite_coefficients):
    # Only these fit arrays enter optimization; heldout arrays and origins are absent.
    zref = float(np.median(points[:, 2]))
    plane = optimize_model(points, normal,
        [('pooled_plane', pooled_start(points, zref, False)),
         ('saved_fit_only_plane', normalized(finite_coefficients, zref))], zref, False)
    if plane['status'] != 'finite_candidate':
        return plane, dict(status='plane_failed', trials=[], invalid_calls=0)
    curve = optimize_model(points, normal,
        [('optimized_plane_d0', np.r_[plane['parameters'], 0.]),
         ('pooled_quadratic', pooled_start(points, zref, True))], zref, True)
    if curve['status'] == 'finite_candidate':
        assert curve['objective'] <= plane['objective'] + 1e-12
    return plane, curve


def self_check():
    xy = np.array([[-.01,-.005],[.005,.01],[.01,-.01],[-.006,.004],[.002,-.007],[.008,.003]])
    points = np.vstack([np.column_stack((xy, a + 23.7 * (xy**2).sum(axis=1))) for a in (1.,1.012)])
    normal = np.tile(np.column_stack((-47.4*xy, np.ones(len(xy)))), (2,1))
    learn = np.array([0,1,2,3,6,7,8,9]); test = np.array([4,5,10,11])
    assert not np.intersect1d(learn,test).size
    coefficients = np.array([[1.,.1,.2],[1.012,-.1,.3]])
    zref = float(np.median(points[learn,2]))
    recovered,d = physical(normalized(coefficients,zref),zref,False)
    np.testing.assert_allclose(curve_energy(points,normal,recovered,d,SCALE,LAMBDA),
                               curve_energy(points,normal,coefficients,0.,SCALE,LAMBDA),atol=1e-12)
    plane,curve = fit_models(points[learn],normal[learn],coefficients)
    assert curve['status']=='finite_candidate' and abs(curve['d']-23.7)<.05
    assert curve['objective']<1e-7
    cost = curve_energy(points[test],normal[test],np.array(curve['coefficients']),curve['d'],SCALE,LAMBDA)
    labels=curve_labels(cost); swapped=curve_labels(cost[:,::-1])
    assert pair_metrics(np.array([0,0,1,1]),labels)['f1']==1.
    np.testing.assert_array_equal(labels[:,None]==labels[None,:],swapped[:,None]==swapped[None,:])
    # Invalid optimization candidates must remain an explicit failure, not disappear.
    failed=optimize_model(points[:2],normal[:2],[('bad',np.full(6,np.nan))],zref,False)
    assert failed['status']=='nonfinite_failure'
    print('continuous graph parameter/d0/fit-only/heldout/permutation/failure self-check passed',flush=True)


def run(root,out,protocol='random_half'):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    previous=root/'gaussian_member_holdout'
    oldmeta=json.loads((previous/'metadata.json').read_text(encoding='utf-8'))
    assert oldmeta['status']=='complete'
    cached={}
    for line in (previous/'models.jsonl').read_text(encoding='utf-8').splitlines():
        model=json.loads(line)
        if model['case_index'] in (9,10,11) and model['protocol']==protocol and model['scale']==SCALE and model['normal_weight']==LAMBDA and model['curvature'] in (0,50):
            key=(model['case_index'],model['seed_offset'],model['curvature'],model['repeat'])
            assert key not in cached
            cached[key]=model
    repeats=range(5) if protocol=='random_half' else range(1)
    assert len(cached)==12*len(repeats)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];started=time.monotonic()
    with (out/'models.jsonl').open('w',encoding='utf-8') as stream:
        for case in (9,10,11):
            for seed in (1000,2000):
                payloads={}; common=None
                for curvature in (0,20,50):
                    folder=root/f'independent_c{curvature}_s{seed}'
                    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));item=meta['cases'][case]
                    assert meta['status']=='complete' and meta['backend']=='analytic_ray_peak_gaussian_CPU'
                    path=folder/item['file'];digest=sha(path)
                    assert digest==item['npz_sha256'] and any(Path(v['path']).resolve()==path.resolve() and v['sha256']==digest for v in oldmeta['inputs'])
                    inputs.append(dict(path=str(path),sha256=digest))
                    with np.load(path,allow_pickle=False) as z:
                        ids=np.unique(z['gaussian_id']);common=ids if common is None else np.intersect1d(common,ids)
                        if curvature in (0,50):payloads[curvature]={name:z[name].copy() for name in ('means','generated_normal','source_origin')}
                for curvature in (0,50):
                    payload=payloads[curvature];p=payload['means'].astype(float);n=payload['generated_normal'].astype(float)
                    assert len(np.unique(payload['source_origin'][common]))==2 # Explicit oracle K2.
                    for repeat in repeats:
                        model=cached[(case,seed,curvature,repeat)];learn=np.array(model['fit_ids']);test=np.array(model['heldout_ids'])
                        if protocol=='random_half':
                            order=np.random.default_rng(700000+seed*100+case*7+repeat).permutation(len(common))
                            np.testing.assert_array_equal(learn,common[np.sort(order[:len(common)//2])])
                            np.testing.assert_array_equal(test,common[np.sort(order[len(common)//2:])])
                        else:
                            np.testing.assert_array_equal(learn,common[p[common,0]<0])
                            np.testing.assert_array_equal(test,common[p[common,0]>0])
                        assert not np.intersect1d(learn,test).size
                        centers=np.array(model['center_ids']);assert np.isin(centers,learn).all()
                        coeff,valid=curve_candidates(p[centers],n[centers],0.);assert len(valid)==2
                        plane,curve=fit_models(p[learn],n[learn],coeff)
                        row=dict(protocol=protocol,case_index=case,seed_offset=seed,curvature=curvature,repeat=repeat,
                                 scale=SCALE,normal_weight=LAMBDA,oracle_K=2,fit_members=len(learn),heldout_members=len(test),
                                 fit_origins=len(np.unique(payload['source_origin'][learn])),heldout_origins=len(np.unique(payload['source_origin'][test])))
                        detail=dict(**row,fit_ids=learn.tolist(),heldout_ids=test.tolist(),models={})
                        for name,result in (('plane',plane),('curve',curve)):
                            pred=np.full(len(test),-1,int)
                            if result['status']=='finite_candidate':
                                costs=curve_energy(p[test],n[test],np.array(result['coefficients']),result['d'],SCALE,LAMBDA)
                                finite=np.isfinite(costs).all(axis=1);pred[finite]=curve_labels(costs[finite])
                            score=pair_metrics(payload['source_origin'][test],pred)
                            row.update({name+'_status':result['status'],name+'_f1':score['f1'],name+'_fp':score['fp'],name+'_fn':score['fn'],
                                        name+'_unknown':int((pred<0).sum()),name+'_groups':len(set(pred.tolist())-{-1}),
                                        name+'_objective':result.get('objective'),name+'_d':result.get('d'),
                                        name+'_successful_starts':sum(t['success'] for t in result['trials'])})
                            detail['models'][name]=dict(**result,heldout_labels=pred.tolist())
                        row['curve_minus_plane']=row['curve_f1']-row['plane_f1'] if row['curve_f1'] is not None and row['plane_f1'] is not None else None
                        rows.append(row);stream.write(json.dumps(detail)+'\n');stream.flush();csv_write(out/'replicates.csv',rows)
                print('continuous graphs case',case,'seed',seed,'finished',round(time.monotonic()-started,2),flush=True)
    assert len(rows)==12*len(repeats)
    metadata=dict(status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),dependency_sha256={name:sha(Path(__file__).parent/name) for name in ('analyze.py','joint_plane_probe.py')},
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,inputs=inputs,
        cached_input_sha256={name:sha(previous/name) for name in ('metadata.json','models.jsonl')},
        elapsed_seconds=time.monotonic()-started,protocol=protocol,
        evidence='CPU analytic existing case9-11, two seeds, curvature0/50, saved random halves or x<0 fit/x>0 heldout with x=0 excluded, scale.002 lambda1. K2 oracle. '
        'Physical shared graph d is unbounded continuous, initialized from fit arrays only, never construction d/gap/origin or heldout. '
        'Same robust normalized graph residual plus unsigned capped axis angle as finite probe; not exact surface distance or calibrated support. '
        'Two deterministic starts each; L-BFGS-B maxiter200 ftol1e-10 gtol1e-6. Plane6 and curve7 parameters. '
        'Best finite initial/optimized candidate chosen only by fit objective; all solver failures retained. Nonconvex, not globally optimized. '
        'Curve includes selected plane at d0; finite saved fit-only plane only initializes plane control. '
        'Predeclared go/no-go: all six curved case-seed mean gains positive, and aggregate planar mean gain nonnegative; else stop expanding this implementation. '
        'Dependent member splits and only two seeds, not significance or true-scene validation. No training or automatic K.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',choices=('random_half','spatial_left_right'),default='random_half');args=parser.parse_args()
    if args.self_check:self_check()
    else:
        if args.root is None or args.output is None:parser.error('root and output required')
        run(args.root,args.output,args.protocol)
