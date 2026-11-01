"""Frozen synthetic 3D necessary-compatibility and complete-link baseline.

prepare writes separate observation and truth files. predict cannot read truth;
score requires a frozen prediction SHA before opening truth. Distances are metres.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / '05_共同表面支持合成' / '共同平面基线.py'
SEED = 20261101
ANGLE_TOL = 1e-12
DIST_TOL = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    def finite_json(x):
        if isinstance(x, dict):
            return {k: finite_json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [finite_json(v) for v in x]
        if isinstance(x, (float, np.floating)) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return digest(path)


def load(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'SHA mismatch: {path}: {actual} != {expected}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cov(h=.01, sx=.04, sy=None, tilt=0.):
    if sy is None:
        sy = sx
    c, s = math.cos(tilt), math.sin(tilt)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return (rotation @ np.diag([sx*sx, sy*sy, h*h]) @ rotation.T).tolist()


def member(mid, xyz, label, h=.01, sx=.04, opacity=.5, tilt=0.):
    observation = dict(id=mid, center=list(map(float, xyz)),
                       covariance=cov(h, sx, tilt=tilt), opacity=float(opacity))
    return observation, label


def case(name, entries, epsilon, eta_deg=0., truth_depths=None,
         assumption='within_hard_bounds', origin=(0., 0., 0.),
         ray=(0., 0., 1.), interval=(.5, 2.5)):
    obs = dict(name=name, members=[x[0] for x in entries], epsilon=float(epsilon),
               eta_rad=math.radians(eta_deg), origin=list(origin), ray=list(ray),
               interval=list(interval))
    truth = dict(name=name, labels={x[0]['id']: x[1] for x in entries},
                 truth_depths=truth_depths or {}, assumption=assumption)
    return obs, truth


def make_cases():
    rng = np.random.default_rng(SEED)
    noisy = []
    for i, (x, h, sx, op) in enumerate(zip((-.025, .0, .025),
                                           (.006, .012, .02), (.04, .06, .08),
                                           (.2, .55, .9))):
        z = 1. + float(rng.uniform(-.001, .001))
        tilt = float(rng.uniform(-math.radians(.5), math.radians(.5)))
        noisy.append(member(f'N{i}', (x, 0., z), 'surface', h, sx, op, tilt))
    near = [member('A', (-.02, 0., 1.), 'near'),
            member('B', (.02, 0., 1.), 'near')]
    far = [member('C', (-.02, 0., 1.02), 'far'),
           member('D', (.02, 0., 1.02), 'far')]
    bridge = [member('A', (0., 0., 1.), 'first'),
              member('B', (0., 0., 1.005), 'first'),
              member('C', (0., 0., 1.010), 'second')]
    long = [member('A', (0., 0., 1.), 'near', sx=2.),
            member('B', (1., 0., 1.02), 'far', sx=2.)]
    wrong = [member('A', (-.02, 0., 1.), 'same'),
             member('B', (.02, 0., 1.), 'same', tilt=math.radians(15.))]
    duplicate = [member('A', (0., 0., 1.), 'same'),
                 member('B', (0., 0., 1.), 'same')]
    inside = [member('A', (-.02, 0., 1.), 'same'),
              member('B', (.02, 0., 1.006), 'same')]
    outside = [member('A', (-.02, 0., 1.), 'left'),
               member('B', (.02, 0., 1.006 + 5e-12), 'right')]
    a = .37
    c, s = math.cos(a), math.sin(a)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    shift = np.array([.3, -.2, .1])
    transformed = []
    for m, label in near + far:
        transformed.append((dict(m, center=(rotation @ m['center'] + shift).tolist(),
                                 covariance=(rotation @ m['covariance'] @ rotation.T).tolist()), label))
    obs_truth = [
        case('hard_bound_single_varying', noisy, .002, 1., {'surface': 1.}),
        case('density_duplicate_single', noisy +
             [(dict(noisy[0][0], id='N3', opacity=.99), 'surface'),
              (dict(noisy[0][0], id='N4', opacity=.01), 'surface')],
             .002, 1., {'surface': 1.}),
        case('separable_layers', near + far, .002, 0., {'near': 1., 'far': 1.02}),
        case('wide_gate_merges_layers', near + far, .012, 0., {'near': 1., 'far': 1.02}),
        case('bridge_P3', bridge, .003, 0., {'first': 1.0025, 'second': 1.01}),
        case('bridge_permuted_ID', [(dict(m, id={'A':'C','B':'A','C':'B'}[m['id']]), label)
                                    for m, label in bridge], .003, 0.,
             {'first': 1.0025, 'second': 1.01}),
        case('long_baseline_false_edge', long, .001, 2., {'near': 1., 'far': 1.02}),
        case('wrong_normal_outside_assumption', wrong, .002, 2., {'same': 1.},
             assumption='violates_eta'),
        case('duplicate_only', duplicate, .002, 0., {'same': 1.}),
        case('distance_boundary_equal', inside, .003, 0., {'same': 1.003}),
        case('distance_boundary_outside', outside, .003, 0.,
             {'left': 1., 'right': 1.006 + 5e-12}, assumption='boundary_control'),
        case('rigid_separable', transformed, .002, 0.,
             {'near': 1., 'far': 1.02}, origin=shift.tolist(),
             ray=(rotation @ np.array([0., 0., 1.])).tolist()),
    ]
    return obs_truth


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pairs = make_cases()
    observations = dict(seed=SEED, angle_tolerance_rad=ANGLE_TOL,
                        distance_tolerance_m=DIST_TOL, helper_sha256=digest(HELPER),
                        cases=[p[0] for p in pairs])
    obs_hash = save(directory / 'observations.json', observations)
    truth = dict(observations_sha256=obs_hash, cases=[p[1] for p in pairs])
    truth_hash = save(directory / 'truth.json', truth)
    print(json.dumps(dict(observations_sha256=obs_hash, truth_sha256=truth_hash,
                          helper_sha256=digest(HELPER), cases=len(pairs))))


def shortest(m):
    mu = np.asarray(m['center'], dtype=np.float64)
    c = np.asarray(m['covariance'], dtype=np.float64)
    if mu.shape != (3,) or c.shape != (3, 3) or not np.isfinite(mu).all() or not np.isfinite(c).all():
        raise ValueError(f'invalid member {m["id"]}')
    if not np.allclose(c, c.T, rtol=1e-12, atol=1e-14*max(1., float(np.max(np.abs(c))))):
        raise ValueError(f'nonsymmetric covariance {m["id"]}')
    eig, axes = np.linalg.eigh((c+c.T)/2)
    if eig[0] <= 0 or eig[0]/eig[2] <= 1e-12 or 1-eig[0]/eig[1] <= 1e-6:
        raise ValueError(f'invalid short axis {m["id"]}')
    return mu, axes[:, 0]


def partition(ids, edges, first=None):
    groups = [(x,) for x in ids]
    if first is not None:
        x, y = first
        groups = [g for g in groups if g != (x,) and g != (y,)] + [tuple(sorted((x, y)))]
    groups.sort()
    while True:
        chosen = next(((i, j) for i in range(len(groups)) for j in range(i+1, len(groups))
                       if all(tuple(sorted((x, y))) in edges for x in groups[i] for y in groups[j])), None)
        if chosen is None:
            return [list(g) for g in groups]
        i, j = chosen
        groups = ([g for k, g in enumerate(groups) if k not in (i, j)] +
                  [tuple(sorted(groups[i] + groups[j]))])
        groups.sort()


def predict_case(c, helper, atol_a, atol_d):
    members = {m['id']: m for m in c['members']}
    ids = sorted(members)
    if len(set(ids)) != len(c['members']):
        raise ValueError('duplicate IDs')
    axes = {x: shortest(members[x]) for x in ids}
    edges = set()
    pairs = []
    for i, x in enumerate(ids):
        for y in ids[i+1:]:
            mi, ni = axes[x]
            mj, nj = axes[y]
            delta = mj-mi
            # atan2 is numerically stable for coincident normals under a rigid rotation;
            # it computes the same unoriented angle as acos(abs(dot)).
            angle = math.atan2(float(np.linalg.norm(np.cross(ni, nj))),
                               abs(float(ni @ nj)))
            distance = max(abs(float(ni @ delta)), abs(float(nj @ delta)))
            angle_limit = 2*c['eta_rad']
            distance_limit = 2*c['epsilon'] + math.sin(c['eta_rad'])*float(np.linalg.norm(delta))
            ok = angle <= angle_limit + atol_a and distance <= distance_limit + atol_d
            if ok:
                edges.add((x, y))
            # Unoriented-axis sign flip must not change either metric.
            flip_ok = (abs(angle-math.atan2(float(np.linalg.norm(np.cross(-ni, nj))),
                                           abs(float((-ni) @ nj)))) < 1e-14
                       and distance == max(abs(float((-ni) @ delta)), abs(float(nj @ delta))))
            if not flip_ok:
                raise AssertionError('normal sign invariance')
            pairs.append(dict(i=x, j=y, angle_rad=angle, angle_limit_rad=angle_limit,
                              distance_m=distance, distance_limit_m=distance_limit,
                              center_separation_m=float(np.linalg.norm(delta)), compatible=ok))
    p3 = next(((x, b, z) for x in ids for z in ids if x < z and (x, z) not in edges
               for b in ids if b not in (x, z) and tuple(sorted((x, b))) in edges
               and tuple(sorted((b, z))) in edges), None)
    if p3:
        a, b, z = p3
        paths = dict(merge_AB=partition(ids, edges, (a, b)),
                     merge_BC=partition(ids, edges, (b, z)))
        status = 'ORDER_AMBIGUOUS'
    else:
        paths = dict(unique=partition(ids, edges))
        status = 'UNIQUE_CLIQUE'
    readouts = {}
    for path, groups in paths.items():
        rows = []
        for group in groups:
            item = dict(name=f'{c["name"]}:{path}:{",".join(group)}',
                        members=[members[x] for x in group], origin=c['origin'],
                        ray=c['ray'], interval=c['interval'], truth_t=None,
                        allow_single=True, normal_signs=None)
            result, details = helper.solve(item, 'uniform')
            rows.append(dict(ids=group, result=result, details=details,
                             low_support=len(group) == 1 or result['unique_members'] < 2))
        readouts[path] = rows
    return dict(name=c['name'], status=status, p3=list(p3) if p3 else None,
                pairs=pairs, paths=paths, readouts=readouts)


def predict(obs_path, obs_sha, output):
    obs = load(obs_path, obs_sha)
    if digest(HELPER) != obs['helper_sha256']:
        raise ValueError('05 helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('fixed_group_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    results = [predict_case(c, helper, obs['angle_tolerance_rad'],
                            obs['distance_tolerance_m']) for c in obs['cases']]
    pred = dict(observations_sha256=obs_sha, helper_sha256=digest(HELPER), cases=results)
    print(json.dumps(dict(prediction_sha256=save(output, pred), cases=len(results),
                          ambiguous=sum(x['status']=='ORDER_AMBIGUOUS' for x in results))))


def score(pred_path, pred_sha, truth_path, truth_sha, output):
    pred = load(pred_path, pred_sha)
    truth = load(truth_path, truth_sha)
    if pred['observations_sha256'] != truth['observations_sha256']:
        raise ValueError('observation SHA mismatch')
    tmap = {c['name']: c for c in truth['cases']}
    rows = []
    for p in pred['cases']:
        t = tmap[p['name']]
        labels = t['labels']
        for path, groups in p['paths'].items():
            split = sum(labels[a] == labels[b] and not any(a in g and b in g for g in groups)
                        for a in labels for b in labels if a < b)
            mixed = sum(labels[a] != labels[b] and any(a in g and b in g for g in groups)
                        for a in labels for b in labels if a < b)
            for g in p['readouts'][path]:
                group_labels = sorted(set(labels[x] for x in g['ids']))
                result = g['result']
                errors = {label: abs(result['t_hat'] - t['truth_depths'][label])
                          for label in group_labels if result['status'] == 'accepted'
                          and t['truth_depths'].get(label) is not None}
                rows.append(dict(case=p['name'], path=path, ids=g['ids'],
                                 graph_status=p['status'], assumption=t['assumption'],
                                 labels=group_labels, split_same_surface_pairs=split,
                                 mixed_cross_surface_pairs=mixed, low_support=g['low_support'],
                                 status=result['status'], reason=result['reason'],
                                 unique_members=result['unique_members'], t_hat=result['t_hat'],
                                 abs_errors_m=errors))
    output = Path(output)
    score_sha = save(output, dict(prediction_sha256=pred_sha,
                                  truth_sha256=truth_sha, rows=rows))
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ('case', 'path', 'ids', 'graph_status', 'assumption', 'labels',
                  'split_same_surface_pairs', 'mixed_cross_surface_pairs',
                  'low_support', 'status', 'reason', 'unique_members', 't_hat',
                  'abs_errors_m')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})
    print(json.dumps(dict(score_sha256=score_sha,
                          score_csv_sha256=digest(output.with_suffix('.csv')), rows=len(rows))))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--directory', required=True)
    p = sub.add_parser('predict'); p.add_argument('--observations', required=True)
    p.add_argument('--observations-sha256', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('score'); p.add_argument('--prediction', required=True)
    p.add_argument('--prediction-sha256', required=True); p.add_argument('--truth', required=True)
    p.add_argument('--truth-sha256', required=True); p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.directory)
    elif a.command == 'predict': predict(a.observations, a.observations_sha256, a.output)
    else: score(a.prediction, a.prediction_sha256, a.truth, a.truth_sha256, a.output)


if __name__ == '__main__':
    main()
