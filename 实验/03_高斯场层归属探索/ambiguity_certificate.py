"""Construct non-uniqueness witnesses for the current pairwise constraints, CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import group_ray
from compatibility_bound import pairwise_compatibility


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_valid(labels, good, compatible):
    assert np.array_equal(labels >= 0, good)
    for k in set(labels) - {-1}:
        ix = np.flatnonzero(labels == k)
        assert compatible[np.ix_(ix, ix)].all()


def possible_moves(labels, compatible):
    groups = {k: np.flatnonzero(labels == k) for k in sorted(set(labels) - {-1})}
    moves = []
    for i in range(len(labels)):
        source = labels[i]
        if source < 0 or len(groups[source]) < 2:
            continue
        targets = [k for k, ids in groups.items() if k != source and compatible[i, ids].all()]
        if targets:
            moves.append((i, int(min(targets))))
    return moves


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(bounds_root, output, angle=15.):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}')
    bound_files = sorted(bounds_root.glob('*/rays.csv'))
    if not bound_files:
        raise ValueError('No completed per-view bound outputs')
    source = Path(__file__).parent
    grouping_hash, bound_hash = sha(source/'analyze.py'), sha(source/'compatibility_bound.py')
    rows, witnesses, input_hashes = [], [], {}
    for bf in bound_files:
        with bf.open(encoding='utf-8-sig') as f:
            bounds = [r for r in csv.DictReader(f) if float(r['angle_degrees']) == angle]
        if not bounds:
            continue
        path = Path(bounds[0]['input_path'])
        digest = sha(path)
        assert str(path) not in input_hashes, f'Duplicate input {path}'
        input_hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as z:
            for b in bounds:
                assert b['input_path'] == str(path) and b['input_sha256'] == digest
                assert b['grouping_source_sha256'] == grouping_hash and b['source_sha256'] == bound_hash
                ray = int(b['ray'])
                assert not z['qualitative'][ray]
                a, end = map(int, z['offsets'][ray:ray+2])
                d, n, v, w = z['depth'][a:end], z['normal'][a:end], z['normal_valid'][a:end], z['weight'][a:end]
                geometry = b['method'] == 'depth_normal'
                good, compatible = pairwise_compatibility(d, n, v, angle, geometry)
                labels, _ = group_ray(d, n, v, .01, angle, geometry)
                partition_valid(labels, good, compatible)
                lower, upper = int(b['greedy_clique_lower_bound']), int(b['greedy_upper_bound'])
                clique = np.asarray(json.loads(b['clique_member_indices']), dtype=int)
                assert len(clique) == lower and len(set(clique)) == lower
                assert good[clique].all()
                if lower > 1:
                    assert not compatible[np.ix_(clique, clique)][np.triu_indices(lower, 1)].any()
                assert upper == len(set(labels) - {-1}) and lower <= upper
                moves = possible_moves(labels, compatible)
                movable = np.array([i for i, _ in moves], dtype=int)
                positive = np.isfinite(w) & (w > 0)
                total_mass = float(w[positive].sum())
                move_mass = float(w[movable][positive[movable]].sum())
                row = dict(input_dir=path.parent.name, view=path.stem, ray=ray, method=b['method'],
                           angle_degrees=angle, members=len(d), valid_members=int(good.sum()),
                           lower_bound=lower, group_count=upper, minimum_count_certified=lower == upper,
                           movable_members=len(moves), has_move=bool(moves),
                           movable_original_mass=move_mass,
                           movable_original_mass_fraction=move_mass/total_mass if total_mass else None,
                           witness_changed_pairs=None, witness_changed_pair_mass=None)
                if moves:
                    i, target = moves[0]
                    moved = labels.copy()
                    moved[i] = target
                    partition_valid(moved, good, compatible)
                    assert len(set(moved)-{-1}) == upper
                    source_group = int(labels[i])
                    affected = ((labels == source_group) | (labels == target)) & (np.arange(len(d)) != i)
                    changed_pairs = int(affected.sum())
                    assert changed_pairs > 0
                    row.update(witness_changed_pairs=changed_pairs,
                               witness_changed_pair_mass=float(w[i]*w[affected].sum()))
                    witnesses.append(dict(**row, gaussian_id=int(z['gaussian_id'][a+i]),
                                          moved_member_index=i, source_group=source_group, target_group=target,
                                          original_labels=labels.tolist(), alternate_labels=moved.tolist(),
                                          lower_bound_clique=clique.tolist(), input_path=str(path), input_sha256=digest))
                rows.append(row)
    if not rows:
        raise ValueError('No matching bound rows')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'rays.csv', rows)
    summary = []
    for method in ('depth', 'depth_normal'):
        for scope in ('all', 'certified_minimum'):
            part = [r for r in rows if r['method'] == method and
                    (scope == 'all' or r['minimum_count_certified'])]
            summary.append(dict(method=method, scope=scope, rays=len(part),
                                rays_with_move=sum(r['has_move'] for r in part),
                                fraction_with_move=float(np.mean([r['has_move'] for r in part])) if part else None,
                                mean_movable_mass_fraction=float(np.mean([r['movable_original_mass_fraction']
                                    for r in part if r['movable_original_mass_fraction'] is not None])) if part else None))
    write_csv(output/'summary.csv', summary)
    (output/'witnesses.json').write_text(json.dumps(witnesses, ensure_ascii=False), encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps(dict(input_sha256=input_hashes,
        bounds_sha256={str(p):sha(p) for p in bound_files}, angle_degrees=angle, eta=.01,
        source_sha256={name:sha(source/name) for name in
                      ('ambiguity_certificate.py','analyze.py','compatibility_bound.py')},
        interpretation='Non-uniqueness of representation partitions under fixed constraints, not physical membership. '
        'Movable members are individually movable from the original partition; they need not move jointly. '
        'No single-member move does not prove uniqueness. Minimum refers only to valid members and current constraints.'),
        indent=2), encoding='utf-8')


def self_check():
    d = np.array([1., 1.008, 1.016])
    n = np.tile([0.,0.,-1.], (3,1)); v = np.ones(3, bool)
    good, compatible = pairwise_compatibility(d,n,v,15.,False)
    labels, _ = group_ray(d,n,v,.01,15.,False)
    partition_valid(labels, good, compatible)
    assert possible_moves(labels, compatible) == [(1,1)]
    assert len(set(labels)) == 2 and not compatible[0,2]
    assert not possible_moves(np.array([0,1,2]), np.eye(3,dtype=bool))
    print('ambiguity certificate self-check passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bounds-root'); p.add_argument('--output')
    p.add_argument('--angle', type=float, default=15.)
    p.add_argument('--self-check', action='store_true'); args=p.parse_args()
    if args.self_check:
        self_check()
    elif args.bounds_root and args.output:
        run(Path(args.bounds_root), Path(args.output), args.angle)
    else:
        p.error('Use --bounds-root and --output')
