"""CPU-only greedy clique lower bound for complete-pairwise ray grouping."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import group_ray


ETA = .01
ANGLES = (15., 30., 60.)
TOL = 1e-12


def pairwise_compatibility(depth, normal, valid, angle, geometry):
    """Mirror group_ray's validity and pair condition, including its tolerance."""
    depth = np.asarray(depth, dtype=float)
    normal = np.asarray(normal, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    length = np.linalg.norm(normal, axis=1)
    good_normal = valid & np.isfinite(normal).all(axis=1) & (length > 1e-8)
    good = np.isfinite(depth) & (depth > 0)
    if geometry:
        good &= good_normal
    unit = np.zeros_like(normal)
    np.divide(normal, length[:, None], out=unit, where=(length > 1e-8)[:, None])
    log_depth = np.zeros(len(depth), dtype=float)
    log_depth[good] = np.log(depth[good])
    distance = np.abs(log_depth[:, None] - log_depth[None, :]) / np.log1p(ETA)
    if geometry:
        angular = np.arccos(np.clip(unit @ unit.T, -1., 1.)) / np.deg2rad(angle)
        distance = np.maximum(distance, angular)
    compatible = (distance <= 1. + TOL) & good[:, None] & good[None, :]
    return good, compatible


def greedy_clique(compatible, valid_indices):
    """Deterministic multi-start greedy clique; return size and original indices."""
    best = []
    valid_indices = np.sort(np.asarray(valid_indices, dtype=int))
    for seed in valid_indices:
        clique = [int(seed)]
        candidates = valid_indices[(valid_indices != seed) & ~compatible[seed, valid_indices]]
        while len(candidates):
            # Degree is measured in the current induced candidate subgraph.
            incompatibility = ~compatible[np.ix_(candidates, candidates)]
            degree = incompatibility.sum(axis=1)
            # candidates is sorted, so argmax resolves ties by original index.
            chosen = int(candidates[int(np.argmax(degree))])
            clique.append(chosen)
            candidates = candidates[(candidates != chosen) & ~compatible[chosen, candidates]]
        if (len(clique) > len(best)
                or (len(clique) == len(best) and tuple(sorted(clique)) < tuple(sorted(best)))):
            best = clique
    assert all(not compatible[a, b] for p, a in enumerate(best) for b in best[p + 1:])
    return len(best), sorted(best)


def compatible_partition_count(depth, normal, valid, angle, geometry):
    labels, _ = group_ray(depth, normal, valid, ETA, angle, geometry)
    return len(set(labels.tolist()) - {-1})


def self_check():
    n = 4
    normal = np.tile([0., 0., -1.], (n, 1))
    valid = np.ones(n, dtype=bool)

    def evaluate(depth, nm=normal, v=valid, geometry=False, angle=15.):
        good, compatible = pairwise_compatibility(depth, nm, v, angle, geometry)
        count, members = greedy_clique(compatible, np.flatnonzero(good).tolist())
        upper = compatible_partition_count(depth, nm, v, angle, geometry)
        assert count <= upper
        return count, members, upper, good, compatible

    lb, _, ub, _, _ = evaluate([1., 1.001, 1.002, 1.003])
    assert (lb, ub) == (1, 1)  # fully compatible
    lb, members, ub, _, _ = evaluate([1., 1.1, 1.21])
    assert lb == ub == 3 and members == [0, 1, 2]  # fully incompatible
    lb, _, _, _, _ = evaluate([1., 1.009, 1.018])
    assert lb == 2  # incompatible endpoints, compatible bridge
    same_depth_normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    d = np.ones(3)
    depth_lb, _, depth_ub, _, _ = evaluate(d, same_depth_normals, np.ones(3, bool), False)
    geom_lb, _, geom_ub, _, _ = evaluate(d, same_depth_normals, np.ones(3, bool), True, 15.)
    assert depth_lb == depth_ub == 1 and geom_lb == geom_ub == 3
    assert geom_lb > depth_ub and geom_lb <= geom_ub
    lb, _, ub, good, _ = evaluate([])
    assert lb == ub == 0 and not good.any()
    invalid_normal = np.array([[0., 0., -1.], [np.nan, 0., 0.], [0., 0., -1.]])
    lb, _, ub, good, _ = evaluate([1., 1.001, 1.002], invalid_normal,
                                  np.array([True, False, True]), geometry=True)
    assert lb == ub == 1 and good.tolist() == [True, False, True]
    lb, _, ub, good, _ = evaluate([np.nan, 0.], normal[:2], np.ones(2, bool), geometry=False)
    assert lb == ub == 0 and not good.any()

    # Pair compatibility agrees with analyze.group_ray for every tested pair,
    # including the normal validity gate and 1e-12 threshold tolerance.
    cases = [
        (np.array([1., 1.01]), normal[:2], valid[:2], False, 15.),
        (np.array([1., 1.01]), normal[:2], valid[:2], True, 15.),
        (np.array([1., 1.001]), invalid_normal[:2], np.array([True, False]), True, 30.),
    ]
    for d, nm, v, geometry, angle in cases:
        good, compatible = pairwise_compatibility(d, nm, v, angle, geometry)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                labels, _ = group_ray(d[[i, j]], nm[[i, j]], v[[i, j]], ETA, angle, geometry)
                expected = bool(good[i] and good[j] and labels[0] == labels[1])
                assert bool(compatible[i, j]) == expected
    print('self-check passed')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_inputs(inputs):
    files = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob('view_*.npz')))
        elif path.is_file() and path.suffix.lower() == '.npz':
            files.append(path)
        else:
            raise FileNotFoundError(f'Input must be an existing directory or NPZ: {path}')
    # Preserve caller order while preventing accidental duplicate analysis.
    unique = list(dict.fromkeys(p.resolve() for p in files))
    if not unique:
        raise ValueError('No view_*.npz files found in the input directories')
    return unique


def analyze(inputs, output):
    files = collect_inputs(inputs)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output must be a new or empty directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(__file__)
    grouping_source = Path(__file__).with_name('analyze.py')
    grouping_source_hash = sha256(grouping_source)
    rows, view_rows, witnesses = [], [], []
    for path in files:
        input_hash = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            offsets, depth = z['offsets'], z['depth']
            normal, valid = z['normal'], z['normal_valid']
            qualitative = z['qualitative']
            gaussian_id = z['gaussian_id']
            nr = len(offsets) - 1
            if (offsets.ndim != 1 or nr < 0 or offsets[0] != 0
                    or offsets[-1] != len(depth) or np.any(np.diff(offsets) < 0)
                    or normal.shape != (len(depth), 3) or len(valid) != len(depth)
                    or len(gaussian_id) != len(depth) or len(z['pixel_xy']) != nr
                    or len(qualitative) != nr):
                raise ValueError(f'Invalid ragged NPZ schema: {path}')
            for ray in range(nr):
                if bool(qualitative[ray]):
                    continue
                start, end = int(offsets[ray]), int(offsets[ray + 1])
                d, nm, v = depth[start:end], normal[start:end], valid[start:end]
                default_depth_groups = compatible_partition_count(d, nm, v, 15., False)
                for method, geometry in (('depth', False), ('depth_normal', True)):
                    for angle in ANGLES:
                        good, compatible = pairwise_compatibility(d, nm, v, angle, geometry)
                        lower, clique = greedy_clique(compatible, np.flatnonzero(good).tolist())
                        # A group_ray partition is a constructive upper bound only
                        # for the same pair constraints. The depth-only count is
                        # retained as a baseline reference for depth_normal rows.
                        upper = compatible_partition_count(d, nm, v, angle, geometry)
                        assert lower <= upper, (path, ray, method, angle, lower, upper)
                        ratio = lower / upper if upper else None
                        row = dict(file=path.name, input_path=str(path), input_sha256=input_hash,
                                   source_sha256=source_hash, grouping_source_sha256=grouping_source_hash,
                                   ray=ray,
                                   x=int(z['pixel_xy'][ray, 0]), y=int(z['pixel_xy'][ray, 1]),
                                   method=method, eta=ETA,
                                   angle_degrees=angle,
                                   greedy_clique_lower_bound=lower,
                                   greedy_upper_bound=upper,
                                   default_depth_greedy_reference=default_depth_groups,
                                   lower_bound_over_upper_bound=ratio,
                                   unknown_members=int((~good).sum()),
                                   valid_members=int(good.sum()),
                                   clique_member_indices=json.dumps(clique, separators=(',', ':')))
                        rows.append(row)
                        witnesses.append(dict(**row,
                                              clique_gaussian_ids=[int(gaussian_id[start + i]) for i in clique],
                                              statement='Greedy clique lower bound; not an exact minimum group count'))
                        view_rows.append(row)
    write_csv(output / 'rays.csv', rows)
    summaries = []
    keys = ('greedy_clique_lower_bound', 'greedy_upper_bound',
            'lower_bound_over_upper_bound', 'unknown_members')
    for input_path in dict.fromkeys(r['input_path'] for r in rows):
        file_name = Path(input_path).name
        for method in ('depth', 'depth_normal'):
            for angle in ANGLES:
                subset = [r for r in rows if r['input_path'] == input_path
                          and r['method'] == method and r['angle_degrees'] == angle]
                summaries.append(dict(file=file_name, input_path=input_path, method=method,
                                      angle_degrees=angle,
                                      rays=len(subset),
                                      **{f'mean_{k}': (float(np.mean([r[k] for r in subset
                                                                       if r[k] is not None]))
                                                       if any(r[k] is not None for r in subset) else None)
                                         for k in keys}))
    write_csv(output / 'views.csv', summaries)
    (output / 'witnesses.json').write_text(json.dumps(witnesses, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(inputs=[dict(path=str(p), sha256=sha256(p)) for p in files],
                    source_path=str(Path(__file__).resolve()), source_sha256=source_hash,
                    grouping_source_path=str(grouping_source.resolve()),
                    grouping_source_sha256=grouping_source_hash,
                    eta=ETA, angles_degrees=list(ANGLES),
                    upper_bound='group_ray count under the same method pair constraints; depth-only count is also retained as a reference',
                    lower_bound='deterministic multi-start greedy clique in the pairwise incompatibility graph',
                    limitation='Greedy clique lower bound is not an exact maximum clique and is not the exact minimum group count')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', help='NPZ or directory; repeat for multiple inputs')
    parser.add_argument('--output', help='New or empty output directory')
    parser.add_argument('--self-check', action='store_true', help='Run small CPU mechanism checks and exit')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.input or not args.output:
        parser.error('normal analysis requires at least one --input and an --output')
    analyze(args.input, args.output)


if __name__ == '__main__':
    main()
