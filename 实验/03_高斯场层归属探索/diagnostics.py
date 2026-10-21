"""CPU-only insertion-order diagnostics for sampled ragged ray NPZ files."""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from analyze import group_ray


ORDERS = ('depth', 'reverse_depth', 'acceptance')


def pair_scores(a, b):
    """Pairwise coassignment scores on members determined by both partitions."""
    a, b = np.asarray(a), np.asarray(b)
    known = (a >= 0) & (b >= 0)
    a, b = a[known], b[known]
    upper = np.triu(np.ones((len(a), len(a)), dtype=bool), 1)
    ca, cb = a[:, None] == a[None, :], b[:, None] == b[None, :]
    tp, fp, fn = int((upper & ca & cb).sum()), int((upper & ~ca & cb).sum()), int((upper & ca & ~cb).sum())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(pair_precision=p, pair_recall=r, pair_f1=f,
                jointly_determined_members=len(a))


def contribution_stats(weights, labels):
    w, lab = np.asarray(weights, float), np.asarray(labels)
    total = float(np.nansum(w[np.isfinite(w) & (w > 0)]))
    known = np.flatnonzero(lab >= 0)
    group_ids, counts = np.unique(lab[known], return_counts=True)
    singleton = set(group_ids[counts == 1].tolist())
    singleton_mask = np.isin(lab, list(singleton))
    masses = [float(np.sum(w[lab == k])) for k in group_ids]
    largest = max(masses, default=0.)
    ordered = np.sort(w[np.isfinite(w) & (w > 0)])[::-1]
    n90 = int(np.searchsorted(np.cumsum(ordered), .9 * total) + 1) if total > 0 and len(ordered) else None
    return dict(max_group_contribution_fraction=largest / total if total > 0 else None,
                candidates_for_90pct_contribution=n90,
                singleton_group_fraction=float(len(singleton) / len(group_ids)) if len(group_ids) else None,
                singleton_member_fraction=float(np.mean(singleton_mask[known])) if len(known) else None,
                singleton_contribution_fraction=float(np.sum(w[singleton_mask]) / total) if total > 0 else None,
                undetermined_member_fraction=float(np.mean(lab < 0)) if len(lab) else None,
                undetermined_contribution_fraction=float(np.sum(w[lab < 0]) / total) if total > 0 else None)


def angles(normals, ids, weights, validity=None):
    n = np.asarray(normals, float)
    lengths = np.linalg.norm(n, axis=1)
    good = np.isfinite(n).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-8)
    if validity is not None:
        good &= np.asarray(validity, dtype=bool)
    ix = np.flatnonzero(good)
    result = dict(normal_valid_coverage=float(len(ix) / len(n)) if len(n) else None,
                  normal_valid_members=len(ix))
    if len(ix) < 2:
        for prefix in ('oriented', 'unoriented'):
            for q in (0, 25, 50, 75, 100):
                result[f'{prefix}_angle_p{q}'] = None
        return result
    unit = n[ix] / lengths[ix, None]
    dot = np.clip(unit @ unit.T, -1, 1)
    iu, ju = np.triu_indices(len(ix), 1)
    oriented = np.degrees(np.arccos(dot[iu, ju]))
    unoriented = np.degrees(np.arccos(np.abs(dot[iu, ju])))
    for prefix, vals in (('oriented', oriented), ('unoriented', unoriented)):
        for q, v in zip((0, 25, 50, 75, 100), np.percentile(vals, (0, 25, 50, 75, 100))):
            result[f'{prefix}_angle_p{q}'] = float(v)
    return result


def weighted_90_subset(weights):
    w = np.asarray(weights, float)
    positive = np.isfinite(w) & (w > 0)
    if not positive.any() or w[positive].sum() <= 0:
        return np.zeros(len(w), dtype=bool)
    ix = np.flatnonzero(positive)
    ix = ix[np.argsort(-w[ix], kind='stable')]
    take = np.searchsorted(np.cumsum(w[ix]), .9 * w[ix].sum(), side='left') + 1
    mask = np.zeros(len(w), dtype=bool)
    mask[ix[:take]] = True
    return mask


def describe_group_normals(normals, weights, labels, validity):
    lengths = np.linalg.norm(normals, axis=1)
    valid_ids = np.flatnonzero(np.asarray(validity, bool) & np.isfinite(normals).all(axis=1)
                               & np.isfinite(lengths) & (lengths > 1e-8))
    groups = [k for k in np.unique(labels) if k >= 0]
    if not groups:
        return dict(largest_group_normal_member_coverage=None, largest_group_normal_90_coverage=None)
    mass = {k: float(np.sum(weights[labels == k])) for k in groups}
    k = max(groups, key=lambda x: (mass[x], -x))
    members = labels == k
    out = dict(largest_group_id=int(k), largest_group_normal_member_coverage=
               float(np.sum(members & np.isin(np.arange(len(labels)), valid_ids)) / np.sum(members)),
               largest_group_contribution_mass=mass[k])
    out.update({f'largest_group_{name}': value for name, value in angles(
        normals[members], np.flatnonzero(members), weights[members], np.asarray(validity)[members]).items()})
    subset = weighted_90_subset(weights[members])
    out['largest_group_90pct_subset_members'] = int(subset.sum())
    out.update({f'largest_group_90_{name}': value for name, value in angles(normals[members][subset],
                np.flatnonzero(members)[subset], weights[members][subset], np.asarray(validity)[members][subset]).items()})
    return out


def expand_inputs(inputs):
    found = []
    for item in inputs:
        p = Path(item)
        found.extend(sorted(p.glob('view_*.npz')) if p.is_dir() else [Path(x) for x in sorted(glob.glob(item))])
    return sorted(set(p.resolve() for p in found if p.suffix == '.npz'))


def run(paths, output, eta=.01, angle=15.):
    output.mkdir(parents=True, exist_ok=True)
    ray_rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            offsets, depth = z['offsets'], z['depth']
            weights, normals, valid = z['weight'], z['normal'], z['normal_valid']
            qual, gids = z['qualitative'], z['gaussian_id']
            nr = len(offsets) - 1
            if offsets[0] != 0 or offsets[-1] != len(depth) or np.any(np.diff(offsets) < 0):
                raise ValueError(f'Invalid ragged offsets: {path}')
            if any(len(x) != len(depth) for x in (weights, normals, valid, gids)) or len(qual) != nr:
                raise ValueError(f'Invalid ragged schema: {path}')
            for ray in range(nr):
                if bool(qual[ray]):
                    continue
                sl = slice(int(offsets[ray]), int(offsets[ray + 1]))
                d, w, n, v = depth[sl], weights[sl], normals[sl], valid[sl]
                if len(d) == 0:
                    continue
                orders = {'depth': np.argsort(d, kind='stable'),
                          'reverse_depth': np.argsort(-d, kind='stable'),
                          'acceptance': np.arange(len(d))}
                partitions = {}
                for method, geometry in (('depth', False), ('depth_normal', True)):
                    for order_name, order in orders.items():
                        labels, flags = group_ray(d, n, v, eta, angle, geometry, order)
                        partitions[method, order_name] = labels
                        row = dict(scan=path.parent.name, view=path.stem, ray=ray, eta=eta, angle=angle,
                                   method=method, order=order_name, members=len(d),
                                   gaussian_ids_json=json.dumps(gids[sl].tolist(), separators=(',', ':')),
                                   groups_json=json.dumps(labels.tolist(), separators=(',', ':')),
                                   compatible_cross_group_pairs=flags['compatible_cross_group_pairs'],
                                   ambiguous_insertions=flags['ambiguous_insertions'], **contribution_stats(w, labels))
                        if method == 'depth' and order_name == 'depth':
                            row.update(describe_group_normals(n, w, labels, v))
                        ray_rows.append(row)
                for method in ('depth', 'depth_normal'):
                    base = partitions[method, 'depth']
                    for order_name in ('reverse_depth', 'acceptance'):
                        ix = next(i for i in range(len(ray_rows)-1, -1, -1)
                                  if ray_rows[i]['scan'] == path.parent.name and ray_rows[i]['view'] == path.stem
                                  and ray_rows[i]['ray'] == ray and ray_rows[i]['method'] == method and ray_rows[i]['order'] == order_name)
                        ray_rows[ix].update(pair_scores(base, partitions[method, order_name]))
    write_csv(output / 'rays.csv', ray_rows)
    keys = ['max_group_contribution_fraction', 'candidates_for_90pct_contribution',
            'singleton_group_fraction', 'singleton_contribution_fraction',
            'undetermined_member_fraction', 'undetermined_contribution_fraction',
            'compatible_cross_group_pairs', 'pair_precision', 'pair_recall', 'pair_f1']
    summaries = []
    for key in sorted(set((r['scan'], r['view'], r['method'], r['order']) for r in ray_rows)):
        subset = [r for r in ray_rows if (r['scan'], r['view'], r['method'], r['order']) == key]
        summaries.append(dict(scan=key[0], view=key[1], method=key[2], order=key[3], rays=len(subset),
                              **{f'mean_{m}': float(np.mean([r[m] for r in subset if r.get(m) is not None]))
                                 if any(r.get(m) is not None for r in subset) else None for m in keys}))
    write_csv(output / 'summary.csv', summaries)
    payload = dict(inputs=[str(p) for p in paths], eta=eta, angle_degrees=angle,
                   interpretation='Algorithmic order dependence only; no order is a truth label. Original contribution is descriptive, not member support or surface probability.',
                   qualitative_rows_excluded=True, summaries=summaries)
    (output / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, help='NPZ, directory, or glob; repeatable')
    parser.add_argument('--output', required=True, help='New or empty diagnostic output directory')
    parser.add_argument('--eta', type=float, default=.01)
    parser.add_argument('--angle', type=float, default=15.)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    result = run(expand_inputs(args.input), out, args.eta, args.angle)
    print(json.dumps(result, indent=2))
