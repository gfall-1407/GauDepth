"""CPU reference grouping and mechanism tests; no GPU imports or scene writes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def group_ray(depth, normal, valid, eta=.01, angle=15., geometry=True):
    """Greedy complete-link reference, NOT globally optimal/hierarchical clustering.

    Stable increasing representative depth (ties: original acceptance index).
    A member may join only if ALL existing members are compatible. Among eligible
    groups minimize worst normalized pair dissimilarity, then creation index.
    Invalid depth / geometry stays -1 (undetermined), never silently discarded.
    """
    if eta <= 0 or not 0 < angle <= 180:
        raise ValueError('Require eta > 0 and angle in (0,180]')
    depth = np.asarray(depth, dtype=float)
    normal = np.asarray(normal, dtype=float)
    n = len(depth)
    good = np.isfinite(depth) & (depth > 0)
    length = np.linalg.norm(normal, axis=1)
    good_normal = np.asarray(valid, dtype=bool) & np.isfinite(normal).all(axis=1) & (length > 1e-8)
    if geometry:
        good &= good_normal
    unit = np.zeros_like(normal)
    np.divide(normal, length[:, None], out=unit, where=(length > 1e-8)[:, None])
    # Log ratio is symmetric and invariant to global metric-unit rescaling.
    logd = np.zeros(n)
    logd[good] = np.log(depth[good])
    distance = np.abs(logd[:, None] - logd[None, :]) / np.log1p(eta)
    if geometry:
        angular = np.arccos(np.clip(unit @ unit.T, -1, 1)) / np.deg2rad(angle)
        distance = np.maximum(distance, angular)
    compatible = (distance <= 1 + 1e-12) & good[:, None] & good[None, :]
    labels = np.full(n, -1, dtype=np.int32)
    groups, ambiguous_insertions = [], 0
    for i in np.argsort(depth, kind='stable'):
        if not good[i]:
            continue
        candidates = [(float(distance[i, members].max()), k)
                      for k, members in enumerate(groups) if compatible[i, members].all()]
        ambiguous_insertions += len(candidates) > 1
        if candidates:
            _, k = min(candidates)
            groups[k].append(int(i))
        else:
            k = len(groups)
            groups.append([int(i)])
        labels[i] = k
    # Every emitted group must obey the advertised full pairwise constraint.
    assert all(compatible[np.ix_(g, g)].all() for g in groups)
    cross = np.triu(compatible & (labels[:, None] != labels[None, :]), 1)
    return labels, dict(ambiguous_insertions=int(ambiguous_insertions),
                        compatible_cross_group_pairs=int(cross.sum()),
                        undetermined=int((labels < 0).sum()))


def pair_metrics(truth, pred):
    """Unknown predictions count as separate items, including unknown-unknown pairs."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    upper = np.triu(np.ones((len(truth), len(truth)), dtype=bool), 1)
    actual = truth[:, None] == truth[None, :]
    predicted = (pred[:, None] == pred[None, :]) & (pred[:, None] >= 0) & (pred[None, :] >= 0)
    tp = int((upper & actual & predicted).sum())
    fp = int((upper & ~actual & predicted).sum())
    fn = int((upper & actual & ~predicted).sum())
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def describe(depth, weight, labels):
    groups = []
    order = np.argsort(depth, kind='stable')
    total = float(np.sum(weight))
    for k in sorted(set(labels) - {-1}):
        ix = np.flatnonzero(labels == k)
        groups.append(dict(group=int(k), members=ix.tolist(), count=len(ix),
                           depth_min=float(np.min(depth[ix])), depth_max=float(np.max(depth[ix])),
                           original_contribution_mass=float(np.sum(weight[ix])),
                           original_contribution_fraction=float(np.sum(weight[ix]) / total) if total > 0 else None))
    overlaps, interleaved = 0, 0
    for a, ga in enumerate(groups):
        for gb in groups[a + 1:]:
            overlaps += max(ga['depth_min'], gb['depth_min']) <= min(ga['depth_max'], gb['depth_max'])
            sequence = labels[order]
            sequence = sequence[(sequence == ga['group']) | (sequence == gb['group'])]
            # A-B-A or B-A-B in sorted representative depth, not necessarily acceptance order.
            runs = 1 + int((sequence[1:] != sequence[:-1]).sum()) if len(sequence) else 0
            interleaved += runs >= 3
    return groups, dict(groups=len(groups), overlapping_group_pairs=int(overlaps),
                        interleaved_group_pairs=int(interleaved), members=len(depth),
                        contribution_mass=total,
                        undetermined_contribution_mass=float(weight[labels < 0].sum()))


def synthetic(output):
    """Fixed labelled generating assumptions, NOT evidence from real scenes."""
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        'single_surface': ([1., 1.002, 1.004, 1.006], [0, 0, 0, 0], [0, 0, 0, 0]),
        'separated_surfaces': ([1., 1.003, 1.04, 1.043], [0, 0, 1, 1], [0, 0, 0, 0]),
        'depth_overlap_direction_separable': ([1., 1.002, 1.004, 1.006], [0, 1, 0, 1], [0, 1, 0, 1]),
        'depth_direction_indistinguishable': ([1., 1.002, 1.004, 1.006], [0, 1, 0, 1], [0, 0, 0, 0]),
        'weak_true_member': ([1., 1.002, 1.004, 1.04, 1.043], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]),
        'bridge': ([1., 1.008, 1.016], [0, 0, 1], [0, 0, 0]),
    }
    rows = []
    for name, (d, truth, direction) in cases.items():
        depth = np.asarray(d)
        normals = np.asarray([[0., 0., -1.] if x == 0 else [1., 0., 0.] for x in direction])
        weight = np.full(len(d), .1)
        if name == 'weak_true_member':
            weight[1] = 1e-7  # Original tiny w could come from a late small T.
        for geometry in [False, True]:
            labels, ambiguity = group_ray(depth, normals, np.ones(len(d), bool), geometry=geometry)
            groups, stats = describe(depth, weight, labels)
            metrics = pair_metrics(truth, labels)
            if name in ['single_surface', 'separated_surfaces', 'weak_true_member', 'bridge']:
                assert metrics['fp'] == metrics['fn'] == 0, (name, geometry, labels)
            if name == 'depth_overlap_direction_separable':
                assert metrics['f1'] == 1. if geometry else metrics['fp'] > 0
                if geometry:
                    assert stats['interleaved_group_pairs'] == 1
            if name == 'depth_direction_indistinguishable':
                assert metrics['fp'] > 0  # Explicit expected identifiability failure.
            if name == 'weak_true_member':
                assert labels[1] == labels[0] and sum(g['count'] for g in groups) == len(d)
            if name == 'bridge':
                assert labels[0] != labels[-1]  # No single-link weak bridge.
            rows.append(dict(case=name, method='depth_normal' if geometry else 'depth',
                             labels=labels.tolist(), truth=truth, weight=weight.tolist(),
                             **metrics, **ambiguity, **stats))
    # Invalid geometry is retained as unknown and creates false splits in pair recall.
    d = np.array([1., 1.001, 1.002])
    nm = np.array([[0, 0, -1], [np.nan, 0, 0], [0, 0, -1.]])
    labels, stats = group_ray(d, nm, np.array([True, False, True]))
    assert labels.tolist() == [0, -1, 0] and stats['undetermined'] == 1
    assert pair_metrics([0, 0, 0], labels)['recall'] == 1 / 3
    # All-singleton grouping must NOT receive perfect F1.
    assert pair_metrics([0, 0, 1, 1], [0, 1, 2, 3])['f1'] == 0
    # Global metric scale leaves grouping invariant; repeat is deterministic.
    base = group_ray(d, np.tile([0., 0., -1.], (3, 1)), np.ones(3, bool))[0]
    scaled = group_ray(d * 1000, np.tile([0., 0., -1.], (3, 1)), np.ones(3, bool))[0]
    np.testing.assert_array_equal(base, scaled)
    payload = dict(evidence='Labelled synthetic generating assumptions only; no real-scene evidence',
                   eta=.01, angle=15, mechanism_checks='passed', cases=rows)
    (output / 'synthetic.json').write_text(json.dumps(payload, indent=2))
    write_csv(output / 'synthetic.csv', [{k: v for k, v in r.items() if k not in ['labels', 'truth', 'weight']} for r in rows])
    print(json.dumps(payload, indent=2))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    source = Path(args.input)
    files = sorted(source.glob('view_*.npz')) if source.is_dir() else [source]
    if not files:
        raise ValueError('No sampled NPZ files found')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Analysis output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    settings = [(args.eta, args.angle)]
    if args.grid:
        settings = [(e, a) for e in [.005, .01, .02] for a in [10., 20.]]
    rows, details, manifests = [], [], []
    figures = 0
    for path in files:
        manifests.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with np.load(path, allow_pickle=False) as z:
            off, depth, weight = z['offsets'], z['depth'], z['weight']
            normals, valid, qualitative = z['normal'], z['normal_valid'], z['qualitative']
            nr = len(off) - 1
            if (off[0] != 0 or off[-1] != len(depth) or np.any(np.diff(off) < 0)
                    or len(normals) != len(depth) or len(weight) != len(depth) or len(valid) != len(depth)
                    or len(qualitative) != nr):
                raise ValueError(f'Invalid ragged schema: {path}')
            for r in range(nr):
                sl = slice(off[r], off[r + 1])
                for eta, angle in settings:
                    labels_for_plot = []
                    for geometry in [False, True]:
                        labels, ambiguity = group_ray(depth[sl], normals[sl], valid[sl], eta, angle, geometry)
                        groups, stats = describe(depth[sl], weight[sl], labels)
                        row = dict(file=path.name, ray=r, x=int(z['pixel_xy'][r, 0]),
                                   y=int(z['pixel_xy'][r, 1]), qualitative=bool(qualitative[r]),
                                   eta=eta, angle=angle, method='depth_normal' if geometry else 'depth',
                                   **stats, **ambiguity)
                        rows.append(row)
                        details.append(dict(**row, group_details=groups, labels=labels.tolist(),
                                            gaussian_id=z['gaussian_id'][sl].tolist()))
                        labels_for_plot.append(labels)
                    if figures < args.plots and (eta, angle) == settings[0]:
                        plot_ray(output / f'ray_{figures:02d}.png', depth[sl], weight[sl],
                                 normals[sl], labels_for_plot, f'{path.stem} ray {r}; qualitative={bool(qualitative[r])}')
                        figures += 1
    write_csv(output / 'rays.csv', rows)
    # Exclude additional hand-picked rays from every statistical summary.
    summaries = []
    for eta, angle in settings:
        for method in ['depth', 'depth_normal']:
            subset = [r for r in rows if not r['qualitative'] and r['eta'] == eta
                      and r['angle'] == angle and r['method'] == method]
            keys = ['groups', 'members', 'undetermined', 'contribution_mass',
                    'undetermined_contribution_mass', 'overlapping_group_pairs',
                    'interleaved_group_pairs', 'ambiguous_insertions', 'compatible_cross_group_pairs']
            summaries.append(dict(eta=eta, angle=angle, method=method, random_rays=len(subset),
                                  **{f'mean_{k}': float(np.mean([r[k] for r in subset])) if subset else None for k in keys}))
    write_csv(output / 'summary.csv', summaries)
    (output / 'groups.json').write_text(json.dumps(details, indent=2))
    payload = dict(inputs=manifests, settings=settings, algorithm='stable-depth greedy complete-link',
                   uncertainty='unknown members and locally compatible cross-group pairs; not calibrated posterior',
                   evidence='Real-scene exploratory candidate groups; no independent labels, no accuracy claim',
                   summaries=summaries, qualitative_rows_excluded=True)
    (output / 'summary.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def plot_ray(path, depth, weight, normal, labels, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    for ax, lab, name in zip(axes[:2], labels, ['depth', 'depth + normal']):
        for k in sorted(set(lab)):
            ix = np.flatnonzero(lab == k)
            ax.scatter(depth[ix], ix, s=20, label='unknown' if k < 0 else f'group {k}')
        ax.set(ylabel='acceptance index', title=name)
        if len(set(lab)) <= 8:
            ax.legend(fontsize=6, loc='best')
    axes[2].scatter(depth, weight, s=15)
    axes[2].set(xlabel='camera-z representative depth', ylabel='original T * alpha')
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--input', help='Sample NPZ or directory; read only')
    p.add_argument('--output', required=True)
    p.add_argument('--eta', type=float, default=.01)
    p.add_argument('--angle', type=float, default=15.)
    p.add_argument('--grid', action='store_true', help='Explicit optional 3 x 2 sensitivity grid')
    p.add_argument('--plots', type=int, default=3)
    a = p.parse_args()
    if a.synthetic:
        synthetic(Path(a.output))
    elif a.input:
        analyze(a)
    else:
        p.error('Provide --synthetic or --input')
