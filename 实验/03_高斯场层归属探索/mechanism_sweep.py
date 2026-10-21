"""Fixed-rule stochastic mechanism probes, not a Gaussian-renderer simulation."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from analyze import group_ray, pair_metrics


def proposals(rng, sources, gap, direction_gap, depth_noise, normal_noise, size):
    truth = np.repeat(np.arange(sources), size)
    depth = 1.0 + gap * truth + rng.normal(0, depth_noise, len(truth))
    theta = np.deg2rad(direction_gap * truth)
    base = np.c_[np.sin(theta), np.zeros(len(truth)), -np.cos(theta)]
    tangent = np.c_[np.cos(theta), np.zeros(len(truth)), np.sin(theta)]
    delta = rng.normal(0, np.deg2rad(normal_noise), len(truth))
    azimuth = rng.uniform(0, 2 * np.pi, len(truth))
    normal = np.cos(delta)[:, None] * base + np.sin(delta)[:, None] * (
        np.cos(azimuth)[:, None] * tangent
        + np.sin(azimuth)[:, None] * np.array([0.0, 1.0, 0.0]))
    order = rng.permutation(len(truth))
    return depth[order], normal[order], truth[order]


def run(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(seed=args.seed, repeats=args.repeats, members_per_source=args.size,
                eta=.01, angle=15., status='running',
                evidence='Phenomenological labelled feature proposals, NOT a physical renderer or real-scene accuracy.',
                caveat='Zero gap is a degenerate feature-identifiability control, not evidence for two physical interfaces at identical depth.',
                depth_noise='camera-z standard deviation at nominal depth 1',
                normal_noise='standard deviation of signed angular displacement around a source direction; uniform tangent azimuth',
                source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                               for n in ('analyze.py', 'mechanism_sweep.py')})
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    fields = ['sources', 'depth_gap', 'direction_gap_deg', 'depth_noise',
              'normal_noise_deg', 'method', 'repeats', 'mean_pair_f1',
              'std_pair_f1', 'mean_false_merge_pairs', 'mean_false_split_pairs',
              'exact_partition_fraction', 'mean_groups']
    conditions = 0
    with (out / 'sweep.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sources in (1, 2):
            gaps = (0.,) if sources == 1 else (0., .005, .02, .05)
            directions = (0.,) if sources == 1 else (0., 10., 30., 60.)
            for gap, angle, dz, dn in itertools.product(gaps, directions, (0., .002, .008), (0., 5., 15., 30.)):
                values = {False: [], True: []}
                for rep in range(args.repeats):
                    rng = np.random.default_rng(args.seed + conditions * args.repeats + rep)
                    d, n, truth = proposals(rng, sources, gap, angle, dz, dn, args.size)
                    for geometry in (False, True):
                        labels, _ = group_ray(d, n, np.ones(len(d), bool), geometry=geometry)
                        score = pair_metrics(truth, labels)
                        values[geometry].append([score['f1'], score['fp'], score['fn'],
                                                 float(score['fp'] == score['fn'] == 0),
                                                 len(set(labels) - {-1})])
                for geometry, data in values.items():
                    arr = np.asarray(data, float)
                    writer.writerow(dict(sources=sources, depth_gap=gap, direction_gap_deg=angle,
                                         depth_noise=dz, normal_noise_deg=dn,
                                         method='depth_normal' if geometry else 'depth', repeats=args.repeats,
                                         mean_pair_f1=arr[:, 0].mean(), std_pair_f1=arr[:, 0].std(),
                                         mean_false_merge_pairs=arr[:, 1].mean(), mean_false_split_pairs=arr[:, 2].mean(),
                                         exact_partition_fraction=arr[:, 3].mean(), mean_groups=arr[:, 4].mean()))
                f.flush()
                conditions += 1
    meta.update(status='complete', conditions=conditions, method_rows=conditions * 2,
                generated_cases=conditions * args.repeats)
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--size', type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1 or args.size < 2:
        parser.error('repeats >= 1 and size >= 2 required')
    run(args)
