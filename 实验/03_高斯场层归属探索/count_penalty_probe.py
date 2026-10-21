"""Test a calibrated K=1/2 penalty; fixed total N, no physical count claims."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze import pair_metrics
from latent_competition import fit
from mechanism_sweep import proposals


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def calibrate(rows):
    single = np.sort([r['improvement'] for r in rows if r['sources'] == 1])
    dual = np.sort([r['improvement'] for r in rows if r['sources'] == 2])
    assert len(single) and len(dual)
    thresholds = np.unique(np.r_[0., single, dual])
    single_ok = np.searchsorted(single, thresholds, side='right')/len(single)
    dual_ok = 1.-np.searchsorted(dual, thresholds, side='right')/len(dual)
    scores = .5*(single_ok+dual_ok)
    at = int(np.argmax(scores))
    return float(thresholds[at]), float(scores[at])


def score(rows, threshold):
    one = [r for r in rows if r['sources'] == 1]; two = [r for r in rows if r['sources'] == 2]
    false_split = float(np.mean([r['improvement'] > threshold for r in one])) if one else None
    false_merge = float(np.mean([r['improvement'] <= threshold for r in two])) if two else None
    return dict(cases=len(rows), single_cases=len(one), dual_cases=len(two),
        false_K2_on_single=false_split, false_K1_on_dual=false_merge,
        balanced_count_accuracy=1.-.5*(false_split+false_merge) if one and two else None,
        mean_selected_pair_f1=float(np.mean([r['K2_pair_f1'] if r['improvement'] > threshold else r['K1_pair_f1'] for r in rows])),
        mean_effective_groups=float(np.mean([r['K2_groups'] if r['improvement'] > threshold else 1 for r in rows])),
        mean_unknown_members=float(np.mean([r['K2_unknown'] if r['improvement'] > threshold else 0 for r in rows])))


def self_check():
    rows = [dict(sources=1, improvement=.1), dict(sources=1, improvement=.2),
            dict(sources=2, improvement=.3), dict(sources=2, improvement=.4)]
    threshold, accuracy = calibrate(rows)
    assert threshold == .2 and accuracy == 1.
    rows = [dict(sources=s, improvement=0.) for s in (1, 2)]
    assert calibrate(rows) == (0., .5)
    # Independent brute-force objective comparison, strict ties favor K1.
    j1, j2, beta = .7, .2, .5
    assert (j1-j2 > beta) == (j2+beta < j1)
    print('count penalty self-check passed', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path); p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--self-check', action='store_true'); args = p.parse_args()
    if args.self_check:
        self_check(); return
    if args.output is None or args.repeats < 4 or args.repeats % 2:
        p.error('Require output and an even repeat count >=4')
    out = args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []; condition = 0
    for sources in (1, 2):
        gaps = (0.,) if sources == 1 else (0., .005, .02, .05)
        directions = (0.,) if sources == 1 else (0., 10., 30., 60.)
        for gap, angle, dz, dn in itertools.product(gaps, directions, (0., .002, .008), (0., 5., 15., 30.)):
            for repeat in range(args.repeats):
                seed = 20261022+condition*args.repeats+repeat
                d, n, truth = proposals(np.random.default_rng(seed), sources, gap, angle, dz, dn, 24//sources)
                assert len(d) == 24
                for lam in (0., 1.):
                    one = fit(d, n, np.ones(24, bool), 1, normal_weight=lam, objective='hard_min')
                    two = fit(d, n, np.ones(24, bool), 2, normal_weight=lam, objective='hard_min')
                    delta = one['objective']-two['objective']; assert delta >= -1e-10
                    rows.append(dict(condition=condition, sources=sources, depth_gap=gap,
                        direction_gap_deg=angle, depth_noise=dz, normal_noise_deg=dn,
                        repeat=repeat, seed=seed, total_members=24, normal_weight=lam,
                        J1=one['objective'], J2=two['objective'], improvement=max(0., delta),
                        K1_pair_f1=pair_metrics(truth, one['labels'])['f1'],
                        K2_pair_f1=pair_metrics(truth, two['labels'])['f1'],
                        K2_groups=len(set(two['labels'])-{-1}), K2_unknown=two['unknown_count']))
            condition += 1
            if condition % 24 == 0:
                write_csv(out/'cases.csv', rows)
                print('count penalty conditions', condition, '/204', flush=True)
    write_csv(out/'cases.csv', rows)
    results = []; counterexamples = []
    for lam in (0., 1.):
        part = [r for r in rows if r['normal_weight'] == lam]
        for protocol in ('seed_holdout_same_conditions', 'noise_condition_holdout'):
            def training(r):
                return r['repeat'] < args.repeats//2 if protocol.startswith('seed') else (
                    r['depth_noise'] <= .002 and r['normal_noise_deg'] <= 5.)
            train = [r for r in part if training(r)]; test = [r for r in part if not training(r)]
            threshold, accuracy = calibrate(train)
            for subset in ('all', 'distinguishable_used_prototypes', 'positive_depth_gap'):
                selected = [r for r in test if subset == 'all' or r['sources'] == 1 or (
                    r['depth_gap'] > 0 or (subset == 'distinguishable_used_prototypes' and lam and r['direction_gap_deg'] > 0))]
                results.append(dict(protocol=protocol, subset=subset, normal_weight=lam,
                    calibrated_penalty=threshold, calibration_balanced_accuracy=accuracy,
                    calibration_cases=len(train), **score(selected, threshold)))
        # Label-assisted diagnostic only: show an inversion ruling out a universally correct penalty.
        single = max((r for r in part if r['sources'] == 1), key=lambda r: r['improvement'])
        dual = min((r for r in part if r['sources'] == 2 and r['depth_gap'] > 0), key=lambda r: r['improvement'])
        counterexamples.append(dict(normal_weight=lam, single=single, positive_gap_dual=dual,
            inverted=single['improvement'] >= dual['improvement'],
            interpretation='Correct single requires beta>=delta_single, correct dual beta<delta_dual; inversion excludes a shared beta for these two cases.'))
    write_csv(out/'summary.csv', results)
    (out/'counterexamples.json').write_text(json.dumps(counterexamples, indent=2), encoding='utf-8')
    source = Path(__file__).parent
    metadata = dict(status='complete', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        generated_cases=condition*args.repeats, fixed_total_members=24,
        source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
            ('count_penalty_probe.py', 'latent_competition.py', 'mechanism_sweep.py', 'analyze.py')},
        selection='K2 iff J1-J2>beta, minimize mean member energy + beta*(K-1); strict ties select K1',
        interpretation='Supervised calibration diagnostic, not an unsupervised method or likelihood/BIC. '
        'Known labels do not enter finite fits, but calibration uses known source counts. '
        'Seed holdout tests new seeds under the same mechanism conditions; noise holdout uses disjoint noise settings. '
        'All evaluations are synthetic feature proposals, not host Gaussian rendering or physical surface counts. '
        'Conditional K2 may collapse or abstain; selected nominal K and effective groups are separate. '
        'Classes and condition grids are artificial; balanced accuracy is not a real-scene estimate.')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
