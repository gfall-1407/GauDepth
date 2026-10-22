"""Repeated Gaussian-pair diagnostics; consistency is not physical accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze import group_ray


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_arrays(ids, depth, normal, valid, labels, geometry, truth=None):
    ids = np.asarray(ids, np.uint64)
    assert len(np.unique(ids)) == len(ids) and np.all(ids < 2**32)
    depth, normal, labels = np.asarray(depth, float), np.asarray(normal, float), np.asarray(labels)
    good = np.isfinite(depth) & (depth > 0)
    length = np.linalg.norm(normal, axis=1)
    if geometry:
        good &= np.asarray(valid, bool) & np.isfinite(normal).all(axis=1) & (length > 1e-8)
    logz = np.zeros(len(depth)); logz[good] = np.log(depth[good])
    distance = np.abs(logz[:, None]-logz[None, :])/np.log1p(.01)
    if geometry:
        unit = np.zeros_like(normal)
        np.divide(normal, length[:, None], out=unit, where=(length > 1e-8)[:, None])
        distance = np.maximum(distance, np.arccos(np.clip(unit@unit.T, -1, 1))/np.deg2rad(15.))
    i, j = np.triu_indices(len(ids), 1)
    known = (labels[i] >= 0) & (labels[j] >= 0)
    assert np.array_equal(known, good[i] & good[j])
    compatible = known & (distance[i, j] <= 1+1e-12)
    same = known & (labels[i] == labels[j])
    assert np.all(~same | compatible)
    key = (np.minimum(ids[i], ids[j]) << np.uint64(32)) | np.maximum(ids[i], ids[j])
    truth_same = np.zeros(len(i), bool) if truth is None else known & (truth[i] == truth[j])
    # Each key occurs once per ray. Columns: raw, known, compatible, same, oracle-positive.
    counts = np.column_stack((np.ones(len(i), np.int64), known, compatible, same, truth_same))
    return key, counts


def reduce_pairs(parts):
    if not parts or not any(len(p[0]) for p in parts):
        return np.empty(0, np.uint64), np.empty((0, 5), np.int64)
    keys = np.concatenate([p[0] for p in parts]); counts = np.concatenate([p[1] for p in parts])
    order = np.argsort(keys, kind='stable'); keys, counts = keys[order], counts[order]
    start = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    return keys[start], np.add.reduceat(counts, start, axis=0)


def summarize(counts, oracle=False, cross=None):
    raw, known, comp, same, truth = counts.T
    repeat_raw = raw >= 2
    if cross is None:
        eligible = known >= 2
        comparisons = known*(known-1)//2
        raw_comparisons = raw*(raw-1)//2
        disagreements = same*(known-same)
    else:
        # cross columns: sum known_view^2, sum same_view*diff_view, known/raw views.
        repeat_raw = cross[:, 3] >= 2
        eligible = cross[:, 2] >= 2
        comparisons = (known*known-cross[:, 0])//2
        raw_comparisons = (raw*raw-cross[:, 4])//2
        disagreements = same*(known-same)-cross[:, 1]
    eligible &= comparisons > 0
    flips = eligible & (same > 0) & (same < known)
    stable_comp = eligible & (comp == known)
    row = dict(unique_observed_pairs=len(raw), repeated_raw_pairs=int(repeat_raw.sum()),
               repeated_known_pairs=int(eligible.sum()),
               raw_pair_observations=int(raw.sum()), known_pair_observations=int(known.sum()),
               same_pair_observations=int(same.sum()), compatible_pair_observations=int(comp.sum()),
               repeated_pair_observations=int(known[eligible].sum()),
               repeated_known_2=int((eligible & (known == 2)).sum()),
               repeated_known_3_to_5=int((eligible & (known >= 3) & (known <= 5)).sum()),
               repeated_known_6_plus=int((eligible & (known >= 6)).sum()),
               flip_pairs=int(flips.sum()), always_compatible_pairs=int(stable_comp.sum()),
               always_compatible_flip_pairs=int((stable_comp & flips).sum()),
               always_incompatible_pairs=int((eligible & (comp == 0)).sum()),
               compatibility_changing_pairs=int((eligible & (comp > 0) & (comp < known)).sum()),
               compared_occurrence_pairs=int(comparisons[eligible].sum()),
               raw_compared_occurrence_pairs=int(raw_comparisons.sum()),
               disagreeing_occurrence_pairs=int(disagreements[eligible].sum()),
               sum_pair_disagreement_rate=float((disagreements[eligible]/comparisons[eligible]).sum()))
    if oracle:
        assert np.all((truth == 0) | (truth == known))
        row.update(stable_wrong_merge_pairs=int((eligible & (same == known) & (truth == 0)).sum()),
                   stable_wrong_split_pairs=int((eligible & (same == 0) & (truth == known)).sum()))
    else:
        row.update(stable_wrong_merge_pairs=None, stable_wrong_split_pairs=None)
    return ratios(row)


def ratios(row):
    for output, a, b in (
        ('known_observation_coverage', 'known_pair_observations', 'raw_pair_observations'),
        ('known_comparison_coverage', 'compared_occurrence_pairs', 'raw_compared_occurrence_pairs'),
        ('repeated_pair_coverage', 'repeated_known_pairs', 'repeated_raw_pairs'),
        ('same_observation_fraction', 'same_pair_observations', 'known_pair_observations'),
        ('flip_pair_fraction', 'flip_pairs', 'repeated_known_pairs'),
        ('equal_pair_disagreement_fraction', 'sum_pair_disagreement_rate', 'repeated_known_pairs'),
        ('always_compatible_flip_fraction', 'always_compatible_flip_pairs', 'always_compatible_pairs'),
        ('occurrence_disagreement_fraction', 'disagreeing_occurrence_pairs', 'compared_occurrence_pairs')):
        row[output] = row[a]/row[b] if row[b] else None
    return row


def across_views(views):
    # Caller must verify identical model SHA; keys are local to one model/case.
    keys, counts = reduce_pairs(views)
    extra = np.zeros((len(keys), 5), np.int64)
    for vk, vc in views:
        at = np.searchsorted(keys, vk)
        nk, ns = vc[:, 1], vc[:, 3]
        extra[at] += np.column_stack((nk*nk, ns*(nk-ns), nk > 0, vc[:, 0] > 0, vc[:, 0]**2))
    return counts, extra


def self_check():
    # Same A/B features; adding C alone changes their grouping context.
    normal = np.tile([0., 0., -1.], (3, 1)); valid = np.ones(3, bool)
    d = np.array([1.008, 1.016, 1.])
    a = group_ray(d[:2], normal[:2], valid[:2], geometry=False)[0]
    b = group_ray(d, normal, valid, geometry=False)[0]
    assert a[0] == a[1] and b[0] != b[1]
    p = pair_arrays(np.array([4, 9]), d[:2], normal[:2], valid[:2], a, False)
    q = pair_arrays(np.array([4, 9, 2]), d, normal, valid, b, False)
    k, c = reduce_pairs([p, q]); r = summarize(c)
    assert r['repeated_known_pairs'] == r['flip_pairs'] == r['always_compatible_flip_pairs'] == 1
    assert r['compared_occurrence_pairs'] == r['disagreeing_occurrence_pairs'] == 1
    c2, ex = across_views([reduce_pairs([p]), reduce_pairs([q])]); r2 = summarize(c2, cross=ex)
    assert r2 == r
    # Independent brute-force cross-view comparison, including unknown observations.
    records = [[(1, 1), (1, 0), (1, 1)], [(1, 0), (0, 0)], [(1, 1)]]
    vv = [(np.array([10], np.uint64), np.array([[len(v), sum(k for k, s in v),
           sum(k for k, s in v), sum(s for k, s in v), 0]], np.int64)) for v in records]
    cc, ee = across_views(vv); rr = summarize(cc, cross=ee)
    comparisons = disagreements = 0
    for x in range(len(records)):
        for y in range(x+1, len(records)):
            for ka, sa in records[x]:
                for kb, sb in records[y]:
                    if ka and kb:
                        comparisons += 1; disagreements += sa != sb
    assert rr['compared_occurrence_pairs'] == comparisons
    assert rr['disagreeing_occurrence_pairs'] == disagreements
    assert rr['raw_compared_occurrence_pairs'] == 11
    assert rr['known_comparison_coverage'] == 7/11
    # Group label renaming changes no pair statistic.
    renamed = np.where(b >= 0, 17+5*b, -1)
    qq = pair_arrays(np.array([4, 9, 2]), d, normal, valid, renamed, False)
    np.testing.assert_array_equal(q[1], qq[1])
    print('cross-ray self-check passed', flush=True)


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def camera_to_world_normals(normal,view):
    matrix=np.asarray(view['world_view_transform'],float);rotation=matrix[:3,:3]
    np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=2e-6,rtol=0)
    np.testing.assert_allclose((np.array([0.,0.,0.,1.])@np.linalg.inv(matrix))[:3],view['camera_center'],atol=2e-6,rtol=2e-6)
    # Stored transposed camera matrix maps row world points to camera points.
    # For rigid rotation, row camera normals map back by its transpose.
    return np.asarray(normal,float)@rotation.T


def world_normal_self_check():
    angle=.7;rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[3,:3]=[.3,-.4,.2]
    view=dict(world_view_transform=matrix.tolist(),camera_center=(np.array([0.,0.,0.,1.])@np.linalg.inv(matrix))[:3].tolist())
    normal=np.array([[1.,2.,3.]])/np.sqrt(14.)
    np.testing.assert_allclose(camera_to_world_normals(normal@rotation,view),normal,atol=1e-14)
    assert np.isclose(np.rad2deg(np.arccos(np.clip(np.abs(normal@(-normal).T),0.,1.)))[0,0],0.,atol=1e-6)
    print('world normal rigid-transform/translation/axis-sign self-check passed',flush=True)


def world_normal_audit(root,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];view_rows=[]
    for scene in ('scan24','scan105','scan65','scan110'):
        folders=[scene,scene+'_more'] if scene in ('scan24','scan105') else [scene]
        views=[];model_sha=None
        for folder in folders:
            directory=root/folder;meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            if model_sha is None:model_sha=meta['model_sha256']
            assert meta['model_sha256']==model_sha
            inputs.append(dict(path=str(directory/'metadata.json'),sha256=sha(directory/'metadata.json')))
            for view in meta['views']:
                path=directory/view['file'];assert sha(path)==view['npz_sha256'];inputs.append(dict(path=str(path),sha256=sha(path)))
                with np.load(path,allow_pickle=False) as z:
                    ids=z['gaussian_id'];normal=z['normal'].astype(float);valid=z['normal_valid'].astype(bool)
                    unique,first,inverse=np.unique(ids,return_index=True,return_inverse=True)
                    lengths=np.linalg.norm(normal,axis=1);valid&=np.isfinite(normal).all(axis=1)&(lengths>1e-8)
                    all_valid=np.ones(len(unique),bool);np.logical_and.at(all_valid,inverse,valid)
                    difference=np.max(np.abs(normal-normal[first[inverse]]),axis=1)
                    spread=np.zeros(len(unique));np.maximum.at(spread,inverse,np.where(np.isfinite(difference),difference,0.))
                    consistent=spread<=1e-6;usable=all_valid&consistent
                    world=camera_to_world_normals(normal[first],view)
                    unit=np.zeros_like(world);unit[usable]=world[usable]/np.linalg.norm(world[usable],axis=1)[:,None]
                    vrow=dict(scene=scene,view=view['view_index'],accepted_records=len(ids),unique_gaussians=len(unique),
                              invalid_records=int((~valid).sum()),invalid_gaussians=int((~all_valid).sum()),
                              inconsistent_gaussians=int((~consistent).sum()),max_finite_repeat_abs_difference=float(spread.max(initial=0.)))
                    view_rows.append(vrow);views.append(dict(view=view['view_index'],ids=unique,normal=unit,usable=usable))
        assert len(views)==3 and len({v['view'] for v in views})==3
        for i in range(3):
            for j in range(i+1,3):
                left,right=views[i],views[j];common,ia,ib=np.intersect1d(left['ids'],right['ids'],return_indices=True)
                known=left['usable'][ia]&right['usable'][ib];dot=np.sum(left['normal'][ia[known]]*right['normal'][ib[known]],axis=1)
                oriented=np.rad2deg(np.arccos(np.clip(dot,-1.,1.)));unsigned=np.rad2deg(np.arccos(np.clip(np.abs(dot),0.,1.)))
                row=dict(scene=scene,view_a=left['view'],view_b=right['view'],unique_a=len(left['ids']),unique_b=len(right['ids']),
                         common_gaussians=len(common),known_common_gaussians=int(known.sum()),unknown_common_gaussians=int((~known).sum()),
                         common_over_union=len(common)/(len(left['ids'])+len(right['ids'])-len(common)))
                for label,values in (('oriented',oriented),('unsigned',unsigned)):
                    for stat,q in (('median',.5),('p95',.95),('max',1.)):
                        row[label+'_'+stat+'_deg']=float(np.quantile(values,q)) if len(values) else None
                rows.append(row)
                np.savez_compressed(out/f'{scene}_{left["view"]}_{right["view"]}.npz',common_ids=common,known=known,oriented_deg=oriented,unsigned_deg=unsigned)
        csv_write(out/'summary.csv',rows);csv_write(out/'views.csv',view_rows)
        print('saved world-normal audit',scene,'finished',flush=True)
    assert len(rows)==len(view_rows)==12
    (out/'metadata.json').write_text(json.dumps(dict(status='complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),inputs=inputs,
        evidence='Read-only 12 saved original-host views, four identical-model-SHA groups. Camera normals rigidly transformed to world coordinates. '
        'One normal per Gaussian per view only if all its accepted records valid and repeat spread <=1e-6; failures retain unknown coverage. '
        'Across-view angles count each common Gaussian once per view pair, not every ray observation. Related pairs are not independent. '
        'Oriented and unsigned angles both retained; no layer threshold or physical accuracy. No 3D means/covariance axes recovered. '
        'Host source/binary provenance gap remains: this characterizes saved normals, not a proof of installed implementation. No fitting or new GPU readout.'),indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--world-normal-audit', action='store_true')
    args = parser.parse_args()
    if args.world_normal_audit:
        if args.self_check:world_normal_self_check();return
        if args.root is None or args.output is None:parser.error('--root and --output required')
        world_normal_audit(args.root,args.output);return
    if args.self_check:
        self_check(); return
    root, out = args.root, args.output
    if root is None or out is None:
        parser.error('--root and --output required')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    sources = []; rows = []; witnesses = []
    tasks = [(scene, [scene, scene+'_more'] if scene in ('scan24', 'scan105') else [scene])
             for scene in ('scan24', 'scan105', 'scan65', 'scan110')]
    analytic = root/'analytic_gaussian_control'
    meta_a = json.loads((analytic/'metadata.json').read_text(encoding='utf-8'))
    assert meta_a['status'] == 'complete' and meta_a['backend'] == 'analytic_ray_peak_gaussian_CPU'
    sources.append(dict(path=str(analytic/'metadata.json'), sha256=sha(analytic/'metadata.json')))
    cases = {v['file']: v for v in meta_a['cases']}
    tasks += [(p.stem, ['analytic_gaussian_control']) for p in sorted(analytic.glob('case_*.npz'))]
    for scene, folders in tasks:
        is_analytic = scene.startswith('case_')
        view_results = {'depth': [], 'depth_normal': []}
        model_sha = None
        for folder in folders:
            base = root/folder
            meta_path = base/'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if not is_analytic:
                if model_sha is not None: assert model_sha == meta['model_sha256']
                model_sha = meta['model_sha256']
                gpath = root/(folder+'_analysis')/'groups.json'
                cached = json.loads(gpath.read_text(encoding='utf-8'))
                lookup = {(r['file'], r['ray'], r['method']): r for r in cached}
                sources += [dict(path=str(gpath), sha256=sha(gpath)),
                            dict(path=str(meta_path), sha256=sha(meta_path))]
                files = sorted(base.glob('view_*.npz'))
            else:
                files = [base/(scene+'.npz')]
            for path in files:
                expected = cases[path.name]['npz_sha256'] if is_analytic else next(
                    v['npz_sha256'] for v in meta['views'] if v['file'] == path.name)
                assert sha(path) == expected
                sources.append(dict(path=str(path), sha256=expected))
                data = np.load(path, allow_pickle=False)
                for method in view_results:
                    geometry = method == 'depth_normal'; parts = []; ray_records = []
                    for ray in range(len(data['offsets'])-1):
                        if data['qualitative'][ray]: continue
                        a, b = map(int, data['offsets'][ray:ray+2]); sl = slice(a, b)
                        ids, d, n, v = [data[k][sl] for k in ('gaussian_id', 'depth', 'normal', 'normal_valid')]
                        truth = data['truth_origin'][sl] if is_analytic else None
                        if is_analytic:
                            labels = group_ray(d, n, v, geometry=geometry)[0]
                        else:
                            saved = lookup[(path.name, ray, method)]
                            assert saved['eta'] == .01 and saved['angle'] == 15.
                            np.testing.assert_array_equal(ids, saved['gaussian_id'])
                            labels = np.asarray(saved['labels'])
                        parts.append(pair_arrays(ids, d, n, v, labels, geometry, truth))
                        ray_records.append((ray, ids, labels))
                    keys, counts = reduce_pairs(parts)
                    view_results[method].append((keys, counts))
                    row = dict(dataset='analytic_cpu' if is_analytic else 'real_host', model=scene,
                               scope='within_view', view=path.stem, method=method,
                               **summarize(counts, is_analytic))
                    rows.append(row)
                    # A few deterministic, auditable pair IDs; full inputs retained, no huge pair table.
                    target = (counts[:, 1] >= 2) & (counts[:, 2] == counts[:, 1]) & (counts[:, 3] > 0) & (counts[:, 3] < counts[:, 1])
                    for at in np.flatnonzero(target)[:3]:
                        pair = [int(keys[at] >> np.uint64(32)), int(keys[at] & np.uint64(0xffffffff))]
                        examples = {}
                        for (ray, ids, labels), (rk, rc) in zip(ray_records, parts):
                            where = np.flatnonzero(rk == keys[at])
                            if not len(where): continue
                            relation = 'same' if rc[where[0], 3] else 'different'
                            if relation in examples: continue
                            ix = [int(np.flatnonzero(ids == gid)[0]) for gid in pair]
                            offset = int(data['offsets'][ray])
                            examples[relation] = dict(ray=ray, pixel=data['pixel_xy'][ray].tolist(),
                                pair_depth=data['depth'][offset+np.asarray(ix)].tolist(),
                                pair_labels=labels[ix].tolist(), accepted_members=len(ids),
                                groups_for_pair=[ids[labels == labels[i]].tolist() for i in ix])
                            if len(examples) == 2: break
                        witnesses.append(dict(model=scene, view=path.stem, method=method,
                            gaussian_ids=pair, examples=examples,
                            raw=int(counts[at, 0]), known=int(counts[at, 1]), compatible=int(counts[at, 2]), same=int(counts[at, 3])))
                data.close()
        if not is_analytic:
            for method, vv in view_results.items():
                counts, extra = across_views(vv)
                rows.append(dict(dataset='real_host', model=scene, scope='cross_view', view='all',
                                 method=method, **summarize(counts, cross=extra)))
        csv_write(out/'by_view.csv', rows)
        print(scene, 'finished', flush=True)
    summary = []
    integer_keys = [k for k in rows[0] if k not in ('dataset', 'model', 'scope', 'view', 'method')
                    and not k.endswith(('_fraction', '_coverage'))]
    for dataset, scope, method in sorted(set((r['dataset'], r['scope'], r['method']) for r in rows)):
        part = [r for r in rows if (r['dataset'], r['scope'], r['method']) == (dataset, scope, method)]
        pooled = {k: sum(r[k] for r in part) if all(r[k] is not None for r in part) else None for k in integer_keys}
        summary.append(dict(dataset=dataset, scope=scope, method=method, units=len(part), **ratios(pooled)))
    csv_write(out/'summary.csv', summary)
    (out/'witnesses.json').write_text(json.dumps(witnesses, indent=2), encoding='utf-8')
    metadata = dict(status='complete', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        source_sha256={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('analyze.py'))},
        numpy=np.__version__, eta=.01, angle=15., inputs=sources,
        evidence='Repeated Gaussian pair coassignment diagnostics, not physical surface accuracy. '
        'All-merge and all-singleton rules can be perfectly stable. Each model/case has its own IDs. '
        'Cross-view observations compared only within identical model SHA; camera normals never directly compared across views. '
        'Cross-ray membership may legitimately vary with local intersections. No independent sample claims. '
        'Within-view pooled equal-pair statistics weight pair-by-view units; cross-view units are model-local unique pairs. '
        'Known observation coverage uses all observations; known comparison coverage measures usable repeated-event comparisons. '
        'Analytic labels are construction origins; analytic CPU is not the host renderer.')
    (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
