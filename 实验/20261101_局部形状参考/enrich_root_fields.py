"""Evaluate fields at frozen root positions; never searches, removes, or selects roots."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import local_shape_reference as ref
from run_fresh_capture import read_npz,EXPECTED_SHA

ROOT_SHA='5260e68a11f16cc2f34691a1eb00217413c75407c9eeb49c74364ad728b6f227'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--roots',type=Path,required=True)
    ap.add_argument('--capture',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.resolve()==args.roots.resolve() or args.output.exists():
        raise ValueError('enrichment output must be a new file; original roots stay immutable')
    roots_sha=hashlib.sha256(args.roots.read_bytes()).hexdigest()
    capture_sha=hashlib.sha256(args.capture.read_bytes()).hexdigest()
    if roots_sha!=ROOT_SHA or capture_sha!=EXPECTED_SHA: raise ValueError('input identity mismatch')
    result=json.loads(args.roots.read_text(encoding='utf8'))
    if result['reference_sha256']!=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest():
        raise ValueError('reference changed since root computation')
    d,h=read_npz(args.capture); raw=d['world_view_transform']; ff=h['world_view_transform']['fortran_order']
    matrix=[[raw[i+4*j if ff else 4*i+j] for j in range(4)] for i in range(4)]
    camera_rotation=[row[:3] for row in matrix[:3]]
    origin=ref.mul(-1,ref.mv(camera_rotation,matrix[3][:3])); fx,fy,cx,cy=d['intrinsics']
    geom={}
    for j,uid in enumerate(d['unique_id']):
        w,x,y,z=ref.unit(list(d['quaternion'][4*j:4*j+4]))
        rotation=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                  [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                  [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
        geom[uid]=ref.gaussian(d['center'][3*j:3*j+3],d['scale_render'][3*j:3*j+3],d['opacity_render'][j],rotation)
    counters={g:dict(roots=0,outside_all_3sigma=0) for g in ('grid_256','grid_512')}
    matched_outside=[]; maximum_normal_difference=0.; capacity=h['ids']['shape'][1]
    for row in result['rays']:
        if row['status']!='ok': continue
        q=row['query_index']; count=row['count']; ids=list(d['ids'][q*capacity:q*capacity+count])
        gs=[geom[uid] for uid in ids]; native_weights=list(d['weight'][q*capacity:q*capacity+count])
        ray=ref.mv(camera_rotation,[(d['xy'][2*q]-cx)/fx,(d['xy'][2*q+1]-cy)/fy,1.])
        for grid in ('grid_256','grid_512'):
            for root_index,root in enumerate(row[grid]['roots']):
                point=ref.add(origin,ref.mul(root['z'],ray)); state=ref.field(gs,point,ray)
                normal=state['surface_normal']
                if normal is not None and ref.dot(normal,ray)>0: normal=ref.mul(-1,normal)
                if normal is None or root['surface_normal_world'] is None:
                    assert normal is None and root['surface_normal_world'] is None
                else:
                    difference=max(abs(a-b) for a,b in zip(normal,root['surface_normal_world']))
                    maximum_normal_difference=max(maximum_normal_difference,difference)
                    if difference>1e-10: raise ValueError('existing actual surface normal did not replay')
                q2=[max(0.,ref.dot(it['delta'],ref.mv(it['p']['Q'],it['delta']))) for it in state['items']]
                within=[i for i,v in enumerate(q2) if v<=9.]
                log_components=[math.log(g['amplitude'])-.5*v for g,v in zip(gs,q2)]
                outside=not within; eigenvalues=state['eigenvalues']; grad=state['gradF']
                # Add only new keys. Existing roots, thresholds, matching and normal stay untouched.
                additions=dict(gradF_world=grad,gradF_world_norm=ref.norm(grad),guide_normal_world=state['normal'],
                    normal_field_eigenvalues=eigenvalues,absolute_eigen_gap=eigenvalues[0]-eigenvalues[1],
                    log_response_sum=state['log_density'],response_sum=math.exp(state['log_density']),
                    maximum_component_response=math.exp(max(log_components)),
                    min_mahalanobis_sq=min(q2),members_within_3sigma=len(within),
                    member_ids_within_3sigma=[int(ids[i]) for i in within],outside_all_3sigma=outside,
                    response_fraction_within_3sigma=sum(state['pi'][i] for i in within),
                    native_weight_sum_within_3sigma=sum(native_weights[i] for i in within))
                assert not set(additions).intersection(root)
                root.update(additions); counters[grid]['roots']+=1; counters[grid]['outside_all_3sigma']+=int(outside)
                if grid=='grid_512' and root.get('matched_admissible') and outside:
                    matched_outside.append(dict(center_index=row['center_index'],flat_pixel=row['flat_pixel'],
                        root_index=root_index,z=root['z'],min_mahalanobis_sq=min(q2),
                        log_response_sum=state['log_density'],response_sum=math.exp(state['log_density'])))
    original=json.loads(args.roots.read_text(encoding='utf8'))
    for before,after in zip(original['rays'],result['rays']):
        for grid in ('grid_256','grid_512'):
            if grid not in before: continue
            assert len(before[grid]['roots'])==len(after[grid]['roots'])
            for old,new in zip(before[grid]['roots'],after[grid]['roots']):
                assert all(new[k]==v for k,v in old.items())
    result['enrichment']=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_roots_sha256=roots_sha,
        capture_sha256=capture_sha,root_search_performed=False,root_selection_performed=False,
        original_fields_unchanged=True,maximum_existing_surface_normal_difference=maximum_normal_difference,
        coordinate_definition='world point = camera_origin + camera_z * world_ray; world_ray is rotated [(u-Cx)/Fx,(v-Cy)/Fy,1], not normalized',
        surface_normal_definition='existing surface_normal_world is normalized actual gradF, flipped to face camera; guide_normal_world is n; gradF_world retains the local F orientation',
        curvature_definition='existing curvature is n^T Hess(log response_sum) n; not mean/Gaussian surface curvature',
        response_definition='sum filtered_opacity_i * exp(-0.5 Mahalanobis_sq_i); dimensionless unnormalized local response, not calibrated occupancy/visibility',
        absolute_eigen_gap_definition='lambda1-lambda2 of sum pi_i gamma_i n_i n_i^T; existing relative gap divides by trace',
        normal_output_parameter_backward='not implemented or verified',
        counters=counters,matched_admissible_outside_3sigma=matched_outside)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result['enrichment'],indent=2))


if __name__=='__main__': main()
