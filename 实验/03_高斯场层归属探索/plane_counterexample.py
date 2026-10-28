"""Coplanar ellipsoid/ray counterexample, with independent closed-form checks."""
import json
import subprocess
import numpy as np
import directional_support_probe as base


def main():
    st=.1; sn=.005; p=np.array([0.,0.,2.]); origin=np.zeros(3); ray=np.array([0.,0.,1.])
    offsets=np.array([-1.,-.5,0.,.5,1.])*st
    cases=[]
    for angle in (0,45,80,89):
        theta=np.deg2rad(angle); n=np.array([np.sin(theta),0.,np.cos(theta)])
        e1=np.array([np.cos(theta),0.,-np.sin(theta)]); e2=np.array([0.,1.,0.])
        R=np.column_stack([e1,e2,n]); scales=np.tile([st,st,sn],(len(offsets),1)); rot=np.tile(R,(len(offsets),1,1))
        denominator=(np.sin(theta)/st)**2+(np.cos(theta)/sn)**2
        for delta in (0.,-sn,sn):
            center=p+delta*n; mu=center+offsets[:,None]*e1
            plane_t=float(n@center/(n@ray)); t,sigma=base.section(mu,scales,rot,origin,ray)
            analytic=2+(offsets*(-np.sin(theta)/st**2)+delta*np.cos(theta)/sn**2)/denominator
            np.testing.assert_allclose(t,analytic,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(sigma,1/np.sqrt(denominator),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose((mu-center)@n,0,atol=1e-15)
            np.testing.assert_allclose(n@(origin+plane_t*ray-center),0,atol=1e-14)
            if delta==0: np.testing.assert_allclose(plane_t,2,atol=1e-14)
            ratio=np.exp(-.5*((plane_t-t)/sigma)**2)
            cases.append({'angle_degrees':angle,'normal_offset':delta,'plane_intersection':plane_t,
                't_star':t.tolist(),'t_star_minus_original_2':(t-2).tolist(),
                't_star_range':[float(t.min()),float(t.max())], 'sigma':sigma.tolist(),
                'response_at_own_plane_over_peak':ratio.tolist(),
                'max_abs_coplanarity_residual':float(np.max(np.abs((mu-center)@n)))})
    result={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip(),
        'ray_origin':origin.tolist(),'ray_direction':ray.tolist(),'original_plane_point':p.tolist(),
        'tangential_scale':st,'normal_scale':sn,'tangent_offsets':offsets.tolist(),'cases':cases,
        'definition':'Analytical synthetic ellipsoids; no training data, renderer weights, real scene, GT, or surface estimator. Offset cases move the common plane and all centers by +/- normal scale. Ratios compare response at the respective plane intersection to each Gaussian peak. Peak offset alone does not determine estimator error.'}
    out=base.OUT/'plane_counterexample.json'
    if out.exists(): raise FileExistsError(out)
    base.OUT.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for c in cases:
        print(f"angle={c['angle_degrees']} delta={c['normal_offset']:+.3f} plane={c['plane_intersection']:.9f} t_range={c['t_star_range']} sigma={c['sigma'][0]:.9f} ratio={c['response_at_own_plane_over_peak']}")


if __name__=='__main__': main()
