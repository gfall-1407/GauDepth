"""Observed refinement diagnostics, not continuous-error certificates."""
import torch


PROBE_PROTOCOL = {
    'id': 'joint_tv_piecewise_linear_probe_v2',
    'threshold': 'log(2)',
    'near_axial': .2001,
    'far_axial': 8.,
    'parameter_dtype': 'float32',
    'field_dtype': 'float32',
    'opacity_policy': '(1-1e-6)*native_3D_filtered_opacity',
    'opacity_factor': 1-1e-6,
    'member_rule': 'full_line_peak_amplitude >= member_epsilon',
    'default_member_epsilon': 1e-6,
    'member_selection_gradient': 'fixed support within each evaluated ray',
    'integration': 'TV of piecewise-linear H on moving differentiable knots',
    'refinement_levels': [2,4,8],
    'min_secant_slope': 1e-4,
    'min_analytic_abs_Hprime': 1e-4,
    'max_near_H': 1e-4,
    'observed_depth_atol_axial': 1e-5,
    'observed_total_TV_atol': 1e-5,
    'observed_gradient_l2_rtol': .01,
    'observed_gradient_l2_atol': 1e-6,
    'continuous_error_certified': False,
    'frozen_reference_parity_verified': False,
    'frozen_gpu_reference_identified': False,
    'reference_differences': {
        'near': 'frozen CPU reference starts at camera z=0, new probe .2001',
        'opacity': 'frozen reference has no new 1e-6 multiplicative shrink',
        'members': 'frozen reference uses >1e-14/N, new probe >=1e-6',
        'precision': 'frozen reference float64, new probe float32',
        'integration': 'frozen reference searches Hprime turning points; new probe interpolates H',
        'tolerances': 'frozen total change 1e-8, ray-distance change 1e-6; probe tolerances above are distinct',
    },
}


def gradient_delta(previous, current, *, rtol=.01, atol=1e-6):
    """Compare vectors on identical parameter indices, not only their norms."""
    delta = (current-previous).double()
    a,b = previous.double(),current.double()
    scale = torch.maximum(a.norm(),b.norm())
    norm=delta.norm()
    finite=bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:
        return dict(finite=False,delta_l2=None,delta_linf=None,relative_l2=None,passed=False)
    return dict(finite=finite,delta_l2=float(norm),delta_linf=float(delta.abs().max()),
                relative_l2=float(norm/scale) if float(scale)>0 else 0.,
                passed=finite and float(norm)<=atol+rtol*float(scale))


def refinement_comparison(previous, current, old_grads, new_grads):
    same_status=(previous['valid']==current['valid'] and
                 previous['has_sampled_root']==current['has_sampled_root'])
    comparable=same_status and previous['valid'] and current['valid']
    gradients={k:gradient_delta(old_grads[k],new_grads[k],
                  rtol=PROBE_PROTOCOL['observed_gradient_l2_rtol'],
                  atol=PROBE_PROTOCOL['observed_gradient_l2_atol']) for k in old_grads}
    depth_delta=abs(previous['depth']-current['depth']) if comparable else None
    tv_delta=abs(previous['total']-current['total'])
    observed_pass=bool(comparable and depth_delta<=PROBE_PROTOCOL['observed_depth_atol_axial']
                      and tv_delta<=PROBE_PROTOCOL['observed_total_TV_atol']
                      and all(g['passed'] for g in gradients.values()))
    return dict(levels=[previous['subdivisions'],current['subdivisions']],
                same_status=same_status,comparable_valid_roots=comparable,
                depth_abs_change=depth_delta,total_TV_abs_change=tv_delta,
                parameter_gradient_changes=gradients,observed_pair_passed=observed_pass,
                continuous_error_certified=False)
