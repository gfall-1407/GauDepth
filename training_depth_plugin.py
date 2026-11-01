"""Opt-in depth route and A/B diagnostics; no candidate readout is defined here."""

import hashlib
import json
from pathlib import Path

import torch


DEPTH_KEYS = ('expected_depth', 'median_depth')
PARAMETER_NAMES = ('xyz', 'scaling', 'rotation')


def validate_ab_output_dir(model_path):
    """Reject an occupied A/B destination before train creates any files."""
    if not model_path:
        raise ValueError('A/B run requires an explicit new model_path')
    output_path = Path(model_path)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f'A/B model_path must be absent or empty: {output_path}')


def reset_warm_start_radii(gaussians):
    """Scene's initialization size may differ from the loaded PLY size."""
    gaussians.max_radii2D = torch.zeros_like(gaussians.get_xyz[:, 0])


def should_log_parameters(iteration, first_iteration, interval):
    return iteration == first_iteration or iteration % interval == 0


def fixed_group_short_axis_ls_z(centers_camera, normals_camera, ray_v, members):
    """Differentiable camera-z LS for an externally fixed member group.

    Camera origin is zero and ray_v=(x,y,1), not a unit ray. This routine does
    no grouping, surface selection, footprint check, or full-image rendering.
    """
    if (centers_camera.ndim != 2 or centers_camera.shape[1] != 3 or
            normals_camera.shape != centers_camera.shape or ray_v.shape != (3,) or
            not torch.isfinite(centers_camera).all() or
            not torch.isfinite(normals_camera).all() or
            not torch.isfinite(ray_v).all() or ray_v[2] != 1):
        raise ValueError('expected finite camera-space [N,3] centers/normals and ray (x,y,1)')
    ids = torch.as_tensor(members, dtype=torch.long, device=centers_camera.device).detach()
    if ids.ndim != 1 or ids.numel() == 0 or torch.any(ids < 0) or torch.any(ids >= centers_camera.shape[0]):
        raise ValueError('fixed group must contain valid member indices')
    normals = normals_camera.index_select(0, ids)
    centers = centers_camera.index_select(0, ids)
    incidence = (normals * ray_v).sum(dim=-1)
    offsets = (normals * centers).sum(dim=-1)
    denominator = (incidence * incidence).sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError('degenerate fixed group')
    return (incidence * offsets).sum() / denominator


def route_depth(render_pkg, mode='baseline', target='expected'):
    """Keep baseline tensors untouched until a reviewed candidate exists.

    identity is a wiring control, not a competing depth estimator. candidate
    fails explicitly so a missing implementation cannot silently run as A/B.
    """
    if target not in ('expected', 'median', 'both'):
        raise ValueError(f'unknown depth target: {target}')
    if mode == 'baseline':
        return render_pkg
    if mode == 'identity':
        for key in DEPTH_KEYS:
            depth = render_pkg[key]
            if depth.ndim != 3 or depth.shape[0] != 1 or not depth.is_floating_point():
                raise ValueError(f'{key} must have floating [1,H,W] shape')
        return dict(render_pkg)
    if mode == 'candidate':
        if 'candidate_depth' not in render_pkg or 'candidate_audit' not in render_pkg:
            raise RuntimeError('candidate requires renderer accepted support')
        candidate = render_pkg['candidate_depth']
        valid = render_pkg['candidate_audit']['candidate'].reshape_as(candidate)
        if candidate.shape != render_pkg['expected_depth'].shape or valid.shape != candidate.shape:
            raise ValueError('candidate depth and mask must match native [1,H,W]')
        output = dict(render_pkg)
        for key in DEPTH_KEYS:
            if target == 'both' or (target == 'expected' and key == 'expected_depth') or (
                    target == 'median' and key == 'median_depth'):
                output[key] = torch.where(valid, candidate, render_pkg[key])
        return output
    raise ValueError(f'unknown depth route: {mode}')


def summarize_candidate_audit(audit, elapsed_ms, peak_allocated_mb=None):
    """Small scalar diagnostics; only call in an opt-in A/B log path."""
    if audit is None:
        return None
    return dict(queries=int(audit['candidate'].numel()),
                selected=int(audit['candidate'].sum().item()),
                overflow=int(audit['overflow'].sum().item()),
                invalid_members=int(audit['invalid_count'].sum().item()),
                no_valid=int(audit['no_valid'].sum().item()),
                group_switch=int(audit['group_switch'].sum().item()),
                low_alpha=int(audit['low_alpha'].sum().item()),
                footprint_fail=int(audit['footprint_fail'].sum().item()),
                insufficient_distinct=int(audit['insufficient_distinct'].sum().item()),
                elapsed_ms=float(elapsed_ms),
                gpu_peak_allocated_mb=None if peak_allocated_mb is None else float(peak_allocated_mb))


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parameter_digest(gaussians):
    """Hash training parameters at the same A/B start; no model data in logs."""
    h = hashlib.sha256()
    h.update(str(gaussians.active_sh_degree).encode('ascii'))
    for group in gaussians.optimizer.param_groups:
        for index, param in enumerate(group['params']):
            data = param.detach().contiguous().cpu().numpy()
            h.update(f'{group["name"]}:{index}'.encode('ascii'))
            h.update(str(data.shape).encode('ascii'))
            h.update(str(data.dtype).encode('ascii'))
            h.update(data.tobytes())
    if hasattr(gaussians, 'filter_3D'):
        data = gaussians.filter_3D.detach().contiguous().cpu().numpy()
        h.update(b'filter_3D')
        h.update(str(data.shape).encode('ascii'))
        h.update(data.tobytes())
    return h.hexdigest()


def parameter_metrics(optimizer, limit=2048):
    """Deterministic first-N sample to limit opt-in logging overhead."""
    output = {}
    before = {}
    for group in optimizer.param_groups:
        name = group.get('name')
        if name not in PARAMETER_NAMES:
            continue
        param = group['params'][0]
        sample = param.reshape(-1)[:limit]
        grad = None if param.grad is None else param.grad.reshape(-1)[:limit]
        output[name] = dict(lr=float(group['lr']), sample_elements=int(sample.numel()),
                            value_rms=float(torch.sqrt(torch.mean(sample.detach()**2)).item()),
                            grad_rms=None if grad is None else
                            float(torch.sqrt(torch.mean(grad.detach()**2)).item()))
        before[name] = sample.detach().clone()
    return output, before


def add_step_metrics(metrics, before, optimizer, applied):
    params = {g.get('name'): g['params'][0] for g in optimizer.param_groups}
    for name, row in metrics.items():
        if applied:
            delta = params[name].reshape(-1)[:len(before[name])].detach() - before[name]
            row['step_rms'] = float(torch.sqrt(torch.mean(delta**2)).item())
        else:
            row['step_rms'] = None
    return metrics


class ABLog:
    def __init__(self, model_path, run_id, interval):
        if not run_id or not run_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError('ab_run_id must use letters, digits, - or _')
        if interval < 1:
            raise ValueError('ab_log_interval must be positive for an A/B run')
        self.path = Path(model_path) / 'ab_trace.jsonl'
        self.interval = interval
        self.stream = self.path.open('x', encoding='utf-8')

    def write(self, row):
        self.stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()
