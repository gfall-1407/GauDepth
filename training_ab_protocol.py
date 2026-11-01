"""Shared, opt-in A/B warm-start and logging protocol from reviewed f864cb1."""

import hashlib
import json
from pathlib import Path

import torch


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
