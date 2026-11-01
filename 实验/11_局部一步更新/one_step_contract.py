"""Guards and one-step optimizer for the *conditional* local intervention.

The caller must supply real full-scene losses and a valid fixed 3x3 root
patch. This module does not resume the original 7001 optimizer or train.
"""
import hashlib
import json
from pathlib import Path

import torch


PLY_SHA256 = '733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7'
PLY_BYTES = 36010340
STEP = 7001
SEED = 0
RAW_GROUPS = ('xyz', 'scaling', 'rotation', 'opacity')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_full_ply(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != PLY_BYTES or digest(path) != PLY_SHA256:
        raise ValueError('wrong_or_missing_full_scan110_7k_ply')
    return path


def appearance_digest(embedding, camera_name_to_uid):
    names = json.dumps(camera_name_to_uid, sort_keys=True, separators=(',', ':')).encode()
    values = embedding.detach().cpu().contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError('nonfinite_appearance')
    return hashlib.sha256(names+values.numpy().tobytes()).hexdigest()


def initialize_shared_fresh_appearance(first, second, camera_name_to_uid,
                                       policy):
    if policy != 'fresh_seed0_pgsr_frozen':
        raise ValueError('appearance_policy_not_confirmed')
    if sorted(camera_name_to_uid.values()) != list(range(len(camera_name_to_uid))):
        raise ValueError('nonbijective_camera_uid_map')
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    first.create_app_model(len(camera_name_to_uid), 3)
    if first.app_model.name != 'PGSR':
        raise ValueError('wrong_appearance_model')
    a = first._appearance_embeddings
    if a is None or a.shape != (len(camera_name_to_uid), 2):
        raise ValueError('appearance_shape_or_camera_map_mismatch')
    second.app_model = first.app_model
    second._appearance_embeddings = torch.nn.Parameter(a.detach().clone(), requires_grad=False)
    second.appearance_network = None
    a.requires_grad_(False)
    result = appearance_digest(a, camera_name_to_uid)
    if result != appearance_digest(second._appearance_embeddings, camera_name_to_uid):
        raise ValueError('appearance_copy_failed')
    return result


def fresh_one_step_optimizer(parameters, group_lr):
    if set(parameters) != set(RAW_GROUPS) or set(group_lr) != set(RAW_GROUPS):
        raise ValueError('wrong_raw_parameter_groups')
    if any(lr <= 0 or not torch.isfinite(torch.tensor(lr)) for lr in group_lr.values()):
        raise ValueError('invalid_group_lr')
    groups = [dict(params=[parameters[name]], lr=float(group_lr[name])*.1, name=name)
              for name in RAW_GROUPS]
    optimizer = torch.optim.Adam(groups, betas=(.9, .999), eps=1e-15)
    if optimizer.state:
        raise AssertionError('optimizer_must_start_empty')
    return optimizer


def host_7001_group_lr(gaussians, opt):
    """Read host schedule after training_setup, before applying the fixed .1 factor."""
    if not hasattr(gaussians, 'xyz_scheduler_args'):
        raise ValueError('host_xyz_schedule_not_initialized')
    return dict(xyz=float(gaussians.xyz_scheduler_args(STEP)),
                scaling=float(opt.scaling_lr), rotation=float(opt.rotation_lr),
                opacity=float(opt.opacity_lr))


def mask_raw_grads(parameters, union_ids, total_rows):
    if (union_ids.ndim != 1 or union_ids.dtype != torch.long or
            bool((union_ids < 0).any()) or bool((union_ids >= total_rows).any()) or
            union_ids.numel() != torch.unique(union_ids).numel()):
        raise ValueError('invalid_union_ids')
    for name in RAW_GROUPS:
        value = parameters[name]
        if value.shape[0] != total_rows or value.grad is None:
            raise ValueError('missing_full_raw_gradient_' + name)
        if not bool(torch.isfinite(value.grad).all()):
            raise ValueError('nonfinite_full_raw_gradient_' + name)
        keep = torch.zeros(total_rows, dtype=torch.bool, device=value.device)
        keep[union_ids.to(value.device)] = True
        value.grad[~keep] = 0


def require_valid_patch_report(report):
    # The actual fixed scan110 preflight currently fails this gate for both
    # centers. No one-step update is permitted until a reviewed branch rule
    # supplies a valid two-center report.
    rays = report.get('rays', {})
    if set(map(int, rays)) != {17, 24} or any(
            row.get('status') != 'valid_local_patch' or
            row.get('branch_revalidation_required_after_update') is not True
            for row in rays.values()):
        raise ValueError('fixed_local_patch_not_valid')
