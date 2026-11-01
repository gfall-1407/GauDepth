"""Same-support four-substep root tracking for a single 3x3 camera patch.

This computes only a local reference-pixel depth and normal. It does not edit
the full depth image, accept a competing root, or claim a visible surface.
"""
import torch
import torch.nn.functional as F

from raw_param_bridge import DTYPE, root_from_shared


def patch_root_normal(arrays, row, raw, selected_root):
    query = row['query']
    center = query // 9
    if query != center*9+4:
        raise ValueError('fixed_query_is_not_3x3_center')
    target_xy = torch.tensor(arrays['xy'][center*9:center*9+9], dtype=DTYPE)
    center_xy = target_xy[4]
    expected_grid = torch.stack([
        torch.stack((center_xy[0]+(j%3-1), center_xy[1]+(j//3-1)))
        for j in range(9)])
    if not torch.equal(target_xy, expected_grid):
        raise ValueError('capture_is_not_full_3x3_grid')
    depths = []
    steps = {}
    for j in range(9):
        # Four equal coordinate substeps from the center for every neighbor.
        if j == 4:
            z, _, _, _ = root_from_shared(arrays, row, raw, selected_root)
            depths.append(z)
            steps[j] = [float(z.detach())]
            continue
        trace = []
        for k in range(1, 5):
            xy = center_xy + (target_xy[j]-center_xy)*(k/4)
            try:
                z, _, _, _ = root_from_shared(arrays, row, raw, selected_root, xy_override=xy)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f'root_tracking_failed:center={center},neighbor={j},'
                                 f'substep={k},xy={xy.tolist()},cause={exc}') from exc
            if not bool(torch.isfinite(z)) or not row['bracket'][0] < float(z.detach()) < row['bracket'][1]:
                raise ValueError('local_root_left_fixed_bracket')
            trace.append(z)
        depths.append(trace[-1])
        steps[j] = [float(value.detach()) for value in trace]
    z = torch.stack(depths).reshape(3, 3)
    fx, fy, cx, cy = torch.tensor(arrays['intrinsics'], dtype=DTYPE)
    xy = target_xy.reshape(3, 3, 2)
    points = torch.stack(((xy[..., 0]-cx)/fx*z,
                          (xy[..., 1]-cy)/fy*z, z), dim=0)
    dy = points[:, 2, 1]-points[:, 0, 1]
    dx = points[:, 1, 2]-points[:, 1, 0]
    normal = F.normalize(torch.cross(dy, dx, dim=0), dim=0)
    if not bool(torch.isfinite(normal).all()):
        raise ValueError('nonfinite_patch_normal')
    return z[1, 1], normal, dict(steps=steps,
                                  patch_depth=z.detach().tolist(),
                                  chosen_ids=row['ids'],
                                  bracket=row['bracket'],
                                  branch_revalidation_required_after_update=True)
