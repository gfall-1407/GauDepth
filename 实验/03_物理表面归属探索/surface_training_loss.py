"""Fixed-pair, reference-assisted mechanism probe, not autonomous layer training.

Replicates RaDe-GS affine ray-plane depth, not an ellipsoid intersection.
Pair IDs, teacher surface depths and teacher weights are fixed outside this
module; gradients update current geometry only. No host CUDA extension imports.
"""
import torch


def paired_depth(xyz, scales, rotation, world_to_camera, fx, fy,
                 width, height, pixels):
    """Depth z for N paired Gaussian/pixel samples; all inputs may be CUDA.

    xyz/scales: (N,3), scales already activated WITH the 3D filter.
    rotation: (N,4) normalized wxyz quaternion, as passed to host CUDA.
    world_to_camera: conventional (4,4), COLUMN-vector transform; pass
    camera.world_view_transform.T, not the transposed host storage matrix.
    pixels: (N,2), image pixel coordinates, with centred host principal point.
    Sort, opacity and membership are fixed externally and not recomputed here.
    Float differences from CUDA fast math are expected; validate real pairs.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or scales.shape != xyz.shape:
        raise ValueError('xyz and scales must both have shape (N,3)')
    if rotation.shape != (len(xyz), 4):
        raise ValueError('rotation must have shape (N,4)')
    kwargs = dict(device=xyz.device, dtype=xyz.dtype)
    view = torch.as_tensor(world_to_camera, **kwargs)
    pixels = torch.as_tensor(pixels, **kwargs)
    if view.shape != (4, 4) or pixels.shape != (len(xyz), 2):
        raise ValueError('world_to_camera=(4,4) and pixels=(N,2) required')
    fx, fy = torch.as_tensor(fx, **kwargs), torch.as_tensor(fy, **kwargs)
    p = xyz @ view[:3, :3].T + view[:3, 3]
    # tc uses UNCLAMPED camera centre; slope computation uses clamped p.
    tc = torch.linalg.vector_norm(p, dim=-1)
    tz = p[:, 2]
    u = torch.clamp(p[:, 0] / tz, min=-1.3*width/(2*fx), max=1.3*width/(2*fx))
    v = torch.clamp(p[:, 1] / tz, min=-1.3*height/(2*fy), max=1.3*height/(2*fy))
    r, x, y, z = rotation.unbind(-1)
    Q = torch.stack((1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                     2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                     2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    # CUDA's GLM constructors are column-major: R_GLM=Q.T; W=c2w.
    RW = Q.transpose(-1, -2) @ view[:3, :3].T
    bigger = (scales[:, 0] < scales[:, 1]).long()
    i0, i1 = (bigger+1) % 3, (bigger+2) % 3
    row = torch.arange(len(xyz), device=xyz.device)
    mi = torch.where(scales[row, i0] > scales[row, i1], i1, i0)
    well = scales[row, mi] > 1e-7
    # The ill-conditioned CUDA branch uses an outer product of the min axis.
    # Avoid division by zero in an unselected branch of torch.where.
    safe_scales = torch.where(well[:, None], scales, torch.ones_like(scales))
    A = RW / safe_scales[:, :, None]
    inverse = A.transpose(-1, -2) @ A
    axis = RW[row, mi]
    inverse = torch.where(well[:, None, None], inverse,
                          axis[:, :, None]*axis[:, None, :])
    uvh = torch.stack((u, v, torch.ones_like(u)), -1)
    uvhm = (inverse @ uvh[:, :, None]).squeeze(-1)
    length = torch.linalg.vector_norm(uvhm, dim=-1, keepdim=True)
    valid = (length[:, 0] > 0) & torch.isfinite(length[:, 0])
    nm = uvhm / torch.where(valid[:, None], length, torch.ones_like(length))
    bn = nm / (nm*uvh).sum(-1, keepdim=True).clamp_min(1e-7)
    px = (v*v+1)*bn[:, 0] - u*v*bn[:, 1] - u*bn[:, 2]
    py = -u*v*bn[:, 0] + (u*u+1)*bn[:, 1] - v*bn[:, 2]
    factor = torch.sqrt((u*tz)**2+(v*tz)**2+tz*tz)/(u*u+v*v+1)
    # Project the actual centre, NOT the clamped coordinates. CUDA p_w uses eps.
    ndcx = (2*fx/width)*p[:, 0]/(tz+1e-7)
    ndcy = (2*fy/height)*p[:, 1]/(tz+1e-7)
    xy = torch.stack((((ndcx+1)*width-1)*.5, ((ndcy+1)*height-1)*.5), -1)
    delta = xy-pixels
    t = px*factor/fx*delta[:, 0] + py*factor/fy*delta[:, 1] + tc
    t = torch.where(valid, t, torch.zeros_like(t))
    nx = (pixels[:, 0]-(width-1)/2)/fx
    ny = (pixels[:, 1]-(height-1)/2)/fy
    return t / torch.sqrt(nx*nx+ny*ny+1)


def fixed_surface_loss(d, target, c, group):
    """Return (loss, stats) for fixed teacher assignments and detached weights.

    Each group must denote ONE (pixel, physical surface) with constant target z.
    loss = mean_g sum_i c_i (d_i-z_g)^2 / sum_i c_i. Zero-mass groups omitted.
    stats['mean_loss'] + stats['variance'] equals loss up to floating roundoff.
    These remain differentiable; stats['decomposition_error'] is diagnostic.
    No opacity input exists, preventing this loss from lowering teacher weights.
    """
    if d.ndim != 1:
        raise ValueError('d must be a flat vector')
    kw = dict(device=d.device, dtype=d.dtype)
    target = torch.as_tensor(target, **kw).detach()
    c = torch.as_tensor(c, **kw).detach()
    group = torch.as_tensor(group, device=d.device, dtype=torch.long).detach()
    if target.shape != d.shape or c.shape != d.shape or group.shape != d.shape:
        raise ValueError('d, target, c, group must have identical shape')
    if not bool(torch.isfinite(c).all() & (c >= 0).all() & torch.isfinite(target).all()):
        raise ValueError('Teacher weights must be finite/nonnegative; targets finite')
    if not len(d):
        zero = d.sum()*0
        return zero, dict(mean_loss=zero, variance=zero, decomposition_error=zero,
                          groups=0, pairs=0)
    labels, inv = torch.unique(group, sorted=True, return_inverse=True)
    def sums(value):
        return torch.zeros(len(labels), **kw).scatter_add(0, inv, value)
    mass = sums(c)
    good = mass > 0
    denom = torch.where(good, mass, torch.ones_like(mass))
    zbar = sums(c*target)/denom
    # Reject an incorrectly merged teacher group instead of hiding target spread.
    if bool((c*(target-zbar[inv]).abs() > 1e-5*c).any()):
        raise ValueError('Each group must have one constant target surface depth')
    mu = sums(c*d)/denom
    per_pair = sums(c*(d-target).square())/denom
    mean_term = (mu-zbar).square()
    variance = sums(c*(d-mu[inv]).square())/denom
    if bool(good.any()):
        loss, mean_loss, var = per_pair[good].mean(), mean_term[good].mean(), variance[good].mean()
    else:
        loss = mean_loss = var = d.sum()*0
    return loss, dict(mean_loss=mean_loss, variance=var,
                      decomposition_error=loss-mean_loss-var,
                      groups=int(good.sum()), pairs=len(d))
