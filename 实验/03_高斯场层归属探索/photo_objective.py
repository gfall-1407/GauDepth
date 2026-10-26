"""Pure NumPy plane-induced observed-image 1-ZNCC costs; no depth selection.

Camera K maps camera coordinates to NumPy integer pixel centres. A caller with
COLMAP corner-coordinate intrinsics must resize K first, then subtract .5 from
cx and cy once before calling. No image recentering is guessed here. Invalid geometry,
out-of-image patches and textureless patches produce unknown (NaN) costs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

VARIANCE_EPS = 64. * np.finfo(np.float64).eps
DENOMINATOR_EPS = 1e-12


def _gray(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("Expected a nonempty grayscale [H,W] image")
    return image


def _camera(camera):
    K = np.asarray(camera["K"], dtype=np.float64)
    c2w = np.asarray(camera["c2w"], dtype=np.float64)
    w2c = np.asarray(camera["w2c"], dtype=np.float64)
    size = tuple(camera["image_size"])
    if K.shape != (3, 3) or c2w.shape != (4, 4) or w2c.shape != (4, 4):
        raise ValueError("Expected K[3,3] and c2w/w2c[4,4]")
    if len(size) != 2 or any(int(x) != x or x <= 0 for x in size):
        raise ValueError("Expected positive integer camera image_size(H,W)")
    if not all(np.isfinite(x).all() for x in (K, c2w, w2c)):
        raise ValueError("Camera matrices must be finite")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0., 0., 1.], rtol=0, atol=1e-12):
        raise ValueError("Expected positive focal lengths and standard pinhole K")
    if not np.allclose(c2w[3], [0., 0., 0., 1.], rtol=0, atol=1e-9) or not np.allclose(w2c[3], [0., 0., 0., 1.], rtol=0, atol=1e-9):
        raise ValueError("Expected rigid homogeneous camera matrices")
    if not np.allclose(w2c @ c2w, np.eye(4), rtol=1e-6, atol=1e-6):
        raise ValueError("Camera c2w and w2c are inconsistent")
    return K, c2w, w2c, tuple(map(int, size))


def _offsets(radius):
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("Patch radius must be a nonnegative integer")
    dy, dx = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), indexing="ij")
    return np.stack((dx, dy), axis=-1).reshape(-1, 2).astype(np.float64)


def _bounds(uv, image_size):
    h, w = image_size
    return (np.isfinite(uv).all(axis=-1) & (uv[..., 0] >= 0.) & (uv[..., 0] <= w-1)
            & (uv[..., 1] >= 0.) & (uv[..., 1] <= h-1))


def bilinear(image, uv):
    """Return (gray samples, coordinate bounds-valid), shaped uv.shape[:-1].

    The closed image boundary is valid; its repeated last neighbour is exactly
    the boundary value. Out-of-bounds/nonfinite coordinates return NaN. The bool
    describes coordinate bounds; nonfinite source pixels also yield NaN samples.
    """
    image = _gray(image)
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError("Expected coordinates [...,2] in (x,y) order")
    shape = uv.shape[:-1]
    flat = uv.reshape(-1, 2)
    valid = _bounds(flat, image.shape)
    samples = np.full(len(flat), np.nan, dtype=np.float64)
    if valid.any():
        points = flat[valid]
        x0, y0 = np.floor(points).astype(np.int64).T
        x1 = np.minimum(x0+1, image.shape[1]-1)
        y1 = np.minimum(y0+1, image.shape[0]-1)
        dx, dy = points[:, 0]-x0, points[:, 1]-y0
        samples[valid] = ((1.-dy)*((1.-dx)*image[y0, x0]+dx*image[y0, x1])
                          + dy*((1.-dx)*image[y1, x0]+dx*image[y1, x1]))
    return samples.reshape(shape), valid.reshape(shape)


def plane_warp(xy, depths, normals, ci, cj, radius=2):
    """Return (target uv[N,K,P,2], valid[N,K,P]) for a reference tangent plane.

    Depth is reference camera-z, in the same length unit as camera translations.
    Normal is reference-camera direction, not a world normal. Patch order is
    row-major dy then dx. Validity includes finite positive intersections and
    projections, a valid normal, and both images' coordinate bounds.
    """
    Ki, c2wi, _, size_i = _camera(ci)
    Kj, _, w2cj, size_j = _camera(cj)
    xy, depths, normals = (np.asarray(x, dtype=np.float64) for x in (xy, depths, normals))
    if xy.ndim != 2 or xy.shape[1] != 2 or depths.ndim != 2 or depths.shape[0] != len(xy) or depths.shape[1] < 1 or normals.shape != (len(xy), 3):
        raise ValueError("Expected xy[N,2], depths[N,K>=1], normals[N,3]")
    patch_xy = xy[:, None, :] + _offsets(radius)[None, :, :]
    inverse_K = np.linalg.inv(Ki)
    centre = np.concatenate((xy, np.ones((len(xy), 1))), axis=-1) @ inverse_K.T
    rays = np.concatenate((patch_xy, np.ones((*patch_xy.shape[:-1], 1))), axis=-1) @ inverse_K.T
    lengths = np.linalg.norm(normals, axis=-1)
    normal_valid = np.isfinite(normals).all(axis=-1) & np.isfinite(lengths) & (lengths > DENOMINATOR_EPS)
    unit = np.zeros_like(normals)
    unit[normal_valid] = normals[normal_valid]/lengths[normal_valid, None]
    denominator = np.einsum("nd,npd->np", unit, rays)
    constant = np.einsum("nd,nd->n", unit, centre)[:, None] * depths
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z_ref = constant[..., None] / denominator[:, None, :]
        local = z_ref[..., None] * rays[:, None, :, :]
        world = local @ c2wi[:3, :3].T + c2wi[:3, 3]
        target = world @ w2cj[:3, :3].T + w2cj[:3, 3]
        projection = target @ Kj.T
        uv = projection[..., :2] / projection[..., 2:]
    valid = (normal_valid[:, None, None] & np.isfinite(depths)[..., None] & (depths[..., None] > 0.)
             & (np.abs(denominator[:, None, :]) > DENOMINATOR_EPS)
             & np.isfinite(z_ref) & (z_ref > 0.) & np.isfinite(target).all(axis=-1)
             & (target[..., 2] > 0.) & _bounds(patch_xy, size_i)[:, None, :] & _bounds(uv, size_j))
    return uv, valid


def photo_cost(refgray, targetgray, xy, depths, normals, ci, cj, radius=2):
    """Return 1-ZNCC[N,K]; unknown patches stay NaN, with no fallback selection.

    Every patch pixel must be legal and finite in both observed images. Each
    population variance must exceed VARIANCE_EPS in supplied intensity units.
    Positive affine brightness/contrast changes cancel within numerical accuracy;
    contrast reversal is not treated as the same signal. Texture does not itself
    prove visibility or depth identifiability (e.g. a zero-baseline cost can be flat).
    """
    refgray, targetgray = _gray(refgray), _gray(targetgray)
    if refgray.shape != tuple(ci["image_size"]) or targetgray.shape != tuple(cj["image_size"]):
        raise ValueError("Observed gray image dimensions must match their cameras")
    xy = np.asarray(xy, dtype=np.float64)
    uv, warp_valid = plane_warp(xy, depths, normals, ci, cj, radius)
    reference_uv = xy[:, None, :] + _offsets(radius)[None, :, :]
    reference, ref_bounds = bilinear(refgray, reference_uv)
    target, target_bounds = bilinear(targetgray, uv)
    ref_finite, target_finite = np.isfinite(reference), np.isfinite(target)
    clean_ref = np.where(ref_finite, reference, 0.)
    clean_target = np.where(target_finite, target, 0.)
    ref_centred = clean_ref-clean_ref.mean(axis=-1, keepdims=True)
    target_centred = clean_target-clean_target.mean(axis=-1, keepdims=True)
    ref_variance = np.mean(ref_centred**2, axis=-1)
    target_variance = np.mean(target_centred**2, axis=-1)
    good = (warp_valid.all(axis=-1) & (ref_bounds & ref_finite).all(axis=-1)[:, None]
            & (target_bounds & target_finite).all(axis=-1)
            & (ref_variance[:, None] > VARIANCE_EPS) & (target_variance > VARIANCE_EPS))
    numerator = np.mean(ref_centred[:, None, :]*target_centred, axis=-1)
    denominator = np.sqrt(ref_variance[:, None]*target_variance)
    correlation = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=correlation, where=good)
    result = np.full_like(correlation, np.nan)
    good &= np.isfinite(correlation)
    result[good] = 1.-np.clip(correlation[good], -1., 1.)
    return result


def _synthetic_camera(centre=(0., 0., 0.), angle_y=0.):
    K = np.array([[140., 0., 80.], [0., 143., 60.], [0., 0., 1.]])
    c, s = np.cos(angle_y), np.sin(angle_y)
    c2w = np.eye(4)
    c2w[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    c2w[:3, 3] = centre
    return dict(K=K, c2w=c2w, w2c=np.linalg.inv(c2w), image_size=(121, 161))


def _render_synthetic_plane(camera, world_normal, constant, pure=None):
    """Analytical world-plane ray intersections; never used by the cost API."""
    K, c2w, _, (h, w) = _camera(camera)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    q = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(K).T
    direction = q @ c2w[:3, :3].T
    z = (constant-np.dot(world_normal, c2w[:3, 3])) / (direction @ world_normal)
    point = c2w[:3, 3]+z[..., None]*direction
    X, Y = point[..., 0], point[..., 1]
    if pure is None:
        image = (.5+.15*np.sin(10*X+4*Y+.2)+.12*np.cos(7*X-13*Y+.8)
                 +.09*np.sin(18*X+11*Y+1.2)+.06*np.cos(25*X-17*Y+.7))
    else:
        image = np.full((h, w), float(pure))
    image[(z <= 0.) | ~np.isfinite(z)] = np.nan
    return image


def self_check():
    ramp = np.arange(20, dtype=np.float64).reshape(4, 5)
    uv = np.array([[[1.25, 2.5], [4., 3.]], [[-1., 2.], [np.nan, 0.]]])
    values, bounds = bilinear(ramp, uv)
    np.testing.assert_allclose(values[0], [13.75, 19.], rtol=0, atol=1e-12)
    assert bounds.tolist() == [[True, True], [False, False]] and np.isnan(values[1]).all()
    ci = _synthetic_camera()
    plus, minus = _synthetic_camera((.5, 0., 0.)), _synthetic_camera((-.5, 0., 0.))
    for camera, expected_x in ((plus, 66.), (minus, 94.)):
        warped, valid = plane_warp([[80., 60.]], [[5.]], [[0., 0., 1.]], ci, camera, radius=0)
        assert valid.all()
        np.testing.assert_allclose(warped[0, 0, 0], [expected_x, 60.], rtol=0, atol=1e-12)
    # Camera K is already in NumPy integer-centre convention; do not shift it twice.
    K_corner = ci["K"].copy()
    K_corner[:2, 2] += .5
    K_numpy = K_corner.copy()
    K_numpy[:2, 2] -= .5
    pixel = np.array([80., 60., 1.])
    np.testing.assert_allclose(np.linalg.inv(K_numpy) @ pixel, [0., 0., 1.], rtol=0, atol=1e-14)
    assert not np.allclose(np.linalg.inv(K_corner) @ pixel, [0., 0., 1.], rtol=0, atol=1e-4)
    cameras = [ci, plus, _synthetic_camera((-.5, .08, 0.), angle_y=.035)]
    xy = np.array([[80., 60.], [65., 47.], [94., 72.], [74., 78.]])
    report = []
    for normal in (np.array([0., 0., 1.]), np.array([.24, -.15, 1.])):
        normal = normal/np.linalg.norm(normal)
        constant = 5.*normal[2]
        q = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(ci["K"]).T
        truth = constant/(q @ normal)
        depths = truth[:, None] * np.array([.8, 1., 1.2])[None, :]
        normals = np.broadcast_to(normal, (len(xy), 3)).copy()
        images = [_render_synthetic_plane(c, normal, constant) for c in cameras]
        same_uv, same_valid = plane_warp(xy, depths, normals, ci, ci)
        patch = xy[:, None, :]+_offsets(2)[None, :, :]
        assert same_valid.all()
        np.testing.assert_allclose(same_uv, np.broadcast_to(patch[:, None], same_uv.shape), rtol=0, atol=1e-12)
        reverse_uv, reverse_valid = plane_warp(xy, depths, -normals, ci, cameras[1])
        forward_uv, forward_valid = plane_warp(xy, depths, normals, ci, cameras[1])
        np.testing.assert_allclose(reverse_uv, forward_uv, rtol=0, atol=1e-12)
        assert np.array_equal(reverse_valid, forward_valid)
        for j in (1, 2):
            costs = photo_cost(images[0], images[j], xy, depths, normals, ci, cameras[j])
            assert np.isfinite(costs).all()
            assert np.all(costs[:, 1] < costs[:, 0]) and np.all(costs[:, 1] < costs[:, 2])
            affine_cost = photo_cost(images[0], images[j]*1.7+.23, xy, depths, normals, ci, cameras[j])
            np.testing.assert_allclose(costs, affine_cost, rtol=0, atol=2e-12)
            report.append(dict(plane_normal=normal.tolist(), neighbour=j, mean_costs=costs.mean(axis=0).tolist()))
        colourless = photo_cost(np.full(ci["image_size"], .3), np.full(ci["image_size"], .8),
            xy, depths, normals, ci, cameras[1])
        assert np.isnan(colourless).all()
        same_costs = photo_cost(images[0], images[0], xy, depths, normals, ci, ci)
        np.testing.assert_allclose(same_costs, 0., rtol=0, atol=2e-14)
    bad_depths = np.array([[5., -5., np.nan]])
    _, bad_valid = plane_warp([[80., 60.]], bad_depths, [[0., 0., 1.]], ci, plus)
    assert bad_valid[:, 0].all() and not bad_valid[:, 1:].any()
    _, parallel = plane_warp([[80., 60.]], [[5.]], [[0., 0., 0.]], ci, plus)
    assert not parallel.any()
    print(json.dumps(dict(status="PASS", synthetic_only=True, textured_plane_checks=report,
        pure_colour="all unknown/NaN", zero_baseline="flat equal costs, not a depth certificate",
        convention="K uses NumPy integer pixel centres; COLMAP corner cx/cy minus.5 once")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run deterministic analytical plane checks only")
    args = parser.parse_args()
    if not args.check:
        parser.error("This module only exposes pure functions and --check; no real-data runner")
    self_check()


if __name__ == "__main__":
    main()
