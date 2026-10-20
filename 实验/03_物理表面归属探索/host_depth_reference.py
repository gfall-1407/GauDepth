"""CPU diagnostic replay of RaDe-GS scale/rotation preprocessing and accepted rays.

Not bit exact CUDA: float32 BLAS, exp, sorting ties and threshold roundoff can
change membership. Compare cached GPU expected/median before trusting a ray.
Inputs default to raw PLY log-scales/logit-opacity and wxyz quaternion. Camera
is cameras.json: rotation=camera-to-world, position=world camera centre, and
width/height/fx/fy at the original resolution. No host CUDA imports required.
"""
import numpy as np


def prepare_gaussians(xyz, scales, rotation, opacity, filter3d, camera,
                      kernel=0.0, resolution=2, activated=False):
    """Return float32 arrays for trace_pixel; activated=True accepts linear s,o.

    Mirrors computeCov2D's scale/rotation branch, including clamp coordinates,
    minimum-scale fallback, mip opacity factors, tile rectangle and radial sort.
    Only standard camera-centred projection, scale_modifier=1 is supported.
    """
    xyz = np.asarray(xyz, np.float32)
    scales = np.asarray(scales, np.float32)
    rotation = np.asarray(rotation, np.float32)
    opacity = np.asarray(opacity, np.float32).reshape(-1)
    n = len(xyz)
    filt = np.zeros((n, 1), np.float32) if filter3d is None else np.asarray(filter3d, np.float32).reshape(n, 1)
    if xyz.shape != (n, 3) or scales.shape != (n, 3) or rotation.shape != (n, 4) or len(opacity) != n:
        raise ValueError('Expected xyz/scales=(N,3), rotation=(N,4), opacity=(N,)')
    if float(resolution) <= 0 or kernel < 0:
        raise ValueError('resolution must be positive and kernel nonnegative')
    width, height = round(camera['width'] / resolution), round(camera['height'] / resolution)
    fx = np.float32(camera['fx'] * width / camera['width'])
    fy = np.float32(camera['fy'] * height / camera['height'])
    cx, cy = np.float32((width - 1) / 2), np.float32((height - 1) / 2)
    c2w = np.asarray(camera['rotation'], np.float32)
    center = np.asarray(camera['position'], np.float32)
    grid = np.array([(width + 15) // 16, (height + 15) // 16])
    out = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
               center=center, c2w=c2w, diagnostic_only=True,
               xy=np.empty((n, 2), np.float32), conic=np.empty((n, 3), np.float32),
               plane=np.empty((n, 3), np.float32), opacity=np.empty(n, np.float32),
               sort_depth=np.empty(n, np.float32), radius=np.zeros(n, np.int32),
               rect_min=np.zeros((n, 2), np.int32), rect_max=np.zeros((n, 2), np.int32),
               valid=np.zeros(n, bool))
    # Column-major GLM R in CUDA equals transpose of the usual wxyz rotation.
    # W in computeCov2D equals c2w. J in CUDA equals J_standard.T.
    for start in range(0, n, 65536):
        end = min(start + 65536, n)
        sl = slice(start, end)
        s = scales[sl] if activated else np.exp(scales[sl])
        o = opacity[sl] if activated else np.exp(-np.logaddexp(np.float32(0), -opacity[sl]))
        ss = s * s
        sf = np.sqrt(ss + filt[sl] * filt[sl])
        of = o * np.sqrt(np.prod(ss, axis=1) / np.prod(sf * sf, axis=1))
        q = rotation[sl]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        r, x, y, z = q.T
        Q = np.stack((1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
                      2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
                      2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
        # Camera rows: world row vectors multiplied by c2w.
        p = (xyz[sl] - center) @ c2w
        tc = np.linalg.norm(p, axis=1)
        out['sort_depth'][sl] = tc
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            u = np.clip(p[:, 0] / p[:, 2], -1.3 * width / (2*fx), 1.3 * width / (2*fx))
            v = np.clip(p[:, 1] / p[:, 2], -1.3 * height / (2*fy), 1.3 * height / (2*fy))
            tz = p[:, 2]
            J = np.zeros((end-start, 3, 3), np.float32)
            J[:, 0, 0], J[:, 1, 1] = fx/tz, fy/tz
            J[:, 2, 0], J[:, 2, 1] = -fx*u/tz, -fy*v/tz
            # M = S * R_GLM * W * J_GLM.
            R = np.swapaxes(Q, 1, 2)
            RW = R @ c2w
            M = (RW @ J) * sf[:, :, None]
            cov = np.swapaxes(M, 1, 2) @ M
            a0, b, c0 = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
            a, c = a0 + np.float32(kernel), c0 + np.float32(kernel)
            det = a*c - b*b
            coef = np.sqrt(np.maximum(np.float32(1e-6), a0*c0-b*b) /
                           np.maximum(np.float32(1e-6), det))
            out['conic'][sl] = np.stack((c/det, -b/det, a/det), axis=1)
            out['opacity'][sl] = of * coef
            # True inverse covariance except the CUDA minimum-scale fallback.
            A = RW / sf[:, :, None]
            cov_inv = np.swapaxes(A, 1, 2) @ A
            # Match CUDA find_min_from_triple tie handling, not np.argmin.
            bigger = (sf[:, 0] < sf[:, 1]).astype(np.int32)
            i0, i1 = (bigger+1) % 3, (bigger+2) % 3
            row = np.arange(end-start)
            mi = np.where(sf[row, i0] > sf[row, i1], i1, i0)
            well = sf[np.arange(end-start), mi] > np.float32(1e-7)
            normal_axis = RW[np.arange(end-start), mi]
            cov_inv[~well] = (normal_axis[:, :, None] * normal_axis[:, None, :])[~well]
            uvh = np.stack((u, v, np.ones_like(u)), axis=1)
            uvhm = np.einsum('nij,nj->ni', cov_inv, uvh)
            nm = uvhm / np.linalg.norm(uvhm, axis=1, keepdims=True)
            vbn = np.sum(nm * uvh, axis=1)
            bn = nm / np.maximum(vbn[:, None], np.float32(1e-7))
            px = (v*v+1)*bn[:, 0] - u*v*bn[:, 1] - u*bn[:, 2]
            py = -u*v*bn[:, 0] + (u*u+1)*bn[:, 1] - v*bn[:, 2]
            # CUDA l is clamped vector length, tc is UNCLAMPED centre distance.
            factor = np.sqrt((u*tz)**2 + (v*tz)**2 + tz*tz) / (u*u+v*v+1)
            plane = np.stack((px*factor/fx, py*factor/fy, tc), axis=1)
            plane[np.isnan(nm[:, 0])] = 0
            out['plane'][sl] = plane
            # p_w includes the same 1e-7 used by preprocess projection.
            ndcx = (2*fx/width) * p[:, 0] / (tz + np.float32(1e-7))
            ndcy = (2*fy/height) * p[:, 1] / (tz + np.float32(1e-7))
            xy = np.stack((((ndcx+1)*width-1)*.5, ((ndcy+1)*height-1)*.5), axis=1)
            out['xy'][sl] = xy
            mid = np.float32(.5)*(a+c)
            lam = mid + np.sqrt(np.maximum(np.float32(.1), mid*mid-det))
            radius_f = np.ceil(3*np.sqrt(lam))
            finite = np.isfinite(radius_f) & np.isfinite(xy).all(axis=1)
            radius = np.where(finite, radius_f, 0).astype(np.int32)
            # C++ int cast truncates toward zero, including offscreen negatives.
            safe_xy = np.where(np.isfinite(xy), xy, 0)
            rect_min = np.clip(np.trunc((safe_xy-radius[:, None])/16), 0, grid).astype(np.int32)
            rect_max = np.clip(np.trunc((safe_xy+radius[:, None]+15)/16), 0, grid).astype(np.int32)
            valid = ((tz > .2) & (det != 0) & finite & (radius > 0) &
                     np.all(rect_max > rect_min, axis=1) & np.isfinite(out['conic'][sl]).all(axis=1))
            out['radius'][sl] = np.where(valid, radius, 0)
            out['rect_min'][sl], out['rect_max'][sl] = rect_min, rect_max
            out['valid'][sl] = valid
    out['order'] = np.argsort(out['sort_depth'], kind='stable')
    return out


def trace_pixel(prepared, x, y):
    """Replay one integer pixel; arrays contain only originally accepted items.

    d = camera z; radial_d = t; T = transmittance BEFORE each accepted item.
    Early-stop item is excluded, just as render_forward.cu. median uses T>.5
    before acceptance, NOT maximum weight or a weighted-median re-sort.
    """
    p = prepared
    if int(x) != x or int(y) != y or not (0 <= x < p['width'] and 0 <= y < p['height']):
        raise ValueError('trace_pixel expects an in-image integer pixel')
    tile = np.array([x//16, y//16])
    mask = p['valid'] & np.all(p['rect_min'] <= tile, axis=1) & np.all(tile < p['rect_max'], axis=1)
    ids = p['order'][mask[p['order']]]
    delta = p['xy'][ids] - np.array([x, y], np.float32)
    co = p['conic'][ids]
    power = np.float32(-.5)*(co[:, 0]*delta[:, 0]**2 + co[:, 2]*delta[:, 1]**2) - co[:, 1]*delta[:, 0]*delta[:, 1]
    alpha = np.minimum(np.float32(.99), p['opacity'][ids]*np.exp(power))
    rln = np.float32(1/np.sqrt(((x-p['cx'])/p['fx'])**2 + ((y-p['cy'])/p['fy'])**2 + 1))
    T, accum, median = np.float32(1), np.float32(0), np.float32(0)
    accepted, ds, ws, aa, ts = [], [], [], [], []
    stopped_id = None
    for j, gid in enumerate(ids):
        if power[j] > 0 or alpha[j] < np.float32(1/255):
            continue
        a = alpha[j]
        test_T = np.float32(T * (1-a))
        if test_T < np.float32(.0001):
            stopped_id = int(gid)
            break
        w = np.float32(a*T)
        t = np.float32(np.dot(p['plane'][gid, :2], delta[j]) + p['plane'][gid, 2])
        accum = np.float32(accum + t*w)
        if T > .5:
            median = t
        accepted.append(gid); ds.append(t); ws.append(w); aa.append(a); ts.append(T)
        T = test_T
    radial = np.asarray(ds, np.float32)
    return dict(ids=np.asarray(accepted, np.int64), d=radial*rln, radial_d=radial,
                w=np.asarray(ws, np.float32), alpha=np.asarray(aa, np.float32),
                T=np.asarray(ts, np.float32), final_T=float(T), mask=float(1-T),
                expected=float(accum*rln/(1-T)) if accepted else 0.,
                median=float(median*rln), rln=float(rln), stopped_id=stopped_id,
                diagnostic_only=True)
