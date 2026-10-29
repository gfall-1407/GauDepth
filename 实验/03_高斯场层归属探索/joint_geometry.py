import json, subprocess
from pathlib import Path
import numpy as np

out = Path(__file__).parent / '结果/20261029_联合几何/joint_geometry.json'
eps = np.finfo(np.float64).eps
xy = [(x, y) for x in (-1., -.5, .5, 1.) for y in (-1., 0., 1.)]
r = np.array([0., 0., 1.])
t = np.linspace(0., 4., 101)

def geometry(points, weights):
    p = weights / weights.sum()
    mean = p @ points
    delta = points - mean
    C = (delta * p[:, None]).T @ delta
    eig, vec = np.linalg.eigh(C)
    n = vec[:, 0]
    denom = float(n @ r)
    D = float(n @ mean / denom) if abs(denom) > 128 * eps else None
    residual = float(p @ (delta @ n) ** 2)
    return dict(eigenvalues=eig.tolist(), residual=residual, denominator=denom, intersection=D)

def alpha_weights(points):
    alpha = .1 * np.exp(-.5 * (points[:, 0]**2 + points[:, 1]**2))
    T = np.concatenate(([1.], np.cumprod(1 - alpha[:-1])))
    return alpha * T

scenes = []
profiles = []
checks = {}
for S in (2., 2.2):
    points = np.array([(x, y, S) for x, y in xy])
    L = max(1., float(np.max(np.abs(points))))
    tol, tol2 = 128 * eps * L, 128 * eps * L**2
    matrices, curves, params = [], [], []
    for x, y in xy:
        beta = (2. - S) / x
        Q = np.array([[1 + beta**2, 0., beta], [0., 1., 0.], [beta, 0., 1.]])
        mu = np.array([x, y, S])
        a = float(r @ Q @ r)
        peak = float(r @ Q @ mu / a)
        amp = float(.1 * np.exp(-.5 * (mu @ Q @ mu - a * peak**2)))
        matrices.append(float(np.linalg.eigvalsh(Q)[0]))
        params.append((peak, a, amp))
        diff = t[:, None] * r - mu
        curves.append(.1 * np.exp(-.5 * np.einsum('ij,jk,ik->i', diff, Q, diff)))
    params, curves = np.array(params), np.array(curves)
    expected_amp = .1 * np.exp(-.5 * (points[:, 0]**2 + points[:, 1]**2))
    checks[f'positive_definite_{S}'] = min(matrices) > 0
    checks[f'analytic_parameters_{S}'] = bool(np.max(np.abs(params - np.column_stack((np.full(12, 2.), np.ones(12), expected_amp)))) <= tol)
    g = geometry(points, alpha_weights(points))
    checks[f'plane_{S}'] = bool(g['eigenvalues'][1] > tol2 and abs(g['eigenvalues'][0]) <= tol2 and g['residual'] <= tol2 and g['intersection'] is not None and abs(g['intersection'] - S) <= tol)
    scenes.append(dict(S=S, min_Q_eigenvalue=min(matrices), geometry=g, length_tolerance=tol, squared_tolerance=tol2))
    profiles.append(curves)
checks['identical_profiles'] = bool(np.max(np.abs(profiles[0] - profiles[1])) <= 128 * eps * 2.2)
points = np.array([(x, y, S) for S in (2., 2.2) for x, y in xy])
mix = geometry(points, alpha_weights(points))
tol2 = 128 * eps * 2.2**2
checks['mixture_rejected'] = bool(mix['eigenvalues'][0] > tol2 and mix['residual'] > tol2)
line = np.array([(x, 0., 2.) for x in (-1., -.5, .5, 1.)])
collinear = geometry(line, alpha_weights(line))
checks['collinear_normal_nonunique'] = bool(abs(collinear['eigenvalues'][1]) <= 128 * eps * 2.**2 and collinear['eigenvalues'][2] > 128 * eps * 2.**2)
result = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), checks=checks, passed=all(checks.values()), max_profile_difference=float(np.max(np.abs(profiles[0]-profiles[1]))), scenes=scenes, mixture=mix, collinear=collinear, note='Synthetic ordered weights; mixture is not a certified surface; collinear normal is nonunique.')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result['passed'] else 1)
