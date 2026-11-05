# Unimodal visibility-contribution regularizer

This experimental regularizer penalizes multiple peaks in the **accepted Gaussian
visibility-contribution sequence** at a pixel. It is not a depth variance loss,
not a point-cloud surface extraction method, and does not guarantee a single
geometric surface.

## Definition and ordering

Use the renderer's existing sorted tile list and its existing per-pixel acceptance
rules. For accepted Gaussians in that order,

- `T_i = product_{j<i} (1-alpha_j)`;
- `w_i = T_i * alpha_i`;
- prepend and append zero: `w_0 = w_{n+1} = 0`;
- `V = sum_{i=0}^n abs(w_{i+1}-w_i)`;
- `M = max_i w_i`;
- `E = max(0, V-2*M)`;
- `L_ray = E / (2*M + epsilon)`.

The nonnegative clamp only guards floating-point cancellation: in exact
arithmetic, `V >= 2*M`. A nonnegative single-peaked sequence, including a flat
peak, has `V=2*M` and zero loss. A rise after a decline creates positive excess
variation. Peak normalization is differentiable; its derivative is not detached.
The loss is not bounded by one and does not directly constrain peak width or
separation in metric depth.

**Ordering caveat:** the current rasterizer sorts by Gaussian-center camera
Euclidean distance (`norm3df(p_view.x, p_view.y, p_view.z)`), not by the exact
per-pixel ray-plane depth. This change preserves that existing order. Calling the
sequence “depth ordered” must be understood in that renderer-specific sense.
Sorting and visibility decisions are discrete and are held fixed in backward.

## Visibility, clipping, and validity

The loss uses exactly the contributors accepted for image compositing:

1. Reject positive Gaussian exponent (`power > 0`).
2. Set `alpha = min(0.99, conic_opacity.w * exp(power))`.
3. Skip `alpha < 1/255` without changing transmittance or inserting a zero sample.
4. If `T * (1-alpha) < 1e-4`, terminate **before accepting that Gaussian**.
5. Otherwise append `w = T*alpha` and update transmittance.

A ray is valid only if it has an accepted contributor and its accumulated mass
`1-T_final` meets `unimodal_min_mass` (default `0.5`). Empty or insufficient-mass
rays emit zero loss and zero gradient. The hard validity gate is not differentiated.
The training loss is the mean over valid rays, with denominator clamped to at
least one, rather than a mean over all image pixels. There is no ground-truth
foreground mask in this definition. Because the gate is based on predicted mass,
a ray can fall below the threshold and stop receiving this regularizer. This
term alone does not prevent opacity collapse; the image and other training
objectives must provide that constraint.

## Full backward

Let `g_i = dL_ray/dw_i`, let `s(x)` be `-1`, `0`, or `1` according to its sign,
and let `m_i` be one only at the earliest accepted maximum. For positive `E`,

```
dV/dw_i = s(w_i-w_{i-1}) + s(w_i-w_{i+1})
D = 2*M + epsilon
g_i = (dV/dw_i - 2*m_i)/D - 2*E*m_i/D^2

dL_ray/dalpha_i = T_i*g_i - sum_{j>i}(g_j*w_j)/(1-alpha_i)
```

The suffix term is essential: changing one alpha changes the transmittance of
**every later accepted contribution**. The implementation propagates this
additional alpha derivative through the existing Gaussian opacity, projected
mean/conic, and preprocessing backward paths. Contributions skipped or excluded
by termination receive no gradient from this loss.

Explicit nonsmooth conventions:

- `abs(0)` uses derivative zero.
- Equal maxima use the first maximum in forward order, not equal splitting.
- `E <= 0` uses zero derivative, including the clamp boundary.
- The new regularizer's alpha derivative is zero when the pre-clipped alpha is
  `>=0.99`, including the clipping boundary.
- No derivative crosses sorting, culling, acceptance, termination, or mass-gate
  boundaries. Central finite differences must avoid these boundaries.

## Enablement and compatibility

`lambda_unimodal` defaults to zero. With zero weight, training does not request
this map and does not add a zero-weight expression to the objective. Existing
render outputs and the disabled six-output Python rasterizer interface remain
available. Enabling the option adds two outputs: the differentiable loss map and
a nondifferentiable validity map, both `[1,H,W]`.

Training options:

```
--lambda_unimodal <nonnegative weight>
--unimodal_min_mass 0.5
--unimodal_epsilon 1e-6
```

Choose a weight experimentally; no recommended weight has been validated by a
training experiment. The regularizer is active from the first iteration when
its weight is positive. Default-disabled behavior is intended to preserve the
baseline; the compiled extension must be rebuilt after installing this change.
The Python extension API and its C++ implementation must come from the same patch.

## Memory and runtime costs

When enabled, the image-state buffer reserves four float planes (16 bytes per
pixel) for statistics. Disabled rendering allocates no extra image-state planes.
When enabled, forward
collects statistics in the existing render pass; backward performs two additional
forward traversals with one pending neighbor per thread. Recomputing the weights
in forward order preserves exact weight ties for the abs subgradient; reconstructing
transmittance by division in reverse can perturb those ties. The first replay sums
`g_i*w_i`; the second subtracts the current prefix to obtain the future suffix. This avoids per-contributor sequence storage, but uses
direct atomic gradient accumulation and has not been performance-profiled.
It can be materially slower for dense tile lists. Disabled behavior preserves the original image-state allocation layout and
numerical output path.

## Focused tests

From the repository root:

```
python -m unittest discover -s tests -p 'test_unimodal_regularizer.py' -v
```

The standard-library suite checks single/flat peaks, two peaks, low-mass gating,
explicit subgradient choices, clipping/skips/termination, and analytic backward
against central finite differences, including 100 seeded random rays. Another 100 seeded randomized rays compare
the two forward pending-neighbor replays against the full analytic derivative, with
skips, alpha caps, and early termination. The
PyTorch CPU test compares a separate autodiff expression against the full alpha
derivative. The optional compiled-CUDA test constructs a tiny four-Gaussian
scene, checks enabled/disabled output equality, the six/eight-output contract,
opacity finite differences, off-axis mean/scale finite differences with depth
outputs disabled, empty/low-mass zero maps and gradients, and depth-output mode
consistency.

Dependency skips are explicit. Passing the mathematical tests does **not** prove
CUDA compilation, numerical agreement of the compiled kernel, optimizer behavior,
or reconstruction quality. Run the optional tests after rebuilding in the actual
CUDA/PyTorch environment, before launching any larger experiment. Sorting and
hard-threshold behavior remain limitations even after those tests pass.
