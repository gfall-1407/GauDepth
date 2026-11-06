# Equal depth segment unimodal regularizer

This revision replaces per-Gaussian weight unimodality from commit 344a16a.
It constrains opacity drops across equal-width depth segments. It does not
introduce continuous Gaussian attenuation or GGGS surface extraction.

## Definition

For K segments over [near, far], t_j = near + j(far-near)/K.
For the contributors actually accepted by the original rasterizer, define

    T_j = product_{i: d_i < t_j} (1 - alpha_i)
    m_k = T_k - T_{k+1},     k = 0, ..., K-1
    A   = T_0 - T_K
    p_k = m_k / A

Each d_i uses this repository's existing per-pixel ray-plane representative
rendered depth, including its ray normalization factor. It is not the Gaussian
list index, the tile sort key, or a continuous attenuation profile through a
Gaussian. All boundaries use the SAME original accepted contributor set,
including original alpha cutoff, alpha cap and early termination. Multiplying
the factors for each boundary is independent of the order of that set, even
when per-pixel representative depths differ from center-based tile ordering.
Consequently, these boundary values are a discrete, representative-depth
transmittance model. They are not claimed to equal GGGS continuous T(t).

Use half-open segments [t_k,t_{k+1}); a contributor exactly at far is excluded.
A contributor before near attenuates all boundary T values; one after far
attenuates none. Invalid nonfinite depths are ignored by this auxiliary term.
This is probability mass conditional on the selected interval, not a claim of
complete whole-ray mass. For equal widths its mode matches the average density.

With p padded by zero at both ends,

    V_p = sum |p_(k+1) - p_k|
    M_p = max p_k
    L   = max(0, V_p - 2 M_p) / (2 M_p + epsilon).

The CUDA arithmetic uses the EXACT raw-mass equivalent,

    L = max(0, V_m - 2 M_m) / (2 M_m + epsilon * A).

The factor A in the denominator must not be omitted. The mask requires at
least one accepted contributor, positive peak and interval mass A > 0 and
A >= unimodal_min_mass. The training script averages over valid pixels, using
zero if none are valid. A flat top and asymmetric unimodal shapes are allowed;
a zero auxiliary loss does not require a narrow or interior peak.

## Parameters

The auxiliary weight remains zero by default, preserving the original training
objective. When enabled, it requires depth preprocessing; the public renderer
requests this automatically even before other geometric losses start.

    --lambda_unimodal      0.0
    --unimodal_from_iter   0      # absolute iteration; enabled when iteration >= this value
    --unimodal_bins        8      # integer, 1 through 32
    --unimodal_near        0.0
    --unimodal_far         10.0
    --unimodal_min_mass    0.5
    --unimodal_epsilon     1e-6

The default range is a placeholder in the repository's rendered depth units,
NOT a validated range for DTU or every camera. Set it to cover the intended
foreground depth distribution before an experiment. A narrow range can hide
other peaks. Excessively coarse bins hide structure; overly fine bins return
toward the over-strict per-Gaussian constraint. Fixed bounds are treated as
constants, not trained variables. The independent unimodal_from_iter gate defaults to 0, preserving first-step
enablement when lambda_unimodal > 0. Before the gate the auxiliary renderer
and loss are disabled; TensorBoard unimodal scalars are logged only when active.
On checkpoint resume, the gate uses the absolute iteration number, not the
number of steps since resuming. Set it to 7000 as an experimental starting
point, not a validated optimum. Negative values are rejected.
On resume, checkpoints do not save these experiment options. Re-pass lambda_unimodal,
unimodal_from_iter, the bin count, and the intended depth range.

## CUDA forward and backward

Forward maintains K+1 boundary transmittances. Every accepted contribution
updates the boundaries behind its representative depth. After rasterization,
the segment masses produce stable excess, peak and the valid mask; colors and
existing depth outputs keep their original accumulation. Only four per-pixel statistics are cached,
not every Gaussian/pixel weight.

For nonnegative masses m_k = T_k - T_(k+1), the stable numerator E is twice
the sum of monotonicity violations around the first maximum k*: max(m_k -
m_(k+1), 0) for k < k*, and max(m_(k+1) - m_k, 0) for k >= k*. This equals
TV(m) - 2*max(m), but avoids subtracting nearly equal aggregate values.
segmentStats selects the first maximum using a strict greater-than comparison.
The first cached plane now stores E (stable excess), not variation; the other
planes remain peak mass, bitcast first-maximum bin, and validity. Hinge equality
uses a zero subgradient; at nonsmooth ties this may differ from the old formula's
gradient. Float32 rounding from transmittance products and adjacent T differences
remains upstream.

Backward replays the exact accepted list twice. The first replay reconstructs
boundary T values using the same factors. The loss derivative with respect to
all m_k includes normalization by A and the selected first-maximum derivative;
the hinge subgradient is zero at equality. Since m_k = T_k - T_(k+1), the
boundary derivatives follow by adjacent differences.
For each accepted Gaussian,

    dL/dalpha_i = - sum_{j: d_i < t_j} (dL/dT_j) T_j / (1-alpha_i).

The derivative of the new auxiliary loss is zero at/above the alpha cap .99.
The existing conic/projection chain propagates it to opacity, screen position
and covariance, then to existing 3D parameters. Colors and SH have no direct
auxiliary-loss gradient. Original losses still update them.

IMPORTANT: hard segment membership, sorting, alpha cutoff, early termination
and the validity gate are held fixed in backward. There is NO derivative
through a depth crossing a bin boundary and no direct ray-depth-spacing
penalty. Geometry gradients arise via projected alpha, not by pretending hard
membership is differentiable. This is a limitation, not proof that geometric
concentration will improve. The fixed-membership GPU geometry test avoids bin
boundaries; boundary discontinuities must not be mistaken for a smooth loss.

## Verification

    python -m unittest discover -s tests -p test_unimodal_regularizer.py -v

CPU checks cover unimodal/plateau/empty distributions, random smooth-branch
alpha finite differences, normalized/raw equivalence, ties, clipping, range
conventions and split-inside-bin mass preservation. A host C++ test compiles
the actual shared CUDA arithmetic header with __device__ removed; this does
NOT compile the renderer's CUDA kernels or check GPU resources.

On a CUDA host, first rebuild this checkout's rasterizer, then require its GPU
suite explicitly (the loader never silently uses site-packages):

    cd submodules/diff-gaussian-rasterization
    python setup.py build_ext --inplace
    cd ../..
    GAUDEPTH_REQUIRE_CUDA_TESTS=1 python -m unittest discover -s tests -p test_unimodal_regularizer.py -v

Two GPU tests compare auxiliary gradients to finite differences and check
unchanged legacy outputs and fixed-bin geometry. They must run after an actual
CUDA rebuild; the old binary is not API-compatible with the new settings.

This revision has not been trained or evaluated on DTU. The previous version's
15 passing checks do not certify this revision. A single depth-segmented peak
is an optimization prior, not a guarantee of correct surface position, smooth
physical density, split invariance across bins, or improvement in mesh metrics.
Mask escape by reducing opacity remains possible and must be monitored together
with image reconstruction, interval mass and valid-pixel fraction.
