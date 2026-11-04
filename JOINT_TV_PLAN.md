# Joint total-variation training candidate: feasibility stage

Theory source: probability-support thread 01a1a5a3-84bc-77c2-8735-1c41fd83000e,
latest joint/sep derivation, read 2026-11-04. This is a research candidate, not a
novelty or reconstruction-quality claim. GGGS and Objects as Volumes overlap.

## Definition and executable boundary

Use the entire anisotropic covariance and global 3D-filtered opacity to form
G_i(x). Set opacity to (1-1e-6) times the native filtered opacity explicitly so
fully saturated native logits do not make log(1-G) infinite. Use the same
definition for joint and sep controls. No Gaussian axis defines the surface.

H(x) = sum_i -0.5 log(1-G_i(x)). Along a ray joint cumulative optical depth is
the total variation of H from near to t; sep sums the individual variations.
First cumulative log(2) crossing is the candidate depth. Native RGB is retained.

joint_tv_depth.py implements the exact first crossing of piecewise-linear H on
adaptive knots. It is an approximation of continuous joint, not a certified
continuous integral. All knot locations, sorting, H evaluations, and crossing
interpolation participate in autograd. No detached implicit-gradient shortcut.
Nested midpoint refinements expose unresolved roots/variation; they do not prove
all extrema have been found. Dense agreement and depth-gradient convergence are
necessary before this can be used for a training experiment.

No root stays no root; depth=0 is a sentinel with a separate valid mask. There is
no Expected fallback. Roots with near H>1e-4 or secant slope<1e-4 are invalid.
An additional analytic |H'(t)| guard rejects continuous turning roots such as
the single-Gaussian a=.75 case. has_sampled_root=False means no root in the
sampled model; it does not generally prove absence in the continuous field.
An entire flat interval or a switching root branch is not assigned a smooth
derivative claim. The solver keeps front-layer history, so joint is not generally
view independent. Regression tests include the .53 recovery case, .30 no-root
case, and separate front/back layers from the theory derivation.

The existing native extension can render RGB. Its integrate() is a different,
inference-only GOF-like cumulative opacity and cannot implement joint TV or its
gradient. This stage uses PyTorch CUDA; no extension rebuild or replacement.

## Minimal training integration (not yet implemented or launched)

Start both arms from scan40 original PLY SHA256
51f5f92bd6015ae98509b58b5aa293d110f4b4f13b8217916fdea0a819d84187 and fresh Adam.
Use identical camera/neighbor/pixel schedule, native RGB loss, 1000 real optimizer
updates, fixed topology/filter, and snapshots at 0/500/1000. Preserve original
input and write each arm to a new directory. Existing matched-plane driver is
scan110-specific and cannot be run on scan40 without adapting and reviewing it.

Use candidate depths for sampled multiview reprojection/photometric constraints,
not a penalty that simply minimizes TV, cancellation, or opacity. Such a penalty
can trivially erase occupancy. An exact reference-point projection requires no
normal assumption. For a photometric patch, each patch pixel needs its own depth;
a constant-depth/frontoparallel patch is an additional assumption and must not be
silently introduced. Visibility must use the same candidate in both cameras;
using native Expected visibility is a changed hybrid assumption to report.

Root existence masks must not be allowed to turn off supervision at no cost.
Before choosing a mask/coverage loss, measure baseline root coverage and test
root birth/death. RGB alone is not proof that candidate geometry coverage remains.
This is a remaining objective-design issue, not solved by the numerical core.

Candidate training/extraction must call the identical candidate solver with the
same near/far, opacity policy, support culling, and numerical resolution. Native
Expected baseline extraction remains unchanged. Cross-readout meshes distinguish
weight changes from readout changes. Evaluate via evaluate_dtu_mesh.py with the
same mesh cleanup/alignment/GT/evaluator seed. Only Accuracy/Completeness/mean
are quality outcomes; RGB, ray tests, and root coverage are supporting diagnostics.

## Immediate bounded GPU checks

1. Independent source review before push; Git-only synchronization to a new G6
   checkout. Recheck hostname, GPU load, commit, native extension identity.
2. Ten CUDA tests: exact/analytic controls, adaptive-grid gradcheck, full
   covariance chain gradcheck, no-root, near-occupied and gradient guard.
3. Read-only original scan40 PLY and COLMAP cameras, 64 spatially spread rays.
   Compare nested 1/2/4 refinement, retained support count, conservative omitted
   TV bound, forward/backward time, and gradient finiteness. This probe is solely
   to decide if a full-scene implementation is computationally feasible.
4. If the O(K^2) adaptive evaluation or all-Gaussian ray selection is too slow,
   stop this prototype before 1000-step training. A tiled support list and a
   batched CUDA evaluator will be needed; isolated easy rays do not prove speed.

No training, full mesh, DTU improvement, server shutdown, or new method validation
has been completed by the files in this feasibility stage.
