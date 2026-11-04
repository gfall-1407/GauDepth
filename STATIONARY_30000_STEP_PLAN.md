# Matched 7k-to-30k continuation after the continuous solver passes GPU checks

This is a concrete implementation plan, not a completed trainer or an executed
experiment. The user's latest instruction supersedes the earlier +1,000-update
final-delivery target: continue each scene's original 7k PLY through 23,000 new
optimizer updates, then deliver total-30k meshes and DTU evaluations. Do not
retrain the first 7k steps. The earlier deployment blocker is resolved; numerical
repair measurements are in G1_STATIONARY_GPU_RESULT.md. Training integration and
full-view candidate extraction remain unfinished. No mesh-quality improvement
is claimed.

## Arms and fixed inputs

Use scan110 original 7k PLY SHA256
733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7 on G1.
Both arms load identical arrays and start fresh Adam. This is PLY continuation,
not restoration of the complete 7k optimizer/RNG/training state. Fix topology, the stored
3D filter, active SH degree, resolution=2, white/black background according to
the original configuration, and all learning-rate schedules. Count actual
optimizer.step calls: 23,000 new updates. For update u=1..23,000, use global
iteration 7,000+u for the shared original 30k learning-rate schedule. Snapshot
labels 7,000/8,000/15,000/20,000/25,000/30,000 correspond to completed new updates
0/1,000/8,000/13,000/18,000/23,000. Save after the optimizer update and validation.
The native train.py skips its final optimizer step and saves before stepping,
so its loop cannot be copied unchanged. Intermediate checkpoints are diagnostics;
only the new total-30k snapshots satisfy the requested final training target.

Repeat this protocol separately for each requested scene, with that scene's own
original 7k PLY/camera identity. Never substitute scan40 and scan110 for one
another. Native original training uses densification/filter changes; this fixed
topology continuation is an explicit common experimental restriction, not a
claim of reproducing a full original 30k schedule.

Both arms retain the original native RGB renderer and the same RGB loss. The
extra sparse geometry term differs only in depth readout: Expected versus joint.
This is a comparison under common fixed anchors and numerical-availability
constraints, not an untouched native baseline. Native-only RGB can be a third
control if required, with the extra difference explicitly reported.

## Minimal observation objective, without a surface-normal assumption

Read COLMAP images.bin tracks: a shared point3D_id supplies two measured 2D
observations. Freeze a pool of such paired pixels from the initial model, then
freeze one seeded camera/pair schedule for both arms. Each sampled pixel has its
own ray and depth; neither Gaussian axes nor a frontoparallel patch define a
surface. No DTU ground-truth point, mesh or evaluator result enters training.

Select initial anchors only where both readouts are finite and joint has a
nondegenerate, refinement-stable root in both views. Use only predeclared track
quality and image-boundary rules. Do not tune pool inclusion against DTU scores.
This biases the first experiment toward regions already reconstructable by both
readouts; report the pool size and spatial/view coverage. It does not test root
birth in background or difficult silhouette regions.

For each pair and readout r, form X1=o1+t1*d1 and X2=o2+t2*d2. Use
`L_geo=mean(norm(X1-X2)/scene_radius)` and
`L=L_native_RGB + lambda_geo*L_geo`. Freeze lambda_geo before either arm runs;
an initial declared value of 0.02 is a pilot setting, not validated optimal.
Initially use 8 paired anchors (16 rays) per update, identical in both arms.
Native Camera discards COLMAP cx/cy and centers at (W-1)/2,(H-1)/2. Preserve raw
COLMAP xy and map ray coordinates as
`u_native=Fx_native*(u_colmap-cx_colmap)/fx_colmap+(W-1)/2` (and v likewise).
Do not simply divide observed pixels by resolution. Check camera image IDs/names,
exclude point3D_id=-1, and use the same mapped coordinates for native Expected
bilinear sampling and joint rays. This reproduces the existing camera model;
image correspondence effects from its centered-intrinsic convention remain a
shared limitation to report.

## No free loss removal through missing roots

Do not dynamically drop invalid depths, resample failed pairs, or shrink the
anchor pool. Both arms perform the same joint availability check even when the
Expected arm only uses Expected depths in its geometry loss. Any scheduled
anchor with a lost root, insufficient slope, changed unresolved branch,
nonconvergent refinement or support-budget overflow terminates the run and saves
the last valid state and diagnostic. A failed arm is not a completed total-30k
result. This fail-closed pilot avoids inventing an untested coverage penalty and
does not force initially rootless background rays to acquire surfaces.

Check before an update and immediately after it on that update's anchors; do not
save an invalid post-update state as a valid snapshot. Periodically check the
entire fixed pool, plus a separate fixed spatial grid whose results are reported
without using it to silently change the training pool. The GPU core probe does
not itself establish these training guards or optimization stability.

## Required numerical and speed gate before launching 23,000 new updates

1. The reviewed seven CUDA synthetic tests and unchanged64-ray diagnosis passed
   on a4156a5 within the limits recorded in G1_STATIONARY_GPU_RESULT.md. Preserve
   these checks for substantive changes; do not rerun unchanged tests as busywork.
   All original rays/states are reported, including the 16 originally rootless
   rays. Check vector gradients, complete-search FD, branch signatures, timings
   and peak allocation. A lower residual or valid=True alone is not a pass.
2. On predefined rays, lower support epsilon from 1e-6 to 1e-8 with a declared
   larger budget; compare gradients with zero-padding on the union of IDs. A
   common support cutoff can bias both refinement levels. Do not truncate a ray
   to meet the budget. Investigate any support-sensitive result before training.
3. Exercise near-coalescing extrema, low slopes and finite-near gradients, then
   check a few real Adam updates on an isolated model clone. Compare native RGB
   gradients with the extra term disabled to verify the training integration.
4. Measure 10 complete matched pilot updates in a separate clone; restart both
   formal arms from identical original arrays and fresh Adam after the pilot.
   Use measured whole-step time to quote the 23,000-update cost. The old 12.49s
   probe cannot predict the new solver.
   A batched support selector and a vectorized critical-point search may be
   necessary; current stationary kernel is one ray at a time.

## Training/readout correspondence and sole quality criterion

Training and candidate depth extraction must share the same continuous field,
opacity factor, near/far, support rule, float64 calculation and root guards.
The current one-ray PyTorch solver is not yet a practical full-view extractor.
Its implicit gradient fixes a numerical issue only if the GPU comparison passes.

At both newly trained total-30k snapshots, render native Expected and candidate joint depths
where feasible, using the same camera resolution, mesh fusion, cleanup,
alignment, scan110 GT and evaluator seed. Cross-readout meshes distinguish
weight changes from readout changes. Run the project's evaluate_dtu_mesh.py
pipeline and report Accuracy, Completeness and their mean in millimeters.
Only these comparable mesh results establish quality; a passing gradient probe,
decreasing geometry loss or faster runtime does not.

Existing mature 30k PLYs may supply an additional readout-only comparison, but
must be labeled pre-existing model, zero new training in this run. They cannot
replace the two newly continued total-30k snapshots.

## Runtime and resource accounting

The latest stationary8 core measures63.04ms/ray median (64 rays sum3.861s), while
the complete multi-method/FD diagnostic takes34.945s. Neither measures a complete
training step or dense extraction. The reviewed checks have external caps of120s and
660 s, respectively (plus 10 s termination grace each); these are limits, not
measured durations. Do not quote the old probe's 12.49 s as a training estimate.

For each scene, let tE and tJ be measured wall seconds per complete Expected and
joint update, including all shared availability checks. On one GPU, training
time is 23,000*(tE+tJ). On two equivalent dedicated GPUs, it is approximately
23,000*max(tE,tJ), plus setup, snapshots, rendering, fusion and DTU evaluation.
Two arms require 46,000 new optimizer updates per scene. Multiple scenes add
their measured workloads; parallelization depends on assigned GPUs.

For scale only, not a forecast: 1 second/update means 6.39 hours per arm;
5 seconds/update means 31.94 hours per arm. The current one-ray Python critical
search may therefore require batching even if gradients pass. Full mesh
extraction needs a separately measured full-view renderer; extrapolating sparse
training-ray timing to a claim that mesh extraction is ready is invalid.

Record initial/final code and input hashes, actual update counters, Adam step
counters, fixed anchor/schedule hashes, measured wall time, GPU memory, output
sizes and free system-disk space. Save only under system-disk isolated outputs;
never overwrite original models or use the nearly-full G1 data disk for output.

Independent mathematical review (/root/stable_tv_math_review) supports the
fixed-track, common-availability, abort-on-failure design as a minimal strict
pilot. It explicitly does not establish general full-scene trainability.
