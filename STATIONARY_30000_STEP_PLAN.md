# Current matched training screen (supersedes the old 30k-only plan)

The current task is a new training experiment, first screened with 1,000 actual
updates from the same original 7k PLY and fresh Adam. The user's current 30k
request concerns pre-existing mature models being evaluated separately; those
zero-update readout comparisons are not outcomes of this training experiment.
This file retains its historical filename to make the superseded plan explicit.

## Method and controls

For each fixed COLMAP track pair, the new readout returns the first continuous
joint-TV root along each camera ray. The field aggregates all retained Gaussian
profiles through their full covariance; it does not infer a surface from a
single Gaussian's shortest axis. Let X=o+t*d and penalize the mean paired
world-point distance divided by scene radius, with declared weight 0.02.
This trains agreement of independently observed, collectively supported points;
it does not impose a single peak or collapse front/back layers into one surface.
Native RGB loss is retained.

Use a common fixed-topology native-RGB continuation control and the new
joint-readout geometry training arm. Extract both Expected and joint meshes
from both checkpoints, giving a 2x2 training/readout comparison. The Expected
geometry arm is an additional old-readout geometry control, not an untouched
native training baseline. No arm claims reproduction of the full original
normal, PatchMatch, densification or opacity-reset recipe. All restrictions,
learning rates and appearance initialization are shared and recorded.

## Frozen fair setup

G1 scan110 original 7k PLY SHA256:
733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7.
Start each pilot and formal arm from this unchanged file with fresh Adam and
identical fresh zero PGSR coefficients. Preserve topology, stored 3D filter,
SH degree, native black background, resolution 2 and kernel settings.
Use update u at learning-rate schedule position 7000+u. Snapshot 8000 means
exactly 1,000 new validated optimizer updates, not complete training-state resume.

Freeze shared COLMAP observations, mapped to the native camera convention, and
the seeded camera/pair schedule before training. The initial 307a7f1 preparation
selected 32 groups of 16 pairs (512 total); record and recheck the regenerated
pool and schedule hashes for subsequent source revisions. Each step uses eight
pairs (16 rays). No DTU ground truth or evaluation metric enters selection or
optimization. Initially unavailable roots are excluded once during preparation;
subsequent root loss may not silently remove terms, shrink the pool or resample.

## Numerical checks and initial results

307a7f1 passed 19 CUDA synthetic tests (5.166 seconds). Its real-model probe
completed in 12.085 seconds with 429,742,592 peak allocated bytes. Conservative
indexing exactly matched full support selection for all 64 rays at both 1e-6
and 1e-8 thresholds. Six declared root-valid rays passed vector-gradient and
depth cutoff comparisons; two rootless rays were reported separately. Batch8
matched single-ray status/depth and took 0.301 versus 0.878 seconds. These are
finite empirical checks, not continuous-error or whole-scene certificates.

Both first 10-step pilots stopped after one attempted update and zero accepted
updates because the post-update guard rejected existing raw opacity infinities.
A read-only check found 543 +Inf logits already in the original PLY, no -Inf or
NaN opacity, and all other PLY fields finite. Their sigmoid opacity is finite.
The repair must preserve these exact indices/signs with zero gradients and Adam
moments, reject new nonfinite values, and retain strict checks on all other
parameters. Do not clamp the original model or describe this as a new Adam
instability. A separate empty-support Inf*0 graph issue also requires repair.

A 256-pixel distributed extraction diagnostic completed in 13.736 seconds,
with 98 valid depths under each readout. It is not a mesh or reliable dense
throughput estimate: scattered pixels occupy small batches. Full-view runtime
must be measured or bounded before promising a DTU result.

## Launch gates and deliverable

After independently reviewing the narrow repair, push to the authorized research
branch, deploy exact SHA, run relevant CUDA regression tests and regenerate the
experiment in a new system-disk directory. Run each short pilot from original
arrays; count actual steps and post-update checks, verify saved snapshots reload,
and retain failed attempts separately. Use bounded jobs and the verified native
CUDA runtime, coordinate GPU occupancy, and do not stop instances.

When pilot checks pass, restart formal controls and candidate from the same 7k
arrays for 1,000 updates each, save after steps 100/500/1000, and report actual
PID, log, elapsed time and optimizer counters. Keep the same field, support rule,
near/far, opacity factor and convergence guards in training and dense extraction.
Use identical fusion, cleanup, alignment, data and evaluation seeds for all
cross-readout meshes. The project's DTU mesh evaluation (Accuracy, Completeness,
and their mean, lower is better) is the only quality criterion. Loss decrease,
CUDA checks, root validity and speed are engineering evidence only. Extend
training only after the comparable mesh results justify it.
