# Continuous stationary-TV repair: GPU checks completed

2026-11-04. Latest tested source is
`a4156a57c2ef6e097e7f23bbc81a81d5b26b97b8`, including a reviewed tail-direction
fix after first repair `0df0fb95cca82d0c0ee69125f2870afd1bda3002`.
Both pushes succeeded and remote SHAs were verified. Seven CUDA tests pass;
the unchanged 64-ray probe completes, with48/48 valid roots passing4-to8
refinement and16 rays still rootless. See G1_STATIONARY_GPU_RESULT.md for costs,
remaining limitations and evidence. This file retains the initial run history.
Branch `research/joint-tv-train-20261104`, remote
`git@github.com:gfall-1407/GauDepth.git`. Explicit renewed user authorization
allowed one normal retry; push succeeded and the exact remote SHA was verified.
At the first repair, five CUDA synthetic tests passed in1.710s and the64-ray
probe completed in29.132s. Nine extra invalids led to the follow-up above.
First-run directory:
`/root/joint_tv_runs/scan110_stationary_0df0fb9_VQNWS4`.

Latest delivery target: each scene's original 7k PLY plus 23,000 new updates,
fresh Adam equally in both arms, then total-30k meshes and DTU evaluation.
See STATIONARY_30000_STEP_PLAN.md. The earlier +1,000-step target is now only an
optional intermediate diagnostic, never the final deliverable. The earlier
denials below remain historical; the new explicit authorization resolved them.

## What changed

`stationary_tv_depth.py` finds sign-changing stationary points of continuous H,
refines them and the first log(2) TV crossing by 40 bisections in float64, and
uses the implicit first-order VJP `-partial_theta A / abs(Hprime(root))`.
Interior stationary-point motion cancels because Hprime=0. The derivative of
H(near) is retained. Backward recomputes only the prefix endpoints with frozen
monotonic directions; it does not differentiate hundreds of quantized abs
increments. Raw model parameters may remain float32, with the profile chain
promoted before exp/sigmoid/filter/quaternion operations. No normal or RGB change.

Finite sign sampling can miss pairs of stationary points. This is not a
continuous-error certificate; first-order derivatives only, on a stable branch.
The implementation records critical residuals/bracket widths, root residual,
root bracket width and distance from segment endpoints. Valid does not certify
convergence. Numerical guards do not replace branch/refinement checks.

Independent static reviews: `/root/stable_tv_math_review` verified the formula
and near derivative; `/root/joint_core_review` approved the final three source
files for bounded CUDA tests/probe only. A zero midpoint direction with nonzero
endpoint change is rejected; finite-near gradcheck was added following review.
All three files parsed locally. The five reviewed synthetic tests then passed
on G1 CUDA; no training update or mesh extraction has executed in this task.

## Root-cause status

The old probe kept identical member IDs across 2/4/8 levels. Member switching
therefore cannot explain its level-to-level gradient difference. Omitted members
can still contribute a common bias; an omitted-TV bound is not a gradient bound.

Float32 zero adjacent-H differences have now been observed on nonzero-slope
intervals, and full-chain float64 reduces gradient drift. This does not isolate
every source of the discrepancy. The probe compares pl32 4/8, fully promoted
pl64 4/8, and stationary64 1/2/4/8 on all the same64 rays and support. It records intervals with
positive dt, zero delta-H and nonzero same-sign endpoint slopes, including the
root prefix. Comparisons use gradient vectors, not only norms. Stationary pairs
must also agree in critical count, directions, first-root segment and positions
(max difference <=1e-7); depth/TV <=1e-5 and gradient L2 <=1e-6+1% scale.

The first four valid stationary rays additionally receive directional finite
differences for xyz/log-scale/rotation/opacity, two step sizes each, with an
independent complete search at each perturbed parameter. They are a verification
subset, not evidence that all real-model gradients pass FD. All 64 states remain
in the report; invalids cannot pass convergence by matching zero sentinels.

## Exact bounded G1 execution for the current reviewed version

G1 host `autodl-container-d6a7459179-693ab6ee`, SSH alias
`GauDepth1-connect.cqa1.seetacloud.com`. Last read-only check: RTX4090D unused,
system disk ~22 GiB free; data disk almost full and must only be read.
Use isolated checkout `/root/joint_tv_code_6d64509`; original checkout unchanged.
Do not shut down the instance.

1. Current reviewed source was pushed; verify remote/source SHA before a rerun.
2. On G1 recheck hostname/GPU processes, remote and clean source state. Fetch
   this branch and detach at a4156a57c2ef6e097e7f23bbc81a81d5b26b97b8.
   Do not switch original repo.
3. Create a fresh directory under `/root/joint_tv_runs` on the system filesystem.
4. Run with `/root/miniconda3/envs/radegs/bin/python`, `PYTHONDONTWRITEBYTECODE=1`.

```bash
timeout -k 10s 120s /root/miniconda3/envs/radegs/bin/python -B \
  -m unittest test_stationary_tv_depth -v
```

Only if all seven tests pass:

```bash
timeout -k 10s 660s /root/miniconda3/envs/radegs/bin/python -B \
  probe_stationary_scene_gpu.py \
  --input-ply /root/autodl-tmp/GauDepth/outtest/scan110/point_cloud/iteration_7000/point_cloud.ply \
  --source-path /root/autodl-tmp/DTU/scan110 \
  --model-id scan110_original7000 \
  --expected-head a4156a57c2ef6e097e7f23bbc81a81d5b26b97b8 \
  --max-seconds 600 \
  --output /root/joint_tv_runs/NEW_UNIQUE_RUN/probe.json
```

Capture UTC start/end, exit status and all stdout/stderr to fresh logs. The probe
writes a started manifest and row journal; only normal completion writes complete.
Nonfinite gradient serialization, timeout or missing rows is a failure, never a
completed result. Recheck original PLY SHA after the probe:
`733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7`.
Download result files only; source synchronization stays Git-only.

## Historical deployment blocker, resolved after renewed authorization

The first push of this commit was rejected before execution. After reading the
trusted user-message history (including explicit permission in the GitHub-branch
context), one retry of the same push was also rejected before execution:

> This retries the previously rejected external push; the available trusted user
> text authorizes implementation and GPU testing, but does not clearly authorize
> publishing this new commit to the remote branch.

No workaround source transfer or alternate publication was attempted.
The previously blocked action was `git push origin research/joint-tv-train-20261104` to
gfall-1407/GauDepth, publishing the reviewed repair commit above. New code has not
reached G1 at that time. A later explicit user approval addressed this specific
repair, and the normal push succeeded. G1 remains running; it was not stopped by
this task.

The previous measured GPU results remain those in G1_SCAN110_GPU_RESULT.md:
64 rays, 48 valid roots, 0/48 complete convergence passes, 12.49 s internal probe.
Those timings do not describe this new stationary solver.
