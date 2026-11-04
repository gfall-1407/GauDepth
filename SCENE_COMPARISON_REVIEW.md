# Scene comparison implementation and G6 readiness, 2026-11-04

This is a reviewable implementation, not a completed reconstruction result. No
server files were changed, no source was pushed, and no training/render job was
started. CPU tests take under one second; no long local experiment was run.

## Actual G6 read-only findings

- SSH alias: `GauDepth6-connect.cqa1.seetacloud.com`.
- Observed hostname: `autodl-container-02b744a905-06a70733`.
- GPU: RTX 4090 D, 24564 MiB, driver 580.76.05; no active compute process at last check.
- Data disk: 45 GB available of 50 GB.
- Code: `/root/autodl-tmp/GauDepth`, clean `main`, commit
  `8474727dd1c1f4fee77a3dc02a6cbc0b38075f21`, remote
  `git@github.com:gfall-1407/GauDepth.git`.
- Environment: `/root/miniconda3/envs/radegs/bin/python`, Torch 2.13.0+cu130,
  CUDA 13.0 available. Imports of Open3D, SciPy, OpenCV and rasterization succeed.
- Native wrapper provides `GaussianRasterizer.integrate`. Loaded binary:
  `/root/miniconda3/envs/radegs/lib/python3.12/site-packages/diff_gaussian_rasterization/_C.cpython-312-x86_64-linux-gnu.so`.
  SHA256 `f97dcd66f55d9ddbf673c1d467238fe6f983b9f0bf062645a87bb12d20aa91df`.
  This is an observed binary identity; its build-source provenance has not yet
  been independently authenticated. No claim of CUDA numerical parity yet.
- `/root/autodl-tmp/DTU/scan110`: 64 image files, 64 depth files,
  `sparse/0/cameras.bin` and `images.bin` present. No normals directory; the
  COLMAP reader does not require that directory for this rendering path.
- `/root/autodl-tmp/DTU_MVS_Data`: scan110 STL, ObsMask110_10.mat,
  Plane110.mat and calibration pos_064.txt exist. The full preflight checks all
  64 calibration files and source points3D.ply before allowing the Scene loader
  to create anything under the data root.
- **Missing on G6:** `GauDepth/outtest/scan110/cfg_args` and original 7k PLY.
  `find /root/autodl-tmp -maxdepth 7 -type f -name point_cloud.ply` returned no
  model. This is a bounded search, not a claim about all mounted storage.
- The known scan110 7k source identity from the earlier matched protocol is
  142892 vertices / 36010340 bytes / SHA256
  `733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7`.
  A local Desktop/GauDepth scan found no PLY with that byte count. The coordinator
  must locate/copy this original model (likely original G5 outtest asset) plus
  its cfg_args into an isolated G6 input directory. Do not substitute the local
  scan110 30k PLY (different SHA) and do not train steps 0–7000 again.

## Concrete code

- `collective_readout.py`: full-image CUDA query of existing native cumulative
  opacity, 22 bisections on camera z in [0.2001, 8].
- `compare_scene_mesh.py`: preflight, copied model staging, all-view Expected /
  Median / collective maps, identical TSDF, native DTU evaluation.
- `scene_compare_io.py`: literal cfg parsing, source hashes, isolation checks,
  camera-based alignment audit.
- `check_collective_native.py`: one-Gaussian GPU analytic smoke, to run only
  after deployment review. CPU reference checks passed; this GPU test has NOT run.
- `test_scene_compare_cpu.py`: seven tests passed: analytic first crossing,
  collective weak-opacity crossing, absent/near-clipped roots, monotonicity and
  permutation, camera similarity alignment, literal config, input-preserving staging.
- `evaluate_dtu_mesh.py`: subprocess status now propagates, exact current Python
  interpreter, argument vector instead of shell string; added eval seed.
- `dtu_eval/eval.py`: deterministic seed 0; same metric/filter/sample algorithms.
  Extra coverage counts go to a separate JSON, existing result keys unchanged.

The local archive base is exact commit `6a9481b` from
`GauDepth_21_LocalPlaneMatched1000`. New files and the two evaluator changes are
provided as a standalone patch against that code. Do NOT deploy the entire
imported history or silently restore its old plane experiment. Apply/review only
this patch on an isolated branch of the live server baseline, through Git.

## Candidate definition and limits

At a candidate point along a ray the existing kernel computes approximately

`A(z) = 1 - product_i(1 - alpha_i * exp(-0.5 * min((t(z)-m_i)/sigma_i, 0)^2))`.

The implementation preserves native tile membership, projected conics, clamp
0.99, alpha rejection 1/255, ray-plane approximation and saturation behavior.
Choose the first z where `A(z) >= 0.5`. It aggregates opacity at the candidate
point, then reads the transmission crossing. This is a **GOF-like cumulative
opacity control**, not the previous joint-total-variation construction, not
proven view-independent 3D local support, and not a novelty claim. An alpha
profile is not geometry GT. A single Gaussian crossing is in front of its mean;
this leading-boundary bias is intentional to test, not hidden.

No threshold sweep, GT fitting, per-ray fallback to Expected, or extra mask is
used. Unsupported/near-clipped/native-outside pixels stay zero and have explicit
status codes. The solver rejects autograd-enabled calls: **candidate training
gradients are not implemented**, and a detached root must not be passed off as
an end-to-end method or silently replaced by straight-through Expected gradients.

All image pixels of all 64 views at the cfg resolution (r=2) are processed.
Expected and Median come from the exact same native render call and same model.
All three methods share source masks, valid positive depth <=8, white rendered
color, voxel size .002, block resolution 16, block count 50000, depth scale 1,
depth max 8, and largest-cluster postprocessing. TSDF uses the original Open3D
CPU implementation **on G6**, not on the user's Windows computer. GPU rendering
is the expensive native component; do not change TSDF algorithm just for speed.

## Sole selection criterion and independent geometry

The exact chain is `evaluate_dtu_mesh.py -> dtu_eval/eval.py --mode mesh`.
The outer script reads recon_post.ply, aligns using training camera centers to
DTU Calibration/cal18/pos_*.txt, and writes recon_aligned.ply. It does not call
its otherwise-defined cull_mesh function. We check sequential 0000-based camera
names, all-view roster, and the same camera-fit residual before evaluating.
Residual >1 mm is a fixed alignment failure, not permission to fit the mesh to GT.

The inner script samples triangles, downsamples at .2 mm, applies ObsMask with
BB/Res, uses independent `Points/stl/stl110_total.ply`, filters the scanner by
Plane110, and averages distances strictly below 20 mm. Its fields are
`mean_d2s` (accuracy), `mean_s2d` (completeness), and `overall` (their average),
in DTU millimetres. Lower is better. Results are in each method's vis/results.json.
The 20 mm truncation and observation region can hide some missing/outlier
geometry; status/coverage counts accompany the scores, not replace them.
Camera-alignment and scan incompleteness are limitations. Conditional ray
references, model consistency and synthetic surfaces are not alternative scores.

## Executable G6 sequence after review and input staging

Use a new Git checkout and an output root outside originals, for example:

```bash
PY=/root/miniconda3/envs/radegs/bin/python
CODE=/root/autodl-tmp/GauDepth_scene_compare
INPUT=/root/autodl-tmp/scene_inputs/scan110_original7000
OUT=/root/autodl-tmp/scene_runs/scan110_7000_collective_v1
cd "$CODE"
"$PY" compare_scene_mesh.py --model "$INPUT" --source /root/autodl-tmp/DTU/scan110 --dtu /root/autodl-tmp/DTU_MVS_Data --output "$OUT" --iteration 7000 --stage preflight
"$PY" check_collective_native.py
"$PY" compare_scene_mesh.py --model "$INPUT" --source /root/autodl-tmp/DTU/scan110 --dtu /root/autodl-tmp/DTU_MVS_Data --output "${OUT}_smoke" --iteration 7000 --stage render --smoke-one-view --extension-receipt /root/autodl-tmp/scene_inputs/native_build_receipt.json
"$PY" compare_scene_mesh.py --model "$INPUT" --source /root/autodl-tmp/DTU/scan110 --dtu /root/autodl-tmp/DTU_MVS_Data --output "$OUT" --iteration 7000 --stage all --extension-receipt /root/autodl-tmp/scene_inputs/native_build_receipt.json
```

The receipt requires `native_sha256`, `source_revision`, `build_log_sha256`; it
must be backed by the actual reviewed build, not generated as a claim solely
from imported package identity. Reuse an existing authenticated receipt if available.
Source synchronization is Git only. Model/data copying is separate and must
verify source/destination hashes. No push or copy has occurred in this task.

On an existing result with completed full maps, `--stage fuse` and then
`--stage evaluate` allow rendering to be separated from CPU fusion/evaluation.
All commands keep the same model/source/dtu/iteration values. A smoke output
cannot enter fusion/evaluation. Existing outputs are refused, never overwritten.

## Matched 7k + 1000-update training protocol

The user's requested next screening is valid, but is **not implemented by the
inference control above**. The existing `train_local_plane_matched.py` has useful
tested initialization/counting logic but its 32-window frozen plane residual is
another method; do not launch that code as this collective candidate.

For the actual training candidate, first provide a reviewed differentiable
readout or explicit regularizer with true gradients. Then reuse the matched
protocol mechanics: copy the same verified 7k PLY to A/B; initialize fresh Adam
and fresh appearance identically in both; record this is a PLY warm start, not
optimizer resume; fix the same seed, all-view camera/neighbor sequence, host
losses, resolution, learning-rate schedule and topology policy. Count 1000 actual
optimizer.step calls at virtual iterations 7001 through 8000 inclusive; save
after the last update. Original train.py skips its terminal optimizer step, so
simply passing --iterations 8000 is insufficient to establish this count.

Save 7000/7500/8000 copies; render every arm with Expected/Median/the candidate
where defined, using the pipeline above. Primary comparison is predeclared
A-native Expected vs B-candidate; A-candidate/B-Expected cross readouts separate
training effects from extraction effects. Same GT and TSDF everywhere. Never
feed DTU GT/ObsMask/Plane or metric values into losses or select on per-ray errors.
Investigate nonfinite loss, bad gradients, zero update, data mismatch and mesh
emptiness before rejecting an algorithm. Early DTU scores screen directions;
they are not final 30k performance or a publication claim.

## Timing and blockers

The machine is ready but currently waits for original 7k input and independent
code/build review. No 1000-update candidate is ready because its gradients are
missing. Do not keep the GPU idle while calling this an active experiment.

Provisional budget after inputs/review: preflight + analytic smoke 2–5 minutes;
measure one full view, extrapolate its actual 64-view runtime; reserve 20–45
minutes for all views plus three TSDF/DTU evaluations. At coordinator-reported
1.88 CNY/hour, this reservation is ~0.63–1.41 CNY, excluding already accrued idle
time. These are planning ranges, **not measured throughput or a promise**. If
the smoke projects beyond the agreed window, report its timing before expansion.
For two 1000-update training arms, timing must be based on the candidate's first
timed updates after its gradient implementation exists; no honest measurement
is available now. Server power remains with the coordinator; this task never
shuts down or deletes an instance.
