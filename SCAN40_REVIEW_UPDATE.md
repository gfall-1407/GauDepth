# Scan40 identity and controlled CUDA build update

This supersedes the scan110 execution commands and missing-build status in
SCENE_COMPARISON_REVIEW.md. The candidate definition and scientific limitations
there remain unchanged. Full-scene candidate execution and training remain NO GO.

## Fixed input and validation

- Source scene: `/root/autodl-tmp/DTU/scan40`.
- Isolated model: `/root/autodl-tmp/scene_inputs/scan40_original7000`.
- Iteration: 7000 only in this reviewed profile. 913575 Gaussians, independently
  reread from the original PLY header during the build audit.
- PLY SHA256: `51f5f92bd6015ae98509b58b5aa293d110f4b4f13b8217916fdea0a819d84187`.
- Exactly 49 cameras 0000 through 0048, 1554x1162 source images, r=2,
  777x581 rendered images. The code validates all of these values.
- `scan40_profile.json` fixes image, camera, points3D, independent STL,
  ObsMask, Plane and all 64 calibration file hashes. The loaded camera roster,
  matrices and intrinsics are recorded; the native centered-principal-point
  convention is checked. Parameters such as resolution, appearance, 3D filter,
  image directory and kernel size are checked against this reviewed profile.
- Original PLY and cfg were copied, not moved. No fresh 0-to-7k training.
- Eight CPU tests pass, including rejection of wrong model hash, Gaussian
  count, image fingerprint, camera roster, image dimensions and resolution.

## Controlled build completed on G6

The user authorized the pinned GLM fetch and isolated build. Actual source:
`/root/autodl-tmp/build_audit/GauDepth_8474727_native`.

GauDepth revision: `8474727dd1c1f4fee77a3dc02a6cbc0b38075f21`.
GLM official origin: `https://github.com/g-truc/glm.git`.
GLM revision: `5c46b9c07008ae65cb81ab79cd677ecc1934b903`.
CUDA compiler: `/usr/local/cuda/bin/nvcc`, 13.0.88.
Torch: 2.13.0+cu130, architecture sm_89, C++20 from the actual build log.
No ninja was installed; the existing PyTorch distutils fallback compiled successfully.
No source changes were necessary. Tracked source and the original repo are clean.

New binary SHA256:
`ecce3c888ae4589d00f6427296acf2fd4df90cd45227a1d173f2ff45ae58d22b`.
Actual loaded path:
`/root/autodl-tmp/build_audit/GauDepth_8474727_native/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-312-x86_64-linux-gnu.so`.

Build log SHA256:
`ea59980fb8ee678b35d4e3e142a9bf255322265bc510ccc243d355f20c369ac2`.
Receipt SHA256:
`72cacf22da94c14fe989d9216f8d3a003032b6d718ff2bed3f5617e1cc0d3e8b`.
The receipt records hashes for 654 source files, compiler and interpreter
versions, build command, loaded wrapper/binary and unchanged original model.
Remote evidence: `native_build.log`, `native_build_receipt.json`,
`native_api_checks.log` under the build directory. The original installed
binary remains unchanged; its old provenance is still unknown. The new receipt
authenticates the new isolated build only.

## GPU API checks actually executed

Tests loaded and asserted the new isolated binary path, took 0.969 seconds of
in-process time, and passed:

| Check | Actual result |
|---|---|
| Single Gaussian analytic opacity | Maximum absolute error 1.19e-7 |
| Two Gaussians each opacity .4 | Collective peak .640000045; half crossing error 7.54e-7 scene units |
| Off-axis, rotated anisotropy | Maximum opacity error 2.98e-7 |
| No crossing | Correctly identified |
| Near clipping | Outside query rejected; already-crossed near endpoint identified |
| Query blocks 1,17,64,257 vs 513 together | Maximum absolute difference 0 in all four cases |
| Degenerate smallest scale 1e-8 | Existing zero-before-plane/.9-after-plane fallback confirmed |

The anisotropic analytic test uses a ray through an off-axis Gaussian mean.
It does not establish exact world-field evaluation at arbitrary other pixels.
Degenerate fallback was confirmed, not fixed. These were bounded calls to the
controlled native API. The full Python image wrapper and real-image GPU timing
have not run because the candidate source patch has not been approved for Git
deployment. No scene accuracy or training-gradient claim follows from these tests.

The exact invocation source is retained locally in
`../native_api_checks_command.py`, SHA256
`f6e229da3767fc9e67557166f2b00d5fd58205018d072bbf0a5eed2616210a9e`.
It was executed as an API test command, not installed into or used to modify
the server source checkout.

## Next permitted single-image check, after source review/deployment

Use Git to deploy the reviewed patch on an isolated checkout of the live base;
do not push the imported local snapshot history as a replacement main. Verify
the new checkout is clean and its revision matches the reviewed patch.

```bash
PY=/root/miniconda3/envs/radegs/bin/python
BUILD=/root/autodl-tmp/build_audit/GauDepth_8474727_native
CODE=/root/autodl-tmp/GauDepth_scene_compare_reviewed
INPUT=/root/autodl-tmp/scene_inputs/scan40_original7000
OUT=/root/autodl-tmp/scene_runs/scan40_7000_collective_smoke_v2
export PYTHONPATH="$BUILD/submodules/diff-gaussian-rasterization"
cd "$CODE"
"$PY" compare_scene_mesh.py --profile scan40_profile.json --model "$INPUT" --source /root/autodl-tmp/DTU/scan40 --dtu /root/autodl-tmp/DTU_MVS_Data --output "$OUT" --iteration 7000 --stage preflight
timeout -k 30s 300s "$PY" compare_scene_mesh.py --profile scan40_profile.json --model "$INPUT" --source /root/autodl-tmp/DTU/scan40 --dtu /root/autodl-tmp/DTU_MVS_Data --output "$OUT" --iteration 7000 --stage render --smoke-one-view --extension-receipt "$BUILD/native_build_receipt.json"
```

Preflight checks hashes before CUDA. The smoke processes every pixel of one
777x581 image, records candidate failure statuses and timing, and sets
full_scene=false. Such output is refused by fusion/evaluation. It is a speed
and correctness check, not an alternative to DTU mesh accuracy.

The old Expected baseline 2.2925406295 mm used the previous binary and unseeded
evaluator. Recompute all final paired readouts under the same new binary and
reviewed evaluator when full-scene execution is authorized. No candidate
training can start without a separate reviewed gradient/regularizer design.
