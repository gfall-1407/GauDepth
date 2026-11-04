# Historical scan65 30k frozen comparison

This adds only a scene profile and this review note to reviewed commit
95d231e47bfefc1dde44473d21a9342e5cb9dfab. No renderer, candidate, TSDF,
evaluator, dependency or training changes. Do not rerun scan24, scan40 or scan110.

## Independently checkable identity

- Original historical source: G1 /root/autodl-tmp/GauDepth/outtest/scan65.
- G6 model: /root/autodl-tmp/scene_inputs/g1_30k_20261104_task18/scan65.
- iteration_30000/point_cloud.ply SHA256
  3e86c5d09ec4cb2cde3e915e469b0b7a6915c344f313c1a0de8c61bfa21d1c6a;
  115943480 bytes; 460087 Gaussians. cfg SHA256
  aaded6a2ba2c91a195a4dafbf90adab930b204f15565a89f60c9981e34d39aff.
- Transfer destination verification has all three scan65 model/config/camera
  files matching source SHA256. Original training commit is not recorded by
  cfg_args; do not invent one. This is historical checkpoint reuse, zero updates.
- scan65 has 49 sequential source views 0000..0048, all RGBA 1554x1162;
  resolution=2 produces 777x581. COLMAP roster independently matches the images.
- /root/autodl-tmp/DTU/scan65 and /root/autodl-tmp/DTU_MVS_Data contain all
  camera, image, STL065, ObsMask65, Plane65 and 64 standard calibration files.
  Profile pins 119 data hashes plus exact model hash, Gaussian count and config.
- Existing reviewed validate_inputs passed on G6 for 121 protected files.
- Raw identity receipt: /root/autodl-tmp/scene_inputs/scan65_30000_identity_20261104.json,
  SHA256 a975dd898df17a803c8523c58ab3483bacda61f94f45d4b31ea4ebe8a47b00a9.
- Profile SHA256: 35d3f315776ab6642727c0763bc4a84d7b16a92ca5d895f1c3bcc3d3c4604f42.

## Execution after independent profile review

Deploy via normal Git push/fetch of research/collective-scan65-30k-20261104
to a new detached checkout /root/autodl-tmp/GauDepth_scan65_30k_reviewed.
Verify reviewed HEAD and clean state. Use radegs Python and the same controlled
native extension, actual SHA256
ecce3c888ae4589d00f6427296acf2fd4df90cd45227a1d173f2ff45ae58d22b.
Native receipt SHA256
72cacf22da94c14fe989d9216f8d3a003032b6d718ff2bed3f5617e1cc0d3e8b.

```
export PATH=/root/miniconda3/envs/radegs/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/build_audit/GauDepth_8474727_native/submodules/diff-gaussian-rasterization
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
ulimit -c 0
COMMON=(--profile scan65_30000_profile.json
  --model /root/autodl-tmp/scene_inputs/g1_30k_20261104_task18/scan65
  --source /root/autodl-tmp/DTU/scan65
  --dtu /root/autodl-tmp/DTU_MVS_Data
  --iteration 30000
  --extension-receipt /root/autodl-tmp/build_audit/GauDepth_8474727_native/native_build_receipt.json)
```

Use fresh smoke output /root/autodl-tmp/scene_runs/scan65_30000_collective_smoke_v1:
CPU suite and preflight, then one-view render with 300s hard timeout/30s grace.
Check exact native/code/model identity, 1 of 49 views, camera alignment <1mm,
777x581 finite depth arrays, coverage/status counts. A failed smoke is retained
and stops this run. Do not fuse smoke frames.

Use separate fresh full output /root/autodl-tmp/scene_runs/scan65_30000_collective_full_v1.
Stages: preflight, all49 render (300s+30s), three sequential fusions (600s+30s),
three DTU evaluations (900s+30s). Capture command, PID, elapsed time and exit code
per stage. No automatic overwrite/retry. Require GPU idle before launch.

Protocol stays identical to paired scan40: Expected/Median/collective_half all
recomputed from one frozen PLY; collective_half 22 bisections, batch131072;
common finite/positive/depth<=8/alpha>=0.5 mask; missing candidate depths remain
zero. CPU Open3D VoxelBlockGrid voxel.002, resolution16, count50000,
depth_scale1/depth_max8, largest cluster and 50-triangle minimum.
evaluate_dtu_mesh.py -> dtu_eval/eval.py remains the only effectiveness criterion,
seed0, density.2, patch60, maxdistance20mm, identical independent GT/alignment.

Record all 49 frame hashes, per-view and aggregate coverage/status counts,
distance exclusions, source before/after hashes and staged model hashes.
Write failure evidence even if any stage exits nonzero. Require >=16GiB free
before launch, >=8GiB free between stages and <=8GiB output. Preserve all
inputs and previous outputs. No local experiment computation or power actions.

Estimate 10-15 minutes including smoke and audits from scan40's measured 6m27s
full run, not a measured scan65 time. This comparison is not the user's final
novel method and does not remove native short-axis fallback. Report adverse
results and failures equally; do not select scenes based on scores.
