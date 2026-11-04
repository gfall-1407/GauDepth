# Historical frozen 30k scan40 comparison

This supersedes the unstarted scan37 7k experiment. No 7k-to-30k training,
optimizer recovery, parameter update or new training is performed.

G6 inventory found no existing iteration_30000 PLY/checkpoint under its model
roots. A preserved historical original scan40 30k model was therefore copied
from the local protected input archive into a new G6 directory:

`/root/autodl-tmp/scene_inputs/scan40_original30000_20261104`

Its source metadata explicitly records
`/root/autodl-tmp/GauDepth/outtest/scan40/point_cloud/iteration_30000/point_cloud.ply`
and model SHA256
`d1df2e5ac7048dc2ba3b76197bb252d59134faf4220242bfb3e39238f6e5e0d8`.
This is the historical model identity, not a new training claim. The original
archive and server 7k files were not changed. Historical metadata remains as
`historical_source_metadata.json` and the new verification as `model_receipt.json`.

Model size 247737716 bytes; 983080 Gaussians. Original cfg SHA256
`70b2a1135b1df09d51ad979f02efeb6d153fb12787e84c0052f350dbff7a1dcb`.
The transferred model and cfg match the historical SHA256 records on G6.
The 49 source RGBA images, camera/COLMAP data and independent DTU GT are the same
verified scan40 assets, with 119 fixed data hashes. Render dimensions 777x581.

## Minimal changes relative to reviewed 586ba468

- Allow explicitly declared iteration 30000 in validate_inputs; the exact
  profile iteration, PLY hash, Gaussian count, configuration and data checks
  still apply. No relabelling a 7k model as 30k.
- Add scan40_30000_profile.json fixing that historical identity.
- Record frozen_model=true and optimizer_updates=0 in scores.json. The
  early_screening_only label remains because this is still a limited method
  screening comparison, even though the input model is historical 30k.
- Add a small input-validation test for accepted 30k identity, mismatched
  profile iteration and altered model bytes. Run the CPU suite on G6 after
  reviewed Git deployment; no experiment computation is run on the user's PC.

No depth readout, renderer, native extension, TSDF setting, alignment or DTU
algorithm is modified. No dependency installation. The controlled native
extension remains SHA256 ecce3c888ae4589d00f6427296acf2fd4df90cd45227a1d173f2ff45ae58d22b.

## Execution after independent review and Git deployment

Use a separate checkout and a fresh output root:
`/root/autodl-tmp/scene_runs/scan40_30000_collective_full_v1`.
Use radegs Python and the existing controlled-extension PYTHONPATH. Invoke
compare_scene_mesh.py with:

```
--profile scan40_30000_profile.json
--model /root/autodl-tmp/scene_inputs/scan40_original30000_20261104
--source /root/autodl-tmp/DTU/scan40
--dtu /root/autodl-tmp/DTU_MVS_Data
--output /root/autodl-tmp/scene_runs/scan40_30000_collective_full_v1
--iteration 30000
--extension-receipt /root/autodl-tmp/build_audit/GauDepth_8474727_native/native_build_receipt.json
```

Run CPU tests and preflight on G6, then all49 render (300+30 seconds), three
sequential fusions (600+30), three DTU evaluations (900+30). Set
PYTHONFAULTHANDLER=1, PYTHONUNBUFFERED=1 and ulimit -c 0. Keep all stage exit
codes, per-view coverage/status maps, seed0 DTU counts/20mm exclusions and all
input before/after hashes. Use the same finite/positive/depth<=8/mask rule,
voxel .002, block_resolution16, block_count50000, depth_scale1, depth_max8,
largest-cluster postprocessing and the same independent STL/ObsMask/Plane.

Recompute Expected, Median and collective_half together on this 30k model.
Do not pair with 7k or historical metrics. Estimate 10-15 minutes using the
completed 7k scene's 9m49s as a reference; this is not a measured 30k duration.
Budget <=8GiB output, with free-space checks between stages. Preserve any
partial/failure directory; do not automatically overwrite or retry.
