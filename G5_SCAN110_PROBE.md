# G5 scan110 adaptation

The 15 synthetic CUDA tests have no scene or model dependency and can run on G5
without changing their numerical definition. They do not invoke native rasterizer
operations or require its rebuild. The real probe now requires --model-id;
scan40 and scan110 are separately identified and cannot share a PLY hash label.

The scan110 identity comes from the preserved native capture metadata:
G5_scan110_native_capture_faa9294_bundle/capture/metadata.json. It records original
7k PLY SHA 733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7,
142892 rows, camera SHA 9dd012bebe2dea0514af3f784dbb8fff23354fd7e9476ae6f9c668c40162d06b,
and images.bin SHA 433bda84b16ae7961619442c5bc74c70ddbb090e38ff61eaf601ff3719bb2794.
These must also match live server files; the probe enforces them before model
evaluation. No fallback to a differently trained scan110 PLY is allowed.

Expected input PLY location from that evidence:
/root/GauDepth_runs/pair12_inputs/scan110/point_cloud.ply.
Expected camera/data source: /root/autodl-tmp/DTU/scan110.
The preserved G1_20261101_12scan_original7000_inventory.txt independently records
the same scan110 PLY SHA at
/root/autodl-tmp/GauDepth/outtest/scan110/point_cloud/iteration_7000/point_cloud.ply.
If the coordinator assigns G1, the same scan110 model-id can be used with this
input path only after live PLY and camera hashes pass; server identity alone is
not a reason to assume files are identical.
GT is not used by this numerical probe. Its result is not interchangeable with
scan40, and a successful probe is not a mesh-quality or training result.

Deployment: use the already authorized research/joint-tv-train-20261104 branch;
fetch the reviewed exact commit into an independent checkout, never switch the
shared server repository. Check hostname/GPU availability/other GPU jobs before
running. Record git HEAD, Python/torch/CUDA/GPU identity, stdout, stderr and exit
codes in a new /root/autodl-tmp/joint_tv_runs/scan110_<unique-run> directory.

With the actual G5 environment verified and this checkout current:

```sh
timeout -k 10s 120s python -B -m unittest test_joint_tv_depth -v
timeout -k 10s 270s python -B probe_joint_scene_gpu.py \
  --model-id scan110_original7000 \
  --input-ply /root/GauDepth_runs/pair12_inputs/scan110/point_cloud.ply \
  --source-path /root/autodl-tmp/DTU/scan110 \
  --expected-head <REVIEWED_COMMIT> \
  --output /root/autodl-tmp/joint_tv_runs/scan110_<unique-run>/probe.json \
  --view 0049 --side 8 --max-seconds 240
```

Only run the second command after the first passes. The outer timeout limits the
whole program; internal timeout is checked between rays. Preserve partial rows
and started_not_complete manifests on failure. No original data/model writes,
training, main-branch updates, extension replacement, or power operation.
