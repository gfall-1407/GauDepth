# G1 scan110 joint-TV feasibility result, 2026-11-04

Decision: **NO GO for 1000-update training with this numerical core.** CUDA
execution works, but observed depth/gradient refinement does not converge on
the real-model probe. This is not a DTU mesh evaluation or evidence of accuracy
improvement. No optimization update or mesh extraction was performed. This
rejects the current numerical implementation for training, not the joint idea
itself. An independent read-only results review agreed with this interpretation.

## Execution and provenance

- Host: autodl-container-d6a7459179-693ab6ee, NVIDIA GeForce RTX 4090 D.
- Python environment: /root/miniconda3/envs/radegs; torch 2.13.0+cu130, CUDA 13.0.
- CUDA synthetic tests: 15/15 passed, unittest time 3.723 s, exit 0;
  tested commit 6d64509e3a7dae7e796a2ad48fc7cc73c2235a69.
- Real probe: commit 5e0bde35d4786babdb59c48fc48ebd90c32de3c9, exit 0,
  UTC start 12:29:47 and end 12:30:05. The intervening change only moved and
  guarded the output root; git diff confirmed the numerical core/tests unchanged.
- Source checkout: /root/joint_tv_code_6d64509, independent of the original repo.
- Original scan110 7k PLY: /root/autodl-tmp/GauDepth/outtest/scan110/point_cloud/iteration_7000/point_cloud.ply.
  SHA256 733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7,
  142892 Gaussians. PLY and both COLMAP hashes matched the retained capture record
  on G1. This is distinct from scan40.
- All new server files are on the system disk in
  /root/joint_tv_runs/scan110_20261104_1224. System disk retained about 22 GiB free;
  the 98%-used data disk was only read. No original model or data was modified.
- Outer timeout: 270 s with 10 s termination grace. Internal budget: 240 s.
  Neither timeout was reached. The instance was not stopped or shut down.

## Measured numerical results

One view (0049), r=2, image 777x581, an 8x8 spatial grid of rays. Every one of
the 64 rays was evaluated: no empty/over-512-member ray was skipped. Retained
members per ray: min 17, median 159, max 485. Full-model profile selection took
0.1481 s. Reported program duration was 12.4897 s (shell wall interval 18 s,
including startup/preflight). Peak **PyTorch allocated**, not total/reserved GPU
memory, was 827653632 bytes = 0.7708 GiB.

At each level 2/4/8, 48 rays had valid sampled roots and 16 had no sampled root.
Root status never changed across these refinements. All computed parameter
gradient groups were finite. These facts do not establish continuous accuracy
or say which rays should intersect ground truth.

| Observed quantity, 48 comparable valid rays | 2 to 4 | 4 to 8 |
|---|---:|---:|
| Depth absolute-change median, axial scene units | 2.6226e-6 | 5.9605e-7 |
| Depth absolute-change max, axial scene units | 3.1996e-4 | 6.1035e-5 |
| Depth within 1e-5 screening tolerance | 34/48 | 46/48 |
| Total-TV absolute-change median | 4.6337e-4 | 1.5950e-4 |
| Total-TV within 1e-5 screening tolerance | 0/48 | 1/48 |
| xyz gradient relative-L2-change median | 18.76% | 22.89% |
| log-scale gradient relative-L2-change median | 20.80% | 21.94% |
| quaternion gradient relative-L2-change median | 37.89% | 40.53% |
| opacity-logit gradient relative-L2-change median | 1.375% | 1.078% |
| All prescribed checks passed together | 0/48 | 0/48 |

The large xyz/rotation changes are not only tiny-vector ratio effects: at 4 to 8,
median absolute L2 differences were 0.7203 for xyz and 0.1183 for quaternions.
Only 2/48 xyz groups and 1/48 quaternion groups passed their gradient check.
This is enough to block training even independently of the whole-ray TV test.

Median per-ray forward/backward times were 0.02386/0.03656/0.06358 s at levels
2/4/8. Level 2 includes a 1.0467 s first-call maximum. These are measured probe
times, not a full-scene training throughput estimate.

The omitted-member TV upper bound had median 1.2710e-5 and max 4.4212e-5.
The membership is the same within each refinement comparison, so omission alone
does not explain the between-level gradient changes. It remains an additional
full-field approximation, and this TV bound is not a depth/gradient error bound.

## Concrete prerequisites before training

1. Separate numerical precision from quadrature/gradient effects on the same
   fixed member sets: GPU float64 versus float32 and 8/16 refinement. Keep root
   failures and every chosen ray; do not lower thresholds or select only easy rays.
   Small depth differences cannot substitute for gradient convergence.
2. If moving piecewise-linear knots remain unstable, implement and review the
   continuous stationary-point TV evaluation and its implicit root derivative.
   The current result does not identify the cause conclusively; float32
   cancellation/sign changes and discretization are hypotheses to test.
3. Validate member-threshold/omission sensitivity separately against a higher
   precision reference. Do not call the new probe identical to the frozen CPU
   protocol; near, opacity scaling, member threshold and integration differ.
4. Implement a concrete observation-driven training objective with no-root
   coverage handling. Merely minimizing TV/cancellation can erase occupancy,
   and dropping invalid pixels can let supervision disappear.
5. Connect that same verified depth reader to training and mesh extraction,
   then implement the matched 7k + fresh Adam, 1000-real-update comparison with
   identical schedules and native RGB. Only evaluate improvement through
   evaluate_dtu_mesh.py with matched settings; none of this is complete yet.

## Evidence integrity

Full JSON, 64-row journal, stdout/stderr, exit codes, timings, runtime and input
identity are preserved in the server run directory. A local copy is under
../gpu_evidence/scan110_20261104_1224 relative to this checkout
(absolute task workspace: C:/Users/guofa/Documents/Codex/2026-11-04/task-16).
The local and server SHA256 values match:

- probe.json: d32d65257e2c5d6624d5c4c35c4870235336a6110231b808ee1f204ce18c5f57
- probe.log: d3aac0d0147c16152145e906e684e2f0640dafe47a7405db552795d48c3e159c
- cuda_tests.log: 4a480d501e4cf206647fc1823201facf04cafb28921c30f62d47995e2b766f6e

The local probe_summary.json is derived from the preserved result with standard
JSON/statistics operations, not from additional model computation.
