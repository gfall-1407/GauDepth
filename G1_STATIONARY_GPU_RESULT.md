# G1 stationary-TV gradient repair: observed refinement passes on valid roots

The repair now passes the declared 4-to-8 refinement checks on all 48 previously
valid scan110 probe rays; all 64 rays were evaluated. Sixteen rays still have no
detected root and are NOT counted as converged valid roots. This establishes a
useful numerical core for further integration, not a 30k training result, full
scene solver certificate, mesh-quality improvement or novelty claim.

## Source, execution and identity

The explicitly authorized push of `0df0fb95cca82d0c0ee69125f2870afd1bda3002`
succeeded; its exact remote SHA was checked. Its real GPU test exposed a tail
direction underflow problem. The independently reviewed follow-up
`a4156a57c2ef6e097e7f23bbc81a81d5b26b97b8` was then pushed to the same branch
`research/joint-tv-train-20261104`, remote gfall-1407/GauDepth, and its remote SHA
was verified. No main branch change, non-Git source transfer or bypass occurred.

Both runs used G1 `autodl-container-d6a7459179-693ab6ee`, RTX4090D,
`/root/miniconda3/envs/radegs/bin/python`, and the clean isolated checkout
`/root/joint_tv_code_6d64509`. New outputs remained on the system disk:

- First repair: `/root/joint_tv_runs/scan110_stationary_0df0fb9_VQNWS4`.
  Five CUDA tests passed in 1.710 s. Probe exit 0, internal 29.132 s,
  UTC 13:07:43 to 13:08:17.
- Tail repair: `/root/joint_tv_runs/scan110_stationary_a4156a5_WJ6Z5p`.
  Seven CUDA tests passed in 2.026 s. Probe exit 0, internal 34.945 s,
  UTC 13:13:57 to 13:14:37. Both used external 120s/660s limits plus 10s grace.

Input scan110 original 7k PLY SHA256 remained
`733bd5ad69246915e7777b740b1891a253624b1d2232df5f19e79a557de78aa7`
before and after both runs. COLMAP hashes matched the registry. Same view0049,
8x8 pixels, field opacity/near/far and full-line member cutoff=1e-6 as the old
probe; no member truncation. Row journals exactly match final reports, 64 rows.
All original models/data were read-only. The instance was not shut down.

## Numerical findings

| Computation | Valid roots | xyz gradient relative-L2 median, 4-to-8 | rotation gradient relative-L2 median, 4-to-8 | Complete valid-root passes |
|---|---:|---:|---:|---:|
| Original sampled TV, float32 | 48 | 22.89% | 40.53% | 0/48 |
| Same sampled-TV method, full float64 parameter chain | 48 | 2.25% | 4.32% | 0/48 |
| Stationary TV after tail repair | 48 | 2.18e-19 | 3.08e-18 | 48/48 |

The last row's maximum relative differences are 2.11e-13 (xyz), 5.17e-14
(log-scale), 2.52e-13 (rotation), 6.63e-15 (opacity logit). Root depth change is
zero at reported precision; total-TV maximum change is 4.44e-16. Stationary
positions agree to <=1.42e-14 on matching branches. Existing acceptance
thresholds were not relaxed.

Precision contributes substantially, but changing precision alone did not make
sampled integration pass. Float32 had zero H differences on some positive-width,
nonzero same-sign derivative intervals (median 3 per ray, max19); float64 had
none in this diagnostic. This is supporting evidence, not a proof that these
specific intervals explain all gradient errors: the full parameter/profile
chain and knot precision also changed.

The first stationary version had only 39 valid roots at 2/4 levels because its
midpoint derivative could underflow to zero in far Gaussian tails. The follow-up
computes derivative signs using a common positive log-domain rescaling, while
keeping the original H, actual root-slope guard and residual calculations.
Analytic narrow-Gaussian regression reproduces the underflow and verifies the
correct root and gradient. All nine additional invalids are recovered at levels
2/4/8. No rootless ray was converted into an invented fallback depth.

Lower refinements are retained: 1-to-2 passes43/47 comparable valid roots and
changes one validity status; 2-to-4 passes47/48 because ray19 gains two extrema
AFTER its first root (depth and gradient unchanged). At4-to8, all48 valid rays
have matching branches. Rootless ray9 still gains two extrema, so only63/64
whole-field branch signatures agree. No claim of certified absence of a root.

Thirty-two random-direction finite differences passed: four parameter groups,
two step sizes, first four valid rays. Every perturbed forward independently
searched stationary points and the first root. Max absolute error4.19e-7. This
does not constitute finite-difference testing of every ray or parameter direction.

## Cost and next boundary

The latest full comparison/FD probe used peak torch allocated memory
1,222,460,416 bytes (1.139GiB), not total GPU reserved memory. Stationary8 forward
plus backward median is63.04ms/ray; the64 such evaluations sum3.861s. Stationary4
median is50.27ms/ray. These are one-ray prototype costs, not complete training
step or full-view extraction timings.

Do not launch23,000 new updates solely from this result. Remaining concrete work:
member-cutoff sensitivity on a fixed support union; native RGB/track-pair
training integration and real Adam-step validation; batched acceleration and
candidate extraction using the same field/solver. The 30k plan specifies
identical original7k arrays, fresh Adam in both arms,23,000 actual updates and
post-update snapshots. A completed trainer and practical full-view candidate
extractor do not yet exist in this branch. No final training ETA is measured.

## Evidence

Repository summaries retain both runs in `evidence/g1_stationary_*_summary.json`,
plus the seven-test CUDA log. Full JSON, journal, stdout, statuses and timestamps
are also downloaded under the task's `gpu_evidence` directory.
Latest probe JSON SHA256:
`67219e3d613a7160aeea9973e902e5eb858c0a684c5b6cdbc23a5217f7f3e7d0`.
Latest probe log:
`edb44455f11ca2302a2c1e13ae85344787a9432a0588264dd27e753fd6528bea`.
CUDA log:
`04a376f9da43a307f5f3532bb6c8f73fa8cb71f8d03a9ec00d6935ca2ddbaeb0`.
Local hashes match the server hash manifest.
