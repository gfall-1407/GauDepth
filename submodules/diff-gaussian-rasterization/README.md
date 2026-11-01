# Differential Gaussian Rasterization

**NOTE**: this is a modified version to support depth & alpha rendering (both forward and backward) from the [original repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization). 

## Accepted Gaussian query (optional)

Pass `support_xy` (contiguous CUDA float32 `[Q,2]`, image pixel coordinates) and
`support_k` (default 128) to `GaussianRasterizer.forward` or `sample_depth`.
The normal return values are followed by `ids[Q,K]`, `support_alpha[Q,K]`,
`count[Q]`, `overflow[Q]`. `ids` contains original Gaussian tensor row indices,
with `-1` padding; `count` is the full accepted count even when it exceeds K.
An overflowed query must use the host depth instead of treating the prefix as
complete. The optional outputs have no gradient. Gather the current Gaussian
parameters by `ids` to construct a differentiable depth. `pixel_grid_xy(H,W,device)`
returns the full raster grid in row-major order. Fractional coordinates are
evaluated as provided, without pixel rounding. Invalid/out-of-image queries
return zero count and padded IDs.
Queries require CUDA's default stream because the existing native forward
launches there; non-default streams are rejected before that forward runs.

The query reads the native sorted tile list and per-Gaussian conic/opacity from
the same forward pass. It applies the original `power > 0`, `alpha < 1/255`, and
`test_T < 1e-4` acceptance/stop rules. `support_alpha` is per-Gaussian alpha,
not transmittance-weighted contribution. The full-grid storage cost is
`8*H*W*K` bytes for IDs and alpha alone; use query chunks if needed.

Local static checks: `python -m py_compile diff_gaussian_rasterization/__init__.py`
and `git diff --check`. The predicates in `support_forward.cu` should be
compared with both `render_forward.cu` and `sample_forward.cu`; CUDA build,
GPU membership equality, and gradient checks require a matching PyTorch/CUDA
environment and have not been run in the Windows source workspace.

```python
rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=cov3D_precomp,
)
```


Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
