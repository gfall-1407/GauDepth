#include "config.h"
#include "rasterizer_impl.h"
#include <torch/extension.h>
#include <cmath>
#include <tuple>

namespace {
__global__ void acceptedCUDA(
    const float2* xy, int Q, int K, int W, int H,
    const uint2* ranges, const uint32_t* sorted_ids,
    const float2* means2D, const float4* conic_opacity,
    int* ids, float* alphas, int* counts, bool* overflow) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q) return;
    const float2 pix = xy[q];
    if (!isfinite(pix.x) || !isfinite(pix.y) || pix.x < 0.f || pix.y < 0.f ||
        pix.x > W - 1 || pix.y > H - 1) return;
    const int tile_x = static_cast<int>(pix.x) / BLOCK_X;
    const int tile_y = static_cast<int>(pix.y) / BLOCK_Y;
    const int tiles_w = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 range = ranges[tile_y * tiles_w + tile_x];
    float T = 1.f;
    int count = 0;
    for (uint32_t j = range.x; j < range.y; ++j) {
        const uint32_t id = sorted_ids[j];
        const float2 mean = means2D[id];
        const float4 con_o = conic_opacity[id];
        const float2 d = {mean.x - pix.x, mean.y - pix.y};
        const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;
        const float alpha = fminf(0.99f, con_o.w * expf(power));
        if (alpha < 1.0f / 255.0f) continue;
        const float test_T = T * (1.f - alpha);
        if (test_T < 0.0001f) break;  // Candidate is not accepted by render/sample.
        if (count < K) {
            ids[q * K + count] = static_cast<int>(id);
            alphas[q * K + count] = alpha;
        }
        ++count;
        T = test_T;
    }
    counts[q] = count;
    overflow[q] = count > K;
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
AcceptedGaussiansCUDA(
    const torch::Tensor& xy,
    const torch::Tensor& geom_buffer,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& range_buffer,
    int P, int R, int W, int H, int K, bool sampled) {
    TORCH_CHECK(xy.is_cuda() && xy.scalar_type() == torch::kFloat32 &&
                xy.dim() == 2 && xy.size(1) == 2 && xy.is_contiguous(),
                "xy must be contiguous CUDA float32 [Q,2]");
    TORCH_CHECK(P >= 0 && R >= 0 && W > 0 && H > 0 && K > 0, "invalid support dimensions");
    TORCH_CHECK(geom_buffer.is_cuda() && binning_buffer.is_cuda() && range_buffer.is_cuda(),
                "raster buffers must be CUDA tensors");
    TORCH_CHECK(xy.device() == geom_buffer.device() && xy.device() == binning_buffer.device() &&
                xy.device() == range_buffer.device(), "query and raster buffers must share a device");
    const int Q = static_cast<int>(xy.size(0));
    auto ids = torch::full({Q, K}, -1, xy.options().dtype(torch::kInt32));
    auto alpha = torch::zeros({Q, K}, xy.options());
    auto count = torch::zeros({Q}, xy.options().dtype(torch::kInt32));
    auto overflow = torch::zeros({Q}, xy.options().dtype(torch::kBool));
    if (!Q || !P || !R) return {ids, alpha, count, overflow};
    TORCH_CHECK(geom_buffer.numel() >= CudaRasterizer::required<CudaRasterizer::GeometryState>(P),
                "geometry buffer too small");
    TORCH_CHECK(binning_buffer.numel() >= CudaRasterizer::required<CudaRasterizer::BinningState>(R),
                "binning buffer too small");
    const size_t tile_count = ((W + BLOCK_X - 1) / BLOCK_X) * ((H + BLOCK_Y - 1) / BLOCK_Y);
    TORCH_CHECK(range_buffer.numel() >= (sampled
                ? CudaRasterizer::required<CudaRasterizer::TileState>(tile_count)
                : CudaRasterizer::required<CudaRasterizer::ImageState>(W * H)),
                "range buffer too small");
    char* g = reinterpret_cast<char*>(geom_buffer.data_ptr());
    char* b = reinterpret_cast<char*>(binning_buffer.data_ptr());
    char* r = reinterpret_cast<char*>(range_buffer.data_ptr());
    auto geom = CudaRasterizer::GeometryState::fromChunk(g, P);
    auto bin = CudaRasterizer::BinningState::fromChunk(b, R);
    const uint2* ranges = sampled
        ? CudaRasterizer::TileState::fromChunk(r, tile_count).gaussian_ranges
        : CudaRasterizer::ImageState::fromChunk(r, W * H).ranges;
    {
        acceptedCUDA<<<(Q + 127) / 128, 128>>>(
            reinterpret_cast<const float2*>(xy.data_ptr<float>()), Q, K, W, H,
            ranges, bin.point_list, geom.means2D, geom.conic_opacity,
            ids.data_ptr<int>(), alpha.data_ptr<float>(), count.data_ptr<int>(),
            overflow.data_ptr<bool>());
        auto error = cudaGetLastError();
        TORCH_CHECK(error == cudaSuccess, "acceptedCUDA: ", cudaGetErrorString(error));
    }
    return {ids, alpha, count, overflow};
}
