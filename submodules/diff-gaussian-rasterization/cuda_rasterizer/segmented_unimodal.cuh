#pragma once
#include <math.h>
// Piecewise transmittance: each accepted Gaussian attenuates at its existing
// per-pixel ray-plane representative depth. Hard depth membership is held fixed
// in backward; this is NOT continuous Gaussian attenuation (nor GGGS).
constexpr int UNIMODAL_MAX_BINS = 32;
__device__ inline float segmentBoundary(int j, int bins, float near, float far) {
    return near + (far - near) * (float(j) / float(bins));
}
__device__ inline void segmentAttenuate(float* t, int bins, float near, float far,
                                      float depth, float alpha) {
    if (!isfinite(depth)) return;
    for (int j = 0; j <= bins; ++j)
        if (depth < segmentBoundary(j, bins, near, far))
            t[j] *= 1.f - alpha;
}
__device__ inline void segmentStats(const float* t, int bins,
                                   float& variation, float& peak, int& argmax) {
    variation = peak = 0.f; argmax = 0;
    float previous = 0.f;
    for (int k = 0; k < bins; ++k) {
        const float mass = t[k] - t[k + 1];
        variation += fabsf(mass - previous);
        if (mass > peak) { peak = mass; argmax = k; }
        previous = mass;
    }
    variation += fabsf(previous);
}
__device__ inline float segmentSign(float v) { return (v > 0.f) - (v < 0.f); }
// Raw-mass form of normalized p loss:
// relu(V_m - 2 M_m) / (2 M_m + epsilon * A).
__device__ inline void segmentBoundaryGradients(const float* t, int bins,
        float epsilon, float upstream, float* dt) {
    float v, m; int argmax;
    segmentStats(t, bins, v, m, argmax);
    const float excess = v - 2.f * m, a = t[0] - t[bins];
    for (int j = 0; j <= bins; ++j) dt[j] = 0.f;
    if (excess <= 0.f || a <= 0.f || m <= 0.f) return;
    const float denominator = 2.f * m + epsilon * a;
    for (int k = 0; k < bins; ++k) {
        const float x = t[k] - t[k + 1];
        const float left = k ? t[k - 1] - t[k] : 0.f;
        const float right = k + 1 < bins ? t[k + 1] - t[k + 2] : 0.f;
        const float h = k == argmax ? 1.f : 0.f;
        const float dv = segmentSign(x - left) + segmentSign(x - right);
        const float g = upstream * ((dv - 2.f * h) / denominator
            - excess * (2.f * h + epsilon) / (denominator * denominator));
        dt[k] += g; dt[k + 1] -= g;
    }
}
