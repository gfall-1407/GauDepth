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
// Twice the total violation of increasing-then-decreasing order around the
// first maximum. For nonnegative masses this equals TV(m) - 2*max(m), but
// avoids cancellation between two nearly equal large quantities.
__device__ inline float segmentExcess(const float* t, int bins, int peak_bin) {
    float violation = 0.f;
    for (int k = 0; k + 1 < bins; ++k) {
        const float a = t[k] - t[k + 1];
        const float b = t[k + 1] - t[k + 2];
        violation += k < peak_bin ? fmaxf(a - b, 0.f) : fmaxf(b - a, 0.f);
    }
    return 2.f * violation;
}
// The branch at equality uses zero as the hinge subgradient. The first maximum
// is selected deterministically. Empty and already-unimodal rays have zero gradient.
__device__ inline void segmentBoundaryGradients(const float* t, int bins,
        float epsilon, float upstream, float* dt) {
    float variation, peak; int argmax;
    segmentStats(t, bins, variation, peak, argmax);
    const float excess = segmentExcess(t, bins, argmax);
    const float mass = t[0] - t[bins];
    for (int j = 0; j <= bins; ++j) dt[j] = 0.f;
    if (excess <= 0.f || mass <= 0.f || peak <= 0.f) return;
    const float denominator = 2.f * peak + epsilon * mass;
    for (int k = 0; k < bins; ++k) {
        const float x = t[k] - t[k + 1];
        float de = 0.f;
        if (k > 0) {
            const float left = t[k - 1] - t[k];
            if (k - 1 < argmax && left > x) de -= 2.f;
            if (k - 1 >= argmax && x > left) de += 2.f;
        }
        if (k + 1 < bins) {
            const float right = t[k + 1] - t[k + 2];
            if (k < argmax && x > right) de += 2.f;
            if (k >= argmax && right > x) de -= 2.f;
        }
        const float h = k == argmax ? 1.f : 0.f;
        const float g = (upstream / denominator) *
            (de - (excess / denominator) * (2.f * h + epsilon));
        dt[k] += g; dt[k + 1] -= g;
    }
}
