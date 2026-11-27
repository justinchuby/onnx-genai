#ifdef MAX_ROI_POOL_HALF_TYPES
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

using u64 = unsigned long long;
using i64 = long long;

struct Bounds {
    u64 begin, end;
};

__device__ u64 clip(i64 edge, u64 limit) {
    if (edge <= 0) return 0;
    return min(static_cast<u64>(edge), limit);
}

__device__ Bounds bin(i64 start, i64 end, i64 index, i64 pooled, u64 limit) {
    // Both endpoints are included in the ROI, but the bin end is exclusive.
    i64 extent = max(end - start + 1, 1LL);
    return {
        clip(start + index * extent / pooled, limit),
        clip(start + ((index + 1) * extent + pooled - 1) / pooled, limit),
    };
}

__device__ float widen(float v) { return v; }
__device__ double widen(double v) { return v; }
#ifdef MAX_ROI_POOL_HALF_TYPES
__device__ float widen(__half v) { return __half2float(v); }
__device__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template <typename T>
__device__ void pool(const T* x, const T* rois, T* y, u64 total, u64 batches,
                     u64 channels, u64 height, u64 width, int ph, int pw,
                     float scale, unsigned* error) {
    using F = decltype(widen(T{}));
    for (u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<u64>(gridDim.x) * blockDim.x) {
        u64 channel_index = i / pw / ph;
        const T* roi = rois + channel_index / channels * 5;
        double batch = widen(roi[0]);
        if (!(batch >= 0.0 && batch < static_cast<double>(batches)
              && trunc(batch) == batch)) {
            atomicOr(error, 1U);
            continue;
        }
        i64 corners[4];
        bool valid = true;
        for (int j = 0; j < 4; ++j) {
            double rounded = round(static_cast<double>(widen(roi[j + 1]) * static_cast<F>(scale)));
            if (!(rounded >= -2147483648.0 && rounded <= 2147483647.0)) {
                valid = false;
                break;
            }
            corners[j] = static_cast<i64>(rounded);
        }
        if (!valid) {
            atomicOr(error, 2U);
            continue;
        }

        Bounds rows = bin(corners[1], corners[3], i / pw % ph, ph, height);
        Bounds cols = bin(corners[0], corners[2], i % pw, pw, width);
        if (rows.begin == rows.end || cols.begin == cols.end) {
            y[i] = static_cast<T>(0.0f);
            continue;
        }
        F best = static_cast<F>(-__int_as_float(0x7f800000));
        u64 plane = (static_cast<u64>(batch) * channels + channel_index % channels) * height;
        for (u64 h = rows.begin; h < rows.end; ++h) {
            for (u64 w = cols.begin; w < cols.end; ++w) {
                F value = widen(x[(plane + h) * width + w]);
                if (value > best) best = value;
            }
        }
        y[i] = static_cast<T>(best);
    }
}

#define ENTRY(tag, T) \
extern "C" __global__ void max_roi_pool_##tag( \
    const T* x, const T* rois, T* y, u64 total, u64 batches, u64 channels, \
    u64 height, u64 width, int ph, int pw, float scale, unsigned* error) { \
    pool(x, rois, y, total, batches, channels, height, width, ph, pw, scale, error); \
}

ENTRY(f32, float)
ENTRY(f64, double)
#ifdef MAX_ROI_POOL_HALF_TYPES
ENTRY(f16, __half)
ENTRY(bf16, __nv_bfloat16)
#endif
