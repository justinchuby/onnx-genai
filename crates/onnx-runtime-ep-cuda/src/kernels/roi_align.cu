using U64 = unsigned long long;
using I64 = long long;

__device__ double load(const void* ptr, int dtype, U64 i) {
    if (dtype == 0) return static_cast<const float*>(ptr)[i];
    if (dtype == 1) return static_cast<const double*>(ptr)[i];
#ifdef ROI_ALIGN_HALF
    if (dtype == 2) return __half2float(static_cast<const __half*>(ptr)[i]);
    return __bfloat162float(static_cast<const __nv_bfloat16*>(ptr)[i]);
#else
    return 0.0;
#endif
}

__device__ void store(void* ptr, int dtype, U64 i, double value) {
    if (dtype == 0) static_cast<float*>(ptr)[i] = static_cast<float>(value);
    else if (dtype == 1) static_cast<double*>(ptr)[i] = value;
#ifdef ROI_ALIGN_HALF
    else if (dtype == 2) static_cast<__half*>(ptr)[i] = __float2half_rn(static_cast<float>(value));
    else static_cast<__nv_bfloat16*>(ptr)[i] = __float2bfloat16_rn(static_cast<float>(value));
#endif
}

struct Axis {
    double start;
    double bin;
    I64 samples;
};

__device__ bool axis(Axis& a, const void* rois, U64 i, int dtype, double scale,
                     U64 bins, I64 ratio, bool half_pixel) {
    const double shift = half_pixel ? 0.5 : 0.0;
    a.start = load(rois, dtype, i) * scale - shift;
    double extent = (load(rois, dtype, i + 2) * scale - shift) - a.start;
    if (!isfinite(extent)) return false;
    if (!half_pixel) extent = fmax(extent, 1.0);
    a.bin = extent / static_cast<double>(bins);
    if (ratio > 0) {
        a.samples = ratio;
    } else {
        const double count = fmax(ceil(a.bin), 0.0);
        if (count >= 9223372036854775808.0) return false;
        a.samples = static_cast<I64>(count);
    }
    return true;
}

__device__ double coordinate(Axis a, U64 bin, I64 sample) {
    return a.start + static_cast<double>(bin) * a.bin
        + (static_cast<double>(sample) + 0.5) * a.bin / static_cast<double>(a.samples);
}

__device__ double sample(const void* x, int dtype, U64 base, U64 h, U64 w,
                         double y, double u, bool max_pool) {
    if (h == 0 || w == 0 || !(y >= -1.0 && y <= static_cast<double>(h))
        || !(u >= -1.0 && u <= static_cast<double>(w))) return 0.0;
    y = fmin(fmax(y, 0.0), static_cast<double>(h - 1));
    u = fmin(fmax(u, 0.0), static_cast<double>(w - 1));
    const U64 yl = static_cast<U64>(y);
    const U64 xl = static_cast<U64>(u);
    const U64 yh = yl == h - 1 ? yl : yl + 1;
    const U64 xh = xl == w - 1 ? xl : xl + 1;
    const double ly = y - static_cast<double>(yl);
    const double lx = u - static_cast<double>(xl);
    const double hy = 1.0 - ly, hx = 1.0 - lx;
    const double a = (hy * hx) * load(x, dtype, base + yl * w + xl);
    const double b = (hy * lx) * load(x, dtype, base + yl * w + xh);
    const double c = (ly * hx) * load(x, dtype, base + yh * w + xl);
    const double d = (ly * lx) * load(x, dtype, base + yh * w + xh);
    if (!max_pool) return a + b + c + d;
    // The pinned ONNX reference takes the maximum of the weighted corners.
    double m = b > a ? b : a;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

extern "C" __global__ void roi_align(
    const void* x, const void* rois, const I64* indices, void* output,
    unsigned int* status, U64 work, U64 output_count, U64 n, U64 c, U64 h, U64 w,
    U64 oh, U64 ow, I64 ratio, double scale, int dtype, int half_pixel, int max_pool) {
    const U64 spatial = oh * ow;
    for (U64 i = blockIdx.x * blockDim.x + threadIdx.x; i < work;
         i += static_cast<U64>(gridDim.x) * blockDim.x) {
        // Even when C is zero, every supplied ROI still has a batch index.
        const U64 r = output_count == 0 ? i : i / (c * spatial);
        const I64 batch = indices[r];
        if (batch < 0 || static_cast<U64>(batch) >= n) {
            atomicOr(status, 1u);
            continue;
        }
        Axis ax, ay;
        if (!axis(ax, rois, r * 4, dtype, scale, ow, ratio, half_pixel != 0)
            || !axis(ay, rois, r * 4 + 1, dtype, scale, oh, ratio, half_pixel != 0)) {
            atomicOr(status, 2u);
            continue;
        }
        if (output_count == 0) continue;
        const U64 base = (static_cast<U64>(batch) * c + (i / spatial) % c) * h * w;
        const U64 ph = (i / ow) % oh, pw = i % ow;
        double result = 0.0;
        if (ax.samples != 0 && ay.samples != 0) {
            for (I64 iy = 0; iy < ay.samples; ++iy) {
                const double y = coordinate(ay, ph, iy);
                for (I64 ix = 0; ix < ax.samples; ++ix) {
                    const double value = sample(x, dtype, base, h, w, y,
                                                coordinate(ax, pw, ix), max_pool != 0);
                    if (!max_pool) result += value;
                    else if ((iy == 0 && ix == 0) || value > result) result = value;
                }
            }
            if (!max_pool) result /= static_cast<double>(ax.samples) * static_cast<double>(ay.samples);
        }
        store(output, dtype, i, result);
    }
}
