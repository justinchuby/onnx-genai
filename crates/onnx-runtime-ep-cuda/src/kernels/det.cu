#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NXRT_HAS_HALF_HEADERS
#endif

using index_t = unsigned long long;

template <typename T> __device__ double widen(T x) { return (double)x; }
template <typename T> __device__ T narrow(double x) { return (T)x; }
#ifdef NXRT_HAS_HALF_HEADERS
template <> __device__ double widen(__half x) { return __half2float(x); }
template <> __device__ double widen(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __half narrow(double x) { return __double2half(x); }
template <> __device__ __nv_bfloat16 narrow(double x) { return __double2bfloat16(x); }
#endif

template <typename T>
__device__ void determinant(const T* x, T* y, double* scratch, index_t n, index_t batch) {
    const unsigned int lane = threadIdx.x;
    if (n < 2) {
        for (index_t b = (index_t)blockIdx.x * blockDim.x + lane; b < batch;
             b += (index_t)gridDim.x * blockDim.x) {
            y[b] = n == 0 ? narrow<T>(1.0) : x[b];
        }
        return;
    }

    __shared__ index_t pivot;
    __shared__ double diagonal;
    __shared__ int done;
    const index_t size = n * n;
    double* a = scratch + blockIdx.x * size;
    for (index_t b = blockIdx.x; b < batch; b += gridDim.x) {
        int has_nan = 0;
        for (index_t i = lane; i < size; i += blockDim.x) {
            const double value = widen(x[b * size + i]);
            a[i] = value;
            has_nan |= isnan(value);
        }
        if (__syncthreads_or(has_nan)) {
            if (lane == 0) y[b] = narrow<T>(nan(""));
            continue;
        }

        double mantissa = 1.0;
        long long exponent = 0;
        for (index_t k = 0; k < n; ++k) {
            if (lane == 0) {
                pivot = k;
                double max_value = fabs(a[k * n + k]);
                for (index_t r = k + 1; r < n && !isnan(max_value); ++r) {
                    const double candidate = fabs(a[r * n + k]);
                    if (candidate > max_value || isnan(candidate)) {
                        pivot = r;
                        max_value = candidate;
                    }
                }
                diagonal = a[pivot * n + k];
                done = isnan(max_value) || max_value == 0.0;
                if (done) {
                    mantissa = isnan(max_value) || isinf(mantissa) ? nan("") : 0.0;
                } else {
                    if (pivot != k) mantissa = -mantissa;
                    if (isinf(diagonal)) {
                        mantissa *= diagonal;
                    } else {
                        int power, correction;
                        const double fraction = frexp(diagonal, &power);
                        mantissa = frexp(mantissa * fraction, &correction);
                        exponent += power + correction;
                    }
                }
            }
            __syncthreads();
            if (done) break;
            if (pivot != k) {
                for (index_t c = k + lane; c < n; c += blockDim.x) {
                    const double tmp = a[k * n + c];
                    a[k * n + c] = a[pivot * n + c];
                    a[pivot * n + c] = tmp;
                }
            }
            __syncthreads();
            for (index_t r = k + 1 + lane; r < n; r += blockDim.x) {
                const double factor = a[r * n + k] / diagonal;
                if (factor == 0.0) continue;
                for (index_t c = k + 1; c < n; ++c) {
                    // Match the CPU's separately rounded elimination steps.
                    a[r * n + c] = __dsub_rn(a[r * n + c], __dmul_rn(factor, a[k * n + c]));
                }
            }
            __syncthreads();
        }
        if (lane == 0) {
            const int power = exponent < -4096 ? -4096 : exponent > 4096 ? 4096 : (int)exponent;
            y[b] = narrow<T>(ldexp(mantissa, power));
        }
    }
}

#define DEFINE_DET(T, suffix) \
extern "C" __global__ void det_##suffix(const T* x, T* y, double* scratch, \
                                       index_t n, index_t batch) { \
    determinant(x, y, scratch, n, batch); \
}
DEFINE_DET(float, f32)
DEFINE_DET(double, f64)
#ifdef NXRT_HAS_HALF_HEADERS
DEFINE_DET(__half, f16)
DEFINE_DET(__nv_bfloat16, bf16)
#endif
