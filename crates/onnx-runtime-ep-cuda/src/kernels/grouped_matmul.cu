#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NXRT_HAS_HALF 1
#endif

template<typename S, typename F> __device__ S store(F v) { return v; }
template<typename F, typename S> __device__ F load(S v) { return v; }
#ifdef NXRT_HAS_HALF
template<> __device__ __half store<__half, float>(float v) { return __float2half_rn(v); }
template<> __device__ __nv_bfloat16 store<__nv_bfloat16, float>(float v) { return __float2bfloat16_rn(v); }
template<> __device__ float load<float, __half>(__half v) { return __half2float(v); }
template<> __device__ float load<float, __nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename S, typename F, typename I>
__device__ void grouped(const B* x, const B* w, const B* indices, const B* b, B* y,
                       unsigned* status, U routes, U lanes, U groups, U selections, U k, U n, int transB) {
    GRID(i, n == 0 ? routes : lanes) {
        const U route = n == 0 ? i : i / n;
        const I group = read<I>(indices, route);
        if (group < 0 || (U)group >= groups) {
            atomicOr(status, 1u);
            continue;
        }
        if (n == 0) continue;
        const U token = route / selections;
        const U col = i % n;
        F sum = F(0);
        for (U j = 0; j < k; ++j) {
            const U offset = transB ? col * k + j : j * n + col;
            sum += load<F>(read<S>(x, token * k + j)) * load<F>(read<S>(w, (U)group * k * n + offset));
        }
        const S dot = store<S>(sum);
        write<S>(y, i, b ? store<S>(load<F>(dot) + load<F>(read<S>(b, (U)group * n + col))) : dot);
    }
}

#define GROUP_ENTRY(S, F, I, NAME) \
extern "C" __global__ void NAME(const B* x, const B* w, const B* idx, const B* b, B* y, \
                    unsigned* status, U routes, U lanes, U groups, U selections, U k, U n, int transB) { \
    grouped<S, F, I>(x,w,idx,b,y,status,routes,lanes,groups,selections,k,n,transB); \
}
#define GROUP_TYPE(S,F,TAG) \
    GROUP_ENTRY(S,F,int,grouped_##TAG##_i32) \
    GROUP_ENTRY(S,F,long long,grouped_##TAG##_i64)
GROUP_TYPE(float,float,f32)
GROUP_TYPE(double,double,f64)

template<typename I>
__device__ void affine_grouped(const B* x, const B* w, const B* indices, const B* scales,
    const B* zeros, B* y, unsigned* status, U routes, U lanes, U groups, U selections,
    U k, U n, U block_size, int zero) {
    const U blocks = k / block_size;
    GRID(i, groups * n * blocks) {
        const float scale = read<float>(scales, i);
        if (!(scale > 0.0f) || !isfinite(scale)) atomicOr(status, 2u);
    }
    GRID(i, n == 0 ? routes : lanes) {
        const U route = n == 0 ? i : i / n;
        const I group = read<I>(indices, route);
        if (group < 0 || (U)group >= groups) {
            atomicOr(status, 1u);
            continue;
        }
        if (n == 0) continue;
        const U row = (U)group * n + i % n;
        const U token = route / selections;
        write<float>(y, i, affine_grouped_dot(x, w, scales, zeros, row, token, k, block_size, zero));
    }
}

#define AFFINE_GROUP_ENTRY(I, NAME) \
extern "C" __global__ void NAME(const B* x, const B* w, const B* idx, const B* scales, \
    const B* zeros, B* y, unsigned* status, U routes, U lanes, U groups, U selections, \
    U k, U n, U block_size, int zero) { \
    affine_grouped<I>(x,w,idx,scales,zeros,y,status,routes,lanes,groups,selections,k,n,block_size,zero); \
}
AFFINE_GROUP_ENTRY(int, affine_grouped_i32)
AFFINE_GROUP_ENTRY(long long, affine_grouped_i64)
#ifdef NXRT_HAS_HALF
GROUP_TYPE(__half,float,f16)
GROUP_TYPE(__nv_bfloat16,float,bf16)
#endif
