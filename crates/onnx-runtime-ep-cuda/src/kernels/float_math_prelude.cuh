#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_HALF 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

template<typename S, typename F> __device__ S store(F v) { return v; }
template<typename F, typename S> __device__ F load(S v) { return v; }
#ifdef NXRT_HAS_HALF
template<> __device__ __half store<__half, float>(float v) { return __float2half_rn(v); }
template<> __device__ __nv_bfloat16 store<__nv_bfloat16, float>(float v) { return __float2bfloat16_rn(v); }
template<> __device__ float load<float, __half>(__half v) { return __half2float(v); }
template<> __device__ float load<float, __nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename S, typename F> __device__ F round_storage(F v) { return load<F>(store<S>(v)); }
using U = unsigned long long;
#define ENTRY(S, F, SUFFIX, NAME, OP) \
extern "C" __global__ void NAME##_##SUFFIX(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) { \
    evaluate<S, F, OP>(x, b, y, lanes, length, inner, alpha); \
}
