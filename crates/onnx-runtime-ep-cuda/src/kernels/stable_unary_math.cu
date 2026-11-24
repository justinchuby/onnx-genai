enum Op { EXPM1, LOG1P };

template<typename S, typename F, int op>
__device__ void evaluate(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) {
    for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < lanes; i += (U)gridDim.x * blockDim.x) {
        const F a = load<F>(x[i]);
        F out;
        if (op == EXPM1) out = expm1(a);
        else out = log1p(a);
        y[i] = store<S>(out);
    }
}

#define DTYPE(S, F, SUFFIX) \
    ENTRY(S, F, SUFFIX, expm1, EXPM1) \
    ENTRY(S, F, SUFFIX, log1p, LOG1P)
DTYPE(float, float, f32)
DTYPE(double, double, f64)
#ifdef NXRT_HAS_HALF
DTYPE(__half, float, f16)
DTYPE(__nv_bfloat16, float, bf16)
#endif
