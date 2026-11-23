enum Op { SCAN };

template<typename S, typename F, int op>
__device__ void evaluate(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) {
    for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < lanes; i += (U)gridDim.x * blockDim.x) {
        const U start = (i / inner) * length * inner + i % inner;
        const F inf = F(__int_as_float(0x7f800000));
        F max_value = -inf, rest = F(0);
        for (U k = 0; k < length; ++k) {
            const U j = start + k * inner;
            const F v = load<F>(x[j]);
            if (isnan(v)) max_value = v;
            else if (!isnan(max_value)) {
                if (v == inf) max_value = v;
                else if (isfinite(v) && max_value != inf) {
                    if (v > max_value) {
                        rest = (F(1) + rest) * exp(max_value - v);
                        max_value = v;
                    } else {
                        rest += exp(v - max_value);
                    }
                }
            }
            y[j] = store<S>(isfinite(max_value) ? max_value + log1p(rest) : max_value);
        }
    }
}

#define DTYPE(S, F, SUFFIX) \
    ENTRY(S, F, SUFFIX, scan, SCAN)
DTYPE(float, float, f32)
DTYPE(double, double, f64)
#ifdef NXRT_HAS_HALF
DTYPE(__half, float, f16)
DTYPE(__nv_bfloat16, float, bf16)
#endif
