enum Op { SWIGLU, GEGLU, GEGLU_TANH };

template<typename S, typename F, int op>
__device__ void evaluate(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) {
    for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < lanes; i += (U)gridDim.x * blockDim.x) {
        const F a = load<F>(x[i]);
        F out;
        if (op == SWIGLU) {
            const F z = round_storage<S>(round_storage<S>(F(alpha)) * a);
            const F e = exp(-fabs(z));
            const F sigmoid = round_storage<S>(z >= F(0) ? F(1) / (F(1) + e) : e / (F(1) + e));
            out = round_storage<S>(a * sigmoid) * load<F>(b[i]);
        } else {
            const F phi = op == GEGLU_TANH
                ? tanh(F(0.7978845608028654) * (a + F(0.044715) * a * a * a))
                : erf(a / F(1.4142135623730951));
            out = ((F(0.5) * a) * (F(1) + phi)) * load<F>(b[i]);
        }
        y[i] = store<S>(out);
    }
}

#define DTYPE(S, F, SUFFIX) \
    ENTRY(S, F, SUFFIX, swiglu, SWIGLU) \
    ENTRY(S, F, SUFFIX, geglu, GEGLU) \
    ENTRY(S, F, SUFFIX, geglu_tanh, GEGLU_TANH)
DTYPE(float, float, f32)
DTYPE(double, double, f64)
#ifdef NXRT_HAS_HALF
DTYPE(__half, float, f16)
DTYPE(__nv_bfloat16, float, bf16)
#endif
