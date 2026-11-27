// One block owns the entire recurrence of each (direction, batch) lane.
using Index = unsigned long long;

struct Half {
    using Storage = unsigned short;
    using Acc = float;
    static __device__ float load(Storage bits) {
        float x;
        asm("cvt.f32.f16 %0, %1;" : "=f"(x) : "h"(bits));
        return x;
    }
    static __device__ Storage store(float x) {
        Storage bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(x));
        return bits;
    }
};

struct BFloat {
    using Storage = unsigned short;
    using Acc = float;
    static __device__ float load(Storage bits) {
        return __uint_as_float((unsigned int)bits << 16);
    }
    static __device__ Storage store(float x) {
        unsigned int bits = __float_as_uint(x);
        if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
        return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    }
};

template <typename T> struct Native {
    using Storage = T;
    using Acc = T;
    static __device__ T load(T x) { return x; }
    static __device__ T store(T x) { return x; }
};

template <typename T> __device__ T activate(T x, int kind, T a, T b) {
    // Discriminants match onnx_runtime_ep_api::rnn::ActivationKind.
    switch (kind) {
    case 0: return tanh(x);
    case 1: return x < T(0) ? T(0) : x;
    case 2: return T(1) / (T(1) + exp(-x));
    case 3: return a*x + b;
    case 4: return x >= T(0) ? x : a*x;
    case 5: return x >= a ? x : T(0);
    case 6: return a*tanh(b*x);
    case 7: {
        T z = a*x + b;
        return z < T(0) ? T(0) : (z > T(1) ? T(1) : z);
    }
    case 8: return x >= T(0) ? x : a*expm1(x);
    case 9: return isinf(x) ? copysign(T(1), x) : x / (T(1) + fabs(x));
    case 10: return x > T(0) ? x + log1p(exp(-x)) : log1p(exp(x));
    default: return x;
    }
}

#define ARGS(Storage) \
    const Storage* x, const Storage* w, const Storage* r, const Storage* bias, \
    const int* lengths, const Storage* initial, Storage* y, Storage* yh, \
    void* scratch, Index batches, Index time, Index inputs, Index hidden, \
    int dirs, int batch_first, int reverse_first, float clip, \
    int act0, float alpha0, float beta0, int act1, float alpha1, float beta1

template <typename D> __device__ void recurrence(ARGS(typename D::Storage)) {
    using Acc = typename D::Acc;
    for (Index lane = blockIdx.x; lane < batches * dirs; lane += gridDim.x) {
        Index dir = lane / batches, batch = lane % batches;
        Index len = lengths ? (Index)lengths[batch] : time;
        Index state = (batch_first ? batch * dirs + dir : lane) * hidden;
        if (len == 0) {
            if (yh)
                for (Index h = threadIdx.x; h < hidden; h += blockDim.x)
                    yh[state+h] = D::store(Acc(0));
        } else {
            auto prev = static_cast<Acc*>(scratch) + lane * 2 * hidden;
            auto next = prev + hidden;
            for (Index h = threadIdx.x; h < hidden; h += blockDim.x)
                prev[h] = initial ? D::load(initial[state+h]) : Acc(0);
            __syncthreads();
            bool reverse = reverse_first || dir == 1;
            for (Index step = 0; step < len; ++step) {
                Index t = reverse ? len - 1 - step : step;
                Index xi = (batch_first ? batch*time + t : t*batches + batch) * inputs;
                Index yi = (batch_first ? (batch*time + t)*dirs + dir :
                            (t*dirs + dir)*batches + batch) * hidden;
                for (Index h = threadIdx.x; h < hidden; h += blockDim.x) {
                    Index row = dir*hidden + h;
                    Acc xw = 0, hr = 0;
                    for (Index i = 0; i < inputs; ++i)
                        xw += D::load(x[xi+i]) * D::load(w[row*inputs+i]);
                    for (Index j = 0; j < hidden; ++j)
                        hr += prev[j] * D::load(r[row*hidden+j]);
                    Index bi = dir*2*hidden + h;
                    Acc b = bias ? D::load(bias[bi]) + D::load(bias[bi+hidden]) : Acc(0);
                    Acc v = xw + hr + b;
                    if (clip >= 0) v = v < Acc(-clip) ? Acc(-clip) : (v > Acc(clip) ? Acc(clip) : v);
                    v = activate(v, dir ? act1 : act0, Acc(dir ? alpha1 : alpha0), Acc(dir ? beta1 : beta0));
                    next[h] = v;
                    if (y) y[yi+h] = D::store(v);
                }
                __syncthreads();
                auto tmp = prev;
                prev = next;
                next = tmp;
            }
            if (yh)
                for (Index h = threadIdx.x; h < hidden; h += blockDim.x)
                    yh[state+h] = D::store(prev[h]);
        }
        if (y)
            for (Index t = len; t < time; ++t) {
                Index yi = (batch_first ? (batch*time + t)*dirs + dir :
                            (t*dirs + dir)*batches + batch) * hidden;
                for (Index h = threadIdx.x; h < hidden; h += blockDim.x)
                    y[yi+h] = D::store(Acc(0));
            }
    }
}

#define ENTRY(name, Codec) \
extern "C" __global__ void name(ARGS(Codec::Storage)) { \
    recurrence<Codec>(x, w, r, bias, lengths, initial, y, yh, scratch, batches, \
                      time, inputs, hidden, dirs, batch_first, reverse_first, clip, \
                      act0, alpha0, beta0, act1, alpha1, beta1); \
}
ENTRY(rnn_f16, Half)
ENTRY(rnn_bf16, BFloat)
ENTRY(rnn_f32, Native<float>)
ENTRY(rnn_f64, Native<double>)
