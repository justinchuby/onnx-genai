using U = unsigned long long;
struct Axis { U input, output, kernel, stride, dilation, pad; };
struct Params {
  U rank, batches, channels, cg, mg, co, offset_cg, mask_c, kv, in_size, out_size;
  Axis axis[32];
};

template<class T> __device__ float widen(T x) { return (float)x; }
__device__ double widen(double x) { return x; }
template<class T, class F> __device__ T narrow(F x) { return (T)x; }
#ifdef DEFORM_CONV_HALF_TYPES
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__device__ float widen(__half x) { return __half2float(x); }
__device__ float widen(__nv_bfloat16 x) { return __bfloat162float(x); }
template<> __device__ __half narrow<__half, float>(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 narrow<__nv_bfloat16, float>(float x) { return __float2bfloat16_rn(x); }
#endif

template<class F> struct Neighbors { U lo, hi; F fraction; };
constexpr U INVALID = ~0ULL;

template<class F>
__device__ __forceinline__ bool neighbors(F pos, U size, U pitch, Neighbors<F>& v) {
  if (!size || !(pos > F(-1) && pos < F(size))) return false;
  const U low = pos < F(0) ? INVALID : (U)pos;
  const F fraction = low == INVALID ? pos + F(1) : pos - F(low);
  const U high = low == INVALID ? 0 : low + 1;
  v = {low == INVALID ? INVALID : low * pitch,
       fraction != F(0) && high < size ? high * pitch : INVALID, fraction};
  return true;
}

template<class F> __device__ __forceinline__ F coord(U origin, U pad, F offset) {
  return (origin >= pad ? F(origin - pad) : -F(pad - origin)) + offset;
}

template<class T, class F>
__device__ __forceinline__ F read(const T* x, U i) {
  return i == INVALID ? F(0) : widen(x[i]);
}

template<class T, class F>
__device__ __forceinline__ F sample_2d(const T* x, const T* off, const Params& p,
    U kh, U kw, U oh, U ow) {
  const Axis& h = p.axis[0];
  const Axis& w = p.axis[1];
  Neighbors<F> r, c;
  if (!neighbors(coord(oh * h.stride + kh * h.dilation, h.pad, F(widen(off[0]))), h.input, w.input, r)
      || !neighbors(coord(ow * w.stride + kw * w.dilation, w.pad, F(widen(off[p.out_size]))), w.input, 1, c))
    return F(0);
  const F top = r.lo == INVALID ? F(0) :
    read<T,F>(x+r.lo, c.lo) * (F(1)-c.fraction) + read<T,F>(x+r.lo, c.hi) * c.fraction;
  const F bottom = r.hi == INVALID ? F(0) :
    read<T,F>(x+r.hi, c.lo) * (F(1)-c.fraction) + read<T,F>(x+r.hi, c.hi) * c.fraction;
  return top * (F(1)-r.fraction) + bottom * r.fraction;
}

template<class T, class F>
__device__ F sample_nd(const T* x, const T* off, const Params& p, U k, U point) {
  Neighbors<F> n[32];
  U pitch = 1, choices = 0;
  for (int d = (int)p.rank - 1; d >= 0; --d) {
    const Axis& a = p.axis[d];
    const U origin = (point % a.output) * a.stride + (k % a.kernel) * a.dilation;
    if (!neighbors(coord(origin, a.pad, F(widen(off[d * p.out_size]))), a.input, pitch, n[d]))
      return F(0);
    if (n[d].lo != INVALID && n[d].hi != INVALID) choices |= 1ULL << d;
    pitch *= a.input;
    point /= a.output;
    k /= a.kernel;
  }
  F sum = F(0);
  for (U corner = choices;; corner = (corner - 1) & choices) {
    U index = 0;
    F weight = F(1);
    for (U d = 0; d < p.rank; ++d) {
      if (n[d].lo == INVALID || ((corner >> d) & 1)) {
        index += n[d].hi;
        weight *= n[d].fraction;
      } else {
        index += n[d].lo;
        weight *= F(1) - n[d].fraction;
      }
    }
    sum += widen(x[index]) * weight;
    if (corner == 0) break;
  }
  return sum;
}

// For each spatial lane reuse one sample across an output-channel tile. Adjacent
// lanes read adjacent offset/mask elements and write adjacent output elements.
template<class T, class F, bool TWO_D, int TILE>
__device__ void deform(const T* x, const T* w, const T* offset, const T* bias,
    const T* mask, T* y, const Params& p) {
  const U spatial_tiles = (p.out_size - 1) / blockDim.x + 1;
  const U channel_tiles = (p.mg - 1) / TILE + 1;
  const U groups = p.co / p.mg;
  for (U block = blockIdx.x; block < p.batches * groups * channel_tiles * spatial_tiles;
       block += gridDim.x) {
    const U position = (block % spatial_tiles) * blockDim.x + threadIdx.x;
    if (position >= p.out_size) continue;
    const U t = block / spatial_tiles;
    const U group = (t / channel_tiles) % groups;
    const U batch = t / (channel_tiles * groups);
    const U channel = group * p.mg + (t % channel_tiles) * TILE;
    const U rows = min((U)TILE, (group + 1) * p.mg - channel);
    const U k_size = p.cg * p.kv;
    const U oh = TWO_D ? position / p.axis[1].output : 0;
    const U ow = TWO_D ? position % p.axis[1].output : 0;
    F sum[TILE] = {};
    for (U c = 0; c < p.cg; ++c) {
      const U ic = group * p.cg + c;
      const U og = ic / p.offset_cg;
      for (U k = 0; k < p.kv; ++k) {
        const U slot = og * p.kv + k;
        const T* off = offset + ((batch * p.mask_c + slot) * p.rank) * p.out_size + position;
        F v = F(0);
        if (p.in_size) {
          const T* source = x + (batch * p.channels + ic) * p.in_size;
          if (TWO_D)
            v = sample_2d<T,F>(source, off, p, k / p.axis[1].kernel, k % p.axis[1].kernel, oh, ow);
          else
            v = sample_nd<T,F>(source, off, p, k, position);
        }
        if (mask) v *= widen(mask[(batch * p.mask_c + slot) * p.out_size + position]);
        const U wi = channel * k_size + c * p.kv + k;
        #pragma unroll
        for (int j = 0; j < TILE; ++j)
          if (j < rows) sum[j] += v * widen(w[wi + j * k_size]);
      }
    }
    #pragma unroll
    for (int j = 0; j < TILE; ++j)
      if (j < rows)
        y[((batch * p.co + channel + j) * p.out_size) + position] =
          narrow<T,F>(sum[j] + (bias ? F(widen(bias[channel + j])) : F(0)));
  }
}

#define ENTRY(NAME, T, F, D, TILE) \
extern "C" __global__ void NAME(const T* x, const T* w, const T* off, const T* b, \
  const T* mask, T* y, const __grid_constant__ Params p) { \
  deform<T, F, D, TILE>(x, w, off, b, mask, y, p); \
}
#define DT(NAME, T, F) \
  ENTRY(deform_##NAME##_2d, T, F, true, 8) \
  ENTRY(deform_##NAME##_2d_single, T, F, true, 1) \
  ENTRY(deform_##NAME##_nd, T, F, false, 4)

#ifdef DEFORM_CONV_HALF_TYPES
DT(f16, __half, float)
DT(bf16, __nv_bfloat16, float)
#else
DT(f32, float, float)
DT(f64, double, double)
#endif
