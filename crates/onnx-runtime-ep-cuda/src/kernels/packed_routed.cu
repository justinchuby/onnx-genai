using U = unsigned long long;
using B = unsigned char;
#define GRID(i, n) for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (U)gridDim.x * blockDim.x)

__device__ U mask(int bits) { return (1ull << bits) - 1; }

// Tensor origins may be unaligned, even when the elements are contiguous.
template<typename T> __device__ T read(const B* base, U i) {
    const B* p = base + i * sizeof(T);
    if ((reinterpret_cast<U>(p) & (alignof(T) - 1)) == 0)
        return *reinterpret_cast<const T*>(p);
    T value;
    B* bytes = reinterpret_cast<B*>(&value);
    #pragma unroll
    for (int j = 0; j < sizeof(T); ++j) bytes[j] = p[j];
    return value;
}

template<typename T> __device__ void write(B* base, U i, T value) {
    B* p = base + i * sizeof(T);
    if ((reinterpret_cast<U>(p) & (alignof(T) - 1)) == 0) {
        *reinterpret_cast<T*>(p) = value;
        return;
    }
    const B* bytes = reinterpret_cast<const B*>(&value);
    #pragma unroll
    for (int j = 0; j < sizeof(T); ++j) p[j] = bytes[j];
}

template<typename Word, bool unpack>
__device__ void packing(const B* input, B* output, const B* count_input,
                        unsigned* status, U lanes, U count, U words, U inner,
                        U radix, U group, U bits, U limit) {
    constexpr U width = sizeof(Word) * 8;
    if (unpack && read<long long>(count_input, 0) != (long long)count) {
        if (blockIdx.x == 0 && threadIdx.x == 0) *status = 2;
        return;
    }
    GRID(i, lanes) {
        const U span = unpack ? count : words;
        const U j = (i / inner) % span;
        const U outer = i / inner / span;
        const U lane = i % inner;
        if (unpack) {
            U bit = (j / group) * bits;
            U index = 0;
            for (U used = 0; used < bits;) {
                const U off = bit % width;
                const U take = min(bits - used, width - off);
                const U word = read<Word>(input, (outer * words + bit / width) * inner + lane);
                index |= ((word >> off) & mask(take)) << used;
                used += take;
                bit += take;
            }
            if (index >= limit) {
                atomicOr(status, 1u);
                continue;
            }
            for (U k = j % group; k != 0; --k) index /= radix;
            output[i] = index % radix;
        } else {
            const U word_bit = j * width;
            const U groups = count / group + (count % group != 0);
            U value = 0;
            for (U g = word_bit / bits; g < groups && g * bits < word_bit + width; ++g) {
                const U start = g * group;
                U index = 0;
                for (U k = min(group, count - start); k != 0; --k) {
                    const U digit = input[(outer * count + start + k - 1) * inner + lane];
                    if (digit >= radix) atomicOr(status, 1u);
                    index = index * radix + digit % radix;
                }
                const U bit = g * bits;
                value |= bit < word_bit ? index >> (word_bit - bit) : index << (bit - word_bit);
            }
            write<Word>(output, i, value);
        }
    }
}

#define PACK_ENTRY(WORD, NAME, UNPACK) \
extern "C" __global__ void NAME(const B* x, B* y, const B* count_input, unsigned* status, \
                  U lanes, U count, U words, U inner, U radix, U group, U bits, U limit) { \
    packing<WORD, UNPACK>(x,y,count_input,status,lanes,count,words,inner,radix,group,bits,limit); \
}
PACK_ENTRY(B, pack_u8, false)
PACK_ENTRY(unsigned, pack_u32, false)
PACK_ENTRY(B, unpack_u8, true)
PACK_ENTRY(unsigned, unpack_u32, true)

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

__device__ int affine_nibble(const B* codes, U row, U col, U width) {
    return (codes[row * ((width + 1) / 2) + col / 2] >> ((col % 2) * 4)) & 15;
}

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
        float sum = 0.0f;
        for (U j = 0; j < k; ++j) {
            const U p = j / block_size;
            const int z = zeros ? affine_nibble(zeros, row, p, blocks) : zero;
            const float decoded = __fmul_rn(affine_nibble(w, row, j, k) - z,
                                             read<float>(scales, row * blocks + p));
            sum += read<float>(x, token * k + j) * decoded;
        }
        write<float>(y, i, sum);
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
