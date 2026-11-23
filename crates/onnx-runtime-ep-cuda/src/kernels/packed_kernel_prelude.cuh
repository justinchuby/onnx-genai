using U = unsigned long long;
using B = unsigned char;
#define GRID(i, n) for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (U)gridDim.x * blockDim.x)

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
