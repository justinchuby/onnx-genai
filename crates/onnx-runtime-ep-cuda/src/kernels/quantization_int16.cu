#include <cuda_fp16.h>
#include <cuda_bf16.h>

using Index = unsigned long long;

__device__ Index scale_index(Index i, Index inner, Index axis, Index block) {
  if (inner == 0) return 0;
  const Index lane = i / inner;
  const Index k = lane % axis;
  if (block == 0) return k;
  return ((lane / axis) * (1 + (axis - 1) / block) + k / block) * inner + i % inner;
}

__device__ double load_real(const void* ptr, Index i, int dtype) {
  if (dtype == 10) return __half2float(((const __half*)ptr)[i]);
  if (dtype == 16) return __bfloat162float(((const __nv_bfloat16*)ptr)[i]);
  if (dtype == 6) return ((const int*)ptr)[i];
  return ((const float*)ptr)[i];
}

__device__ void store_real(void* ptr, Index i, int dtype, float value) {
  if (dtype == 10) ((__half*)ptr)[i] = __float2half_rn(value);
  else if (dtype == 16) ((__nv_bfloat16*)ptr)[i] = __float2bfloat16_rn(value);
  else ((float*)ptr)[i] = value;
}

__device__ double narrow(double value, int precision) {
  if (precision == 10) return __half2float(__double2half(value));
  if (precision == 16) return __bfloat162float(__double2bfloat16(value));
  return value;
}

__device__ int load_code(const unsigned short* ptr, Index i, int is_signed) {
  const int bits = ptr[i];
  return is_signed ? (bits ^ 32768) - 32768 : bits;
}

extern "C" __global__ void quantize16(
    const void* x, const void* scale, const unsigned short* zp, unsigned short* y,
    Index n, Index inner, Index axis, Index block, int x_type, int scale_type,
    int precision, int is_signed) {
  for (Index i = (Index)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (Index)gridDim.x * blockDim.x) {
    const Index p = scale_index(i, inner, axis, block);
    const double v = load_real(x, i, x_type);
    const double s = load_real(scale, p, scale_type);
    const double quotient = precision == 1
        ? __fdiv_rn((float)v, (float)s)
        : narrow(narrow(v, precision) / narrow(s, precision), precision);
    const int z = zp ? load_code(zp, p, is_signed) : 0;
    const double q = (isnan(quotient) ? 0.0 : rint(quotient)) + z;
    const double bounded = fmin(fmax(q, is_signed ? -32768.0 : 0.0),
                                is_signed ? 32767.0 : 65535.0);
    y[i] = (unsigned short)(int)bounded;
  }
}

extern "C" __global__ void dequantize16(
    const unsigned short* x, const void* scale, const unsigned short* zp, void* y,
    Index n, Index inner, Index axis, Index block, int y_type, int scale_type,
    int /* unused */, int is_signed) {
  for (Index i = (Index)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (Index)gridDim.x * blockDim.x) {
    const Index p = scale_index(i, inner, axis, block);
    const int z = zp ? load_code(zp, p, is_signed) : 0;
    const float value = (float)(load_code(x, i, is_signed) - z) * (float)load_real(scale, p, scale_type);
    store_real(y, i, y_type, value);
  }
}
