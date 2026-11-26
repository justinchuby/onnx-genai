//! CUDA indexed element movement: `GatherElements`, `ScatterElements`, and
//! `ScatterND`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::scatter::{ScatterReduction as Reduction, validate_nd_index_type};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use super::elementwise::u64_bytes;
use super::movement::PersistentMetadata;
use crate::device_metadata::DeviceMetadataCache;
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
pub const SCATTER_CAPTURE_ERROR_INDEX: u32 = 256;
pub const GATHER_ELEMENTS_CAPTURE_ERROR_INDEX: u32 = 1_024;
pub const SCATTER_ND_CAPTURE_ERROR_INDEX: u32 = 8_192;
const SOURCE: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

extern "C" __global__ void gather_elements(
    const unsigned char* data, const void* indices, unsigned char* output,
    const unsigned long long* meta, int rank, int axis, int elem_bytes,
    unsigned long long elements, int index_is_i64, unsigned int* capture_error) {
  const unsigned long long* index_dims = meta;
  const unsigned long long* index_strides = meta + rank;
  const unsigned long long* data_strides = meta + 2 * rank;
  for (unsigned long long linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < elements; linear += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = linear, data_offset = 0;
    for (int d = 0; d < rank; ++d) {
      unsigned long long coordinate = rem / index_strides[d];
      rem %= index_strides[d];
      if (d == axis) {
        long long selected = index_is_i64
            ? ((const long long*)indices)[linear]
            : (long long)((const int*)indices)[linear];
        unsigned long long dim = meta[3 * rank + d];
        if ((selected >= 0 && (unsigned long long)selected >= dim) ||
            (selected < 0 && (0ull - (unsigned long long)selected) > dim)) {
          if (capture_error) atomicOr(capture_error, 1024u);
          continue;
        }
        coordinate = selected < 0 ? dim - (0ull - (unsigned long long)selected)
                                  : (unsigned long long)selected;
      }
      data_offset += coordinate * data_strides[d];
    }
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[linear * elem_bytes + byte] = data[data_offset * elem_bytes + byte];
  }
}

template <typename T> __device__ float scatter_load(T value);
template <> __device__ float scatter_load<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ float scatter_load<__half>(__half value) { return __half2float(value); }
template <> __device__ float scatter_load<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
#endif

template <typename T> __device__ T scatter_store(float value);
template <> __device__ float scatter_store<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ __half scatter_store<__half>(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 scatter_store<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}
#endif

template <typename Index>
__device__ bool scatter_offset(
    const Index* indices, unsigned long long linear,
    const unsigned long long* meta, int rank, int axis,
    unsigned long long* data_offset) {
  const unsigned long long* index_strides = meta;
  const unsigned long long* data_strides = meta + rank;
  const unsigned long long* data_dims = meta + 2 * rank;
  unsigned long long rem = linear;
  *data_offset = 0;
  for (int d = 0; d < rank; ++d) {
    unsigned long long coordinate = rem / index_strides[d];
    rem %= index_strides[d];
    if (d == axis) {
      long long raw = (long long)indices[linear];
      if (raw >= 0) {
        coordinate = (unsigned long long)raw;
        if (coordinate >= data_dims[d]) return false;
      } else {
        unsigned long long magnitude = 0ull - (unsigned long long)raw;
        if (magnitude > data_dims[d]) return false;
        coordinate = data_dims[d] - magnitude;
      }
    }
    *data_offset += coordinate * data_strides[d];
  }
  return true;
}

template <typename Data, typename Index>
__device__ void scatter_float_impl(
    Data* output, const Index* indices, const Data* updates,
    const unsigned long long* meta, int rank, int axis,
    unsigned long long elements, int reduction, unsigned int* capture_error) {
  if (blockIdx.x || threadIdx.x) return;
  for (unsigned long long linear = 0; linear < elements; ++linear) {
    unsigned long long data_offset;
    if (!scatter_offset(indices, linear, meta, rank, axis, &data_offset)) {
      if (capture_error) atomicOr(capture_error, 256u);
      continue;
    }
    if (reduction == 0) {
      output[data_offset] = updates[linear];
      continue;
    }
    float current = scatter_load<Data>(output[data_offset]);
    float update = scatter_load<Data>(updates[linear]);
    if (reduction == 1) output[data_offset] = scatter_store<Data>(current + update);
    else if (reduction == 2) output[data_offset] = scatter_store<Data>(current * update);
    else if (reduction == 3) {
      float value = (isnan(current) || isnan(update)) ? __int_as_float(0x7fc00000)
                                                       : fmaxf(current, update);
      output[data_offset] = scatter_store<Data>(value);
    } else {
      float value = (isnan(current) || isnan(update)) ? __int_as_float(0x7fc00000)
                                                       : fminf(current, update);
      output[data_offset] = scatter_store<Data>(value);
    }
  }
}

template <typename Index>
__device__ void scatter_i64_impl(
    long long* output, const Index* indices, const long long* updates,
    const unsigned long long* meta, int rank, int axis,
    unsigned long long elements, int reduction, unsigned int* capture_error) {
  if (blockIdx.x || threadIdx.x) return;
  for (unsigned long long linear = 0; linear < elements; ++linear) {
    unsigned long long data_offset;
    if (!scatter_offset(indices, linear, meta, rank, axis, &data_offset)) {
      if (capture_error) atomicOr(capture_error, 256u);
      continue;
    }
    long long update = updates[linear];
    if (reduction == 0) output[data_offset] = update;
    else if (reduction == 1)
      output[data_offset] = (long long)((unsigned long long)output[data_offset] +
                                        (unsigned long long)update);
    else if (reduction == 2)
      output[data_offset] = (long long)((unsigned long long)output[data_offset] *
                                        (unsigned long long)update);
    else if (reduction == 3) output[data_offset] = output[data_offset] > update ? output[data_offset] : update;
    else output[data_offset] = output[data_offset] < update ? output[data_offset] : update;
  }
}

#define DEFINE_SCATTER_FLOAT(DATA, DATA_SUFFIX, INDEX, INDEX_SUFFIX) \
extern "C" __global__ void scatter_##DATA_SUFFIX##_##INDEX_SUFFIX( \
    DATA* output, const INDEX* indices, const DATA* updates, \
    const unsigned long long* meta, int rank, int axis, \
    unsigned long long elements, int reduction, unsigned int* capture_error) { \
  scatter_float_impl(output, indices, updates, meta, rank, axis, elements, reduction, \
                     capture_error); \
}

#define DEFINE_SCATTER_I64(INDEX, INDEX_SUFFIX) \
extern "C" __global__ void scatter_i64_##INDEX_SUFFIX( \
    long long* output, const INDEX* indices, const long long* updates, \
    const unsigned long long* meta, int rank, int axis, \
    unsigned long long elements, int reduction, unsigned int* capture_error) { \
  scatter_i64_impl(output, indices, updates, meta, rank, axis, elements, reduction, \
                   capture_error); \
}

DEFINE_SCATTER_FLOAT(float, f32, int, i32)
DEFINE_SCATTER_FLOAT(float, f32, long long, i64)
DEFINE_SCATTER_I64(int, i32)
DEFINE_SCATTER_I64(long long, i64)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_SCATTER_FLOAT(__half, f16, int, i32)
DEFINE_SCATTER_FLOAT(__half, f16, long long, i64)
DEFINE_SCATTER_FLOAT(__nv_bfloat16, bf16, int, i32)
DEFINE_SCATTER_FLOAT(__nv_bfloat16, bf16, long long, i64)
#endif

template <typename Index>
__device__ bool scatter_nd_offset(
    const Index* indices, unsigned long long tuple,
    const unsigned long long* meta, int data_rank, int index_depth,
    unsigned long long* data_offset) {
  const unsigned long long* data_strides = meta;
  const unsigned long long* data_dims = meta + data_rank;
  *data_offset = 0;
  for (int dimension = 0; dimension < index_depth; ++dimension) {
    const long long raw = indices[tuple * index_depth + dimension];
    unsigned long long coordinate;
    if (raw >= 0) {
      coordinate = (unsigned long long)raw;
      if (coordinate >= data_dims[dimension]) return false;
    } else {
      const unsigned long long magnitude = 0ull - (unsigned long long)raw;
      if (magnitude > data_dims[dimension]) return false;
      coordinate = data_dims[dimension] - magnitude;
    }
    *data_offset += coordinate * data_strides[dimension];
  }
  return true;
}

template <typename Data, typename Index>
__device__ void scatter_nd_float_impl(
    Data* output, const Index* indices, const Data* updates,
    const unsigned long long* meta, int data_rank, int index_depth,
    unsigned long long tuples, unsigned long long slice_length, int reduction,
    unsigned int* capture_error) {
  if (blockIdx.x || threadIdx.x) return;
  for (unsigned long long tuple = 0; tuple < tuples; ++tuple) {
    unsigned long long destination;
    if (!scatter_nd_offset(
            indices, tuple, meta, data_rank, index_depth, &destination)) {
      if (capture_error) atomicOr(capture_error, 8192u);
      continue;
    }
    for (unsigned long long element = 0; element < slice_length; ++element) {
      const unsigned long long output_index = destination + element;
      const unsigned long long update_index = tuple * slice_length + element;
      if (reduction == 0) {
        output[output_index] = updates[update_index];
        continue;
      }
      const float current = scatter_load<Data>(output[output_index]);
      const float update = scatter_load<Data>(updates[update_index]);
      if (reduction == 1)
        output[output_index] = scatter_store<Data>(current + update);
      else if (reduction == 2)
        output[output_index] = scatter_store<Data>(current * update);
      else if (reduction == 3) {
        const float value = (isnan(current) || isnan(update))
            ? __int_as_float(0x7fc00000) : fmaxf(current, update);
        output[output_index] = scatter_store<Data>(value);
      } else {
        const float value = (isnan(current) || isnan(update))
            ? __int_as_float(0x7fc00000) : fminf(current, update);
        output[output_index] = scatter_store<Data>(value);
      }
    }
  }
}

template <typename Index>
__device__ void scatter_nd_i64_impl(
    long long* output, const Index* indices, const long long* updates,
    const unsigned long long* meta, int data_rank, int index_depth,
    unsigned long long tuples, unsigned long long slice_length, int reduction,
    unsigned int* capture_error) {
  if (blockIdx.x || threadIdx.x) return;
  for (unsigned long long tuple = 0; tuple < tuples; ++tuple) {
    unsigned long long destination;
    if (!scatter_nd_offset(
            indices, tuple, meta, data_rank, index_depth, &destination)) {
      if (capture_error) atomicOr(capture_error, 8192u);
      continue;
    }
    for (unsigned long long element = 0; element < slice_length; ++element) {
      const unsigned long long output_index = destination + element;
      const long long update = updates[tuple * slice_length + element];
      if (reduction == 0) output[output_index] = update;
      else if (reduction == 1)
        output[output_index] =
            (long long)((unsigned long long)output[output_index] +
                        (unsigned long long)update);
      else if (reduction == 2)
        output[output_index] =
            (long long)((unsigned long long)output[output_index] *
                        (unsigned long long)update);
      else if (reduction == 3)
        output[output_index] =
            output[output_index] > update ? output[output_index] : update;
      else
        output[output_index] =
            output[output_index] < update ? output[output_index] : update;
    }
  }
}

#define DEFINE_SCATTER_ND(DATA, SUFFIX, INDEX, IDX, IMPL) \
extern "C" __global__ void scatter_nd_##SUFFIX##_##IDX( \
    DATA* output, const INDEX* indices, const DATA* updates, \
    const unsigned long long* meta, int data_rank, int index_depth, \
    unsigned long long tuples, unsigned long long slice_length, int reduction, \
    unsigned int* capture_error) { \
  IMPL(output, indices, updates, meta, data_rank, index_depth, \
       tuples, slice_length, reduction, capture_error); \
}

#define DEFINE_SCATTER_ND_TYPES(DATA, SUFFIX, IMPL) \
DEFINE_SCATTER_ND(DATA, SUFFIX, int, i32, IMPL) \
DEFINE_SCATTER_ND(DATA, SUFFIX, long long, i64, IMPL)
DEFINE_SCATTER_ND_TYPES(float, f32, scatter_nd_float_impl)
DEFINE_SCATTER_ND_TYPES(long long, i64, scatter_nd_i64_impl)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_SCATTER_ND_TYPES(__half, f16, scatter_nd_float_impl)
DEFINE_SCATTER_ND_TYPES(__nv_bfloat16, bf16, scatter_nd_float_impl)
#endif

// Keep the update sum separate from the original so untouched values can be
// copied bit-for-bit. The integer sum is exact even across 64-bit overflow:
// each atomic add of the low limb contributes its own carry to the high limb.
using U64 = unsigned long long;
struct FloatMean { double sum; U64 count; };
struct IntMean { U64 low, high, count; };
template <typename Data> __device__ Data scatter_mean_store(double value);
template <> __device__ float scatter_mean_store<float>(double value) { return (float)value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ __half scatter_mean_store<__half>(double value) {
  return __double2half(value);
}
template <> __device__ __nv_bfloat16 scatter_mean_store<__nv_bfloat16>(double value) {
  return __double2bfloat16(value);
}
#endif
template <typename Data> struct Mean {
  using State = FloatMean;
  static __device__ void add(State* state, Data value) {
    atomicAdd(&state->sum, (double)scatter_load(value));
    atomicAdd(&state->count, 1ull);
  }
  static __device__ Data finish(State state, Data original) {
    return state.count == 0 ? original : scatter_mean_store<Data>(
        (state.sum + (double)scatter_load(original)) / (state.count + 1));
  }
};

template <> struct Mean<long long> {
  using State = IntMean;
  static __device__ void add(State* state, long long value) {
    U64 low = (U64)value;
    U64 old = atomicAdd(&state->low, low);
    int carry = old + low < old;
    int extension = carry - (value < 0);
    if (extension) atomicAdd(&state->high, (U64)extension);
    atomicAdd(&state->count, 1ull);
  }
  static __device__ long long finish(State state, long long original) {
    if (!state.count) return original;
    U64 low = state.low + (U64)original;
    U64 high = state.high + (low < state.low) - (original < 0);
    U64 count = state.count + 1;
    bool negative = (long long)high < 0;
    if (negative) {
      low = 0ull - low;
      high = ~high + (low == 0);
    }
    U64 quotient;
    if (!high) {
      quotient = low / count;
    } else {
      // The quotient fits in 64 bits, so high < count. Divide the low bits
      // into the initial remainder, then restore sign for truncation to zero.
      quotient = 0;
      for (int bit = 63; bit >= 0; --bit) {
        bool carry = high >> 63;
        high = (high << 1) | ((low >> bit) & 1);
        if (carry || high >= count) {
          high -= count;
          quotient |= 1ull << bit;
        }
      }
    }
    return (long long)(negative ? 0ull - quotient : quotient);
  }
};

template <typename Data, typename Index, bool Nd>
__device__ void scatter_mean_impl(
    typename Mean<Data>::State* states, const Index* indices, const Data* updates,
    const U64* meta, int rank, int axis_or_depth, U64 elements, U64 slice_length,
    unsigned int* capture_error) {
  const U64 stride = (U64)blockDim.x * gridDim.x;
  const U64 thread = (U64)blockDim.x * blockIdx.x + threadIdx.x;
  if (Nd && slice_length >= warpSize) {
    // One lookup per tuple rather than per element, with coalesced slice I/O.
    const unsigned int lane = threadIdx.x % warpSize;
    const U64 tuples = elements / slice_length;
    for (U64 tuple = thread / warpSize; tuple < tuples; tuple += stride / warpSize) {
      U64 at = 0;
      int valid = 0;
      if (lane == 0) {
        valid = scatter_nd_offset(indices, tuple, meta, rank, axis_or_depth, &at);
        if (!valid && capture_error) atomicOr(capture_error, 8192u);
      }
      if (!__shfl_sync(0xffffffffu, valid, 0)) continue;
      at = __shfl_sync(0xffffffffu, at, 0);
      for (U64 element = lane; element < slice_length; element += warpSize)
        Mean<Data>::add(states + at + element, updates[tuple * slice_length + element]);
    }
    return;
  }
  for (U64 i = thread; i < elements; i += stride) {
    U64 at;
    if (Nd) {
      const U64 tuple = i / slice_length;
      if (!scatter_nd_offset(indices, tuple, meta, rank, axis_or_depth, &at)) {
        if (capture_error) atomicOr(capture_error, 8192u);
        continue;
      }
      at += i - tuple * slice_length;
    } else if (!scatter_offset(indices, i, meta, rank, axis_or_depth, &at)) {
      if (capture_error) atomicOr(capture_error, 256u);
      continue;
    }
    Mean<Data>::add(states + at, updates[i]);
  }
}

#define DEFINE_MEAN_SCATTER(DATA, SUFFIX, INDEX, IDX, PREFIX, ND) \
extern "C" __global__ void PREFIX##_mean_##SUFFIX##_##IDX( \
    Mean<DATA>::State* states, const INDEX* indices, const DATA* updates, \
    const U64* meta, int rank, int axis_or_depth, U64 elements, U64 slice_length, \
    unsigned int* capture_error) { \
  scatter_mean_impl<DATA, INDEX, ND>(states, indices, updates, meta, rank, axis_or_depth, \
                                    elements, slice_length, capture_error); \
}
#define DEFINE_MEAN(DATA, SUFFIX) \
DEFINE_MEAN_SCATTER(DATA, SUFFIX, int, i32, scatter, false) \
DEFINE_MEAN_SCATTER(DATA, SUFFIX, long long, i64, scatter, false) \
DEFINE_MEAN_SCATTER(DATA, SUFFIX, int, i32, scatter_nd, true) \
DEFINE_MEAN_SCATTER(DATA, SUFFIX, long long, i64, scatter_nd, true) \
extern "C" __global__ void scatter_mean_finish_##SUFFIX( \
    DATA* output, const DATA* data, const Mean<DATA>::State* states, U64 elements) { \
  const U64 stride = (U64)blockDim.x * gridDim.x; \
  for (U64 i = (U64)blockDim.x * blockIdx.x + threadIdx.x; i < elements; i += stride) \
    output[i] = Mean<DATA>::finish(states[i], data[i]); \
}
DEFINE_MEAN(float, f32)
DEFINE_MEAN(long long, i64)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_MEAN(__half, f16)
DEFINE_MEAN(__nv_bfloat16, bf16)
#endif
"#;

fn axis(op: &str, raw: i64, rank: usize) -> Result<usize> {
    let normalized = if raw < 0 { raw + rank as i64 } else { raw };
    if normalized < 0 || normalized as usize >= rank {
        Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: axis out of range"
        )))
    } else {
        Ok(normalized as usize)
    }
}

fn require_dense(op: &str, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if inputs.iter().any(|v| !v.is_contiguous()) || outputs.iter().any(|v| !v.is_contiguous()) {
        Err(not_implemented(format!("{op} with non-contiguous tensors")))
    } else {
        Ok(())
    }
}

fn validate_indices(
    runtime: &CudaRuntime,
    indices: &TensorView,
    dim: usize,
    op: &str,
) -> Result<()> {
    if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: indices must be Int32 or Int64"
        )));
    }
    // Eager-fast path: skip the blocking index-bounds D2H (numerically inert —
    // a correct model never trips it; the captured path relies on the device
    // error latch instead of a per-op host readback).
    if runtime.eager_sync_deferred() {
        return Ok(());
    }
    let mut bytes = vec![0_u8; indices.dtype.storage_bytes(indices.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(indices.data_ptr::<u8>() as *const c_void))? };
    }
    for raw in bytes.chunks_exact(indices.dtype.byte_size()) {
        let value = match indices.dtype {
            DataType::Int32 => i32::from_ne_bytes(raw.try_into().unwrap()) as i64,
            DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
            _ => unreachable!("validated above"),
        };
        let in_range = if value >= 0 {
            (value as u64) < dim as u64
        } else {
            value.unsigned_abs() <= dim as u64
        };
        if !in_range {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: index {value} out of range"
            )));
        }
    }
    Ok(())
}

fn validate_nd_indices(
    runtime: &CudaRuntime,
    indices: &TensorView,
    data_shape: &[usize],
    index_depth: usize,
) -> Result<()> {
    let mut bytes = vec![0_u8; indices.dtype.storage_bytes(indices.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(indices.data_ptr::<u8>() as *const c_void))? };
    }
    for (linear, raw) in bytes.chunks_exact(indices.dtype.byte_size()).enumerate() {
        let value = match indices.dtype {
            DataType::Int32 => i64::from(i32::from_ne_bytes(raw.try_into().unwrap())),
            DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
            _ => unreachable!("validated above"),
        };
        let dimension = linear % index_depth;
        let size = data_shape[dimension];
        let in_range = if value >= 0 {
            (value as u64) < size as u64
        } else {
            value.unsigned_abs() <= size as u64
        };
        if !in_range {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ScatterND: index {value} out of range at tuple dimension {dimension}"
            )));
        }
    }
    Ok(())
}

fn scatter_elements_metadata_bytes(
    data_shape: &[usize],
    indices_shape: &[usize],
) -> Result<Vec<u8>> {
    let mut values = compute_contiguous_strides(indices_shape)?
        .into_iter()
        .map(|value| value as u64)
        .collect::<Vec<_>>();
    values.extend(
        compute_contiguous_strides(data_shape)?
            .into_iter()
            .map(|value| value as u64),
    );
    values.extend(data_shape.iter().map(|&value| value as u64));
    Ok(u64_bytes(&values).to_vec())
}

fn gather_elements_metadata_bytes(
    data_shape: &[usize],
    indices_shape: &[usize],
) -> Result<Vec<u8>> {
    let mut values = indices_shape
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    values.extend(
        compute_contiguous_strides(indices_shape)?
            .into_iter()
            .map(|value| value as u64),
    );
    values.extend(
        compute_contiguous_strides(data_shape)?
            .into_iter()
            .map(|value| value as u64),
    );
    values.extend(data_shape.iter().map(|&value| value as u64));
    Ok(u64_bytes(&values).to_vec())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterMetadataKey {
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ScatterMetadataCache {
    storage: DeviceMetadataCache<ScatterMetadataKey>,
}

impl ScatterMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            storage: DeviceMetadataCache::new(runtime),
        }
    }

    fn prepare(&mut self, data_shape: &[usize], indices_shape: &[usize]) -> Result<CUdeviceptr> {
        self.storage.get_or_prepare(
            "ScatterElements metadata",
            |key| key.data_shape == data_shape && key.indices_shape == indices_shape,
            || {
                Ok((
                    ScatterMetadataKey {
                        data_shape: data_shape.to_vec(),
                        indices_shape: indices_shape.to_vec(),
                    },
                    scatter_elements_metadata_bytes(data_shape, indices_shape)?,
                ))
            },
        )
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.storage.device_graph_resource()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterCaptureSignature {
    data_dtype: DataType,
    indices_dtype: DataType,
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterNdCaptureSignature {
    data_dtype: DataType,
    indices_dtype: DataType,
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
    updates_shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GatherElementsMetadataKey {
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct GatherElementsMetadataCache {
    storage: DeviceMetadataCache<GatherElementsMetadataKey>,
}

impl GatherElementsMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            storage: DeviceMetadataCache::new(runtime),
        }
    }

    fn prepare(&mut self, data_shape: &[usize], indices_shape: &[usize]) -> Result<CUdeviceptr> {
        self.storage.get_or_prepare(
            "GatherElements metadata",
            |key| key.data_shape == data_shape && key.indices_shape == indices_shape,
            || {
                Ok((
                    GatherElementsMetadataKey {
                        data_shape: data_shape.to_vec(),
                        indices_shape: indices_shape.to_vec(),
                    },
                    gather_elements_metadata_bytes(data_shape, indices_shape)?,
                ))
            },
        )
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.storage.device_graph_resource()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GatherElementsCaptureSignature {
    data_dtype: DataType,
    indices_dtype: DataType,
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

pub struct GatherElementsFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GatherElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherElementsKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(0),
            metadata: Mutex::new(GatherElementsMetadataCache::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct GatherElementsKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    metadata: Mutex<GatherElementsMetadataCache>,
    warmed_signature: Mutex<Option<GatherElementsCaptureSignature>>,
}

impl Kernel for GatherElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: expected 2 inputs and 1 output".into(),
            ));
        }
        require_dense("GatherElements", inputs, outputs)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        let output = &mut outputs[0];
        if data.shape.len() != indices.shape.len() {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: data and indices must have equal rank".into(),
            ));
        }
        let rank = data.shape.len();
        let axis = axis("GatherElements", self.axis, rank)?;
        if output.dtype != data.dtype || output.shape != indices.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: output must match indices shape and data dtype".into(),
            ));
        }
        for d in 0..rank {
            if d != axis && indices.shape[d] > data.shape[d] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GatherElements: indices dimension exceeds data at axis {d}"
                )));
            }
        }
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: indices must be Int32 or Int64".into(),
            ));
        }
        let capturing = self.runtime.is_capturing()?;
        let signature = GatherElementsCaptureSignature {
            data_dtype: data.dtype,
            indices_dtype: indices.dtype,
            data_shape: data.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
        };
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep GatherElements: capture signature lock was poisoned".into(),
            )
        })?;
        if capturing && warmed_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: shape or dtype changed during CUDA graph capture; warm the exact signature first".into(),
            ));
        }
        if !capturing {
            validate_indices(&self.runtime, indices, data.shape[axis], "GatherElements")?;
        }
        if output.numel() == 0 {
            if !capturing {
                *warmed_signature = Some(signature);
            }
            return Ok(());
        }
        let elem_bytes = i32::try_from(data.dtype.byte_size()).map_err(|_| {
            EpError::KernelFailed("cuda_ep GatherElements: element width exceeds i32".into())
        })?;
        if elem_bytes == 0 {
            return Err(not_implemented("GatherElements for variable-width dtype"));
        }
        let func = self
            .runtime
            .nvrtc_function("indexing_ops", SOURCE, "gather_elements")?;
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep GatherElements: metadata lock was poisoned".into())
        })?;
        let mut metadata_candidate = metadata.clone();
        let meta_ptr = metadata_candidate.prepare(data.shape, indices.shape)?;
        let data_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rank = i32::try_from(rank).map_err(|_| {
            EpError::KernelFailed("cuda_ep GatherElements: rank exceeds i32".into())
        })?;
        let axis = axis as i32;
        let elements = output.numel() as u64;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&data_ptr)
            .arg(&indices_ptr)
            .arg(&output_ptr)
            .arg(&meta_ptr)
            .arg(&rank)
            .arg(&axis)
            .arg(&elem_bytes)
            .arg(&elements)
            .arg(&index_is_i64)
            .arg(&capture_error);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    (elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32),
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch gather_elements", e))?;
        if !capturing {
            self.runtime.synchronize()?;
            *metadata = metadata_candidate;
            *warmed_signature = Some(signature);
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.metadata
            .lock()
            .ok()
            .and_then(|metadata| metadata.device_graph_resource())
            .into_iter()
            .collect()
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "GatherElements must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "GatherElements capture signature lock was poisoned",
            ),
        }
    }
}

fn scatter_data_suffix(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float16 => "f16",
        DataType::Float32 => "f32",
        DataType::BFloat16 => "bf16",
        DataType::Int64 => "i64",
        _ => unreachable!("validated scatter data type"),
    }
}

fn scatter_entry(prefix: &str, data: DataType, indices: DataType) -> String {
    let index = if indices == DataType::Int32 {
        "i32"
    } else {
        "i64"
    };
    format!("{prefix}_{}_{index}", scatter_data_suffix(data))
}

fn copy_scatter_data(
    runtime: &CudaRuntime,
    data: &TensorView,
    output: &mut TensorMut,
) -> Result<()> {
    if data.numel() == 0 {
        return Ok(());
    }
    // SAFETY: both views are dense with matching shape and dtype.
    unsafe {
        runtime.dtod_async(
            cuptr(data.data_ptr::<u8>() as *const c_void),
            cuptr(output.data_ptr_mut::<u8>() as *const c_void),
            data.dtype.storage_bytes(data.numel()),
        )
    }
}

/// Mean state is disposable within one stream-ordered call, not a kernel cache.
/// Each floating output has a double sum/u64 count; Int64 adds a second sum limb.
fn mean_workspace_layout(
    reduction: Reduction,
    inputs: &[TensorMetadata<'_>],
) -> Result<WorkspaceLayout<1>> {
    let count = |shape: &[usize]| -> Result<usize> {
        if shape.contains(&0) {
            return Ok(0);
        }
        shape.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim)).ok_or_else(|| EpError::KernelFailed("cuda_ep Scatter mean: shape element count exceeds addressable capacity; reduce data/update dimensions".into()))
    };
    let region = if reduction != Reduction::Mean {
        std::alloc::Layout::new::<()>()
    } else {
        let [data, indices, updates] = inputs else {
            return Err(EpError::KernelFailed("cuda_ep Scatter mean: workspace planning needs data, indices and updates; provide all three input metadata slots".into()));
        };
        if !data.present || !indices.present || !updates.present {
            return Err(EpError::KernelFailed("cuda_ep Scatter mean: required data/indices/updates metadata is absent; bind all three inputs before preparing workspace".into()));
        }
        let elements = count(data.shape)?;
        let updates = count(updates.shape)?;
        let elements = if updates == 0 { 0 } else { elements };
        let layout = match data.dtype {
            DataType::Int64 => std::alloc::Layout::array::<[u64; 3]>(elements),
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 => {
                std::alloc::Layout::array::<[u64; 2]>(elements)
            }
            dtype => {
                return Err(not_implemented(format!(
                    "Scatter mean data dtype {dtype:?}; use Float16, BFloat16, Float32, or Int64"
                )));
            }
        };
        layout.map_err(|_| EpError::KernelFailed("cuda_ep Scatter mean: accumulator state exceeds addressable capacity; reduce the output shape".into()))?
    };
    WorkspaceLayout::new([region])
}

fn mean_workspace_requirement(
    reduction: Reduction,
    inputs: &[TensorMetadata<'_>],
) -> Result<WorkspaceRequirement> {
    let layout = mean_workspace_layout(reduction, inputs)?;
    Ok(if layout.bytes() == 0 {
        WorkspaceRequirement::NONE
    } else {
        WorkspaceRequirement {
            bytes: layout.bytes() as u64,
            alignment: layout.alignment(),
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_mean(
    runtime: &CudaRuntime,
    used: WorkspaceView,
    prefix: &str,
    data: &TensorView,
    index_dtype: DataType,
    output: CUdeviceptr,
    indices: CUdeviceptr,
    updates: CUdeviceptr,
    meta: CUdeviceptr,
    rank: i32,
    axis_or_depth: i32,
    elements: u64,
    slice_length: u64,
    capture_error: CUdeviceptr,
) -> Result<()> {
    let output_elements = data.numel() as u64;
    if output_elements == 0 {
        return Ok(());
    }
    let bytes = used.bytes();
    let scratch = cuptr(used.ptr().0.cast_const());
    let scatter_entry = scatter_entry(&format!("{prefix}_mean"), data.dtype, index_dtype);
    let finish_entry = format!("scatter_mean_finish_{}", scatter_data_suffix(data.dtype));
    let scatter = runtime.nvrtc_function("indexing_ops_v4", SOURCE, &scatter_entry)?;
    let finish = runtime.nvrtc_function("indexing_ops_v4", SOURCE, &finish_entry)?;
    let config = |n: u64| LaunchConfig {
        grid_dim: (n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    runtime.bind()?;
    // SAFETY: the workspace covers every output and remains owned through
    // completion, including graph replay. All three commands share a stream.
    unsafe { cudarc::driver::result::memset_d8_async(scratch, 0, bytes, runtime.stream_ptr()) }
        .map_err(|err| driver_err("zero Scatter mean workspace", err))?;
    let mut builder = runtime.stream().launch_builder(&scatter);
    builder
        .arg(&scratch)
        .arg(&indices)
        .arg(&updates)
        .arg(&meta)
        .arg(&rank)
        .arg(&axis_or_depth)
        .arg(&elements)
        .arg(&slice_length)
        .arg(&capture_error);
    unsafe { builder.launch(config(elements)) }
        .map_err(|err| driver_err(&format!("launch {scatter_entry}"), err))?;
    let original = cuptr(data.data_ptr::<u8>() as *const c_void);
    let mut builder = runtime.stream().launch_builder(&finish);
    builder
        .arg(&output)
        .arg(&original)
        .arg(&scratch)
        .arg(&output_elements);
    unsafe { builder.launch(config(output_elements)) }
        .map_err(|err| driver_err(&format!("launch {finish_entry}"), err))?;
    Ok(())
}

pub struct ScatterElementsFactory {
    pub runtime: Arc<CudaRuntime>,
    pub opset: u64,
}

pub struct ScatterNdFactory {
    pub runtime: Arc<CudaRuntime>,
    pub opset: u64,
}

impl KernelFactory for ScatterNdFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ScatterNdKernel {
            runtime: self.runtime.clone(),
            reduction: Reduction::parse(node, self.opset).map_err(EpError::KernelFailed)?,
            opset: self.opset,
            metadata: Mutex::new(PersistentMetadata::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct ScatterNdKernel {
    runtime: Arc<CudaRuntime>,
    reduction: Reduction,
    opset: u64,
    metadata: Mutex<PersistentMetadata>,
    warmed_signature: Mutex<Option<ScatterNdCaptureSignature>>,
}

impl ScatterNdKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep ScatterND: capture signature lock was poisoned; recreate the kernel"
                    .into(),
            )
        })?;
        let previous_signature = warmed_signature.take();
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterND: expected 3 inputs and 1 output".into(),
            ));
        }
        require_dense("ScatterND", inputs, outputs)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        let updates = &inputs[2];
        let output = &mut outputs[0];
        validate_nd_index_type("ScatterND", self.opset, indices.dtype)
            .map_err(EpError::KernelFailed)?;
        if !matches!(
            data.dtype,
            DataType::Float16 | DataType::Float32 | DataType::BFloat16 | DataType::Int64
        ) {
            return Err(not_implemented(format!(
                "ScatterND supports Float16, Float32, BFloat16, and Int64 data, got {:?}",
                data.dtype
            )));
        }
        if updates.dtype != data.dtype || output.dtype != data.dtype || output.shape != data.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterND: data, updates, and output must match".into(),
            ));
        }
        if data.shape.is_empty() || indices.shape.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterND: data and indices must have rank at least 1".into(),
            ));
        }
        let index_depth = *indices.shape.last().expect("rank checked above");
        if index_depth > data.shape.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ScatterND: index tuple length {index_depth} exceeds data rank {}",
                data.shape.len()
            )));
        }
        let mut expected_updates_shape = indices.shape[..indices.shape.len() - 1].to_vec();
        expected_updates_shape.extend_from_slice(&data.shape[index_depth..]);
        if updates.shape != expected_updates_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ScatterND: updates shape {:?} must be {expected_updates_shape:?}",
                updates.shape
            )));
        }
        if matches!(data.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers("ScatterND")?;
        }

        let metadata = std::array::from_fn::<_, 3, _>(|i| {
            TensorMetadata::new(inputs[i].dtype, inputs[i].shape, !inputs[i].is_absent())
        });
        let [mean_scratch] = mean_workspace_layout(self.reduction, &metadata)?.bind(workspace).map_err(|cause| EpError::KernelFailed(format!("cuda_ep ScatterND mean: {cause}; prepare and pass the complete accumulator workspace")))?;
        let capturing = self.runtime.is_capturing()?;
        if capturing && mean_scratch.bytes() != 0 {
            self.runtime.require_registered_workspace_capture(
                mean_scratch,
                "ScatterND mean accumulators",
            )?;
        }
        let signature = ScatterNdCaptureSignature {
            data_dtype: data.dtype,
            indices_dtype: indices.dtype,
            data_shape: data.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
            updates_shape: updates.shape.to_vec(),
        };
        if capturing && previous_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterND: shape or dtype changed during CUDA graph capture; warm the exact signature first".into(),
            ));
        }
        let mean = self.reduction == Reduction::Mean;
        if !capturing
            && index_depth != 0
            && (!mean || !self.runtime.eager_sync_deferred() || updates.numel() == 0)
        {
            validate_nd_indices(&self.runtime, indices, data.shape, index_depth)?;
        }
        let tuples = indices.shape[..indices.shape.len() - 1]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep ScatterND: tuple count overflow".into())
            })?;
        let slice_length = data.shape[index_depth..]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep ScatterND: slice length overflow".into())
            })?;
        if tuples != 0
            && let Some(d) = data.shape[..index_depth].iter().position(|&n| n == 0)
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ScatterND: nonempty index tuples cannot address zero-length data dimension {d}"
            )));
        }
        if !mean || tuples == 0 {
            copy_scatter_data(&self.runtime, data, output)?;
        }
        if tuples == 0 || (slice_length == 0 && index_depth == 0) {
            *warmed_signature = Some(signature);
            return Ok(());
        }

        let mut metadata_values = compute_contiguous_strides(data.shape)?
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>();
        metadata_values.extend(data.shape.iter().map(|&value| value as u64));
        let mut metadata_cache = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterND: metadata lock was poisoned".into())
        })?;
        let metadata_candidate = metadata_cache.stage(&metadata_values, "ScatterND metadata")?;
        let metadata_ptr = metadata_candidate.ptr("ScatterND")?;
        // Empty slices still require tuple validation, but never access values.
        let output_ptr = if slice_length == 0 {
            0
        } else {
            cuptr(output.data_ptr_mut::<u8>() as *const c_void)
        };
        let indices_ptr = if index_depth == 0 {
            0
        } else {
            cuptr(indices.data_ptr::<u8>() as *const c_void)
        };
        let updates_ptr = if slice_length == 0 {
            0
        } else {
            cuptr(updates.data_ptr::<u8>() as *const c_void)
        };
        let data_rank = i32::try_from(data.shape.len())
            .map_err(|_| EpError::KernelFailed("cuda_ep ScatterND: rank exceeds i32".into()))?;
        let index_depth = i32::try_from(index_depth).map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterND: index depth exceeds i32".into())
        })?;
        let tuples = tuples as u64;
        let slice_length = slice_length as u64;
        let reduction = self.reduction as i32;
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        if mean && slice_length != 0 {
            launch_scatter_mean(
                &self.runtime,
                mean_scratch,
                "scatter_nd",
                data,
                indices.dtype,
                output_ptr,
                indices_ptr,
                updates_ptr,
                metadata_ptr,
                data_rank,
                index_depth,
                updates.numel() as u64,
                slice_length,
                capture_error,
            )?;
        } else {
            let entry = scatter_entry("scatter_nd", data.dtype, indices.dtype);
            let function = self
                .runtime
                .nvrtc_function("indexing_ops_v4", SOURCE, &entry)?;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&output_ptr)
                .arg(&indices_ptr)
                .arg(&updates_ptr)
                .arg(&metadata_ptr)
                .arg(&data_rank)
                .arg(&index_depth)
                .arg(&tuples)
                .arg(&slice_length)
                .arg(&reduction)
                .arg(&capture_error);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        }
        if !capturing {
            *metadata_cache = metadata_candidate;
        }
        *warmed_signature = Some(signature);
        Ok(())
    }
}

impl Kernel for ScatterNdKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        mean_workspace_requirement(self.reduction, inputs)
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.metadata
            .lock()
            .ok()
            .and_then(|metadata| metadata.device_graph_resource())
            .into_iter()
            .collect()
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ScatterND must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ScatterND capture signature lock was poisoned",
            ),
        }
    }
}

impl KernelFactory for ScatterElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ScatterElementsKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(0),
            reduction: Reduction::parse(node, self.opset).map_err(EpError::KernelFailed)?,
            metadata: Mutex::new(ScatterMetadataCache::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct ScatterElementsKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    reduction: Reduction,
    metadata: Mutex<ScatterMetadataCache>,
    warmed_signature: Mutex<Option<ScatterCaptureSignature>>,
}

impl ScatterElementsKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep ScatterElements: capture signature lock was poisoned; recreate the kernel"
                    .into(),
            )
        })?;
        let previous_signature = warmed_signature.take();
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: expected 3 inputs and 1 output".into(),
            ));
        }
        require_dense("ScatterElements", inputs, outputs)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        let updates = &inputs[2];
        let output = &mut outputs[0];
        if indices.shape != updates.shape || indices.shape.len() != data.shape.len() {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: indices and updates must have equal rank and shape"
                    .into(),
            ));
        }
        if updates.dtype != data.dtype || output.dtype != data.dtype || output.shape != data.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: data, updates, and output must match".into(),
            ));
        }
        if !matches!(
            data.dtype,
            DataType::Float16 | DataType::Float32 | DataType::BFloat16 | DataType::Int64
        ) {
            return Err(not_implemented(format!(
                "ScatterElements supports Float16, Float32, BFloat16, and Int64 data, got {:?}",
                data.dtype
            )));
        }
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "ScatterElements supports Int32 and Int64 indices, got {:?}",
                indices.dtype
            )));
        }
        if matches!(data.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers("ScatterElements")?;
        }
        let rank = data.shape.len();
        let axis = axis("ScatterElements", self.axis, rank)?;
        for d in 0..rank {
            if d != axis && indices.shape[d] > data.shape[d] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep ScatterElements: indices dimension exceeds data at axis {d}"
                )));
            }
        }
        if data.shape[axis] == 0 && indices.numel() != 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ScatterElements: indices must be empty when data axis {axis} has length zero"
            )));
        }
        let metadata = std::array::from_fn::<_, 3, _>(|i| {
            TensorMetadata::new(inputs[i].dtype, inputs[i].shape, !inputs[i].is_absent())
        });
        let [mean_scratch] = mean_workspace_layout(self.reduction, &metadata)?.bind(workspace).map_err(|cause| EpError::KernelFailed(format!("cuda_ep ScatterElements mean: {cause}; prepare and pass the complete accumulator workspace")))?;
        let capturing = self.runtime.is_capturing()?;
        if capturing && mean_scratch.bytes() != 0 {
            self.runtime.require_registered_workspace_capture(
                mean_scratch,
                "ScatterElements mean accumulators",
            )?;
        }
        let signature = ScatterCaptureSignature {
            data_dtype: data.dtype,
            indices_dtype: indices.dtype,
            data_shape: data.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
        };
        if capturing && previous_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: shape or dtype changed during CUDA graph capture; warm the exact signature first".into(),
            ));
        }
        if !capturing {
            validate_indices(&self.runtime, indices, data.shape[axis], "ScatterElements")?;
        }
        let mean = self.reduction == Reduction::Mean;
        if !mean || indices.numel() == 0 {
            copy_scatter_data(&self.runtime, data, output)?;
        }
        if indices.numel() == 0 {
            *warmed_signature = Some(signature);
            return Ok(());
        }
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterElements: metadata lock was poisoned".into())
        })?;
        let mut metadata_candidate = metadata.clone();
        let meta_ptr = metadata_candidate.prepare(data.shape, indices.shape)?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let updates_ptr = cuptr(updates.data_ptr::<u8>() as *const c_void);
        let rank = i32::try_from(rank).map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterElements: rank exceeds i32".into())
        })?;
        let axis = axis as i32;
        let elements = u64::try_from(indices.numel()).map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterElements: element count exceeds u64".into())
        })?;
        let reduction = self.reduction as i32;
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        if mean {
            launch_scatter_mean(
                &self.runtime,
                mean_scratch,
                "scatter",
                data,
                indices.dtype,
                output_ptr,
                indices_ptr,
                updates_ptr,
                meta_ptr,
                rank,
                axis,
                elements,
                1,
                capture_error,
            )?;
        } else {
            let entry = scatter_entry("scatter", data.dtype, indices.dtype);
            let func = self
                .runtime
                .nvrtc_function("indexing_ops_v4", SOURCE, &entry)?;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&output_ptr)
                .arg(&indices_ptr)
                .arg(&updates_ptr)
                .arg(&meta_ptr)
                .arg(&rank)
                .arg(&axis)
                .arg(&elements)
                .arg(&reduction)
                .arg(&capture_error);
            // SAFETY: the entry point is selected from the validated data/index
            // dtypes; metadata contains three rank-length u64 arrays.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        }
        if !capturing {
            *metadata = metadata_candidate;
        }
        *warmed_signature = Some(signature);
        Ok(())
    }
}

impl Kernel for ScatterElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        mean_workspace_requirement(self.reduction, inputs)
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.metadata
            .lock()
            .ok()
            .and_then(|metadata| metadata.device_graph_resource())
            .into_iter()
            .collect()
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ScatterElements must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ScatterElements capture signature lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;

    #[test]
    fn indexed_metadata_preserves_the_distinct_kernel_table_abis() {
        let decode = |bytes: Vec<u8>| {
            bytes
                .chunks_exact(8)
                .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let scatter = decode(
            scatter_elements_metadata_bytes(&[2, 3], &[1, 2])
                .expect("fixture shape has representable contiguous strides"),
        );
        assert_eq!(
            scatter,
            [2, 1, 3, 1, 2, 3],
            "ScatterElements consumes index strides, data strides, data dimensions"
        );
        let gather = decode(
            gather_elements_metadata_bytes(&[2, 3], &[1, 2])
                .expect("fixture shape has representable contiguous strides"),
        );
        assert_eq!(
            gather,
            [1, 2, 2, 1, 3, 1, 2, 3],
            "GatherElements also consumes leading index dimensions"
        );
    }

    fn metadata<'a>(
        dtype: DataType,
        data: &'a [usize],
        indices: &'a [usize],
        updates: &'a [usize],
    ) -> [TensorMetadata<'a>; 3] {
        [
            TensorMetadata::new(dtype, data, true),
            TensorMetadata::new(DataType::Int64, indices, true),
            TensorMetadata::new(dtype, updates, true),
        ]
    }

    #[test]
    fn mean_uses_one_peak_with_exact_float_or_integer_state() {
        for (dtype, size) in [
            (DataType::Float16, 16),
            (DataType::BFloat16, 16),
            (DataType::Float32, 16),
            (DataType::Int64, 24),
        ] {
            let inputs = metadata(dtype, &[3, 5], &[2, 1], &[2, 5]);
            let requirement = mean_workspace_requirement(Reduction::Mean, &inputs).unwrap();
            assert_eq!(requirement.bytes, 15 * size);
            assert_eq!(requirement.alignment, std::mem::align_of::<u64>());
            assert_eq!(requirement.lifetime, WorkspaceLifetime::SessionPersistent);
        }
    }

    #[test]
    fn nd_depth_zero_keeps_nonempty_mean_updates_and_true_empty_work_is_zero() {
        let depth_zero = metadata(DataType::Float32, &[4], &[2, 0], &[2, 4]);
        assert_eq!(
            mean_workspace_requirement(Reduction::Mean, &depth_zero)
                .unwrap()
                .bytes,
            64
        );
        for inputs in [
            metadata(DataType::Float32, &[4], &[0, 1], &[0]),
            metadata(DataType::Float32, &[4, 0], &[2, 1], &[2, 0]),
        ] {
            assert_eq!(
                mean_workspace_requirement(Reduction::Mean, &inputs).unwrap(),
                WorkspaceRequirement::NONE
            );
        }
        assert_eq!(
            mean_workspace_requirement(Reduction::Add, &depth_zero).unwrap(),
            WorkspaceRequirement::NONE
        );
    }

    #[test]
    fn mean_rejects_absent_inputs_and_unaddressable_shapes_before_reservation() {
        assert!(
            mean_workspace_requirement(Reduction::Mean, &[])
                .unwrap_err()
                .to_string()
                .contains("all three input")
        );
        let mut inputs = metadata(DataType::Float32, &[4], &[1], &[1]);
        inputs[2].present = false;
        assert!(
            mean_workspace_requirement(Reduction::Mean, &inputs)
                .unwrap_err()
                .to_string()
                .contains("absent")
        );
        for shape in [&[usize::MAX, 2][..], &[isize::MAX as usize / 8][..]] {
            let inputs = metadata(DataType::Int64, shape, &[1], &[1]);
            let error = mean_workspace_requirement(Reduction::Mean, &inputs)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("capacity") && error.contains("reduce"),
                "{error}"
            );
        }
    }

    #[test]
    fn mean_borrows_exact_used_span_and_rejects_missing_short_or_unaligned_views() {
        let inputs = metadata(DataType::Int64, &[4], &[1], &[1]);
        let layout = mean_workspace_layout(Reduction::Mean, &inputs).unwrap();
        let base = DevicePtrMut(0x1000usize as *mut _);
        for view in [
            None,
            Some(WorkspaceView::new(base, 95)),
            Some(WorkspaceView::new(DevicePtrMut(0x1001usize as *mut _), 96)),
        ] {
            assert!(
                layout
                    .bind(view)
                    .unwrap_err()
                    .to_string()
                    .contains("prepare and pass")
            );
        }
        let [used] = layout.bind(Some(WorkspaceView::new(base, 1024))).unwrap();
        assert_eq!(used.bytes(), 96);
        assert_eq!(used.ptr().0, base.0);
    }
}
