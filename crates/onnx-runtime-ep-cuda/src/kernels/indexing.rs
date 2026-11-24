//! CUDA indexed element movement: `GatherElements`, `ScatterElements`, and
//! `ScatterND`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::scatter::{ScatterReduction as Reduction, validate_nd_index_type};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use super::movement::PersistentMetadata;
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

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
template <typename Data> struct Mean {
  using State = FloatMean;
  static __device__ void add(State* state, Data value) {
    atomicAdd(&state->sum, (double)scatter_load(value));
    atomicAdd(&state->count, 1ull);
  }
  static __device__ Data finish(State state, Data original) {
    return state.count == 0 ? original : scatter_store<Data>(
        (float)((state.sum + (double)scatter_load(original)) / (state.count + 1)));
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
  for (U64 i = (U64)blockDim.x * blockIdx.x + threadIdx.x; i < elements; i += stride) {
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

fn upload_meta(runtime: &Arc<CudaRuntime>, values: &[usize]) -> Result<Arc<GraphDeviceAllocation>> {
    let values = values.iter().map(|&v| v as u64).collect::<Vec<_>>();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values.as_slice()),
        )
    };
    let allocation = GraphDeviceAllocation::allocate(runtime, bytes.len().max(1))?;
    if !bytes.is_empty()
        && let Err(error) = unsafe { runtime.htod(bytes, allocation.ptr()) }
    {
        return Err(error);
    }
    runtime.staged_warm_cache_mutation("indexing metadata allocation/upload")?;
    Ok(allocation)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterMetadataKey {
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ScatterMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<ScatterMetadataKey>,
    allocation: Option<Arc<GraphDeviceAllocation>>,
}

impl ScatterMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            allocation: None,
        }
    }

    fn prepare(&mut self, data_shape: &[usize], indices_shape: &[usize]) -> Result<CUdeviceptr> {
        let key = ScatterMetadataKey {
            data_shape: data_shape.to_vec(),
            indices_shape: indices_shape.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            let ptr = self.allocation.as_ref().map_or_else(
                || {
                    Err(EpError::KernelFailed(
                        "cuda_ep ScatterElements: cached metadata lost its device allocation"
                            .into(),
                    ))
                },
                |allocation| Ok(allocation.ptr()),
            )?;
            if self.runtime.is_capturing()? {
                self.runtime.require_registered_address_capture(
                    self.device_graph_resource()
                        .expect("cached ScatterElements allocation is present")
                        .identity(),
                    "ScatterElements metadata",
                )?;
            }
            return Ok(ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: shape changed during CUDA graph capture; warm the exact shape first".into(),
            ));
        }
        if self.allocation.is_some() {
            self.runtime.drain_for_unmap()?;
        }

        let mut meta = compute_contiguous_strides(indices_shape)
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        meta.extend(
            compute_contiguous_strides(data_shape)
                .into_iter()
                .map(|value| value as usize),
        );
        meta.extend(data_shape.iter().copied());
        let allocation = upload_meta(&self.runtime, &meta)?;
        let ptr = allocation.ptr();
        self.key = Some(key);
        self.allocation = Some(allocation);
        Ok(ptr)
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.allocation
            .as_ref()
            .map(GraphDeviceAllocation::device_graph_resource)
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
    runtime: Arc<CudaRuntime>,
    key: Option<GatherElementsMetadataKey>,
    allocation: Option<Arc<GraphDeviceAllocation>>,
}

impl GatherElementsMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            allocation: None,
        }
    }

    fn prepare(&mut self, data_shape: &[usize], indices_shape: &[usize]) -> Result<CUdeviceptr> {
        let key = GatherElementsMetadataKey {
            data_shape: data_shape.to_vec(),
            indices_shape: indices_shape.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            let ptr = self.allocation.as_ref().map_or_else(
                || {
                    Err(EpError::KernelFailed(
                        "cuda_ep GatherElements: cached metadata lost its device allocation".into(),
                    ))
                },
                |allocation| Ok(allocation.ptr()),
            )?;
            if self.runtime.is_capturing()? {
                self.runtime.require_registered_address_capture(
                    self.device_graph_resource()
                        .expect("cached GatherElements allocation is present")
                        .identity(),
                    "GatherElements metadata",
                )?;
            }
            return Ok(ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherElements: shape changed during CUDA graph capture; warm the exact shape first".into(),
            ));
        }
        if self.allocation.is_some() {
            self.runtime.drain_for_unmap()?;
        }

        let mut meta = indices_shape.to_vec();
        meta.extend(
            compute_contiguous_strides(indices_shape)
                .into_iter()
                .map(|value| value as usize),
        );
        meta.extend(
            compute_contiguous_strides(data_shape)
                .into_iter()
                .map(|value| value as usize),
        );
        meta.extend(data_shape.iter().copied());
        let allocation = upload_meta(&self.runtime, &meta)?;
        let ptr = allocation.ptr();
        self.key = Some(key);
        self.allocation = Some(allocation);
        Ok(ptr)
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.allocation
            .as_ref()
            .map(GraphDeviceAllocation::device_graph_resource)
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

#[derive(Default)]
struct ScatterMeanScratch(Mutex<Option<Arc<GraphDeviceAllocation>>>);

impl ScatterMeanScratch {
    fn resource(&self) -> Option<DeviceGraphResource> {
        self.0
            .lock()
            .ok()?
            .as_ref()
            .map(GraphDeviceAllocation::device_graph_resource)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        runtime: &Arc<CudaRuntime>,
        capturing: bool,
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
            // The caller already validated indices; no output values exist.
            return Ok(());
        }
        let state_bytes = if data.dtype == DataType::Int64 {
            24
        } else {
            16
        };
        let bytes = data.numel().checked_mul(state_bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep Scatter mean: workspace size overflow".into())
        })?;
        let mut allocation = self.0.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep Scatter mean: workspace lock was poisoned".into())
        })?;
        if allocation
            .as_ref()
            .is_none_or(|value| value.bytes() < bytes)
        {
            if capturing {
                return Err(EpError::KernelFailed(
                    "cuda_ep Scatter mean: warm the exact shape before capture to allocate workspace".into(),
                ));
            }
            let fresh = GraphDeviceAllocation::allocate(runtime, bytes)?;
            runtime.staged_warm_cache_mutation("Scatter mean workspace allocation")?;
            *allocation = Some(fresh);
        }
        let scratch = allocation.as_ref().expect("workspace reserved").ptr();
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
            mean: ScatterMeanScratch::default(),
            metadata: Mutex::new(PersistentMetadata::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct ScatterNdKernel {
    runtime: Arc<CudaRuntime>,
    reduction: Reduction,
    opset: u64,
    mean: ScatterMeanScratch,
    metadata: Mutex<PersistentMetadata>,
    warmed_signature: Mutex<Option<ScatterNdCaptureSignature>>,
}

impl Kernel for ScatterNdKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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

        let capturing = self.runtime.is_capturing()?;
        let signature = ScatterNdCaptureSignature {
            data_dtype: data.dtype,
            indices_dtype: indices.dtype,
            data_shape: data.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
            updates_shape: updates.shape.to_vec(),
        };
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterND: capture signature lock was poisoned".into())
        })?;
        if capturing && warmed_signature.as_ref() != Some(&signature) {
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
        if !mean || tuples == 0 || slice_length == 0 {
            copy_scatter_data(&self.runtime, data, output)?;
        }
        if tuples == 0 || slice_length == 0 {
            // With an empty data axis no mean kernel can latch an invalid index.
            // Nonempty index tensors in this case must fail before touching output.
            if data.numel() == 0 && indices.numel() != 0 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep ScatterElements: indices must be empty when data axis {axis} has length zero"
                )));
            }
            if !capturing {
                *warmed_signature = Some(signature);
            }
            return Ok(());
        }

        let mut metadata_values = compute_contiguous_strides(data.shape)
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>();
        metadata_values.extend(data.shape.iter().map(|&value| value as u64));
        let mut metadata_cache = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterND: metadata lock was poisoned".into())
        })?;
        let metadata_candidate = metadata_cache.stage(&metadata_values, "ScatterND")?;
        let metadata_ptr = metadata_candidate.ptr("ScatterND")?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let indices_ptr = if index_depth == 0 {
            0
        } else {
            cuptr(indices.data_ptr::<u8>() as *const c_void)
        };
        let updates_ptr = cuptr(updates.data_ptr::<u8>() as *const c_void);
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
        if mean {
            self.mean.launch(
                &self.runtime,
                capturing,
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
            .chain(self.mean.resource())
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
            mean: ScatterMeanScratch::default(),
            metadata: Mutex::new(ScatterMetadataCache::new(self.runtime.clone())),
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct ScatterElementsKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    reduction: Reduction,
    mean: ScatterMeanScratch,
    metadata: Mutex<ScatterMetadataCache>,
    warmed_signature: Mutex<Option<ScatterCaptureSignature>>,
}

impl Kernel for ScatterElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        let capturing = self.runtime.is_capturing()?;
        let signature = ScatterCaptureSignature {
            data_dtype: data.dtype,
            indices_dtype: indices.dtype,
            data_shape: data.shape.to_vec(),
            indices_shape: indices.shape.to_vec(),
        };
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep ScatterElements: capture signature lock was poisoned".into(),
            )
        })?;
        if capturing && warmed_signature.as_ref() != Some(&signature) {
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
            if !capturing {
                *warmed_signature = Some(signature);
            }
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
            self.mean.launch(
                &self.runtime,
                capturing,
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
            .chain(self.mean.resource())
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
