//! CUDA indexed element movement: `GatherElements` and `ScatterElements`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
pub const SCATTER_CAPTURE_ERROR_INDEX: u32 = 256;
const SOURCE: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

extern "C" __global__ void gather_elements(
    const unsigned char* data, const long long* indices, unsigned char* output,
    const unsigned long long* meta, int rank, int axis, int elem_bytes,
    unsigned long long elements) {
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
        long long selected = indices[linear];
        if (selected < 0) selected += (long long)meta[3 * rank + d];
        coordinate = (unsigned long long)selected;
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

fn upload_meta(
    runtime: &CudaRuntime,
    values: &[usize],
) -> Result<cudarc::driver::sys::CUdeviceptr> {
    let values = values.iter().map(|&v| v as u64).collect::<Vec<_>>();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values.as_slice()),
        )
    };
    let ptr = runtime.alloc_raw(bytes.len().max(1))?;
    if !bytes.is_empty()
        && let Err(error) = unsafe { runtime.htod(bytes, ptr) }
    {
        let _ = unsafe { runtime.free_raw(ptr) };
        return Err(error);
    }
    Ok(ptr)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterMetadataKey {
    data_shape: Vec<usize>,
    indices_shape: Vec<usize>,
}

#[derive(Debug)]
struct ScatterMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<ScatterMetadataKey>,
    ptr: CUdeviceptr,
}

impl ScatterMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            ptr: 0,
        }
    }

    fn prepare(&mut self, data_shape: &[usize], indices_shape: &[usize]) -> Result<CUdeviceptr> {
        let key = ScatterMetadataKey {
            data_shape: data_shape.to_vec(),
            indices_shape: indices_shape.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            return Ok(self.ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep ScatterElements: shape changed during CUDA graph capture; warm the exact shape first".into(),
            ));
        }
        if self.ptr != 0 {
            self.runtime.synchronize()?;
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
        let ptr = upload_meta(&self.runtime, &meta)?;
        if self.ptr != 0 {
            // SAFETY: synchronization above completed every prior launch using
            // the old cache-owned pointer.
            unsafe { self.runtime.free_raw(self.ptr) }?;
        }
        self.key = Some(key);
        self.ptr = ptr;
        Ok(ptr)
    }
}

impl Drop for ScatterMetadataCache {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the cache exclusively owns this persistent allocation.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScatterCaptureSignature {
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
        }))
    }
}

struct GatherElementsKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
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
        validate_indices(&self.runtime, indices, data.shape[axis], "GatherElements")?;
        if output.numel() == 0 {
            return Ok(());
        }
        let elem_bytes = i32::try_from(data.dtype.byte_size()).map_err(|_| {
            EpError::KernelFailed("cuda_ep GatherElements: element width exceeds i32".into())
        })?;
        if elem_bytes == 0 {
            return Err(not_implemented("GatherElements for variable-width dtype"));
        }
        let mut meta = indices.shape.to_vec();
        meta.extend(
            compute_contiguous_strides(indices.shape)
                .into_iter()
                .map(|v| v as usize),
        );
        meta.extend(
            compute_contiguous_strides(data.shape)
                .into_iter()
                .map(|v| v as usize),
        );
        meta.extend(data.shape);
        let meta_ptr = upload_meta(&self.runtime, &meta)?;
        let result = (|| {
            let func = self
                .runtime
                .nvrtc_function("indexing_ops", SOURCE, "gather_elements")?;
            let data_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
            let indices_ptr = cuptr(indices.data_ptr::<i64>() as *const c_void);
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            let rank = rank as i32;
            let axis = axis as i32;
            let elements = output.numel() as u64;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&data_ptr)
                .arg(&indices_ptr)
                .arg(&output_ptr)
                .arg(&meta_ptr)
                .arg(&rank)
                .arg(&axis)
                .arg(&elem_bytes)
                .arg(&elements);
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
            self.runtime.synchronize()
        })();
        let free = unsafe { self.runtime.free_raw(meta_ptr) };
        result.and(free)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "GatherElements allocates/uploads/frees per-call indexing metadata and synchronizes the stream",
        )
    }
}

#[derive(Clone, Copy)]
enum Reduction {
    None = 0,
    Add = 1,
    Mul = 2,
    Max = 3,
    Min = 4,
}

pub struct ScatterElementsFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ScatterElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let reduction = match node.attr("reduction") {
            None => Reduction::None,
            Some(attribute) => match attribute.as_str() {
                Some("none") => Reduction::None,
                Some("add") => Reduction::Add,
                Some("mul") => Reduction::Mul,
                Some("max") => Reduction::Max,
                Some("min") => Reduction::Min,
                Some(value) => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep ScatterElements: unsupported reduction {value:?}"
                    )));
                }
                None => {
                    return Err(EpError::KernelFailed(
                        "cuda_ep ScatterElements: reduction must be a string".into(),
                    ));
                }
            },
        };
        Ok(Box::new(ScatterElementsKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(0),
            reduction,
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
        unsafe {
            self.runtime.dtod_async(
                cuptr(data.data_ptr::<u8>() as *const c_void),
                cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                data.dtype.storage_bytes(data.numel()),
            )?
        };
        if indices.numel() == 0 {
            if !capturing {
                *warmed_signature = Some(signature);
            }
            return Ok(());
        }
        let data_suffix = match data.dtype {
            DataType::Float16 => "f16",
            DataType::Float32 => "f32",
            DataType::BFloat16 => "bf16",
            DataType::Int64 => "i64",
            _ => unreachable!("validated above"),
        };
        let index_suffix = match indices.dtype {
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            _ => unreachable!("validated above"),
        };
        let entry = format!("scatter_{data_suffix}_{index_suffix}");
        let func = self
            .runtime
            .nvrtc_function("indexing_ops_v2", SOURCE, &entry)?;
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ScatterElements: metadata lock was poisoned".into())
        })?;
        let meta_ptr = metadata.prepare(data.shape, indices.shape)?;
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
        let capture_error = if capturing {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
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
        if !capturing {
            *warmed_signature = Some(signature);
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
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
