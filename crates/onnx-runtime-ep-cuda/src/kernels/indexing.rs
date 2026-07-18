//! CUDA indexed element movement: `GatherElements` and `ScatterElements`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
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

extern "C" __global__ void scatter_f32(
    float* output, const long long* indices, const float* updates,
    const unsigned long long* meta, int rank, int axis,
    unsigned long long elements, int reduction) {
  if (blockIdx.x || threadIdx.x) return;
  const unsigned long long* index_strides = meta;
  const unsigned long long* data_strides = meta + rank;
  const unsigned long long* data_dims = meta + 2 * rank;
  for (unsigned long long linear = 0; linear < elements; ++linear) {
    unsigned long long rem = linear, data_offset = 0;
    for (int d = 0; d < rank; ++d) {
      unsigned long long coordinate = rem / index_strides[d];
      rem %= index_strides[d];
      if (d == axis) {
        long long selected = indices[linear];
        if (selected < 0) selected += (long long)data_dims[d];
        coordinate = (unsigned long long)selected;
      }
      data_offset += coordinate * data_strides[d];
    }
    float update = updates[linear];
    if (reduction == 0) output[data_offset] = update;
    else if (reduction == 1) output[data_offset] += update;
    else if (reduction == 2) output[data_offset] *= update;
    else if (reduction == 3) {
      float current = output[data_offset];
      output[data_offset] = (isnan(current) || isnan(update)) ? __int_as_float(0x7fc00000)
                                                               : fmaxf(current, update);
    } else {
      float current = output[data_offset];
      output[data_offset] = (isnan(current) || isnan(update)) ? __int_as_float(0x7fc00000)
                                                               : fminf(current, update);
    }
  }
}

extern "C" __global__ void scatter_i64(
    long long* output, const long long* indices, const long long* updates,
    const unsigned long long* meta, int rank, int axis,
    unsigned long long elements, int reduction) {
  if (blockIdx.x || threadIdx.x) return;
  const unsigned long long* index_strides = meta;
  const unsigned long long* data_strides = meta + rank;
  const unsigned long long* data_dims = meta + 2 * rank;
  for (unsigned long long linear = 0; linear < elements; ++linear) {
    unsigned long long rem = linear, data_offset = 0;
    for (int d = 0; d < rank; ++d) {
      unsigned long long coordinate = rem / index_strides[d];
      rem %= index_strides[d];
      if (d == axis) {
        long long selected = indices[linear];
        if (selected < 0) selected += (long long)data_dims[d];
        coordinate = (unsigned long long)selected;
      }
      data_offset += coordinate * data_strides[d];
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
    if indices.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: indices must be Int64"
        )));
    }
    let mut bytes = vec![0_u8; indices.dtype.storage_bytes(indices.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(indices.data_ptr::<u8>() as *const c_void))? };
    }
    for raw in bytes.chunks_exact(8) {
        let value = i64::from_ne_bytes(raw.try_into().unwrap());
        let normalized = if value < 0 { value + dim as i64 } else { value };
        if normalized < 0 || normalized >= dim as i64 {
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
    if !bytes.is_empty() {
        if let Err(error) = unsafe { runtime.htod(bytes, ptr) } {
            let _ = unsafe { runtime.free_raw(ptr) };
            return Err(error);
        }
    }
    Ok(ptr)
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
                        (elements.div_ceil(BLOCK as u64).min(65_535).max(1) as u32),
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
    fn cuda_graph_compatible(&self) -> bool {
        false
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
        }))
    }
}

struct ScatterElementsKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    reduction: Reduction,
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
        if !matches!(data.dtype, DataType::Float32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "ScatterElements supports Float32 and Int64, got {:?}",
                data.dtype
            )));
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
        validate_indices(&self.runtime, indices, data.shape[axis], "ScatterElements")?;
        unsafe {
            self.runtime.dtod(
                cuptr(data.data_ptr::<u8>() as *const c_void),
                cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                data.dtype.storage_bytes(data.numel()),
            )?
        };
        if indices.numel() == 0 {
            return Ok(());
        }
        let mut meta = compute_contiguous_strides(indices.shape)
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();
        meta.extend(
            compute_contiguous_strides(data.shape)
                .into_iter()
                .map(|v| v as usize),
        );
        meta.extend(data.shape.iter().copied());
        let meta_ptr = upload_meta(&self.runtime, &meta)?;
        let result = (|| {
            let entry = if data.dtype == DataType::Float32 {
                "scatter_f32"
            } else {
                "scatter_i64"
            };
            let func = self.runtime.nvrtc_function("indexing_ops", SOURCE, entry)?;
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            let indices_ptr = cuptr(indices.data_ptr::<i64>() as *const c_void);
            let updates_ptr = cuptr(updates.data_ptr::<u8>() as *const c_void);
            let rank = rank as i32;
            let axis = axis as i32;
            let elements = indices.numel() as u64;
            let reduction = self.reduction as i32;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&output_ptr)
                .arg(&indices_ptr)
                .arg(&updates_ptr)
                .arg(&meta_ptr)
                .arg(&rank)
                .arg(&axis)
                .arg(&elements)
                .arg(&reduction);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err("launch ScatterElements", e))?;
            self.runtime.synchronize()
        })();
        let free = unsafe { self.runtime.free_raw(meta_ptr) };
        result.and(free)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
