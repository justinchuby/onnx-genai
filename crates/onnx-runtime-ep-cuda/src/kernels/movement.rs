//! Dtype-agnostic CUDA kernels for ONNX construction and movement operators.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use super::elementwise::{broadcast_strides, u64_bytes};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MOVEMENT_SOURCE: &str = r#"
extern "C" __global__ void expand_bytes(
    const unsigned char* input, unsigned char* output,
    const unsigned long long* metadata, int rank,
    int elem_bytes, unsigned long long elements) {
  const unsigned long long* dims = metadata;
  const unsigned long long* strides = metadata + rank;
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = out, src = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
      unsigned long long coord = rem % dims[axis];
      rem /= dims[axis];
      src += coord * strides[axis];
    }
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}

extern "C" __global__ void transpose_bytes(
    const unsigned char* input, unsigned char* output,
    const unsigned long long* metadata, int rank,
    int elem_bytes, unsigned long long elements) {
  const unsigned long long* dims = metadata;
  const unsigned long long* strides = metadata + rank;
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = out, src = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
      unsigned long long coord = rem % dims[axis];
      rem /= dims[axis];
      src += coord * strides[axis];
    }
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}

extern "C" __global__ void slice_bytes(
    const unsigned char* input, unsigned char* output,
    const unsigned long long* dims, const long long* strides,
    long long base, int rank, int elem_bytes, unsigned long long elements) {
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = out;
    long long src = base;
    for (int axis = rank - 1; axis >= 0; --axis) {
      unsigned long long coord = rem % dims[axis];
      rem /= dims[axis];
      src += (long long)coord * strides[axis];
    }
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}

extern "C" __global__ void tile_bytes(
    const unsigned char* input, unsigned char* output,
    const unsigned long long* metadata, int rank,
    int elem_bytes, unsigned long long elements) {
  const unsigned long long* out_dims = metadata;
  const unsigned long long* in_dims = metadata + rank;
  const unsigned long long* in_strides = metadata + 2 * rank;
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = out, src = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
      unsigned long long coord = rem % out_dims[axis];
      rem /= out_dims[axis];
      src += (coord % in_dims[axis]) * in_strides[axis];
    }
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}

extern "C" __global__ void concat_chunk_bytes(
    const unsigned char* input, unsigned char* output,
    unsigned long long elements, unsigned long long input_axis,
    unsigned long long output_axis, unsigned long long axis_prefix,
    unsigned long long inner, int elem_bytes) {
  for (unsigned long long src = blockIdx.x * blockDim.x + threadIdx.x; src < elements;
       src += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long inner_index = src % inner;
    unsigned long long axis_index = (src / inner) % input_axis;
    unsigned long long outer = src / (inner * input_axis);
    unsigned long long dst = ((outer * output_axis + axis_prefix + axis_index) * inner)
                           + inner_index;
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[dst * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}

extern "C" __global__ void split_chunk_bytes(
    const unsigned char* input, unsigned char* output,
    unsigned long long elements, unsigned long long input_axis,
    unsigned long long output_axis, unsigned long long axis_prefix,
    unsigned long long inner, int elem_bytes) {
  for (unsigned long long dst = blockIdx.x * blockDim.x + threadIdx.x; dst < elements;
       dst += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long inner_index = dst % inner;
    unsigned long long axis_index = (dst / inner) % output_axis;
    unsigned long long outer = dst / (inner * output_axis);
    unsigned long long src = ((outer * input_axis + axis_prefix + axis_index) * inner)
                           + inner_index;
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[dst * elem_bytes + byte] = input[src * elem_bytes + byte];
  }
}
"#;

fn arity(
    op: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    out: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() != out {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: expected {min}..={max} inputs and {out} outputs, got {} and {}",
            inputs.len(),
            outputs.len()
        )));
    }
    Ok(())
}

fn require_dense(op: &str, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if inputs.iter().any(|v| !v.is_absent() && !v.is_contiguous())
        || outputs.iter().any(|v| !v.is_contiguous())
    {
        return Err(not_implemented(format!(
            "{op} with non-contiguous input/output"
        )));
    }
    Ok(())
}

fn fixed_width(op: &str, dtype: DataType) -> Result<usize> {
    let bytes = dtype.byte_size();
    if bytes == 0 {
        Err(not_implemented(format!(
            "{op} for packed or variable-width dtype {dtype:?}"
        )))
    } else {
        Ok(bytes)
    }
}

fn product(dims: &[usize], op: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep {op}: shape product overflow")))
    })
}

fn grid(elements: usize) -> u32 {
    (elements as u64).div_ceil(BLOCK as u64).clamp(1, 65_535) as u32
}

fn host_ints(runtime: &CudaRuntime, view: &TensorView, op: &str) -> Result<Vec<i64>> {
    if !matches!(view.dtype, DataType::Int32 | DataType::Int64) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: integer metadata input must be Int32 or Int64, got {:?}",
            view.dtype
        )));
    }
    if !view.is_contiguous() {
        return Err(not_implemented(format!("{op} with strided metadata input")));
    }
    let mut bytes = vec![0u8; view.dtype.storage_bytes(view.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))? };
    }
    Ok(match view.dtype {
        DataType::Int32 => bytes
            .chunks_exact(4)
            .map(|v| i32::from_ne_bytes(v.try_into().unwrap()) as i64)
            .collect(),
        DataType::Int64 => bytes
            .chunks_exact(8)
            .map(|v| i64::from_ne_bytes(v.try_into().unwrap()))
            .collect(),
        _ => unreachable!(),
    })
}

fn launch_metadata(
    runtime: &CudaRuntime,
    entry: &'static str,
    input: &TensorView,
    output: &mut TensorMut,
    metadata: &[u64],
) -> Result<()> {
    let elements = output.numel();
    if elements == 0 {
        return Ok(());
    }
    let rank = i32::try_from(output.shape.len())
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep {entry}: rank exceeds i32")))?;
    let elem_bytes = i32::try_from(fixed_width(entry, input.dtype)?).map_err(|_| {
        EpError::KernelFailed(format!("cuda_ep {entry}: element width exceeds i32"))
    })?;
    let elements_u64 = elements as u64;
    let scalar_metadata = [0_u64];
    let bytes = u64_bytes(if metadata.is_empty() {
        &scalar_metadata
    } else {
        metadata
    });
    let func = runtime.nvrtc_function("movement_ops", MOVEMENT_SOURCE, entry)?;
    let metadata_ptr = runtime.alloc_raw(bytes.len())?;
    if let Err(error) = unsafe { runtime.htod(bytes, metadata_ptr) } {
        let _ = unsafe { runtime.free_raw(metadata_ptr) };
        return Err(error);
    }
    let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
    let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
    let mut builder = runtime.stream().launch_builder(&func);
    builder
        .arg(&input_ptr)
        .arg(&output_ptr)
        .arg(&metadata_ptr)
        .arg(&rank)
        .arg(&elem_bytes)
        .arg(&elements_u64);
    let launch = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid(elements), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| driver_err(&format!("launch {entry}"), e));
    let sync = launch.and_then(|_| runtime.synchronize());
    let free = unsafe { runtime.free_raw(metadata_ptr) };
    sync.and(free)
}

fn copy_reshape(
    runtime: &CudaRuntime,
    op: &str,
    input: &TensorView,
    output: &mut TensorMut,
) -> Result<()> {
    if input.dtype != output.dtype || input.numel() != output.numel() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: input/output dtype and element count must match"
        )));
    }
    let bytes = input.dtype.storage_bytes(input.numel());
    if bytes != 0 {
        unsafe {
            runtime.dtod(
                cuptr(input.data_ptr::<u8>() as *const c_void),
                cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                bytes,
            )?
        };
    }
    Ok(())
}

pub struct ReshapeFactory {
    pub runtime: Arc<CudaRuntime>,
}
pub struct SqueezeFactory {
    pub runtime: Arc<CudaRuntime>,
}

macro_rules! copy_factory {
    ($factory:ident, $kernel:ident, $op:literal, $min:literal, $max:literal) => {
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new($kernel {
                    runtime: self.runtime.clone(),
                }))
            }
        }
        #[derive(Debug)]
        struct $kernel {
            runtime: Arc<CudaRuntime>,
        }
        impl Kernel for $kernel {
            fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
                arity($op, inputs, outputs, $min, $max, 1)?;
                require_dense($op, inputs, outputs)?;
                copy_reshape(&self.runtime, $op, &inputs[0], &mut outputs[0])
            }
            fn supports_strided_input(&self, _idx: usize) -> bool {
                false
            }
            fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
                onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                    "{} uses the copy path rather than a capture-validated zero-copy view",
                    $op
                ))
            }
        }
    };
}
copy_factory!(ReshapeFactory, ReshapeKernel, "Reshape", 1, 2);
copy_factory!(SqueezeFactory, SqueezeKernel, "Squeeze", 1, 2);

pub struct UnsqueezeFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for UnsqueezeFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(UnsqueezeKernel {
            runtime: self.runtime.clone(),
            axes: node
                .attr("axes")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
        }))
    }
}
#[derive(Debug)]
struct UnsqueezeKernel {
    runtime: Arc<CudaRuntime>,
    axes: Option<Vec<i64>>,
}
impl Kernel for UnsqueezeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Unsqueeze", inputs, outputs, 1, 2, 1)?;
        require_dense("Unsqueeze", inputs, outputs)?;
        let axes_len = if inputs.get(1).is_some_and(|v| !v.is_absent()) {
            host_ints(&self.runtime, &inputs[1], "Unsqueeze")?.len()
        } else {
            self.axes
                .as_ref()
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep Unsqueeze: axes input or attribute is required".into(),
                    )
                })?
                .len()
        };
        if outputs[0].shape.len() != inputs[0].shape.len() + axes_len {
            return Err(EpError::KernelFailed(
                "cuda_ep Unsqueeze: output rank mismatch".into(),
            ));
        }
        copy_reshape(&self.runtime, "Unsqueeze", &inputs[0], &mut outputs[0])
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Unsqueeze reads runtime axes on the host and uses a non-validated copy path",
        )
    }
}

pub struct ExpandFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for ExpandFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ExpandKernel {
            runtime: self.runtime.clone(),
        }))
    }
}
#[derive(Debug)]
struct ExpandKernel {
    runtime: Arc<CudaRuntime>,
}
impl Kernel for ExpandKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Expand", inputs, outputs, 2, 2, 1)?;
        require_dense("Expand", inputs, outputs)?;
        if inputs[0].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Expand: output dtype must match input".into(),
            ));
        }
        let out_shape = outputs[0].shape.to_vec();
        let expected =
            onnx_runtime_ir::broadcast_shapes(inputs[0].shape, &out_shape).map_err(EpError::Ir)?;
        if expected != out_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Expand: output shape {out_shape:?}, expected broadcast shape {expected:?}"
            )));
        }
        let mut metadata = out_shape.iter().map(|&v| v as u64).collect::<Vec<_>>();
        metadata.extend(broadcast_strides(inputs[0].shape, &out_shape));
        launch_metadata(
            &self.runtime,
            "expand_bytes",
            &inputs[0],
            &mut outputs[0],
            &metadata,
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Expand allocates/uploads/frees per-call broadcast metadata and synchronizes the stream",
        )
    }
}

pub struct TransposeFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for TransposeFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let perm = node
            .attr("perm")
            .and_then(Attribute::as_ints)
            .map(<[i64]>::to_vec);
        Ok(Box::new(TransposeKernel {
            runtime: self.runtime.clone(),
            perm,
        }))
    }
}
#[derive(Debug)]
struct TransposeKernel {
    runtime: Arc<CudaRuntime>,
    perm: Option<Vec<i64>>,
}
impl Kernel for TransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Transpose", inputs, outputs, 1, 1, 1)?;
        require_dense("Transpose", inputs, outputs)?;
        if inputs[0].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Transpose: output dtype must match input".into(),
            ));
        }
        let rank = inputs[0].shape.len();
        let perm = self
            .perm
            .clone()
            .unwrap_or_else(|| (0..rank as i64).rev().collect());
        if perm.len() != rank {
            return Err(EpError::KernelFailed(
                "cuda_ep Transpose: perm rank mismatch".into(),
            ));
        }
        let in_strides = compute_contiguous_strides(inputs[0].shape);
        let mut seen = vec![false; rank];
        let mut expected = Vec::with_capacity(rank);
        let mut metadata = outputs[0]
            .shape
            .iter()
            .map(|&v| v as u64)
            .collect::<Vec<_>>();
        for &axis in &perm {
            let axis = usize::try_from(axis).map_err(|_| {
                EpError::KernelFailed("cuda_ep Transpose: negative perm axis".into())
            })?;
            if axis >= rank || seen[axis] {
                return Err(EpError::KernelFailed(
                    "cuda_ep Transpose: perm must be an axis permutation".into(),
                ));
            }
            seen[axis] = true;
            expected.push(inputs[0].shape[axis]);
            metadata.push(in_strides[axis] as u64);
        }
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Transpose: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        launch_metadata(
            &self.runtime,
            "transpose_bytes",
            &inputs[0],
            &mut outputs[0],
            &metadata,
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Transpose allocates/uploads/frees per-call permutation metadata and synchronizes the stream",
        )
    }
}

#[derive(Clone, Copy)]
struct SlicePlan {
    start: i64,
    step: i64,
    count: usize,
}
fn slice_plan(
    shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Vec<SlicePlan>> {
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(EpError::KernelFailed(
            "cuda_ep Slice: starts/ends/axes/steps length mismatch".into(),
        ));
    }
    let rank = shape.len();
    let mut plan = shape
        .iter()
        .map(|&count| SlicePlan {
            start: 0,
            step: 1,
            count,
        })
        .collect::<Vec<_>>();
    for i in 0..starts.len() {
        let axis = if axes[i] < 0 {
            axes[i] + rank as i64
        } else {
            axes[i]
        };
        if !(0..rank as i64).contains(&axis) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Slice: axis {} out of range",
                axes[i]
            )));
        }
        let dim = shape[axis as usize] as i64;
        let step = steps[i];
        if step == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Slice: step must be non-zero".into(),
            ));
        }
        if dim == 0 {
            plan[axis as usize] = SlicePlan {
                start: 0,
                step,
                count: 0,
            };
            continue;
        }
        let mut start = starts[i];
        let mut end = ends[i];
        if start < 0 {
            start += dim;
        }
        if end < 0 {
            end += dim;
        }
        let (start, end) = if step < 0 {
            (start.clamp(0, dim - 1), end.clamp(-1, dim - 1))
        } else {
            (start.clamp(0, dim), end.clamp(0, dim))
        };
        let count = if step > 0 && end > start {
            ((end - start + step - 1) / step) as usize
        } else if step < 0 && start > end {
            ((start - end + (-step) - 1) / (-step)) as usize
        } else {
            0
        };
        plan[axis as usize] = SlicePlan { start, step, count };
    }
    Ok(plan)
}

pub struct SliceFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for SliceFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SliceKernel {
            runtime: self.runtime.clone(),
        }))
    }
}
#[derive(Debug)]
struct SliceKernel {
    runtime: Arc<CudaRuntime>,
}
impl Kernel for SliceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Slice", inputs, outputs, 3, 5, 1)?;
        require_dense("Slice", inputs, outputs)?;
        if inputs[0].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Slice: output dtype must match data".into(),
            ));
        }
        let starts = host_ints(&self.runtime, &inputs[1], "Slice")?;
        let ends = host_ints(&self.runtime, &inputs[2], "Slice")?;
        let axes = if inputs.get(3).is_some_and(|v| !v.is_absent()) {
            host_ints(&self.runtime, &inputs[3], "Slice")?
        } else {
            (0..starts.len() as i64).collect()
        };
        let steps = if inputs.get(4).is_some_and(|v| !v.is_absent()) {
            host_ints(&self.runtime, &inputs[4], "Slice")?
        } else {
            vec![1; starts.len()]
        };
        let plan = slice_plan(inputs[0].shape, &starts, &ends, &axes, &steps)?;
        let expected = plan.iter().map(|p| p.count).collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Slice: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        if outputs[0].numel() == 0 {
            return Ok(());
        }
        if expected.is_empty() {
            return copy_reshape(&self.runtime, "Slice", &inputs[0], &mut outputs[0]);
        }
        let contiguous = compute_contiguous_strides(inputs[0].shape);
        let dims = expected.iter().map(|&v| v as u64).collect::<Vec<_>>();
        let strides = plan
            .iter()
            .zip(&contiguous)
            .map(|(p, &s)| p.step * s)
            .collect::<Vec<_>>();
        let base = plan
            .iter()
            .zip(&contiguous)
            .map(|(p, &s)| p.start * s)
            .sum::<i64>();
        let func = self
            .runtime
            .nvrtc_function("movement_ops", MOVEMENT_SOURCE, "slice_bytes")?;
        let dims_ptr = self
            .runtime
            .alloc_raw(std::mem::size_of_val(dims.as_slice()))?;
        let strides_bytes = unsafe {
            std::slice::from_raw_parts(
                strides.as_ptr().cast::<u8>(),
                std::mem::size_of_val(strides.as_slice()),
            )
        };
        let strides_ptr = match self.runtime.alloc_raw(strides_bytes.len()) {
            Ok(ptr) => ptr,
            Err(error) => {
                let _ = unsafe { self.runtime.free_raw(dims_ptr) };
                return Err(error);
            }
        };
        if let Err(error) = unsafe { self.runtime.htod(u64_bytes(&dims), dims_ptr) } {
            let _ = unsafe { self.runtime.free_raw(dims_ptr) };
            let _ = unsafe { self.runtime.free_raw(strides_ptr) };
            return Err(error);
        }
        if let Err(error) = unsafe { self.runtime.htod(strides_bytes, strides_ptr) } {
            let _ = unsafe { self.runtime.free_raw(dims_ptr) };
            let _ = unsafe { self.runtime.free_raw(strides_ptr) };
            return Err(error);
        }
        let input_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let rank = expected.len() as i32;
        let elem_bytes = fixed_width("Slice", inputs[0].dtype)? as i32;
        let elements = outputs[0].numel() as u64;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&dims_ptr)
            .arg(&strides_ptr)
            .arg(&base)
            .arg(&rank)
            .arg(&elem_bytes)
            .arg(&elements);
        let launch = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid(outputs[0].numel()), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch slice_bytes", e));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free_dims = unsafe { self.runtime.free_raw(dims_ptr) };
        let free_strides = unsafe { self.runtime.free_raw(strides_ptr) };
        sync.and(free_dims).and(free_strides)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Slice reads runtime bounds on the host, allocates per-call metadata, and synchronizes the stream",
        )
    }
}

pub struct TileFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for TileFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TileKernel {
            runtime: self.runtime.clone(),
        }))
    }
}
#[derive(Debug)]
struct TileKernel {
    runtime: Arc<CudaRuntime>,
}
impl Kernel for TileKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Tile", inputs, outputs, 2, 2, 1)?;
        require_dense("Tile", inputs, outputs)?;
        if inputs[0].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Tile: output dtype must match input".into(),
            ));
        }
        let repeats = host_ints(&self.runtime, &inputs[1], "Tile")?;
        if repeats.len() != inputs[0].shape.len() || repeats.iter().any(|&r| r < 0) {
            return Err(EpError::KernelFailed(
                "cuda_ep Tile: repeats must be non-negative and match input rank".into(),
            ));
        }
        let expected = inputs[0]
            .shape
            .iter()
            .zip(&repeats)
            .map(|(&d, &r)| d * r as usize)
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(
                "cuda_ep Tile: output shape does not match repeats".into(),
            ));
        }
        let mut metadata = expected.iter().map(|&v| v as u64).collect::<Vec<_>>();
        metadata.extend(inputs[0].shape.iter().map(|&v| v as u64));
        metadata.extend(
            compute_contiguous_strides(inputs[0].shape)
                .iter()
                .map(|&v| v as u64),
        );
        launch_metadata(
            &self.runtime,
            "tile_bytes",
            &inputs[0],
            &mut outputs[0],
            &metadata,
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Tile reads repeats on the host, allocates per-call metadata, and synchronizes the stream",
        )
    }
}

pub struct ConcatFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for ConcatFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ConcatKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(0),
        }))
    }
}
#[derive(Debug)]
struct ConcatKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
}
impl Kernel for ConcatKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep Concat: requires inputs and one output".into(),
            ));
        }
        require_dense("Concat", inputs, outputs)?;
        let rank = inputs[0].shape.len();
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if !(0..rank as i64).contains(&axis) {
            return Err(EpError::KernelFailed(
                "cuda_ep Concat: axis out of range".into(),
            ));
        }
        let axis = axis as usize;
        let dtype = inputs[0].dtype;
        let elem_bytes = fixed_width("Concat", dtype)? as i32;
        let mut expected = inputs[0].shape.to_vec();
        expected[axis] = 0;
        for input in inputs {
            if input.dtype != dtype
                || input.shape.len() != rank
                || (0..rank).any(|d| d != axis && input.shape[d] != inputs[0].shape[d])
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep Concat: incompatible input dtype or shape".into(),
                ));
            }
            expected[axis] += input.shape[axis];
        }
        if outputs[0].dtype != dtype || outputs[0].shape != expected {
            return Err(EpError::KernelFailed(
                "cuda_ep Concat: output dtype or shape mismatch".into(),
            ));
        }
        let inner = product(&expected[axis + 1..], "Concat")? as u64;
        let output_axis = expected[axis] as u64;
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let func =
            self.runtime
                .nvrtc_function("movement_ops", MOVEMENT_SOURCE, "concat_chunk_bytes")?;
        let mut prefix = 0u64;
        for input in inputs {
            let elements = input.numel() as u64;
            if elements != 0 {
                let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
                let input_axis = input.shape[axis] as u64;
                let mut builder = self.runtime.stream().launch_builder(&func);
                builder
                    .arg(&input_ptr)
                    .arg(&output_ptr)
                    .arg(&elements)
                    .arg(&input_axis)
                    .arg(&output_axis)
                    .arg(&prefix)
                    .arg(&inner)
                    .arg(&elem_bytes);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid(input.numel()), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| driver_err("launch concat_chunk_bytes", e))?;
            }
            prefix += input.shape[axis] as u64;
        }
        self.runtime.synchronize()
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Concat performs a trailing host stream synchronization",
        )
    }
}

fn even_split(dim: usize, n: usize) -> Result<Vec<usize>> {
    if n == 0 {
        return Err(EpError::KernelFailed("cuda_ep Split: zero outputs".into()));
    }
    if dim.is_multiple_of(n) {
        return Ok(vec![dim / n; n]);
    }
    let chunk = dim / n + 1;
    if chunk * (n - 1) > dim {
        return Err(EpError::KernelFailed(
            "cuda_ep Split: too many outputs for axis extent".into(),
        ));
    }
    let mut sizes = vec![chunk; n - 1];
    sizes.push(dim - chunk * (n - 1));
    Ok(sizes)
}

/// Resolve a (possibly negative) axis attribute against a concrete rank.
fn resolve_split_axis(axis_attr: i64, rank: usize) -> Option<usize> {
    let axis = if axis_attr < 0 {
        axis_attr + rank as i64
    } else {
        axis_attr
    };
    if (0..rank as i64).contains(&axis) {
        Some(axis as usize)
    } else {
        None
    }
}

/// Per-output offsets/sizes that are fully known at build time for the static,
/// single-data-input Split form. Precomputing these once lets the launch path
/// avoid any host read of split sizes and, crucially, drop the trailing stream
/// synchronization so the op is safe to record inside a CUDA graph capture.
#[derive(Debug, Clone)]
struct StaticSplitPlan {
    axis: usize,
    axis_extent: usize,
    sizes: Vec<usize>,
}

/// Detect the CUDA-graph-capturable Split form at build time.
///
/// Capturable requires a single data input (no runtime split-size tensor),
/// statically known split sizes (from the `split` attribute or an even split
/// derived from `num_outputs`/output count), and an axis that resolves to a
/// concrete in-range dimension. The two-input / data-dependent form returns
/// `None` and keeps the host-read-plus-synchronize path.
fn build_static_split_plan(
    node: &Node,
    input_shapes: &[Vec<usize>],
    axis_attr: i64,
    split: Option<&[i64]>,
    num_outputs: Option<i64>,
) -> Option<StaticSplitPlan> {
    // A resolved data shape must be available; the test/introspection path that
    // supplies no shapes cannot be planned statically.
    let data_shape = input_shapes.first()?;
    if data_shape.is_empty() {
        return None;
    }
    // Reject the data-dependent form: any wired second input carries runtime
    // split sizes that would require a host read.
    if input_shapes.get(1).is_some_and(|shape| !shape.is_empty()) {
        return None;
    }
    let axis = resolve_split_axis(axis_attr, data_shape.len())?;
    let axis_extent = data_shape[axis];
    let output_count = node.outputs.len();
    if output_count == 0 {
        return None;
    }
    let sizes: Vec<usize> = if let Some(split) = split {
        split
            .iter()
            .map(|&value| usize::try_from(value).ok())
            .collect::<Option<Vec<_>>>()?
    } else {
        let count = num_outputs
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(output_count);
        even_split(axis_extent, count).ok()?
    };
    if sizes.len() != output_count || sizes.iter().sum::<usize>() != axis_extent {
        return None;
    }
    Some(StaticSplitPlan {
        axis,
        axis_extent,
        sizes,
    })
}

pub struct SplitFactory {
    pub runtime: Arc<CudaRuntime>,
}
impl KernelFactory for SplitFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(Attribute::as_int).unwrap_or(0);
        let split = node
            .attr("split")
            .and_then(Attribute::as_ints)
            .map(<[i64]>::to_vec);
        let num_outputs = node.attr("num_outputs").and_then(Attribute::as_int);
        let static_plan =
            build_static_split_plan(node, input_shapes, axis, split.as_deref(), num_outputs);
        Ok(Box::new(SplitKernel {
            runtime: self.runtime.clone(),
            axis,
            split,
            num_outputs,
            static_plan,
        }))
    }
}
#[derive(Debug)]
struct SplitKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    split: Option<Vec<i64>>,
    num_outputs: Option<i64>,
    /// Precomputed capturable plan for the static, single-data-input form; the
    /// dynamic (runtime split-size) form leaves this `None`.
    static_plan: Option<StaticSplitPlan>,
}
impl SplitKernel {
    /// Launch one copy kernel per output on the runtime stream. Validates each
    /// output's dtype/shape against the chosen split. Does not synchronize:
    /// the copies are ordered on the stream and callers add a host sync only
    /// for the dynamic form.
    fn launch_copies(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        axis: usize,
        sizes: &[usize],
    ) -> Result<()> {
        let dtype = inputs[0].dtype;
        let elem_bytes = fixed_width("Split", dtype)? as i32;
        let inner = product(&inputs[0].shape[axis + 1..], "Split")? as u64;
        let input_axis = inputs[0].shape[axis] as u64;
        let input_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let func =
            self.runtime
                .nvrtc_function("movement_ops", MOVEMENT_SOURCE, "split_chunk_bytes")?;
        let mut prefix = 0u64;
        for (output, &size) in outputs.iter_mut().zip(sizes) {
            let mut expected = inputs[0].shape.to_vec();
            expected[axis] = size;
            if output.dtype != dtype || output.shape != expected {
                return Err(EpError::KernelFailed(
                    "cuda_ep Split: output dtype or shape mismatch".into(),
                ));
            }
            let elements = output.numel() as u64;
            if elements != 0 {
                let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
                let output_axis = size as u64;
                let mut builder = self.runtime.stream().launch_builder(&func);
                builder
                    .arg(&input_ptr)
                    .arg(&output_ptr)
                    .arg(&elements)
                    .arg(&input_axis)
                    .arg(&output_axis)
                    .arg(&prefix)
                    .arg(&inner)
                    .arg(&elem_bytes);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid(output.numel()), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| driver_err("launch split_chunk_bytes", e))?;
            }
            prefix += size as u64;
        }
        Ok(())
    }
}
impl Kernel for SplitKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        arity("Split", inputs, outputs, 1, 2, outputs.len())?;
        if outputs.is_empty() {
            return Err(EpError::KernelFailed("cuda_ep Split: no outputs".into()));
        }
        require_dense("Split", inputs, outputs)?;
        let rank = inputs[0].shape.len();
        // Capturable fast path: the static, single-data-input form was planned
        // at build time, so no host read of split sizes and no trailing
        // synchronization are needed. Copies are ordered on the stream, which is
        // exactly what makes this recordable inside a CUDA graph capture.
        if let Some(plan) = &self.static_plan {
            let runtime_split_input = inputs.get(1).is_some_and(|v| !v.is_absent());
            if !runtime_split_input
                && plan.axis < rank
                && inputs[0].shape[plan.axis] == plan.axis_extent
                && plan.sizes.len() == outputs.len()
            {
                return self.launch_copies(inputs, outputs, plan.axis, &plan.sizes);
            }
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if !(0..rank as i64).contains(&axis) {
            return Err(EpError::KernelFailed(
                "cuda_ep Split: axis out of range".into(),
            ));
        }
        let axis = axis as usize;
        let sizes_i64 = if inputs.get(1).is_some_and(|v| !v.is_absent()) {
            host_ints(&self.runtime, &inputs[1], "Split")?
        } else if let Some(split) = &self.split {
            split.clone()
        } else {
            let n = self
                .num_outputs
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(outputs.len());
            even_split(inputs[0].shape[axis], n)?
                .into_iter()
                .map(|v| v as i64)
                .collect()
        };
        let sizes = sizes_i64
            .into_iter()
            .map(|v| {
                usize::try_from(v)
                    .map_err(|_| EpError::KernelFailed("cuda_ep Split: negative split size".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if sizes.len() != outputs.len() || sizes.iter().sum::<usize>() != inputs[0].shape[axis] {
            return Err(EpError::KernelFailed(
                "cuda_ep Split: split sizes do not match outputs/axis".into(),
            ));
        }
        self.launch_copies(inputs, outputs, axis, &sizes)?;
        self.runtime.synchronize()
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.static_plan.is_some() {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Split reads runtime split sizes on the host and performs a trailing stream synchronization",
            )
        }
    }
}
