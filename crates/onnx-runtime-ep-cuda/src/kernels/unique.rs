//! Bounded, flattened CUDA `Unique` with a device-workspace metadata phase.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, KernelSizedOutputMetadata,
    KernelSizedOutputPolicy, Result, TensorMetadata, TensorMut, TensorView, WorkspaceLifetime,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{Attribute, DataType, Node, Shape, TensorLayout};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaDeviceCapabilities, CudaRuntime, cuptr};

pub const MAX_UNIQUE_ELEMENTS: usize = 1024;

const SOURCE: &str = r#"
__device__ __forceinline__ bool unique_equal(float a, float b) {
  return (isnan(a) && isnan(b)) || a == b;
}

__device__ __forceinline__ bool unique_before(
    float a, unsigned int ia, float b, unsigned int ib) {
  if (ia == 0xffffffffu) return false;
  if (ib == 0xffffffffu) return true;
  const bool a_nan = isnan(a);
  const bool b_nan = isnan(b);
  if (a_nan != b_nan) return !a_nan;
  if (!a_nan && a < b) return true;
  if (!a_nan && b < a) return false;
  return ia < ib;
}

extern "C" __global__ void unique_plan_f32(
    const float* input, unsigned int n, int sorted,
    unsigned int* first, unsigned int* counts, unsigned int* inverse,
    unsigned long long* unique_count) {
  extern __shared__ unsigned char raw[];
  float* keys = reinterpret_cast<float*>(raw);
  unsigned int* indices =
      reinterpret_cast<unsigned int*>(keys + blockDim.x);
  const unsigned int tid = threadIdx.x;
  if (tid < n) {
    keys[tid] = input[tid];
    indices[tid] = tid;
  } else {
    keys[tid] = 0.0f;
    indices[tid] = 0xffffffffu;
  }
  __syncthreads();

  for (unsigned int width = 2; width <= blockDim.x; width <<= 1) {
    for (unsigned int stride = width >> 1; stride > 0; stride >>= 1) {
      const unsigned int peer = tid ^ stride;
      if (peer > tid) {
        const bool ascending = (tid & width) == 0;
        const bool peer_before =
            unique_before(keys[peer], indices[peer], keys[tid], indices[tid]);
        const bool self_before =
            unique_before(keys[tid], indices[tid], keys[peer], indices[peer]);
        if ((ascending && peer_before) || (!ascending && self_before)) {
          const float key = keys[tid];
          keys[tid] = keys[peer];
          keys[peer] = key;
          const unsigned int index = indices[tid];
          indices[tid] = indices[peer];
          indices[peer] = index;
        }
      }
      __syncthreads();
    }
  }

  if (tid != 0) return;
  unsigned int groups = 0;
  for (unsigned int position = 0; position < n; ++position) {
    const unsigned int original = indices[position];
    if (position == 0 || !unique_equal(keys[position - 1], keys[position])) {
      first[groups] = original;
      counts[groups] = 0;
      ++groups;
    } else if (original < first[groups - 1]) {
      first[groups - 1] = original;
    }
    inverse[original] = groups - 1;
    ++counts[groups - 1];
  }

  if (!sorted) {
    unsigned int output_group = 0;
    // Reuse shared storage after sorting: keys holds temporary first indices
    // and indices holds temporary counts. counts[sorted_group] becomes the
    // sorted-to-first-appearance map after its original count is saved.
    unsigned int* temp_first = reinterpret_cast<unsigned int*>(keys);
    unsigned int* temp_counts = indices;
    for (unsigned int original = 0; original < n; ++original) {
      const unsigned int sorted_group = inverse[original];
      if (first[sorted_group] == original) {
        temp_first[output_group] = original;
        temp_counts[output_group] = counts[sorted_group];
        counts[sorted_group] = output_group;
        ++output_group;
      }
    }
    for (unsigned int original = 0; original < n; ++original)
      inverse[original] = counts[inverse[original]];
    for (unsigned int group = 0; group < groups; ++group) {
      first[group] = temp_first[group];
      counts[group] = temp_counts[group];
    }
  }
  *unique_count = groups;
}

extern "C" __global__ void unique_materialize_f32(
    const float* input, unsigned int n,
    const unsigned int* first, const unsigned int* counts,
    const unsigned int* inverse, const unsigned long long* unique_count_ptr,
    float* y, long long* indices,
    long long* inverse_indices, long long* output_counts) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int unique_count = (unsigned int)*unique_count_ptr;
  if (i < unique_count) {
    if (y) y[i] = input[first[i]];
    if (indices) indices[i] = (long long)first[i];
    if (output_counts) output_counts[i] = (long long)counts[i];
  }
  if (i < n && inverse_indices)
    inverse_indices[i] = (long long)inverse[i];
}
"#;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniqueExecutionStats {
    pub metadata_launches: u64,
    pub materialize_launches: u64,
    pub d2h_bytes: u64,
    pub full_input_d2h_bytes: u64,
    pub workspace_bytes: u64,
}

static METADATA_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static MATERIALIZE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static WORKSPACE_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn unique_execution_stats() -> UniqueExecutionStats {
    UniqueExecutionStats {
        metadata_launches: METADATA_LAUNCHES.load(Ordering::Relaxed),
        materialize_launches: MATERIALIZE_LAUNCHES.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        full_input_d2h_bytes: 0,
        workspace_bytes: WORKSPACE_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_unique_execution_stats() {
    METADATA_LAUNCHES.store(0, Ordering::Relaxed);
    MATERIALIZE_LAUNCHES.store(0, Ordering::Relaxed);
    D2H_BYTES.store(0, Ordering::Relaxed);
    WORKSPACE_BYTES.store(0, Ordering::Relaxed);
}

pub struct UniqueFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for UniqueFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if node.attr("axis").is_some() {
            return Err(not_implemented(
                "Unique with axis; the bounded CUDA slice supports flattened input only",
            ));
        }
        let sorted = match node.attr("sorted") {
            None => true,
            Some(Attribute::Int(0)) => false,
            Some(Attribute::Int(1)) => true,
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep Unique: attribute 'sorted' must be 0 or 1".into(),
                ));
            }
        };
        Ok(Box::new(UniqueKernel {
            runtime: self.runtime.clone(),
            sorted,
            max_elements: MAX_UNIQUE_ELEMENTS.min(
                usize::try_from(self.runtime.capabilities().max_threads_per_block())
                    .unwrap_or(MAX_UNIQUE_ELEMENTS),
            ),
        }))
    }
}

struct UniqueKernel {
    runtime: Arc<CudaRuntime>,
    sorted: bool,
    max_elements: usize,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceLayout {
    first: usize,
    counts: usize,
    inverse: usize,
    unique_count: usize,
    bytes: usize,
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: workspace alignment overflow".into()))
}

fn workspace_layout(elements: usize) -> Result<WorkspaceLayout> {
    if elements == 0 {
        return Ok(WorkspaceLayout {
            first: 0,
            counts: 0,
            inverse: 0,
            unique_count: 0,
            bytes: 0,
        });
    }
    let u32_bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: workspace size overflow".into()))?;
    let first = 0;
    let counts = u32_bytes;
    let inverse = counts
        .checked_add(u32_bytes)
        .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: workspace size overflow".into()))?;
    let unique_count = align_up(
        inverse.checked_add(u32_bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep Unique: workspace size overflow".into())
        })?,
        std::mem::align_of::<u64>(),
    )?;
    let bytes = unique_count
        .checked_add(std::mem::size_of::<u64>())
        .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: workspace size overflow".into()))?;
    Ok(WorkspaceLayout {
        first,
        counts,
        inverse,
        unique_count,
        bytes,
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: input shape overflow".into()))
    })
}

impl UniqueKernel {
    fn validate_input(&self, inputs: &[TensorView]) -> Result<usize> {
        if inputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Unique: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.dtype != DataType::Float32 {
            return Err(not_implemented(format!(
                "Unique dtype {:?}; the bounded CUDA slice supports Float32 only",
                input.dtype
            )));
        }
        if !input.is_contiguous() {
            return Err(not_implemented(
                "Unique with strided input; the bounded CUDA slice requires contiguous input",
            ));
        }
        let elements = checked_numel(input.shape)?;
        if elements > self.max_elements {
            return Err(not_implemented(format!(
                "Unique with {elements} flattened elements; this bounded CUDA algorithm supports \
                 at most {} elements on device {:?}",
                self.max_elements, input.device
            )));
        }
        Ok(elements)
    }

    fn workspace(
        &self,
        workspace: Option<WorkspaceView>,
        elements: usize,
    ) -> Result<(WorkspaceView, WorkspaceLayout)> {
        let layout = workspace_layout(elements)?;
        let workspace = workspace.ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Unique: governed workspace of {} bytes was not supplied",
                layout.bytes
            ))
        })?;
        if workspace.bytes() < layout.bytes || workspace.ptr().is_null() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Unique: governed workspace is {} bytes, need {}",
                workspace.bytes(),
                layout.bytes
            )));
        }
        Ok((workspace, layout))
    }

    fn prepare(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
        workspace: Option<WorkspaceView>,
    ) -> Result<Vec<Option<KernelSizedOutputMetadata>>> {
        if !(1..=4).contains(&requested_outputs.len()) || !requested_outputs[0] {
            return Err(EpError::KernelFailed(
                "cuda_ep Unique: expected 1..=4 output slots with required Y present".into(),
            ));
        }
        let elements = self.validate_input(inputs)?;
        if elements == 0 {
            return Ok(unique_metadata(elements, 0, requested_outputs));
        }
        let (workspace, layout) = self.workspace(workspace, elements)?;
        let base = cuptr(workspace.ptr().as_ptr::<u8>().cast::<c_void>());
        let first = base + layout.first as u64;
        let counts = base + layout.counts as u64;
        let inverse = base + layout.inverse as u64;
        let unique_count = base + layout.unique_count as u64;
        let capacity = elements.next_power_of_two();
        let function = self
            .runtime
            .nvrtc_function("unique_f32_v1", SOURCE, "unique_plan_f32")?;
        if capacity > self.max_elements {
            return Err(not_implemented(format!(
                "Unique launch needs {capacity} threads, device limit is {}",
                self.max_elements
            )));
        }
        let input = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let n = u32::try_from(elements)
            .map_err(|_| EpError::KernelFailed("cuda_ep Unique: input is too large".into()))?;
        let sorted = i32::from(self.sorted);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input)
            .arg(&n)
            .arg(&sorted)
            .arg(&first)
            .arg(&counts)
            .arg(&inverse)
            .arg(&unique_count);
        let launch = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (capacity as u32, 1, 1),
            shared_mem_bytes: u32::try_from(capacity * 8).map_err(|_| {
                EpError::KernelFailed("cuda_ep Unique: shared-memory size overflow".into())
            })?,
        };
        unsafe { builder.launch(launch) }
            .map_err(|error| driver_err("launch Unique metadata phase", error))?;
        METADATA_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        WORKSPACE_BYTES.store(layout.bytes as u64, Ordering::Relaxed);

        let mut count_bytes = [0u8; std::mem::size_of::<u64>()];
        unsafe { self.runtime.dtoh(&mut count_bytes, unique_count)? };
        D2H_BYTES.fetch_add(count_bytes.len() as u64, Ordering::Relaxed);
        let unique_count = usize::try_from(u64::from_ne_bytes(count_bytes)).map_err(|_| {
            EpError::KernelFailed("cuda_ep Unique: unique count exceeds usize".into())
        })?;
        if unique_count > elements {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Unique: device returned invalid count {unique_count} for {elements} inputs"
            )));
        }
        Ok(unique_metadata(elements, unique_count, requested_outputs))
    }

    fn materialize(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let elements = self.validate_input(inputs)?;
        if outputs.is_empty() || outputs.len() > 4 || outputs[0].is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep Unique: invalid positional outputs".into(),
            ));
        }
        if elements == 0 {
            return Ok(());
        }
        let (workspace, layout) = self.workspace(workspace, elements)?;
        let base = cuptr(workspace.ptr().as_ptr::<u8>().cast::<c_void>());
        let first = base + layout.first as u64;
        let counts = base + layout.counts as u64;
        let inverse = base + layout.inverse as u64;
        let unique_count_ptr = base + layout.unique_count as u64;
        let unique_count = outputs[0].shape.first().copied().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep Unique: Y output must have rank 1".into())
        })?;
        validate_outputs(outputs, inputs[0].dtype, elements, unique_count)?;

        let mut ptr = |slot: usize| -> u64 {
            outputs
                .get_mut(slot)
                .filter(|output| !output.is_absent())
                .map_or(0, |output| {
                    cuptr(output.data_ptr_mut::<u8>() as *const c_void)
                })
        };
        let input = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let y = ptr(0);
        let indices = ptr(1);
        let inverse_indices = ptr(2);
        let output_counts = ptr(3);
        let n = elements as u32;
        let function =
            self.runtime
                .nvrtc_function("unique_f32_v1", SOURCE, "unique_materialize_f32")?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input)
            .arg(&n)
            .arg(&first)
            .arg(&counts)
            .arg(&inverse)
            .arg(&unique_count_ptr)
            .arg(&y)
            .arg(&indices)
            .arg(&inverse_indices)
            .arg(&output_counts);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(
                u32::try_from(elements).map_err(|_| {
                    EpError::KernelFailed("cuda_ep Unique: launch size exceeds u32".into())
                })?,
            ))
        }
        .map_err(|error| driver_err("launch Unique materialization phase", error))?;
        MATERIALIZE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        self.runtime.synchronize()
    }
}

fn unique_metadata(
    elements: usize,
    unique_count: usize,
    requested_outputs: &[bool],
) -> Vec<Option<KernelSizedOutputMetadata>> {
    requested_outputs
        .iter()
        .enumerate()
        .map(|(slot, &requested)| {
            requested.then(|| KernelSizedOutputMetadata {
                shape: if slot == 2 {
                    vec![elements]
                } else {
                    vec![unique_count]
                },
                dtype: if slot == 0 {
                    DataType::Float32
                } else {
                    DataType::Int64
                },
            })
        })
        .collect()
}

fn validate_outputs(
    outputs: &[TensorMut],
    input_dtype: DataType,
    elements: usize,
    unique_count: usize,
) -> Result<()> {
    for (slot, output) in outputs.iter().enumerate() {
        if output.is_absent() {
            continue;
        }
        let (dtype, shape) = if slot == 0 {
            (input_dtype, vec![unique_count])
        } else if slot == 2 {
            (DataType::Int64, vec![elements])
        } else {
            (DataType::Int64, vec![unique_count])
        };
        if output.dtype != dtype || output.shape != shape || !output.is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Unique: output slot {slot} must be contiguous {dtype:?}{shape:?}, got \
                 {:?}{:?}",
                output.dtype, output.shape
            )));
        }
    }
    Ok(())
}

impl Kernel for UniqueKernel {
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
        Err(EpError::KernelFailed(
            "cuda_ep Unique requires governed workspace; call execute_with_workspace".into(),
        ))
    }

    fn has_kernel_sized_outputs(&self) -> bool {
        true
    }

    fn kernel_sized_output_policy(&self) -> KernelSizedOutputPolicy {
        KernelSizedOutputPolicy::DeviceWorkspace
    }

    fn prepare_kernel_sized_device(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
        workspace: Option<WorkspaceView>,
    ) -> Result<Vec<Option<KernelSizedOutputMetadata>>> {
        self.prepare(inputs, requested_outputs, workspace)
    }

    fn materialize_kernel_sized_device(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.materialize(inputs, outputs, workspace)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        let elements = inputs
            .first()
            .filter(|input| input.present)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep Unique: missing input metadata".into()))
            .and_then(|input| checked_numel(input.shape))?;
        let layout = workspace_layout(elements)?;
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.bytes).map_err(|_| {
                EpError::KernelFailed("cuda_ep Unique: workspace exceeds u64".into())
            })?,
            alignment: std::mem::align_of::<u64>(),
            lifetime: WorkspaceLifetime::StepScoped,
            role: MemoryRole::Workspace { step_scoped: true },
        })
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let requested: Vec<bool> = outputs.iter().map(|output| !output.is_absent()).collect();
        let metadata = self.prepare(inputs, &requested, workspace)?;
        for (slot, (metadata, output)) in metadata.iter().zip(outputs.iter()).enumerate() {
            if let Some(metadata) = metadata
                && (output.dtype != metadata.dtype || output.shape != metadata.shape)
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Unique: native output slot {slot} does not match device metadata"
                )));
            }
        }
        self.materialize(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "Unique uses the DeviceWorkspace two-phase path: an 8-byte count D2H synchronization \
             must precede dynamic ORT output allocation",
        )
    }
}

pub(crate) fn unsupported_reason(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
    input_layouts: &[TensorLayout],
    capabilities: CudaDeviceCapabilities,
) -> Option<String> {
    let reject = || -> std::result::Result<(), String> {
        if node.inputs.len() != 1 || !(1..=4).contains(&node.outputs.len()) {
            return Err("requires one input and 1..=4 positional outputs".into());
        }
        if node.attr("axis").is_some() {
            return Err("axis mode is not in the bounded flattened CUDA slice".into());
        }
        match node.attr("sorted").and_then(Attribute::as_int) {
            None | Some(0 | 1) => {}
            Some(value) => return Err(format!("attribute 'sorted' must be 0 or 1, got {value}")),
        }
        if input_dtypes.first() != Some(&DataType::Float32) {
            return Err(format!(
                "input dtype must be Float32, got {:?}",
                input_dtypes.first().copied().unwrap_or(DataType::Undefined)
            ));
        }
        let shape = input_shapes
            .first()
            .ok_or_else(|| "missing input shape metadata".to_string())?;
        let static_shape = onnx_runtime_ir::as_static_shape(shape)
            .ok_or_else(|| "input shape must be static for the bounded CUDA claim".to_string())?;
        let elements = static_shape.iter().try_fold(1usize, |product, &extent| {
            product
                .checked_mul(extent)
                .ok_or_else(|| "input element count overflows usize".to_string())
        })?;
        let limit = MAX_UNIQUE_ELEMENTS.min(
            usize::try_from(capabilities.max_threads_per_block()).unwrap_or(MAX_UNIQUE_ELEMENTS),
        );
        if elements > limit {
            return Err(format!(
                "flattened input has {elements} elements; bounded CUDA Unique limit is {limit}"
            ));
        }
        if input_layouts
            .first()
            .is_some_and(|layout| !layout.is_contiguous(&static_shape))
        {
            return Err("input layout must be contiguous".into());
        }
        Ok(())
    };
    reject().err().map(|reason| format!("Unique: {reason}"))
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::DeviceId;

    use super::*;

    #[test]
    fn workspace_layout_is_checked_and_compact() {
        let layout = workspace_layout(6).unwrap();
        assert_eq!(layout.first, 0);
        assert_eq!(layout.counts, 24);
        assert_eq!(layout.inverse, 48);
        assert_eq!(layout.unique_count, 72);
        assert_eq!(layout.bytes, 80);
        assert_eq!(workspace_layout(0).unwrap().bytes, 0);
    }

    #[test]
    fn claim_declines_axis_dtype_dynamic_large_and_strided_inputs() {
        let capabilities = CudaDeviceCapabilities::for_test((8, 9), 24, 24 * 1024 * 1024);
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Unique",
            vec![Some(onnx_runtime_ir::ValueId(0))],
            vec![onnx_runtime_ir::ValueId(1)],
        );
        let shape = onnx_runtime_ir::static_shape([8]);
        assert!(
            unsupported_reason(
                &node,
                std::slice::from_ref(&shape),
                &[DataType::Float32],
                &[TensorLayout::contiguous()],
                capabilities,
            )
            .is_none()
        );
        node.attributes.insert("axis".into(), Attribute::Int(0));
        assert!(
            unsupported_reason(
                &node,
                std::slice::from_ref(&shape),
                &[DataType::Float32],
                &[TensorLayout::contiguous()],
                capabilities,
            )
            .unwrap()
            .contains("axis mode")
        );

        node.attributes.clear();
        assert!(
            unsupported_reason(
                &node,
                std::slice::from_ref(&shape),
                &[DataType::Float16],
                &[TensorLayout::contiguous()],
                capabilities,
            )
            .unwrap()
            .contains("Float32")
        );
        let dynamic = vec![onnx_runtime_ir::Dim::Symbolic(onnx_runtime_ir::SymbolId(7))];
        assert!(
            unsupported_reason(
                &node,
                &[dynamic],
                &[DataType::Float32],
                &[TensorLayout::contiguous()],
                capabilities,
            )
            .unwrap()
            .contains("static")
        );
        let too_large = onnx_runtime_ir::static_shape([MAX_UNIQUE_ELEMENTS + 1]);
        assert!(
            unsupported_reason(
                &node,
                &[too_large],
                &[DataType::Float32],
                &[TensorLayout::contiguous()],
                capabilities,
            )
            .unwrap()
            .contains("limit")
        );
        assert!(
            unsupported_reason(
                &node,
                std::slice::from_ref(&shape),
                &[DataType::Float32],
                &[TensorLayout::strided(vec![2])],
                capabilities,
            )
            .unwrap()
            .contains("contiguous")
        );
    }

    struct DeviceOutput {
        ptr: u64,
        shape: Vec<usize>,
        strides: Vec<i64>,
        dtype: DataType,
        present: bool,
    }

    impl DeviceOutput {
        fn view_mut(&mut self) -> TensorMut<'_> {
            let view = TensorMut::new(
                DevicePtrMut(self.ptr as usize as *mut c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cuda(0),
            );
            if self.present {
                view
            } else {
                view.mark_absent()
            }
        }
    }

    fn bytes_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn read_i64(bytes: &[u8]) -> Vec<i64> {
        bytes
            .chunks_exact(8)
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    struct GpuCaseResult {
        outputs: Vec<Option<Vec<u8>>>,
        shapes: Vec<Option<Vec<usize>>>,
        stats: UniqueExecutionStats,
    }

    fn run_gpu_case(
        runtime: Arc<CudaRuntime>,
        values: &[f32],
        sorted: bool,
        requested: &[bool],
    ) -> GpuCaseResult {
        reset_unique_execution_stats();
        let input_bytes = bytes_f32(values);
        let input_ptr = runtime.alloc_raw(input_bytes.len().max(1)).unwrap();
        if !input_bytes.is_empty() {
            unsafe { runtime.htod(&input_bytes, input_ptr).unwrap() };
        }
        let input_shape = vec![values.len()];
        let input_strides = vec![1];
        let input = TensorView::new(
            DevicePtr(input_ptr as usize as *const c_void),
            DataType::Float32,
            &input_shape,
            &input_strides,
            DeviceId::cuda(0),
        );
        let kernel = UniqueKernel {
            runtime: runtime.clone(),
            sorted,
            max_elements: MAX_UNIQUE_ELEMENTS
                .min(usize::try_from(runtime.capabilities().max_threads_per_block()).unwrap()),
        };
        let layout = workspace_layout(values.len()).unwrap();
        let workspace_ptr = if layout.bytes == 0 {
            0
        } else {
            runtime.alloc_raw(layout.bytes).unwrap()
        };
        let workspace = (layout.bytes != 0).then(|| {
            WorkspaceView::new(
                DevicePtrMut(workspace_ptr as usize as *mut c_void),
                layout.bytes,
            )
        });
        let transfers_before = runtime.transfer_counts();
        let metadata = kernel
            .prepare_kernel_sized_device(&[input], requested, workspace)
            .unwrap();
        let mut device_outputs: Vec<DeviceOutput> = metadata
            .iter()
            .map(|metadata| match metadata {
                Some(metadata) => {
                    let elements = metadata.shape.iter().product::<usize>();
                    DeviceOutput {
                        ptr: runtime
                            .alloc_raw((elements * metadata.dtype.byte_size()).max(1))
                            .unwrap(),
                        shape: metadata.shape.clone(),
                        strides: onnx_runtime_ir::compute_contiguous_strides(&metadata.shape),
                        dtype: metadata.dtype,
                        present: true,
                    }
                }
                None => DeviceOutput {
                    ptr: 0,
                    shape: Vec::new(),
                    strides: Vec::new(),
                    dtype: DataType::Undefined,
                    present: false,
                },
            })
            .collect();
        let mut output_views: Vec<_> = device_outputs
            .iter_mut()
            .map(DeviceOutput::view_mut)
            .collect();
        kernel
            .materialize_kernel_sized_device(&[input], &mut output_views, workspace)
            .unwrap();
        runtime.synchronize().unwrap();
        let transfers_after_algorithm = runtime.transfer_counts();
        assert_eq!(
            transfers_after_algorithm.device_to_host - transfers_before.device_to_host,
            if values.is_empty() { 0 } else { 1 },
            "CUDA Unique may copy only the compact count during its algorithm"
        );

        let mut host_outputs = Vec::with_capacity(device_outputs.len());
        let mut shapes = Vec::with_capacity(device_outputs.len());
        for output in &device_outputs {
            if !output.present {
                host_outputs.push(None);
                shapes.push(None);
                continue;
            }
            let mut bytes =
                vec![0u8; output.shape.iter().product::<usize>() * output.dtype.byte_size()];
            if !bytes.is_empty() {
                unsafe { runtime.dtoh(&mut bytes, output.ptr).unwrap() };
            }
            host_outputs.push(Some(bytes));
            shapes.push(Some(output.shape.clone()));
        }
        let stats = unique_execution_stats();
        for output in device_outputs {
            if output.ptr != 0 {
                unsafe { runtime.free_raw(output.ptr).unwrap() };
            }
        }
        if workspace_ptr != 0 {
            unsafe { runtime.free_raw(workspace_ptr).unwrap() };
        }
        unsafe { runtime.free_raw(input_ptr).unwrap() };
        GpuCaseResult {
            outputs: host_outputs,
            shapes,
            stats,
        }
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn gpu_flattened_unique_matches_cpu_semantics_and_transfer_contract() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CUDA Unique test: CUDA runtime unavailable");
            return;
        };
        let registry = crate::kernels::build_cuda_registry(runtime.clone());
        let factory = registry
            .lookup("Unique", "", 11)
            .expect("CUDA Unique must be present in the real factory registry");
        let node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Unique",
            vec![Some(onnx_runtime_ir::ValueId(0))],
            vec![onnx_runtime_ir::ValueId(1)],
        );
        let registered_kernel = factory.create(&node, &[vec![6]]).unwrap();
        let capture = registered_kernel.capture_support();
        assert!(!capture.is_supported());
        assert!(capture.reason().unwrap().contains("8-byte count D2H"));
        reset_unique_execution_stats();
        let capture_error = runtime
            .begin_graph_capture(&[registered_kernel.as_ref()])
            .expect_err("Unique capture must be rejected before either DeviceWorkspace phase");
        assert!(
            capture_error
                .to_string()
                .contains("rejected before begin_capture")
        );
        assert!(capture_error.to_string().contains("DeviceWorkspace"));
        assert_eq!(unique_execution_stats(), UniqueExecutionStats::default());
        assert!(!runtime.is_capturing().unwrap());
        assert!(crate::kernels::CUDA_COVERED_OPS.contains(&"Unique"));
        assert_eq!(
            crate::kernels::cuda_supported_dtypes_for_op("Unique", ""),
            &[DataType::Float32, DataType::Int64]
        );

        let GpuCaseResult {
            outputs,
            shapes,
            stats,
        } = run_gpu_case(
            runtime.clone(),
            &[2., 1., 1., 3., 4., 3.],
            false,
            &[true, true, true, true],
        );
        assert_eq!(
            shapes,
            [Some(vec![4]), Some(vec![4]), Some(vec![6]), Some(vec![4])]
        );
        assert_eq!(read_f32(outputs[0].as_ref().unwrap()), [2., 1., 3., 4.]);
        assert_eq!(read_i64(outputs[1].as_ref().unwrap()), [0, 1, 3, 4]);
        assert_eq!(read_i64(outputs[2].as_ref().unwrap()), [0, 1, 1, 2, 3, 2]);
        assert_eq!(read_i64(outputs[3].as_ref().unwrap()), [1, 2, 2, 1]);
        assert_eq!(stats.metadata_launches, 1);
        assert_eq!(stats.materialize_launches, 1);
        assert_eq!(stats.d2h_bytes, 8);
        assert_eq!(stats.full_input_d2h_bytes, 0);
        assert_eq!(
            stats.workspace_bytes,
            workspace_layout(6).unwrap().bytes as u64
        );

        let GpuCaseResult { outputs, stats, .. } = run_gpu_case(
            runtime.clone(),
            &[3., 1., 3., 2., 1.],
            true,
            &[true, false, true, false],
        );
        assert_eq!(read_f32(outputs[0].as_ref().unwrap()), [1., 2., 3.]);
        assert!(outputs[1].is_none());
        assert_eq!(read_i64(outputs[2].as_ref().unwrap()), [2, 0, 2, 1, 0]);
        assert!(outputs[3].is_none());
        assert_eq!(stats.metadata_launches, 1);
        assert_eq!(stats.materialize_launches, 1);
        assert_eq!(stats.d2h_bytes, 8);

        for values in [&[][..], &[7., 7., 7.][..], &[3., 1., 2.][..]] {
            let GpuCaseResult { outputs, stats, .. } =
                run_gpu_case(runtime.clone(), values, true, &[true]);
            let expected: &[f32] = if values.is_empty() {
                &[]
            } else if values[0] == 7. {
                &[7.]
            } else {
                &[1., 2., 3.]
            };
            assert_eq!(read_f32(outputs[0].as_ref().unwrap()), expected);
            assert_eq!(stats.full_input_d2h_bytes, 0);
        }

        let first_nan = f32::from_bits(0x7fc0_0001);
        let second_nan = f32::from_bits(0x7fc0_1234);
        let GpuCaseResult { outputs, .. } = run_gpu_case(
            runtime,
            &[first_nan, second_nan, -0.0, 0.0],
            true,
            &[true, true, true, true],
        );
        let y = read_f32(outputs[0].as_ref().unwrap());
        assert_eq!(y[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(y[1].to_bits(), first_nan.to_bits());
        assert_eq!(read_i64(outputs[1].as_ref().unwrap()), [2, 0]);
        assert_eq!(read_i64(outputs[2].as_ref().unwrap()), [1, 1, 0, 0]);
        assert_eq!(read_i64(outputs[3].as_ref().unwrap()), [2, 2]);
    }

    #[test]
    #[ignore = "CUDA phase accounting probe; run explicitly on an idle GPU"]
    fn gpu_unique_phase_accounting() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CUDA Unique accounting: CUDA runtime unavailable");
            return;
        };
        let warm_values: Vec<f32> = (0..64).map(|index| (index % 17) as f32).collect();
        let _ = run_gpu_case(
            runtime.clone(),
            &warm_values,
            true,
            &[true, true, true, true],
        );
        println!(
            "# n,total_host_us,metadata_launches,materialize_launches,input_h2d_bytes,count_d2h_bytes,workspace_bytes"
        );
        for elements in [64usize, 256, 1024] {
            if elements > runtime.capabilities().max_threads_per_block() as usize {
                continue;
            }
            let values: Vec<f32> = (0..elements)
                .map(|index| ((index * 37) % (elements / 2 + 1)) as f32)
                .collect();
            let start = std::time::Instant::now();
            let GpuCaseResult { stats, .. } =
                run_gpu_case(runtime.clone(), &values, true, &[true, true, true, true]);
            println!(
                "# {elements},{},{},{},{},{},{}",
                start.elapsed().as_micros(),
                stats.metadata_launches,
                stats.materialize_launches,
                elements * std::mem::size_of::<f32>(),
                stats.d2h_bytes,
                stats.workspace_bytes,
            );
        }
    }
}
