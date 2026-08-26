//! Bounded CUDA `NonMaxSuppression` with device-workspace-sized output.

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
use crate::runtime::{CudaRuntime, cuptr};

pub const MAX_NMS_BOXES: usize = 256;
pub const MAX_NMS_GROUPS: usize = 256;

const SOURCE: &str = r#"
__device__ __forceinline__ bool nms_before(
    float a, unsigned int ia, float b, unsigned int ib) {
  if (ia == 0xffffffffu) return false;
  if (ib == 0xffffffffu) return true;
  int ka = __float_as_int(a);
  int kb = __float_as_int(b);
  ka ^= (int)(((unsigned int)(ka >> 31)) >> 1);
  kb ^= (int)(((unsigned int)(kb >> 31)) >> 1);
  if (ka != kb) return ka > kb;
  return ia < ib;
}

__device__ __forceinline__ float4 nms_box(
    const float* values, int center_point_box) {
  if (center_point_box) {
    const float xc = values[0], yc = values[1];
    const float w = values[2], h = values[3];
    return make_float4(yc - h * 0.5f, xc - w * 0.5f,
                       yc + h * 0.5f, xc + w * 0.5f);
  }
  return make_float4(fminf(values[0], values[2]),
                     fminf(values[1], values[3]),
                     fmaxf(values[0], values[2]),
                     fmaxf(values[1], values[3]));
}

__device__ __forceinline__ float nms_iou(float4 a, float4 b) {
  const float ih = fmaxf(fminf(a.z, b.z) - fmaxf(a.x, b.x), 0.0f);
  const float iw = fmaxf(fminf(a.w, b.w) - fmaxf(a.y, b.y), 0.0f);
  const float intersection = ih * iw;
  const float aa = fmaxf(a.z - a.x, 0.0f) * fmaxf(a.w - a.y, 0.0f);
  const float ba = fmaxf(b.z - b.x, 0.0f) * fmaxf(b.w - b.y, 0.0f);
  const float u = aa + ba - intersection;
  return u > 0.0f ? intersection / u : 0.0f;
}

extern "C" __global__ void nms_prepare_f32(
    const float* boxes, const float* scores,
    const long long* max_output_ptr, const float* iou_threshold_ptr,
    const float* score_threshold_ptr, unsigned int boxes_count,
    unsigned int classes, int center_point_box,
    unsigned int* selected, unsigned int* selected_counts) {
  extern __shared__ unsigned char raw[];
  float* ordered_scores = reinterpret_cast<float*>(raw);
  unsigned int* ordered_indices =
      reinterpret_cast<unsigned int*>(ordered_scores + blockDim.x);
  const unsigned int tid = threadIdx.x;
  const unsigned int group = blockIdx.x;
  const unsigned int batch = group / classes;
  const unsigned int cls = group % classes;
  const float threshold =
      score_threshold_ptr ? *score_threshold_ptr
                          : __int_as_float((int)0xff800000u);
  const unsigned int score_offset =
      (batch * classes + cls) * boxes_count;
  if (tid < boxes_count && scores[score_offset + tid] > threshold) {
    ordered_scores[tid] = scores[score_offset + tid];
    ordered_indices[tid] = tid;
  } else {
    ordered_scores[tid] = 0.0f;
    ordered_indices[tid] = 0xffffffffu;
  }
  __syncthreads();

  for (unsigned int width = 2; width <= blockDim.x; width <<= 1) {
    for (unsigned int stride = width >> 1; stride > 0; stride >>= 1) {
      const unsigned int peer = tid ^ stride;
      if (peer > tid) {
        const bool ascending = (tid & width) == 0;
        const bool peer_before = nms_before(
            ordered_scores[peer], ordered_indices[peer],
            ordered_scores[tid], ordered_indices[tid]);
        const bool self_before = nms_before(
            ordered_scores[tid], ordered_indices[tid],
            ordered_scores[peer], ordered_indices[peer]);
        if ((ascending && peer_before) || (!ascending && self_before)) {
          const float score = ordered_scores[tid];
          ordered_scores[tid] = ordered_scores[peer];
          ordered_scores[peer] = score;
          const unsigned int index = ordered_indices[tid];
          ordered_indices[tid] = ordered_indices[peer];
          ordered_indices[peer] = index;
        }
      }
      __syncthreads();
    }
  }

  if (tid != 0) return;
  long long requested = max_output_ptr ? *max_output_ptr : 0;
  if (requested < 0) {
    selected_counts[group] = 0xffffffffu;
    return;
  }
  const unsigned int limit =
      min((unsigned long long)boxes_count, (unsigned long long)requested);
  const float iou_threshold = iou_threshold_ptr ? *iou_threshold_ptr : 0.0f;
  unsigned int kept = 0;
  for (unsigned int position = 0; position < boxes_count && kept < limit; ++position) {
    const unsigned int candidate = ordered_indices[position];
    if (candidate == 0xffffffffu) break;
    const float4 candidate_box = nms_box(
        boxes + (batch * boxes_count + candidate) * 4, center_point_box);
    bool retain = true;
    for (unsigned int prior = 0; prior < kept; ++prior) {
      const unsigned int prior_index = selected[group * boxes_count + prior];
      const float4 prior_box = nms_box(
          boxes + (batch * boxes_count + prior_index) * 4, center_point_box);
      if (nms_iou(candidate_box, prior_box) > iou_threshold) {
        retain = false;
        break;
      }
    }
    if (retain) selected[group * boxes_count + kept++] = candidate;
  }
  selected_counts[group] = kept;
}

extern "C" __global__ void nms_count(
    const unsigned int* selected_counts, unsigned int groups,
    unsigned long long* selected_count) {
  if (blockIdx.x || threadIdx.x) return;
  unsigned long long total = 0;
  for (unsigned int group = 0; group < groups; ++group) {
    if (selected_counts[group] == 0xffffffffu) {
      *selected_count = 0xffffffffffffffffull;
      return;
    }
    total += selected_counts[group];
  }
  *selected_count = total;
}

extern "C" __global__ void nms_materialize(
    const unsigned int* selected, const unsigned int* selected_counts,
    unsigned int boxes_count, unsigned int classes, unsigned int groups,
    long long* output) {
  const unsigned int group = blockIdx.x;
  const unsigned int rank = threadIdx.x;
  if (group >= groups || rank >= selected_counts[group]) return;
  unsigned long long offset = 0;
  for (unsigned int prior = 0; prior < group; ++prior)
    offset += selected_counts[prior];
  const unsigned long long row = offset + rank;
  output[row * 3] = (long long)(group / classes);
  output[row * 3 + 1] = (long long)(group % classes);
  output[row * 3 + 2] =
      (long long)selected[group * boxes_count + rank];
}
"#;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NmsExecutionStats {
    pub prepare_launches: u64,
    pub count_launches: u64,
    pub materialize_launches: u64,
    pub d2h_bytes: u64,
    pub full_input_d2h_bytes: u64,
    pub workspace_bytes: u64,
}

static PREPARE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static COUNT_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static MATERIALIZE_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static WORKSPACE_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn nms_execution_stats() -> NmsExecutionStats {
    NmsExecutionStats {
        prepare_launches: PREPARE_LAUNCHES.load(Ordering::Relaxed),
        count_launches: COUNT_LAUNCHES.load(Ordering::Relaxed),
        materialize_launches: MATERIALIZE_LAUNCHES.load(Ordering::Relaxed),
        d2h_bytes: D2H_BYTES.load(Ordering::Relaxed),
        full_input_d2h_bytes: 0,
        workspace_bytes: WORKSPACE_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_nms_execution_stats() {
    PREPARE_LAUNCHES.store(0, Ordering::Relaxed);
    COUNT_LAUNCHES.store(0, Ordering::Relaxed);
    MATERIALIZE_LAUNCHES.store(0, Ordering::Relaxed);
    D2H_BYTES.store(0, Ordering::Relaxed);
    WORKSPACE_BYTES.store(0, Ordering::Relaxed);
}

pub struct NonMaxSuppressionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for NonMaxSuppressionFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let center_point_box = node
            .attr("center_point_box")
            .and_then(Attribute::as_int)
            .unwrap_or(0);
        if !matches!(center_point_box, 0 | 1) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep NonMaxSuppression: center_point_box must be 0 or 1, got {center_point_box}"
            )));
        }
        Ok(Box::new(NonMaxSuppressionKernel {
            runtime: self.runtime.clone(),
            center_point_box: center_point_box as i32,
        }))
    }
}

struct NonMaxSuppressionKernel {
    runtime: Arc<CudaRuntime>,
    center_point_box: i32,
}

#[derive(Clone, Copy)]
struct Geometry {
    boxes: usize,
    classes: usize,
    groups: usize,
}

#[derive(Clone, Copy)]
struct WorkspaceLayout {
    selected: usize,
    counts: usize,
    total_count: usize,
    bytes: usize,
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: workspace alignment overflow".into()))
}

fn workspace_layout(geometry: Geometry) -> Result<WorkspaceLayout> {
    if geometry.groups == 0 || geometry.boxes == 0 {
        return Ok(WorkspaceLayout {
            selected: 0,
            counts: 0,
            total_count: 0,
            bytes: 0,
        });
    }
    let selected_bytes = geometry
        .groups
        .checked_mul(geometry.boxes)
        .and_then(|items| items.checked_mul(4))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: workspace size overflow".into()))?;
    let counts = selected_bytes;
    let total_count = align_up(
        counts
            .checked_add(geometry.groups * 4)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: workspace size overflow".into()))?,
        8,
    )?;
    Ok(WorkspaceLayout {
        selected: 0,
        counts,
        total_count,
        bytes: total_count + 8,
    })
}

impl NonMaxSuppressionKernel {
    fn geometry(&self, inputs: &[TensorView]) -> Result<Geometry> {
        validate_views(inputs)?;
        let batches = inputs[0].shape[0];
        let boxes = inputs[0].shape[1];
        let classes = inputs[1].shape[1];
        let groups = batches
            .checked_mul(classes)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: group count overflow".into()))?;
        if boxes > MAX_NMS_BOXES || groups > MAX_NMS_GROUPS {
            return Err(not_implemented(format!(
                "NonMaxSuppression geometry boxes={boxes}, batch×classes={groups}; bounded CUDA \
                 limits are {MAX_NMS_BOXES} boxes and {MAX_NMS_GROUPS} groups"
            )));
        }
        Ok(Geometry {
            boxes,
            classes,
            groups,
        })
    }

    fn workspace(
        &self,
        workspace: Option<WorkspaceView>,
        geometry: Geometry,
    ) -> Result<(WorkspaceView, WorkspaceLayout)> {
        let layout = workspace_layout(geometry)?;
        let workspace = workspace.ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep NMS: governed workspace of {} bytes was not supplied",
                layout.bytes
            ))
        })?;
        if workspace.bytes() < layout.bytes || workspace.ptr().is_null() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep NMS: governed workspace is {} bytes, need {}",
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
        if requested_outputs != [true] {
            return Err(EpError::KernelFailed(
                "cuda_ep NMS: required selected_indices output must be present".into(),
            ));
        }
        let geometry = self.geometry(inputs)?;
        if geometry.groups == 0 || geometry.boxes == 0 {
            return Ok(vec![Some(KernelSizedOutputMetadata {
                shape: vec![0, 3],
                dtype: DataType::Int64,
            })]);
        }
        let (workspace, layout) = self.workspace(workspace, geometry)?;
        let base = cuptr(workspace.ptr().as_ptr::<u8>().cast::<c_void>());
        let selected = base + layout.selected as u64;
        let counts = base + layout.counts as u64;
        let total_count = base + layout.total_count as u64;
        let boxes_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let scores_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let scalar = |slot: usize| {
            inputs
                .get(slot)
                .filter(|input| !input.is_absent())
                .map_or(0, |input| cuptr(input.data_ptr::<u8>() as *const c_void))
        };
        let max_output = scalar(2);
        let iou_threshold = scalar(3);
        let score_threshold = scalar(4);
        let boxes = geometry.boxes as u32;
        let classes = geometry.classes as u32;
        let capacity = geometry.boxes.next_power_of_two().max(1);
        let center = self.center_point_box;
        let function = self
            .runtime
            .nvrtc_function("nms_f32_v1", SOURCE, "nms_prepare_f32")?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&boxes_ptr)
            .arg(&scores_ptr)
            .arg(&max_output)
            .arg(&iou_threshold)
            .arg(&score_threshold)
            .arg(&boxes)
            .arg(&classes)
            .arg(&center)
            .arg(&selected)
            .arg(&counts);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (geometry.groups as u32, 1, 1),
                block_dim: (capacity as u32, 1, 1),
                shared_mem_bytes: (capacity * 8) as u32,
            })
        }
        .map_err(|error| driver_err("launch NMS prepare phase", error))?;
        PREPARE_LAUNCHES.fetch_add(1, Ordering::Relaxed);

        let count_function = self
            .runtime
            .nvrtc_function("nms_f32_v1", SOURCE, "nms_count")?;
        let groups = geometry.groups as u32;
        let mut builder = self.runtime.stream().launch_builder(&count_function);
        builder.arg(&counts).arg(&groups).arg(&total_count);
        unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }
            .map_err(|error| driver_err("launch NMS count phase", error))?;
        COUNT_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        WORKSPACE_BYTES.store(layout.bytes as u64, Ordering::Relaxed);

        let mut count_bytes = [0u8; 8];
        unsafe { self.runtime.dtoh(&mut count_bytes, total_count)? };
        D2H_BYTES.fetch_add(8, Ordering::Relaxed);
        let raw_count = u64::from_ne_bytes(count_bytes);
        if raw_count == u64::MAX {
            return Err(EpError::KernelFailed(
                "cuda_ep NMS: max_output_boxes_per_class must be non-negative".into(),
            ));
        }
        let count = usize::try_from(raw_count)
            .map_err(|_| EpError::KernelFailed("cuda_ep NMS: count exceeds usize".into()))?;
        if count > geometry.groups * geometry.boxes {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep NMS: device returned invalid count {count}"
            )));
        }
        Ok(vec![Some(KernelSizedOutputMetadata {
            shape: vec![count, 3],
            dtype: DataType::Int64,
        })])
    }

    fn materialize(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let geometry = self.geometry(inputs)?;
        if outputs.len() != 1
            || outputs[0].is_absent()
            || outputs[0].dtype != DataType::Int64
            || outputs[0].shape.len() != 2
            || outputs[0].shape[1] != 3
            || !outputs[0].is_contiguous()
        {
            return Err(EpError::KernelFailed(
                "cuda_ep NMS: selected_indices must be contiguous Int64 [selected,3]".into(),
            ));
        }
        if geometry.groups == 0 || geometry.boxes == 0 || outputs[0].shape[0] == 0 {
            return Ok(());
        }
        let (workspace, layout) = self.workspace(workspace, geometry)?;
        let base = cuptr(workspace.ptr().as_ptr::<u8>().cast::<c_void>());
        let selected = base + layout.selected as u64;
        let counts = base + layout.counts as u64;
        let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let boxes = geometry.boxes as u32;
        let classes = geometry.classes as u32;
        let groups = geometry.groups as u32;
        let function = self
            .runtime
            .nvrtc_function("nms_f32_v1", SOURCE, "nms_materialize")?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&selected)
            .arg(&counts)
            .arg(&boxes)
            .arg(&classes)
            .arg(&groups)
            .arg(&output);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (groups, 1, 1),
                block_dim: (MAX_NMS_BOXES as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch NMS materialization phase", error))?;
        MATERIALIZE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        self.runtime.synchronize()
    }
}

impl Kernel for NonMaxSuppressionKernel {
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
        Err(EpError::KernelFailed(
            "cuda_ep NMS requires governed workspace; call execute_with_workspace".into(),
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
        let boxes = inputs
            .first()
            .filter(|input| input.present)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: missing boxes metadata".into()))?;
        let scores = inputs
            .get(1)
            .filter(|input| input.present)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: missing scores metadata".into()))?;
        if boxes.shape.len() != 3 || scores.shape.len() != 3 {
            return Err(EpError::KernelFailed(
                "cuda_ep NMS: boxes and scores metadata must be rank 3".into(),
            ));
        }
        let geometry = Geometry {
            boxes: boxes.shape[1],
            classes: scores.shape[1],
            groups: boxes.shape[0]
                .checked_mul(scores.shape[1])
                .ok_or_else(|| EpError::KernelFailed("cuda_ep NMS: group count overflow".into()))?,
        };
        let layout = workspace_layout(geometry)?;
        Ok(WorkspaceRequirement {
            bytes: layout.bytes as u64,
            alignment: 8,
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
        let metadata = self.prepare(inputs, &[true], workspace)?;
        let expected = metadata[0].as_ref().expect("required NMS output");
        if outputs.len() != 1
            || outputs[0].dtype != expected.dtype
            || outputs[0].shape != expected.shape
        {
            return Err(EpError::KernelFailed(
                "cuda_ep NMS: native output does not match prepared metadata".into(),
            ));
        }
        self.materialize(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "NonMaxSuppression uses the DeviceWorkspace two-phase path: \
             an 8-byte count D2H synchronization must precede dynamic ORT output allocation",
        )
    }
}

fn validate_views(inputs: &[TensorView]) -> Result<()> {
    if !(2..=5).contains(&inputs.len()) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep NMS: expected 2..=5 inputs, got {}",
            inputs.len()
        )));
    }
    let boxes = &inputs[0];
    let scores = &inputs[1];
    if boxes.dtype != DataType::Float32
        || scores.dtype != DataType::Float32
        || boxes.shape.len() != 3
        || boxes.shape[2] != 4
        || scores.shape.len() != 3
        || scores.shape[0] != boxes.shape[0]
        || scores.shape[2] != boxes.shape[1]
        || !boxes.is_contiguous()
        || !scores.is_contiguous()
    {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep NMS: expected contiguous Float32 boxes [B,N,4] and scores [B,C,N], got \
             {:?}{:?} and {:?}{:?}",
            boxes.dtype, boxes.shape, scores.dtype, scores.shape
        )));
    }
    for (slot, dtype) in [
        (2, DataType::Int64),
        (3, DataType::Float32),
        (4, DataType::Float32),
    ] {
        if let Some(input) = inputs.get(slot).filter(|input| !input.is_absent())
            && (input.dtype != dtype || !input.shape.is_empty() || !input.is_contiguous())
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep NMS: optional input {slot} must be a contiguous {dtype:?} scalar"
            )));
        }
    }
    Ok(())
}

pub(crate) fn unsupported_reason(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
    input_layouts: &[TensorLayout],
) -> Option<String> {
    let reject = || -> std::result::Result<(), String> {
        if !(2..=5).contains(&node.inputs.len()) || node.outputs.len() != 1 {
            return Err("requires 2..=5 inputs and one output".into());
        }
        match node.attr("center_point_box").and_then(Attribute::as_int) {
            None | Some(0 | 1) => {}
            Some(value) => return Err(format!("center_point_box must be 0 or 1, got {value}")),
        }
        let boxes = input_shapes
            .first()
            .and_then(|shape| onnx_runtime_ir::as_static_shape(shape))
            .ok_or_else(|| "boxes shape must be static".to_string())?;
        let scores = input_shapes
            .get(1)
            .and_then(|shape| onnx_runtime_ir::as_static_shape(shape))
            .ok_or_else(|| "scores shape must be static".to_string())?;
        if boxes.len() != 3
            || boxes[2] != 4
            || scores.len() != 3
            || scores[0] != boxes[0]
            || scores[2] != boxes[1]
        {
            return Err(format!(
                "expected boxes [B,N,4] and scores [B,C,N], got {boxes:?} and {scores:?}"
            ));
        }
        let groups = boxes[0]
            .checked_mul(scores[1])
            .ok_or_else(|| "batch×classes overflows usize".to_string())?;
        if boxes[1] > MAX_NMS_BOXES || groups > MAX_NMS_GROUPS {
            return Err(format!(
                "bounded CUDA limits are {MAX_NMS_BOXES} boxes and {MAX_NMS_GROUPS} \
                 batch×class groups; got {} and {groups}",
                boxes[1]
            ));
        }
        for (slot, name) in [
            (2usize, "max_output_boxes_per_class"),
            (3usize, "iou_threshold"),
            (4usize, "score_threshold"),
        ] {
            if node.inputs.get(slot).is_some_and(Option::is_some) {
                let shape = input_shapes
                    .get(slot)
                    .and_then(|shape| onnx_runtime_ir::as_static_shape(shape))
                    .ok_or_else(|| {
                        format!("input {slot} ('{name}') shape must be known static rank-0")
                    })?;
                if !shape.is_empty() {
                    return Err(format!(
                        "input {slot} ('{name}') must be a scalar with shape [], got {shape:?}"
                    ));
                }
            }
        }
        for (slot, dtype) in [
            (0, DataType::Float32),
            (1, DataType::Float32),
            (2, DataType::Int64),
            (3, DataType::Float32),
            (4, DataType::Float32),
        ] {
            if node.inputs.get(slot).is_some_and(Option::is_some)
                && input_dtypes.get(slot) != Some(&dtype)
            {
                return Err(format!("input {slot} must be {dtype:?}"));
            }
            if node.inputs.get(slot).is_some_and(Option::is_some)
                && input_layouts.get(slot).is_some_and(|layout| {
                    input_shapes
                        .get(slot)
                        .and_then(|shape| onnx_runtime_ir::as_static_shape(shape))
                        .is_none_or(|shape| !layout.is_contiguous(&shape))
                })
            {
                return Err(format!("input {slot} must be contiguous"));
            }
        }
        Ok(())
    };
    reject()
        .err()
        .map(|reason| format!("NonMaxSuppression: {reason}"))
}
