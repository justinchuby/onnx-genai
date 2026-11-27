use std::{alloc::Layout, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg, result::memset_d8_async};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata,
    TensorMut, TensorView, WorkspaceExecutionPreparation, WorkspaceLayout, WorkspaceLifetime,
    WorkspacePreparation, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::Node;
use onnx_runtime_ir::max_unpool::{self as rules, Geometry};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = r#"
using u64 = unsigned long long;
extern "C" __global__ void select_sources(const long long* indices, u64 n,
    u64 outputs, u64* records) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (u64)gridDim.x * blockDim.x) {
    const long long target = indices[i];
    if (target < 0 || (u64)target >= outputs) atomicMax(&records[0], i + 1);
    else atomicMax(&records[(u64)target + 1], i + 1);
  }
}

extern "C" __global__ void write_values(const unsigned char* input,
    const u64* records, u64 n, int width, unsigned char* output) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (u64)gridDim.x * blockDim.x) {
    const u64 winner = records[i + 1];
    for (int b = 0; b < width; ++b) {
      output[i * width + b] = winner ? input[(winner - 1) * width + b] : 0;
    }
  }
}
"#;

fn launch_config(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// One word holds an invalid source position, then one word per output holds
// the last source position (one-based so zero means unwritten).
fn scratch(outputs: usize) -> Result<WorkspaceLayout<1>> {
    let region = outputs.checked_add(1)
        .and_then(|n| Layout::array::<u64>(n).ok())
        .ok_or_else(|| EpError::KernelFailed(
            "MaxUnpool: CUDA scatter workspace exceeds addressable storage; reduce the output shape".into()
        ))?;
    WorkspaceLayout::new([region])
}

pub struct MaxUnpoolFactory {
    pub version: u64,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MaxUnpoolFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let rank = shapes
            .first()
            .ok_or_else(|| {
                EpError::KernelFailed("MaxUnpool: missing X shape; supply X and I".into())
            })?
            .len();
        let geometry = Geometry::from_node(node, rank).map_err(EpError::KernelFailed)?;
        Ok(Box::new(MaxUnpool {
            version: self.version,
            geometry,
            runtime: self.runtime.clone(),
        }))
    }
}

struct MaxUnpool {
    version: u64,
    geometry: Geometry,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for MaxUnpool {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let invalid = EpError::KernelFailed;
        let Some(x) = inputs.first().filter(|input| input.present) else {
            return Err(invalid(
                "MaxUnpool: provide X to size CUDA workspace".into(),
            ));
        };
        let width = rules::element_bytes(self.version, x.dtype).map_err(invalid)?;
        if rules::checked_bytes("X", x.shape, width).map_err(invalid)? == 0 {
            return Ok(WorkspacePreparation::Ready(WorkspaceRequirement::NONE));
        }
        let shape = match outputs {
            [OutputMetadata::Tensor { shape, .. }] => shape,
            [OutputMetadata::Unresolved { .. }] => return Ok(WorkspacePreparation::Deferred),
            _ => {
                return Err(invalid(
                    "MaxUnpool: supply one output to size CUDA workspace".into(),
                ));
            }
        };
        let elements = rules::checked_bytes("output", shape, width).map_err(invalid)? / width;
        let layout = scratch(elements)?;
        Ok(WorkspacePreparation::Ready(WorkspaceRequirement {
            bytes: layout.bytes() as u64,
            alignment: layout.alignment(),
            lifetime: WorkspaceLifetime::StepScoped,
        }))
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        _: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
        _: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let [output] = outputs else {
            return Err(EpError::KernelFailed(
                "MaxUnpool: supply one bound output to size CUDA workspace".into(),
            ));
        };
        let metadata = inputs
            .first()
            .map(|x| TensorMetadata::new(x.dtype, x.shape, !x.is_absent()));
        self.workspace_requirement(metadata.as_slice(), &[OutputMetadata::from_view(output)])?
            .require_ready()
            .map(WorkspaceExecutionPreparation::Ready)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let invalid = EpError::KernelFailed;
        if !(2..=3).contains(&inputs.len())
            || inputs[..2].iter().any(TensorView::is_absent)
            || outputs.len() != 1
            || outputs[0].is_absent()
        {
            return Err(invalid(
                "MaxUnpool: supply X, I, optional output_shape and one output".into(),
            ));
        }
        let x = &inputs[0];
        let indices = &inputs[1];
        let output = &mut outputs[0];
        let width = rules::element_bytes(self.version, x.dtype).map_err(invalid)?;
        rules::validate_i64("I", indices.dtype).map_err(invalid)?;
        rules::validate_indices_shape(x.shape, indices.shape).map_err(invalid)?;
        if !x.is_contiguous() || !indices.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "MaxUnpool requires contiguous CUDA tensors",
            ));
        }
        let input_bytes = rules::checked_bytes("X", x.shape, width).map_err(invalid)?;
        rules::checked_bytes("I", indices.shape, 8).map_err(invalid)?;
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "MaxUnpool requires eager CUDA execution to check index bounds",
            ));
        }
        let explicit = inputs
            .get(2)
            .filter(|input| !input.is_absent())
            .map(|input| {
                rules::validate_i64("output_shape", input.dtype).map_err(invalid)?;
                rules::validate_control_shape(x.shape.len(), input.shape).map_err(invalid)?;
                super::index_transform::device_i64(
                    &self.runtime,
                    input,
                    "MaxUnpool",
                    "output_shape",
                )
            })
            .transpose()?;
        let expected = self
            .geometry
            .output_shape(x.shape, explicit.as_deref())
            .map_err(invalid)?;
        if output.dtype != x.dtype || output.shape != expected.as_slice() {
            return Err(invalid(format!(
                "MaxUnpool: output is {:?} {:?}; bind {:?} {expected:?}",
                output.dtype, output.shape, x.dtype
            )));
        }
        let bytes = rules::checked_bytes("output", &expected, width).map_err(invalid)?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>().cast_const().cast());
        self.runtime.bind()?;
        let clear = |ptr, bytes| {
            unsafe { memset_d8_async(ptr, 0, bytes, self.runtime.stream().cu_stream()) }
                .map_err(|error| driver_err("clear MaxUnpool CUDA buffer", error))
        };
        if input_bytes == 0 {
            if bytes != 0 {
                clear(output_ptr, bytes)?;
            }
            return Ok(());
        }
        let elements = (bytes / width) as u64;
        let n = (input_bytes / width) as u64;
        let layout = scratch(elements as usize)?;
        let [region] = layout.bind(workspace)?;
        let records = cuptr(region.ptr().0.cast_const());
        let select = self
            .runtime
            .nvrtc_function("max_unpool_v1", SOURCE, "select_sources")?;
        clear(records, layout.bytes())?;
        let index_ptr = cuptr(indices.data_ptr::<u8>().cast());
        let mut launcher = self.runtime.stream().launch_builder(&select);
        launcher
            .arg(&index_ptr)
            .arg(&n)
            .arg(&elements)
            .arg(&records);
        unsafe { launcher.launch(launch_config(n)) }
            .map_err(|error| driver_err("select MaxUnpool CUDA source positions", error))?;
        let mut status = [0; 8];
        unsafe { self.runtime.dtoh(&mut status, records)? };
        let bad = u64::from_ne_bytes(status);
        if bad != 0 {
            let position = bad - 1;
            let mut value = [0; 8];
            unsafe { self.runtime.dtoh(&mut value, index_ptr + position * 8)? };
            let index = i64::from_ne_bytes(value);
            return Err(invalid(format!(
                "MaxUnpool: I[{position}]={index} is outside [0, {elements}); supply flat output indices"
            )));
        }
        let write = self
            .runtime
            .nvrtc_function("max_unpool_v1", SOURCE, "write_values")?;
        let x_ptr = cuptr(x.data_ptr::<u8>().cast());
        let width = width as i32;
        let mut launcher = self.runtime.stream().launch_builder(&write);
        launcher
            .arg(&x_ptr)
            .arg(&records)
            .arg(&elements)
            .arg(&width)
            .arg(&output_ptr);
        unsafe { launcher.launch(launch_config(elements)) }
            .map_err(|error| driver_err("write MaxUnpool CUDA output", error))?;
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "MaxUnpool reads CUDA index-validation status before writing output",
        )
    }
}
