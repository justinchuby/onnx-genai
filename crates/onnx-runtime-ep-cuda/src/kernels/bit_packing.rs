use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::packed_routed::{fail, pack_layout, validate};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = concat!(
    include_str!("packed_kernel_prelude.cuh"),
    include_str!("bit_packing.cu")
);

pub(super) struct Factory(pub Arc<CudaRuntime>);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native {
            op: Op::from_node(node).map_err(EpError::KernelFailed)?,
            runtime: Arc::clone(&self.0),
        }))
    }
}

struct Native {
    op: Op,
    runtime: Arc<CudaRuntime>,
}

fn launch_config(lanes: usize) -> LaunchConfig {
    const BLOCK: u32 = 256;
    LaunchConfig {
        grid_dim: (lanes.div_ceil(BLOCK as usize).clamp(1, 65_535) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Kernel for Native {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "opset 29 packing and routing currently reads validation status from the GPU; use eager execution",
        )
    }

    fn workspace_requirement(
        &self,
        _inputs: &[TensorMetadata<'_>],
        outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            super::packed_validation::requirement(super::packed_validation::packing_needed(
                self.op,
                outputs.first().and_then(|output| match output {
                    onnx_runtime_ep_api::OutputMetadata::Tensor { shape, .. } => {
                        Some(shape.as_ref())
                    }
                    _ => None,
                }),
            )),
        ))
    }

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(x, y, None)
    }

    fn execute_with_workspace(
        &self,
        x: &[TensorView],
        y: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let op = self.op;
        validate(op, x, y)?;
        if x.iter()
            .any(|t| t.dtype != DataType::Undefined && !t.is_contiguous())
            || !y[0].is_contiguous()
        {
            return Err(fail(
                op,
                "CUDA requires contiguous operands; materialize strided tensors first",
            ));
        }
        let ptr = |t: &TensorView| cuptr(t.data_ptr::<u8>() as *const c_void);
        let input = ptr(&x[0]);
        let output = cuptr(y[0].data_ptr_mut::<u8>() as *const c_void);
        let stream = self.runtime.stream();
        let l = pack_layout(op, &x[0], &y[0], None)?;
        if !super::packed_validation::packing_needed(op, Some(y[0].shape)) {
            return Ok(());
        }
        // Keep tensor data on the device; only the four-byte validation result
        // is read back. This kernel is eager-only until it uses the capture latch.
        let status = super::packed_validation::status(workspace, op.name())?;
        // SAFETY: the checked caller-owned status region covers four bytes.
        unsafe { self.runtime.htod(&[0; 4], status)? };

        let unpack = matches!(op, Op::Unpack(_));
        let entry = match (unpack, l.word_bits) {
            (true, 8) => "unpack_u8",
            (true, _) => "unpack_u32",
            (false, 8) => "pack_u8",
            (false, _) => "pack_u32",
        };
        let func = self.runtime.nvrtc_function("bit_packing", SOURCE, entry)?;
        let count_input = if unpack { ptr(&x[1]) } else { 0 };
        let (lanes, count, words, inner) = (
            l.lanes as u64,
            l.count as u64,
            l.words as u64,
            l.inner as u64,
        );
        let p = l.packing;
        let (group, bits) = (p.group as u64, p.bits as u64);
        let mut launch = stream.launch_builder(&func);
        launch
            .arg(&input)
            .arg(&output)
            .arg(&count_input)
            .arg(&status)
            .arg(&lanes)
            .arg(&count)
            .arg(&words)
            .arg(&inner)
            .arg(&p.radix)
            .arg(&group)
            .arg(&bits)
            .arg(&p.limit);
        // SAFETY: tensor types and extents were validated, and arguments match PACK_ENTRY.
        unsafe { launch.launch(launch_config(l.lanes)) }
            .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        let mut bytes = [0; 4];
        // SAFETY: the live flag covers four bytes; dtoh drains the compute stream.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        let bits = u32::from_ne_bytes(bytes);
        if bits == 0 {
            return Ok(());
        }
        Err(fail(
            op,
            match op {
                Op::Pack(p, _) => format!(
                    "input contains a code outside [0, {}); fix X or the packing attributes",
                    p.radix
                ),
                Op::Unpack(_) if bits == 2 => format!(
                    "count must be nonnegative and equal the output axis extent {}",
                    l.count
                ),
                Op::Unpack(p) => format!(
                    "payload contains a group index outside [0, {}); check the radix and group_size or regenerate X",
                    p.limit
                ),
                Op::GroupedMatMul { .. } => unreachable!("registered Pack/Unpack op"),
            },
        ))
    }
}
