//! GPU kernels, and the plumbing that turns an EP tensor view into a launch.
//!
//! # Scope of this first backend
//!
//! Only the operators listed in [`SUPPORTED_OPS`] are claimed, and only for the
//! dtypes and shapes each factory verifies. Everything else is rejected by
//! `supports_op` with a reason naming exactly what was unsupported, so a graph
//! falls back to another EP node by node instead of failing at load time or —
//! worse — silently producing wrong numbers.
//!
//! # Why shapes are compile-time constants
//!
//! `ExecutionProvider::get_kernel` is handed the concrete input shapes, so each
//! kernel object already knows its dimensions. Passing them as `#[comptime]`
//! lets CubeCL fold the index arithmetic and specialise the loop bounds, and
//! costs nothing at runtime because CubeCL caches compiled kernels: a
//! transformer reuses the same handful of shapes for its whole run.

pub mod elementwise;
pub mod matmul;

use std::sync::Arc;

use cubecl::prelude::*;
use onnx_runtime_ep_api::{EpError, Kernel, OpKey, OpRegistry, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

use crate::context::CubeclContext;
use crate::memory::Resolved;

/// One entry in this EP's operator surface, used both to populate the internal
/// [`OpRegistry`] and to advertise kernels across the plugin ABIs.
#[derive(Debug, Clone, Copy)]
pub struct CubeclOpDescriptor {
    pub op_type: &'static str,
    pub domain: &'static str,
    pub since_version: i32,
    pub supported_dtypes: &'static [DataType],
}

/// f32 is the only element type this round supports.
///
/// wgpu guarantees f32 storage buffers everywhere; f16 needs the `shader-f16`
/// feature, which is optional on Vulkan and absent on much of the WebGPU
/// baseline. Claiming f16 before probing for that feature would produce a
/// provider that loads and then fails at first dispatch, which rule 5 forbids.
const F32_ONLY: &[DataType] = &[DataType::Float32];

/// Every operator this EP claims.
pub const SUPPORTED_OPS: &[CubeclOpDescriptor] = &[
    CubeclOpDescriptor {
        op_type: "Add",
        domain: "",
        since_version: 7,
        supported_dtypes: F32_ONLY,
    },
    CubeclOpDescriptor {
        op_type: "Mul",
        domain: "",
        since_version: 7,
        supported_dtypes: F32_ONLY,
    },
    CubeclOpDescriptor {
        op_type: "Relu",
        domain: "",
        since_version: 6,
        supported_dtypes: F32_ONLY,
    },
    CubeclOpDescriptor {
        op_type: "MatMul",
        domain: "",
        since_version: 9,
        supported_dtypes: F32_ONLY,
    },
];

/// Build the registry that backs `supports_op`/`get_kernel`.
pub fn build_registry<R: Runtime>(context: Arc<CubeclContext<R>>) -> OpRegistry {
    let mut registry = OpRegistry::new();
    registry.register(
        OpKey::new("Add", "", 7),
        Box::new(elementwise::BinaryFactory::new(
            context.clone(),
            elementwise::BinaryOp::Add,
        )),
    );
    registry.register(
        OpKey::new("Mul", "", 7),
        Box::new(elementwise::BinaryFactory::new(
            context.clone(),
            elementwise::BinaryOp::Mul,
        )),
    );
    registry.register(
        OpKey::new("Relu", "", 6),
        Box::new(elementwise::ReluFactory::new(context.clone())),
    );
    registry.register(
        OpKey::new("MatMul", "", 9),
        Box::new(matmul::MatMulFactory::new(context)),
    );
    registry
}

/// Number of threads per cube for the 1-D elementwise launches.
///
/// 256 is the largest workgroup size guaranteed by the WebGPU baseline
/// (`maxComputeInvocationsPerWorkgroup`), so it is the widest cube that is
/// portable across every adapter both backends may land on.
pub(crate) const CUBE_DIM: u32 = 256;

/// Split `elements` into a 1-D cube count for [`CUBE_DIM`]-wide cubes.
pub(crate) fn cube_count(elements: usize) -> CubeCount {
    let cubes = elements.div_ceil(CUBE_DIM as usize).max(1) as u32;
    CubeCount::Static(cubes, 1, 1)
}

/// Resolve a read-only input to the CubeCL handle it lives in.
pub(crate) fn input_handle<R: Runtime>(
    context: &CubeclContext<R>,
    view: &TensorView<'_>,
    what: &str,
) -> Result<Resolved> {
    if view.is_absent() {
        return Err(EpError::InvalidTensorView {
            reason: format!("cubecl_ep: required input '{what}' was not supplied"),
        });
    }
    require_contiguous(view.is_contiguous(), what)?;
    let addr = view.data.as_ptr::<u8>().wrapping_add(view.byte_offset);
    context.table.resolve(addr.cast(), view.byte_size())
}

/// Resolve an output to the CubeCL handle it must be written into.
pub(crate) fn output_handle<R: Runtime>(
    context: &CubeclContext<R>,
    view: &mut TensorMut<'_>,
    what: &str,
) -> Result<Resolved> {
    if view.is_absent() {
        return Err(EpError::InvalidTensorView {
            reason: format!("cubecl_ep: required output '{what}' was not supplied"),
        });
    }
    require_contiguous(view.is_contiguous(), what)?;
    let byte_size = view.byte_size();
    let addr = view.data_ptr_mut::<u8>().wrapping_add(view.byte_offset);
    context.table.resolve(addr.cast(), byte_size)
}

fn require_contiguous(contiguous: bool, what: &str) -> Result<()> {
    if contiguous {
        return Ok(());
    }
    Err(EpError::InvalidTensorView {
        reason: format!(
            "cubecl_ep: tensor '{what}' is strided, and these kernels index buffers linearly. \
             Insert a contiguous copy before this node, or run it on an EP that handles \
             arbitrary strides."
        ),
    })
}

/// Reject a dtype this EP does not implement, naming the one it does.
pub(crate) fn require_f32(dtype: DataType, op: &str, what: &str) -> Result<()> {
    if dtype == DataType::Float32 {
        return Ok(());
    }
    Err(EpError::KernelFailed(format!(
        "cubecl_ep: {op} received {dtype:?} for '{what}', but the cubecl backends currently \
         implement f32 only. Cast the tensor to f32, or assign this node to another EP."
    )))
}

/// Shared object-safe kernel wrapper so factories can return closures over the
/// context without each defining its own struct boilerplate.
pub(crate) struct FnKernel<F>(pub F)
where
    F: Fn(&[TensorView<'_>], &mut [TensorMut<'_>]) -> Result<()> + Send + Sync;

impl<F> Kernel for FnKernel<F>
where
    F: Fn(&[TensorView<'_>], &mut [TensorMut<'_>]) -> Result<()> + Send + Sync,
{
    fn execute(&self, inputs: &[TensorView<'_>], outputs: &mut [TensorMut<'_>]) -> Result<()> {
        (self.0)(inputs, outputs)
    }
}
