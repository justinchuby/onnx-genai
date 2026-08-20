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

/// Dtypes claimed on an adapter without f16.
///
/// wgpu guarantees f32 storage buffers on every adapter, so this set is always
/// safe to advertise.
const F32_ONLY: &[DataType] = &[DataType::Float32];

/// Dtypes claimed on an adapter that passed [`crate::runtime::supports_f16`].
///
/// Which of the two sets a host is told about is decided at device-open time,
/// not compile time: `shader-f16` is optional in the WebGPU baseline, and
/// advertising f16 on an adapter that lacks it would give a provider that
/// loads, is handed an f16 node, and fails at first dispatch.
const F32_AND_F16: &[DataType] = &[DataType::Float32, DataType::Float16];

/// Every operator this EP claims.
pub const SUPPORTED_OPS: &[CubeclOpDescriptor] = &[
    CubeclOpDescriptor {
        op_type: "Add",
        domain: "",
        since_version: 7,
        supported_dtypes: F32_AND_F16,
    },
    CubeclOpDescriptor {
        op_type: "Mul",
        domain: "",
        since_version: 7,
        supported_dtypes: F32_AND_F16,
    },
    CubeclOpDescriptor {
        op_type: "Relu",
        domain: "",
        since_version: 6,
        supported_dtypes: F32_AND_F16,
    },
    CubeclOpDescriptor {
        op_type: "MatMul",
        domain: "",
        since_version: 9,
        supported_dtypes: F32_AND_F16,
    },
];

/// The operator surface to advertise for a device, given its f16 probe result.
///
/// Returns descriptors whose `supported_dtypes` reflect what that specific
/// adapter can actually run, so a host never learns about a dtype this device
/// would then refuse.
pub fn supported_ops_for(f16_available: bool) -> Vec<CubeclOpDescriptor> {
    SUPPORTED_OPS
        .iter()
        .map(|descriptor| CubeclOpDescriptor {
            supported_dtypes: if f16_available { F32_AND_F16 } else { F32_ONLY },
            ..*descriptor
        })
        .collect()
}

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

/// Which float type a launch is instantiated with.
///
/// The kernels are generic over `F: Float`, so the element type is chosen once
/// per launch here and threaded through as a type parameter. Keeping it an enum
/// rather than branching on `DataType` at each call site means adding a type is
/// a compile error everywhere it must be handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementKind {
    F32,
    F16,
}

impl ElementKind {
    /// Pick the launch type for `dtype`, or explain why it cannot be used.
    ///
    /// `f16_available` is the device probe result, not a build-time constant:
    /// the same binary runs on adapters that do and do not expose f16, and the
    /// refusal has to name the missing device feature so the reader knows this
    /// is a hardware limit rather than a gap in this EP.
    pub(crate) fn resolve(
        dtype: DataType,
        f16_available: bool,
        op: &str,
        what: &str,
    ) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(ElementKind::F32),
            DataType::Float16 if f16_available => Ok(ElementKind::F16),
            DataType::Float16 => Err(EpError::KernelFailed(format!(
                "cubecl_ep: {op} received Float16 for '{what}', but this adapter does not report \
                 f16 buffer and arithmetic support (WebGPU 'shader-f16' / wgpu SHADER_F16). Cast \
                 the tensor to f32, assign this node to another EP, or run on an adapter with \
                 f16 support."
            ))),
            other => Err(EpError::KernelFailed(format!(
                "cubecl_ep: {op} received {other:?} for '{what}', but the cubecl backends \
                 implement f32 and f16 only. Cast the tensor, or assign this node to another EP."
            ))),
        }
    }
}

/// Resolve one launch type for a whole operator, refusing mixed dtypes.
///
/// ONNX requires every float operand of these ops to share a type, and the
/// kernels are monomorphic in `F`, so a mismatch has to be caught here rather
/// than reinterpreting one operand's bytes as the other's type.
pub(crate) fn launch_kind(
    operands: &[(DataType, &str)],
    f16_available: bool,
    op: &str,
) -> Result<ElementKind> {
    let mut resolved: Option<(ElementKind, &str)> = None;
    for (dtype, what) in operands {
        let kind = ElementKind::resolve(*dtype, f16_available, op, what)?;
        match resolved {
            None => resolved = Some((kind, what)),
            Some((first, first_what)) if first != kind => {
                return Err(EpError::KernelFailed(format!(
                    "cubecl_ep: {op} was given {first:?} for '{first_what}' and {kind:?} for \
                     '{what}'. These kernels are compiled for a single float type per node; \
                     insert a Cast so both operands match."
                )));
            }
            Some(_) => {}
        }
    }
    resolved.map(|(kind, _)| kind).ok_or_else(|| {
        EpError::KernelFailed(format!("cubecl_ep: {op} has no operands to type-check"))
    })
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
