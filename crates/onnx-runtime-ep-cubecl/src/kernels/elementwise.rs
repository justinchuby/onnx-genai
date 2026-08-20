//! Elementwise kernels: `Add`, `Mul`, `Relu`.
//!
//! # Broadcasting
//!
//! Full ONNX multidirectional broadcasting is not implemented. Two cases are
//! claimed because they cover the overwhelming majority of real graphs and both
//! are checkable up front:
//!
//! * identical shapes, and
//! * one operand with a single element (a scalar bias or scale).
//!
//! Anything else is rejected by the factory with the two shapes printed, so the
//! node falls back to another EP rather than being silently mis-broadcast.

use std::sync::Arc;

use cubecl::prelude::*;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{
    CUBE_DIM, ElementKind, FnKernel, cube_count, input_handle, launch_kind, output_handle,
};
use crate::context::CubeclContext;

#[cube(launch_unchecked)]
fn binary_same_shape<F: Float>(lhs: &[F], rhs: &[F], out: &mut [F], #[comptime] op: BinaryOp) {
    let index = ABSOLUTE_POS;
    if index < out.len() {
        let a = lhs[index];
        let b = rhs[index];
        out[index] = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Mul => a * b,
        };
    }
}

#[cube(launch_unchecked)]
fn binary_scalar_rhs<F: Float>(lhs: &[F], rhs: &[F], out: &mut [F], #[comptime] op: BinaryOp) {
    let index = ABSOLUTE_POS;
    if index < out.len() {
        let a = lhs[index];
        let b = rhs[0];
        out[index] = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Mul => a * b,
        };
    }
}

#[cube(launch_unchecked)]
fn binary_scalar_lhs<F: Float>(lhs: &[F], rhs: &[F], out: &mut [F], #[comptime] op: BinaryOp) {
    let index = ABSOLUTE_POS;
    if index < out.len() {
        let a = lhs[0];
        let b = rhs[index];
        out[index] = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Mul => a * b,
        };
    }
}

#[cube(launch_unchecked)]
fn relu_kernel<F: Float>(input: &[F], out: &mut [F]) {
    let index = ABSOLUTE_POS;
    if index < out.len() {
        out[index] = input[index].max(F::new(0.0_f32));
    }
}

/// Which elementwise binary operation a launch performs.
///
/// Carried as a `#[comptime]` argument so both operators share one kernel body
/// and one code path, while still specialising to a single instruction in the
/// generated shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Mul,
}

impl BinaryOp {
    fn op_type(self) -> &'static str {
        match self {
            BinaryOp::Add => "Add",
            BinaryOp::Mul => "Mul",
        }
    }
}

/// How the two operands line up, decided once when the kernel is created.
#[derive(Debug, Clone, Copy)]
enum Pairing {
    SameShape,
    ScalarRhs,
    ScalarLhs,
}

pub struct BinaryFactory<R: Runtime> {
    context: Arc<CubeclContext<R>>,
    op: BinaryOp,
}

impl<R: Runtime> BinaryFactory<R> {
    pub fn new(context: Arc<CubeclContext<R>>, op: BinaryOp) -> Self {
        Self { context, op }
    }
}

impl<R: Runtime> KernelFactory for BinaryFactory<R> {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let op = self.op;
        let [lhs_shape, rhs_shape] = input_shapes else {
            return Err(EpError::KernelFailed(format!(
                "cubecl_ep: {} node '{}' has {} inputs, expected exactly 2",
                op.op_type(),
                node.name,
                input_shapes.len()
            )));
        };
        let lhs_len: usize = lhs_shape.iter().product();
        let rhs_len: usize = rhs_shape.iter().product();
        let (pairing, out_len) = if lhs_shape == rhs_shape {
            (Pairing::SameShape, lhs_len)
        } else if rhs_len == 1 {
            (Pairing::ScalarRhs, lhs_len)
        } else if lhs_len == 1 {
            (Pairing::ScalarLhs, rhs_len)
        } else {
            return Err(EpError::KernelFailed(format!(
                "cubecl_ep: {} node '{}' needs broadcasting between {lhs_shape:?} and \
                 {rhs_shape:?}, which these backends do not implement. Only equal shapes and a \
                 single-element operand are supported; assign this node to another EP.",
                op.op_type(),
                node.name,
            )));
        };

        let context = self.context.clone();
        Ok(Box::new(FnKernel(
            move |inputs: &[TensorView<'_>], outputs: &mut [TensorMut<'_>]| {
                let [lhs, rhs] = inputs else {
                    return Err(EpError::KernelFailed(format!(
                        "cubecl_ep: {} expected 2 inputs at execution, got {}",
                        op.op_type(),
                        inputs.len()
                    )));
                };
                let Some(out) = outputs.first_mut() else {
                    return Err(EpError::KernelFailed(format!(
                        "cubecl_ep: {} expected 1 output at execution, got 0",
                        op.op_type()
                    )));
                };
                let kind = launch_kind(
                    &[(lhs.dtype, "A"), (rhs.dtype, "B"), (out.dtype, "C")],
                    context.f16,
                    op.op_type(),
                )?;

                let lhs_res = input_handle(&context, lhs, "A")?;
                let rhs_res = input_handle(&context, rhs, "B")?;
                let out_res = output_handle(&context, out, "C")?;

                // The kernels are generic over `F`, so the element type is a type
                // parameter and the two arms below are separate monomorphisations
                // of one body rather than two implementations to keep in sync.
                macro_rules! dispatch {
                    ($float:ty) => {
                        // SAFETY: every handle was resolved from a live allocation of
                        // this provider, and `resolve` already checked that each buffer
                        // holds at least the element count passed here, so no launch
                        // can index past the memory it was given.
                        unsafe {
                            match pairing {
                                Pairing::SameShape => {
                                    binary_same_shape::launch_unchecked::<$float, R>(
                                        &context.client,
                                        cube_count(out_len),
                                        CubeDim::new_1d(CUBE_DIM),
                                        BufferArg::from_raw_parts(lhs_res.handle, out_len),
                                        BufferArg::from_raw_parts(rhs_res.handle, out_len),
                                        BufferArg::from_raw_parts(out_res.handle, out_len),
                                        op,
                                    )
                                }
                                Pairing::ScalarRhs => {
                                    binary_scalar_rhs::launch_unchecked::<$float, R>(
                                        &context.client,
                                        cube_count(out_len),
                                        CubeDim::new_1d(CUBE_DIM),
                                        BufferArg::from_raw_parts(lhs_res.handle, out_len),
                                        BufferArg::from_raw_parts(rhs_res.handle, 1),
                                        BufferArg::from_raw_parts(out_res.handle, out_len),
                                        op,
                                    )
                                }
                                Pairing::ScalarLhs => {
                                    binary_scalar_lhs::launch_unchecked::<$float, R>(
                                        &context.client,
                                        cube_count(out_len),
                                        CubeDim::new_1d(CUBE_DIM),
                                        BufferArg::from_raw_parts(lhs_res.handle, 1),
                                        BufferArg::from_raw_parts(rhs_res.handle, out_len),
                                        BufferArg::from_raw_parts(out_res.handle, out_len),
                                        op,
                                    )
                                }
                            }
                        }
                    };
                }
                match kind {
                    ElementKind::F32 => dispatch!(f32),
                    ElementKind::F16 => dispatch!(half::f16),
                }
                Ok(())
            },
        )))
    }
}

pub struct ReluFactory<R: Runtime> {
    context: Arc<CubeclContext<R>>,
}

impl<R: Runtime> ReluFactory<R> {
    pub fn new(context: Arc<CubeclContext<R>>) -> Self {
        Self { context }
    }
}

impl<R: Runtime> KernelFactory for ReluFactory<R> {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let [shape] = input_shapes else {
            return Err(EpError::KernelFailed(format!(
                "cubecl_ep: Relu node '{}' has {} inputs, expected exactly 1",
                node.name,
                input_shapes.len()
            )));
        };
        let len: usize = shape.iter().product();
        let context = self.context.clone();
        Ok(Box::new(FnKernel(
            move |inputs: &[TensorView<'_>], outputs: &mut [TensorMut<'_>]| {
                let Some(input) = inputs.first() else {
                    return Err(EpError::KernelFailed(
                        "cubecl_ep: Relu expected 1 input at execution, got 0".to_string(),
                    ));
                };
                let Some(out) = outputs.first_mut() else {
                    return Err(EpError::KernelFailed(
                        "cubecl_ep: Relu expected 1 output at execution, got 0".to_string(),
                    ));
                };
                let kind =
                    launch_kind(&[(input.dtype, "X"), (out.dtype, "Y")], context.f16, "Relu")?;

                let input_res = input_handle(&context, input, "X")?;
                let out_res = output_handle(&context, out, "Y")?;

                macro_rules! dispatch {
                    ($float:ty) => {
                        // SAFETY: as in the binary kernels, both handles come from
                        // live allocations that `resolve` sized against `len`.
                        unsafe {
                            relu_kernel::launch_unchecked::<$float, R>(
                                &context.client,
                                cube_count(len),
                                CubeDim::new_1d(CUBE_DIM),
                                BufferArg::from_raw_parts(input_res.handle, len),
                                BufferArg::from_raw_parts(out_res.handle, len),
                            );
                        }
                    };
                }
                match kind {
                    ElementKind::F32 => dispatch!(f32),
                    ElementKind::F16 => dispatch!(half::f16),
                }
                Ok(())
            },
        )))
    }
}
