//! Native CPU execution for ONNX `Einsum` (opset 12).
//!
//! The equation is parsed exactly once, by [`EinsumPlan`], when the
//! shape-specialized kernel is built. Execution consumes only the plan's
//! structural classification and axis maps.

use std::borrow::Cow;
use std::cell::RefCell;

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput,
};
use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumClassification, EinsumContractionPlan, EinsumInput,
    EinsumOperandPlan, EinsumPermutationPlan, EinsumPlan, EinsumReductionPlan, Node, Shape,
    compute_contiguous_strides,
};
use rayon::prelude::*;

use super::{check_arity, matmul::MatMulKernel, to_dense_bytes, write_dense_bytes};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::next_index;

/// Diagnostic execution switch used by `benches/einsum.rs`.
///
/// `oracle` routes arithmetic through the canonical plan's generic
/// high-precision evaluator instead of the native lowering. It is a
/// correctness diagnostic, not a replaceable performance baseline. The default is
/// `optimized`. It is read only when a kernel is constructed.
pub const EINSUM_MODE_ENV: &str = "NXRT_CPU_EINSUM_MODE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionMode {
    Optimized,
    Oracle,
}

#[derive(Default)]
struct EinsumScratch {
    f32_output: Vec<f32>,
}

/// Shape-specialized CPU Einsum kernel.
pub struct EinsumKernel {
    plan: EinsumPlan,
    matmul: MatMulKernel,
    scratch: RefCell<EinsumScratch>,
    mode: ExecutionMode,
    flops: Option<u64>,
    #[cfg(test)]
    last_route: std::sync::atomic::AtomicU8,
}

/// Factory for [`EinsumKernel`].
pub struct EinsumFactory;

impl KernelFactory for EinsumFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let equation = equation(node)?;
        if input_shapes.is_empty() {
            return Err(EpError::KernelFailed(
                "Einsum: expected at least one input shape; the canonical planner cannot build an \
                 execution plan without operand ranks. HOW: provide inferred input shapes before \
                 kernel construction."
                    .into(),
            ));
        }
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let plan = EinsumPlan::build_for_shapes(equation, &input_shape_refs).map_err(|error| {
            EpError::KernelFailed(format!(
                "Einsum: canonical planning failed for `{equation}`: {error}"
            ))
        })?;
        if let EinsumClassification::Unsupported(reason) = plan.classification() {
            return Err(EpError::KernelFailed(format!(
                "Einsum equation `{}` is valid but unsupported by the native CPU EP: {reason}. \
                 Supported native classes are view/diagonal, reduction or elementwise product, \
                 and binary GEMM/BMM-compatible contractions.",
                plan.equation()
            )));
        }
        let mode = execution_mode()?;
        let flops = match plan.classification() {
            EinsumClassification::Gemm(gemm) => gemm_flops(gemm),
            _ => None,
        };
        Ok(Box::new(EinsumKernel {
            plan,
            matmul: MatMulKernel::default(),
            scratch: RefCell::new(EinsumScratch::default()),
            mode,
            flops,
            #[cfg(test)]
            last_route: std::sync::atomic::AtomicU8::new(0),
        }))
    }
}

fn equation(node: &Node) -> Result<&str> {
    let attribute = node.attr("equation").ok_or_else(|| {
        EpError::KernelFailed(
            "Einsum: missing required string attribute `equation`. HOW: export the opset-12 node \
             with its ONNX equation attribute."
                .into(),
        )
    })?;
    attribute.as_str().ok_or_else(|| {
        EpError::KernelFailed(
            "Einsum: attribute `equation` must be valid UTF-8 STRING data. HOW: encode an ASCII \
             opset-12 einsum equation such as `ik,kj->ij`."
                .into(),
        )
    })
}

fn execution_mode() -> Result<ExecutionMode> {
    match std::env::var(EINSUM_MODE_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("oracle") => Ok(ExecutionMode::Oracle),
        Ok(value) if value.eq_ignore_ascii_case("optimized") || value.trim().is_empty() => {
            Ok(ExecutionMode::Optimized)
        }
        Ok(value) => Err(EpError::KernelFailed(format!(
            "Einsum: {EINSUM_MODE_ENV}={value:?} is invalid. HOW: use `optimized` (default) or \
             `oracle` for a high-precision correctness diagnostic."
        ))),
        Err(std::env::VarError::NotPresent) => Ok(ExecutionMode::Optimized),
        Err(std::env::VarError::NotUnicode(_)) => Err(EpError::KernelFailed(format!(
            "Einsum: {EINSUM_MODE_ENV} is not valid UTF-8. HOW: unset it or use `optimized` or \
             `oracle`."
        ))),
    }
}

/// Claim-time capability check shared with [`crate::CpuExecutionProvider`].
///
/// Returning the planner's structured rejection before ORT compiles the node
/// lets another CPU provider take legal but not-yet-native general
/// contractions instead of failing session creation after assignment.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let equation = match equation(node) {
        Ok(equation) => equation,
        Err(error) => return Some(error.to_string()),
    };
    if shapes.len() != input_dtypes.len() {
        return Some(format!(
            "Einsum `{equation}` received {} input shape(s) but {} input dtype(s); \
             HOW: finish graph type/shape inference before EP placement",
            shapes.len(),
            input_dtypes.len()
        ));
    }
    if shapes.is_empty() {
        return Some(format!(
            "Einsum `{equation}` has no inputs; ONNX Einsum requires at least one operand"
        ));
    }
    let inputs: Vec<_> = shapes
        .iter()
        .zip(input_dtypes)
        .map(|(shape, &dtype)| EinsumInput::new(dtype, shape))
        .collect();
    match EinsumPlan::build(equation, &inputs) {
        Ok(plan) => {
            if let Some((index, dtype)) = input_dtypes
                .iter()
                .copied()
                .enumerate()
                .find(|(_, dtype)| !matches!(dtype, DataType::Float32 | DataType::Float16))
            {
                return Some(format!(
                    "Einsum `{equation}` input #{index} has canonical dtype {dtype:?}, but the \
                     native CPU kernel supports only Float32 and Float16. HOW: cast every operand \
                     to Float32 or Float16, or use another execution provider."
                ));
            }
            match plan.classification() {
                EinsumClassification::Unsupported(reason) => Some(format!(
                    "Einsum `{}` is valid but not a native CPU lowering: {reason}. Supported \
                     native classes are view/diagonal, reduction or elementwise product, and \
                     binary GEMM/BMM-compatible contractions.",
                    plan.equation()
                )),
                _ => None,
            }
        }
        Err(error) => Some(format!(
            "Einsum canonical planning rejected `{equation}`: {error}"
        )),
    }
}

impl Kernel for EinsumKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.matmul.set_constant_inputs(constant_inputs);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.validate_execution(inputs, outputs)?;
        match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(permutation)
            | EinsumClassification::DiagonalView(permutation) => {
                self.record_route(1);
                self.execute_view_copy(inputs, outputs, permutation)
            }
            EinsumClassification::ReductionOrElementwise(reduction) => {
                self.record_route(if self.mode == ExecutionMode::Oracle {
                    4
                } else {
                    2
                });
                self.execute_reduction(inputs, outputs, reduction)
            }
            EinsumClassification::Gemm(gemm) if self.mode == ExecutionMode::Optimized => {
                self.execute_gemm(inputs, outputs, gemm)
            }
            EinsumClassification::Gemm(_) => {
                self.record_route(4);
                self.execute_oracle(inputs, outputs)
            }
            EinsumClassification::Unsupported(reason) => Err(EpError::KernelFailed(format!(
                "Einsum equation `{}` reached execution with an unsupported canonical plan: \
                 {reason}",
                self.plan.equation()
            ))),
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn may_produce_views(&self) -> bool {
        matches!(
            self.plan.classification(),
            EinsumClassification::ViewOnlyPermutation(_) | EinsumClassification::DiagonalView(_)
        )
    }

    fn view_outputs(
        &self,
        inputs: &[TensorView],
        output_shapes: &[Vec<usize>],
        num_outputs: usize,
    ) -> Option<Vec<ViewOutput>> {
        if num_outputs != 1 || output_shapes.len() != 1 {
            return None;
        }
        let dtype = inputs.first()?.dtype;
        if !matches!(dtype, DataType::Float32 | DataType::Float16)
            || inputs.iter().any(|input| input.dtype != dtype)
        {
            return None;
        }
        let permutation = match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(permutation)
            | EinsumClassification::DiagonalView(permutation) => permutation,
            _ => return None,
        };
        let input = inputs.get(permutation.input())?;
        if input.dtype.byte_size() == 0 {
            return None;
        }
        let shapes: Vec<_> = inputs.iter().map(|input| input.shape).collect();
        let resolved = self.plan.resolve_concrete_output_shape(&shapes).ok()?;
        if resolved != output_shapes[0] {
            return None;
        }
        let layout = permutation_layout(
            input,
            &self.plan.operands()[permutation.input()],
            permutation,
        )
        .ok()?;
        Some(vec![ViewOutput {
            input_index: permutation.input(),
            shape: layout.shape,
            strides: layout.strides,
            byte_offset: input.byte_offset,
        }])
    }

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
    }
}

impl EinsumKernel {
    fn validate_execution(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(
            "Einsum",
            inputs,
            outputs,
            self.plan.operands().len(),
            self.plan.operands().len(),
            1,
        )?;
        for (index, input) in inputs.iter().enumerate() {
            input.validate()?;
            if !input.device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} is on {:?}; the native CPU kernel requires \
                     host-accessible inputs. HOW: place or copy the tensor to CPU before this node.",
                    self.plan.equation(),
                    input.device
                )));
            }
            if !matches!(input.dtype, DataType::Float32 | DataType::Float16) {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} has unsupported dtype {:?}; expected Float32 or \
                     Float16. BFloat16 is not in the canonical ONNX opset-12 Einsum contract. \
                     HOW: cast every operand to Float32 or Float16 before this node.",
                    self.plan.equation(),
                    input.dtype
                )));
            }
            if input.dtype != inputs[0].dtype {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} has dtype {:?}, but input #0 has {:?}; ONNX \
                     Einsum operands must share one dtype",
                    self.plan.equation(),
                    input.dtype,
                    inputs[0].dtype
                )));
            }
        }
        outputs[0].validate()?;
        if !outputs[0].device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output is on {:?}; the native CPU kernel requires a host-accessible \
                 output",
                self.plan.equation(),
                outputs[0].device
            )));
        }
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output dtype {:?} must match input dtype {:?}",
                self.plan.equation(),
                outputs[0].dtype,
                inputs[0].dtype
            )));
        }
        let shapes: Vec<_> = inputs.iter().map(|input| input.shape).collect();
        let expected = self
            .plan
            .resolve_concrete_output_shape(&shapes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` runtime shape validation failed: {error}",
                    self.plan.equation()
                ))
            })?;
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output shape {:?} does not match canonical resolved shape \
                 {expected:?}",
                self.plan.equation(),
                outputs[0].shape
            )));
        }
        Ok(())
    }

    fn execute_view_copy(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        permutation: &EinsumPermutationPlan,
    ) -> Result<()> {
        let input = &inputs[permutation.input()];
        let layout = permutation_layout(
            input,
            &self.plan.operands()[permutation.input()],
            permutation,
        )?;
        let view = TensorView::new(
            input.data,
            input.dtype,
            &layout.shape,
            &layout.strides,
            input.device,
        )
        .with_byte_offset(input.byte_offset);
        let dense = to_dense_bytes(&view)?;
        write_dense_bytes(&mut outputs[0], &dense)
    }

    fn execute_reduction(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        reduction: &EinsumReductionPlan,
    ) -> Result<()> {
        self.execute_generic(
            inputs,
            outputs,
            reduction.iteration_axes(),
            reduction.output_rank(),
            reduction.operand_axis_mappings(),
            self.mode == ExecutionMode::Oracle,
        )
    }

    fn execute_oracle(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut iteration_axes = self.plan.output_axes().to_vec();
        iteration_axes.extend_from_slice(self.plan.reduction_axes());
        let output_rank = self.plan.output_axes().len();
        let mappings: Vec<Vec<usize>> = self
            .plan
            .operands()
            .iter()
            .map(|operand| {
                operand
                    .unique_axes()
                    .iter()
                    .map(|operand_axis| {
                        iteration_axes
                            .iter()
                            .position(|axis| axis == &operand_axis.axis())
                            .expect("validated supported plan maps every operand axis")
                    })
                    .collect()
            })
            .collect();
        self.execute_generic(
            inputs,
            outputs,
            &iteration_axes,
            output_rank,
            &mappings,
            true,
        )
    }

    fn execute_generic(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        iteration_axes: &[EinsumAxis],
        output_rank: usize,
        mappings: &[Vec<usize>],
        high_precision: bool,
    ) -> Result<()> {
        let iteration_shape = axes_shape(&self.plan, iteration_axes)?;
        let output_shape = &iteration_shape[..output_rank];
        let reduction_shape = &iteration_shape[output_rank..];
        let output_len = checked_numel("output", output_shape)?;
        let reduction_len = checked_numel("reduction", reduction_shape)?;

        let layouts: Vec<_> = inputs
            .iter()
            .zip(self.plan.operands())
            .map(|(input, operand)| unique_operand_layout(input, operand))
            .collect::<Result<_>>()?;
        let views: Vec<_> = inputs
            .iter()
            .zip(&layouts)
            .map(|(input, layout)| {
                TensorView::new(
                    input.data,
                    input.dtype,
                    &layout.shape,
                    &layout.strides,
                    input.device,
                )
                .with_byte_offset(input.byte_offset)
            })
            .collect();
        let dense: Vec<Cow<'_, [f32]>> = views
            .iter()
            .map(|view| to_dense_f32_widen("Einsum", view))
            .collect::<Result<_>>()?;
        let operand_iteration_strides: Vec<Vec<usize>> = layouts
            .iter()
            .zip(mappings)
            .map(|(layout, mapping)| {
                let dense_strides = compute_contiguous_strides(&layout.shape);
                let mut strides = vec![0usize; iteration_axes.len()];
                for (unique_axis, &iteration_axis) in mapping.iter().enumerate() {
                    let iter_extent = iteration_shape[iteration_axis];
                    strides[iteration_axis] = if layout.shape[unique_axis] == 1 && iter_extent != 1
                    {
                        0
                    } else {
                        usize::try_from(dense_strides[unique_axis]).map_err(|_| {
                            EpError::KernelFailed(format!(
                                "Einsum `{}` produced a negative dense stride for operand axis \
                                 {unique_axis}",
                                self.plan.equation()
                            ))
                        })?
                    };
                }
                Ok(strides)
            })
            .collect::<Result<_>>()?;

        let mut scratch = self.scratch.borrow_mut();
        resize_f32(&mut scratch.f32_output, output_len)?;
        scratch.f32_output.fill(0.0);
        let identity_mappings = mappings
            .iter()
            .all(|mapping| mapping.iter().copied().eq(0..iteration_axes.len()));
        let aligned_dense = layouts
            .iter()
            .zip(&dense)
            .all(|(layout, data)| layout.shape == iteration_shape && data.len() == output_len);
        if !high_precision && reduction_len == 1 && identity_mappings && aligned_dense {
            const PARALLEL_ELEMENTWISE_MIN_ELEMS: usize = 64 * 1024;
            let evaluate = |index: usize| {
                dense
                    .iter()
                    .fold(1.0f32, |product, operand| product * operand[index])
            };
            if output_len >= PARALLEL_ELEMENTWISE_MIN_ELEMS {
                scratch
                    .f32_output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(index, output)| *output = evaluate(index));
            } else {
                scratch
                    .f32_output
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, output)| *output = evaluate(index));
            }
            return write_dense_f32_narrow("Einsum", &mut outputs[0], &scratch.f32_output);
        }
        if !high_precision
            && dense.len() == 1
            && reduction_len != 0
            && identity_mappings
            && layouts[0].shape == iteration_shape
            && dense[0].len() == output_len.saturating_mul(reduction_len)
        {
            const PARALLEL_REDUCTION_MIN_ELEMS: usize = 64 * 1024;
            let data = dense[0].as_ref();
            let reduce_one = |(output, values): (&mut f32, &[f32])| {
                *output = values.iter().copied().sum();
            };
            if data.len() >= PARALLEL_REDUCTION_MIN_ELEMS && output_len > 1 {
                scratch
                    .f32_output
                    .par_iter_mut()
                    .zip(data.par_chunks(reduction_len))
                    .for_each(reduce_one);
            } else {
                scratch
                    .f32_output
                    .iter_mut()
                    .zip(data.chunks(reduction_len))
                    .for_each(reduce_one);
            }
            return write_dense_f32_narrow("Einsum", &mut outputs[0], &scratch.f32_output);
        }
        if output_len != 0 && reduction_len != 0 {
            let mut output_index = vec![0usize; output_rank];
            for output_offset in 0..output_len {
                let mut reduction_index = vec![0usize; reduction_shape.len()];
                let mut first = true;
                let mut sum_f32 = 0.0f32;
                let mut sum_f64 = 0.0f64;
                while first || next_index(reduction_shape, &mut reduction_index) {
                    first = false;
                    let mut product_f32 = 1.0f32;
                    let mut product_f64 = 1.0f64;
                    for ((data, strides), _operand) in dense
                        .iter()
                        .zip(&operand_iteration_strides)
                        .zip(self.plan.operands())
                    {
                        let mut offset = 0usize;
                        for axis in 0..iteration_axes.len() {
                            let index = if axis < output_rank {
                                output_index[axis]
                            } else {
                                reduction_index[axis - output_rank]
                            };
                            offset = offset
                                .checked_add(index.checked_mul(strides[axis]).ok_or_else(|| {
                                    geometry_overflow(self.plan.equation(), "operand offset")
                                })?)
                                .ok_or_else(|| {
                                    geometry_overflow(self.plan.equation(), "operand offset")
                                })?;
                        }
                        let value = *data.get(offset).ok_or_else(|| {
                            EpError::KernelFailed(format!(
                                "Einsum `{}` canonical operand offset {offset} exceeded a dense \
                                 operand with {} element(s)",
                                self.plan.equation(),
                                data.len()
                            ))
                        })?;
                        product_f32 *= value;
                        product_f64 *= f64::from(value);
                    }
                    if high_precision {
                        sum_f64 += product_f64;
                    } else {
                        sum_f32 += product_f32;
                    }
                    if reduction_shape.is_empty() {
                        break;
                    }
                }
                scratch.f32_output[output_offset] = if high_precision {
                    sum_f64 as f32
                } else {
                    sum_f32
                };
                if output_offset + 1 < output_len {
                    let advanced = next_index(output_shape, &mut output_index);
                    debug_assert!(advanced);
                }
            }
        }
        write_dense_f32_narrow("Einsum", &mut outputs[0], &scratch.f32_output)
    }

    fn execute_gemm(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        gemm: &EinsumContractionPlan,
    ) -> Result<()> {
        let geometry = self
            .plan
            .resolve_concrete_gemm_geometry(
                &inputs.iter().map(|input| input.shape).collect::<Vec<_>>(),
            )
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` could not resolve GEMM geometry: {error}",
                    self.plan.equation()
                ))
            })?
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` canonical classification changed before GEMM execution",
                    self.plan.equation()
                ))
            })?;
        let left_ordered =
            ordered_operand_layout(&inputs[0], &self.plan.operands()[0], gemm.left_axis_order())?;
        let right_ordered = ordered_operand_layout(
            &inputs[1],
            &self.plan.operands()[1],
            gemm.right_axis_order(),
        )?;
        let left = collapse_operand_layout(
            &left_ordered,
            gemm.batch_axes().len(),
            gemm.left_free_axes().len(),
            gemm.contract_axes().len(),
        );
        let right = collapse_operand_layout(
            &right_ordered,
            gemm.batch_axes().len(),
            gemm.contract_axes().len(),
            gemm.right_free_axes().len(),
        );
        let output = collapsed_output_layout(&self.plan, gemm, &outputs[0]);

        let output_aliases_input = inputs
            .iter()
            .any(|input| views_may_overlap(input, &outputs[0]));
        if !output_aliases_input
            && let (Some(left), Some(right), Some(output)) = (left, right, output)
        {
            self.record_route(3);
            let left_view = TensorView::new(
                inputs[0].data,
                inputs[0].dtype,
                &left.shape,
                &left.strides,
                inputs[0].device,
            )
            .with_byte_offset(inputs[0].byte_offset);
            let right_view = TensorView::new(
                inputs[1].data,
                inputs[1].dtype,
                &right.shape,
                &right.strides,
                inputs[1].device,
            )
            .with_byte_offset(inputs[1].byte_offset);
            let output_view = TensorMut::new(
                outputs[0].data,
                outputs[0].dtype,
                &output.shape,
                &output.strides,
                outputs[0].device,
            )
            .with_byte_offset(outputs[0].byte_offset);
            return self
                .matmul
                .execute(&[left_view, right_view], &mut [output_view]);
        }

        self.record_route(5);
        let left_view = TensorView::new(
            inputs[0].data,
            inputs[0].dtype,
            &left_ordered.shape,
            &left_ordered.strides,
            inputs[0].device,
        )
        .with_byte_offset(inputs[0].byte_offset);
        let right_view = TensorView::new(
            inputs[1].data,
            inputs[1].dtype,
            &right_ordered.shape,
            &right_ordered.strides,
            inputs[1].device,
        )
        .with_byte_offset(inputs[1].byte_offset);
        let left_dense = to_dense_f32_widen("Einsum", &left_view)?;
        let right_dense = to_dense_f32_widen("Einsum", &right_view)?;
        let batch_rank = gemm.batch_axes().len();
        let left_shape = flattened_gemm_shape(
            &left_ordered.shape[..batch_rank],
            geometry.m(),
            geometry.k(),
        );
        let right_shape = flattened_gemm_shape(
            &right_ordered.shape[..batch_rank],
            geometry.k(),
            geometry.n(),
        );
        let left_strides = compute_contiguous_strides(&left_shape);
        let right_strides = compute_contiguous_strides(&right_shape);
        let left_f32 = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(left_dense.as_ptr().cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let right_f32 = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(right_dense.as_ptr().cast()),
            DataType::Float32,
            &right_shape,
            &right_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let canonical_shape =
            flattened_gemm_shape(geometry.batch_shape(), geometry.m(), geometry.n());
        let canonical_strides = compute_contiguous_strides(&canonical_shape);
        let canonical_len = checked_numel("GEMM output", &canonical_shape)?;
        let mut scratch = self.scratch.borrow_mut();
        resize_f32(&mut scratch.f32_output, canonical_len)?;
        let canonical_output = TensorMut::new(
            onnx_runtime_ep_api::DevicePtrMut(scratch.f32_output.as_mut_ptr().cast()),
            DataType::Float32,
            &canonical_shape,
            &canonical_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        self.matmul
            .execute(&[left_f32, right_f32], &mut [canonical_output])?;
        write_canonical_output(&self.plan, gemm, &scratch.f32_output, &mut outputs[0])
    }

    #[cfg(test)]
    fn record_route(&self, route: u8) {
        self.last_route
            .store(route, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(not(test))]
    fn record_route(&self, _route: u8) {}
}

#[derive(Clone, Debug)]
struct Layout {
    shape: Vec<usize>,
    strides: Vec<i64>,
}

fn unique_operand_layout(input: &TensorView, operand: &EinsumOperandPlan) -> Result<Layout> {
    let mut shape = Vec::with_capacity(operand.unique_axes().len());
    let mut strides = Vec::with_capacity(operand.unique_axes().len());
    for axis in operand.unique_axes() {
        let &first = axis.input_axes().first().ok_or_else(|| {
            EpError::KernelFailed("Einsum canonical operand axis has no physical axis".into())
        })?;
        shape.push(input.shape[first]);
        let stride = axis.input_axes().iter().try_fold(0i64, |sum, &physical| {
            sum.checked_add(input.strides[physical]).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum diagonal stride overflowed for input #{} axis {physical}",
                    operand.input()
                ))
            })
        })?;
        strides.push(stride);
    }
    Ok(Layout { shape, strides })
}

fn permutation_layout(
    input: &TensorView,
    operand: &EinsumOperandPlan,
    permutation: &EinsumPermutationPlan,
) -> Result<Layout> {
    let unique = unique_operand_layout(input, operand)?;
    let mut shape = Vec::with_capacity(permutation.output_to_operand_axis().len());
    let mut strides = Vec::with_capacity(permutation.output_to_operand_axis().len());
    for &axis in permutation.output_to_operand_axis() {
        shape.push(*unique.shape.get(axis).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{}` permutation references missing operand axis {axis}",
                permutation.input()
            ))
        })?);
        strides.push(unique.strides[axis]);
    }
    Ok(Layout { shape, strides })
}

fn ordered_operand_layout(
    input: &TensorView,
    operand: &EinsumOperandPlan,
    order: &[Option<usize>],
) -> Result<Layout> {
    let unique = unique_operand_layout(input, operand)?;
    let mut shape = Vec::with_capacity(order.len());
    let mut strides = Vec::with_capacity(order.len());
    for entry in order {
        match entry {
            Some(axis) => {
                shape.push(*unique.shape.get(*axis).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Einsum input #{} GEMM order references missing unique axis {axis}",
                        operand.input()
                    ))
                })?);
                strides.push(unique.strides[*axis]);
            }
            None => {
                shape.push(1);
                strides.push(0);
            }
        }
    }
    Ok(Layout { shape, strides })
}

fn collapse_operand_layout(
    ordered: &Layout,
    batch_rank: usize,
    rows_rank: usize,
    columns_rank: usize,
) -> Option<Layout> {
    if ordered.shape.len() != batch_rank + rows_rank + columns_rank {
        return None;
    }
    let (rows, row_stride) = collapse_group(
        &ordered.shape[batch_rank..batch_rank + rows_rank],
        &ordered.strides[batch_rank..batch_rank + rows_rank],
    )?;
    let (columns, column_stride) = collapse_group(
        &ordered.shape[batch_rank + rows_rank..],
        &ordered.strides[batch_rank + rows_rank..],
    )?;
    let mut shape = ordered.shape[..batch_rank].to_vec();
    shape.extend([rows, columns]);
    let mut strides = ordered.strides[..batch_rank].to_vec();
    strides.extend([row_stride, column_stride]);
    Some(Layout { shape, strides })
}

fn collapse_group(shape: &[usize], strides: &[i64]) -> Option<(usize, i64)> {
    if shape.is_empty() {
        return Some((1, 0));
    }
    let extent = shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))?;
    if extent == 0 {
        return Some((0, 1));
    }
    let mut base_stride = None;
    let mut inner_extent = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim <= 1 {
            continue;
        }
        match base_stride {
            None => base_stride = Some(stride),
            Some(base) => {
                let expected = i128::from(base) * i128::try_from(inner_extent).ok()?;
                if expected != i128::from(stride) {
                    return None;
                }
            }
        }
        inner_extent = inner_extent.checked_mul(dim)?;
    }
    Some((extent, base_stride.unwrap_or(0)))
}

fn collapsed_output_layout(
    plan: &EinsumPlan,
    gemm: &EinsumContractionPlan,
    output: &TensorMut,
) -> Option<Layout> {
    let canonical_shape = canonical_output_shape(plan, gemm).ok()?;
    let canonical_strides = canonical_output_strides(plan, gemm, output).ok()?;
    collapse_operand_layout(
        &Layout {
            shape: canonical_shape,
            strides: canonical_strides,
        },
        gemm.batch_axes().len(),
        gemm.left_free_axes().len(),
        gemm.right_free_axes().len(),
    )
}

fn canonical_output_shape(plan: &EinsumPlan, gemm: &EinsumContractionPlan) -> Result<Vec<usize>> {
    let mut axes = Vec::new();
    axes.extend_from_slice(gemm.batch_axes());
    axes.extend_from_slice(gemm.left_free_axes());
    axes.extend_from_slice(gemm.right_free_axes());
    axes_shape(plan, &axes)
}

fn canonical_output_strides(
    plan: &EinsumPlan,
    gemm: &EinsumContractionPlan,
    output: &TensorMut,
) -> Result<Vec<i64>> {
    if gemm.output_permutation().len() != output.shape.len() {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{}` output permutation rank {} does not match output rank {}",
            plan.equation(),
            gemm.output_permutation().len(),
            output.shape.len()
        )));
    }
    let mut canonical = vec![None; gemm.output_permutation().len()];
    for (requested, &canonical_axis) in gemm.output_permutation().iter().enumerate() {
        let slot = canonical.get_mut(canonical_axis).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{}` output permutation references canonical axis {canonical_axis}",
                plan.equation()
            ))
        })?;
        if slot.replace(output.strides[requested]).is_some() {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output permutation repeats canonical axis {canonical_axis}",
                plan.equation()
            )));
        }
    }
    canonical
        .into_iter()
        .map(|stride| {
            stride.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` output permutation omitted a canonical axis",
                    plan.equation()
                ))
            })
        })
        .collect()
}

fn write_canonical_output(
    plan: &EinsumPlan,
    gemm: &EinsumContractionPlan,
    data: &[f32],
    output: &mut TensorMut,
) -> Result<()> {
    let shape = canonical_output_shape(plan, gemm)?;
    let strides = canonical_output_strides(plan, gemm, output)?;
    let mut canonical = TensorMut::new(output.data, output.dtype, &shape, &strides, output.device)
        .with_byte_offset(output.byte_offset);
    write_dense_f32_narrow("Einsum", &mut canonical, data)
}

fn flattened_gemm_shape(batch_shape: &[usize], rows: usize, columns: usize) -> Vec<usize> {
    let mut shape = batch_shape.to_vec();
    shape.extend([rows, columns]);
    shape
}

fn axes_shape(plan: &EinsumPlan, axes: &[EinsumAxis]) -> Result<Vec<usize>> {
    axes.iter()
        .map(|axis| {
            plan.logical_axes()
                .iter()
                .find(|logical| logical.axis() == *axis)
                .and_then(|logical| logical.dimension().as_static())
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Einsum `{}` execution plan has no concrete extent for {axis}",
                        plan.equation()
                    ))
                })
        })
        .collect()
}

fn checked_numel(label: &str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))
        .ok_or_else(|| EpError::KernelFailed(format!("Einsum {label} element count overflowed")))
}

fn geometry_overflow(equation: &str, target: &str) -> EpError {
    EpError::KernelFailed(format!(
        "Einsum `{equation}` {target} overflowed usize; HOW: use smaller concrete dimensions"
    ))
}

fn gemm_flops(gemm: &EinsumContractionPlan) -> Option<u64> {
    let geometry = gemm.geometry();
    [
        geometry.batch().as_static()?,
        geometry.m().as_static()?,
        geometry.k().as_static()?,
        geometry.n().as_static()?,
        2,
    ]
    .into_iter()
    .try_fold(1u64, |value, factor| {
        value.checked_mul(u64::try_from(factor).ok()?)
    })
}

fn resize_f32(buffer: &mut Vec<f32>, len: usize) -> Result<()> {
    if len > buffer.len() {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum could not reserve {} bytes of bounded Float32 workspace: {error}",
                    len.saturating_mul(std::mem::size_of::<f32>())
                ))
            })?;
    }
    buffer.resize(len, 0.0);
    Ok(())
}

fn views_may_overlap(input: &TensorView, output: &TensorMut) -> bool {
    fn byte_range(
        base: usize,
        byte_offset: usize,
        shape: &[usize],
        strides: &[i64],
        element_size: usize,
    ) -> Option<(i128, i128)> {
        if checked_numel("alias-check", shape).ok()? == 0 {
            return Some((0, 0));
        }
        let (minimum, maximum) = crate::strided::addressed_elem_range(shape, strides);
        let origin = i128::try_from(base)
            .ok()?
            .checked_add(i128::try_from(byte_offset).ok()?)?;
        let element_size = i128::try_from(element_size).ok()?;
        let start = origin.checked_add(i128::from(minimum).checked_mul(element_size)?)?;
        let end = origin
            .checked_add(i128::from(maximum).checked_mul(element_size)?)?
            .checked_add(element_size)?;
        Some((start, end))
    }

    let element_size = input.dtype.byte_size();
    if element_size == 0 || output.dtype.byte_size() != element_size {
        return true;
    }
    let Some(input_range) = byte_range(
        input.data.0 as usize,
        input.byte_offset,
        input.shape,
        input.strides,
        element_size,
    ) else {
        return true;
    };
    let Some(output_range) = byte_range(
        output.data.0 as usize,
        output.byte_offset,
        output.shape,
        output.strides,
        element_size,
    ) else {
        return true;
    };
    input_range.0 < output_range.1 && output_range.0 < input_range.1
}

/// Current reusable Float32 scratch capacity held by an Einsum kernel.
///
/// Used by the benchmark to report steady-state workspace rather than infer it
/// from tensor shapes.
#[doc(hidden)]
pub fn benchmark_scratch_capacity_bytes(kernel: &dyn Kernel) -> Option<usize> {
    kernel
        .as_any()
        .downcast_ref::<EinsumKernel>()
        .map(|kernel| {
            kernel
                .scratch
                .borrow()
                .f32_output
                .capacity()
                .saturating_mul(std::mem::size_of::<f32>())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::abi::OrtGraphView;
    use onnx_runtime_ep_api::{DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::{Attribute, DeviceId, Dim, FrozenGraph, Graph, NodeId, static_shape};

    fn kernel(equation: &str, shapes: &[Vec<usize>], mode: ExecutionMode) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(equation.as_bytes().to_vec()),
        );
        let input_shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
        let plan = EinsumPlan::build_for_shapes(equation, &input_shape_refs).unwrap();
        Box::new(EinsumKernel {
            flops: None,
            plan,
            matmul: MatMulKernel::default(),
            scratch: RefCell::new(EinsumScratch::default()),
            mode,
            last_route: std::sync::atomic::AtomicU8::new(0),
        })
    }

    fn route(kernel: &dyn Kernel) -> u8 {
        kernel
            .as_any()
            .downcast_ref::<EinsumKernel>()
            .unwrap()
            .last_route
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: got {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn permutation_and_diagonal_are_zero_copy_views() {
        let permutation = kernel("abc->bca", &[vec![2, 3, 4]], ExecutionMode::Optimized);
        let input = Owned::f32(&[2, 3, 4], &(0..24).map(|x| x as f32).collect::<Vec<_>>());
        let specs = permutation
            .view_outputs(&[input.view()], &[vec![3, 4, 2]], 1)
            .expect("permutation must be a view");
        assert_eq!(specs[0].shape, [3, 4, 2]);
        assert_eq!(specs[0].strides, [4, 1, 12]);

        let diagonal = kernel("ii->i", &[vec![3, 3]], ExecutionMode::Optimized);
        let matrix = Owned::f32(&[3, 3], &(0..9).map(|x| x as f32).collect::<Vec<_>>());
        let specs = diagonal
            .view_outputs(&[matrix.view()], &[vec![3]], 1)
            .expect("diagonal must be a view");
        assert_eq!(specs[0].strides, [4]);

        let scalar = kernel("->", &[vec![]], ExecutionMode::Optimized);
        let value = Owned::f32(&[], &[7.5]);
        let specs = scalar
            .view_outputs(&[value.view()], &[vec![]], 1)
            .expect("rank-0 identity must be a view");
        assert!(specs[0].shape.is_empty());
        assert!(specs[0].strides.is_empty());
    }

    #[test]
    fn reduction_elementwise_and_outer_product_follow_plan_mappings() {
        let reduce = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2]);
        reduce.execute(&[x.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), [6., 15.]);
        assert_eq!(route(&*reduce), 2);

        let outer = kernel("i,j->ij", &[vec![2], vec![3]], ExecutionMode::Optimized);
        let left = Owned::f32(&[2], &[2., 3.]);
        let right = Owned::f32(&[3], &[5., 7., 11.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        outer
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [10., 14., 22., 15., 21., 33.]);
    }

    #[test]
    fn gemm_and_transpose_required_bmm_use_matmul_lowering() {
        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5.]);
        assert_eq!(route(&*gemm), 3);

        let bmm = kernel(
            "bik,bjk->bij",
            &[vec![2, 2, 3], vec![2, 4, 3]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(
            &[2, 2, 3],
            &[1., 2., 3., 4., 5., 6., 1., 0., 1., 2., 1., 0.],
        );
        let b = Owned::f32(
            &[2, 4, 3],
            &[
                1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 2., 3., 3., 2., 1., 2., 0., 1.,
                0., 1., 2.,
            ],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 4]);
        bmm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            [
                1., 2., 3., 6., 4., 5., 6., 15., 4., 4., 3., 2., 4., 8., 4., 1.
            ]
        );
        assert_eq!(route(&*bmm), 3);
    }

    #[test]
    fn broadcast_bmm_rank_one_and_zero_dimensions_are_supported() {
        let bmm = kernel(
            "mk,...kn->...mn",
            &[vec![2, 3], vec![2, 3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(
            &[2, 3, 2],
            &[1., 0., 0., 1., 1., 0., 2., 1., 1., 0., 0., 2.],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        bmm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5., 4., 7., 13., 16.]);

        let dot = kernel("i,i->", &[vec![3], vec![3]], ExecutionMode::Optimized);
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let y = Owned::f32(&[3], &[4., 5., 6.]);
        let mut scalar = Owned::zeros_f32(&[]);
        dot.execute(&[x.view(), y.view()], &mut [scalar.view_mut()])
            .unwrap();
        assert_eq!(scalar.to_f32(), [32.]);

        let zero = kernel(
            "ik,kj->ij",
            &[vec![0, 3], vec![3, 4]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[0, 3], &[]);
        let b = Owned::f32(&[3, 4], &[1.; 12]);
        let mut out = Owned::zeros_f32(&[0, 4]);
        zero.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert!(out.to_f32().is_empty());
    }

    #[test]
    fn multi_axis_output_permutation_and_diagonal_contraction_are_correct() {
        let multi = kernel(
            "abxy,xycd->dcab",
            &[vec![2, 1, 2, 2], vec![2, 2, 2, 2]],
            ExecutionMode::Optimized,
        );
        let left = Owned::f32(&[2, 1, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let right = Owned::f32(
            &[2, 2, 2, 2],
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 2, 1]);
        multi
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            [90., 202., 110., 254., 100., 228., 120., 280.]
        );
        assert_eq!(route(&*multi), 5);

        let diagonal = kernel(
            "iik,kj->ij",
            &[vec![2, 2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let left = Owned::f32(
            &[2, 2, 3],
            &[1., 2., 3., 99., 99., 99., 99., 99., 99., 4., 5., 6.],
        );
        let right = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        diagonal
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 5., 10., 11.]);

        // The output permutation forces the materialized fallback while the
        // left operand has an inserted broadcast batch axis. Its materialized
        // shape must retain batch=1 rather than pretending it owns the resolved
        // batch=3 storage.
        let equation = "abxy,...xycd->d...cab";
        let shapes = [vec![2, 1, 2, 2], vec![3, 2, 2, 2, 2]];
        let broadcasted = kernel(equation, &shapes, ExecutionMode::Optimized);
        let oracle = kernel(equation, &shapes, ExecutionMode::Oracle);
        let left = Owned::f32(&shapes[0], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let right = Owned::f32(
            &shapes[1],
            &(1..=48).map(|value| value as f32 / 8.0).collect::<Vec<_>>(),
        );
        let mut actual = Owned::zeros_f32(&[2, 3, 2, 2, 1]);
        let mut expected = Owned::zeros_f32(&[2, 3, 2, 2, 1]);
        broadcasted
            .execute(&[left.view(), right.view()], &mut [actual.view_mut()])
            .unwrap();
        oracle
            .execute(&[left.view(), right.view()], &mut [expected.view_mut()])
            .unwrap();
        assert_close(&actual.to_f32(), &expected.to_f32(), 1e-5);
        assert_eq!(route(&*broadcasted), 5);
    }

    #[test]
    fn float16_and_noncontiguous_inputs_match_expected_values() {
        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros(DataType::Float16, &[2, 2]);
        gemm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_close(&out.to_f16_as_f32(), &[4., 2., 10., 5.], 0.05);

        let reduce = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut reduced = Owned::zeros(DataType::Float16, &[2]);
        reduce
            .execute(&[a.view()], &mut [reduced.view_mut()])
            .unwrap();
        assert_close(&reduced.to_f16_as_f32(), &[6., 15.], 0.05);

        let transpose = kernel("ij->ji", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut transposed = Owned::zeros(DataType::Float16, &[3, 2]);
        transpose
            .execute(&[a.view()], &mut [transposed.view_mut()])
            .unwrap();
        assert_eq!(
            transposed.to_u16_bits(),
            [
                a.to_u16_bits()[0],
                a.to_u16_bits()[3],
                a.to_u16_bits()[1],
                a.to_u16_bits()[4],
                a.to_u16_bits()[2],
                a.to_u16_bits()[5],
            ],
            "view/copy semantics must preserve Float16 payload bits exactly"
        );

        let noncontiguous = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]).with_view(&[2, 3], &[1, 2]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        noncontiguous
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5.]);
    }

    #[test]
    fn compute_fallback_is_alias_safe() {
        let permutation_kernel = kernel("ij->ji", &[vec![2, 2]], ExecutionMode::Optimized);
        let mut tensor = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let shape = tensor.shape.clone();
        let strides = tensor.strides.clone();
        let ptr = tensor.bytes.as_mut_ptr();
        let input = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(ptr.cast_const().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let output_shape = [2usize, 2];
        let output_strides = [2i64, 1];
        let mut output = TensorMut::new(
            DevicePtrMut(ptr.cast()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            DeviceId::cpu(),
        );
        permutation_kernel
            .execute(&[input], std::slice::from_mut(&mut output))
            .unwrap();
        tensor.shape = shape;
        tensor.strides = strides;
        assert_eq!(tensor.to_f32(), [1., 3., 2., 4.]);

        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 2], vec![2, 2]],
            ExecutionMode::Optimized,
        );
        let mut left = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let right = Owned::f32(&[2, 2], &[2., 0., 0., 3.]);
        let left_shape = left.shape.clone();
        let left_strides = left.strides.clone();
        let left_ptr = left.bytes.as_mut_ptr();
        let left_input = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(left_ptr.cast_const().cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            DeviceId::cpu(),
        );
        let mut aliased_output = TensorMut::new(
            DevicePtrMut(left_ptr.cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            DeviceId::cpu(),
        );
        gemm.execute(
            &[left_input, right.view()],
            std::slice::from_mut(&mut aliased_output),
        )
        .unwrap();
        assert_eq!(left.to_f32(), [2., 6., 6., 12.]);
        assert_eq!(route(&*gemm), 5);
    }

    #[test]
    fn oracle_mode_is_high_precision_and_non_vacuously_selected() {
        let optimized = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let oracle_kernel = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Oracle,
        );
        let a = Owned::f32(&[2, 3], &[1e10, 1., -1e10, 3., 4., 5.]);
        let b = Owned::f32(&[3, 2], &[1., 2., 1., 1., 1., 0.]);
        let mut fast = Owned::zeros_f32(&[2, 2]);
        let mut oracle = Owned::zeros_f32(&[2, 2]);
        optimized
            .execute(&[a.view(), b.view()], &mut [fast.view_mut()])
            .unwrap();
        oracle_kernel
            .execute(&[a.view(), b.view()], &mut [oracle.view_mut()])
            .unwrap();
        assert_eq!(route(&*optimized), 3);
        assert_eq!(route(&*oracle_kernel), 4);
        assert_eq!(oracle.to_f32(), [1., 2e10, 12., 10.]);
        assert_close(&fast.to_f32(), &oracle.to_f32(), 1024.0);
    }

    fn einsum_graph(dtype: DataType) -> FrozenGraph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        let left = graph.create_named_value("A", dtype, static_shape([2, 3]));
        let right = graph.create_named_value("B", dtype, static_shape([3, 2]));
        let output = graph.create_named_value("C", dtype, static_shape([2, 2]));
        graph.add_input(left);
        graph.add_input(right);
        let mut node = Node::new(
            NodeId(0),
            "Einsum",
            vec![Some(left), Some(right)],
            vec![output],
        );
        node.attributes
            .insert("equation".into(), Attribute::String(b"ik,kj->ij".to_vec()));
        graph.insert_node(node);
        graph.add_output(output);
        FrozenGraph::build(graph).unwrap()
    }

    #[test]
    fn provider_placement_declines_bfloat16_and_reaches_float16_float32() {
        let provider = crate::CpuExecutionProvider::new();

        for dtype in [DataType::Float32, DataType::Float16] {
            let frozen = einsum_graph(dtype);
            let view = frozen.view();
            let node_index = view.nodes().next().expect("one Einsum node");
            let support = provider.supports_node(&view, node_index, 24);
            assert!(
                support.is_supported(),
                "{dtype:?} Einsum must be reachable through normal provider placement: {support:?}"
            );
            let claims = OrtGraphView::new(&view).query_capabilities(&provider);
            assert_eq!(
                claims.len(),
                1,
                "{dtype:?} Einsum must produce one non-vacuous provider capability"
            );
            let kernel = provider
                .get_kernel(view.node(node_index), &[vec![2, 3], vec![3, 2]], 24)
                .unwrap();
            let (left, right) = match dtype {
                DataType::Float32 => (
                    Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                    Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]),
                ),
                DataType::Float16 => (
                    Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                    Owned::f16(&[3, 2], &[1., 0., 0., 1., 1., 0.]),
                ),
                _ => unreachable!(),
            };
            let mut output = Owned::zeros(dtype, &[2, 2]);
            kernel
                .execute(&[left.view(), right.view()], &mut [output.view_mut()])
                .unwrap();
            let actual = match dtype {
                DataType::Float32 => output.to_f32(),
                DataType::Float16 => output.to_f16_as_f32(),
                _ => unreachable!(),
            };
            assert_close(&actual, &[4., 2., 10., 5.], 0.05);
        }

        let frozen = einsum_graph(DataType::BFloat16);
        let view = frozen.view();
        let node_index = view.nodes().next().expect("one Einsum node");
        let support = provider.supports_node(&view, node_index, 24);
        assert!(!support.is_supported());
        let reason = support.reason().expect("BFloat16 decline must explain why");
        assert!(
            reason.contains("unsupported opset-12 dtype BFloat16"),
            "{reason}"
        );
        assert!(
            reason.contains("not part of the canonical Einsum contract"),
            "{reason}"
        );
        assert!(reason.contains("HOW:"), "{reason}");
        assert!(
            OrtGraphView::new(&view)
                .query_capabilities(&provider)
                .is_empty(),
            "BFloat16 Einsum must not reach compilation through provider placement"
        );
    }

    #[test]
    fn unsupported_general_contraction_is_declined_before_kernel_creation() {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(b"ij,jk,kl->il".to_vec()),
        );
        let shapes = [
            static_shape([2, 3]),
            static_shape([3, 4]),
            static_shape([4, 5]),
        ];
        let dtypes = [DataType::Float32; 3];
        let reason = unsupported_reason(&node, &shapes, &dtypes).unwrap();
        assert!(reason.contains("3-input contraction"));

        let provider = crate::CpuExecutionProvider::new();
        let support = provider.supports_op(
            &node,
            12,
            &shapes,
            &dtypes,
            &[
                onnx_runtime_ir::TensorLayout::contiguous(),
                onnx_runtime_ir::TensorLayout::contiguous(),
                onnx_runtime_ir::TensorLayout::contiguous(),
            ],
        );
        assert!(!support.is_supported());
        assert!(support.reason().unwrap().contains("3-input contraction"));
    }

    #[test]
    fn runtime_shape_and_dtype_errors_are_actionable() {
        let shape_kernel = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1.; 6]);
        let b = Owned::f32(&[4, 2], &[1.; 8]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let error = shape_kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("runtime shape validation failed"),
            "unexpected error: {error}"
        );
        assert!(
            error.contains("input #1") || error.contains("label `k`"),
            "error did not identify the rejected operand/axis: {error}"
        );

        let int_shape = vec![Dim::Static(2)];
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
        let reason =
            unsupported_reason(&node, &[int_shape], &[DataType::Int32]).expect("must decline");
        assert!(reason.contains("supports only Float32 and Float16"));

        let direct = kernel("i->i", &[vec![2]], ExecutionMode::Optimized);
        let input = Owned::bf16(&[2], &[1.0, 2.0]);
        assert!(
            direct
                .view_outputs(&[input.view()], &[vec![2]], 1)
                .is_none(),
            "BFloat16 must not bypass kernel dtype validation through a view output"
        );
        let mut output = Owned::zeros(DataType::BFloat16, &[2]);
        let error = direct
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("BFloat16 is not in the canonical ONNX opset-12 Einsum contract"),
            "{error}"
        );
        assert!(error.contains("HOW:"), "{error}");
    }
}
