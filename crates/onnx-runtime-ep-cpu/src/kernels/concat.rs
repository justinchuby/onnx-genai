//! `Concat`: join a list of tensors along one axis (`docs/ORT2.md` §4.4).
//!
//! The executor supplies the single, pre-shaped output allocation. Contiguous
//! inputs are copied directly into disjoint output slabs with bulk memcpy;
//! strided inputs fall back to gathering elements directly into those same
//! slabs, without materializing intermediate buffers.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::elem_size;

/// Concat kernel carrying the raw `axis` attribute (may be negative).
pub struct ConcatKernel {
    axis: i64,
}

/// Factory reading the required `axis` attribute.
pub struct ConcatFactory;

impl KernelFactory for ConcatFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0);
        Ok(Box::new(ConcatKernel { axis }))
    }
}

fn shape_product(op: &str, dims: &[usize], context: &str) -> Result<usize> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{op}: WHAT: shape product overflow while computing {context}. \
                 WHY: the tensor shape is too large to address on this platform. \
                 HOW: use smaller tensor dimensions."
            ))
        })
    })
}

fn strided_offset(flat: usize, shape: &[usize], strides: &[i64]) -> isize {
    let mut remainder = flat;
    let mut offset = 0i64;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim != 0 {
            offset += (remainder % dim) as i64 * stride;
            remainder /= dim;
        }
    }
    offset as isize
}

impl Kernel for ConcatKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() {
            return Err(EpError::KernelFailed(
                "Concat: WHAT: received no inputs. WHY: concatenation requires at least one \
                 tensor. HOW: connect one or more tensors to the Concat node."
                    .into(),
            ));
        }
        if outputs.is_empty() {
            return Err(EpError::KernelFailed(
                "Concat: WHAT: received no output view. WHY: the kernel needs the executor's \
                 pre-shaped output allocation. HOW: allocate and pass one Concat output."
                    .into(),
            ));
        }
        let rank = inputs[0].shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Concat: WHAT: input 0 is scalar (shape []). WHY: a scalar has no axis to \
                 concatenate. HOW: provide rank-1-or-higher inputs."
                    .into(),
            ));
        }

        let resolved_axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if resolved_axis < 0 || resolved_axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Concat: WHAT: axis {} is out of range for input 0 shape {:?} (rank {rank}). \
                 WHY: the axis must identify an existing dimension. \
                 HOW: use an axis in [-{rank}, {rank}).",
                self.axis, inputs[0].shape
            )));
        }
        let axis = resolved_axis as usize;
        let output = &mut outputs[0];
        output.validate().map_err(|err| {
            EpError::KernelFailed(format!(
                "Concat: WHAT: output view {:?} is invalid: {err}. \
                 WHY: Concat writes directly into the executor-provided output allocation. \
                 HOW: provide a valid, pre-shaped contiguous output view.",
                output.shape
            ))
        })?;
        if !output.is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "Concat: WHAT: output shape {:?} has non-contiguous strides {:?}. \
                 WHY: direct slab copies require a contiguous executor-provided output. \
                 HOW: allocate the Concat output in contiguous row-major layout.",
                output.shape, output.strides
            )));
        }

        let esize = elem_size(output.dtype).map_err(|err| {
            EpError::KernelFailed(format!(
                "Concat: WHAT: output dtype {:?} has no supported fixed-width byte layout: {err}. \
                 WHY: byte-agnostic Concat copies whole fixed-width elements. \
                 HOW: use a fixed-width tensor dtype.",
                output.dtype
            ))
        })?;
        let mut expected_shape = inputs[0].shape.to_vec();
        expected_shape[axis] = 0;

        // Validate every input before writing any output bytes, so failures cannot
        // leave a partially-written result.
        for (i, input) in inputs.iter().enumerate() {
            input.validate().map_err(|err| {
                EpError::KernelFailed(format!(
                    "Concat: WHAT: input {i} view {:?} is invalid: {err}. \
                     WHY: Concat must safely address every source element. \
                     HOW: provide a valid tensor view with matching shape/strides.",
                    input.shape
                ))
            })?;
            if input.shape.len() != rank {
                return Err(EpError::KernelFailed(format!(
                    "Concat: WHAT: input {i} shape {:?} has rank {}, but input 0 shape {:?} has \
                     rank {rank}. WHY: all inputs must have the same rank. \
                     HOW: reshape input {i} to rank {rank} before concatenating along axis {axis}.",
                    input.shape,
                    input.shape.len(),
                    inputs[0].shape
                )));
            }
            if input.dtype != output.dtype {
                return Err(EpError::KernelFailed(format!(
                    "Concat: WHAT: input {i} dtype {:?} differs from output dtype {:?}. \
                     WHY: Concat joins tensors of one dtype. \
                     HOW: cast input {i} to {:?} before concatenating along axis {axis}.",
                    input.dtype, output.dtype, output.dtype
                )));
            }
            for dim in 0..rank {
                if dim != axis && input.shape[dim] != inputs[0].shape[dim] {
                    return Err(EpError::KernelFailed(format!(
                        "Concat: WHAT: input {i} shape {:?} is incompatible with input 0 shape \
                         {:?} at dimension {dim} while concatenating axis {axis}. \
                         WHY: every non-concat dimension must match. \
                         HOW: make dimension {dim} equal to {} for input {i}.",
                        input.shape, inputs[0].shape, inputs[0].shape[dim]
                    )));
                }
            }
            expected_shape[axis] = expected_shape[axis]
                .checked_add(input.shape[axis])
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Concat: WHAT: concatenated axis {axis} overflows while adding input {i} \
                         shape {:?}. WHY: the output axis is too large to address. \
                         HOW: reduce the input extents along axis {axis}.",
                        input.shape
                    ))
                })?;
        }

        if output.shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "Concat: WHAT: executor supplied output shape {:?}, but inputs require {:?} for \
                 axis {axis}. WHY: the output must contain every input exactly once. \
                 HOW: allocate the Concat output with shape {:?}.",
                output.shape, expected_shape, expected_shape
            )));
        }

        let outer = shape_product("Concat", &expected_shape[..axis], "outer block count")?;
        let inner_elems =
            shape_product("Concat", &expected_shape[axis + 1..], "inner element count")?;
        let inner_bytes = inner_elems.checked_mul(esize).ok_or_else(|| {
            EpError::KernelFailed(
                "Concat: WHAT: inner byte count overflowed. WHY: the trailing dimensions and \
                 element size exceed addressable memory. HOW: use smaller tensor dimensions."
                    .into(),
            )
        })?;
        let total_elems = shape_product("Concat", &expected_shape, "output element count")?;
        let _total_bytes = total_elems.checked_mul(esize).ok_or_else(|| {
            EpError::KernelFailed(
                "Concat: WHAT: output byte count overflowed. WHY: the output shape and element \
                 size exceed addressable memory. HOW: use smaller tensor dimensions."
                    .into(),
            )
        })?;
        if total_elems == 0 {
            return Ok(());
        }

        let dst = output.data_ptr_mut::<u8>();
        let mut axis_prefix = 0usize;
        for input in inputs {
            let axis_len = input.shape[axis];
            if axis_len == 0 {
                continue;
            }
            let slab_bytes = axis_len.checked_mul(inner_bytes).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Concat: WHAT: input slab byte count overflowed for shape {:?}, axis {axis}. \
                     WHY: the axis extent and inner byte count exceed addressable memory. \
                     HOW: use smaller tensor dimensions.",
                    input.shape
                ))
            })?;

            if input.is_contiguous() {
                let src = input.data_ptr::<u8>();
                for outer_idx in 0..outer {
                    let src_offset = outer_idx * slab_bytes;
                    let dst_offset = (outer_idx * expected_shape[axis] + axis_prefix) * inner_bytes;
                    // SAFETY: validated contiguous input/output views are bounds-gated by the
                    // executor. SSA supplies a distinct output allocation. Each (input, outer)
                    // pair owns a unique destination slab, and this single-threaded kernel
                    // writes every slab exactly once, so ranges neither overlap nor race.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.add(src_offset),
                            dst.add(dst_offset),
                            slab_bytes,
                        );
                    }
                }
            } else {
                // A strided view cannot be memcpy'd as a slab. Gather each logical element
                // directly into its final output position; no dense temporary is allocated.
                let src = input.data_ptr::<u8>();
                for outer_idx in 0..outer {
                    let outer_offset =
                        strided_offset(outer_idx, &input.shape[..axis], &input.strides[..axis]);
                    for axis_idx in 0..axis_len {
                        for inner_idx in 0..inner_elems {
                            let inner_offset = strided_offset(
                                inner_idx,
                                &input.shape[axis + 1..],
                                &input.strides[axis + 1..],
                            );
                            let src_elem = outer_offset
                                + axis_idx as isize * input.strides[axis] as isize
                                + inner_offset;
                            let dst_elem =
                                (outer_idx * expected_shape[axis] + axis_prefix + axis_idx)
                                    * inner_elems
                                    + inner_idx;
                            // SAFETY: the source index is within the validated, executor
                            // bounds-gated strided view; destination logical indices are unique.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    src.offset(src_elem * esize as isize),
                                    dst.add(dst_elem * esize),
                                    esize,
                                );
                            }
                        }
                    }
                }
            }
            axis_prefix += axis_len;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::DataType;

    fn run_result(axis: i64, ins: &[&Owned], out: &mut Owned) -> Result<()> {
        let views: Vec<_> = ins.iter().map(|o| o.view()).collect();
        ConcatKernel { axis }.execute(&views, &mut [out.view_mut()])
    }

    fn run(axis: i64, ins: &[&Owned], out: &mut Owned) {
        run_result(axis, ins, out).unwrap();
    }

    #[test]
    fn concat_axis0_f32() {
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[1, 2], &[5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        run(0, &[&a, &b], &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn concat_middle_axis_f32() {
        let a = Owned::f32(&[2, 1, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 2, 2], &[5., 6., 7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 3, 2]);
        run(1, &[&a, &b], &mut out);
        assert_eq!(
            out.to_f32(),
            vec![1., 2., 5., 6., 7., 8., 3., 4., 9., 10., 11., 12.]
        );
    }

    #[test]
    fn concat_last_axis_three_inputs_u8() {
        let a = Owned::u8(&[2, 1], &[1, 2]);
        let b = Owned::u8(&[2, 2], &[3, 4, 5, 6]);
        let c = Owned::u8(&[2, 1], &[7, 8]);
        let mut out = Owned::zeros(DataType::Uint8, &[2, 4]);
        run(-1, &[&a, &b, &c], &mut out);
        assert_eq!(out.bytes, vec![1, 3, 4, 7, 2, 5, 6, 8]);
    }

    #[test]
    fn concat_int64_dtype_agnostic() {
        let a = Owned::i64(&[2], &[10, 20]);
        let b = Owned::i64(&[3], &[30, 40, 50]);
        let mut out = Owned::zeros(DataType::Int64, &[5]);
        run(0, &[&a, &b], &mut out);
        assert_eq!(out.to_i64(), vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn concat_gathers_non_contiguous_input_without_temporary() {
        // Backing [2,3] row-major, exposed transposed as [3,2] with strides [1,3].
        let transposed = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]).with_view(&[3, 2], &[1, 3]);
        let dense = Owned::f32(&[3, 1], &[7., 8., 9.]);
        let mut out = Owned::zeros_f32(&[3, 3]);
        run(1, &[&transposed, &dense], &mut out);
        assert_eq!(out.to_f32(), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]);
    }

    #[test]
    fn concat_rejects_axis_out_of_bounds_with_actionable_error() {
        let input = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let err = run_result(2, &[&input], &mut out).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("axis 2"));
        assert!(message.contains("shape [2, 2]"));
        assert!(message.contains("WHY:"));
        assert!(message.contains("HOW:"));
    }

    #[test]
    fn concat_rejects_mismatched_non_axis_dimension_before_writing() {
        let a = Owned::f32(&[2, 1, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[3, 1, 2], &[5., 6., 7., 8., 9., 10.]);
        let mut out = Owned::f32(&[2, 2, 2], &[99.; 8]);
        let err = run_result(1, &[&a, &b], &mut out).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("input 1 shape [3, 1, 2]"));
        assert!(message.contains("input 0 shape [2, 1, 2]"));
        assert!(message.contains("axis 1"));
        assert!(message.contains("WHY:"));
        assert!(message.contains("HOW:"));
        assert_eq!(out.to_f32(), vec![99.; 8]);
    }
    #[test]
    fn concat_bf16_preserves_element_bits() {
        let a = Owned::bf16(&[1, 2], &[1., -2.]);
        let b = Owned::bf16(&[1, 2], &[3., 4.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2, 2]);
        run(0, &[&a, &b], &mut out);
        let mut expected = a.to_u16_bits();
        expected.extend(b.to_u16_bits());
        assert_eq!(out.to_u16_bits(), expected);
    }
}
