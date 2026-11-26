//! `Compress` selection along an optional axis.

use std::ops::Range;

use onnx_runtime_ep_api::{
    EpError, HostKernelData, HostKernelInput, HostKernelOutput, Kernel, KernelFactory, Result,
    TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType, Node, checked_numel};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};

fn error(detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("Compress: {detail}"))
}

pub struct CompressKernel {
    axis: Option<i64>,
}

pub struct CompressFactory;

impl KernelFactory for CompressFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CompressKernel {
            axis: node.attr("axis").and_then(Attribute::as_int),
        }))
    }
}

struct Selection {
    shape: Vec<usize>,
    axis_len: usize,
    selected: Vec<usize>,
    outer: usize,
    inner: usize,
    count: usize,
}

impl Selection {
    fn ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        // Avoid walking logical blocks when a zero dimension leaves no data to move.
        let outer = if self.count == 0 { 0 } else { self.outer };
        (0..outer).flat_map(move |outer| {
            self.selected.iter().map(move |&index| {
                let start = (outer * self.axis_len + index) * self.inner;
                start..start + self.inner
            })
        })
    }
}

impl CompressKernel {
    fn selection(&self, shape: &[usize], condition: &[u8]) -> Result<Selection> {
        let numel = |dims: &[usize]| {
            checked_numel(dims).ok_or_else(|| {
                error(format!(
                    "shape {shape:?} overflows addressable element count"
                ))
            })
        };
        let (mut output, axis) = match self.axis {
            Some(requested) => {
                let rank = shape.len();
                let axis = if requested < 0 {
                    requested + rank as i64
                } else {
                    requested
                };
                if !(0..rank as i64).contains(&axis) {
                    return Err(error(format!(
                        "axis {requested} is outside [-{rank}, {rank}); supply an axis within the input rank"
                    )));
                }
                (shape.to_vec(), axis as usize)
            }
            None => (vec![numel(shape)?], 0),
        };
        let axis_len = output[axis];
        let selected: Vec<_> = condition
            .iter()
            .take(axis_len)
            .enumerate()
            .filter_map(|(index, &keep)| (keep != 0).then_some(index))
            .collect();
        output[axis] = selected.len();
        let count = numel(&output)?;
        let (outer, inner) = if count == 0 {
            (0, 0)
        } else {
            numel(shape)?;
            (numel(&output[..axis])?, numel(&output[axis + 1..])?)
        };
        Ok(Selection {
            shape: output,
            axis_len,
            selected,
            outer,
            inner,
            count,
        })
    }
}

impl Kernel for CompressKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Compress", inputs, outputs, 2, 2, 1)?;
        let input = &inputs[0];
        let condition = &inputs[1];
        if condition.dtype != DataType::Bool || condition.shape.len() != 1 {
            return Err(error("condition must be a one-dimensional Bool tensor"));
        }
        if outputs[0].dtype != input.dtype {
            return Err(error("output dtype must match input dtype"));
        }

        let selection = self.selection(input.shape, &to_dense_bytes(condition)?)?;
        if outputs[0].shape != selection.shape {
            return Err(error(
                "output shape does not match the selected condition elements",
            ));
        }

        let element_size = elem_size(input.dtype)?;
        let src = to_dense_bytes(input)?;
        let byte_count = selection
            .count
            .checked_mul(element_size)
            .ok_or_else(|| error("output shape overflows addressable bytes"))?;
        let mut out = Vec::with_capacity(byte_count);
        for range in selection.ranges() {
            out.extend_from_slice(&src[range.start * element_size..range.end * element_size]);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn execute_managed_host(
        &self,
        inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        let [HostKernelInput::Strings(input), condition] = inputs else {
            return Err(error(
                "expected managed tensor(string) data and a one-dimensional Bool condition",
            ));
        };
        let HostKernelInput::Bool {
            shape: [len],
            bytes,
        } = condition
        else {
            return Err(error("condition must be a one-dimensional Bool tensor"));
        };
        if *len != bytes.len() {
            return Err(error(format!(
                "condition shape [{len}] requires {len} Bool bytes, got {}",
                bytes.len()
            )));
        }
        if checked_numel(input.shape) != Some(input.values.len()) {
            return Err(error(format!(
                "input shape {:?} does not match the supplied {} strings; provide one string per element",
                input.shape,
                input.values.len()
            )));
        }
        let selection = self.selection(input.shape, bytes)?;
        let mut strings = Vec::with_capacity(selection.count);
        for range in selection.ranges() {
            strings.extend_from_slice(&input.values[range]);
        }
        Ok(vec![HostKernelOutput {
            shape: selection.shape,
            data: HostKernelData::Strings(strings),
        }])
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn selects_axis_zero() {
        let input = Owned::i32(&[3, 2], &[1, 2, 3, 4, 5, 6]);
        let condition = Owned::bool_(&[3], &[true, false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 2]);
        CompressKernel { axis: Some(0) }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![1, 2, 5, 6]);
    }

    #[test]
    fn selects_negative_axis_with_short_condition() {
        let input = Owned::i32(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let condition = Owned::bool_(&[2], &[false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 1]);
        CompressKernel { axis: Some(-1) }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![2, 5]);
    }

    #[test]
    fn omitting_axis_flattens_before_selection() {
        let input = Owned::i32(&[2, 2], &[1, 2, 3, 4]);
        let condition = Owned::bool_(&[4], &[false, true, false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        CompressKernel { axis: None }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![2, 4]);
    }
    #[test]
    fn compress_bf16_preserves_element_bits() {
        let x = Owned::bf16(&[3], &[1., -2., 3.]);
        let c = Owned::bool_(&[3], &[true, false, true]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2]);
        CompressKernel { axis: Some(0) }
            .execute(&[x.view(), c.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_u16_bits(),
            vec![x.to_u16_bits()[0], x.to_u16_bits()[2]]
        );
    }
}
