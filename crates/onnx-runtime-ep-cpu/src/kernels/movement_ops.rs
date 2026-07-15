//! Pure metadata data-movement ops: `Flatten`, `Squeeze`, `Size`, and `Trilu`
//! (`docs/ORT2.md` §4.4).
//!
//! `Flatten` and `Squeeze` only change a tensor's *shape*, never its row-major
//! element order, and the runtime pre-allocates the output with the target
//! shape (computed by shape inference). Each kernel therefore moves raw element
//! bytes through [`to_dense_bytes`]/[`write_dense_bytes`], serving every
//! fixed-width dtype uniformly. `Squeeze`'s optional `axes` input is consumed
//! upstream when the output shape is built, so the kernel ignores it.
//!
//! `Size` reports the input's total element count as a rank-0 `int64` scalar; it
//! reads only shape metadata and is dtype-agnostic on its input.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

/// Stateless `Flatten` kernel (row-major byte copy into the pre-shaped output).
pub struct FlattenKernel;

/// Factory for [`FlattenKernel`] (the `axis` attribute only affects the output
/// shape, which is resolved upstream).
pub struct FlattenFactory;

impl KernelFactory for FlattenFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FlattenKernel))
    }
}

impl Kernel for FlattenKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Flatten", inputs, outputs, 1, 1, 1)?;
        let bytes = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Stateless `Squeeze` kernel (row-major byte copy into the pre-shaped output).
pub struct SqueezeKernel;

/// Factory for [`SqueezeKernel`] (`axes` from attribute or input 1 only affects
/// the output shape, which is resolved upstream).
pub struct SqueezeFactory;

impl KernelFactory for SqueezeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SqueezeKernel))
    }
}

impl Kernel for SqueezeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // data (+ optional axes input, which is metadata only here).
        check_arity("Squeeze", inputs, outputs, 1, 2, 1)?;
        let bytes = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Stateless `Size` kernel: input element count as a rank-0 `int64` scalar.
pub struct SizeKernel;

/// Factory for [`SizeKernel`] (no attributes).
pub struct SizeFactory;

impl KernelFactory for SizeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SizeKernel))
    }
}

impl Kernel for SizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Size", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "Size: output must be Int64, got {:?}. WHY: ONNX Size yields an int64 scalar. \
                 HOW: allocate the output as Int64.",
                outputs[0].dtype
            )));
        }
        let n = numel(inputs[0].shape) as i64;
        write_dense_bytes(&mut outputs[0], &n.to_le_bytes())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Triangular matrix masking over the final two dimensions.
pub struct TriluKernel {
    upper: bool,
}

pub struct TriluFactory;

impl KernelFactory for TriluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TriluKernel {
            upper: node.attr("upper").and_then(Attribute::as_int).unwrap_or(1) != 0,
        }))
    }
}

impl Kernel for TriluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Trilu", inputs, outputs, 1, 2, 1)?;
        let input = &inputs[0];
        let k = match inputs.get(1) {
            Some(k_input) if !k_input.is_absent() => {
                if k_input.dtype != DataType::Int64 || !k_input.shape.is_empty() {
                    return Err(EpError::KernelFailed(
                        "Trilu: k input must be a scalar Int64 tensor".into(),
                    ));
                }
                to_dense_i64(k_input)?[0]
            }
            _ => 0,
        };
        if input.shape.len() < 2 {
            return Err(EpError::KernelFailed(
                "Trilu: input must have rank of at least 2".into(),
            ));
        }
        if outputs[0].dtype != input.dtype || outputs[0].shape != input.shape {
            return Err(EpError::KernelFailed(
                "Trilu: output must have the input's dtype and shape".into(),
            ));
        }
        let element_size = super::elem_size(input.dtype)?;
        let rows = input.shape[input.shape.len() - 2];
        let cols = input.shape[input.shape.len() - 1];
        let matrices = numel(&input.shape[..input.shape.len() - 2]);
        let src = to_dense_bytes(input)?;
        let mut out = vec![0; src.len()];
        for matrix in 0..matrices {
            for row in 0..rows {
                for col in 0..cols {
                    let keep = if self.upper {
                        (col as i64 - row as i64) >= k
                    } else {
                        (col as i64 - row as i64) <= k
                    };
                    if keep {
                        let offset = (matrix * rows * cols + row * cols + col) * element_size;
                        out[offset..offset + element_size]
                            .copy_from_slice(&src[offset..offset + element_size]);
                    }
                }
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{static_shape, DataType, Graph, Node, NodeId, TensorData, WeightRef};

    fn i64_bytes(data: &[i64]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Build a Trilu-14 graph with an initializer-backed optional `k` input.
    fn trilu_model(upper: bool, k: Option<i64>) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 14);
        let input = graph.create_named_value("input", DataType::Float32, static_shape([2, 3]));
        graph.add_input(input);
        let mut inputs = vec![Some(input)];
        if let Some(k) = k {
            let k_input =
                graph.create_named_value("k", DataType::Int64, static_shape(std::iter::empty()));
            graph.set_initializer(
                k_input,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Int64,
                    vec![],
                    i64_bytes(&[k]),
                )),
            );
            inputs.push(Some(k_input));
        }
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2, 3]));
        let mut node = Node::new(NodeId(0), "Trilu", inputs, vec![output]);
        node.attributes
            .insert("upper".into(), Attribute::Int(upper as i64));
        let node_id = graph.insert_node(node);
        graph.add_output(output);
        (graph, node_id)
    }

    fn trilu_kernel(upper: bool, k: Option<i64>) -> Box<dyn Kernel> {
        let (graph, node_id) = trilu_model(upper, k);
        TriluFactory
            .create(graph.node(node_id), &[])
            .expect("create Trilu kernel")
    }

    #[test]
    fn flatten_copies_row_major() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        FlattenKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn squeeze_copies_bytes_int64() {
        let a = Owned::i64(&[1, 3, 1], &[7, 8, 9]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        SqueezeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![7, 8, 9]);
    }

    #[test]
    fn squeeze_with_axes_input_ignores_axes_data() {
        let a = Owned::f32(&[1, 2], &[3., 4.]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[2]);
        SqueezeKernel
            .execute(&[a.view(), axes.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 4.]);
    }

    #[test]
    fn size_reports_element_count() {
        let a = Owned::f32(&[2, 3, 4], &[0.0; 24]);
        let mut out = Owned::zeros(DataType::Int64, &[]);
        SizeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![24]);
    }

    #[test]
    fn size_of_scalar_is_one() {
        let a = Owned::f32(&[], &[5.0]);
        let mut out = Owned::zeros(DataType::Int64, &[]);
        SizeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![1]);
    }

    #[test]
    fn trilu_upper_reads_positive_k_input() {
        let input = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let k = Owned::i64(&[], &[1]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        trilu_kernel(true, Some(1))
            .execute(&[input.view(), k.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 2., 3., 0., 0., 6.]);
    }

    #[test]
    fn trilu_lower_reads_negative_k_input() {
        let input = Owned::f32(&[3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let k = Owned::i64(&[], &[-1]);
        let mut out = Owned::zeros_f32(&[3, 3]);
        trilu_kernel(false, Some(-1))
            .execute(&[input.view(), k.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 0., 0., 4., 0., 0., 7., 8., 0.]);
    }

    #[test]
    fn trilu_defaults_k_to_zero_when_input_is_absent() {
        let input = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        trilu_kernel(true, None)
            .execute(&[input.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 0., 5., 6.]);
    }

    #[test]
    fn trilu_lower_applies_to_each_batched_matrix() {
        let input = Owned::i64(&[2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let k = Owned::i64(&[], &[0]);
        let mut out = Owned::zeros(DataType::Int64, &[2, 2, 2]);
        trilu_kernel(false, Some(0))
            .execute(&[input.view(), k.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![1, 0, 3, 4, 5, 0, 7, 8]);
    }
}
