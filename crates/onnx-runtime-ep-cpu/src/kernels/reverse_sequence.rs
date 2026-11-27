//! Native CPU ReverseSequence, copying whole trailing slices without conversion.

use onnx_runtime_ep_api::reverse_sequence::Axes;
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_bytes, to_dense_i64, write_dense_bytes};

pub struct ReverseSequenceFactory(pub u64);

impl KernelFactory for ReverseSequenceFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReverseSequence {
            axes: Axes::from_node(node)?,
            version: self.0,
        }))
    }
}

struct ReverseSequence {
    axes: Axes,
    version: u64,
}

impl Kernel for ReverseSequence {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ReverseSequence", inputs, outputs, 2, 2, 1)?;
        let plan = self
            .axes
            .plan(self.version, &inputs[0], &inputs[1], &outputs[0])?;
        let lengths = to_dense_i64(&inputs[1])?;
        plan.validate_lengths(lengths.iter().copied())?;
        if plan.bytes == 0 {
            return Ok(());
        }

        let input = to_dense_bytes(&inputs[0])?;
        let mut output = vec![0; plan.bytes];
        let offset = |batch, time| {
            let block = if plan.batch_first {
                batch * plan.time + time
            } else {
                time * plan.batch + batch
            };
            block * plan.block_bytes
        };
        for (batch, length) in lengths.into_iter().enumerate() {
            let length = length as usize;
            for time in 0..plan.time {
                let source = if time < length {
                    length - 1 - time
                } else {
                    time
                };
                let src = offset(batch, source);
                let dst = offset(batch, time);
                output[dst..dst + plan.block_bytes]
                    .copy_from_slice(&input[src..src + plan.block_bytes]);
            }
        }
        write_dense_bytes(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, DataType, NodeId, ValueId, static_shape};

    fn node() -> Node {
        Node::new(
            NodeId(0),
            "ReverseSequence",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        )
    }

    #[test]
    fn reverse_sequence_cpu_exact_bytes_both_layouts() {
        let ep = CpuExecutionProvider::default();
        let mut node = node();
        // Distinct NaN payloads, signed zero, infinities and subnormals.
        let bits = [
            0x7f81, 0x7fc3, 0x8000, 0x0000, 0xff80, 0x7f80, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x4040,
            0xc040, 0x3f00, 0xbf00, 0x3fc0, 0xbfc0, 0x4000, 0xc000,
        ];
        let shape = [3, 3, 2];
        let shapes = [shape.to_vec(), vec![3]];
        let dtypes = [DataType::BFloat16, DataType::Int64];
        let lengths = Owned::i64(&[3], &[3, 0, 2]);
        let input = Owned::bf16_bits(&shape, &bits);
        for (batch_first, expected) in [
            (
                false,
                [
                    0x3f00, 0xbf00, 0x8000, 0x0000, 0x4040, 0xc040, 0x0001, 0x8001, 0x3f80, 0xbf80,
                    0xff80, 0x7f80, 0x7f81, 0x7fc3, 0x3fc0, 0xbfc0, 0x4000, 0xc000,
                ],
            ),
            (
                true,
                [
                    0xff80, 0x7f80, 0x8000, 0x0000, 0x7f81, 0x7fc3, 0x0001, 0x8001, 0x3f80, 0xbf80,
                    0x4040, 0xc040, 0x3fc0, 0xbfc0, 0x3f00, 0xbf00, 0x4000, 0xc000,
                ],
            ),
        ] {
            if batch_first {
                node.attributes
                    .insert("batch_axis".into(), Attribute::Int(0));
                node.attributes
                    .insert("time_axis".into(), Attribute::Int(1));
            }
            let claim_shapes = [static_shape(shape), static_shape([3])];
            for opset in [10, 28] {
                let claim = ep.supports_op(&node, opset, &claim_shapes, &dtypes, &[]);
                assert!(claim.is_supported(), "opset {opset}: {claim:?}");
                let kernel = ep.get_kernel(&node, &shapes, opset).unwrap();
                let mut output = Owned::zeros(DataType::BFloat16, &shape);
                kernel
                    .execute(&[input.view(), lengths.view()], &mut [output.view_mut()])
                    .unwrap();
                assert_eq!(
                    output.to_u16_bits(),
                    expected,
                    "opset {opset}, batch_first={batch_first}"
                );
            }
        }
    }

    #[test]
    fn reverse_sequence_cpu_admission_and_empty_lengths() {
        let ep = CpuExecutionProvider::default();
        let mut node = node();
        let shapes = [static_shape([0, 2]), static_shape([2])];
        for (opset, dtypes, reason) in [
            (9, [DataType::Uint8, DataType::Int64], "since opset 10"),
            (
                27,
                [DataType::Float8E5M2, DataType::Int64],
                "not supported by native execution (opset 27)",
            ),
            (28, [DataType::Uint8, DataType::Int32], "must be Int64"),
            (28, [DataType::String, DataType::Int64], "managed-string"),
        ] {
            let claim = ep.supports_op(&node, opset, &shapes, &dtypes, &[]);
            assert!(claim.reason().unwrap().contains(reason), "{claim:?}");
        }
        let dtypes = [DataType::Uint8, DataType::Int64];
        assert!(
            ep.supports_op(&node, 10, &shapes, &dtypes, &[])
                .is_supported()
        );
        let kernel = ep.get_kernel(&node, &[vec![0, 2], vec![2]], 10).unwrap();
        let input = Owned::u8(&[0, 2], &[]);
        let mut output = Owned::zeros(DataType::Uint8, &[0, 2]);
        for raw in [i64::MIN, 1, i64::MAX] {
            let lengths = Owned::i64(&[2], &[0, raw]);
            let error = kernel
                .execute(&[input.view(), lengths.view()], &mut [output.view_mut()])
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("[1] = {raw} is outside 0..=0"))
            );
        }
        let lengths = Owned::i64(&[2], &[0, 0]);
        kernel
            .execute(&[input.view(), lengths.view()], &mut [output.view_mut()])
            .unwrap();
        assert!(output.to_u8().is_empty());

        let shapes = [static_shape([4, 2]), static_shape([1])];
        let claim = ep.supports_op(&node, 28, &shapes, &dtypes, &[]);
        assert!(claim.reason().unwrap().contains("one length per batch"));
        node.attributes
            .insert("time_axis".into(), Attribute::Int(1));
        let claim = ep.supports_op(&node, 28, &shapes, &dtypes, &[]);
        assert!(claim.reason().unwrap().contains("must be distinct"));
    }
}
