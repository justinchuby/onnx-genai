//! Inverse NCHW spatial-block permutations.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::space_depth::{Direction, Order, SpaceDepth};
use onnx_runtime_ir::{Attribute, Node};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};

pub struct LegacySpaceToDepthFactory;

impl KernelFactory for LegacySpaceToDepthFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let block = node
            .attr("blocksize")
            .and_then(Attribute::as_int)
            .ok_or_else(|| EpError::KernelFailed("SpaceToDepth: blocksize is required".into()))?;
        if block <= 0 {
            return Err(EpError::KernelFailed(
                "SpaceToDepth: blocksize must be positive".into(),
            ));
        }
        Ok(Box::new(LegacySpaceToDepth(block as usize)))
    }
}

struct LegacySpaceToDepth(usize);

impl Kernel for LegacySpaceToDepth {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("SpaceToDepth", inputs, outputs, 1, 1, 1)?;
        let input = &inputs[0];
        let shape @ [n, c, h, w] = <[usize; 4]>::try_from(input.shape)
            .map_err(|_| EpError::KernelFailed("SpaceToDepth: input must have rank 4".into()))?;
        let block = self.0;
        if h % block != 0 || w % block != 0 {
            return Err(EpError::KernelFailed(format!(
                "SpaceToDepth: spatial dimensions {h}x{w} must be divisible by blocksize {block}"
            )));
        }
        let output_shape = [n, c * block * block, h / block, w / block];
        if outputs[0].dtype != input.dtype || outputs[0].shape != output_shape {
            return Err(EpError::KernelFailed(format!(
                "SpaceToDepth: output must have dtype {:?} and shape {output_shape:?}",
                input.dtype
            )));
        }
        let element_size = elem_size(input.dtype)?;
        let input_bytes = to_dense_bytes(input)?;
        let mut output_bytes = vec![0; input_bytes.len()];
        if !input_bytes.is_empty() {
            permute::<true, false>(shape, block, element_size, &input_bytes, &mut output_bytes);
        }
        write_dense_bytes(&mut outputs[0], &output_bytes)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

struct SpaceDepthKernel {
    params: SpaceDepth,
    op: String,
}

pub struct SpaceDepthFactory;

impl KernelFactory for SpaceDepthFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let params = SpaceDepth::parse(node, node.local_opset().unwrap_or(1))
            .map_err(|detail| EpError::KernelFailed(format!("{}: {detail}", node.op_type)))?;
        Ok(Box::new(SpaceDepthKernel {
            params,
            op: node.op_type.clone(),
        }))
    }
}

impl Kernel for SpaceDepthKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(&self.op, inputs, outputs, 1, 1, 1)?;
        let input = &inputs[0];
        let output_shape = self
            .params
            .output_shape(input.shape)
            .map_err(|detail| EpError::KernelFailed(format!("{}: {detail}", self.op)))?;
        if outputs[0].dtype != input.dtype || outputs[0].shape != output_shape {
            return Err(EpError::KernelFailed(format!(
                "{}: output must have dtype {:?} and shape {output_shape:?}",
                self.op, input.dtype,
            )));
        }

        let element_size = SpaceDepth::element_bytes(input.dtype)
            .map_err(|detail| EpError::KernelFailed(format!("{}: {detail}", self.op)))?;
        if input.shape.contains(&0) {
            input.validate()?;
            outputs[0].validate()?;
            return Ok(());
        }
        let input_bytes = to_dense_bytes(input)?;
        let mut output_bytes = vec![0; input_bytes.len()];
        let block = self.params.blocksize() as usize;
        let spatial: [usize; 4] = match self.params.direction {
            Direction::SpaceToDepth => input.shape.try_into().expect("rank already validated"),
            Direction::DepthToSpace => output_shape,
        };
        let order = (self.params.direction, self.params.order);
        (match order {
            (Direction::SpaceToDepth, Order::Dcr) => permute::<true, false>,
            (Direction::SpaceToDepth, Order::Crd) => permute::<true, true>,
            (Direction::DepthToSpace, Order::Dcr) => permute::<false, false>,
            (Direction::DepthToSpace, Order::Crd) => permute::<false, true>,
        })(
            spatial,
            block,
            element_size,
            &input_bytes,
            &mut output_bytes,
        );
        write_dense_bytes(&mut outputs[0], &output_bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn permute<const TO_DEPTH: bool, const CRD: bool>(
    [n, c, h, w]: [usize; 4],
    block: usize,
    element_size: usize,
    input: &[u8],
    output: &mut [u8],
) {
    let depth_h = h / block;
    let depth_w = w / block;
    let area = block * block;
    let depth_c = c * area;
    for batch in 0..n {
        for channel in 0..c {
            for block_h in 0..block {
                for block_w in 0..block {
                    let offset = block_h * block + block_w;
                    let depth_channel = if CRD {
                        channel * area + offset
                    } else {
                        offset * c + channel
                    };
                    for y in 0..depth_h {
                        for x in 0..depth_w {
                            let spatial_index = ((batch * c + channel) * h + y * block + block_h)
                                * w
                                + x * block
                                + block_w;
                            let depth_index =
                                ((batch * depth_c + depth_channel) * depth_h + y) * depth_w + x;
                            let (src, dst) = if TO_DEPTH {
                                (spatial_index, depth_index)
                            } else {
                                (depth_index, spatial_index)
                            };
                            output[dst * element_size..(dst + 1) * element_size].copy_from_slice(
                                &input[src * element_size..(src + 1) * element_size],
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn kernel(op: &str, opset: i64, mode: Option<&str>) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), op, vec![], vec![]);
        node.version = Some(opset);
        node.attributes
            .insert("blocksize".into(), Attribute::Int(2));
        if let Some(mode) = mode {
            node.attributes
                .insert("mode".into(), Attribute::String(mode.as_bytes().to_vec()));
        }
        SpaceDepthFactory.create(&node, &[]).unwrap()
    }

    #[test]
    fn space_to_depth_bf16_preserves_element_bits() {
        let vals: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = Owned::bf16(&[1, 1, 4, 4], &vals);
        let mut output = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 4, 2, 2]);
        kernel("SpaceToDepth", 13, None)
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_bf16_as_f32(),
            vec![
                1., 3., 9., 11., 2., 4., 10., 12., 5., 7., 13., 15., 6., 8., 14., 16.
            ]
        );
    }

    #[test]
    fn rearranges_each_spatial_offset_into_channels() {
        let input = Owned::f32(
            &[1, 1, 4, 4],
            &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let mut output = Owned::zeros_f32(&[1, 4, 2, 2]);
        kernel("SpaceToDepth", 13, None)
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_f32(),
            vec![
                1., 3., 9., 11., 2., 4., 10., 12., 5., 7., 13., 15., 6., 8., 14., 16.
            ]
        );
    }

    #[test]
    fn interleaves_channels_within_each_spatial_offset() {
        let input = Owned::i64(&[1, 2, 2, 2], &[1, 2, 3, 4, 11, 12, 13, 14]);
        let mut output = Owned::zeros(onnx_runtime_ir::DataType::Int64, &[1, 8, 1, 1]);
        kernel("SpaceToDepth", 13, None)
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i64(), vec![1, 11, 2, 12, 3, 13, 4, 14]);
    }

    #[test]
    fn space_depth_modes_match_independent_values() {
        let sequential = [0., 1., 2., 3., 4., 5., 6., 7.];
        let interleaved = [0., 4., 1., 5., 2., 6., 3., 7.];
        for (op, src, dst) in [
            ("SpaceToDepth", [1, 2, 2, 2], [1, 8, 1, 1]),
            ("DepthToSpace", [1, 8, 1, 1], [1, 2, 2, 2]),
        ] {
            let dcr = if op == "SpaceToDepth" {
                interleaved
            } else {
                [0., 2., 4., 6., 1., 3., 5., 7.]
            };
            for (opset, mode, expected) in [
                (1, None, dcr),
                (28, None, dcr),
                (28, Some("DCR"), dcr),
                (28, Some("CRD"), sequential),
            ] {
                let input = Owned::f32(&src, &sequential);
                let mut output = Owned::zeros_f32(&dst);
                kernel(op, opset, mode)
                    .execute(&[input.view()], &mut [output.view_mut()])
                    .unwrap();
                assert_eq!(output.to_f32(), expected, "{op}-{opset} {mode:?}");
            }
        }
    }

    #[test]
    fn space_depth_claims_reject_bad_mode_and_geometry() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::{DataType, ValueId, static_shape};

        let cpu = crate::CpuExecutionProvider::new();
        for (op, first, valid, invalid) in [
            ("SpaceToDepth", 28, [1, 2, 4, 6], [1, 2, 4, 5]),
            ("DepthToSpace", 11, [1, 8, 2, 3], [1, 6, 2, 3]),
        ] {
            let mut node = Node::new(NodeId(0), op, vec![Some(ValueId(0))], vec![ValueId(1)]);
            node.attributes
                .insert("blocksize".into(), Attribute::Int(2));
            node.attributes
                .insert("mode".into(), Attribute::String(b"CRD".to_vec()));
            let claims = |opset, dims, dtype| {
                cpu.supports_op(&node, opset, &[static_shape(dims)], &[dtype], &[])
                    .is_supported()
            };
            assert!(claims(first, valid, DataType::Int64), "{op}");
            assert!(!claims(first - 1, valid, DataType::Int64), "{op}");
            assert!(!claims(first, invalid, DataType::Int64), "{op}");
            assert!(!claims(first, valid, DataType::String), "{op}");
        }
    }

    #[test]
    fn ort_plugin_keeps_the_space_to_depth_13_contract() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::{DataType, ValueId, static_shape};

        let plugin = crate::CpuExecutionProvider::for_ort_plugin();
        let (_, descriptors) = crate::build_cpu_registry_with_descriptors();
        let spatial: Vec<_> = descriptors
            .iter()
            .filter(|d| {
                d.domain.is_empty() && matches!(d.op_type.as_str(), "SpaceToDepth" | "DepthToSpace")
            })
            .collect();
        let [descriptor] = spatial.as_slice() else {
            panic!(
                "unexpected number of plugin spatial descriptors: {}",
                spatial.len()
            );
        };
        assert_eq!(descriptor.op_type, "SpaceToDepth");
        assert_eq!(descriptor.since_version, 13);
        assert_eq!(descriptor.end_version, u64::MAX);
        assert_eq!(
            descriptor.supported_dtypes,
            &[
                DataType::Float32,
                DataType::Float16,
                DataType::BFloat16,
                DataType::Float64
            ]
        );

        let mut node = Node::new(
            NodeId(0),
            "SpaceToDepth",
            vec![Some(ValueId(0))],
            vec![ValueId(1)],
        );
        node.attributes
            .insert("blocksize".into(), Attribute::Int(2));
        node.attributes
            .insert("mode".into(), Attribute::String(b"CRD".to_vec()));
        let shapes = [static_shape([1, 2, 2, 2])];
        let claims = |opset| {
            plugin
                .supports_op(&node, opset, &shapes, &[DataType::Float32], &[])
                .is_supported()
        };
        assert!(!claims(12));
        assert!(claims(28));
        let kernel = plugin.get_kernel(&node, &[vec![1, 2, 2, 2]], 28).unwrap();
        let input = Owned::f32(&[1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let mut output = Owned::zeros_f32(&[1, 8, 1, 1]);
        kernel
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), [0., 4., 1., 5., 2., 6., 3., 7.]);
    }
}
