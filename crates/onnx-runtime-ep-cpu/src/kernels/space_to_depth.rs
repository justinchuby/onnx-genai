//! `SpaceToDepth` rearranges spatial blocks into the channel dimension.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};

pub struct SpaceToDepthKernel {
    blocksize: usize,
}

pub struct SpaceToDepthFactory;

impl KernelFactory for SpaceToDepthFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let blocksize = node
            .attr("blocksize")
            .and_then(Attribute::as_int)
            .ok_or_else(|| EpError::KernelFailed("SpaceToDepth: blocksize is required".into()))?;
        if blocksize <= 0 {
            return Err(EpError::KernelFailed(
                "SpaceToDepth: blocksize must be positive".into(),
            ));
        }
        Ok(Box::new(SpaceToDepthKernel {
            blocksize: blocksize as usize,
        }))
    }
}

impl Kernel for SpaceToDepthKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("SpaceToDepth", inputs, outputs, 1, 1, 1)?;
        let input = &inputs[0];
        if input.shape.len() != 4 {
            return Err(EpError::KernelFailed(
                "SpaceToDepth: input must have rank 4".into(),
            ));
        }
        let [n, c, h, w] = <[usize; 4]>::try_from(input.shape).unwrap();
        let block = self.blocksize;
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
        let output_h = h / block;
        let output_w = w / block;
        for batch in 0..n {
            for channel in 0..c {
                for block_h in 0..block {
                    for block_w in 0..block {
                        let output_channel = (block_h * block + block_w) * c + channel;
                        for output_y in 0..output_h {
                            for output_x in 0..output_w {
                                let input_y = output_y * block + block_h;
                                let input_x = output_x * block + block_w;
                                let input_index =
                                    ((batch * c + channel) * h + input_y) * w + input_x;
                                let output_index = ((batch * output_shape[1] + output_channel)
                                    * output_h
                                    + output_y)
                                    * output_w
                                    + output_x;
                                output_bytes[output_index * element_size
                                    ..(output_index + 1) * element_size]
                                    .copy_from_slice(
                                        &input_bytes[input_index * element_size
                                            ..(input_index + 1) * element_size],
                                    );
                            }
                        }
                    }
                }
            }
        }
        write_dense_bytes(&mut outputs[0], &output_bytes)
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
    fn space_to_depth_bf16_preserves_element_bits() {
        let vals: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = Owned::bf16(&[1, 1, 4, 4], &vals);
        let mut output = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 4, 2, 2]);
        SpaceToDepthKernel { blocksize: 2 }
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
        SpaceToDepthKernel { blocksize: 2 }
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
        SpaceToDepthKernel { blocksize: 2 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i64(), vec![1, 11, 2, 12, 3, 13, 4, 14]);
    }
}
