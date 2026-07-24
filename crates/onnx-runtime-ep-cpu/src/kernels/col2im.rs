//! `Col2Im`: fold batched columns into an N-dimensional image.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

pub struct Col2ImKernel {
    dilations: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

pub struct Col2ImFactory;

impl KernelFactory for Col2ImFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Col2ImKernel {
            dilations: node
                .attr("dilations")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
            pads: node
                .attr("pads")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
            strides: node
                .attr("strides")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
        }))
    }
}

fn coordinates(mut value: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = value % shape[axis];
        value /= shape[axis];
    }
    result
}

impl Kernel for Col2ImKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Col2Im", inputs, outputs, 3, 3, 1)?;
        if inputs[0].shape.len() != 3 {
            return Err(EpError::KernelFailed(
                "Col2Im: input must have rank 3 [N,C*K,L]".into(),
            ));
        }
        let image: Vec<usize> = to_dense_i64(&inputs[1])?
            .into_iter()
            .map(|d| {
                usize::try_from(d).map_err(|_| {
                    EpError::KernelFailed("Col2Im: image_shape must be non-negative".into())
                })
            })
            .collect::<Result<_>>()?;
        let block: Vec<usize> = to_dense_i64(&inputs[2])?
            .into_iter()
            .map(|d| {
                usize::try_from(d).ok().filter(|&d| d > 0).ok_or_else(|| {
                    EpError::KernelFailed("Col2Im: block_shape must be positive".into())
                })
            })
            .collect::<Result<_>>()?;
        let rank = image.len();
        if rank < 2 || block.len() != rank {
            return Err(EpError::KernelFailed(
                "Col2Im: image_shape and block_shape must have the same rank of at least 2".into(),
            ));
        }
        let dilations = if self.dilations.is_empty() {
            vec![1; rank]
        } else {
            self.dilations.clone()
        };
        let strides = if self.strides.is_empty() {
            vec![1; rank]
        } else {
            self.strides.clone()
        };
        let pads = if self.pads.is_empty() {
            vec![0; rank * 2]
        } else {
            self.pads.clone()
        };
        if dilations.len() != rank
            || strides.len() != rank
            || pads.len() != rank * 2
            || dilations.iter().any(|&v| v <= 0)
            || strides.iter().any(|&v| v <= 0)
            || pads.iter().any(|&v| v < 0)
        {
            return Err(EpError::KernelFailed("Col2Im: dilations/strides must be positive and pads must contain 2 values per spatial axis".into()));
        }
        let block_elements = numel(&block);
        if !inputs[0].shape[1].is_multiple_of(block_elements) {
            return Err(EpError::KernelFailed(
                "Col2Im: input channel-block dimension is not divisible by block_shape product"
                    .into(),
            ));
        }
        let channels = inputs[0].shape[1] / block_elements;
        let mut blocks = Vec::with_capacity(rank);
        for axis in 0..rank {
            let receptive = dilations[axis] * (block[axis] as i64 - 1) + 1;
            let available = image[axis] as i64 + pads[axis] + pads[axis + rank] - receptive;
            blocks.push(if available < 0 {
                0
            } else {
                (available / strides[axis] + 1) as usize
            });
        }
        let locations = numel(&blocks);
        if inputs[0].shape[2] != locations {
            return Err(EpError::KernelFailed(format!(
                "Col2Im: input has {} columns, expected {locations} for image_shape/block_shape",
                inputs[0].shape[2]
            )));
        }
        let mut expected = vec![inputs[0].shape[0], channels];
        expected.extend_from_slice(&image);
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "Col2Im: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }

        let input = to_dense_f32_widen("Col2Im", &inputs[0])?;
        let image_elements = numel(&image);
        let mut output = vec![0.0; numel(&expected)];
        for n in 0..inputs[0].shape[0] {
            for location in 0..locations {
                let block_position = coordinates(location, &blocks);
                for kernel in 0..block_elements {
                    let kernel_position = coordinates(kernel, &block);
                    let mut image_position = Vec::with_capacity(rank);
                    let mut valid = true;
                    for axis in 0..rank {
                        let p = block_position[axis] as i64 * strides[axis]
                            + kernel_position[axis] as i64 * dilations[axis]
                            - pads[axis];
                        if p < 0 || p >= image[axis] as i64 {
                            valid = false;
                            break;
                        }
                        image_position.push(p as usize);
                    }
                    if !valid {
                        continue;
                    }
                    let image_offset = coordinates_to_offset(&image_position, &image);
                    for c in 0..channels {
                        let source = (n * channels * block_elements + c * block_elements + kernel)
                            * locations
                            + location;
                        let target = (n * channels + c) * image_elements + image_offset;
                        output[target] += input[source];
                    }
                }
            }
        }
        write_dense_f32_narrow("Col2Im", &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

fn coordinates_to_offset(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(shape)
        .fold(0, |offset, (&coord, &extent)| offset * extent + coord)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn col2im_bf16_matches_widened_f32_reference() {
        let vals: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let image = Owned::i64(&[2], &[3, 3]);
        let block = Owned::i64(&[2], &[2, 2]);

        let input_f32 = Owned::f32(&[1, 4, 4], &vals);
        let mut ref_out = Owned::zeros_f32(&[1, 1, 3, 3]);
        Col2ImKernel {
            dilations: vec![],
            pads: vec![],
            strides: vec![],
        }
        .execute(
            &[input_f32.view(), image.view(), block.view()],
            &mut [ref_out.view_mut()],
        )
        .unwrap();

        let input_bf16 = Owned::bf16(&[1, 4, 4], &vals);
        let mut bf16_out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 1, 3, 3]);
        Col2ImKernel {
            dilations: vec![],
            pads: vec![],
            strides: vec![],
        }
        .execute(
            &[input_bf16.view(), image.view(), block.view()],
            &mut [bf16_out.view_mut()],
        )
        .unwrap();

        for (&r, &g) in ref_out.to_f32().iter().zip(bf16_out.to_bf16_as_f32().iter()) {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "col2im bf16 {g} vs f32 {r}"
            );
        }
    }

    fn kernel(dilations: Vec<i64>, pads: Vec<i64>, strides: Vec<i64>) -> Col2ImKernel {
        Col2ImKernel {
            dilations,
            pads,
            strides,
        }
    }

    #[test]
    fn col2im_sums_overlapping_2d_windows() {
        let input = Owned::f32(
            &[1, 4, 4],
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
        );
        let image = Owned::i64(&[2], &[3, 3]);
        let block = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[1, 1, 3, 3]);
        kernel(vec![], vec![], vec![])
            .execute(
                &[input.view(), image.view(), block.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 7., 6., 12., 34., 22., 11., 27., 16.]);
    }

    #[test]
    fn col2im_honors_dilation_padding_and_stride() {
        let input = Owned::f32(
            &[1, 4, 4],
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
        );
        let image = Owned::i64(&[2], &[3, 3]);
        let block = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[1, 1, 3, 3]);
        kernel(vec![1, 1], vec![1, 1, 1, 1], vec![2, 2])
            .execute(
                &[input.view(), image.view(), block.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![13., 10., 14., 7., 4., 8., 15., 12., 16.]);
    }

    #[test]
    fn col2im_folds_3d_blocks() {
        let input = Owned::f32(&[1, 8, 1], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let image = Owned::i64(&[3], &[2, 2, 2]);
        let block = Owned::i64(&[3], &[2, 2, 2]);
        let mut out = Owned::zeros_f32(&[1, 1, 2, 2, 2]);
        kernel(vec![], vec![], vec![])
            .execute(
                &[input.view(), image.view(), block.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }
}
