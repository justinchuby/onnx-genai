//! `AffineGrid` sampling-grid generation.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

pub struct AffineGridKernel {
    align_corners: bool,
}

pub struct AffineGridFactory;

impl KernelFactory for AffineGridFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(AffineGridKernel {
            align_corners: node
                .attr("align_corners")
                .and_then(Attribute::as_int)
                .unwrap_or(0)
                != 0,
        }))
    }
}

fn coordinate(index: usize, extent: usize, align_corners: bool) -> f32 {
    if align_corners {
        if extent <= 1 {
            0.0
        } else {
            2.0 * index as f32 / (extent - 1) as f32 - 1.0
        }
    } else {
        (2.0 * index as f32 + 1.0) / extent as f32 - 1.0
    }
}

impl Kernel for AffineGridKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("AffineGrid", inputs, outputs, 2, 2, 1)?;
        let theta_shape = inputs[0].shape;
        if theta_shape.len() != 3 || !matches!(theta_shape[1..], [2, 3] | [3, 4]) {
            return Err(EpError::KernelFailed(
                "AffineGrid: theta must have shape [N,2,3] or [N,3,4]".into(),
            ));
        }
        let size = to_dense_i64(&inputs[1])?;
        let spatial_rank = theta_shape[1];
        if size.len() != spatial_rank + 2 || size[0] < 0 || size[1] < 0 {
            return Err(EpError::KernelFailed(format!(
                "AffineGrid: size must contain N, C, and {spatial_rank} non-negative spatial dimensions"
            )));
        }
        let dims: Vec<usize> = size[2..]
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    EpError::KernelFailed("AffineGrid: size dimensions must be non-negative".into())
                })
            })
            .collect::<Result<_>>()?;
        let batch = theta_shape[0];
        if size[0] as usize != batch {
            return Err(EpError::KernelFailed(format!(
                "AffineGrid: theta batch {batch} does not match size batch {}",
                size[0]
            )));
        }
        let expected_shape = if spatial_rank == 2 {
            vec![batch, dims[0], dims[1], 2]
        } else {
            vec![batch, dims[0], dims[1], dims[2], 3]
        };
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "AffineGrid: output shape {:?}, expected {expected_shape:?}",
                outputs[0].shape
            )));
        }

        let theta = to_dense_f32_widen("AffineGrid", &inputs[0])?;
        let mut out = Vec::with_capacity(expected_shape.iter().product());
        if spatial_rank == 2 {
            let (height, width) = (dims[0], dims[1]);
            for n in 0..batch {
                let t = &theta[n * 6..n * 6 + 6];
                for y in 0..height {
                    let y = coordinate(y, height, self.align_corners);
                    for x in 0..width {
                        let x = coordinate(x, width, self.align_corners);
                        out.push(t[0] * x + t[1] * y + t[2]);
                        out.push(t[3] * x + t[4] * y + t[5]);
                    }
                }
            }
        } else {
            let (depth, height, width) = (dims[0], dims[1], dims[2]);
            for n in 0..batch {
                let t = &theta[n * 12..n * 12 + 12];
                for z in 0..depth {
                    let z = coordinate(z, depth, self.align_corners);
                    for y in 0..height {
                        let y = coordinate(y, height, self.align_corners);
                        for x in 0..width {
                            let x = coordinate(x, width, self.align_corners);
                            out.push(t[0] * x + t[1] * y + t[2] * z + t[3]);
                            out.push(t[4] * x + t[5] * y + t[6] * z + t[7]);
                            out.push(t[8] * x + t[9] * y + t[10] * z + t[11]);
                        }
                    }
                }
            }
        }
        write_dense_f32_narrow("AffineGrid", &mut outputs[0], &out)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn affine_grid_bf16_matches_widened_f32_reference() {
        let theta_vals = [1.0f32, 0.25, -0.5, 0.1, 1.0, 0.75];
        let theta_f32 = Owned::f32(&[1, 2, 3], &theta_vals);
        let size = Owned::i64(&[4], &[1, 1, 2, 3]);
        let mut ref_out = Owned::zeros_f32(&[1, 2, 3, 2]);
        AffineGridKernel {
            align_corners: true,
        }
        .execute(&[theta_f32.view(), size.view()], &mut [ref_out.view_mut()])
        .unwrap();

        let theta_bf16 = Owned::bf16(&[1, 2, 3], &theta_vals);
        let mut bf16_out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 2, 3, 2]);
        AffineGridKernel {
            align_corners: true,
        }
        .execute(
            &[theta_bf16.view(), size.view()],
            &mut [bf16_out.view_mut()],
        )
        .unwrap();

        for (&r, &g) in ref_out
            .to_f32()
            .iter()
            .zip(bf16_out.to_bf16_as_f32().iter())
        {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "affine_grid bf16 {g} vs f32 {r}"
            );
        }
    }

    #[test]
    fn affine_grid_2d_honors_align_corners() {
        let theta = Owned::f32(&[1, 2, 3], &[1., 0., 0., 0., 1., 0.]);
        let size = Owned::i64(&[4], &[1, 1, 2, 3]);
        let mut out = Owned::zeros_f32(&[1, 2, 3, 2]);
        AffineGridKernel {
            align_corners: false,
        }
        .execute(&[theta.view(), size.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![
                -0.6666666, -0.5, 0., -0.5, 0.6666666, -0.5, -0.6666666, 0.5, 0., 0.5, 0.6666666,
                0.5
            ]
        );

        let mut out = Owned::zeros_f32(&[1, 2, 3, 2]);
        AffineGridKernel {
            align_corners: true,
        }
        .execute(&[theta.view(), size.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![-1., -1., 0., -1., 1., -1., -1., 1., 0., 1., 1., 1.]
        );
    }

    #[test]
    fn affine_grid_3d_applies_translation() {
        let theta = Owned::f32(
            &[1, 3, 4],
            &[1., 0., 0., 0.25, 0., 1., 0., -0.5, 0., 0., 1., 0.75],
        );
        let size = Owned::i64(&[5], &[1, 1, 2, 2, 2]);
        let mut out = Owned::zeros_f32(&[1, 2, 2, 2, 3]);
        AffineGridKernel {
            align_corners: true,
        }
        .execute(&[theta.view(), size.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![
                -0.75, -1.5, -0.25, 1.25, -1.5, -0.25, -0.75, 0.5, -0.25, 1.25, 0.5, -0.25, -0.75,
                -1.5, 1.75, 1.25, -1.5, 1.75, -0.75, 0.5, 1.75, 1.25, 0.5, 1.75
            ]
        );
    }
}
