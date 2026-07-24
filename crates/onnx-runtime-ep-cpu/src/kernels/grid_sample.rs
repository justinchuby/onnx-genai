//! `GridSample`: 2-D and volumetric sampling over normalized coordinate grids.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Linear,
    Nearest,
    Cubic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaddingMode {
    Zeros,
    Border,
    Reflection,
}

pub struct GridSampleKernel {
    mode: Mode,
    padding_mode: PaddingMode,
    align_corners: bool,
    since_version: u32,
}

pub struct GridSampleFactory {
    pub since_version: u32,
}

impl KernelFactory for GridSampleFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node
            .attr("mode")
            .and_then(Attribute::as_str)
            .unwrap_or("linear")
        {
            "linear" | "bilinear" => Mode::Linear,
            "nearest" => Mode::Nearest,
            "cubic" | "bicubic" => Mode::Cubic,
            other => {
                return Err(EpError::KernelFailed(format!(
                    "GridSample: unsupported mode {other:?}; expected linear, nearest, or cubic"
                )));
            }
        };
        let padding_mode = match node
            .attr("padding_mode")
            .and_then(Attribute::as_str)
            .unwrap_or("zeros")
        {
            "zeros" => PaddingMode::Zeros,
            "border" => PaddingMode::Border,
            "reflection" => PaddingMode::Reflection,
            other => {
                return Err(EpError::KernelFailed(format!(
                    "GridSample: unsupported padding_mode {other:?}; expected zeros, border, or reflection"
                )));
            }
        };
        Ok(Box::new(GridSampleKernel {
            mode,
            padding_mode,
            align_corners: node
                .attr("align_corners")
                .and_then(Attribute::as_int)
                .unwrap_or(0)
                != 0,
            since_version: self.since_version,
        }))
    }
}

impl GridSampleKernel {
    fn unnormalize(&self, coordinate: f32, size: usize) -> f32 {
        if self.align_corners {
            coordinate.mul_add(
                (size.saturating_sub(1)) as f32 / 2.0,
                (size.saturating_sub(1)) as f32 / 2.0,
            )
        } else {
            coordinate.mul_add(size as f32 / 2.0, (size as f32 - 1.0) / 2.0)
        }
    }

    fn reflect(&self, coordinate: f32, size: usize) -> f32 {
        if size <= 1 {
            return 0.0;
        }
        let (low, high) = if self.align_corners {
            (0.0, (size - 1) as f32)
        } else {
            (-0.5, size as f32 - 0.5)
        };
        let span = high - low;
        let mut offset = (coordinate - low) % (2.0 * span);
        if offset < 0.0 {
            offset += 2.0 * span;
        }
        (if offset <= span {
            low + offset
        } else {
            high - (offset - span)
        })
        .clamp(0.0, (size - 1) as f32)
    }

    fn source_coordinate(&self, coordinate: f32, size: usize) -> f32 {
        let coordinate = self.unnormalize(coordinate, size);
        match self.padding_mode {
            PaddingMode::Zeros => coordinate,
            PaddingMode::Border => coordinate.clamp(0.0, (size.saturating_sub(1)) as f32),
            PaddingMode::Reflection => self.reflect(coordinate, size),
        }
    }

    fn map_index(&self, index: isize, size: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        match self.padding_mode {
            PaddingMode::Zeros => usize::try_from(index).ok().filter(|&i| i < size),
            PaddingMode::Border => Some(index.clamp(0, size as isize - 1) as usize),
            PaddingMode::Reflection => Some(self.reflect(index as f32, size).round() as usize),
        }
    }

    fn sample_2d(
        &self,
        input: &[f32],
        offset: usize,
        height: usize,
        width: usize,
        y: isize,
        x: isize,
    ) -> f32 {
        let Some(y) = self.map_index(y, height) else {
            return 0.0;
        };
        let Some(x) = self.map_index(x, width) else {
            return 0.0;
        };
        input[offset + y * width + x]
    }

    fn sample_3d(
        &self,
        input: &[f32],
        offset: usize,
        depth: usize,
        height: usize,
        width: usize,
        z: isize,
        y: isize,
        x: isize,
    ) -> f32 {
        let Some(z) = self.map_index(z, depth) else {
            return 0.0;
        };
        let Some(y) = self.map_index(y, height) else {
            return 0.0;
        };
        let Some(x) = self.map_index(x, width) else {
            return 0.0;
        };
        input[offset + (z * height + y) * width + x]
    }

    fn cubic_weight(distance: f32) -> f32 {
        let a = -0.75;
        let distance = distance.abs();
        if distance <= 1.0 {
            (a + 2.0) * distance.powi(3) - (a + 3.0) * distance.powi(2) + 1.0
        } else if distance < 2.0 {
            a * distance.powi(3) - 5.0 * a * distance.powi(2) + 8.0 * a * distance - 4.0 * a
        } else {
            0.0
        }
    }
}

impl Kernel for GridSampleKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GridSample", inputs, outputs, 2, 2, 1)?;
        let x_shape = inputs[0].shape;
        let grid_shape = inputs[1].shape;
        let rank = x_shape.len();
        if !matches!(rank, 4 | 5) {
            return Err(EpError::KernelFailed(
                "GridSample: X must have rank 4 (2-D) or 5 (volumetric)".into(),
            ));
        }
        if rank == 5 && self.since_version < 20 {
            return Err(EpError::KernelFailed(
                "GridSample: 3D input requires opset 20 or later".into(),
            ));
        }
        if grid_shape.len() != rank
            || grid_shape[0] != x_shape[0]
            || grid_shape[rank - 1] != rank - 2
        {
            return Err(EpError::KernelFailed(format!(
                "GridSample: grid must have shape [N, spatial..., {}] with the same batch as X",
                rank - 2
            )));
        }
        if rank == 5 && self.mode == Mode::Cubic {
            return Err(EpError::KernelFailed(
                "GridSample: cubic mode is defined only for 2-D input".into(),
            ));
        }

        let expected_output = if rank == 4 {
            vec![x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
        } else {
            vec![
                x_shape[0],
                x_shape[1],
                grid_shape[1],
                grid_shape[2],
                grid_shape[3],
            ]
        };
        if outputs[0].shape != expected_output {
            return Err(EpError::KernelFailed(format!(
                "GridSample: output shape {:?}, expected {expected_output:?}",
                outputs[0].shape
            )));
        }
        if x_shape[2..].contains(&0) {
            return Err(EpError::KernelFailed(
                "GridSample: input spatial dimensions must be non-zero".into(),
            ));
        }

        let input = to_dense_f32_widen("GridSample", &inputs[0])?;
        let grid = to_dense_f32_widen("GridSample", &inputs[1])?;
        let mut output = Vec::with_capacity(expected_output.iter().product());
        if rank == 4 {
            let (batch, channels, height, width) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
            let (out_height, out_width) = (grid_shape[1], grid_shape[2]);
            for n in 0..batch {
                for c in 0..channels {
                    let offset = (n * channels + c) * height * width;
                    for oy in 0..out_height {
                        for ox in 0..out_width {
                            let grid_offset = ((n * out_height + oy) * out_width + ox) * 2;
                            let x = self.source_coordinate(grid[grid_offset], width);
                            let y = self.source_coordinate(grid[grid_offset + 1], height);
                            let value = match self.mode {
                                Mode::Nearest => self.sample_2d(
                                    &input,
                                    offset,
                                    height,
                                    width,
                                    y.round_ties_even() as isize,
                                    x.round_ties_even() as isize,
                                ),
                                Mode::Linear => {
                                    let x0 = x.floor() as isize;
                                    let y0 = y.floor() as isize;
                                    let dx = x - x0 as f32;
                                    let dy = y - y0 as f32;
                                    let top = self.sample_2d(&input, offset, height, width, y0, x0)
                                        * (1.0 - dx)
                                        + self.sample_2d(&input, offset, height, width, y0, x0 + 1)
                                            * dx;
                                    let bottom =
                                        self.sample_2d(&input, offset, height, width, y0 + 1, x0)
                                            * (1.0 - dx)
                                            + self.sample_2d(
                                                &input,
                                                offset,
                                                height,
                                                width,
                                                y0 + 1,
                                                x0 + 1,
                                            ) * dx;
                                    top * (1.0 - dy) + bottom * dy
                                }
                                Mode::Cubic => {
                                    let x0 = x.floor() as isize;
                                    let y0 = y.floor() as isize;
                                    let mut value = 0.0;
                                    for iy in -1..=2 {
                                        let wy = Self::cubic_weight(y - (y0 + iy) as f32);
                                        for ix in -1..=2 {
                                            let wx = Self::cubic_weight(x - (x0 + ix) as f32);
                                            value += wy
                                                * wx
                                                * self.sample_2d(
                                                    &input,
                                                    offset,
                                                    height,
                                                    width,
                                                    y0 + iy,
                                                    x0 + ix,
                                                );
                                        }
                                    }
                                    value
                                }
                            };
                            output.push(value);
                        }
                    }
                }
            }
        } else {
            let (batch, channels, depth, height, width) =
                (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
            let (out_depth, out_height, out_width) = (grid_shape[1], grid_shape[2], grid_shape[3]);
            for n in 0..batch {
                for c in 0..channels {
                    let offset = (n * channels + c) * depth * height * width;
                    for oz in 0..out_depth {
                        for oy in 0..out_height {
                            for ox in 0..out_width {
                                let grid_offset =
                                    (((n * out_depth + oz) * out_height + oy) * out_width + ox) * 3;
                                let x = self.source_coordinate(grid[grid_offset], width);
                                let y = self.source_coordinate(grid[grid_offset + 1], height);
                                let z = self.source_coordinate(grid[grid_offset + 2], depth);
                                let value = match self.mode {
                                    Mode::Nearest => self.sample_3d(
                                        &input,
                                        offset,
                                        depth,
                                        height,
                                        width,
                                        z.round_ties_even() as isize,
                                        y.round_ties_even() as isize,
                                        x.round_ties_even() as isize,
                                    ),
                                    Mode::Linear => {
                                        let x0 = x.floor() as isize;
                                        let y0 = y.floor() as isize;
                                        let z0 = z.floor() as isize;
                                        let dx = x - x0 as f32;
                                        let dy = y - y0 as f32;
                                        let dz = z - z0 as f32;
                                        let mut value = 0.0;
                                        for (z_index, wz) in [(z0, 1.0 - dz), (z0 + 1, dz)] {
                                            for (y_index, wy) in [(y0, 1.0 - dy), (y0 + 1, dy)] {
                                                for (x_index, wx) in [(x0, 1.0 - dx), (x0 + 1, dx)]
                                                {
                                                    value += wz
                                                        * wy
                                                        * wx
                                                        * self.sample_3d(
                                                            &input, offset, depth, height, width,
                                                            z_index, y_index, x_index,
                                                        );
                                                }
                                            }
                                        }
                                        value
                                    }
                                    Mode::Cubic => unreachable!("cubic rank-5 input was rejected"),
                                };
                                output.push(value);
                            }
                        }
                    }
                }
            }
        }
        write_dense_f32_narrow("GridSample", &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx < 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{build_cpu_registry, testutil::Owned};
    use onnx_runtime_ir::{Node, NodeId};

    #[test]
    fn grid_sample_bf16_matches_widened_f32_reference() {
        let x_vals = [1.0f32, 2.0, 3.0, 4.0];
        let grid_vals = [-1.0f32, -1.0, 0.0, 0.0];
        let x_f32 = Owned::f32(&[1, 1, 2, 2], &x_vals);
        let grid_f32 = Owned::f32(&[1, 1, 2, 2], &grid_vals);
        let mut ref_out = Owned::zeros_f32(&[1, 1, 1, 2]);
        kernel(16, Mode::Linear, PaddingMode::Zeros, true)
            .execute(&[x_f32.view(), grid_f32.view()], &mut [ref_out.view_mut()])
            .unwrap();

        let x_bf16 = Owned::bf16(&[1, 1, 2, 2], &x_vals);
        let grid_bf16 = Owned::bf16(&[1, 1, 2, 2], &grid_vals);
        let mut bf16_out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 1, 1, 2]);
        kernel(16, Mode::Linear, PaddingMode::Zeros, true)
            .execute(
                &[x_bf16.view(), grid_bf16.view()],
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
                "grid_sample bf16 {g} vs f32 {r}"
            );
        }
    }

    fn kernel(
        since_version: u32,
        mode: Mode,
        padding_mode: PaddingMode,
        align_corners: bool,
    ) -> GridSampleKernel {
        GridSampleKernel {
            mode,
            padding_mode,
            align_corners,
            since_version,
        }
    }

    #[test]
    fn nearest_and_linear_2d_sample_expected_pixels() {
        let x = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let grid = Owned::f32(&[1, 1, 2, 2], &[-1., -1., 0., 0.]);
        let mut nearest = Owned::zeros_f32(&[1, 1, 1, 2]);
        kernel(16, Mode::Nearest, PaddingMode::Zeros, true)
            .execute(&[x.view(), grid.view()], &mut [nearest.view_mut()])
            .unwrap();
        assert_eq!(nearest.to_f32(), vec![1., 1.]);

        let mut linear = Owned::zeros_f32(&[1, 1, 1, 2]);
        kernel(16, Mode::Linear, PaddingMode::Zeros, true)
            .execute(&[x.view(), grid.view()], &mut [linear.view_mut()])
            .unwrap();
        assert_eq!(linear.to_f32(), vec![1., 2.5]);
    }

    #[test]
    fn padding_modes_and_align_corners_map_coordinates_differently() {
        let x = Owned::f32(&[1, 1, 1, 3], &[10., 20., 30.]);
        let grid = Owned::f32(&[1, 1, 3, 2], &[-2., 0., 2., 0., -1.5, 0.]);
        let mut zeros = Owned::zeros_f32(&[1, 1, 1, 3]);
        kernel(16, Mode::Nearest, PaddingMode::Zeros, true)
            .execute(&[x.view(), grid.view()], &mut [zeros.view_mut()])
            .unwrap();
        assert_eq!(zeros.to_f32(), vec![0., 0., 10.]);

        let mut border = Owned::zeros_f32(&[1, 1, 1, 3]);
        kernel(16, Mode::Nearest, PaddingMode::Border, true)
            .execute(&[x.view(), grid.view()], &mut [border.view_mut()])
            .unwrap();
        assert_eq!(border.to_f32(), vec![10., 30., 10.]);

        let mut reflection = Owned::zeros_f32(&[1, 1, 1, 3]);
        kernel(16, Mode::Nearest, PaddingMode::Reflection, true)
            .execute(&[x.view(), grid.view()], &mut [reflection.view_mut()])
            .unwrap();
        assert_eq!(reflection.to_f32(), vec![20., 20., 10.]);

        let edge = Owned::f32(&[1, 1, 1, 2], &[4., 8.]);
        let grid = Owned::f32(&[1, 1, 1, 2], &[-1., 0.]);
        let mut unaligned = Owned::zeros_f32(&[1, 1, 1, 1]);
        kernel(16, Mode::Linear, PaddingMode::Zeros, false)
            .execute(&[edge.view(), grid.view()], &mut [unaligned.view_mut()])
            .unwrap();
        assert_eq!(unaligned.to_f32(), vec![2.]);
        let mut aligned = Owned::zeros_f32(&[1, 1, 1, 1]);
        kernel(16, Mode::Linear, PaddingMode::Zeros, true)
            .execute(&[edge.view(), grid.view()], &mut [aligned.view_mut()])
            .unwrap();
        assert_eq!(aligned.to_f32(), vec![4.]);
    }

    #[test]
    fn cubic_uses_onnx_cubic_coefficient() {
        let x = Owned::f32(&[1, 1, 1, 4], &[0., 10., 20., 30.]);
        let grid = Owned::f32(&[1, 1, 1, 2], &[-0.25, 0.]);
        let mut out = Owned::zeros_f32(&[1, 1, 1, 1]);
        kernel(16, Mode::Cubic, PaddingMode::Border, true)
            .execute(&[x.view(), grid.view()], &mut [out.view_mut()])
            .unwrap();
        assert!((out.to_f32()[0] - 11.660_156).abs() < 1e-5);
    }

    #[test]
    fn volumetric_nearest_and_trilinear_sample_expected_values() {
        let x = Owned::f32(&[1, 1, 2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let grid = Owned::f32(&[1, 1, 2, 1, 3], &[-1., -1., -1., 0., 0., 0.]);
        let mut nearest = Owned::zeros_f32(&[1, 1, 1, 2, 1]);
        kernel(20, Mode::Nearest, PaddingMode::Zeros, true)
            .execute(&[x.view(), grid.view()], &mut [nearest.view_mut()])
            .unwrap();
        assert_eq!(nearest.to_f32(), vec![0., 0.]);
        let mut linear = Owned::zeros_f32(&[1, 1, 1, 2, 1]);
        kernel(20, Mode::Linear, PaddingMode::Zeros, true)
            .execute(&[x.view(), grid.view()], &mut [linear.view_mut()])
            .unwrap();
        assert_eq!(linear.to_f32(), vec![0., 3.5]);
    }

    #[test]
    fn registry_gates_volumetric_input_to_opset20() {
        let registry = build_cpu_registry();
        let node = Node::new(NodeId(0), "GridSample", vec![], vec![]);
        let input_2d = Owned::f32(&[1, 1, 1, 1], &[1.]);
        let grid_2d = Owned::f32(&[1, 1, 1, 2], &[0., 0.]);
        let input_3d = Owned::f32(&[1, 1, 1, 1, 1], &[1.]);
        let grid_3d = Owned::f32(&[1, 1, 1, 1, 3], &[0., 0., 0.]);

        for opset in [16, 20] {
            let mut output = Owned::zeros_f32(&[1, 1, 1, 1]);
            registry
                .lookup("GridSample", "", opset)
                .unwrap()
                .create(&node, &[])
                .unwrap()
                .execute(&[input_2d.view(), grid_2d.view()], &mut [output.view_mut()])
                .unwrap();
            assert_eq!(output.to_f32(), vec![1.]);
        }

        let mut output = Owned::zeros_f32(&[1, 1, 1, 1, 1]);
        let err = registry
            .lookup("GridSample", "", 16)
            .unwrap()
            .create(&node, &[])
            .unwrap()
            .execute(&[input_3d.view(), grid_3d.view()], &mut [output.view_mut()])
            .unwrap_err();
        assert!(err.to_string().contains("3D input requires opset 20"));

        registry
            .lookup("GridSample", "", 20)
            .unwrap()
            .create(&node, &[])
            .unwrap()
            .execute(&[input_3d.view(), grid_3d.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), vec![1.]);
    }
}
