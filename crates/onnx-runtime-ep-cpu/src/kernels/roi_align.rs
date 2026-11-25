use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    DataType, Node,
    roi_align::{Pool, RoiAlign},
};

use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};

pub(super) struct Factory(pub u64);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native(
            RoiAlign::parse(node, self.0).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct Native(RoiAlign);

impl Kernel for Native {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let [x, rois, indices] = inputs else {
            return Err(EpError::KernelFailed(
                "RoiAlign requires X, rois and batch_indices".into(),
            ));
        };
        let [y] = outputs else {
            return Err(EpError::KernelFailed("RoiAlign requires one output".into()));
        };
        let params = self.0;
        params
            .validate_dtypes(Some(x.dtype), Some(rois.dtype), Some(indices.dtype))
            .map_err(EpError::KernelFailed)?;
        RoiAlign::validate_shapes(Some(x.shape), Some(rois.shape), Some(indices.shape), |&n| {
            Some(n)
        })
        .map_err(EpError::KernelFailed)?;
        let [n, channels, height, width] = <[usize; 4]>::try_from(x.shape).expect("validated rank");
        let expected = [
            rois.shape[0],
            channels,
            params.output_height,
            params.output_width,
        ];
        if y.shape != expected || y.dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "RoiAlign Y has shape {:?} and dtype {:?}; expected {expected:?} and {:?}",
                y.shape, y.dtype, x.dtype
            )));
        }
        let batch = to_dense::<i64>(indices)?;
        let (x, rois) = if x.dtype == DataType::Float64 {
            (to_dense::<f64>(x)?, to_dense::<f64>(rois)?)
        } else {
            let widen = |t| -> Result<Vec<f64>> {
                Ok(to_dense_f32_widen("RoiAlign", t)?
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect())
            };
            (widen(x)?, widen(rois)?)
        };
        let mut output = Vec::with_capacity(y.numel());
        for (r, (coords, batch)) in rois.chunks_exact(4).zip(batch).enumerate() {
            let b = usize::try_from(batch)
                .ok()
                .filter(|&b| b < n)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "RoiAlign batch_indices[{r}] is {batch}; use an index in [0, {n})"
                    ))
                })?;
            let scale = f64::from(params.spatial_scale);
            let shift = if params.half_pixel { 0.5 } else { 0.0 };
            let [x0, y0, x1, y1] = <[f64; 4]>::try_from(coords)
                .expect("four coordinates")
                .map(|v| v * scale - shift);
            let axis = |start: f64, end: f64, bins| -> Option<_> {
                let mut extent = end - start;
                if !extent.is_finite() {
                    return None;
                }
                if !params.half_pixel {
                    extent = extent.max(1.0);
                }
                let bin = extent / bins as f64;
                let samples = if params.sampling_ratio > 0 {
                    params.sampling_ratio
                } else {
                    let samples = bin.ceil().max(0.0);
                    if samples >= i64::MAX as f64 {
                        return None;
                    }
                    samples as i64
                };
                Some((bin, samples))
            };
            let ((bin_x, nx), (bin_y, ny)) = axis(x0, x1, params.output_width)
                .zip(axis(y0, y1, params.output_height))
                .ok_or_else(|| EpError::KernelFailed(format!(
                    "RoiAlign rois[{r}] has non-finite scaled coordinates or an unrepresentable adaptive sampling grid; check rois and spatial_scale"
                )))?;
            for c in 0..channels {
                let base = ((b * channels + c) * height) * width;
                let plane = &x[base..base + height * width];
                for ph in 0..params.output_height {
                    for pw in 0..params.output_width {
                        let mut accum = 0.0;
                        if nx != 0 && ny != 0 {
                            for iy in 0..ny {
                                let sy =
                                    y0 + ph as f64 * bin_y + (iy as f64 + 0.5) * bin_y / ny as f64;
                                for ix in 0..nx {
                                    let sx = x0
                                        + pw as f64 * bin_x
                                        + (ix as f64 + 0.5) * bin_x / nx as f64;
                                    let val = sample(plane, height, width, sy, sx, params.pool);
                                    if params.pool == Pool::Average {
                                        accum += val;
                                    } else if (iy == 0 && ix == 0) || val > accum {
                                        accum = val;
                                    }
                                }
                            }
                            if params.pool == Pool::Average {
                                accum /= nx as f64 * ny as f64;
                            }
                        }
                        output.push(accum);
                    }
                }
            }
        }
        if y.dtype == DataType::Float64 {
            write_dense(y, &output)
        } else {
            let output: Vec<_> = output.into_iter().map(|v| v as f32).collect();
            write_dense_f32_narrow("RoiAlign", y, &output)
        }
    }
}

fn sample(plane: &[f64], height: usize, width: usize, y: f64, x: f64, pool: Pool) -> f64 {
    if height == 0 || width == 0 {
        return 0.0;
    }
    if !(-1.0..=height as f64).contains(&y) || !(-1.0..=width as f64).contains(&x) {
        // Pinned ONNX uses zero weights and corner index zero for padding.
        return 0.0 * plane[0];
    }
    let y = y.clamp(0.0, (height - 1) as f64);
    let x = x.clamp(0.0, (width - 1) as f64);
    let (yl, xl) = (y as usize, x as usize);
    let (yh, xh) = ((yl + 1).min(height - 1), (xl + 1).min(width - 1));
    let (ly, lx) = (y - yl as f64, x - xl as f64);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let terms = [
        (hy * hx) * plane[yl * width + xl],
        (hy * lx) * plane[yl * width + xh],
        (ly * hx) * plane[yh * width + xl],
        (ly * lx) * plane[yh * width + xh],
    ];
    match pool {
        Pool::Average => terms.into_iter().sum(),
        // The pinned ONNX reference takes the maximum of the weighted corners.
        Pool::Max => terms
            .into_iter()
            .reduce(|a, b| if b > a { b } else { a })
            .expect("four corners"),
    }
}
