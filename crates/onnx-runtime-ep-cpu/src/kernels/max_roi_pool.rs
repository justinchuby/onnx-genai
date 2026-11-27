use std::ops::Range;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::max_roi_pool::{Attributes, validate_types};
use onnx_runtime_ir::{DataType, Node};

use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};

const OP: &str = "MaxRoiPool";

pub struct Factory(pub u64);

struct MaxRoiPool {
    attrs: Attributes,
    version: u64,
}

fn fail(detail: String) -> EpError {
    EpError::KernelFailed(format!("{OP}: {detail}"))
}

impl KernelFactory for Factory {
    fn unsupported_reason(
        &self,
        node: &Node,
        opset: u64,
        shapes: &[onnx_runtime_ir::Shape],
        input_dtypes: &[DataType],
    ) -> Option<String> {
        onnx_runtime_ir::max_roi_pool::validate_claim(node, opset, shapes, input_dtypes)
            .err()
            .map(|reason| format!("{OP}: {reason}"))
    }

    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MaxRoiPool {
            attrs: Attributes::parse(node).map_err(fail)?,
            version: self.0,
        }))
    }
}

// Exact floor/ceil of the rational bin edges, followed by clipping. The
// inclusive extent is at most 2^32 and the bin index at most i32::MAX.
fn bin(start: i64, end: i64, index: i64, pooled: i64, limit: usize) -> Range<usize> {
    let extent = (end - start + 1).max(1);
    let clip = |edge: i64| ((start + edge).max(0) as usize).min(limit);
    clip(index * extent / pooled)..clip(((index + 1) * extent + pooled - 1) / pooled)
}

impl MaxRoiPool {
    fn run<T: Copy + PartialOrd + Default>(
        &self,
        x: &[T],
        rois: &[T],
        x_shape: &[usize],
        scale: impl Fn(T, f32) -> f64,
        neg_inf: T,
    ) -> Result<Vec<T>> {
        let [batches, channels, height, width] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
        let [ph, pw] = self.attrs.pooled;
        let mut output = Vec::new();
        for (r, roi) in rois.chunks_exact(5).enumerate() {
            let batch = scale(roi[0], 1.0);
            if !batch.is_finite() || batch < 0.0 || batch.fract() != 0.0 || batch >= batches as f64
            {
                return Err(fail(format!(
                    "rois row {r} batch_id {batch} must be an integer in [0,{batches})"
                )));
            }
            let mut corners = [0; 4];
            for (dst, (&v, name)) in corners
                .iter_mut()
                .zip(roi[1..].iter().zip(["x1", "y1", "x2", "y2"]))
            {
                let rounded = scale(v, self.attrs.spatial_scale).round();
                if !(f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&rounded) {
                    return Err(fail(format!(
                        "rois row {r} {name} must have a finite scaled and rounded coordinate in the signed 32-bit range"
                    )));
                }
                *dst = rounded as i64;
            }
            let [x1, y1, x2, y2] = corners;
            for channel in 0..channels {
                for iy in 0..ph {
                    let rows = bin(y1, y2, iy, ph, height);
                    for ix in 0..pw {
                        let columns = bin(x1, x2, ix, pw, width);
                        if rows.is_empty() || columns.is_empty() {
                            output.push(T::default());
                            continue;
                        }
                        let mut best = neg_inf;
                        for y in rows.clone() {
                            let row = ((batch as usize * channels + channel) * height + y) * width;
                            for &value in &x[row + columns.start..row + columns.end] {
                                if value > best {
                                    best = value;
                                }
                            }
                        }
                        output.push(best);
                    }
                }
            }
        }
        Ok(output)
    }
}

impl Kernel for MaxRoiPool {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([x, rois], [y]) = (inputs, outputs) else {
            return Err(fail(
                "expected two inputs X and rois and one output Y".into(),
            ));
        };
        validate_types(
            self.version,
            [Some(x.dtype), Some(rois.dtype), Some(y.dtype)],
        )
        .map_err(fail)?;
        let shape = self.attrs.output_shape(x.shape, rois.shape).map_err(fail)?;
        if y.shape != shape {
            return Err(fail(format!("Y shape {:?}, expected {shape:?}", y.shape)));
        }
        if shape.contains(&0) {
            return Ok(());
        }
        if x.dtype == DataType::Float64 {
            let values = self.run(
                &to_dense::<f64>(x)?,
                &to_dense::<f64>(rois)?,
                x.shape,
                |v, s| v * f64::from(s),
                f64::NEG_INFINITY,
            )?;
            write_dense(y, &values)
        } else {
            let values = self.run(
                &to_dense_f32_widen(OP, x)?,
                &to_dense_f32_widen(OP, rois)?,
                x.shape,
                |v, s| f64::from(v * s),
                f32::NEG_INFINITY,
            )?;
            write_dense_f32_narrow(OP, y, &values)
        }
    }
}
