//! Shared checked axis/output geometry for Softmax and LogSoftmax.

use super::float_workspace::FloatWorkspace;
use onnx_runtime_ep_api::{EpError, OutputMetadata, Result, TensorMetadata};
use onnx_runtime_ir::DataType;

pub(super) struct SoftmaxGeometry {
    pub(super) outer: usize,
    pub(super) width: usize,
    pub(super) inner: usize,
    pub(super) count: usize,
    pub(super) output_dtype: DataType,
    pub(super) workspace: FloatWorkspace<1>,
}

impl SoftmaxGeometry {
    pub(super) fn new(
        op: &'static str,
        raw_axis: i64,
        coerce_2d: bool,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<Self> {
        let fail = |reason: String| EpError::KernelFailed(format!("{op}: {reason}"));
        let [input] = inputs else {
            return Err(fail(format!(
                "provide exactly one present X input, got {} slots",
                inputs.len()
            )));
        };
        if !input.present || !super::FLOAT_DTYPES.contains(&input.dtype) {
            return Err(fail(format!(
                "X must be a present floating tensor, got present={} dtype {:?}",
                input.present, input.dtype
            )));
        }
        let shape = input.shape;
        let rank = shape.len();
        let axis=if raw_axis<0 {raw_axis.checked_add(rank as i64)} else {Some(raw_axis)}
            .filter(|&axis|axis>=0 && (axis as usize)<rank)
            .ok_or_else(||fail(format!("axis {raw_axis} out of range for rank {rank}; provide rank >= 1 and axis in [-rank, rank)")))? as usize;
        let [output] = outputs else {
            return Err(fail(format!(
                "provide exactly one Y output, got {} slots",
                outputs.len()
            )));
        };
        let dtype = match output {
            OutputMetadata::Tensor { dtype, shape: out } => {
                if out.as_ref() != shape {
                    return Err(fail(format!(
                        "Y shape {out:?} must match X shape {shape:?}"
                    )));
                }
                *dtype
            }
            OutputMetadata::Unresolved { dtype } => *dtype,
            OutputMetadata::Absent => {
                return Err(fail(
                    "required Y output is absent; bind one floating output matching X shape".into(),
                ));
            }
        };
        if !super::FLOAT_DTYPES.contains(&dtype) {
            return Err(fail(format!("Y dtype {dtype:?} must be floating point")));
        }
        let product = |dims: &[usize]| {
            dims.iter().rev().try_fold(1usize,|n,&d|n.checked_mul(d)).ok_or_else(||fail(format!("shape {shape:?} has overflowing dense strides; reduce the tensor dimensions")))
        };
        let count = product(shape)?;
        let outer = product(&shape[..axis])?;
        let (width, inner) = if coerce_2d {
            (product(&shape[axis..])?, 1)
        } else {
            (shape[axis], product(&shape[axis + 1..])?)
        };
        Ok(Self {
            outer,
            width,
            inner,
            count,
            output_dtype: dtype,
            workspace: FloatWorkspace::new(op, [("Y staging", count)])?,
        })
    }
}
