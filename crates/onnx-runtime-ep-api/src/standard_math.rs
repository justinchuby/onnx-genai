//! Concrete tensor validation shared by standard math kernels.

use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    DataType, Node, Shape,
    standard_math::{DTYPES, Math, OPS},
};

/// Check the known parts of the input contract without requiring concrete dimensions.
pub fn unsupported_reason(node: &Node, shapes: &[Shape], dtypes: &[DataType]) -> Option<String> {
    if !matches!(node.domain.as_str(), "" | "ai.onnx")
        || !OPS.iter().any(|&(name, _)| name == node.op_type)
    {
        return None;
    }
    let check = || {
        let op = Math::from_node(node)?;
        if node.inputs.len() != op.input_count() || node.outputs.len() != 1 {
            return Err(format!(
                "{}: expected {} input(s) and one output",
                op.name(),
                op.input_count()
            ));
        }
        let mut known = dtypes
            .iter()
            .copied()
            .filter(|&dtype| dtype != DataType::Undefined);
        if let Some(first) = known.next()
            && (!DTYPES.contains(&first) || known.any(|dtype| dtype != first))
        {
            return Err(format!(
                "{}: input dtypes {dtypes:?} must match and be f16, bf16, f32 or f64",
                op.name()
            ));
        }
        if let Some(shape) = shapes.first() {
            op.axis(shape.len())?;
        }
        Ok(())
    };
    check().err()
}

/// Number of independent scans, prefix length, and distance between prefix elements.
#[derive(Clone, Copy)]
pub struct Scan {
    pub lanes: usize,
    pub length: usize,
    pub inner: usize,
}

pub fn validate(op: Math, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<Option<Scan>> {
    let fail = |detail| EpError::KernelFailed(format!("{}: {detail}", op.name()));
    if inputs.len() != op.input_count() || outputs.len() != 1 {
        return Err(fail(format!(
            "expected {} input(s) and one output, got {} and {}; check the node's operands",
            op.input_count(),
            inputs.len(),
            outputs.len()
        )));
    }
    let x = &inputs[0];
    if !DTYPES.contains(&x.dtype) {
        return Err(fail(format!(
            "unsupported input type {:?}; expected f16, bf16, f32 or f64",
            x.dtype
        )));
    }
    for (name, dtype, shape) in inputs[1..]
        .iter()
        .map(|b| ("B", b.dtype, b.shape))
        .chain(outputs.iter().map(|y| ("output", y.dtype, y.shape)))
    {
        if dtype != x.dtype || shape != x.shape {
            return Err(fail(format!(
                "{name} has type {dtype:?} and shape {shape:?}; expected {:?} and {:?} to match the first input (no broadcasting)",
                x.dtype, x.shape
            )));
        }
    }
    let Some(axis) = op.axis(x.shape.len()).map_err(EpError::KernelFailed)? else {
        return Ok(None);
    };
    if x.shape.contains(&0) {
        return Ok(Some(Scan {
            lanes: 0,
            length: 0,
            inner: 0,
        }));
    }
    let overflow = || {
        fail(format!(
            "shape {:?} exceeds the addressable element count",
            x.shape
        ))
    };
    let len = x
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(overflow)?;
    let length = x.shape[axis];
    let inner = x.shape[axis + 1..].iter().product();
    Ok(Some(Scan {
        lanes: len / length,
        length,
        inner,
    }))
}
