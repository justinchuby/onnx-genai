//! Shared ONNX contract for the native loss kernels.

use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Dim, Node, Shape};

pub const OP: &str = "NegativeLogLikelihoodLoss";
pub const VERSIONS: [u64; 3] = [12, 13, 22];
pub const EDGE_DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
    DataType::Int32,
    DataType::Int64,
];
const FLOATS: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];
pub const INPUT_DTYPES: &[(usize, &[DataType])] = &[
    (0, FLOATS),
    (1, &[DataType::Int32, DataType::Int64]),
    (2, FLOATS),
];
pub const OUTPUT_DTYPES: &[(usize, &[DataType])] = &[(0, FLOATS)];

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Reduction {
    None = 0,
    Sum = 1,
    Mean = 2,
}

#[derive(Clone, Copy)]
pub struct Loss {
    pub reduction: Reduction,
    pub ignore_index: Option<i64>,
    version: u64,
}

pub struct Dimensions {
    pub classes: usize,
    pub spatial: usize,
    pub elements: usize,
}

impl Loss {
    pub fn from_node(node: &Node, version: u64) -> Result<Self> {
        let reduction = match node.attr("reduction") {
            None => Reduction::Mean,
            Some(attr) => match attr.as_str() {
                Some("none") => Reduction::None,
                Some("sum") => Reduction::Sum,
                Some("mean") => Reduction::Mean,
                _ => return Err(fail("reduction must be 'none', 'sum', or 'mean'")),
            },
        };
        let ignore_index = node
            .attr("ignore_index")
            .map(|attr| {
                attr.as_int()
                    .ok_or_else(|| fail("ignore_index must be an integer"))
            })
            .transpose()?;
        Ok(Self {
            reduction,
            ignore_index,
            version,
        })
    }

    fn validate_dtypes(&self, x: DataType, target: DataType, weight: DataType) -> Result<()> {
        match x {
            DataType::Undefined | DataType::Float16 | DataType::Float32 | DataType::Float64 => {}
            DataType::BFloat16 if self.version >= VERSIONS[2] => {}
            DataType::BFloat16 => {
                return Err(fail(
                    "BF16 requires opset 22 or later; upgrade the model or use a pre-22 floating-point input type",
                ));
            }
            _ => {
                return Err(fail(format!(
                    "input must be f16, f32, f64, or (since opset 22) BF16, got {x:?}"
                )));
            }
        }
        if !matches!(
            target,
            DataType::Undefined | DataType::Int32 | DataType::Int64
        ) {
            return Err(fail(format!(
                "target must be Int32 or Int64, got {target:?}"
            )));
        }
        if weight != DataType::Undefined && x != DataType::Undefined && weight != x {
            return Err(fail(format!(
                "weight dtype {weight:?} must match input dtype {x:?}"
            )));
        }
        Ok(())
    }

    pub fn validate(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<Dimensions> {
        if !(2..=3).contains(&inputs.len())
            || outputs.len() != 1
            || inputs[..2].iter().any(TensorView::is_absent)
            || outputs[0].is_absent()
        {
            return Err(fail(
                "expected input, target, optional weight, and one loss output",
            ));
        }
        let (x, target, out) = (&inputs[0], &inputs[1], &outputs[0]);
        let weight = weight(inputs);
        self.validate_dtypes(
            x.dtype,
            target.dtype,
            weight.map_or(DataType::Undefined, |v| v.dtype),
        )?;
        if x.dtype == DataType::Undefined
            || target.dtype == DataType::Undefined
            || weight.is_some_and(|v| v.dtype == DataType::Undefined)
        {
            return Err(fail("concrete input tensors must have defined dtypes"));
        }
        if x.shape.len() < 2
            || target.shape.len() != x.shape.len() - 1
            || target.shape[0] != x.shape[0]
            || target.shape[1..] != x.shape[2..]
        {
            return Err(fail(format!(
                "expected input [N,C,...] and target [N,...], got {:?} and {:?}",
                x.shape, target.shape
            )));
        }
        let classes = x.shape[1];
        if let Some(w) = weight
            && w.shape != [classes]
        {
            return Err(fail(format!(
                "weight must have shape [{classes}], got {:?}",
                w.shape
            )));
        }
        let shape = if self.reduction == Reduction::None {
            target.shape
        } else {
            &[]
        };
        if out.shape != shape || out.dtype != x.dtype {
            return Err(fail(format!(
                "loss must have dtype {:?} and shape {shape:?}, got {:?} {:?}",
                x.dtype, out.dtype, out.shape
            )));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| fail("input or target element count overflows usize"))
        };
        let spatial = if target.shape.contains(&0) {
            0
        } else {
            x.shape[2..].iter().copied().try_fold(1, product)?
        };
        let elements = product(x.shape[0], spatial)?;
        product(elements, classes)?;
        for input in inputs.iter().filter(|v| !v.is_absent()) {
            input.validate()?;
        }
        out.validate()?;
        Ok(Dimensions {
            classes,
            spatial,
            elements,
        })
    }

    /// Validate before converting to an offset; the ignore value may be outside the class range.
    pub fn class(&self, value: i64, pos: usize, classes: usize) -> Result<Option<usize>> {
        if self.ignore_index == Some(value) {
            return Ok(None);
        }
        usize::try_from(value).ok().filter(|&v| v < classes).map(Some).ok_or_else(|| {
            fail(format!("target[{pos}] is {value}; expected a class in [0, {classes}) or ignore_index {:?}", self.ignore_index))
        })
    }
}

pub fn weight<'a, 't>(inputs: &'a [TensorView<'t>]) -> Option<&'a TensorView<'t>> {
    inputs.get(2).filter(|v| !v.is_absent())
}

pub fn unsupported_reason(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<String> {
    if node.op_type != OP || !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    let check = || {
        let loss = Loss::from_node(node, opset)?;
        let dtype = |i| dtypes.get(i).copied().unwrap_or(DataType::Undefined);
        loss.validate_dtypes(dtype(0), dtype(1), dtype(2))?;
        if let Some(x) = shapes.first() {
            if x.len() < 2 {
                return Err(fail(format!("input must have shape [N,C,...], got {x:?}")));
            }
            if let Some(target) = shapes.get(1) {
                let expected = x.iter().take(1).chain(x.iter().skip(2));
                if target.len() != x.len() - 1
                    || expected
                        .zip(target.iter())
                        .any(|(a, b)| matches!((a, b), (Dim::Static(a), Dim::Static(b)) if a != b))
                {
                    return Err(fail(format!(
                        "target must match input shape without the class axis, got {target:?} for {x:?}"
                    )));
                }
            }
        }
        Ok(())
    };
    check().err().map(|err| err.to_string())
}

fn fail(detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{OP}: {detail}"))
}
