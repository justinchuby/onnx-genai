//! Shared attributes and dense indexing for ONNX MeanVarianceNormalization-29.

use onnx_runtime_ir::{DataType, Node, Shape};

use crate::{EpError, Result};

pub const OP: &str = "MeanVarianceNormalization";
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float32,
    DataType::Float64,
];

pub fn unsupported_reason(node: &Node, shapes: &[Shape], dtypes: &[DataType]) -> Option<String> {
    if node.op_type != OP || !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    if let Some(dtype) = dtypes.first()
        && !DTYPES.contains(dtype)
    {
        return Some(format!(
            "{OP}: X has dtype {dtype:?}; expected one of {DTYPES:?}"
        ));
    }
    let result = Attributes::parse(node).and_then(|attrs| match shapes.first() {
        Some(shape) => attrs.reduction_mask(shape.len()).map(|_| ()),
        None => Ok(()),
    });
    result.err().map(|error| error.to_string())
}

pub struct Attributes {
    axes: Vec<i64>,
    pub epsilon: f32,
}

impl Attributes {
    pub fn parse(node: &Node) -> Result<Self> {
        let axes = match node.attr("axes") {
            None => vec![0, 2, 3],
            Some(attr) => attr
                .as_ints()
                .ok_or_else(|| {
                    EpError::KernelFailed(format!("{OP}: axes must be a list of integers"))
                })?
                .to_vec(),
        };
        let epsilon = match node.attr("epsilon") {
            None => 1e-9,
            Some(attr) => attr
                .as_float()
                .ok_or_else(|| EpError::KernelFailed(format!("{OP}: epsilon must be a float")))?,
        };
        Ok(Self { axes, epsilon })
    }

    fn reduction_mask(&self, rank: usize) -> Result<Vec<bool>> {
        let mut reduce = vec![self.axes.is_empty(); rank];
        let rank = rank as i64;
        for &axis in &self.axes {
            let normalized = if axis < 0 { axis + rank } else { axis };
            if !(0..rank).contains(&normalized) {
                return Err(EpError::KernelFailed(format!(
                    "{OP}: axes contains {axis}, out of range for X rank {rank}; use axes in [-{rank}, {rank})"
                )));
            }
            Ok(reduce)
        }

        pub fn layout(&self, shape: &[usize]) -> Result<Layout> {
            let reduce = self.reduction_mask(shape.len())?;
            if std::mem::replace(&mut reduce[normalized as usize], true) {
                return Err(EpError::KernelFailed(format!(
                    "{OP}: axes contains duplicate axis {axis} for rank {rank}; specify each axis once"
                )));
            }
        }
        if shape.contains(&0) {
            return Ok(Layout {
                kept: vec![],
                reduced: vec![],
                groups: 0,
                width: 0,
            });
        }

        let mut kept = Vec::<AxisRun>::new();
        let mut reduced = Vec::<AxisRun>::new();
        let mut groups = 1;
        let mut width = 1;
        let mut stride = 1usize;
        let mut previous = None;
        for (&extent, is_reduced) in shape.iter().zip(reduce).rev() {
            if extent == 1 {
                continue;
            }
            let next_stride = stride.checked_mul(extent).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{OP}: X shape {shape:?} overflows element indexing"
                ))
            })?;
            let (runs, count) = if is_reduced {
                (&mut reduced, &mut width)
            } else {
                (&mut kept, &mut groups)
            };
            if previous == Some(is_reduced) {
                runs.last_mut().expect("previous run exists").extent *= extent;
            } else {
                runs.push(AxisRun { extent, stride });
            }
            *count *= extent;
            stride = next_stride;
            previous = Some(is_reduced);
        }
        Ok(Layout {
            kept,
            reduced,
            groups,
            width,
        })
    }
}

/// Consecutive dense axes with the same reduction membership, innermost first.
pub struct AxisRun {
    pub extent: usize,
    pub stride: usize,
}

pub struct Layout {
    pub kept: Vec<AxisRun>,
    pub reduced: Vec<AxisRun>,
    pub groups: usize,
    pub width: usize,
}

/// Convert an index over either set of axes into its contribution to the dense
/// tensor offset. The outermost run needs no modulus.
#[inline]
pub fn offset(mut index: usize, runs: &[AxisRun]) -> usize {
    let Some((outer, inner)) = runs.split_last() else {
        return 0;
    };
    let mut offset = 0;
    for run in inner {
        offset += (index % run.extent) * run.stride;
        index /= run.extent;
    }
    offset + index * outer.stride
}
