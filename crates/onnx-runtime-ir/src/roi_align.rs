//! Shared admission and attribute contract for ONNX RoiAlign.

use crate::{Attribute, DataType, Node, Shape, is_default_domain};

pub const VERSIONS: [u64; 3] = [10, 16, 22];
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Average,
    Max,
}

#[derive(Clone, Copy)]
pub struct RoiAlign {
    pub output_height: usize,
    pub output_width: usize,
    pub sampling_ratio: i64,
    pub spatial_scale: f32,
    pub half_pixel: bool,
    pub pool: Pool,
    opset: u64,
}

impl RoiAlign {
    pub fn parse(node: &Node, opset: u64) -> Result<Self, String> {
        if opset < VERSIONS[0] {
            return Err(format!("RoiAlign requires opset 10 or later; got {opset}"));
        }
        if node.inputs.len() != 3
            || node.inputs.iter().any(Option::is_none)
            || node.outputs.len() != 1
        {
            return Err("RoiAlign requires X, rois, batch_indices, and one output".into());
        }
        let int = |name, default| match node.attr(name) {
            None => Ok(default),
            Some(attr) => attr
                .as_int()
                .ok_or_else(|| format!("RoiAlign {name} must be an integer")),
        };
        let extent = |name| {
            let value = int(name, 1)?;
            usize::try_from(value)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("RoiAlign {name} must be a positive extent; got {value}"))
        };
        let string = |name, default| match node.attr(name) {
            None => Ok(default),
            Some(attr) => attr
                .as_str()
                .ok_or_else(|| format!("RoiAlign {name} must be a UTF-8 string")),
        };
        let sampling_ratio = int("sampling_ratio", 0)?;
        if sampling_ratio < 0 {
            return Err(format!(
                "RoiAlign sampling_ratio must be nonnegative (0 for adaptive); got {sampling_ratio}"
            ));
        }
        let spatial_scale = match node.attr("spatial_scale") {
            None => 1.0,
            Some(Attribute::Float(value)) if value.is_finite() => *value,
            _ => return Err("RoiAlign spatial_scale must be a finite float".into()),
        };
        if opset < 16 && node.attr("coordinate_transformation_mode").is_some() {
            return Err(
                "RoiAlign coordinate_transformation_mode requires opset 16 or later".into(),
            );
        }
        let default_coord = if opset < 16 {
            "output_half_pixel"
        } else {
            "half_pixel"
        };
        let half_pixel = match string("coordinate_transformation_mode", default_coord)? {
            "half_pixel" => true,
            "output_half_pixel" => false,
            value => {
                return Err(format!(
                    "RoiAlign coordinate_transformation_mode must be half_pixel or output_half_pixel; got {value:?}"
                ));
            }
        };
        let pool = match string("mode", "avg")? {
            "avg" => Pool::Average,
            "max" => Pool::Max,
            value => return Err(format!("RoiAlign mode must be avg or max; got {value:?}")),
        };
        Ok(Self {
            output_height: extent("output_height")?,
            output_width: extent("output_width")?,
            sampling_ratio,
            spatial_scale,
            half_pixel,
            pool,
            opset,
        })
    }

    pub fn validate_dtypes(
        &self,
        x: Option<DataType>,
        rois: Option<DataType>,
        indices: Option<DataType>,
    ) -> Result<(), String> {
        for (name, dtype) in [("X", x), ("rois", rois)] {
            if let Some(dtype) = dtype {
                if dtype == DataType::BFloat16 && self.opset < 22 {
                    return Err(format!(
                        "RoiAlign {name} uses BFloat16, which requires opset 22; got {}",
                        self.opset
                    ));
                }
                if !DTYPES.contains(&dtype) {
                    return Err(format!(
                        "RoiAlign {name} must be floating point; got {dtype:?}"
                    ));
                }
            }
        }
        if let (Some(x), Some(rois)) = (x, rois)
            && x != rois
        {
            return Err(format!(
                "RoiAlign rois dtype {rois:?} must match X dtype {x:?}"
            ));
        }
        if let Some(indices) = indices
            && indices != DataType::Int64
        {
            return Err(format!(
                "RoiAlign batch_indices must be Int64; got {indices:?}"
            ));
        }
        Ok(())
    }

    pub fn validate_shapes<D>(
        x: Option<&[D]>,
        rois: Option<&[D]>,
        indices: Option<&[D]>,
        known: impl Fn(&D) -> Option<usize>,
    ) -> Result<(), String> {
        for (name, shape, rank) in [
            ("X", x, 4),
            ("rois", rois, 2),
            ("batch_indices", indices, 1),
        ] {
            if let Some(shape) = shape
                && shape.len() != rank
            {
                return Err(format!(
                    "RoiAlign {name} must have rank {rank}; got {}",
                    shape.len()
                ));
            }
        }
        if let Some(width) = rois.and_then(|shape| known(&shape[1]))
            && width != 4
        {
            return Err(format!(
                "RoiAlign rois must have shape [num_rois, 4]; got {width} coordinates per ROI"
            ));
        }
        if let (Some(rois), Some(indices)) = (
            rois.and_then(|shape| known(&shape[0])),
            indices.and_then(|shape| known(&shape[0])),
        ) && rois != indices
        {
            return Err(format!(
                "RoiAlign needs one batch index per ROI; got {indices} indices for {rois} rois"
            ));
        }
        Ok(())
    }
}

pub fn unsupported_reason(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<String> {
    if node.op_type != "RoiAlign" || !is_default_domain(&node.domain) {
        return None;
    }
    let result = RoiAlign::parse(node, opset).and_then(|params| {
        let dtype = |i| {
            dtypes
                .get(i)
                .copied()
                .filter(|&dtype| dtype != DataType::Undefined)
        };
        params.validate_dtypes(dtype(0), dtype(1), dtype(2))?;
        // Claim callers may provide empty shapes when ranks have not been inferred.
        let shape = |i| {
            shapes
                .get(i)
                .map(Vec::as_slice)
                .filter(|s: &&[_]| !s.is_empty())
        };
        RoiAlign::validate_shapes(shape(0), shape(1), shape(2), |d| d.as_static())
    });
    result.err()
}
