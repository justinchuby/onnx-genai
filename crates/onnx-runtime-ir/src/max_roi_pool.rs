//! Shared shape and dtype contract for ONNX MaxRoiPool.

use crate::{Attribute, DataType, Node, Shape};

pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

#[derive(Clone, Copy)]
pub struct Attributes {
    pub pooled: [i64; 2],
    pub spatial_scale: f32,
}

impl Attributes {
    pub fn parse(node: &Node) -> Result<Self, String> {
        let Some(Attribute::Ints(pooled)) = node.attr("pooled_shape") else {
            return Err(
                "pooled_shape must be specified as two positive integers [height, width]".into(),
            );
        };
        let [height, width] = pooled.as_slice() else {
            return Err(format!(
                "pooled_shape must have two dimensions, got {pooled:?}"
            ));
        };
        if *height <= 0 || *width <= 0 {
            return Err(format!(
                "pooled_shape dimensions must be positive, got {pooled:?}"
            ));
        }
        let spatial_scale = match node.attr("spatial_scale") {
            None => 1.0,
            Some(Attribute::Float(scale)) if scale.is_finite() => *scale,
            other => {
                return Err(format!(
                    "spatial_scale must be a finite float, got {other:?}"
                ));
            }
        };
        Ok(Self {
            pooled: [*height, *width],
            spatial_scale,
        })
    }

    pub fn native_pooled(self) -> Result<[i32; 2], String> {
        let convert = |d| {
            i32::try_from(d).map_err(|_| {
                format!(
                    "native MaxRoiPool supports pooled_shape dimensions up to {}, got {:?}",
                    i32::MAX,
                    self.pooled
                )
            })
        };
        Ok([convert(self.pooled[0])?, convert(self.pooled[1])?])
    }

    pub fn output_shape(self, x: &[usize], rois: &[usize]) -> Result<[usize; 4], String> {
        validate_shapes(Some(x.len()), Some(rois.len()), rois.get(1).copied())?;
        let [ph, pw] = self.native_pooled()?;
        let output = [rois[0], x[1], ph as usize, pw as usize];
        if !output.contains(&0)
            && output
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .is_none()
        {
            return Err(format!("Y shape {output:?} has too many elements"));
        }
        Ok(output)
    }
}

pub fn validate_shapes(
    x_rank: Option<usize>,
    rois_rank: Option<usize>,
    roi_columns: Option<usize>,
) -> Result<(), String> {
    if let Some(rank) = x_rank
        && rank != 4
    {
        return Err(format!("X must have rank 4 [N,C,H,W], got rank {rank}"));
    }
    if let Some(rank) = rois_rank
        && rank != 2
    {
        return Err(format!(
            "rois must have rank 2 [num_rois,5], got rank {rank}"
        ));
    }
    if let Some(columns) = roi_columns
        && columns != 5
    {
        return Err(format!(
            "rois must have 5 columns [batch_id,x1,y1,x2,y2], got {columns}"
        ));
    }
    Ok(())
}

pub fn validate_types(
    opset: u64,
    types: [Option<DataType>; 3],
) -> Result<Option<DataType>, String> {
    let mut shared = None;
    for (name, dtype) in ["X", "rois", "Y"].into_iter().zip(types) {
        let Some(dtype) = dtype.filter(|t| *t != DataType::Undefined) else {
            continue;
        };
        if !DTYPES.contains(&dtype) || (dtype == DataType::BFloat16 && opset < 22) {
            return Err(format!(
                "{name} dtype {dtype:?} is unsupported at opset {opset}; use float16, float32 or float64 (bfloat16 requires opset 22+)"
            ));
        }
        if let Some(previous) = shared
            && previous != dtype
        {
            return Err(format!(
                "{name} dtype {dtype:?} must match {previous:?} for X, rois and Y"
            ));
        }
        shared = Some(dtype);
    }
    Ok(shared)
}

pub fn validate_claim(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Result<(), String> {
    Attributes::parse(node)?.native_pooled()?;
    if node.inputs.len() != 2 || node.inputs.iter().any(Option::is_none) || node.outputs.len() != 1
    {
        return Err("expected two inputs X and rois, and one output Y".into());
    }
    validate_types(
        opset,
        [dtypes.first().copied(), dtypes.get(1).copied(), None],
    )?;
    let rois = shapes.get(1);
    validate_shapes(
        shapes.first().map(Vec::len),
        rois.map(Vec::len),
        rois.and_then(|s| s.get(1)).and_then(|d| d.as_static()),
    )
}
