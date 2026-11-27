//! Shared ONNX MaxUnpool type and shape rules.

use crate::{DataType, Node};

pub fn validate_claim(node: &Node, version: u64, dtypes: &[DataType]) -> Result<(), String> {
    if !(2..=3).contains(&node.inputs.len())
        || node.inputs[..2].iter().any(Option::is_none)
        || node.outputs.len() != 1
        || node.outputs[0].is_none()
        || dtypes.len() != node.inputs.len()
    {
        return Err("MaxUnpool: supply X, I, optional output_shape and one output".into());
    }
    element_bytes(version, dtypes[0])?;
    validate_i64("I", dtypes[1])?;
    if node.inputs.get(2).is_some_and(Option::is_some) {
        validate_i64("output_shape", dtypes[2])?;
    }
    Ok(())
}

pub fn element_bytes(version: u64, dtype: DataType) -> Result<usize, String> {
    match (version, dtype) {
        (9.., DataType::Float16) | (22.., DataType::BFloat16) => Ok(2),
        (9.., DataType::Float32) => Ok(4),
        (9.., DataType::Float64) => Ok(8),
        _ => Err(format!(
            "MaxUnpool: X dtype {dtype:?} is not valid at opset {version}; use float16, float32, float64, or bfloat16 with opset 22 or later"
        )),
    }
}

pub fn validate_i64(name: &str, dtype: DataType) -> Result<(), String> {
    if dtype != DataType::Int64 {
        return Err(format!(
            "MaxUnpool: {name} has dtype {dtype:?}; supply an Int64 tensor"
        ));
    }
    Ok(())
}

pub fn validate_control_shape(rank: usize, shape: &[usize]) -> Result<(), String> {
    if shape != [rank] {
        return Err(format!(
            "MaxUnpool: output_shape has tensor shape {shape:?}; supply a rank-1 Int64 tensor of length {rank}"
        ));
    }
    Ok(())
}

pub fn validate_indices_shape(x: &[usize], indices: &[usize]) -> Result<(), String> {
    if x != indices {
        return Err(format!(
            "MaxUnpool: I shape {indices:?} must match X shape {x:?}"
        ));
    }
    Ok(())
}

pub struct Geometry {
    kernel: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
}

impl Geometry {
    pub fn from_node(node: &Node, rank: usize) -> Result<Self, String> {
        let spatial = rank.checked_sub(2).ok_or_else(|| {
            format!("MaxUnpool: X rank {rank} is too small; supply [N, C, spatial...]")
        })?;
        let read =
            |name: &str, len: usize, minimum: i64, default: Option<i64>| {
                let values =
                    match node.attr(name) {
                        Some(attr) => attr
                            .as_ints()
                            .ok_or_else(|| format!("MaxUnpool: {name} must be a list of integers"))?
                            .to_vec(),
                        None => vec![default.ok_or_else(|| {
                    format!("MaxUnpool: missing required attribute {name}; supply {len} values")
                })?; len],
                    };
                if values.len() != len || values.iter().any(|&value| value < minimum) {
                    return Err(format!(
                        "MaxUnpool: {name}={values:?}; supply {len} values each >= {minimum}"
                    ));
                }
                Ok(values)
            };
        Ok(Self {
            kernel: read("kernel_shape", spatial, 1, None)?,
            strides: read("strides", spatial, 1, Some(1))?,
            pads: read("pads", spatial * 2, 0, Some(0))?,
        })
    }

    pub fn spatial_extent(&self, axis: usize, input: i64) -> Result<i64, String> {
        let value = i128::from(self.strides[axis]) * (i128::from(input) - 1)
            + i128::from(self.kernel[axis])
            - i128::from(self.pads[axis])
            - i128::from(self.pads[axis + self.kernel.len()]);
        extent(value).ok_or_else(|| {
            format!("MaxUnpool: computed spatial extent {value} at axis {axis} is not addressable; check kernel_shape, strides and pads")
        })
    }

    pub fn output_shape(
        &self,
        x: &[usize],
        explicit: Option<&[i64]>,
    ) -> Result<Vec<usize>, String> {
        let rank = self.kernel.len() + 2;
        if x.len() != rank {
            return Err(format!(
                "MaxUnpool: X shape {x:?} does not match the configured rank {rank}"
            ));
        }
        if let Some(shape) = explicit {
            validate_control_shape(rank, &[shape.len()])?;
            return shape
                .iter()
                .enumerate()
                .map(|(axis, &dim)| explicit_extent(axis, dim).map(|dim| dim as usize))
                .collect();
        }
        let mut shape = x[..2].to_vec();
        for (axis, &dim) in x[2..].iter().enumerate() {
            let dim = i64::try_from(dim).map_err(|_| {
                format!("MaxUnpool: X spatial extent {dim} at axis {axis} exceeds Int64; reduce the input shape")
            })?;
            shape.push(self.spatial_extent(axis, dim)? as usize);
        }
        Ok(shape)
    }
}

fn extent(value: i128) -> Option<i64> {
    (0..=isize::MAX as i128)
        .contains(&value)
        .then_some(value as i64)
}

pub fn explicit_extent(axis: usize, dim: i64) -> Result<i64, String> {
    extent(i128::from(dim)).ok_or_else(|| {
        format!(
            "MaxUnpool: output_shape[{axis}]={dim}; supply an extent in 0..={}",
            isize::MAX
        )
    })
}

pub fn checked_bytes(name: &str, shape: &[usize], width: usize) -> Result<usize, String> {
    if shape.contains(&0) {
        return Ok(0);
    }
    crate::checked_numel(shape)
        .and_then(|n| n.checked_mul(width))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| format!("MaxUnpool: {name} shape {shape:?} exceeds addressable storage; reduce the tensor size"))
}
