//! Shared shape rules for the native ONNX DFT paths.

use crate::{DataType, Node};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spectrum {
    Full,
    OneSidedOutput,
    OneSidedInput,
}

impl Spectrum {
    pub fn from_flags(opset: u64, inverse: bool, onesided: bool) -> Self {
        match (onesided, inverse && opset >= 20) {
            (false, _) => Self::Full,
            (true, false) => Self::OneSidedOutput,
            (true, true) => Self::OneSidedInput,
        }
    }

    pub fn validate_components(self, components: usize) -> Result<(), String> {
        match (self, components) {
            (Self::Full, 1 | 2) | (Self::OneSidedOutput, 1) | (Self::OneSidedInput, 2) => Ok(()),
            (Self::OneSidedOutput, _) => Err(format!(
                "onesided=1 is valid only for real input (last dimension 1), got {components}"
            )),
            (Self::OneSidedInput, _) => Err(format!(
                "inverse one-sided DFT requires complex input (last dimension 2), got {components}"
            )),
            (Self::Full, _) => Err(format!(
                "last input dimension must be 1 (real) or 2 (complex), got {components}"
            )),
        }
    }

    pub fn output_components(self) -> usize {
        if self == Self::OneSidedInput { 1 } else { 2 }
    }

    pub fn transform_length(self, explicit: Option<usize>, extent: usize) -> Result<usize, String> {
        let length = match explicit {
            Some(length) => length,
            None if self == Self::OneSidedInput => extent
                .checked_sub(1)
                .and_then(|bins| bins.checked_mul(2))
                .ok_or_else(|| format!("cannot infer inverse real length from {extent} bins; provide a positive dft_length"))?,
            None => extent,
        };
        if length == 0 {
            return Err("dft_length must be positive; provide an explicit positive dft_length when the default is zero".into());
        }
        Ok(length)
    }

    pub fn output_extent(self, length: usize) -> usize {
        if self == Self::OneSidedOutput {
            length / 2 + 1
        } else {
            length
        }
    }

    pub fn input_extent(self, length: usize) -> usize {
        if self == Self::OneSidedInput {
            length / 2 + 1
        } else {
            length
        }
    }
}

pub fn default_axis(node: &Node, opset: u64) -> i64 {
    if opset >= 20 {
        -2
    } else {
        node.attr("axis")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(1)
    }
}

pub fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let r = i64::try_from(rank).map_err(|_| format!("input rank {rank} exceeds i64"))?;
    let normalized = if axis < 0 { axis + r } else { axis };
    if normalized < 0 || normalized >= r - 1 {
        return Err(format!(
            "axis {axis} is invalid for rank {rank}; the last dimension is the complex component and cannot be transformed"
        ));
    }
    Ok(normalized as usize)
}

pub fn explicit_length(length: i64) -> Result<usize, String> {
    usize::try_from(length)
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            format!("dft_length must be positive and fit the native index size, got {length}")
        })
}

pub fn is_length_dtype(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int32 | DataType::Int64)
}
