//! Shared ONNX contract for the inverse NCHW space/depth permutations.

use crate::{Attribute, DataType, Dim, Node, Shape};

/// Fixed-width storage types from the ONNX IR4 schema; strings are not byte movers.
pub const DTYPES: &[DataType] = &[
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::BFloat16,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Bool,
    DataType::Complex64,
    DataType::Complex128,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    SpaceToDepth,
    DepthToSpace,
}

impl Direction {
    pub fn from_node(node: &Node) -> Option<Self> {
        if !matches!(node.domain.as_str(), "" | "ai.onnx") {
            return None;
        }
        match node.op_type.as_str() {
            "SpaceToDepth" => Some(Self::SpaceToDepth),
            "DepthToSpace" => Some(Self::DepthToSpace),
            _ => None,
        }
    }

    fn mode_since(self) -> u64 {
        match self {
            Self::SpaceToDepth => 28,
            Self::DepthToSpace => 11,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Flatten the block offset first, then the channel.
    Dcr,
    /// Flatten the channel first, then the block offset.
    Crd,
}

#[derive(Clone, Copy, Debug)]
pub enum AxisScale {
    Keep,
    Multiply(i64),
    Divide(i64),
}

impl AxisScale {
    pub fn apply(self, axis: usize, dim: i64) -> Result<i64, String> {
        let name = ["batch", "channel", "height", "width"][axis];
        match self {
            Self::Keep => Ok(dim),
            Self::Multiply(factor) => dim.checked_mul(factor).ok_or_else(|| {
                format!("{name} dimension {dim} times {factor} overflows; reduce the input extent or blocksize")
            }),
            Self::Divide(divisor) if dim % divisor == 0 => Ok(dim / divisor),
            Self::Divide(divisor) => Err(format!(
                "{name} dimension {dim} must be divisible by {divisor}; check the input shape and blocksize"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpaceDepth {
    pub direction: Direction,
    pub order: Order,
    block: i64,
    area: i64,
}

impl SpaceDepth {
    pub fn parse(node: &Node, opset: u64) -> Result<Self, String> {
        let direction = Direction::from_node(node)
            .ok_or_else(|| format!("unexpected operator {}", node.op_type))?;
        let block = node
            .attr("blocksize")
            .and_then(Attribute::as_int)
            .filter(|&value| value > 0)
            .ok_or_else(|| "attribute 'blocksize' must be a positive integer".to_string())?;
        let area = block.checked_mul(block).ok_or_else(|| {
            format!("blocksize squared overflows for {block}; use a smaller blocksize")
        })?;
        let order = match node.attr("mode") {
            None => Order::Dcr,
            Some(_) if opset < direction.mode_since() => {
                return Err(format!(
                    "attribute 'mode' is not defined at opset {opset}; omit it for DCR or use opset {} or newer",
                    direction.mode_since()
                ));
            }
            Some(mode) => match mode.as_str() {
                Some("DCR") => Order::Dcr,
                Some("CRD") => Order::Crd,
                _ => return Err(format!("attribute 'mode' must be DCR or CRD, got {mode:?}")),
            },
        };
        Ok(Self {
            direction,
            order,
            block,
            area,
        })
    }

    pub fn blocksize(self) -> i64 {
        self.block
    }

    pub fn axis_scales(self) -> [AxisScale; 4] {
        use AxisScale::{Divide, Keep, Multiply};
        match self.direction {
            Direction::SpaceToDepth => [
                Keep,
                Multiply(self.area),
                Divide(self.block),
                Divide(self.block),
            ],
            Direction::DepthToSpace => [
                Keep,
                Divide(self.area),
                Multiply(self.block),
                Multiply(self.block),
            ],
        }
    }

    pub fn rank_error(rank: usize) -> String {
        format!("input must have rank 4 [N, C, H, W], got rank {rank}")
    }

    pub fn validate_claim(
        node: &Node,
        opset: u64,
        shapes: &[Shape],
        dtypes: &[DataType],
    ) -> Result<(), String> {
        if !matches!(node.inputs.as_slice(), [Some(_)])
            || !matches!(node.outputs.as_slice(), [Some(_)])
        {
            return Err("expected exactly one input and one output".into());
        }
        let [dtype] = dtypes else {
            return Err(format!("expected one input dtype, got {}", dtypes.len()));
        };
        Self::element_bytes(*dtype)?;
        let params = Self::parse(node, opset)?;
        if let [shape] = shapes {
            // IR uses an empty vector for both missing rank and a scalar.
            if !shape.is_empty() && shape.len() != 4 {
                return Err(Self::rank_error(shape.len()));
            }
            for (axis, (scale, dim)) in params.axis_scales().into_iter().zip(shape).enumerate() {
                if let Dim::Static(dim) = dim {
                    Self::scale_extent(scale, axis, *dim)?;
                }
            }
        }
        Ok(())
    }

    pub fn element_bytes(dtype: DataType) -> Result<usize, String> {
        if DTYPES.contains(&dtype) {
            Ok(dtype.byte_size())
        } else {
            Err(format!(
                "input dtype {dtype:?} is unsupported; use a fixed-width tensor type from the ONNX schema"
            ))
        }
    }

    pub fn output_shape(self, shape: &[usize]) -> Result<[usize; 4], String> {
        if shape.len() != 4 {
            return Err(Self::rank_error(shape.len()));
        }
        let mut output = [0; 4];
        for (axis, (scale, &dim)) in self.axis_scales().into_iter().zip(shape).enumerate() {
            output[axis] = Self::scale_extent(scale, axis, dim)?;
        }
        Ok(output)
    }

    fn scale_extent(scale: AxisScale, axis: usize, dim: usize) -> Result<usize, String> {
        let dim = i64::try_from(dim)
            .map_err(|_| format!("input extent {dim} on axis {axis} exceeds the ONNX i64 range"))?;
        let value = scale.apply(axis, dim)?;
        usize::try_from(value).map_err(|_| {
            format!("output extent {value} on axis {axis} exceeds the host size range")
        })
    }
}
