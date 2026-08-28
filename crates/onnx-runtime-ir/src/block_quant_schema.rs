//! Canonical `pkg.nxrt` block-quantized operator v1 contract.
//!
//! The operator version remains v1 while the development-stage schema is
//! changed atomically across every in-repo producer and consumer. Planar
//! formats carry a packed weight and a separate UE8M0 auxiliary scale tensor;
//! interleaved GGUF formats carry no auxiliary scale input.

use crate::{DataType, Node};

pub const BLOCK_QUANT_LAYOUT_VERSION: i64 = 1;
pub const FP4_MICROSCALE_BLOCK: usize = 32;
pub const FP4_PACK_FACTOR: usize = 2;

pub const BLOCK_QUANTIZED_MATMUL_INPUT_COUNT: usize = 4;
pub const BLOCK_QUANTIZED_MATMUL_INPUT_NAMES: [&str; BLOCK_QUANTIZED_MATMUL_INPUT_COUNT] =
    ["A", "packed_B", "aux_scale_B", "bias"];
pub const BQMM_ACTIVATION: usize = 0;
pub const BQMM_WEIGHT: usize = 1;
pub const BQMM_SCALE: usize = 2;
pub const BQMM_BIAS: usize = 3;

pub const BLOCK_QUANTIZED_MOE_INPUT_COUNT: usize = 12;
pub const BLOCK_QUANTIZED_MOE_INPUT_NAMES: [&str; BLOCK_QUANTIZED_MOE_INPUT_COUNT] = [
    "input",
    "router_logits",
    "fc1_experts_weights",
    "fc1_experts_bias",
    "fc2_experts_weights",
    "fc2_experts_bias",
    "fc3_experts_weights",
    "fc3_experts_bias",
    "router_weights",
    "fc1_experts_aux_scale",
    "fc2_experts_aux_scale",
    "fc3_experts_aux_scale",
];
pub const BQMOE_INPUT: usize = 0;
pub const BQMOE_ROUTER_LOGITS: usize = 1;
pub const BQMOE_FC1_WEIGHT: usize = 2;
pub const BQMOE_FC1_BIAS: usize = 3;
pub const BQMOE_FC2_WEIGHT: usize = 4;
pub const BQMOE_FC2_BIAS: usize = 5;
pub const BQMOE_FC3_WEIGHT: usize = 6;
pub const BQMOE_FC3_BIAS: usize = 7;
pub const BQMOE_ROUTER_WEIGHTS: usize = 8;
pub const BQMOE_FC1_SCALE: usize = 9;
pub const BQMOE_FC2_SCALE: usize = 10;
pub const BQMOE_FC3_SCALE: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanarBlockFormat {
    /// `F8_E4M3FN` weight plus a 2-D `F8_E8M0` scale grid.
    BlockFp8,
    /// Two E2M1 nibbles per `I8` byte plus one `F8_E8M0` scale per 32 inputs.
    Fp4Planar,
}

impl PlanarBlockFormat {
    pub fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "block_fp8" => Ok(Self::BlockFp8),
            "fp4_planar" => Ok(Self::Fp4Planar),
            _ => Err(format!(
                "unsupported planar format '{value}'; expected block_fp8 or fp4_planar"
            )),
        }
    }

    pub const fn capability_str(self) -> &'static str {
        match self {
            Self::BlockFp8 => "block_fp8",
            Self::Fp4Planar => "fp4_planar",
        }
    }

    pub const fn weight_dtype(self) -> DataType {
        match self {
            Self::BlockFp8 => DataType::Float8E4M3FN,
            Self::Fp4Planar => DataType::Int8,
        }
    }

    pub const fn weight_dtype_name(self) -> &'static str {
        match self {
            Self::BlockFp8 => "F8_E4M3",
            Self::Fp4Planar => "I8",
        }
    }

    pub const fn scale_dtype(self) -> DataType {
        DataType::Float8E8M0
    }

    pub const fn scale_dtype_name(self) -> &'static str {
        "F8_E8M0"
    }

    pub const fn pack_factor(self) -> usize {
        match self {
            Self::BlockFp8 => 1,
            Self::Fp4Planar => FP4_PACK_FACTOR,
        }
    }

    pub const fn kernel_id(self) -> i32 {
        match self {
            Self::BlockFp8 => 0,
            Self::Fp4Planar => 1,
        }
    }
}

/// Explicit planar scale geometry carried by operator attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanarBlockGeometry {
    pub format: PlanarBlockFormat,
    pub block_out: usize,
    pub block_in: usize,
}

impl PlanarBlockGeometry {
    pub fn new(
        format: PlanarBlockFormat,
        block_out: usize,
        block_in: usize,
    ) -> std::result::Result<Self, String> {
        if block_out == 0 || block_in == 0 {
            return Err(format!(
                "{} requires positive block geometry, got [{block_out}, {block_in}]",
                format.capability_str()
            ));
        }
        if format == PlanarBlockFormat::Fp4Planar
            && (block_out != 1 || block_in != FP4_MICROSCALE_BLOCK)
        {
            return Err(format!(
                "fp4_planar block geometry must be [1, {FP4_MICROSCALE_BLOCK}], got [{block_out}, {block_in}]"
            ));
        }
        Ok(Self {
            format,
            block_out,
            block_in,
        })
    }
}

/// Require the one canonical development-stage layout version.
pub fn require_layout_v1(node: &Node, op: &str) -> std::result::Result<(), String> {
    let version = node
        .attr("block_layout_version")
        .ok_or_else(|| format!("{op}: missing required integer attribute 'block_layout_version'"))?
        .as_int()
        .ok_or_else(|| format!("{op}: block_layout_version must be an integer"))?;
    if version != BLOCK_QUANT_LAYOUT_VERSION {
        return Err(format!(
            "{op}: block_layout_version must be {BLOCK_QUANT_LAYOUT_VERSION}, got {version}"
        ));
    }
    Ok(())
}

/// Parse the planar properties for one projection.
///
/// Planar formats require both block attributes. Interleaved formats forbid
/// them, preventing two competing layout authorities from reaching a kernel.
pub fn planar_geometry_from_node(
    node: &Node,
    op: &str,
    format_attr: &str,
    block_out_attr: &str,
    block_in_attr: &str,
) -> std::result::Result<Option<PlanarBlockGeometry>, String> {
    let format = node
        .attr(format_attr)
        .ok_or_else(|| format!("{op}: missing required string attribute '{format_attr}'"))?
        .as_str()
        .ok_or_else(|| format!("{op}: attribute '{format_attr}' must be a UTF-8 string"))?;
    let parsed = PlanarBlockFormat::parse(format).ok();
    match parsed {
        Some(format) => {
            let block_out = positive_usize_attr(node, op, block_out_attr)?;
            let block_in = positive_usize_attr(node, op, block_in_attr)?;
            PlanarBlockGeometry::new(format, block_out, block_in).map(Some)
        }
        None => {
            for name in [block_out_attr, block_in_attr] {
                if node.attr(name).is_some() {
                    return Err(format!(
                        "{op}: attribute '{name}' is valid only for block_fp8 or fp4_planar"
                    ));
                }
            }
            Ok(None)
        }
    }
}

fn positive_usize_attr(node: &Node, op: &str, name: &str) -> std::result::Result<usize, String> {
    let value = node
        .attr(name)
        .ok_or_else(|| format!("{op}: missing required integer attribute '{name}'"))?
        .as_int()
        .ok_or_else(|| format!("{op}: attribute '{name}' must be an integer"))?;
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| format!("{op}: attribute '{name}' must be positive, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Attribute, Node, NodeId};

    fn node(format: &str) -> Node {
        let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
        node.domain = "pkg.nxrt".into();
        node.attributes.insert(
            "format".into(),
            Attribute::String(format.as_bytes().to_vec()),
        );
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        node
    }

    #[test]
    fn planar_properties_are_single_source() {
        let mut fp8 = node("block_fp8");
        fp8.attributes
            .insert("block_size_out".into(), Attribute::Int(128));
        fp8.attributes
            .insert("block_size_in".into(), Attribute::Int(128));
        assert_eq!(
            planar_geometry_from_node(
                &fp8,
                "BlockQuantizedMatMul",
                "format",
                "block_size_out",
                "block_size_in",
            )
            .unwrap(),
            Some(PlanarBlockGeometry {
                format: PlanarBlockFormat::BlockFp8,
                block_out: 128,
                block_in: 128,
            })
        );

        let mut interleaved = node("mxfp4");
        interleaved
            .attributes
            .insert("block_size_in".into(), Attribute::Int(32));
        assert!(
            planar_geometry_from_node(
                &interleaved,
                "BlockQuantizedMatMul",
                "format",
                "block_size_out",
                "block_size_in",
            )
            .unwrap_err()
            .contains("valid only")
        );
    }

    #[test]
    fn fp4_geometry_is_fixed() {
        assert!(PlanarBlockGeometry::new(PlanarBlockFormat::Fp4Planar, 1, 32).is_ok());
        assert!(PlanarBlockGeometry::new(PlanarBlockFormat::Fp4Planar, 2, 32).is_err());
        assert!(PlanarBlockGeometry::new(PlanarBlockFormat::Fp4Planar, 1, 16).is_err());
    }
}
