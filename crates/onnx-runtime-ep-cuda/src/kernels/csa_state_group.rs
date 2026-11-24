//! Property-typed capability gate for graph-threaded CSA state.
//! State and scratch ownership/admission belong to native session bindings and
//! caller workspace; this module contains no allocator, quota or lease ledger.
use onnx_runtime_ep_api::EpError;
use onnx_runtime_ir::Node;

/// Record/cache encoding a graph declares for a CSA state group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CsaCacheFormat {
    F32,
    Fp8E4m3Block64,
    Fp4E2m1Block32,
}

impl CsaCacheFormat {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "f32" => Some(Self::F32),
            "fp8_e4m3_block64" => Some(Self::Fp8E4m3Block64),
            "fp4_e2m1_block32" => Some(Self::Fp4E2m1Block32),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Fp8E4m3Block64 => "fp8_e4m3_block64",
            Self::Fp4E2m1Block32 => "fp4_e2m1_block32",
        }
    }
}

/// Why a CSA state group was refused. Every variant carries the property that
/// decided it (design-discipline: "carry the reason"), so the next case is
/// diagnosable and a test can match the exact cause rather than a string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CsaStateGroupRefusal {
    /// The declared compression ratio is neither CSA (4) nor HCA (128).
    UnsupportedRatio { ratio: usize },
    /// The graph declares a `cache_format` string this build does not know.
    UnknownCacheFormat { raw: String },
    /// Ratio-4 (CSA) records are hybrid FP8/BF16; any other cache format cannot
    /// host the learned-index compressor.
    Ratio4RequiresFp8Cache { cache_format: &'static str },
    /// Ratio-4 selection reads a 128-wide FP4 index key; a different index head
    /// dim is a different, unsupported contract.
    Ratio4RequiresIndexHeadDim128 { index_head_dim: usize },
    /// Ratio-128 (HCA) attention-compressor records are f32 or hybrid FP8/BF16,
    /// never FP4.
    Ratio128RejectsFp4,
    /// A head geometry with a zero axis cannot describe a real attention op.
    InvalidHeadGeometry { num_heads: usize, head_dim: usize },
    /// The rotary sub-dimension cannot exceed the head dimension.
    RopeExceedsHeadDim {
        qk_rope_head_dim: usize,
        head_dim: usize,
    },
    /// v1 threads one device per state group; a batch that fans out across
    /// devices is ambiguous and fails closed rather than guessing.
    MultiDeviceAmbiguity { device_count: u32 },
    /// A required `past_* -> present_*` state edge is absent from the node, so
    /// there is nothing to thread — fail closed rather than fabricate state.
    MissingStateEdge { which: &'static str },
    /// The op supports this ratio but the C1 runtime slice threads only
    /// ratio-128 (HCA). Ratio-4 (CSA) and MTP recurrence are follow-up slices
    /// and are typed-refused here rather than silently threaded.
    UnsupportedC1Ratio { ratio: usize },
}

impl std::fmt::Display for CsaStateGroupRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRatio { ratio } => {
                write!(
                    f,
                    "compression_ratio {ratio} is neither CSA (4) nor HCA (128)"
                )
            }
            Self::UnknownCacheFormat { raw } => write!(f, "unknown cache_format {raw:?}"),
            Self::Ratio4RequiresFp8Cache { cache_format } => {
                write!(f, "ratio-4 records are hybrid FP8/BF16, not {cache_format}")
            }
            Self::Ratio4RequiresIndexHeadDim128 { index_head_dim } => write!(
                f,
                "ratio-4 selection needs a 128-wide index key, not {index_head_dim}"
            ),
            Self::Ratio128RejectsFp4 => {
                write!(
                    f,
                    "ratio-128 attention-compressor records are f32 or FP8/BF16, not FP4"
                )
            }
            Self::InvalidHeadGeometry {
                num_heads,
                head_dim,
            } => write!(
                f,
                "invalid head geometry num_heads={num_heads} head_dim={head_dim}"
            ),
            Self::RopeExceedsHeadDim {
                qk_rope_head_dim,
                head_dim,
            } => write!(
                f,
                "qk_rope_head_dim {qk_rope_head_dim} exceeds head_dim {head_dim}"
            ),
            Self::MultiDeviceAmbiguity { device_count } => write!(
                f,
                "a CSA state group spans {device_count} devices; v1 requires exactly one"
            ),
            Self::MissingStateEdge { which } => {
                write!(f, "required state edge {which} is missing from the node")
            }
            Self::UnsupportedC1Ratio { ratio } => write!(
                f,
                "the C1 runtime slice threads only ratio-128 (HCA); ratio-{ratio} is a follow-up slice"
            ),
        }
    }
}

impl From<CsaStateGroupRefusal> for EpError {
    fn from(refusal: CsaStateGroupRefusal) -> Self {
        EpError::KernelFailed(format!(
            "CompressedSparseAttention state group refused: {refusal}"
        ))
    }
}

/// Property-typed identity of one CSA state group, derived from the graph node.
///
/// Support is decided from these properties alone — no op/model name, no shape
/// allowlist — so an unseen but property-compatible layer is accepted and an
/// unsupported one is refused with its exact reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CsaStateGroupDescriptor {
    pub ratio: usize,
    pub cache_format: CsaCacheFormat,
    pub num_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub index_head_dim: usize,
    pub device_count: u32,
    pub has_compressed_state: bool,
    pub has_carry_state: bool,
}

impl CsaStateGroupDescriptor {
    /// Parse the property set from a `pkg.nxrt::CompressedSparseAttention` node.
    ///
    /// Only the presence/values of declared attributes and the state input
    /// edges are read; numerics stay in the kernel. Returns a typed refusal if
    /// the declared `cache_format` string is unknown to this build.
    pub(crate) fn from_node(node: &Node, device_count: u32) -> Result<Self, CsaStateGroupRefusal> {
        let ratio = attr_usize(node, "compression_ratio").unwrap_or(0);
        let raw_format = node
            .attr("cache_format")
            .and_then(|attribute| attribute.as_str())
            .unwrap_or("f32");
        let cache_format = CsaCacheFormat::parse(raw_format).ok_or_else(|| {
            CsaStateGroupRefusal::UnknownCacheFormat {
                raw: raw_format.to_string(),
            }
        })?;
        Ok(Self {
            ratio,
            cache_format,
            num_heads: attr_usize(node, "num_heads").unwrap_or(0),
            head_dim: attr_usize(node, "head_dim").unwrap_or(0),
            qk_rope_head_dim: attr_usize(node, "qk_rope_head_dim").unwrap_or(0),
            index_head_dim: attr_usize(node, "index_head_dim").unwrap_or(0),
            device_count,
            // Frozen v1 threads the compressed records as input 6 and the carry
            // as input 7; a node missing either has no state edge to thread.
            has_compressed_state: node.inputs.get(6).map(Option::is_some).unwrap_or(false),
            has_carry_state: node.inputs.get(7).map(Option::is_some).unwrap_or(false),
        })
    }

    /// Decide support from the declared properties, carrying the exact reason on
    /// refusal. Accepts iff the layer is a property-compatible CSA (ratio-4) or
    /// HCA (ratio-128) group this build can thread on a single device.
    pub(crate) fn validate(&self) -> Result<(), CsaStateGroupRefusal> {
        if self.ratio != 4 && self.ratio != 128 {
            return Err(CsaStateGroupRefusal::UnsupportedRatio { ratio: self.ratio });
        }
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(CsaStateGroupRefusal::InvalidHeadGeometry {
                num_heads: self.num_heads,
                head_dim: self.head_dim,
            });
        }
        if self.qk_rope_head_dim > self.head_dim {
            return Err(CsaStateGroupRefusal::RopeExceedsHeadDim {
                qk_rope_head_dim: self.qk_rope_head_dim,
                head_dim: self.head_dim,
            });
        }
        if self.device_count != 1 {
            return Err(CsaStateGroupRefusal::MultiDeviceAmbiguity {
                device_count: self.device_count,
            });
        }
        if !self.has_compressed_state {
            return Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compressed_kv",
            });
        }
        if !self.has_carry_state {
            return Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compression_carry",
            });
        }
        match self.ratio {
            4 => {
                if self.cache_format != CsaCacheFormat::Fp8E4m3Block64 {
                    return Err(CsaStateGroupRefusal::Ratio4RequiresFp8Cache {
                        cache_format: self.cache_format.as_str(),
                    });
                }
                if self.index_head_dim != 128 {
                    return Err(CsaStateGroupRefusal::Ratio4RequiresIndexHeadDim128 {
                        index_head_dim: self.index_head_dim,
                    });
                }
            }
            128 => {
                if self.cache_format == CsaCacheFormat::Fp4E2m1Block32 {
                    return Err(CsaStateGroupRefusal::Ratio128RejectsFp4);
                }
            }
            _ => unreachable!("ratio guarded above"),
        }
        Ok(())
    }

    /// Stricter than [`validate`](Self::validate): the C1 runtime slice threads
    /// exactly one ratio-128 (HCA) state group end-to-end. Ratio-4 (CSA) and
    /// any other ratio are op-supported but out of C1 scope, so they are
    /// typed-refused here rather than silently threaded through native decode.
    /// This is the gate the native-decode integration calls before admitting a
    /// CSA layer to the compressed state-threading path; that caller lands in
    /// the follow-up slice, so it is exercised by tests today.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn validate_c1_runtime(&self) -> Result<(), CsaStateGroupRefusal> {
        self.validate()?;
        if self.ratio != 128 {
            return Err(CsaStateGroupRefusal::UnsupportedC1Ratio { ratio: self.ratio });
        }
        Ok(())
    }
}

/// Bytes reserved for a CSA state group, split by state class so residency is
/// inspectable (e.g. "compressed < dense" for the HCA memory advantage).
fn attr_usize(node: &Node, name: &str) -> Option<usize> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
}
#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Node, NodeId};

    fn hca_node(cache_format: &str, with_state: bool) -> Node {
        // Inputs 0..=7 present means the frozen-v1 compressed (6) and carry (7)
        // state edges exist; `with_state=false` truncates them.
        let input_count = if with_state { 8 } else { 6 };
        let inputs = (0..input_count)
            .map(|_| Some(onnx_runtime_ir::ValueId(0)))
            .collect();
        let mut node = Node::new(NodeId(0), "CompressedSparseAttention", inputs, vec![]);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));
        node.attributes
            .insert("num_heads".into(), Attribute::Int(64));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(512));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes.insert(
            "cache_format".into(),
            Attribute::String(cache_format.into()),
        );
        node
    }

    #[test]
    fn hca_descriptor_accepts_property_compatible_ratio128() {
        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert_eq!(descriptor.ratio, 128);
        assert_eq!(descriptor.cache_format, CsaCacheFormat::F32);
        assert!(descriptor.validate().is_ok());
    }

    #[test]
    fn ratio128_rejects_fp4_by_property() {
        let node = hca_node("fp4_e2m1_block32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio128RejectsFp4)
        );
    }

    #[test]
    fn ratio4_requires_fp8_and_index_head_dim_128() {
        let mut node = hca_node("f32", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio4RequiresFp8Cache {
                cache_format: "f32"
            })
        );

        node.attributes.insert(
            "cache_format".into(),
            Attribute::String("fp8_e4m3_block64".into()),
        );
        // index_head_dim missing (0) -> refused for the index-key contract.
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio4RequiresIndexHeadDim128 { index_head_dim: 0 })
        );

        node.attributes
            .insert("index_head_dim".into(), Attribute::Int(128));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert!(descriptor.validate().is_ok());
    }

    #[test]
    fn unsupported_ratio_and_geometry_and_multidevice_and_missing_edges_refuse() {
        let mut node = hca_node("f32", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(2));
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 1)
                .unwrap()
                .validate(),
            Err(CsaStateGroupRefusal::UnsupportedRatio { ratio: 2 })
        );

        let mut node = hca_node("f32", true);
        node.attributes.insert("head_dim".into(), Attribute::Int(0));
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 1)
                .unwrap()
                .validate(),
            Err(CsaStateGroupRefusal::InvalidHeadGeometry {
                num_heads: 64,
                head_dim: 0
            })
        );

        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 2).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::MultiDeviceAmbiguity { device_count: 2 })
        );

        let node = hca_node("f32", false);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compressed_kv"
            })
        );
    }

    #[test]
    fn unknown_cache_format_is_typed() {
        let node = hca_node("bf16_block16", true);
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 1),
            Err(CsaStateGroupRefusal::UnknownCacheFormat {
                raw: "bf16_block16".into()
            })
        );
    }

    #[test]
    fn c1_runtime_admits_only_ratio128() {
        // Ratio-128 (HCA) is the one C1 slice group and is admitted.
        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert!(descriptor.validate_c1_runtime().is_ok());

        // Ratio-4 (CSA) is a valid op config but out of C1 scope: it validates
        // at the op level yet is typed-refused by the C1 runtime gate.
        let mut node = hca_node("fp8_e4m3_block64", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        node.attributes
            .insert("index_head_dim".into(), Attribute::Int(128));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 1).unwrap();
        assert!(
            descriptor.validate().is_ok(),
            "ratio-4 is a valid op config"
        );
        assert_eq!(
            descriptor.validate_c1_runtime(),
            Err(CsaStateGroupRefusal::UnsupportedC1Ratio { ratio: 4 }),
            "ratio-4 is out of C1 scope"
        );
    }
}
