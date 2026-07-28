//! LoRA adapter capabilities (native LoRA design §G).
//!
//! Purely additive, forward-compatible metadata declaring which LoRA adapters a
//! model ships with and how a runtime may apply them. Every field is optional,
//! so this keeps the same schema major version and is safely ignored by older
//! readers.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::Deserialize;

/// Author-declared LoRA adapter capabilities for a model (design §G).
///
/// This is the metadata contract only — it declares intent and defaults. When
/// [`target_manifest`](Self::target_manifest) is present it is authoritative;
/// runtimes validate it against the loaded graph rather than inferring target
/// structure from graph naming.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct LoraCapabilities {
    /// Adapter directories or package-relative paths shipped with the model.
    ///
    /// Each entry is a PEFT adapter (an `adapter_config.json` plus its
    /// `adapter_model.safetensors`). Absent or empty means the model declares no
    /// bundled adapters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available: Vec<String>,

    /// The adapter applied by default when a session does not select one, given
    /// as an entry of [`available`](Self::available). Absent means base-only.
    #[serde(default)]
    pub default: Option<String>,

    /// Author policy describing which module types the shipped adapters target
    /// (for example `"attention_projections"` or `"all_linear"`). Advisory only:
    /// the runtime derives the authoritative target set from the graph.
    #[serde(default)]
    pub target_module_policy: Option<String>,

    /// Whether the model supports swapping the active adapter without rebuilding
    /// the session. Phase 1 is single-fixed-adapter, so a reader may treat an
    /// absent value as `false`.
    #[serde(default)]
    pub supports_hot_swap: Option<bool>,

    /// Authoritative mapping from semantic LoRA modules to graph projections.
    ///
    /// Absent preserves compatibility with older metadata and asks the runtime
    /// to use its fail-loud graph-discovery fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_manifest: Option<LoraTargetManifest>,
}

/// Explicit LoRA target map for one model graph.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct LoraTargetManifest {
    /// Base projections that adapters may target.
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub targets: Vec<LoraTargetDescriptor>,
}

/// One declared base projection and its optional fused child slices.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LoraTargetDescriptor {
    /// Semantic base-module name, for example `q_proj`, `qkv_proj`, or
    /// `linear_attn.in_proj_qkv`.
    #[schemars(length(min = 1))]
    pub module_name: String,

    /// Decoder layer containing this projection.
    pub layer_index: usize,

    /// Exact ONNX node name of the base projection.
    #[schemars(length(min = 1))]
    pub node_name: String,

    /// Exact ONNX value name produced by the base projection.
    #[schemars(length(min = 1))]
    pub output_name: String,

    /// Base projection inner dimension.
    #[schemars(range(min = 1))]
    pub k: usize,

    /// Full base projection output dimension.
    #[schemars(range(min = 1))]
    pub n: usize,

    /// Child semantic role to output slice for a fused projection.
    ///
    /// Empty means the declared module spans the complete output.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub slices: BTreeMap<String, LoraTargetSlice>,

    /// Optional adapter rank policy for a direct target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub rank: Option<usize>,

    /// Optional adapter alpha policy for a direct target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub alpha: Option<f32>,
}

/// One semantic child within a fused projection output.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LoraTargetSlice {
    /// Zero-based column offset in the fused output.
    pub offset: usize,

    /// Number of output columns occupied by this child.
    #[schemars(range(min = 1))]
    pub width: usize,

    /// Optional adapter rank policy for this child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub rank: Option<usize>,

    /// Optional adapter alpha policy for this child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub alpha: Option<f32>,
}
