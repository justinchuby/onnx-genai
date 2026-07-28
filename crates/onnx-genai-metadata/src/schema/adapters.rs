//! LoRA adapter capabilities (native LoRA design §G).
//!
//! Purely additive, forward-compatible metadata declaring which LoRA adapters a
//! model ships with and how a runtime may apply them. Every field is optional,
//! so this keeps the same schema major version and is safely ignored by older
//! readers.

use schemars::JsonSchema;
use serde::Deserialize;

/// Author-declared LoRA adapter capabilities for a model (design §G).
///
/// This is the metadata contract only — it declares intent and defaults; the
/// runtime still discovers and validates each adapter against the actual graph
/// (target manifest, §C) before applying it.
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
}
