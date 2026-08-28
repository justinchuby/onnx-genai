//! The one registry for optional, versioned semantic extensions.
//!
//! Core schema conformance is deliberately not represented here. A reader that
//! accepts a schema version owns its SSA, state, shape, and output invariants;
//! those cannot be switched on with a package flag. This registry is only for
//! independently implemented semantic modules whose exact identity and version
//! a package declares through a typed extension surface.

use crate::version::{
    CANONICAL_SPECULATION_SCHEMA_VERSION, DFLASH_SCHEMA_VERSION, INITIAL_SCHEMA_VERSION,
    SchemaVersion, TOKEN_CONTEXT_SCHEMA_VERSION, TOOL_PROTOCOL_SCHEMA_VERSION,
};

/// One immutable identifier/version pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtensionId {
    pub identity: &'static str,
    pub version: &'static str,
}

impl ExtensionId {
    pub const fn new(identity: &'static str, version: &'static str) -> Self {
        Self { identity, version }
    }

    pub fn wire_name(self) -> String {
        format!("{}@{}", self.identity, self.version)
    }

    pub fn matches_wire_name(self, value: &str) -> bool {
        value
            .strip_suffix(self.version)
            .is_some_and(|identity| identity.strip_suffix('@') == Some(self.identity))
    }
}

/// The only legal fallback for an extension requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackClass {
    /// A package that declares the extension cannot run without this exact pair.
    SemanticRequired,
    /// Runtime policy may decline this path and preserve semantics generically.
    ///
    /// No package extension may use this class. It is included so generated
    /// guidance can state the boundary explicitly.
    OptimizationOptional,
}

impl FallbackClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SemanticRequired => "semantic-required (fail closed)",
            Self::OptimizationOptional => "optimization-optional (generic/isolated fallback)",
        }
    }
}

/// Current implementation coverage for the registry row, not a package claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportStatus {
    /// The named consumer implements the exact pair.
    Implemented,
    /// The declaration is valid but the consumer admits only a documented subset.
    Partial,
    /// The schema recognizes the pair but the current consumer refuses it.
    KnownButUnavailable,
}

impl SupportStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Partial => "partial; see admission consumer",
            Self::KnownButUnavailable => "known, but unavailable (fail closed)",
        }
    }
}

/// Typed metadata surface that declares an extension requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionSurface {
    ToolProtocol,
    ComponentContract,
    ComponentAdapter,
    PackageAdapter,
    AdapterLoader,
    Speculative,
    StateCheckpoint,
}

impl ExtensionSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ToolProtocol => "tool protocol",
            Self::ComponentContract => "component contract",
            Self::ComponentAdapter => "component adapter ABI",
            Self::PackageAdapter => "package adapter service",
            Self::AdapterLoader => "adapter artifact loader",
            Self::Speculative => "speculative execution contract",
            Self::StateCheckpoint => "state checkpoint adapter",
        }
    }
}

/// A built-in optional semantic extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionDescriptor {
    pub id: ExtensionId,
    pub surface: ExtensionSurface,
    pub schema_floor: SchemaVersion,
    pub declaration: &'static str,
    pub normative_reference: &'static str,
    pub admission_consumer: &'static str,
    pub fallback: FallbackClass,
    pub status: SupportStatus,
}

const V1: SchemaVersion = INITIAL_SCHEMA_VERSION;

pub const TAGGED_JSON_V1: ExtensionId = ExtensionId::new("tagged-json", "v1");
pub const ATEM_XML_V1: ExtensionId = ExtensionId::new("atem-xml", "v1");
pub const TOKEN_CONTEXT_V1: ExtensionId = ExtensionId::new("onnx-genai.token-context", "1");
pub const SPECULATIVE_V1: ExtensionId = ExtensionId::new("onnx-genai.speculative", "1");
pub const DFLASH_FLAT_BLOCK_V1: ExtensionId = ExtensionId::new("onnx-genai.dflash-flat-block", "1");
pub const DFLASH_FLAT_BLOCK_V2: ExtensionId = ExtensionId::new("onnx-genai.dflash-flat-block", "2");
pub const GRAMMAR_GUIDANCE_V1: ExtensionId = ExtensionId::new("onnx-genai.grammar-guidance", "1");
pub const TELEMETRY_V1: ExtensionId = ExtensionId::new("onnx-genai.telemetry", "1");
pub const PARAMETER_OVERLAY_V1: ExtensionId = ExtensionId::new("onnx-genai.parameter-overlay", "1");
pub const IMAGE_PREPROCESS_V1: ExtensionId = ExtensionId::new("onnx-genai.image-preprocess", "1");
pub const VIDEO_PREPROCESS_V1: ExtensionId = ExtensionId::new("onnx-genai.video-preprocess", "1");
pub const AUDIO_PREPROCESS_V1: ExtensionId = ExtensionId::new("onnx-genai.audio-preprocess", "1");
pub const TEXT_ASSEMBLY_V1: ExtensionId = ExtensionId::new("onnx-genai.text-assembly", "1");
pub const ADAPTERS_V1: ExtensionId = ExtensionId::new("onnx-genai.adapters", "1");
pub const ADAPTERS_JSON_V1: ExtensionId = ExtensionId::new("onnx-genai.adapters.json", "1");
pub const ORT_LORA_ADAPTER_V1: ExtensionId = ExtensionId::new("onnxruntime.lora-adapter", "1");
pub const ADAPTERS_HF_PEFT_V1: ExtensionId = ExtensionId::new("onnx-genai.adapters.hf-peft", "1");
pub const ADAPTERS_SAFETENSORS_V1: ExtensionId =
    ExtensionId::new("onnx-genai.adapters.safetensors", "1");
pub const KV_CHECKPOINT_V1: ExtensionId = ExtensionId::new("onnx-genai.kv-checkpoint", "1");

/// Every built-in optional semantic extension, declared once.
///
/// Entries describe existing typed extension surfaces and current consumers.
/// They are neither a catalogue of core schema semantics nor a hardware/backend
/// feature list.
pub const BUILTIN_EXTENSIONS: &[ExtensionDescriptor] = &[
    ExtensionDescriptor {
        id: TAGGED_JSON_V1,
        surface: ExtensionSurface::ToolProtocol,
        schema_floor: TOOL_PROTOCOL_SCHEMA_VERSION,
        declaration: "package.tool_protocol",
        normative_reference: "§4.3a, §15",
        admission_consumer: "onnx_genai_server::tool_protocol::resolve",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ATEM_XML_V1,
        surface: ExtensionSurface::ToolProtocol,
        schema_floor: TOOL_PROTOCOL_SCHEMA_VERSION,
        declaration: "package.tool_protocol",
        normative_reference: "§4.3a, §15",
        admission_consumer: "onnx_genai_server::tool_protocol::resolve",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: TOKEN_CONTEXT_V1,
        surface: ExtensionSurface::ComponentContract,
        schema_floor: TOKEN_CONTEXT_SCHEMA_VERSION,
        declaration: "pipeline.workflow.components.<name>.contract",
        normative_reference: "§12, §15",
        admission_consumer: "onnx_genai_metadata::validate_token_context_component",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: SPECULATIVE_V1,
        surface: ExtensionSurface::Speculative,
        schema_floor: CANONICAL_SPECULATION_SCHEMA_VERSION,
        declaration: "speculative.identity + speculative.version",
        normative_reference: "§13",
        admission_consumer: "onnx_genai_engine::WorkflowExecutionAdmission",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Partial,
    },
    ExtensionDescriptor {
        id: DFLASH_FLAT_BLOCK_V1,
        surface: ExtensionSurface::Speculative,
        schema_floor: DFLASH_SCHEMA_VERSION,
        declaration: "speculative.proposal_execution { kind: dflash_flat_block, version }",
        normative_reference: "§13",
        admission_consumer: "onnx_genai_engine::WorkflowExecutionAdmission",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Partial,
    },
    ExtensionDescriptor {
        id: DFLASH_FLAT_BLOCK_V2,
        surface: ExtensionSurface::Speculative,
        schema_floor: DFLASH_SCHEMA_VERSION,
        declaration: "speculative.proposal_execution { kind: dflash_flat_block, version }",
        normative_reference: "§13",
        admission_consumer: "onnx_genai_engine::WorkflowExecutionAdmission",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::KnownButUnavailable,
    },
    ExtensionDescriptor {
        id: GRAMMAR_GUIDANCE_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§7",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: TELEMETRY_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§7",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: PARAMETER_OVERLAY_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§7, §9",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: IMAGE_PREPROCESS_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§10, §15",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: VIDEO_PREPROCESS_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§10, §15",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: AUDIO_PREPROCESS_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§10, §15",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: TEXT_ASSEMBLY_V1,
        surface: ExtensionSurface::ComponentAdapter,
        schema_floor: V1,
        declaration: "pipeline.workflow.components.<name>.implementation { kind: adapter }",
        normative_reference: "§7",
        admission_consumer: "onnx_genai_engine::pipeline::workflow",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ADAPTERS_V1,
        surface: ExtensionSurface::PackageAdapter,
        schema_floor: V1,
        declaration: "adapters.application_capability",
        normative_reference: "§9",
        admission_consumer: "onnx_genai_metadata::validate_adapter_service",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ADAPTERS_JSON_V1,
        surface: ExtensionSurface::AdapterLoader,
        schema_floor: V1,
        declaration: "adapters.artifacts.<name>.weights[].loader_capability",
        normative_reference: "§9",
        admission_consumer: "onnx_genai_metadata::validate_adapter_service",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ORT_LORA_ADAPTER_V1,
        surface: ExtensionSurface::AdapterLoader,
        schema_floor: V1,
        declaration: "adapters.artifacts.<name>.weights[].loader_capability",
        normative_reference: "§9",
        admission_consumer: "onnx_genai_metadata::validate_adapter_service",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ADAPTERS_HF_PEFT_V1,
        surface: ExtensionSurface::AdapterLoader,
        schema_floor: V1,
        declaration: "adapters.artifacts.<name>.weights[].loader_capability",
        normative_reference: "§9",
        admission_consumer: "onnx_genai_metadata::validate_adapter_service",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: ADAPTERS_SAFETENSORS_V1,
        surface: ExtensionSurface::AdapterLoader,
        schema_floor: V1,
        declaration: "adapters.artifacts.<name>.weights[].loader_capability",
        normative_reference: "§9",
        admission_consumer: "onnx_genai_metadata::validate_adapter_service",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::Implemented,
    },
    ExtensionDescriptor {
        id: KV_CHECKPOINT_V1,
        surface: ExtensionSurface::StateCheckpoint,
        schema_floor: V1,
        declaration: "pipeline.workflow.serving.state_service.groups.<name>.checkpoint",
        normative_reference: "§12.6",
        admission_consumer: "runtime checkpoint adapter registration",
        fallback: FallbackClass::SemanticRequired,
        status: SupportStatus::KnownButUnavailable,
    },
];

/// The schema-version obligations a conforming reader owns without negotiation.
pub const CORE_CONFORMANCE: &[(SchemaVersion, &str)] = &[
    (
        V1,
        "typed workflow SSA/dataflow, tensor shape as the sole rank authority, typed state, effects, and validation invariants",
    ),
    (
        crate::version::BATCHING_SCHEMA_VERSION,
        "batch layouts, padding/ownership levels, and video preprocessing syntax",
    ),
    (
        crate::version::TOKEN_AUTHORITY_SCHEMA_VERSION,
        "package-owned numeric token authority and explicit EOS policy",
    ),
    (
        TOOL_PROTOCOL_SCHEMA_VERSION,
        "the typed tool-protocol declaration surface (the selected protocol remains an extension)",
    ),
    (
        TOKEN_CONTEXT_SCHEMA_VERSION,
        "the typed token-context declaration surface (the module remains an extension)",
    ),
    (
        crate::version::OUTPUT_PROTOCOL_SCHEMA_VERSION,
        "output families, stream identities, and typed revision operations",
    ),
    (
        CANONICAL_SPECULATION_SCHEMA_VERSION,
        "the typed canonical speculative declaration surface (the selected proposer remains an extension)",
    ),
];

/// Look up a built-in extension by its exact declared identity and version.
pub fn find(identity: &str, version: &str) -> Option<&'static ExtensionDescriptor> {
    BUILTIN_EXTENSIONS
        .iter()
        .find(|descriptor| descriptor.id.identity == identity && descriptor.id.version == version)
}

/// Render the committed, discoverable extension registry.
pub fn extension_registry_markdown() -> String {
    use std::fmt::Write;

    let mut document = String::from(
        "# Metadata extension registry\n\n\
This generated document is the complete built-in registry for **optional, \
versioned semantic modules**. Its source is \
`onnx_genai_metadata::extensions::BUILTIN_EXTENSIONS`; do not edit this file \
by hand. The normative rules live in \
[`INFERENCE_METADATA_DECISIONS.md`](INFERENCE_METADATA_DECISIONS.md).\n\n\
## Core conformance is not a capability\n\n\
A reader that accepts a schema version must implement that version's typed \
SSA/dataflow, shape/rank, state, output, transaction, and validation rules. \
Those obligations are not package flags and are never advertised or negotiated \
as extensions.\n\n\
| Schema floor | Reader obligations added at that floor |\n\
| --- | --- |\n",
    );
    for (version, obligations) in CORE_CONFORMANCE {
        writeln!(document, "| `{version}` | {obligations} |")
            .expect("writing to String cannot fail");
    }

    document.push_str(
        "\n## Optional semantic extensions\n\n\
Packages declare an extension only through its typed declaration location. The \
reader selects the exact identity/version at the named admission consumer; \
otherwise it rejects before execution. `partial` and `unavailable` rows are \
not fallback permission.\n\n\
| Identity | Surface | Schema floor | Declaration location | Normative reference | Admission consumer | Fallback | Current status |\n\
| --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for descriptor in BUILTIN_EXTENSIONS {
        writeln!(
            document,
            "| `{}@{}` | {} | `{}` | `{}` | {} | `{}` | {} | {} |",
            descriptor.id.identity,
            descriptor.id.version,
            descriptor.surface.as_str(),
            descriptor.schema_floor,
            descriptor.declaration,
            descriptor.normative_reference,
            descriptor.admission_consumer,
            descriptor.fallback.as_str(),
            descriptor.status.as_str(),
        )
        .expect("writing to String cannot fail");
    }

    document.push_str(
        "\n## Open extension surfaces\n\n\
The registry lists built-ins only. A producer may use a namespaced external \
identity at a typed surface (`component.contract`, `implementation.adapter`, \
`package.tool_protocol`, constraint dialect, profile kind, or checkpoint \
adapter), with an exact version. A runtime that has no matching implementation \
must fail closed; parsing a string is not admission.\n\n\
## Runtime optimizations and policies\n\n\
Continuous batching, graph capture, scheduling, placement, storage tiering, \
allocation budgets, and kernel selection are runtime decisions, not package \
extension identifiers. Typed component/state contracts determine whether an \
optimization preserves semantics; if it does not run, the runtime uses a \
generic or isolated path where one is semantically valid. They therefore have \
the `optimization-optional` fallback class and do not appear in the table \
above.\n\n\
## Author and reader guidance\n\n\
Authors: use a core schema field for standard semantics, a typed exact \
identity/version for an optional module, and no metadata field for deployment \
policy. Readers: gate the schema version first, resolve each required semantic \
extension exactly at its declared consumer, and reject unknown or ambiguous \
pairs before mutation. Do not infer behavior from model, vendor, backend, \
filename, tensor name, or optimization availability.\n",
    );
    document
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_exact_semantic_requirements() {
        assert!(BUILTIN_EXTENSIONS.iter().all(|descriptor| {
            descriptor.fallback == FallbackClass::SemanticRequired
                && !descriptor.id.identity.is_empty()
                && !descriptor.id.version.is_empty()
                && descriptor.schema_floor <= crate::version::SUPPORTED_SCHEMA_VERSION
        }));
    }

    #[test]
    fn generated_registry_has_no_core_capability_entries() {
        let registry = extension_registry_markdown();
        for core in [
            "workflow_ssa",
            "typed_emit",
            "continuous_batching",
            "streaming_emit",
        ] {
            assert!(
                !registry.contains(&format!("`{core}@")),
                "{core} is core or runtime policy, not an extension"
            );
        }
    }
}
