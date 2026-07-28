//! Consume and validate `onnx_runtime.*` model metadata hints.
//!
//! ONNX models can embed execution hints directly in the graph using the
//! `onnx_runtime.` metadata namespace (see `docs/MODEL_METADATA.md`). These
//! hints are the fourth and lowest-priority source of execution guidance,
//! written by the model author or export tool into `ModelProto.metadata_props`,
//! `GraphProto.metadata_props`, and `NodeProto.metadata_props`.
//!
//! This module turns those raw `key → value` string pairs into a validated,
//! typed [`MetadataHints`] structure. The behaviour matches the "Validation"
//! and "Priority Resolution" sections of `docs/MODEL_METADATA.md`:
//!
//! 1. Scan every `onnx_runtime.*` key at the model, graph, and node levels.
//! 2. Warn on keys that are not recognised at their level (typo detection).
//! 3. Validate each value against the type the key declares (int, bool, enum,
//!    string), warning on values that fail to parse.
//! 4. Resolve the winning value when several sources set the same hint, using
//!    the documented source-priority order.
//! 5. Report contradicting `force` device hints as hard errors.
//!
//! The scanner is deliberately source-agnostic: [`MetadataHints::scan`] takes an
//! iterator of [`HintEntry`] values tagged with a [`HintSource`], so hints from
//! a programmatic builder API, an `execution_hints.json` file, or an
//! `inference_metadata.yaml` section can be merged with the model's own
//! metadata through the same priority and validation logic. The convenience
//! constructor [`MetadataHints::from_model`] feeds only the model's embedded
//! `onnx_runtime.*` metadata, which is what the load path has available today.

use std::collections::BTreeMap;

use crate::model::Model;

/// The namespace prefix every runtime hint key carries (`onnx_runtime.`).
///
/// Keeps our keys from colliding with `onnx.` (reserved by the ONNX spec),
/// `com.microsoft.` (ORT internal), or any other runtime's namespace.
pub const NAMESPACE_PREFIX: &str = "onnx_runtime.";

/// Where a hint originated, in descending priority order.
///
/// When more than one source sets the same key, the value from the
/// highest-priority source wins. Mirrors the "Priority Resolution" table in
/// `docs/MODEL_METADATA.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HintSource {
    /// Embedded `onnx_runtime.*` metadata on the ONNX model (lowest priority).
    OnnxMetadata,
    /// The `execution_hints` section of an `inference_metadata.yaml` package.
    InferenceMetadataYaml,
    /// A user-supplied `execution_hints.json` file.
    ExecutionHintsJson,
    /// A programmatic builder API call (`.placement_hint(...)`, highest).
    ProgrammaticBuilder,
}

impl HintSource {
    /// Numeric priority; higher wins. Derived from declaration order so the
    /// ordering stays in one place.
    fn priority(self) -> u8 {
        match self {
            HintSource::OnnxMetadata => 0,
            HintSource::InferenceMetadataYaml => 1,
            HintSource::ExecutionHintsJson => 2,
            HintSource::ProgrammaticBuilder => 3,
        }
    }
}

/// The graph location a metadata entry was attached to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintScope {
    /// `ModelProto.metadata_props` — applies to the whole model.
    Model,
    /// `GraphProto.metadata_props` — applies to a named graph. Graph-level and
    /// model-level keys share one namespace, so both resolve together.
    Graph {
        /// The graph's name (may be empty for an unnamed graph).
        graph_name: String,
    },
    /// `NodeProto.metadata_props` — applies to a single node.
    Node {
        /// The node's name (may be empty for an unnamed node).
        node_name: String,
    },
}

/// A single raw `key = value` metadata entry to be scanned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintEntry {
    /// Where the entry was attached.
    pub scope: HintScope,
    /// Which source produced it.
    pub source: HintSource,
    /// The full key, including the `onnx_runtime.` prefix.
    pub key: String,
    /// The raw string value as stored in the model.
    pub value: String,
}

/// The value type a hint key declares, used to validate raw strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HintValueType {
    /// Free-form string (device names, kernel names, group ids, patterns).
    Text,
    /// Boolean parsed from `true` / `false` (case-insensitive).
    Boolean,
    /// Signed 64-bit integer.
    Integer,
    /// One of a fixed set of string tokens.
    Enumerated(&'static [&'static str]),
}

impl HintValueType {
    /// A short human-readable description of what a valid value looks like,
    /// used for the `expected` field of [`MetadataWarning::InvalidValue`].
    fn expected(self) -> &'static str {
        match self {
            HintValueType::Text => "a string",
            HintValueType::Boolean => "a boolean (\"true\" or \"false\")",
            HintValueType::Integer => "an integer",
            HintValueType::Enumerated(allowed) => match allowed {
                ["prefer", "force"] => "one of: prefer, force",
                ["high", "low", "normal"] => "one of: high, low, normal",
                _ => "one of a fixed set of tokens",
            },
        }
    }

    /// Parse `raw` into a [`HintValue`], or `None` if it does not match.
    fn parse(self, raw: &str) -> Option<HintValue> {
        match self {
            HintValueType::Text => Some(HintValue::Text(raw.to_string())),
            HintValueType::Boolean => match raw.to_ascii_lowercase().as_str() {
                "true" => Some(HintValue::Boolean(true)),
                "false" => Some(HintValue::Boolean(false)),
                _ => None,
            },
            HintValueType::Integer => raw.trim().parse::<i64>().ok().map(HintValue::Integer),
            HintValueType::Enumerated(allowed) => allowed
                .iter()
                .find(|token| **token == raw)
                .map(|token| HintValue::Enumerated((*token).to_string())),
        }
    }
}

/// A validated, typed hint value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintValue {
    /// A free-form string value.
    Text(String),
    /// A parsed boolean.
    Boolean(bool),
    /// A parsed integer.
    Integer(i64),
    /// A validated enumeration token.
    Enumerated(String),
}

impl HintValue {
    /// The string payload of a [`HintValue::Text`] or
    /// [`HintValue::Enumerated`], if this value is one of those.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            HintValue::Text(text) | HintValue::Enumerated(text) => Some(text.as_str()),
            _ => None,
        }
    }

    /// The boolean payload of a [`HintValue::Boolean`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HintValue::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    /// The integer payload of a [`HintValue::Integer`].
    pub fn as_int(&self) -> Option<i64> {
        match self {
            HintValue::Integer(value) => Some(*value),
            _ => None,
        }
    }
}

/// A single recognised hint key and the value type it expects.
struct KnownHint {
    key: &'static str,
    value_type: HintValueType,
}

/// Strength enum shared by `onnx_runtime.device.strength` and colocation.
const STRENGTH_TOKENS: &[&str] = &["prefer", "force"];
/// Eviction-priority enum for `onnx_runtime.memory.priority`.
const MEMORY_PRIORITY_TOKENS: &[&str] = &["high", "low", "normal"];

/// Keys recognised on `NodeProto.metadata_props`.
const NODE_HINTS: &[KnownHint] = &[
    KnownHint {
        key: "onnx_runtime.device",
        value_type: HintValueType::Text,
    },
    KnownHint {
        key: "onnx_runtime.device.strength",
        value_type: HintValueType::Enumerated(STRENGTH_TOKENS),
    },
    KnownHint {
        key: "onnx_runtime.memory.pin",
        value_type: HintValueType::Boolean,
    },
    KnownHint {
        key: "onnx_runtime.memory.priority",
        value_type: HintValueType::Enumerated(MEMORY_PRIORITY_TOKENS),
    },
    KnownHint {
        key: "onnx_runtime.scheduling.cuda_graph",
        value_type: HintValueType::Boolean,
    },
    KnownHint {
        key: "onnx_runtime.scheduling.overlap",
        value_type: HintValueType::Boolean,
    },
    KnownHint {
        key: "onnx_runtime.group",
        value_type: HintValueType::Text,
    },
    KnownHint {
        key: "onnx_runtime.layer",
        value_type: HintValueType::Integer,
    },
    KnownHint {
        key: "onnx_runtime.offloadable",
        value_type: HintValueType::Boolean,
    },
    KnownHint {
        key: "onnx_runtime.kernel",
        value_type: HintValueType::Text,
    },
];

/// Keys recognised on `GraphProto.metadata_props` or
/// `ModelProto.metadata_props`.
const GRAPH_HINTS: &[KnownHint] = &[
    KnownHint {
        key: "onnx_runtime.model.num_layers",
        value_type: HintValueType::Integer,
    },
    KnownHint {
        key: "onnx_runtime.model.layer_pattern",
        value_type: HintValueType::Text,
    },
    KnownHint {
        key: "onnx_runtime.model.architecture",
        value_type: HintValueType::Text,
    },
    KnownHint {
        key: "onnx_runtime.memory.arena_gpu_mb",
        value_type: HintValueType::Integer,
    },
    KnownHint {
        key: "onnx_runtime.memory.arena_cpu_mb",
        value_type: HintValueType::Integer,
    },
    KnownHint {
        key: "onnx_runtime.memory.prefetch",
        value_type: HintValueType::Text,
    },
    KnownHint {
        key: "onnx_runtime.version",
        value_type: HintValueType::Text,
    },
];

fn lookup(registry: &'static [KnownHint], key: &str) -> Option<HintValueType> {
    registry
        .iter()
        .find(|hint| hint.key == key)
        .map(|hint| hint.value_type)
}

/// A problem found while scanning `onnx_runtime.*` metadata.
///
/// Field shapes match the enum documented in `docs/MODEL_METADATA.md`. For
/// model-level and graph-level entries the `node` field carries the graph name
/// (empty for the top-level or an unnamed graph) rather than a node name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetadataWarning {
    /// A key in the `onnx_runtime.` namespace that is not recognised at the
    /// location it appears (likely a typo or a misplaced key).
    UnknownKey {
        /// Location label: the node name, or the graph name for graph/model
        /// level entries.
        node: String,
        /// The full offending key.
        key: String,
    },
    /// A recognised key whose value does not parse as the declared type.
    InvalidValue {
        /// Location label (see [`MetadataWarning::UnknownKey`]).
        node: String,
        /// The full key.
        key: String,
        /// The raw value that failed to parse.
        value: String,
        /// A description of what a valid value looks like.
        expected: &'static str,
    },
    /// Two sources force the same node onto different devices — unsatisfiable.
    ConflictingForce {
        /// The node the conflict is on.
        node: String,
        /// One contributing source.
        source_a: HintSource,
        /// The other contributing source.
        source_b: HintSource,
    },
}

impl MetadataWarning {
    /// Whether this warning is a hard error. Contradicting `force` device
    /// hints are unsatisfiable and therefore fatal; unknown keys and malformed
    /// values are advisory.
    pub fn is_error(&self) -> bool {
        matches!(self, MetadataWarning::ConflictingForce { .. })
    }
}

/// The strength of a device placement hint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlacementStrength {
    /// Advisory placement; a higher-priority source may override it.
    #[default]
    Prefer,
    /// Mandatory placement; contradicting forces are an error.
    Force,
}

/// Resolved node-level execution hints.
///
/// Every field is optional: an absent hint leaves the field `None`, so callers
/// get a safe default rather than a synthesised value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeHints {
    /// Preferred device string, e.g. `"gpu"`, `"gpu:0"`, `"cpu"`, `"npu"`.
    pub device: Option<String>,
    /// Placement strength; only meaningful alongside [`NodeHints::device`].
    pub device_strength: Option<PlacementStrength>,
    /// Pin this node's output tensors in memory.
    pub memory_pin: Option<bool>,
    /// Eviction priority (`high`, `low`, `normal`).
    pub memory_priority: Option<String>,
    /// Include this node in a CUDA graph capture region.
    pub cuda_graph: Option<bool>,
    /// Allow this node to overlap with adjacent ops.
    pub overlap: Option<bool>,
    /// Colocation group; nodes sharing a group stay on one device.
    pub group: Option<String>,
    /// Logical transformer-layer index.
    pub layer: Option<i64>,
    /// Whether this node may be offloaded to CPU under GPU pressure.
    pub offloadable: Option<bool>,
    /// Preferred kernel implementation.
    pub kernel: Option<String>,
}

/// Resolved model-level and graph-level execution hints.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelHints {
    /// Total transformer layers (enables layer-range hints).
    pub num_layers: Option<i64>,
    /// Naming pattern for layer nodes, e.g. `"model.layers.{}"`.
    pub layer_pattern: Option<String>,
    /// Model architecture hint, e.g. `"llama"`.
    pub architecture: Option<String>,
    /// Suggested GPU arena size in MB.
    pub arena_gpu_mb: Option<i64>,
    /// Suggested CPU arena size in MB.
    pub arena_cpu_mb: Option<i64>,
    /// Comma-separated tensor names to prefetch.
    pub prefetch: Option<String>,
    /// Metadata schema version string.
    pub version: Option<String>,
}

/// The full result of scanning a model's `onnx_runtime.*` metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetadataHints {
    /// Resolved model-level / graph-level hints.
    pub model: ModelHints,
    /// Resolved per-node hints, keyed by node name.
    pub nodes: BTreeMap<String, NodeHints>,
    /// Warnings and errors gathered during the scan, in discovery order.
    pub warnings: Vec<MetadataWarning>,
}

impl MetadataHints {
    /// Whether the scan produced any hard errors (contradicting `force` hints).
    pub fn has_errors(&self) -> bool {
        self.warnings.iter().any(MetadataWarning::is_error)
    }

    /// Scan an arbitrary set of hint entries from any mix of sources.
    ///
    /// Entries are validated against the key registry, then the winning value
    /// for each key is chosen by source priority. This is the general entry
    /// point; [`MetadataHints::from_model`] is the model-only convenience.
    pub fn scan(entries: impl IntoIterator<Item = HintEntry>) -> Self {
        let mut scanner = Scanner::default();
        for entry in entries {
            scanner.ingest(entry);
        }
        scanner.finish()
    }

    /// Scan the `onnx_runtime.*` metadata embedded in a loaded [`Model`].
    ///
    /// Reads model-level `metadata_props`, and — when the source protobuf is
    /// retained — graph-level and node-level `metadata_props`. All entries are
    /// tagged [`HintSource::OnnxMetadata`], the lowest priority.
    pub fn from_model(model: &Model) -> Self {
        Self::scan(model_hint_entries(model))
    }
}

/// The winning contribution for one `(scope, key)` pair.
struct Winner {
    source: HintSource,
    value: HintValue,
}

/// One source's device placement for a node: the device it named and the
/// strength that source attached to it (defaulting to `prefer`).
#[derive(Default)]
struct DevicePlacement {
    device: Option<String>,
    strength: Option<PlacementStrength>,
}

#[derive(Default)]
struct Scanner {
    model: ModelHints,
    nodes: BTreeMap<String, NodeHints>,
    warnings: Vec<MetadataWarning>,
    /// Highest-priority winner seen so far for each `(node_key, full_key)`.
    /// A `None` node_key denotes model/graph level.
    winners: BTreeMap<(Option<String>, String), Winner>,
    /// Per-node, per-source device placement. Keeping device and strength
    /// paired by source is what lets a `force` from one source override a
    /// `prefer` from another and lets contradicting forces be detected.
    device_placements: BTreeMap<String, BTreeMap<HintSource, DevicePlacement>>,
}

impl Scanner {
    fn ingest(&mut self, entry: HintEntry) {
        if !entry.key.starts_with(NAMESPACE_PREFIX) {
            return;
        }
        let (registry, node_label, node_key) = match &entry.scope {
            HintScope::Model => (GRAPH_HINTS, String::new(), None),
            HintScope::Graph { graph_name } => (GRAPH_HINTS, graph_name.clone(), None),
            HintScope::Node { node_name } => {
                (NODE_HINTS, node_name.clone(), Some(node_name.clone()))
            }
        };

        let Some(value_type) = lookup(registry, &entry.key) else {
            self.warnings.push(MetadataWarning::UnknownKey {
                node: node_label,
                key: entry.key,
            });
            return;
        };

        let Some(typed) = value_type.parse(&entry.value) else {
            self.warnings.push(MetadataWarning::InvalidValue {
                node: node_label,
                key: entry.key,
                value: entry.value,
                expected: value_type.expected(),
            });
            return;
        };

        // Device placement is resolved specially: a `force` overrides a
        // `prefer` regardless of source priority. Device and strength are kept
        // paired per source so each source's intent stays intact until the end.
        if let HintScope::Node { node_name } = &entry.scope {
            match (entry.key.as_str(), &typed) {
                ("onnx_runtime.device", HintValue::Text(device)) => {
                    self.device_placements
                        .entry(node_name.clone())
                        .or_default()
                        .entry(entry.source)
                        .or_default()
                        .device = Some(device.clone());
                }
                ("onnx_runtime.device.strength", HintValue::Enumerated(token)) => {
                    self.device_placements
                        .entry(node_name.clone())
                        .or_default()
                        .entry(entry.source)
                        .or_default()
                        .strength = Some(parse_strength(token));
                }
                _ => {}
            }
        }

        self.record_winner(node_key, entry.key, entry.source, typed);
    }

    /// Keep the highest-priority contribution for a `(node, key)` pair.
    fn record_winner(
        &mut self,
        node_key: Option<String>,
        key: String,
        source: HintSource,
        value: HintValue,
    ) {
        let slot = (node_key, key);
        match self.winners.get(&slot) {
            Some(existing) if existing.source.priority() >= source.priority() => {}
            _ => {
                self.winners.insert(slot, Winner { source, value });
            }
        }
    }

    fn finish(mut self) -> MetadataHints {
        self.settle_device_placement();
        self.apply_winners();
        MetadataHints {
            model: self.model,
            nodes: self.nodes,
            warnings: self.warnings,
        }
    }

    /// Choose the effective device per node from each source's paired
    /// device/strength placement, and flag contradicting forces.
    fn settle_device_placement(&mut self) {
        let placements = std::mem::take(&mut self.device_placements);
        for (node_name, by_source) in placements {
            // One (source, device, strength) contribution per source that named
            // a device. A source that only set a strength contributes nothing.
            let contributions: Vec<(HintSource, String, PlacementStrength)> = by_source
                .into_iter()
                .filter_map(|(source, placement)| {
                    placement
                        .device
                        .map(|device| (source, device, placement.strength.unwrap_or_default()))
                })
                .collect();
            if contributions.is_empty() {
                continue;
            }

            let forced: Vec<&(HintSource, String, PlacementStrength)> = contributions
                .iter()
                .filter(|(_, _, strength)| *strength == PlacementStrength::Force)
                .collect();
            if let Some(first) = forced.first()
                && let Some(conflicting) = forced.iter().find(|(_, device, _)| *device != first.1)
            {
                self.warnings.push(MetadataWarning::ConflictingForce {
                    node: node_name.clone(),
                    source_a: first.0,
                    source_b: conflicting.0,
                });
            }

            // A force wins over any prefer; otherwise the highest-priority
            // source wins.
            let effective = contributions
                .iter()
                .max_by_key(|(source, _, strength)| {
                    (
                        (*strength == PlacementStrength::Force) as u8,
                        source.priority(),
                    )
                })
                .map(|(_, device, strength)| (device.clone(), *strength));
            if let Some((device, resolved_strength)) = effective {
                let node = self.nodes.entry(node_name.clone()).or_default();
                node.device = Some(device);
                node.device_strength = Some(resolved_strength);
            }
        }
    }

    fn apply_winners(&mut self) {
        let winners = std::mem::take(&mut self.winners);
        for ((node_key, key), winner) in winners {
            match node_key {
                None => apply_model_hint(&mut self.model, &key, winner.value),
                Some(node_name) => {
                    // Device fields were already settled with strength/force
                    // semantics; skip re-applying the raw winner for them.
                    if key == "onnx_runtime.device" || key == "onnx_runtime.device.strength" {
                        continue;
                    }
                    let node = self.nodes.entry(node_name).or_default();
                    apply_node_hint(node, &key, winner.value);
                }
            }
        }
    }
}

fn parse_strength(token: &str) -> PlacementStrength {
    match token {
        "force" => PlacementStrength::Force,
        _ => PlacementStrength::Prefer,
    }
}

fn apply_model_hint(model: &mut ModelHints, key: &str, value: HintValue) {
    match key {
        "onnx_runtime.model.num_layers" => model.num_layers = value.as_int(),
        "onnx_runtime.model.layer_pattern" => {
            model.layer_pattern = value.as_str().map(str::to_string)
        }
        "onnx_runtime.model.architecture" => {
            model.architecture = value.as_str().map(str::to_string)
        }
        "onnx_runtime.memory.arena_gpu_mb" => model.arena_gpu_mb = value.as_int(),
        "onnx_runtime.memory.arena_cpu_mb" => model.arena_cpu_mb = value.as_int(),
        "onnx_runtime.memory.prefetch" => model.prefetch = value.as_str().map(str::to_string),
        "onnx_runtime.version" => model.version = value.as_str().map(str::to_string),
        _ => {}
    }
}

fn apply_node_hint(node: &mut NodeHints, key: &str, value: HintValue) {
    match key {
        "onnx_runtime.memory.pin" => node.memory_pin = value.as_bool(),
        "onnx_runtime.memory.priority" => node.memory_priority = value.as_str().map(str::to_string),
        "onnx_runtime.scheduling.cuda_graph" => node.cuda_graph = value.as_bool(),
        "onnx_runtime.scheduling.overlap" => node.overlap = value.as_bool(),
        "onnx_runtime.group" => node.group = value.as_str().map(str::to_string),
        "onnx_runtime.layer" => node.layer = value.as_int(),
        "onnx_runtime.offloadable" => node.offloadable = value.as_bool(),
        "onnx_runtime.kernel" => node.kernel = value.as_str().map(str::to_string),
        _ => {}
    }
}

/// Collect every `onnx_runtime.*` entry embedded in a loaded model.
fn model_hint_entries(model: &Model) -> Vec<HintEntry> {
    let mut entries = Vec::new();
    for (key, value) in &model.metadata.metadata_props {
        if key.starts_with(NAMESPACE_PREFIX) {
            entries.push(HintEntry {
                scope: HintScope::Model,
                source: HintSource::OnnxMetadata,
                key: key.clone(),
                value: value.clone(),
            });
        }
    }

    if let Some(proto) = model.retained_proto()
        && let Some(graph) = proto.graph.as_ref()
    {
        for entry in &graph.metadata_props {
            if entry.key.starts_with(NAMESPACE_PREFIX) {
                entries.push(HintEntry {
                    scope: HintScope::Graph {
                        graph_name: graph.name.clone(),
                    },
                    source: HintSource::OnnxMetadata,
                    key: entry.key.clone(),
                    value: entry.value.clone(),
                });
            }
        }
        for node in &graph.node {
            for entry in &node.metadata_props {
                if entry.key.starts_with(NAMESPACE_PREFIX) {
                    entries.push(HintEntry {
                        scope: HintScope::Node {
                            node_name: node.name.clone(),
                        },
                        source: HintSource::OnnxMetadata,
                        key: entry.key.clone(),
                        value: entry.value.clone(),
                    });
                }
            }
        }
    }

    entries
}

#[cfg(test)]
mod tests;
