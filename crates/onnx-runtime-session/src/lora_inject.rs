//! Load-time native-LoRA injection (design `docs/NATIVE_LORA_DESIGN.md` §B, §C).
//!
//! Two cooperating passes, both operating on the freshly-loaded [`Graph`] IR,
//! mirroring the structure of the sibling [`crate::fp16_decode`] rewrite:
//!
//! 1. [`build_manifest`] (design §C, §H — **P3**) walks the graph and maps each
//!    semantic PEFT target module (`q_proj`, `k_proj`, …) to a concrete graph
//!    target: the base [`MatMulNBits`](https://onnx.ai) node, its output value
//!    (the injection point), the activation input, and the inner/outer dims. It
//!    detects — **structurally**, not by name guessing — whether a projection is
//!    kept split (Qwen3), fused into one `qkv_proj` (Qwen2.5), or is a layout it
//!    does not recognize (Qwen3.5 linear attention), and **fails loud** on any
//!    module it cannot resolve to exactly one node + slice.
//!
//! 2. [`inject`] (design §B — **P2b**) rewrites the graph, adding a separate
//!    fp16/fp32 delta branch onto each targeted projection's output:
//!    `r = MatMul(x, A_t)`, `delta = MatMul(r, B_t)`, `scaled = Mul(delta, s)`,
//!    `Y = Add(Y_base, scaled)`, rewiring the base output's consumers onto `Y`.
//!    The base `MatMulNBits` is never touched (no requantize). `A_t`/`B_t` are
//!    exposed as **overridable optional graph inputs** (design §B.3, P1) with
//!    zero-rank defaults (`[K, 0]` / `[0, N]`, empty bytes) so that, when unfed,
//!    the delta is a provable no-op and the graph runs base-only. A fused
//!    `qkv_proj` receives one branch per Q/K/V slice, concatenated in verified
//!    GQA order so each PEFT factor lands on the correct output columns.
//!
//! # Crate placement and the format-agnostic boundary
//!
//! This lives in `onnx-runtime-session` (not `onnx-genai-engine`) because the
//! rewrite needs the IR and the [`Executor::build_with_overrides`] override
//! mechanism, and because `onnx-genai-engine` depends on `onnx-runtime-session`
//! and not the reverse — the pass therefore cannot reference the engine's PEFT
//! `LoadedAdapter`. Instead it consumes a small, adapter-format-agnostic
//! [`LoraAdapterSpec`] built from plain [`TensorData`]. The engine's future
//! `LoraManager` (P4) is responsible for translating a parsed PEFT adapter into
//! this spec (already-transposed `A_t = [K, r]`, `B_t = [r, N]`). The session
//! crate stays LoRA-format-agnostic, exactly as the executor's `OptionalOverride`
//! carries nothing LoRA-specific.
//!
//! [`Executor::build_with_overrides`]: crate::executor
//! [`MatMulNBits`]: https://github.com/microsoft/onnxruntime

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use onnx_runtime_ep_api::{
    AdapterId, LoraFactorInput, LoraModuleId, LoraPoolId, LoraPoolRegistration,
    LoraPoolRegistry, LoraWeightPool,
};
use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
};

/// The op type of every int4 base projection this pass injects onto. Both the
/// Qwen2.5 (`.../MatMul_Q4` node name, `com.microsoft` domain) and Qwen3
/// (`.../MatMulNBits` node name) exports use this same op type; only the node
/// and weight *names* differ (design §H), which is why discovery keys on the op
/// type and structural dims rather than on any name pattern.
const BASE_OP: &str = "MatMulNBits";

/// The attention op whose head configuration determines a fused `qkv_proj`'s
/// Q/K/V slice widths (design §H: `num_heads` / `kv_num_heads`).
const ATTENTION_OP: &str = "GroupQueryAttention";

// ===========================================================================
// Public spec — the adapter-format-agnostic input to the injection pass.
// ===========================================================================

/// One targeted projection of a LoRA adapter, oriented for ONNX `MatMul`
/// (already transposed at load, design §A): `a_t` is `[K, rank]` and `b_t` is
/// `[rank, N]`, where `N` is *this module's* output width (the slice width for a
/// fused target). The A/B bytes are validated for shape/dtype here but are fed
/// at run time through the override inputs, not baked into the graph; only
/// `scale` is baked as a constant.
#[derive(Clone, Debug)]
pub struct LoraModuleSpec {
    /// Semantic PEFT module name after the layer index, e.g. `self_attn.q_proj`
    /// or `mlp.gate_proj`. Only the trailing `*_proj` leaf token is significant
    /// for resolution; any block prefix (`self_attn`, `attn`, `mlp`, …) is
    /// ignored so the same adapter resolves across export naming variants.
    pub module_name: String,
    /// Decoder layer index this module belongs to.
    pub layer_index: usize,
    /// LoRA rank `r` (must equal `a_t.dims[1] == b_t.dims[0]`).
    pub rank: usize,
    /// Per-module scale `alpha / rank` (design §A). Baked as a constant.
    pub scale: f32,
    /// Contiguous `[K, rank]` factor.
    pub a_t: TensorData,
    /// Contiguous `[rank, N]` factor.
    pub b_t: TensorData,
}

/// A decoded adapter reduced to the graph-injection essentials.
#[derive(Clone, Debug)]
pub struct LoraAdapterSpec {
    /// Human-readable adapter name (diagnostics only).
    pub name: String,
    /// The targeted modules, one per PEFT `q_proj`/`k_proj`/… factor pair.
    pub modules: Vec<LoraModuleSpec>,
}

/// The subset of a [`LoraAdapterSpec`] that [`build_manifest`] needs to resolve
/// a target: which module, in which layer.
#[derive(Clone, Debug)]
pub struct LoraTarget {
    pub module_name: String,
    pub layer_index: usize,
}

// ===========================================================================
// Manifest — the graph-derived target map (design §C).
// ===========================================================================

/// Q/K/V role within a fused `qkv_proj` projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QkvRole {
    Q,
    K,
    V,
}

impl QkvRole {
    fn as_str(self) -> &'static str {
        match self {
            QkvRole::Q => "q_proj",
            QkvRole::K => "k_proj",
            QkvRole::V => "v_proj",
        }
    }
}

/// The geometry of a fused `qkv_proj` projection: the fused output width and the
/// three ordered Q/K/V output slices, derived structurally from the layer's
/// [`GroupQueryAttention`](ATTENTION_OP) head configuration (design §H). The
/// `order` is always `[Q, K, V]` — the GQA-packed layout — and the offsets are
/// validated to sum to `fused_n`, so a wrong offset can never silently corrupt
/// attention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedGroup {
    /// The fused base node.
    pub node_id: NodeId,
    /// The fused projection's total output width `N = N_q + N_k + N_v`.
    pub fused_n: usize,
    /// The three ordered `(role, offset, width)` slices, in `[Q, K, V]` order.
    pub slices: [(QkvRole, usize, usize); 3],
}

/// Where a targeted module's delta lands on its base node's output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// A standalone projection: the delta spans the whole `[.., N]` output.
    Direct,
    /// One Q/K/V slice of a fused `qkv_proj`: `group` carries the full fused
    /// geometry and `role` is this module's slice.
    FusedSlice { group: FusedGroup, role: QkvRole },
}

/// A resolved, validated mapping of one semantic PEFT module to a concrete graph
/// target (design §C).
#[derive(Clone, Debug)]
pub struct TargetEntry {
    /// Semantic identity, e.g. `layers.0.q_proj` (diagnostics).
    pub semantic: String,
    /// The base `MatMulNBits` node the delta attaches to.
    pub node_id: NodeId,
    /// The base node's output value — the injection point.
    pub base_output: ValueId,
    /// The base node's activation input `x` (input slot 0).
    pub activation: ValueId,
    /// Inner dimension `K` (input features), from the base node's `K` attribute.
    pub k: usize,
    /// This module's output width `N` (the slice width when fused).
    pub n: usize,
    /// The branch element type — the activation value's dtype.
    pub dtype: DataType,
    /// Whether the target is standalone or one slice of a fused projection.
    pub placement: Placement,
}

/// The per-model manifest: one [`TargetEntry`] per requested module, in the same
/// order as the input targets.
#[derive(Clone, Debug)]
pub struct LoraManifest {
    pub entries: Vec<TargetEntry>,
}

/// One concrete adapter feed produced by the injection: the name of an
/// overridable optional graph input (`A_t` or `B_t` of a targeted module) paired
/// with the exact factor bytes to bind when the adapter is *active*. The engine's
/// `LoraManager` (P4) hands these to the session, which feeds them by name every
/// run so the delta applies; when the adapter is deactivated the feeds are
/// dropped and the zero-rank defaults collapse the branch to base-only.
///
/// The names are authored here (next to where the override inputs are created)
/// so the feed side never has to reconstruct the injection's naming convention.
#[derive(Clone, Debug)]
pub struct OverrideFeed {
    /// The overridable optional graph input's name, e.g.
    /// `lora.<adapter>.layers.0.q_proj.A_t`.
    pub input_name: String,
    /// The factor bytes to bind: `A_t = [K, rank]` or `B_t = [rank, N]`.
    pub data: TensorData,
}

/// The result of a successful injection: the set of value ids that became
/// overridable optional inputs (feed into [`Executor::build_with_overrides`]),
/// the concrete per-module `A_t`/`B_t` feeds (design §D activation), plus the
/// manifest that drove the rewrite.
///
/// [`Executor::build_with_overrides`]: crate::executor
#[derive(Debug)]
pub struct LoraInjection {
    pub override_ids: HashSet<ValueId>,
    /// The named `A_t`/`B_t` feeds for every *targeted* module, in the same
    /// order as `manifest.entries`. Untargeted fused Q/K/V slices are never
    /// fed (their permanently zero-rank branches stay base-only) and so are
    /// absent here.
    pub feeds: Vec<OverrideFeed>,
    pub manifest: LoraManifest,
}

// ===========================================================================
// Errors — typed, fail-loud (design §C "Fail loud").
// ===========================================================================

/// Errors raised while resolving the target manifest or rewriting the graph.
/// Every failure names the offending module so an operator can act on it; the
/// pass never silently skips a projection or produces a partial mapping.
#[derive(Debug, thiserror::Error)]
pub enum LoraInjectError {
    #[error(
        "LoRA target module {module:?} (layer {layer}) resolves to no {BASE_OP} node in the \
         graph; the adapter targets a projection this export does not contain (unrecognized \
         layout, e.g. a linear-attention or renamed projection)"
    )]
    UnresolvedModule { module: String, layer: usize },

    #[error(
        "LoRA target module {module:?} (layer {layer}) resolves to {count} candidate {BASE_OP} \
         nodes ({nodes}); the mapping is ambiguous"
    )]
    AmbiguousModule {
        module: String,
        layer: usize,
        count: usize,
        nodes: String,
    },

    #[error("{BASE_OP} node {node:?} is missing the required integer attribute {attr:?}")]
    MissingAttribute { node: String, attr: String },

    #[error("{BASE_OP} node {node:?} has no activation input in slot 0")]
    MissingActivation { node: String },

    #[error(
        "fused projection node {node:?} (N={fused_n}) has no {ATTENTION_OP} in layer {layer} to \
         derive Q/K/V slice widths from; cannot resolve a fused q/k/v target"
    )]
    MissingAttention { node: String, layer: usize, fused_n: usize },

    #[error(
        "fused projection node {node:?} (N={fused_n}) is inconsistent with {ATTENTION_OP} \
         num_heads={num_heads} kv_num_heads={kv_num_heads}: {reason}"
    )]
    FusedGeometry {
        node: String,
        fused_n: usize,
        num_heads: usize,
        kv_num_heads: usize,
        reason: String,
    },

    #[error(
        "LoRA module {module:?}: factor rank disagreement — A_t is {a_rank} wide but B_t is \
         {b_rank} tall (both must equal the rank)"
    )]
    RankMismatch {
        module: String,
        a_rank: usize,
        b_rank: usize,
    },

    #[error(
        "LoRA module {module:?}: {factor} has shape {actual:?}, but the resolved target requires \
         {expected:?}"
    )]
    ShapeMismatch {
        module: String,
        factor: &'static str,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error(
        "LoRA module {module:?}: adapter factor dtype {adapter:?} does not match the base \
         activation dtype {activation:?}; the delta branch runs in the activation dtype"
    )]
    DtypeMismatch {
        module: String,
        adapter: DataType,
        activation: DataType,
    },

    #[error("internal: manifest has {entries} entries but the adapter spec has {modules} modules")]
    SpecCountMismatch { entries: usize, modules: usize },

    #[error(
        "LoRA module {module:?}: could not admit factor pages into the adapter weight pool: \
         {source}"
    )]
    PoolAdmission {
        module: String,
        #[source]
        source: onnx_runtime_ep_api::LoraPoolError,
    },
}

// ===========================================================================
// Discovery — parsing the graph into a projection index (design §C, §H).
// ===========================================================================

/// One discovered base projection node, keyed by `(layer_index, proj_token)`.
struct ProjectionNode {
    node_id: NodeId,
    node_name: String,
    proj_token: String,
    layer: usize,
    k: usize,
    n: usize,
    activation: ValueId,
    base_output: ValueId,
}

/// Extract the layer index from a node/weight name of either the slash form
/// (`/model/layers.0/attn/...`) or the dot form (`model.layers.0.attn....`) by
/// scanning for the `layers` token followed by an integer.
fn extract_layer(name: &str) -> Option<usize> {
    let idx = name.find("layers")?;
    let rest = &name[idx + "layers".len()..];
    let rest = rest.strip_prefix('.').or_else(|| rest.strip_prefix('/'))?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Extract the projection token (`qkv_proj`, `q_proj`, `gate_proj`, …) from a
/// weight tensor name: the `.`-separated component immediately before the
/// `MatMul` / `MatMulNBits` op component (`model.layers.0.attn.qkv_proj.MatMul.
/// weight_Q4` → `qkv_proj`; `model.layers.0.attn.q_proj.MatMulNBits.qweight` →
/// `q_proj`).
fn proj_token_from_weight(weight_name: &str) -> Option<String> {
    let parts: Vec<&str> = weight_name.split('.').collect();
    let op_pos = parts
        .iter()
        .position(|p| *p == "MatMul" || *p == BASE_OP)?;
    if op_pos == 0 {
        return None;
    }
    Some(parts[op_pos - 1].to_string())
}

/// Extract the projection token from a slash-form node name: the second-to-last
/// path segment (`/model/layers.0/attn/qkv_proj/MatMul_Q4` → `qkv_proj`).
fn proj_token_from_node_name(node_name: &str) -> Option<String> {
    let parts: Vec<&str> = node_name.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() >= 2 {
        Some(parts[parts.len() - 2].to_string())
    } else {
        None
    }
}

/// The trailing `*_proj`-style leaf token of a semantic PEFT module name
/// (`self_attn.q_proj` → `q_proj`).
fn leaf_token(module_name: &str) -> &str {
    module_name.rsplit('.').next().unwrap_or(module_name)
}

/// True when a projection token names a fused Q/K/V projection.
fn is_fused_qkv_token(token: &str) -> bool {
    token == "qkv_proj" || token == "qkv"
}

/// Read a required positive integer attribute from a base node.
fn required_int_attr(node: &Node, attr: &str) -> Result<usize, LoraInjectError> {
    node.attr(attr)
        .and_then(|a| a.as_int())
        .filter(|v| *v > 0)
        .map(|v| v as usize)
        .ok_or_else(|| LoraInjectError::MissingAttribute {
            node: node_label(node),
            attr: attr.to_string(),
        })
}

/// A stable label for a node in diagnostics: its ONNX name, or `#id` if unnamed.
fn node_label(node: &Node) -> String {
    if node.name.is_empty() {
        format!("#{}", node.id.0)
    } else {
        node.name.clone()
    }
}

/// Index every `MatMulNBits` base projection node by `(layer, proj_token)`.
/// Nodes whose layer or projection token cannot be parsed are skipped (they are
/// not layer projections, e.g. `lm_head`); an unresolved *target* later fails
/// loud rather than silently mapping to nothing.
fn index_projections(graph: &Graph) -> Result<Vec<ProjectionNode>, LoraInjectError> {
    let mut out = Vec::new();
    for (node_id, node) in graph.nodes.iter() {
        if node.op_type != BASE_OP {
            continue;
        }
        // Parse layer + projection token from the weight name (input slot 1)
        // first — it is present on every quantized projection and is stable
        // across the slash/dot naming variants — then fall back to the node
        // name.
        let weight_name = node
            .inputs
            .get(1)
            .and_then(|slot| *slot)
            .and_then(|vid| graph.value(vid).name.as_deref());
        let layer = weight_name
            .and_then(extract_layer)
            .or_else(|| extract_layer(&node.name));
        let proj_token = weight_name
            .and_then(proj_token_from_weight)
            .or_else(|| proj_token_from_node_name(&node.name));
        let (Some(layer), Some(proj_token)) = (layer, proj_token) else {
            continue;
        };
        let k = required_int_attr(node, "K")?;
        let n = required_int_attr(node, "N")?;
        let activation =
            node.inputs
                .first()
                .and_then(|slot| *slot)
                .ok_or_else(|| LoraInjectError::MissingActivation {
                    node: node_label(node),
                })?;
        let base_output = *node.outputs.first().ok_or_else(|| {
            LoraInjectError::MissingActivation {
                node: node_label(node),
            }
        })?;
        out.push(ProjectionNode {
            node_id,
            node_name: node_label(node),
            proj_token,
            layer,
            k,
            n,
            activation,
            base_output,
        });
    }
    Ok(out)
}

/// Find the `GroupQueryAttention` node's `(num_heads, kv_num_heads)` for a
/// layer, so fused Q/K/V slice widths can be derived structurally.
fn attention_heads(graph: &Graph, layer: usize) -> Option<(usize, usize)> {
    for (_, node) in graph.nodes.iter() {
        if node.op_type != ATTENTION_OP {
            continue;
        }
        if extract_layer(&node.name) != Some(layer) {
            continue;
        }
        let num_heads = node.attr("num_heads").and_then(|a| a.as_int());
        let kv_num_heads = node.attr("kv_num_heads").and_then(|a| a.as_int());
        if let (Some(h), Some(kv)) = (num_heads, kv_num_heads) {
            if h > 0 && kv > 0 {
                return Some((h as usize, kv as usize));
            }
        }
    }
    None
}

/// Compute the fused Q/K/V geometry of a fused projection from the layer's GQA
/// head configuration (design §H), validating that the slices exactly tile the
/// fused width.
fn fused_group(
    graph: &Graph,
    proj: &ProjectionNode,
) -> Result<FusedGroup, LoraInjectError> {
    let fused_n = proj.n;
    let (num_heads, kv_num_heads) = attention_heads(graph, proj.layer).ok_or_else(|| {
        LoraInjectError::MissingAttention {
            node: proj.node_name.clone(),
            layer: proj.layer,
            fused_n,
        }
    })?;
    let group_units = num_heads + 2 * kv_num_heads;
    if group_units == 0 || fused_n % group_units != 0 {
        return Err(LoraInjectError::FusedGeometry {
            node: proj.node_name.clone(),
            fused_n,
            num_heads,
            kv_num_heads,
            reason: format!(
                "N ({fused_n}) is not divisible by num_heads + 2*kv_num_heads ({group_units}), \
                 so a per-head slice width cannot be derived"
            ),
        });
    }
    let head_dim = fused_n / group_units;
    let n_q = num_heads * head_dim;
    let n_kv = kv_num_heads * head_dim;
    if n_q + 2 * n_kv != fused_n {
        return Err(LoraInjectError::FusedGeometry {
            node: proj.node_name.clone(),
            fused_n,
            num_heads,
            kv_num_heads,
            reason: format!("Q({n_q}) + K({n_kv}) + V({n_kv}) != N({fused_n})"),
        });
    }
    Ok(FusedGroup {
        node_id: proj.node_id,
        fused_n,
        slices: [
            (QkvRole::Q, 0, n_q),
            (QkvRole::K, n_q, n_kv),
            (QkvRole::V, n_q + n_kv, n_kv),
        ],
    })
}

/// Resolve a single requested target to a validated [`TargetEntry`], failing
/// loud when it cannot be mapped to exactly one node + slice (design §C).
fn resolve_target(
    graph: &Graph,
    projections: &[ProjectionNode],
    target: &LoraTarget,
) -> Result<TargetEntry, LoraInjectError> {
    let leaf = leaf_token(&target.module_name);
    let semantic = format!("layers.{}.{}", target.layer_index, leaf);

    // 1) Direct match: a node in this layer whose projection token equals the
    //    requested leaf token (the split layout, e.g. Qwen3 q/k/v, or any
    //    standalone o_proj / gate_proj / up_proj / down_proj, or a directly
    //    targeted fused/linear token like in_proj_qkv).
    let direct: Vec<&ProjectionNode> = projections
        .iter()
        .filter(|p| p.layer == target.layer_index && p.proj_token == leaf)
        .collect();
    if direct.len() > 1 {
        return Err(LoraInjectError::AmbiguousModule {
            module: target.module_name.clone(),
            layer: target.layer_index,
            count: direct.len(),
            nodes: direct
                .iter()
                .map(|p| p.node_name.clone())
                .collect::<Vec<_>>()
                .join(", "),
        });
    }
    if let Some(p) = direct.first() {
        return Ok(TargetEntry {
            semantic,
            node_id: p.node_id,
            base_output: p.base_output,
            activation: p.activation,
            k: p.k,
            n: p.n,
            dtype: graph.value(p.activation).dtype,
            placement: Placement::Direct,
        });
    }

    // 2) Fused Q/K/V: the requested module is q/k/v and the layer has a single
    //    fused qkv projection. Slice offsets come from GQA head dims.
    let role = match leaf {
        "q_proj" => Some(QkvRole::Q),
        "k_proj" => Some(QkvRole::K),
        "v_proj" => Some(QkvRole::V),
        _ => None,
    };
    if let Some(role) = role {
        let fused: Vec<&ProjectionNode> = projections
            .iter()
            .filter(|p| p.layer == target.layer_index && is_fused_qkv_token(&p.proj_token))
            .collect();
        if fused.len() > 1 {
            return Err(LoraInjectError::AmbiguousModule {
                module: target.module_name.clone(),
                layer: target.layer_index,
                count: fused.len(),
                nodes: fused
                    .iter()
                    .map(|p| p.node_name.clone())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        if let Some(p) = fused.first() {
            let group = fused_group(graph, p)?;
            let (_, _, width) = group
                .slices
                .iter()
                .copied()
                .find(|(r, _, _)| *r == role)
                .expect("Q/K/V slice present in a fused group");
            return Ok(TargetEntry {
                semantic,
                node_id: p.node_id,
                base_output: p.base_output,
                activation: p.activation,
                k: p.k,
                n: width,
                dtype: graph.value(p.activation).dtype,
                placement: Placement::FusedSlice { group, role },
            });
        }
    }

    Err(LoraInjectError::UnresolvedModule {
        module: target.module_name.clone(),
        layer: target.layer_index,
    })
}

/// Build the per-model target manifest (design §C, **P3**). Fails loud if any
/// target cannot be resolved to exactly one node + slice.
pub fn build_manifest(
    graph: &Graph,
    targets: &[LoraTarget],
) -> Result<LoraManifest, LoraInjectError> {
    let projections = index_projections(graph)?;
    let mut entries = Vec::with_capacity(targets.len());
    for target in targets {
        entries.push(resolve_target(graph, &projections, target)?);
    }
    Ok(LoraManifest { entries })
}

// ===========================================================================
// Injection — the graph rewrite (design §B, §B.3, **P2b**).
// ===========================================================================

/// Encode a scalar `scale` as little-endian bytes for a scale constant of the
/// branch dtype.
fn scale_bytes(scale: f32, dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float16 => half::f16::from_f32(scale).to_le_bytes().to_vec(),
        DataType::BFloat16 => half::bf16::from_f32(scale).to_le_bytes().to_vec(),
        _ => scale.to_le_bytes().to_vec(),
    }
}

/// Clone a shape, replacing (or appending) its last dimension with `last`.
fn shape_with_last(base: &Shape, last: Dim) -> Shape {
    let mut shape = base.clone();
    if shape.is_empty() {
        shape.push(last);
    } else {
        let idx = shape.len() - 1;
        shape[idx] = last;
    }
    shape
}

/// Validate a spec module's factor dims/dtype against the resolved target's
/// geometry, failing loud on any mismatch (a wrong shape would silently
/// mis-map).
fn validate_module(
    spec: &LoraModuleSpec,
    k: usize,
    n: usize,
    dtype: DataType,
) -> Result<(), LoraInjectError> {
    if spec.a_t.dtype != spec.b_t.dtype {
        return Err(LoraInjectError::DtypeMismatch {
            module: spec.module_name.clone(),
            adapter: spec.a_t.dtype,
            activation: spec.b_t.dtype,
        });
    }
    if spec.a_t.dtype != dtype {
        return Err(LoraInjectError::DtypeMismatch {
            module: spec.module_name.clone(),
            adapter: spec.a_t.dtype,
            activation: dtype,
        });
    }
    let a_rank = *spec.a_t.dims.get(1).unwrap_or(&usize::MAX);
    let b_rank = *spec.b_t.dims.first().unwrap_or(&usize::MAX);
    if a_rank != spec.rank || b_rank != spec.rank {
        return Err(LoraInjectError::RankMismatch {
            module: spec.module_name.clone(),
            a_rank,
            b_rank,
        });
    }
    if spec.a_t.dims != vec![k, spec.rank] {
        return Err(LoraInjectError::ShapeMismatch {
            module: spec.module_name.clone(),
            factor: "A_t",
            actual: spec.a_t.dims.clone(),
            expected: vec![k, spec.rank],
        });
    }
    if spec.b_t.dims != vec![spec.rank, n] {
        return Err(LoraInjectError::ShapeMismatch {
            module: spec.module_name.clone(),
            factor: "B_t",
            actual: spec.b_t.dims.clone(),
            expected: vec![spec.rank, n],
        });
    }
    Ok(())
}

/// Inputs to build one delta sub-branch (`scaled = scale * (x @ A_t) @ B_t`).
struct BranchPlan {
    /// Unique value-name prefix, e.g. `lora.adapter.layers.0.q_proj`.
    prefix: String,
    /// Inner dim `K`.
    k: usize,
    /// Output width `N` (the slice width when fused).
    width: usize,
    /// Branch dtype (the activation dtype).
    dtype: DataType,
    /// Per-module scale, baked as a constant.
    scale: f32,
    /// The activation value `x`.
    activation: ValueId,
    /// The base output value, only used to shape the delta.
    base_output: ValueId,
}

/// Insert one delta sub-branch and return its `scaled` output value, appending
/// the `A_t`/`B_t` value ids to `override_ids`. The branch is a provable no-op
/// when `A_t`/`B_t` are unfed (their zero-rank defaults collapse the delta to a
/// zero-filled `[.., width]`).
fn insert_branch(
    graph: &mut Graph,
    plan: &BranchPlan,
    override_ids: &mut HashSet<ValueId>,
) -> ValueId {
    let rank_symbol = graph.create_symbol(Some(format!("{}.lora_r", plan.prefix)));

    // A_t: overridable optional input, declared [K, r], default [K, 0].
    let a_t = graph.create_named_value(
        format!("{}.A_t", plan.prefix),
        plan.dtype,
        vec![Dim::Static(plan.k), Dim::Symbolic(rank_symbol)],
    );
    graph.add_input(a_t);
    graph.set_initializer(
        a_t,
        WeightRef::Inline(TensorData::from_raw(plan.dtype, vec![plan.k, 0], Vec::new())),
    );
    override_ids.insert(a_t);

    // B_t: overridable optional input, declared [r, N], default [0, N].
    let b_t = graph.create_named_value(
        format!("{}.B_t", plan.prefix),
        plan.dtype,
        vec![Dim::Symbolic(rank_symbol), Dim::Static(plan.width)],
    );
    graph.add_input(b_t);
    graph.set_initializer(
        b_t,
        WeightRef::Inline(TensorData::from_raw(
            plan.dtype,
            vec![0, plan.width],
            Vec::new(),
        )),
    );
    override_ids.insert(b_t);

    let activation_shape = graph.value(plan.activation).shape.clone();
    let base_shape = graph.value(plan.base_output).shape.clone();

    // r_int = MatMul(x, A_t) -> [.., r]
    let rmid = graph.create_named_value(
        format!("{}.r_int", plan.prefix),
        plan.dtype,
        shape_with_last(&activation_shape, Dim::Symbolic(rank_symbol)),
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(plan.activation), Some(a_t)],
        vec![rmid],
    ));

    // delta = MatMul(r_int, B_t) -> [.., width]
    let delta = graph.create_named_value(
        format!("{}.delta", plan.prefix),
        plan.dtype,
        shape_with_last(&base_shape, Dim::Static(plan.width)),
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(rmid), Some(b_t)],
        vec![delta],
    ));

    // scaled = Mul(delta, scale)
    let scale_value = graph.create_named_value(format!("{}.scale", plan.prefix), plan.dtype, vec![]);
    graph.set_initializer(
        scale_value,
        WeightRef::Inline(TensorData::from_raw(
            plan.dtype,
            vec![],
            scale_bytes(plan.scale, plan.dtype),
        )),
    );
    let scaled = graph.create_named_value(
        format!("{}.scaled", plan.prefix),
        plan.dtype,
        shape_with_last(&base_shape, Dim::Static(plan.width)),
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(delta), Some(scale_value)],
        vec![scaled],
    ));

    scaled
}

/// Rewire consumers of `base_output` onto a fresh value produced by
/// `Add(base_output, addend)`, returning the new value. The base node is left
/// untouched; only its downstream edges move.
fn add_delta_onto_base(
    graph: &mut Graph,
    prefix: &str,
    base_output: ValueId,
    addend: ValueId,
) -> ValueId {
    let base_shape = graph.value(base_output).shape.clone();
    let dtype = graph.value(base_output).dtype;
    let y_new = graph.create_named_value(format!("{prefix}.Y"), dtype, base_shape);
    // Move existing consumers (and any graph-output slot) to Y before the Add
    // itself becomes a consumer of base_output (mirrors Graph::insert_on_edge).
    graph.replace_all_uses(base_output, y_new);
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(base_output), Some(addend)],
        vec![y_new],
    ));
    y_new
}

/// Inject a standalone (split) projection's delta branch.
fn inject_direct(
    graph: &mut Graph,
    entry: &TargetEntry,
    spec: &LoraModuleSpec,
    adapter_name: &str,
    override_ids: &mut HashSet<ValueId>,
    feeds: &mut Vec<OverrideFeed>,
) -> Result<(), LoraInjectError> {
    validate_module(spec, entry.k, entry.n, entry.dtype)?;
    let prefix = format!("lora.{adapter_name}.{}", entry.semantic);
    let scaled = insert_branch(
        graph,
        &BranchPlan {
            prefix: prefix.clone(),
            k: entry.k,
            width: entry.n,
            dtype: entry.dtype,
            scale: spec.scale,
            activation: entry.activation,
            base_output: entry.base_output,
        },
        override_ids,
    );
    push_module_feeds(&prefix, spec, feeds);
    add_delta_onto_base(graph, &prefix, entry.base_output, scaled);
    Ok(())
}

/// Record the `A_t`/`B_t` feeds for one targeted module under `prefix`, matching
/// the input names [`insert_branch`] created (`{prefix}.A_t` / `{prefix}.B_t`).
fn push_module_feeds(prefix: &str, spec: &LoraModuleSpec, feeds: &mut Vec<OverrideFeed>) {
    feeds.push(OverrideFeed {
        input_name: format!("{prefix}.A_t"),
        data: spec.a_t.clone(),
    });
    feeds.push(OverrideFeed {
        input_name: format!("{prefix}.B_t"),
        data: spec.b_t.clone(),
    });
}

/// Inject a fused `qkv_proj` projection: one delta sub-branch per Q/K/V slice,
/// concatenated in `[Q, K, V]` order and added onto the fused base output. Every
/// slice gets an overridable branch — targeted roles carry the adapter's factor
/// geometry, and roles absent from this adapter default to a zero-rank no-op
/// (leaving a hook a later adapter can feed). This keeps each factor's natural
/// `[r, N_slice]` `B_t` (no feed-side padding) and bakes the slice offsets into
/// the concat order, so a wrong slice can never silently corrupt attention.
fn inject_fused(
    graph: &mut Graph,
    group: &FusedGroup,
    base_output: ValueId,
    activation: ValueId,
    k: usize,
    dtype: DataType,
    role_specs: &HashMap<QkvRole, &LoraModuleSpec>,
    adapter_name: &str,
    layer_semantic: &str,
    override_ids: &mut HashSet<ValueId>,
    feeds: &mut Vec<OverrideFeed>,
) -> Result<(), LoraInjectError> {
    let mut slice_values = Vec::with_capacity(group.slices.len());
    for (role, _offset, width) in group.slices {
        let prefix = format!("lora.{adapter_name}.{layer_semantic}.{}", role.as_str());
        let (scale, targeted_spec) = match role_specs.get(&role) {
            Some(spec) => {
                validate_module(spec, k, width, dtype)?;
                (spec.scale, Some(*spec))
            }
            // Untargeted slice: a permanently zero-rank branch (scale is
            // irrelevant against the empty delta).
            None => (1.0, None),
        };
        let scaled = insert_branch(
            graph,
            &BranchPlan {
                prefix: prefix.clone(),
                k,
                width,
                dtype,
                scale,
                activation,
                base_output,
            },
            override_ids,
        );
        // Only a targeted slice is ever fed; an untargeted slice keeps its
        // zero-rank default so the fused output stays base-only for that role.
        if let Some(spec) = targeted_spec {
            push_module_feeds(&prefix, spec, feeds);
        }
        slice_values.push(scaled);
    }

    // delta_fused = Concat([q, k, v], axis=-1) -> [.., fused_n]
    let base_shape = graph.value(base_output).shape.clone();
    let fused_prefix = format!("lora.{adapter_name}.{layer_semantic}.qkv");
    let concat = graph.create_named_value(
        format!("{fused_prefix}.delta_fused"),
        dtype,
        shape_with_last(&base_shape, Dim::Static(group.fused_n)),
    );
    let mut concat_node = Node::new(
        NodeId(0),
        "Concat",
        slice_values.iter().map(|v| Some(*v)).collect(),
        vec![concat],
    );
    concat_node
        .attributes
        .insert("axis".to_string(), onnx_runtime_ir::Attribute::Int(-1));
    graph.insert_node(concat_node);

    add_delta_onto_base(graph, &fused_prefix, base_output, concat);
    Ok(())
}

/// Inject an adapter into the graph given a pre-built manifest (design §B,
/// **P2b**). `manifest.entries[i]` must correspond to `adapter.modules[i]`.
/// Returns the set of overridable optional input value ids for
/// [`Executor::build_with_overrides`].
///
/// [`Executor::build_with_overrides`]: crate::executor
pub fn inject(
    graph: &mut Graph,
    manifest: &LoraManifest,
    adapter: &LoraAdapterSpec,
) -> Result<HashSet<ValueId>, LoraInjectError> {
    let mut feeds = Vec::new();
    inject_collecting_feeds(graph, manifest, adapter, &mut feeds)
}

/// Inner injection that also records the concrete `A_t`/`B_t` feeds for each
/// targeted module (design §D activation). [`inject`] discards the feeds (its
/// callers only need the override ids); [`inject_lora_adapter`] keeps them.
fn inject_collecting_feeds(
    graph: &mut Graph,
    manifest: &LoraManifest,
    adapter: &LoraAdapterSpec,
    feeds: &mut Vec<OverrideFeed>,
) -> Result<HashSet<ValueId>, LoraInjectError> {
    if manifest.entries.len() != adapter.modules.len() {
        return Err(LoraInjectError::SpecCountMismatch {
            entries: manifest.entries.len(),
            modules: adapter.modules.len(),
        });
    }
    let mut override_ids = HashSet::new();

    // Group fused-target modules by their base node so each fused projection is
    // rewritten exactly once, with all three Q/K/V slices.
    struct FusedPlan<'a> {
        group: FusedGroup,
        base_output: ValueId,
        activation: ValueId,
        k: usize,
        dtype: DataType,
        layer_semantic: String,
        role_specs: HashMap<QkvRole, &'a LoraModuleSpec>,
    }
    let mut fused_plans: BTreeMap<usize, FusedPlan> = BTreeMap::new();

    for (entry, spec) in manifest.entries.iter().zip(&adapter.modules) {
        match &entry.placement {
            Placement::Direct => {
                inject_direct(graph, entry, spec, &adapter.name, &mut override_ids, feeds)?;
            }
            Placement::FusedSlice { group, role } => {
                let layer_semantic = entry
                    .semantic
                    .rsplit_once('.')
                    .map(|(prefix, _)| prefix.to_string())
                    .unwrap_or_else(|| entry.semantic.clone());
                let plan = fused_plans.entry(group.node_id.0 as usize).or_insert_with(|| {
                    FusedPlan {
                        group: group.clone(),
                        base_output: entry.base_output,
                        activation: entry.activation,
                        k: entry.k,
                        dtype: entry.dtype,
                        layer_semantic,
                        role_specs: HashMap::new(),
                    }
                });
                plan.role_specs.insert(*role, spec);
            }
        }
    }

    for (_, plan) in fused_plans {
        inject_fused(
            graph,
            &plan.group,
            plan.base_output,
            plan.activation,
            plan.k,
            plan.dtype,
            &plan.role_specs,
            &adapter.name,
            &plan.layer_semantic,
            &mut override_ids,
            feeds,
        )?;
    }

    Ok(override_ids)
}

/// Resolve the target manifest and inject an adapter in one step (the entry
/// point a `LoraManager` (P4) calls). Fails loud if any targeted module cannot
/// be resolved.
pub fn inject_lora_adapter(
    graph: &mut Graph,
    adapter: &LoraAdapterSpec,
) -> Result<LoraInjection, LoraInjectError> {
    let targets: Vec<LoraTarget> = adapter
        .modules
        .iter()
        .map(|m| LoraTarget {
            module_name: m.module_name.clone(),
            layer_index: m.layer_index,
        })
        .collect();
    let manifest = build_manifest(graph, &targets)?;
    let mut feeds = Vec::new();
    let override_ids = inject_collecting_feeds(graph, &manifest, adapter, &mut feeds)?;
    Ok(LoraInjection {
        override_ids,
        feeds,
        manifest,
    })
}

// ===========================================================================
// Phase-2 grouped injection (design §J.1, §J.5) — the `GroupedLoraDelta` op.
// ===========================================================================

/// The `pkg.nxrt` custom op emitted by the grouped injection path.
const GROUPED_OP: &str = "GroupedLoraDelta";
/// The private operator domain the grouped op is registered under (mirrors the
/// other `pkg.nxrt` custom ops in `build_cpu_registry`).
const NXRT_DOMAIN: &str = "pkg.nxrt";
/// The single reserved adapter id a pool-of-one (single-adapter unify, §J.5)
/// routes every row to. Multi-tenant batching (P2 follow-up) assigns distinct
/// ids per active adapter.
const SINGLE_ADAPTER_ID: AdapterId = AdapterId(0);

/// The result of a successful grouped injection (design §J.1). The graph now
/// carries one [`GroupedLoraDelta`](GROUPED_OP) op per targeted projection,
/// each reading the shared per-run `segments` routing input and reading its
/// factor pages from the registered [`LoraWeightPool`]; the base int4 weights
/// are never touched.
///
/// Unlike the Phase-1 [`inject`] path, the factors are *not* graph inputs — they
/// live in the pool, admitted once at load time — so the only per-run input the
/// caller must feed (beyond the model's activations) is `segments`.
pub struct GroupedLoraInjection {
    /// The name of the shared non-constant per-run routing input (`Int32`,
    /// shape `[tokens]`): `segments[row]` is the adapter id that row routes to,
    /// or a negative value / the reserved null id for a base-only row. It has no
    /// initializer, so the executor treats it as a required runtime input that
    /// is never a graph constant (never prepacked).
    pub segments_input: String,
    /// The copyable identifier baked into every emitted op's `pool_id` attribute.
    /// It does not own the registry entry; dropping this injection releases it.
    pub pool_id: LoraPoolId,
    /// The populated pool. Its registry entry is owned by this injection's
    /// private registration and is removed when the injection is dropped.
    pub pool: Arc<LoraWeightPool>,
    /// The adapter id every row routes to for the single-adapter unify case.
    pub adapter_id: AdapterId,
    pub manifest: LoraManifest,
    _registration: LoraPoolRegistration,
}

/// Round a byte count up to the pool's page alignment so a computed capacity is
/// never short of what [`LoraWeightPool::admit`] will actually reserve.
fn aligned_page_bytes(bytes: usize) -> u64 {
    const ALIGN: usize = 64;
    (bytes.div_ceil(ALIGN) * ALIGN) as u64
}

/// Borrow one already-transposed factor of a spec as a pool admission input.
fn factor_input(data: &TensorData) -> LoraFactorInput<'_> {
    LoraFactorInput {
        dtype: data.dtype,
        rows: data.dims.first().copied().unwrap_or(0),
        cols: data.dims.get(1).copied().unwrap_or(0),
        bytes: &data.data,
    }
}

/// Emit one `GroupedLoraDelta` op reading `(activation, segments)` and producing
/// a `[.., width]` delta shaped from `base_output`, returning the delta value.
/// The op carries the `pkg.nxrt` domain and the five integer attributes the CPU
/// kernel factory requires (`K`, `N`, `module_id`, `max_rank`, `pool_id`).
#[allow(clippy::too_many_arguments)]
fn emit_grouped_op(
    graph: &mut Graph,
    prefix: &str,
    activation: ValueId,
    segments: ValueId,
    base_output: ValueId,
    k: usize,
    width: usize,
    module_id: u32,
    max_rank: usize,
    pool_id: u64,
    dtype: DataType,
) -> ValueId {
    let base_shape = graph.value(base_output).shape.clone();
    let delta = graph.create_named_value(
        format!("{prefix}.delta"),
        dtype,
        shape_with_last(&base_shape, Dim::Static(width)),
    );
    let mut node = Node::new(
        NodeId(0),
        GROUPED_OP,
        vec![Some(activation), Some(segments)],
        vec![delta],
    );
    node.domain = NXRT_DOMAIN.to_string();
    node.version = Some(1);
    node.attributes
        .insert("K".to_string(), Attribute::Int(k as i64));
    node.attributes
        .insert("N".to_string(), Attribute::Int(width as i64));
    node.attributes
        .insert("module_id".to_string(), Attribute::Int(module_id as i64));
    // `max_rank` is a per-op upper bound the kernel validates against; a
    // permanently zero-rank (untargeted fused slice) still needs a positive
    // bound, so never emit zero.
    node.attributes.insert(
        "max_rank".to_string(),
        Attribute::Int(max_rank.max(1) as i64),
    );
    node.attributes
        .insert("pool_id".to_string(), Attribute::Int(pool_id as i64));
    graph.insert_node(node);
    delta
}

/// One planned op emission, resolved after the pool has been built and
/// registered (so `pool_id` is known).
struct GroupedOpPlan {
    prefix: String,
    activation: ValueId,
    base_output: ValueId,
    k: usize,
    width: usize,
    module_id: u32,
    max_rank: usize,
    dtype: DataType,
    /// `None` for a standalone projection (delta added directly); `Some(role)`
    /// for one slice of a fused `qkv_proj` (deltas concatenated then added).
    fused: Option<(NodeId, QkvRole)>,
}

/// Inject an adapter as Phase-2 [`GroupedLoraDelta`](GROUPED_OP) ops (design
/// §J.1, §J.5), the batched-LoRA unify of the Phase-1 4-node subgraph.
///
/// This is the second selector of the injection pass (the first is [`inject`],
/// the Phase-1 4-node path, which is kept in place — §J.5 retirement is a later
/// phase once parity and performance are proven). It:
///
/// 1. builds a fresh [`LoraWeightPool`] sized to hold exactly this adapter,
///    admits every targeted module's `A_t`/`B_t` factor pair (adapter-major,
///    64-byte aligned pages), and registers it in the process
///    [`LoraPoolRegistry`];
/// 2. creates one shared non-constant `segments` routing input;
/// 3. emits one `GroupedLoraDelta` op per standalone projection, folded onto the
///    base output by the reused Phase-1 `Add`; and one op per Q/K/V slice of a
///    fused `qkv_proj`, concatenated in `[Q, K, V]` order and added once (§J's
///    simpler "three op instances sharing one Add" option — no sub-descriptors,
///    so each factor keeps its natural `[r, N_slice]` `B_t` and the slice
///    offsets are baked into the concat order exactly as the Phase-1 fused path
///    does).
///
/// For the single-adapter case every row routes to [`SINGLE_ADAPTER_ID`] (feed
/// `segments = [0, 0, …]`), and the op's group-by-adapter dense path runs the
/// same two fp32-accumulator matmuls as the Phase-1 branch, in the same order —
/// so it is bit-parity with [`inject`] on the §E golden test.
pub fn inject_grouped(
    graph: &mut Graph,
    manifest: &LoraManifest,
    adapter: &LoraAdapterSpec,
) -> Result<GroupedLoraInjection, LoraInjectError> {
    if manifest.entries.len() != adapter.modules.len() {
        return Err(LoraInjectError::SpecCountMismatch {
            entries: manifest.entries.len(),
            modules: adapter.modules.len(),
        });
    }

    // --- Grouping pass: separate standalone projections from fused Q/K/V. ---
    struct DirectAcc<'a> {
        entry: &'a TargetEntry,
        spec: &'a LoraModuleSpec,
    }
    struct FusedAcc<'a> {
        group: FusedGroup,
        base_output: ValueId,
        activation: ValueId,
        k: usize,
        dtype: DataType,
        layer_semantic: String,
        role_specs: HashMap<QkvRole, &'a LoraModuleSpec>,
    }
    let mut directs: Vec<DirectAcc> = Vec::new();
    let mut fused_plans: BTreeMap<usize, FusedAcc> = BTreeMap::new();

    for (entry, spec) in manifest.entries.iter().zip(&adapter.modules) {
        match &entry.placement {
            Placement::Direct => {
                validate_module(spec, entry.k, entry.n, entry.dtype)?;
                directs.push(DirectAcc { entry, spec });
            }
            Placement::FusedSlice { group, role } => {
                let layer_semantic = entry
                    .semantic
                    .rsplit_once('.')
                    .map(|(prefix, _)| prefix.to_string())
                    .unwrap_or_else(|| entry.semantic.clone());
                let acc = fused_plans
                    .entry(group.node_id.0 as usize)
                    .or_insert_with(|| FusedAcc {
                        group: group.clone(),
                        base_output: entry.base_output,
                        activation: entry.activation,
                        k: entry.k,
                        dtype: entry.dtype,
                        layer_semantic,
                        role_specs: HashMap::new(),
                    });
                acc.role_specs.insert(*role, spec);
            }
        }
    }

    // --- Build the pool: assign module ids and admit every page. ---
    // Capacity is sized to hold exactly this adapter (aligned page bytes plus a
    // small slack), so nothing is evicted at load time; the real byte budget
    // governs admission one layer up in the engine (design §J.2 control plane).
    let mut capacity: u64 = 0;
    let account = |cap: &mut u64, data: &TensorData| {
        *cap = cap.saturating_add(aligned_page_bytes(data.data.len()));
    };
    for d in &directs {
        account(&mut capacity, &d.spec.a_t);
        account(&mut capacity, &d.spec.b_t);
    }
    for f in fused_plans.values() {
        for (_role, _offset, _width) in f.group.slices {
            // Untargeted slices admit an empty zero-rank page (0 bytes → one
            // aligned slot); targeted slices admit their real factors.
            capacity = capacity.saturating_add(2 * aligned_page_bytes(0));
        }
        for spec in f.role_specs.values() {
            account(&mut capacity, &spec.a_t);
            account(&mut capacity, &spec.b_t);
        }
    }
    // Slack so page-alignment rounding can never overrun the ceiling.
    capacity = capacity.saturating_add(64 * (2 * manifest.entries.len() as u64 + 8));

    let mut pool = LoraWeightPool::with_capacity_bytes(capacity);
    let mut next_module_id: u32 = 0;
    let mut plans: Vec<GroupedOpPlan> = Vec::new();

    for d in &directs {
        let module_id = next_module_id;
        next_module_id += 1;
        let prefix = format!("lora.{}.{}", adapter.name, d.entry.semantic);
        pool.admit(
            SINGLE_ADAPTER_ID,
            LoraModuleId(module_id),
            factor_input(&d.spec.a_t),
            factor_input(&d.spec.b_t),
            d.spec.scale,
        )
        .map_err(|source| LoraInjectError::PoolAdmission {
            module: d.spec.module_name.clone(),
            source,
        })?;
        plans.push(GroupedOpPlan {
            prefix,
            activation: d.entry.activation,
            base_output: d.entry.base_output,
            k: d.entry.k,
            width: d.entry.n,
            module_id,
            max_rank: d.spec.rank,
            dtype: d.entry.dtype,
            fused: None,
        });
    }

    for f in fused_plans.values() {
        for (role, _offset, width) in f.group.slices {
            let module_id = next_module_id;
            next_module_id += 1;
            let prefix = format!("lora.{}.{}.{}", adapter.name, f.layer_semantic, role.as_str());
            let (max_rank, module_name) = match f.role_specs.get(&role) {
                Some(spec) => {
                    validate_module(spec, f.k, width, f.dtype)?;
                    pool.admit(
                        SINGLE_ADAPTER_ID,
                        LoraModuleId(module_id),
                        factor_input(&spec.a_t),
                        factor_input(&spec.b_t),
                        spec.scale,
                    )
                    .map_err(|source| LoraInjectError::PoolAdmission {
                        module: spec.module_name.clone(),
                        source,
                    })?;
                    (spec.rank, spec.module_name.clone())
                }
                None => {
                    // Untargeted slice: a permanently zero-rank page yields a
                    // zero delta for that role (the fused output stays base-only
                    // there), mirroring the Phase-1 zero-rank default branch.
                    let a0 = LoraFactorInput {
                        dtype: f.dtype,
                        rows: f.k,
                        cols: 0,
                        bytes: &[],
                    };
                    let b0 = LoraFactorInput {
                        dtype: f.dtype,
                        rows: 0,
                        cols: width,
                        bytes: &[],
                    };
                    pool.admit(SINGLE_ADAPTER_ID, LoraModuleId(module_id), a0, b0, 1.0)
                        .map_err(|source| LoraInjectError::PoolAdmission {
                            module: format!("{}.{} (untargeted)", f.layer_semantic, role.as_str()),
                            source,
                        })?;
                    (1, String::new())
                }
            };
            let _ = module_name;
            plans.push(GroupedOpPlan {
                prefix,
                activation: f.activation,
                base_output: f.base_output,
                k: f.k,
                width,
                module_id,
                max_rank,
                dtype: f.dtype,
                fused: Some((f.group.node_id, role)),
            });
        }
    }

    let pool = Arc::new(pool);
    let registration = LoraPoolRegistry::global().register_owned(Arc::clone(&pool));
    let pool_id = registration.pool_id();

    // --- Create the single shared non-constant `segments` routing input. ---
    let tokens_symbol = graph.create_symbol(Some("lora.segments.tokens".to_string()));
    let segments = graph.create_named_value(
        "lora.segments".to_string(),
        DataType::Int32,
        vec![Dim::Symbolic(tokens_symbol)],
    );
    graph.add_input(segments);

    // --- Emit ops: standalone deltas added directly, fused slices concatenated
    //     in [Q, K, V] order and added once. ---
    let mut fused_slice_values: BTreeMap<usize, Vec<(QkvRole, ValueId)>> = BTreeMap::new();
    let mut fused_meta: BTreeMap<usize, (FusedGroup, ValueId, DataType, String)> = BTreeMap::new();
    for f in fused_plans.values() {
        fused_meta.insert(
            f.group.node_id.0 as usize,
            (
                f.group.clone(),
                f.base_output,
                f.dtype,
                f.layer_semantic.clone(),
            ),
        );
    }

    for plan in &plans {
        let delta = emit_grouped_op(
            graph,
            &plan.prefix,
            plan.activation,
            segments,
            plan.base_output,
            plan.k,
            plan.width,
            plan.module_id,
            plan.max_rank,
            pool_id.0,
            plan.dtype,
        );
        match plan.fused {
            None => {
                add_delta_onto_base(graph, &plan.prefix, plan.base_output, delta);
            }
            Some((node_id, role)) => {
                fused_slice_values
                    .entry(node_id.0 as usize)
                    .or_default()
                    .push((role, delta));
            }
        }
    }

    // Concat each fused group's [Q, K, V] slices, then a single Add onto base.
    for (node_key, mut slices) in fused_slice_values {
        let (group, base_output, dtype, layer_semantic) = fused_meta
            .remove(&node_key)
            .expect("fused metadata recorded for every fused group");
        // Order slices as [Q, K, V] to match the fused base output columns.
        let order = |role: QkvRole| group.slices.iter().position(|(r, _, _)| *r == role);
        slices.sort_by_key(|(role, _)| order(*role));
        let fused_prefix = format!("lora.{}.{}.qkv", adapter.name, layer_semantic);
        let base_shape = graph.value(base_output).shape.clone();
        let concat = graph.create_named_value(
            format!("{fused_prefix}.delta_fused"),
            dtype,
            shape_with_last(&base_shape, Dim::Static(group.fused_n)),
        );
        let mut concat_node = Node::new(
            NodeId(0),
            "Concat",
            slices.iter().map(|(_, v)| Some(*v)).collect(),
            vec![concat],
        );
        concat_node
            .attributes
            .insert("axis".to_string(), Attribute::Int(-1));
        graph.insert_node(concat_node);
        add_delta_onto_base(graph, &fused_prefix, base_output, concat);
    }

    Ok(GroupedLoraInjection {
        segments_input: "lora.segments".to_string(),
        pool_id,
        pool,
        adapter_id: SINGLE_ADAPTER_ID,
        manifest: manifest.clone(),
        _registration: registration,
    })
}

/// Resolve the target manifest and inject an adapter as grouped
/// [`GroupedLoraDelta`](GROUPED_OP) ops in one step (the grouped analogue of
/// [`inject_lora_adapter`]). Fails loud if any targeted module cannot be
/// resolved.
pub fn inject_grouped_lora_adapter(
    graph: &mut Graph,
    adapter: &LoraAdapterSpec,
) -> Result<GroupedLoraInjection, LoraInjectError> {
    let targets: Vec<LoraTarget> = adapter
        .modules
        .iter()
        .map(|m| LoraTarget {
            module_name: m.module_name.clone(),
            layer_index: m.layer_index,
        })
        .collect();
    let manifest = build_manifest(graph, &targets)?;
    inject_grouped(graph, &manifest, adapter)
}

#[cfg(test)]
mod tests;
