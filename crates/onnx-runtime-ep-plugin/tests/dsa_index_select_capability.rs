//! Session-load capability integration for `pkg.nxrt::DsaIndexSelect` v1.
//!
//! These tests drive the **full** cpu-plugin capability path end-to-end — the
//! same predicate ORT's `ep_get_capability` applies at session load:
//!
//! ```text
//! C_ABI_claims = query_capabilities(cpu_ep) ∩ { nodes where for_node ≠ Declined }
//! ```
//!
//! The two halves are:
//!
//! 1. **Trait half** — the CPU EP claim gate
//!    (`CpuExecutionProvider::supports_op` → `dsa_index_select::unsupported_reason`),
//!    wired into `GetCapability` by the merged #2053 revision. It is the only
//!    half that sees input **dtypes**, so it is the half that must catch dtype
//!    violations (non-`Float32` `attention_bias`, mismatched or non-float
//!    query/key/weights storage) at capability time.
//! 2. **Shape half** — the plugin shape rule
//!    (`ShapeInference::for_node` → `build_dsa_index_select`). It sees only
//!    shapes/attrs, so it catches ABI-attribute, rank and static-dimension
//!    violations; it deliberately returns `DsaIndexSelect { top_k }` for a
//!    dtype-only violation because dtypes are not visible to shape inference.
//!
//! The native CUDA `DsaIndexSelect` kernel (#2076, this branch) claims the same
//! LATENT subset on the GPU side and delegates its dtype gate to the very same
//! CPU validator, so proving the CPU plugin claims valid `Float32`/`Float16`/
//! `BFloat16` storage nodes and **declines every capability-time-invalid node
//! without a post-claim hard fail** pins the cross-provider claim contract from
//! the session-load side. Post-claim session-load drops (over-claiming a node
//! the plugin cannot shape, or under-claiming valid low-precision storage) are
//! exactly the defect class that gated the earlier #2053 revisions.

use onnx_runtime_ep_api::EpConfig;
use onnx_runtime_ep_api::abi::OrtGraphView;
use onnx_runtime_ep_api::provider::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_plugin::compute::ShapeInference;
use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, GraphView, GraphViewCache, Node, NodeId};

// Baseline GLM-shaped DSA index-selection geometry (batch=2, q_seq=3, heads=4,
// head_dim=8, key_seq=16, top_k=4). Every invalid spec below flips exactly one
// axis off this baseline so each test pins exactly one rejection.
const BATCH: usize = 2;
const Q_SEQ: usize = 3;
const HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const KEY_SEQ: usize = 16;
const TOP_K: i64 = 4;

/// A single-node `DsaIndexSelect` graph specification. `valid` builds the
/// baseline; callers mutate one field to construct exactly one violation.
struct DsaSpec {
    domain: String,
    version: i64,
    query_dtype: DataType,
    key_dtype: DataType,
    weights_dtype: DataType,
    bias_dtype: DataType,
    query_shape: Vec<usize>,
    key_shape: Vec<usize>,
    weights_shape: Vec<usize>,
    bias_shape: Vec<usize>,
    top_k: Attribute,
    scale: Option<Attribute>,
    weights_scale: Option<Attribute>,
    extra_attr: Option<(&'static str, Attribute)>,
}

impl DsaSpec {
    /// Baseline: property-compatible node with `storage` for query/key/weights
    /// and the mandatory `Float32` `attention_bias`.
    fn valid(storage: DataType) -> Self {
        Self {
            domain: "pkg.nxrt".into(),
            version: 1,
            query_dtype: storage,
            key_dtype: storage,
            weights_dtype: storage,
            bias_dtype: DataType::Float32,
            query_shape: vec![BATCH, Q_SEQ, HEADS, HEAD_DIM],
            key_shape: vec![BATCH, KEY_SEQ, HEAD_DIM],
            weights_shape: vec![BATCH, Q_SEQ, HEADS],
            bias_shape: vec![BATCH, 1, Q_SEQ, KEY_SEQ],
            top_k: Attribute::Int(TOP_K),
            scale: Some(Attribute::Float(0.125)),
            weights_scale: Some(Attribute::Float(0.5)),
            extra_attr: None,
        }
    }

    fn shape(dims: &[usize]) -> Vec<Dim> {
        dims.iter().map(|&d| Dim::Static(d)).collect()
    }

    fn build(&self) -> Graph {
        let mut graph = Graph::default();
        graph.opset_imports.insert(String::new(), 17);
        graph
            .opset_imports
            .insert(self.domain.clone(), self.version as u64);

        let query =
            graph.create_named_value("query", self.query_dtype, Self::shape(&self.query_shape));
        let key = graph.create_named_value("key", self.key_dtype, Self::shape(&self.key_shape));
        let weights = graph.create_named_value(
            "weights",
            self.weights_dtype,
            Self::shape(&self.weights_shape),
        );
        let bias = graph.create_named_value(
            "attention_bias",
            self.bias_dtype,
            Self::shape(&self.bias_shape),
        );
        for value in [query, key, weights, bias] {
            graph.add_input(value);
        }
        let output = graph.create_named_value("selected_indices", DataType::Int64, vec![]);
        graph.add_output(output);

        let mut node = Node::new(
            NodeId(0),
            "DsaIndexSelect",
            vec![Some(query), Some(key), Some(weights), Some(bias)],
            vec![output],
        );
        node.domain = self.domain.clone();
        node.version = Some(self.version);
        node.attributes.insert("top_k".into(), self.top_k.clone());
        if let Some(scale) = &self.scale {
            node.attributes.insert("scale".into(), scale.clone());
        }
        if let Some(weights_scale) = &self.weights_scale {
            node.attributes
                .insert("weights_scale".into(), weights_scale.clone());
        }
        if let Some((name, value)) = &self.extra_attr {
            node.attributes.insert((*name).into(), value.clone());
        }
        graph.insert_node(node);
        graph
    }

    fn input_shapes(&self) -> Vec<Vec<Option<usize>>> {
        [
            &self.query_shape,
            &self.key_shape,
            &self.weights_shape,
            &self.bias_shape,
        ]
        .iter()
        .map(|dims| dims.iter().copied().map(Some).collect())
        .collect()
    }
}

/// Result of driving the full session-load capability predicate on a spec.
struct Capability {
    /// The CPU EP claim gate (`GetCapability` trait half) claimed the node.
    trait_supported: bool,
    /// The plugin shape rule declined the node.
    shape_declined: bool,
    /// `ShapeInference::for_node` result, for exact `DsaIndexSelect { top_k }` checks.
    shape: ShapeInference,
    /// The combined C-ABI predicate: `trait_supported && !shape_declined`.
    ///
    /// This mirrors `ep_get_capability` exactly — a node ORT keeps at session
    /// load. `false` means the node is excluded at capability time (partitioned
    /// around), never a post-claim hard fail.
    cabi_claimed: bool,
}

fn make_cpu_ep() -> CpuExecutionProvider {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&EpConfig::default()).unwrap();
    ep
}

fn evaluate(spec: &DsaSpec) -> Capability {
    let ep = make_cpu_ep();
    let graph = spec.build();
    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);
    let node_idx = view.nodes().next().unwrap();
    let node = view.node(node_idx);

    // Trait half: the CPU EP claim gate, driven at the node's effective opset —
    // the same opset `query_capabilities` derives from `opset_imports`.
    let opset = graph.effective_opset(node).unwrap_or(0);
    let trait_supported = ep.supports_node(&view, node_idx, opset).is_supported();

    // Shape half: the plugin shape rule.
    let shape = ShapeInference::for_node(node, &spec.input_shapes(), 1);
    let shape_declined = matches!(shape, ShapeInference::Declined { .. });

    // Full C-ABI path: the trait-only `query_capabilities` first half must agree
    // with the direct `supports_node` claim (single-node graph), and the
    // combined predicate is what ORT keeps at session load.
    let ort_view = OrtGraphView::new(&view);
    let trait_claims = ort_view.query_capabilities(&ep);
    assert_eq!(
        trait_claims.is_empty(),
        !trait_supported,
        "query_capabilities (C-ABI trait half) must agree with supports_node for the single node"
    );

    Capability {
        trait_supported,
        shape_declined,
        shape,
        cabi_claimed: trait_supported && !shape_declined,
    }
}

// ─── Valid storage: claimed at session load ─────────────────────────────────

/// The CPU plugin must claim a property-compatible `DsaIndexSelect` node for
/// each supported storage dtype — `Float32`, `Float16` and `BFloat16` — at
/// session load. The `Float16`/`BFloat16` claims exercise the dtype edge union
/// the merged #2053 revision advertises; a catch-all `Float32`-only constraint
/// would have wrongly dropped the low-precision GLM storage path.
#[test]
fn dsa_index_select_session_load_claims_f32_f16_bf16_storage() {
    for storage in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let cap = evaluate(&DsaSpec::valid(storage));
        assert!(
            cap.trait_supported,
            "CPU claim gate must claim {storage:?} storage DsaIndexSelect at capability time"
        );
        assert!(
            !cap.shape_declined,
            "plugin shape rule must not decline valid {storage:?} storage DsaIndexSelect"
        );
        assert!(
            matches!(cap.shape, ShapeInference::DsaIndexSelect { top_k } if top_k == TOP_K as usize),
            "shape rule must infer DsaIndexSelect {{ top_k = {TOP_K} }} for {storage:?}, got {:?}",
            cap.shape
        );
        assert!(
            cap.cabi_claimed,
            "full session-load capability path must keep valid {storage:?} storage DsaIndexSelect"
        );
    }
}

// ─── Capability-time rejects: excluded, never a post-claim hard fail ─────────

/// `attention_bias` carries the `-inf`/finfo.min causal sentinel and is
/// `Float32`-only. A `Float16` bias must be declined **at capability time** by
/// the CPU claim gate (the dtype-aware half). The shape rule cannot see the
/// dtype, so it still returns `DsaIndexSelect` — proving the trait half is what
/// keeps the malformed node out of the session-load claim set.
#[test]
fn dsa_index_select_capability_rejects_non_f32_bias() {
    let mut spec = DsaSpec::valid(DataType::Float16);
    spec.bias_dtype = DataType::Float16;
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline non-Float32 attention_bias at capability time"
    );
    assert!(
        !cap.shape_declined,
        "shape rule cannot see dtypes, so it must still shape the node (trait half rejects)"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the non-Float32 bias node at session load"
    );
}

/// Query/key/weights must share one storage dtype. A mixed
/// `Float16`-query/`Float32`-key node must be declined at capability time by the
/// CPU claim gate (again the dtype-aware half).
#[test]
fn dsa_index_select_capability_rejects_mismatched_storage_dtype() {
    let mut spec = DsaSpec::valid(DataType::Float16);
    spec.key_dtype = DataType::Float32;
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline mismatched query/key storage dtype at capability time"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the mismatched-storage node at session load"
    );
}

/// Non-float storage (e.g. `Int64` query) must be declined at capability time by
/// the CPU claim gate; the op only supports float compute storage.
#[test]
fn dsa_index_select_capability_rejects_non_float_query_storage() {
    let mut spec = DsaSpec::valid(DataType::Float32);
    spec.query_dtype = DataType::Int64;
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline non-float query storage at capability time"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the non-float-query node at session load"
    );
}

/// An attribute outside the frozen v1 ABI must be declined at capability time by
/// **both** halves — the CPU claim gate and the plugin shape rule.
#[test]
fn dsa_index_select_capability_rejects_unknown_attribute() {
    let mut spec = DsaSpec::valid(DataType::BFloat16);
    spec.extra_attr = Some(("axis", Attribute::Int(-1)));
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline an out-of-ABI attribute at capability time"
    );
    assert!(
        cap.shape_declined,
        "plugin shape rule must also decline an out-of-ABI attribute"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the unknown-attribute node at session load"
    );
}

/// A static cross-input dimension conflict (query head_dim 8 vs key head_dim 9)
/// must be declined at capability time by both halves.
#[test]
fn dsa_index_select_capability_rejects_inconsistent_static_dims() {
    let mut spec = DsaSpec::valid(DataType::Float32);
    spec.key_shape = vec![BATCH, KEY_SEQ, HEAD_DIM + 1];
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline a query/key head_dim conflict at capability time"
    );
    assert!(
        cap.shape_declined,
        "plugin shape rule must also decline a query/key head_dim conflict"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the inconsistent-dims node at session load"
    );
}

/// A wrong opset (v2) must be excluded from the session-load claim set. The CPU
/// registry follows ONNX since-version semantics (an op registered since v1 is
/// claimable at any opset ≥ 1), so the **plugin shape rule** is the half that
/// pins the exact `pkg.nxrt` v1 dispatch and declines v2 — keeping the node out
/// of the combined C-ABI claim set at capability time.
#[test]
fn dsa_index_select_capability_rejects_wrong_opset() {
    let mut spec = DsaSpec::valid(DataType::Float32);
    spec.version = 2;
    let cap = evaluate(&spec);
    assert!(
        cap.trait_supported,
        "CPU registry follows ONNX since-version semantics, so it still claims DsaIndexSelect at opset 2 — the shape half is the gate here"
    );
    assert!(
        cap.shape_declined,
        "plugin shape rule dispatches only pkg.nxrt v1, so opset 2 must decline"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the wrong-opset node at session load"
    );
}

/// A wrong domain must be declined at capability time by both halves.
#[test]
fn dsa_index_select_capability_rejects_wrong_domain() {
    let mut spec = DsaSpec::valid(DataType::Float32);
    spec.domain = "com.example".into();
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU registry must not claim DsaIndexSelect under a foreign domain"
    );
    assert!(
        cap.shape_declined,
        "plugin shape rule dispatches only pkg.nxrt, so a foreign domain must decline"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the wrong-domain node at session load"
    );
}

/// A zero `top_k` must be declined at capability time by both halves.
#[test]
fn dsa_index_select_capability_rejects_zero_top_k() {
    let mut spec = DsaSpec::valid(DataType::Float32);
    spec.top_k = Attribute::Int(0);
    let cap = evaluate(&spec);
    assert!(
        !cap.trait_supported,
        "CPU claim gate must decline top_k = 0 at capability time"
    );
    assert!(
        cap.shape_declined,
        "plugin shape rule must decline top_k = 0"
    );
    assert!(
        !cap.cabi_claimed,
        "full capability path must exclude the zero-top_k node at session load"
    );
}
