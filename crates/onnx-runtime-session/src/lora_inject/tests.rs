//! Tests for the native-LoRA manifest builder (P3) and injection pass (P2b),
//! plus the golden numerics suite (design `docs/NATIVE_LORA_DESIGN.md` §E).
//!
//! Two families:
//!
//!   * **Discovery** (`build_manifest`) — structural, no execution. Synthetic
//!     graphs shaped like the three verified exports (§H): Qwen2.5 fused
//!     `qkv_proj`, Qwen3 split `q/k/v/o_proj`, and Qwen3.5 linear-attention
//!     `in_proj_qkv`. They assert the layout is detected structurally (by node
//!     `N` versus the GQA head dims, and the projection token) and that an
//!     unresolvable target **fails loud** with a typed error naming the module.
//!
//!   * **Numerics** (`inject` + `Executor::build_with_overrides`) — execution.
//!     A plain fp32 `MatMul` stands in for the int4 base projection (the base
//!     op is never touched by the pass, so its quantization is irrelevant to
//!     the delta arithmetic — this simplification is called out explicitly in
//!     the golden test). They feed `A_t`/`B_t` through the P1 override
//!     mechanism and assert `Y == Y_base + scale * ((x @ A_t) @ B_t)`, the
//!     unfed base-only no-op, a rank change, correct fused-QKV slice placement,
//!     and optimizer survival.


use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
    static_shape,
};
use onnx_genai_metadata::{LoraTargetDescriptor, LoraTargetManifest, LoraTargetSlice};
use onnx_runtime_loader::WeightStore;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::{
    build_manifest, inject, inject_grouped, inject_grouped_lora_adapter, inject_grouped_multi,
    FusedGroup, LoraAdapterSpec, LoraInjectError, LoraManifest, LoraModuleSpec, LoraTarget,
    Placement, QkvRole, TargetEntry,
};
use crate::Tensor;
use crate::executor::{auto_detect_cpu_ep, Executor};
use onnx_runtime_ep_api::{AdapterId, LoraPoolRegistry};

const F32: DataType = DataType::Float32;

// ===========================================================================
// Shared graph-construction helpers.
// ===========================================================================

/// Little-endian f32 bytes for an inline initializer.
fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Insert a `MatMulNBits` base projection node (the op discovery keys on),
/// naming its weight input so `build_manifest` can parse `(layer, proj_token)`
/// exactly as it does on a real export (§H). Returns the node id and its output
/// value. The weight/activation shapes are irrelevant to discovery.
fn add_matmulnbits(
    graph: &mut Graph,
    node_name: &str,
    weight_name: &str,
    activation: ValueId,
    k: usize,
    n: usize,
) -> (NodeId, ValueId) {
    let weight = graph.create_named_value(weight_name, DataType::Uint8, static_shape([1]));
    let out = graph.create_named_value(
        format!("{node_name}.out"),
        F32,
        static_shape([1, n]),
    );
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(activation), Some(weight)],
        vec![out],
    );
    node.name = node_name.to_string();
    node.domain = "com.microsoft".to_string();
    node.attributes.insert("K".to_string(), Attribute::Int(k as i64));
    node.attributes.insert("N".to_string(), Attribute::Int(n as i64));
    node.attributes.insert("bits".to_string(), Attribute::Int(4));
    node.attributes
        .insert("block_size".to_string(), Attribute::Int(32));
    let id = graph.insert_node(node);
    (id, out)
}

/// Insert a `GroupQueryAttention` node carrying the head configuration that
/// determines a fused projection's Q/K/V slice widths (§H).
fn add_gqa(graph: &mut Graph, node_name: &str, num_heads: usize, kv_num_heads: usize) {
    let din = graph.create_named_value(format!("{node_name}.in"), F32, static_shape([1]));
    let dout = graph.create_named_value(format!("{node_name}.out"), F32, static_shape([1]));
    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![Some(din)],
        vec![dout],
    );
    node.name = node_name.to_string();
    node.attributes
        .insert("num_heads".to_string(), Attribute::Int(num_heads as i64));
    node.attributes.insert(
        "kv_num_heads".to_string(),
        Attribute::Int(kv_num_heads as i64),
    );
    graph.insert_node(node);
}

/// A shared activation input `x` for the discovery graphs.
fn add_activation(graph: &mut Graph, k: usize) -> ValueId {
    let x = graph.create_named_value("hidden", F32, static_shape([1, k]));
    graph.add_input(x);
    x
}

fn declared_fused_qkv(node_name: &str, k: usize, n: usize) -> LoraTargetManifest {
    let slices = BTreeMap::from([
        ("q_proj".to_string(), LoraTargetSlice {
            offset: 0, width: 3584, rank: None, alpha: None,
        }),
        ("k_proj".to_string(), LoraTargetSlice {
            offset: 3584, width: 512, rank: None, alpha: None,
        }),
        ("v_proj".to_string(), LoraTargetSlice {
            offset: 4096, width: 512, rank: None, alpha: None,
        }),
    ]);
    LoraTargetManifest {
        targets: vec![LoraTargetDescriptor {
            module_name: "self_attn.qkv_proj".to_string(),
            layer_index: 0,
            node_name: node_name.to_string(),
            output_name: format!("{node_name}.out"),
            k,
            n,
            slices,
            rank: None,
            alpha: None,
        }],
    }
}

// ===========================================================================
// P3 — discovery / manifest tests (structural, no execution).
// ===========================================================================

// Qwen3-style SPLIT attention: q/k/v/o_proj are four separate MatMulNBits
// nodes with `.MatMulNBits.qweight` weights. Every target resolves Direct to
// its own node with the node's full N. (§H: Qwen3-0.6b q N=2048, k/v N=1024.)
#[test]
fn discovery_split_qwen3_resolves_direct() {
    let mut graph = Graph::new();
    let k = 2048;
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/q_proj/MatMulNBits",
        "model.layers.0.attn.q_proj.MatMulNBits.qweight",
        x,
        k,
        2048,
    );
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/k_proj/MatMulNBits",
        "model.layers.0.attn.k_proj.MatMulNBits.qweight",
        x,
        k,
        1024,
    );
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/v_proj/MatMulNBits",
        "model.layers.0.attn.v_proj.MatMulNBits.qweight",
        x,
        k,
        1024,
    );
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/o_proj/MatMulNBits",
        "model.layers.0.attn.o_proj.MatMulNBits.qweight",
        x,
        2048,
        2048,
    );
    // A split layout does not need GQA to resolve, but a real export has one.
    add_gqa(&mut graph, "/model/layers.0/attn/GroupQueryAttention", 16, 8);

    let targets = vec![
        LoraTarget { module_name: "self_attn.q_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.k_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.v_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.o_proj".into(), layer_index: 0 },
    ];
    let manifest = build_manifest(&graph, &targets, None).expect("split layout resolves");
    assert_eq!(manifest.entries.len(), 4);
    for (entry, expected_n) in manifest.entries.iter().zip([2048, 1024, 1024, 2048]) {
        assert!(
            matches!(entry.placement, Placement::Direct),
            "{} must be a Direct target, got {:?}",
            entry.semantic,
            entry.placement
        );
        assert_eq!(entry.n, expected_n, "{}", entry.semantic);
        assert_eq!(entry.k, if entry.semantic.ends_with("o_proj") { 2048 } else { k });
    }
}

// Qwen2.5-style FUSED attention: one `qkv_proj` MatMulNBits (§H node
// `/model/layers.0/attn/qkv_proj/MatMul_Q4`, weight
// `model.layers.0.attn.qkv_proj.MatMul.weight_Q4`, K=3584 N=4608, GQA
// num_heads=28 kv_num_heads=4). q/k/v targets resolve to FusedSlice with the
// verified offsets q[0:3584] k[3584:4096] v[4096:4608].
#[test]
fn discovery_fused_qwen25_resolves_slices() {
    let mut graph = Graph::new();
    let k = 3584;
    let x = add_activation(&mut graph, k);
    let (qkv_id, _) = add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/qkv_proj/MatMul_Q4",
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4",
        x,
        k,
        4608,
    );
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/o_proj/MatMul_Q4",
        "model.layers.0.attn.o_proj.MatMul.weight_Q4",
        x,
        3584,
        3584,
    );
    add_gqa(&mut graph, "/model/layers.0/attn/GroupQueryAttention", 28, 4);

    let targets = vec![
        LoraTarget { module_name: "self_attn.q_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.k_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.v_proj".into(), layer_index: 0 },
    ];
    let manifest = build_manifest(&graph, &targets, None).expect("fused layout resolves");

    let expected = [
        (QkvRole::Q, 0usize, 3584usize),
        (QkvRole::K, 3584, 512),
        (QkvRole::V, 4096, 512),
    ];
    for (entry, (role, offset, width)) in manifest.entries.iter().zip(expected) {
        match &entry.placement {
            Placement::FusedSlice { group, role: got_role } => {
                assert_eq!(group.node_id, qkv_id);
                assert_eq!(group.fused_n, 4608);
                assert_eq!(*got_role, role, "{}", entry.semantic);
                assert_eq!(entry.n, width, "{} slice width", entry.semantic);
                let slice = group.slices.iter().find(|(r, _, _)| *r == role).unwrap();
                assert_eq!((slice.1, slice.2), (offset, width), "{} offset", entry.semantic);
            }
            other => panic!("{} expected FusedSlice, got {other:?}", entry.semantic),
        }
        // All three slices share the fused node's activation input and K.
        assert_eq!(entry.k, k);
    }
    // The [Q, K, V] slices exactly tile the fused width.
    if let Placement::FusedSlice { group, .. } = &manifest.entries[0].placement {
        let sum: usize = group.slices.iter().map(|(_, _, w)| *w).sum();
        assert_eq!(sum, group.fused_n);
    }
}

#[test]
fn declared_manifest_matches_graph_discovery_for_fused_qkv() {
    let mut graph = Graph::new();
    let k = 3584;
    let node_name = "/model/layers.0/attn/qkv_proj/MatMul_Q4";
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph, node_name,
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4", x, k, 4608,
    );
    add_gqa(&mut graph, "/model/layers.0/attn/GroupQueryAttention", 28, 4);
    let targets = vec![
        LoraTarget { module_name: "self_attn.q_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.k_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "self_attn.v_proj".into(), layer_index: 0 },
    ];

    let discovered = build_manifest(&graph, &targets, None).expect("graph discovery");
    let declaration = declared_fused_qkv(node_name, k, 4608);
    let declared =
        build_manifest(&graph, &targets, Some(&declaration)).expect("declared manifest");

    assert_eq!(declared.entries.len(), discovered.entries.len());
    for (declared, discovered) in declared.entries.iter().zip(&discovered.entries) {
        assert_eq!(declared.semantic, discovered.semantic);
        assert_eq!(declared.node_id, discovered.node_id);
        assert_eq!(declared.base_output, discovered.base_output);
        assert_eq!(declared.activation, discovered.activation);
        assert_eq!(declared.k, discovered.k);
        assert_eq!(declared.n, discovered.n);
        assert_eq!(declared.dtype, discovered.dtype);
        assert_eq!(declared.placement, discovered.placement);
    }
}

#[test]
fn declared_manifest_missing_node_returns_typed_error() {
    let mut graph = Graph::new();
    let x = add_activation(&mut graph, 3584);
    add_matmulnbits(
        &mut graph, "/model/layers.0/attn/qkv_proj/MatMul_Q4",
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4", x, 3584, 4608,
    );
    let declaration =
        declared_fused_qkv("/model/layers.0/attn/missing/MatMul_Q4", 3584, 4608);
    let targets = [LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];

    let error = build_manifest(&graph, &targets, Some(&declaration))
        .expect_err("missing declared node must fail");
    assert!(matches!(error, LoraInjectError::DeclaredNodeMissing { .. }));
}

#[test]
fn declared_manifest_dimension_mismatch_returns_typed_error() {
    let mut graph = Graph::new();
    let node_name = "/model/layers.0/attn/qkv_proj/MatMul_Q4";
    let x = add_activation(&mut graph, 3584);
    add_matmulnbits(
        &mut graph, node_name,
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4", x, 3584, 4608,
    );
    let declaration = declared_fused_qkv(node_name, 4096, 4608);
    let targets = [LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];

    let error = build_manifest(&graph, &targets, Some(&declaration))
        .expect_err("declared K mismatch must fail");
    assert!(matches!(
        error,
        LoraInjectError::DeclaredDimensionMismatch {
            dimension: "K",
            declared: 4096,
            actual: 3584,
            ..
        }
    ));
}

#[test]
fn absent_declared_manifest_falls_back_to_graph_discovery() {
    let mut graph = Graph::new();
    let x = add_activation(&mut graph, 2048);
    add_matmulnbits(
        &mut graph, "/model/layers.0/attn/q_proj/MatMulNBits",
        "model.layers.0.attn.q_proj.MatMulNBits.qweight", x, 2048, 2048,
    );
    let targets = [LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];

    let manifest = build_manifest(&graph, &targets, None).expect("fallback discovery");
    assert_eq!(manifest.entries.len(), 1);
    assert!(matches!(manifest.entries[0].placement, Placement::Direct));
}

// Qwen3.5-style LINEAR attention: the projection is `in_proj_qkv`, a token the
// pass does not recognize as q/k/v or as a fused `qkv_proj`. A q_proj target
// must FAIL LOUD (UnresolvedModule) rather than silently mapping onto the
// linear-attention projection.
#[test]
fn discovery_linear_attn_fails_loud() {
    let mut graph = Graph::new();
    let k = 2048;
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/linear_attn/in_proj_qkv/MatMul_Q4",
        "model.layers.0.linear_attn.in_proj_qkv.MatMul.weight_Q4",
        x,
        k,
        4096,
    );

    let targets = vec![LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];
    let err = build_manifest(&graph, &targets, None).expect_err("q_proj must not resolve");
    match err {
        LoraInjectError::UnresolvedModule { module, layer } => {
            assert_eq!(module, "self_attn.q_proj");
            assert_eq!(layer, 0);
        }
        other => panic!("expected UnresolvedModule, got {other:?}"),
    }
}

// A directly named `in_proj_qkv` target DOES resolve Direct — proving the
// linear-attention failure above is a fused/split mismatch, not an inability
// to see the node at all.
#[test]
fn discovery_linear_attn_direct_token_resolves() {
    let mut graph = Graph::new();
    let k = 2048;
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/linear_attn/in_proj_qkv/MatMul_Q4",
        "model.layers.0.linear_attn.in_proj_qkv.MatMul.weight_Q4",
        x,
        k,
        4096,
    );

    let targets = vec![LoraTarget {
        module_name: "linear_attn.in_proj_qkv".into(),
        layer_index: 0,
    }];
    let manifest = build_manifest(&graph, &targets, None).expect("direct token resolves");
    assert!(matches!(manifest.entries[0].placement, Placement::Direct));
    assert_eq!(manifest.entries[0].n, 4096);
}

// A fused q/k/v target with no GroupQueryAttention in the layer cannot derive
// slice widths and must fail loud with MissingAttention (never guess).
#[test]
fn discovery_fused_missing_attention_fails_loud() {
    let mut graph = Graph::new();
    let k = 3584;
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/qkv_proj/MatMul_Q4",
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4",
        x,
        k,
        4608,
    );
    // No GQA node inserted.

    let targets = vec![LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];
    let err = build_manifest(&graph, &targets, None).expect_err("no GQA => cannot slice");
    match err {
        LoraInjectError::MissingAttention { fused_n, layer, .. } => {
            assert_eq!(fused_n, 4608);
            assert_eq!(layer, 0);
        }
        other => panic!("expected MissingAttention, got {other:?}"),
    }
}

// A fused projection whose N is inconsistent with the GQA head counts fails
// loud with FusedGeometry rather than silently producing wrong slice widths.
#[test]
fn discovery_fused_geometry_mismatch_fails_loud() {
    let mut graph = Graph::new();
    let k = 3584;
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/attn/qkv_proj/MatMul_Q4",
        "model.layers.0.attn.qkv_proj.MatMul.weight_Q4",
        x,
        k,
        4608,
    );
    // num_heads + 2*kv = 30 + 2*4 = 38, and 4608 % 38 != 0.
    add_gqa(&mut graph, "/model/layers.0/attn/GroupQueryAttention", 30, 4);

    let targets = vec![LoraTarget {
        module_name: "self_attn.q_proj".into(),
        layer_index: 0,
    }];
    let err = build_manifest(&graph, &targets, None).expect_err("bad geometry must fail");
    assert!(
        matches!(err, LoraInjectError::FusedGeometry { .. }),
        "expected FusedGeometry, got {err:?}"
    );
}

// A MatMulNBits node missing its `K` attribute fails loud (the manifest cannot
// fabricate an inner dimension).
#[test]
fn discovery_missing_attribute_fails_loud() {
    let mut graph = Graph::new();
    let x = add_activation(&mut graph, 8);
    let weight = graph.create_named_value(
        "model.layers.0.mlp.gate_proj.MatMulNBits.qweight",
        DataType::Uint8,
        static_shape([1]),
    );
    let out = graph.create_named_value("gate.out", F32, static_shape([1, 8]));
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(x), Some(weight)],
        vec![out],
    );
    node.name = "/model/layers.0/mlp/gate_proj/MatMulNBits".to_string();
    // Only N is set; K is missing.
    node.attributes.insert("N".to_string(), Attribute::Int(8));
    graph.insert_node(node);

    let targets = vec![LoraTarget {
        module_name: "mlp.gate_proj".into(),
        layer_index: 0,
    }];
    let err = build_manifest(&graph, &targets, None).expect_err("missing K must fail");
    match err {
        LoraInjectError::MissingAttribute { attr, .. } => assert_eq!(attr, "K"),
        other => panic!("expected MissingAttribute, got {other:?}"),
    }
}

// MLP projections (gate/up/down) resolve Direct, confirming discovery is not
// attention-specific.
#[test]
fn discovery_mlp_projections_resolve_direct() {
    let mut graph = Graph::new();
    let x = add_activation(&mut graph, 3584);
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/mlp/gate_proj/MatMul_Q4",
        "model.layers.0.mlp.gate_proj.MatMul.weight_Q4",
        x,
        3584,
        18944,
    );
    add_matmulnbits(
        &mut graph,
        "/model/layers.0/mlp/down_proj/MatMul_Q4",
        "model.layers.0.mlp.down_proj.MatMul.weight_Q4",
        x,
        18944,
        3584,
    );
    let targets = vec![
        LoraTarget { module_name: "mlp.gate_proj".into(), layer_index: 0 },
        LoraTarget { module_name: "mlp.down_proj".into(), layer_index: 0 },
    ];
    let manifest = build_manifest(&graph, &targets, None).expect("mlp resolves");
    assert!(matches!(manifest.entries[0].placement, Placement::Direct));
    assert_eq!(manifest.entries[0].n, 18944);
    assert!(matches!(manifest.entries[1].placement, Placement::Direct));
    assert_eq!(manifest.entries[1].n, 3584);
}

// ===========================================================================
// Numerics helpers.
// ===========================================================================

/// Reference `delta[m, n] = scale * (x[m,k] @ A_t[k,r]) @ B_t[r,n]` in f64.
fn reference_delta(
    m: usize,
    k: usize,
    r: usize,
    n: usize,
    scale: f32,
    x: &[f32],
    a_t: &[f32],
    b_t: &[f32],
) -> Vec<f32> {
    let mut rmid = vec![0.0f64; m * r];
    for i in 0..m {
        for j in 0..r {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += x[i * k + p] as f64 * a_t[p * r + j] as f64;
            }
            rmid[i * r + j] = acc;
        }
    }
    let mut delta = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..r {
                acc += rmid[i * r + p] * b_t[p * n + j] as f64;
            }
            delta[i * n + j] = (acc * scale as f64) as f32;
        }
    }
    delta
}

/// Reference base `base[m,n] = x[m,k] @ W[k,n]` in f64.
fn reference_base(m: usize, k: usize, n: usize, x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += x[i * k + p] as f64 * w[p * n + j] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// Build a plain fp32 `MatMul` base projection `base = x @ W` and register
/// `base` as the graph output. Returns `(node id, x, base output)`. This stands
/// in for the int4 `MatMulNBits` base — the injection pass never touches the
/// base op, so its quantization is orthogonal to the delta arithmetic tested
/// here (design §E simplification).
fn add_base_matmul(
    graph: &mut Graph,
    m: usize,
    k: usize,
    n: usize,
    w: &[f32],
) -> (NodeId, ValueId, ValueId) {
    let x = graph.create_named_value("x", F32, static_shape([m, k]));
    graph.add_input(x);
    let weight = graph.create_named_value("W", F32, static_shape([k, n]));
    graph.set_initializer(
        weight,
        WeightRef::Inline(TensorData::from_raw(F32, vec![k, n], f32_le_bytes(w))),
    );
    let base = graph.create_named_value("base", F32, static_shape([m, n]));
    let id = graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(weight)],
        vec![base],
    ));
    graph.add_output(base);
    (id, x, base)
}

fn f32_module(
    module_name: &str,
    rank: usize,
    scale: f32,
    a_t: &[f32],
    b_t: &[f32],
    k: usize,
    n: usize,
) -> LoraModuleSpec {
    LoraModuleSpec {
        module_name: module_name.to_string(),
        layer_index: 0,
        rank,
        scale,
        a_t: TensorData::from_raw(F32, vec![k, rank], f32_le_bytes(a_t)),
        b_t: TensorData::from_raw(F32, vec![rank, n], f32_le_bytes(b_t)),
    }
}

fn direct_entry(node_id: NodeId, base: ValueId, x: ValueId, k: usize, n: usize) -> TargetEntry {
    TargetEntry {
        semantic: "layers.0.q_proj".to_string(),
        node_id,
        base_output: base,
        activation: x,
        k,
        n,
        dtype: F32,
        placement: Placement::Direct,
    }
}

fn count_op(graph: &Graph, op: &str) -> usize {
    graph.nodes.iter().filter(|(_, n)| n.op_type == op).count()
}

fn has_input_named(graph: &Graph, name: &str) -> bool {
    graph
        .inputs
        .iter()
        .any(|vid| graph.value(*vid).name.as_deref() == Some(name))
}

// ===========================================================================
// P2b — golden numerics (execution).
// ===========================================================================

// Golden: with A_t/B_t fed, Y == Y_base + scale * ((x @ A_t) @ B_t) element-wise
// within fp tolerance. Base is a plain fp32 MatMul (simplification noted above).
#[test]
fn golden_direct_fed_matches_reference() {
    let (m, k, n, r) = (2, 4, 3, 2);
    let scale = 0.5f32;
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0];
    let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale, &a, &b, k, n)],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject");
    assert_eq!(overrides.len(), 2, "A_t and B_t become overridable inputs");

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();

    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let a_t = Tensor::from_f32(&[k, r], &a).unwrap();
    let b_t = Tensor::from_f32(&[r, n], &b).unwrap();
    let outputs = exec
        .run(&[
            ("x", &x_t),
            ("lora.adapter.layers.0.q_proj.A_t", &a_t),
            ("lora.adapter.layers.0.q_proj.B_t", &b_t),
        ])
        .unwrap();

    let base = reference_base(m, k, n, &x, &w);
    let delta = reference_delta(m, k, r, n, scale, &x, &a, &b);
    let expected: Vec<f32> = base.iter().zip(&delta).map(|(a, b)| a + b).collect();
    let got = outputs[0].to_vec_f32();
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-3, "got {got:?} expected {expected:?}");
    }
}

// Unfed base-only no-op: with A_t/B_t unfed, the zero-rank defaults collapse
// the delta MatMul (k == 0 zero-fill on CPU) and Y is bit-identical to the base.
#[test]
fn golden_direct_unfed_is_base_only() {
    let (m, k, n, r) = (2, 4, 3, 2);
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0];
    let a = vec![0.7f32; k * r];
    let b = vec![-0.4f32; r * n];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, 2.0, &a, &b, k, n)],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject");
    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();

    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let outputs = exec.run(&[("x", &x_t)]).unwrap();
    let base = reference_base(m, k, n, &x, &w);
    assert_eq!(outputs[0].to_vec_f32(), base, "unfed => base-only, exactly");
}

// Rank change: the same injected graph accepts different fed ranks (r=2 then
// r=5) without rebuild-side geometry baked in, because A_t/B_t carry the rank
// as a symbolic dim resolved at feed time (P1).
#[test]
fn golden_direct_rank_change() {
    let (m, k, n) = (2, 4, 3);
    let scale = 1.5f32;
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let x: Vec<f32> = vec![0.3, -0.7, 1.1, 0.2, -1.0, 0.5, 0.9, -0.1];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    // The declared spec rank is irrelevant to a fed override's rank; use r=2
    // for the initial spec (validated against the [K,r]/[r,N] factors below).
    let r0 = 2;
    let a0: Vec<f32> = (0..k * r0).map(|i| (i as f32) * 0.1).collect();
    let b0: Vec<f32> = (0..r0 * n).map(|i| (i as f32) * 0.1).collect();
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r0, scale, &a0, &b0, k, n)],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject");
    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();

    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let base = reference_base(m, k, n, &x, &w);

    for r in [2usize, 5usize] {
        let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.07 - 0.2).collect();
        let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.03 + 0.15).collect();
        let a_t = Tensor::from_f32(&[k, r], &a).unwrap();
        let b_t = Tensor::from_f32(&[r, n], &b).unwrap();
        let outputs = exec
            .run(&[
                ("x", &x_t),
                ("lora.adapter.layers.0.q_proj.A_t", &a_t),
                ("lora.adapter.layers.0.q_proj.B_t", &b_t),
            ])
            .unwrap();
        let delta = reference_delta(m, k, r, n, scale, &x, &a, &b);
        let expected: Vec<f32> = base.iter().zip(&delta).map(|(a, b)| a + b).collect();
        let got = outputs[0].to_vec_f32();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "r={r}: got {got:?} expected {expected:?}");
        }
    }
}

// ===========================================================================
// Fused-QKV numerics (execution): three deltas land on the correct slices.
// ===========================================================================

/// A tiny fused geometry: num_heads=2, kv_num_heads=1, head_dim=2 =>
/// fused_n = (2 + 2*1) * 2 = 8, with q[0:4] k[4:6] v[6:8].
fn tiny_fused_group(node_id: NodeId) -> FusedGroup {
    FusedGroup {
        node_id,
        fused_n: 8,
        slices: [
            (QkvRole::Q, 0, 4),
            (QkvRole::K, 4, 2),
            (QkvRole::V, 6, 2),
        ],
    }
}

fn fused_entry(
    node_id: NodeId,
    base: ValueId,
    x: ValueId,
    k: usize,
    role: QkvRole,
    width: usize,
    group: &FusedGroup,
) -> TargetEntry {
    TargetEntry {
        semantic: format!("layers.0.{}", role.as_str()),
        node_id,
        base_output: base,
        activation: x,
        k,
        n: width,
        dtype: F32,
        placement: Placement::FusedSlice { group: group.clone(), role },
    }
}

// All three Q/K/V deltas fed: each lands on its own output slice and none leaks
// into another slice's columns.
#[test]
fn fused_qkv_slices_land_correctly() {
    let (m, k) = (2, 3);
    let fused_n = 8;
    let w: Vec<f32> = (0..k * fused_n).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, fused_n, &w);
    let group = tiny_fused_group(node_id);

    // Distinct ranks and scales per role to prove independence.
    let (rq, rk, rv) = (2usize, 3usize, 1usize);
    let (sq, sk, sv) = (0.5f32, 1.0f32, 2.0f32);
    let aq: Vec<f32> = (0..k * rq).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let bq: Vec<f32> = (0..rq * 4).map(|i| (i as f32) * 0.05).collect();
    let ak: Vec<f32> = (0..k * rk).map(|i| (i as f32) * -0.1 + 0.3).collect();
    let bk: Vec<f32> = (0..rk * 2).map(|i| (i as f32) * 0.07).collect();
    let av: Vec<f32> = (0..k * rv).map(|i| (i as f32) * 0.2).collect();
    let bv: Vec<f32> = (0..rv * 2).map(|i| (i as f32) * -0.1 + 0.4).collect();

    let manifest = LoraManifest {
        entries: vec![
            fused_entry(node_id, base_v, x_v, k, QkvRole::Q, 4, &group),
            fused_entry(node_id, base_v, x_v, k, QkvRole::K, 2, &group),
            fused_entry(node_id, base_v, x_v, k, QkvRole::V, 2, &group),
        ],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![
            f32_module("self_attn.q_proj", rq, sq, &aq, &bq, k, 4),
            f32_module("self_attn.k_proj", rk, sk, &ak, &bk, k, 2),
            f32_module("self_attn.v_proj", rv, sv, &av, &bv, k, 2),
        ],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject fused");
    // Three roles => three A_t/B_t pairs.
    assert_eq!(overrides.len(), 6);

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();

    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let outputs = exec
        .run(&[
            ("x", &x_t),
            ("lora.adapter.layers.0.q_proj.A_t", &Tensor::from_f32(&[k, rq], &aq).unwrap()),
            ("lora.adapter.layers.0.q_proj.B_t", &Tensor::from_f32(&[rq, 4], &bq).unwrap()),
            ("lora.adapter.layers.0.k_proj.A_t", &Tensor::from_f32(&[k, rk], &ak).unwrap()),
            ("lora.adapter.layers.0.k_proj.B_t", &Tensor::from_f32(&[rk, 2], &bk).unwrap()),
            ("lora.adapter.layers.0.v_proj.A_t", &Tensor::from_f32(&[k, rv], &av).unwrap()),
            ("lora.adapter.layers.0.v_proj.B_t", &Tensor::from_f32(&[rv, 2], &bv).unwrap()),
        ])
        .unwrap();

    let base = reference_base(m, k, fused_n, &x, &w);
    let dq = reference_delta(m, k, rq, 4, sq, &x, &aq, &bq);
    let dk = reference_delta(m, k, rk, 2, sk, &x, &ak, &bk);
    let dv = reference_delta(m, k, rv, 2, sv, &x, &av, &bv);

    let mut expected = base.clone();
    for i in 0..m {
        for j in 0..4 {
            expected[i * fused_n + 0 + j] += dq[i * 4 + j];
        }
        for j in 0..2 {
            expected[i * fused_n + 4 + j] += dk[i * 2 + j];
        }
        for j in 0..2 {
            expected[i * fused_n + 6 + j] += dv[i * 2 + j];
        }
    }
    let got = outputs[0].to_vec_f32();
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-3, "got {got:?} expected {expected:?}");
    }
}

// Partial fused adapter (only q targeted): the untargeted K/V slices default to
// zero-rank no-op hooks, so those columns equal the base exactly while q's
// columns carry the delta.
#[test]
fn fused_qkv_partial_leaves_untargeted_slices_base() {
    let (m, k) = (2, 3);
    let fused_n = 8;
    let w: Vec<f32> = (0..k * fused_n).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, fused_n, &w);
    let group = tiny_fused_group(node_id);

    let rq = 2usize;
    let sq = 0.75f32;
    let aq: Vec<f32> = (0..k * rq).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let bq: Vec<f32> = (0..rq * 4).map(|i| (i as f32) * 0.05).collect();

    let manifest = LoraManifest {
        entries: vec![fused_entry(node_id, base_v, x_v, k, QkvRole::Q, 4, &group)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", rq, sq, &aq, &bq, k, 4)],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject partial fused");
    // Even untargeted K/V slices get override hooks (a later adapter can feed).
    assert_eq!(overrides.len(), 6);

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();

    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    // Feed only q; K/V stay at their zero-rank defaults.
    let outputs = exec
        .run(&[
            ("x", &x_t),
            ("lora.adapter.layers.0.q_proj.A_t", &Tensor::from_f32(&[k, rq], &aq).unwrap()),
            ("lora.adapter.layers.0.q_proj.B_t", &Tensor::from_f32(&[rq, 4], &bq).unwrap()),
        ])
        .unwrap();

    let base = reference_base(m, k, fused_n, &x, &w);
    let dq = reference_delta(m, k, rq, 4, sq, &x, &aq, &bq);
    let got = outputs[0].to_vec_f32();
    for i in 0..m {
        for j in 0..4 {
            let e = base[i * fused_n + j] + dq[i * 4 + j];
            assert!((got[i * fused_n + j] - e).abs() < 1e-3, "q col mismatch");
        }
        for j in 4..8 {
            let e = base[i * fused_n + j];
            assert!(
                (got[i * fused_n + j] - e).abs() < 1e-6,
                "untargeted col {j} must equal base"
            );
        }
    }
}

// ===========================================================================
// Optimizer-survival regression (design §E "must survive optimization").
// ===========================================================================

// The injected A/B override inputs and every LoRA node (2 MatMul, Mul, Add)
// must survive constant folding + dead-node elimination at both Basic and All
// levels, the output must stay wired, and fed numerics must be unchanged.
#[test]
fn injected_branch_survives_optimization() {
    for level in [crate::OptimizationLevel::Basic, crate::OptimizationLevel::All] {
        let (m, k, n, r) = (2, 4, 3, 2);
        let scale = 0.5f32;
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0];
        let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
        let manifest = LoraManifest {
            entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
        };
        let adapter = LoraAdapterSpec {
            name: "adapter".into(),
            modules: vec![f32_module("self_attn.q_proj", r, scale, &a, &b, k, n)],
        };
        let overrides = inject(&mut graph, &manifest, &adapter).expect("inject");

        // Base MatMul + 2 branch MatMuls = 3; one Mul; one Add.
        assert_eq!(count_op(&graph, "MatMul"), 3);
        assert_eq!(count_op(&graph, "Mul"), 1);
        assert_eq!(count_op(&graph, "Add"), 1);

        crate::optimize_graph(&mut graph, level).expect("optimize");

        // Every injected structure survives.
        assert_eq!(count_op(&graph, "MatMul"), 3, "{level:?}: branch MatMuls dropped");
        assert_eq!(count_op(&graph, "Mul"), 1, "{level:?}: scale Mul dropped");
        assert_eq!(count_op(&graph, "Add"), 1, "{level:?}: delta Add dropped");
        assert!(
            has_input_named(&graph, "lora.adapter.layers.0.q_proj.A_t"),
            "{level:?}: A_t override input dropped"
        );
        assert!(
            has_input_named(&graph, "lora.adapter.layers.0.q_proj.B_t"),
            "{level:?}: B_t override input dropped"
        );

        // Fed numerics are unchanged after optimization.
        let mut exec = Executor::build_with_overrides(
            graph,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
            &overrides,
        )
        .unwrap();
        let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
        let a_t = Tensor::from_f32(&[k, r], &a).unwrap();
        let b_t = Tensor::from_f32(&[r, n], &b).unwrap();
        let outputs = exec
            .run(&[
                ("x", &x_t),
                ("lora.adapter.layers.0.q_proj.A_t", &a_t),
                ("lora.adapter.layers.0.q_proj.B_t", &b_t),
            ])
            .unwrap();
        let base = reference_base(m, k, n, &x, &w);
        let delta = reference_delta(m, k, r, n, scale, &x, &a, &b);
        let expected: Vec<f32> = base.iter().zip(&delta).map(|(a, b)| a + b).collect();
        let got = outputs[0].to_vec_f32();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{level:?}: got {got:?} expected {expected:?}");
        }

        // And unfed still collapses to base-only after optimization.
        let outputs = exec.run(&[("x", &x_t)]).unwrap();
        assert_eq!(outputs[0].to_vec_f32(), base, "{level:?}: unfed must be base-only");
    }
}

// ===========================================================================
// P2c — grouped injection (GroupedLoraDelta) single-adapter parity (§J.5).
// ===========================================================================

/// Build an `Int32` routing tensor.
fn i32_segments(len: usize, data: &[i32]) -> Tensor {
    assert_eq!(data.len(), len);
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::from_raw(DataType::Int32, vec![len], &bytes).unwrap()
}

/// Run the Phase-1 4-node direct injection and return the output row-major f32.
fn run_phase1_direct(
    m: usize,
    k: usize,
    n: usize,
    r: usize,
    scale: f32,
    w: &[f32],
    x: &[f32],
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale, a, b, k, n)],
    };
    let overrides = inject(&mut graph, &manifest, &adapter).expect("inject");
    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &overrides,
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], x).unwrap();
    let a_t = Tensor::from_f32(&[k, r], a).unwrap();
    let b_t = Tensor::from_f32(&[r, n], b).unwrap();
    exec.run(&[
        ("x", &x_t),
        ("lora.adapter.layers.0.q_proj.A_t", &a_t),
        ("lora.adapter.layers.0.q_proj.B_t", &b_t),
    ])
    .unwrap()[0]
        .to_vec_f32()
}

// §E golden parity: the grouped GroupedLoraDelta op, run as a pool-of-one with
// every row routed to adapter 0, is bit-parity with the Phase-1 4-node subgraph.
#[test]
fn grouped_injection_drop_removes_registry_entry() {
    let (m, k, n, r) = (2, 4, 3, 2);
    let w = vec![0.0f32; k * n];
    let a = vec![0.0f32; k * r];
    let b = vec![0.0f32; r * n];
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, 1.0, &a, &b, k, n)],
    };
    let pool_id = {
        let injection = inject_grouped(&mut graph, &manifest, &adapter).expect("inject_grouped");
        assert!(LoraPoolRegistry::global().get(injection.pool_id).is_some());
        injection.pool_id
    };
    assert!(LoraPoolRegistry::global().get(pool_id).is_none());
}

#[test]
fn grouped_single_adapter_matches_phase1_subgraph() {
    let (m, k, n, r) = (2, 4, 3, 2);
    let scale = 0.5f32;
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0];
    let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();

    let phase1 = run_phase1_direct(m, k, n, r, scale, &w, &x, &a, &b);

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale, &a, &b, k, n)],
    };
    let injection = inject_grouped(&mut graph, &manifest, &adapter).expect("inject_grouped");
    let pool_id = injection.pool_id;

    // One custom op, one Add, one shared segments input, no A_t/B_t inputs.
    assert_eq!(count_op(&graph, "GroupedLoraDelta"), 1);
    assert_eq!(count_op(&graph, "Add"), 1);
    assert_eq!(count_op(&graph, "MatMul"), 1, "only the base MatMul remains");
    assert!(has_input_named(&graph, "lora.segments"));
    assert!(!has_input_named(&graph, "lora.adapter.layers.0.q_proj.A_t"));

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &HashSet::new(),
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let seg = i32_segments(m, &vec![0i32; m]);
    let grouped = exec
        .run(&[("x", &x_t), ("lora.segments", &seg)])
        .unwrap()[0]
        .to_vec_f32();

    LoraPoolRegistry::global().unregister(pool_id);

    assert_eq!(phase1.len(), grouped.len());
    for (p, g) in phase1.iter().zip(&grouped) {
        assert!(
            (p - g).abs() < 1e-6,
            "grouped {grouped:?} must match Phase-1 {phase1:?}"
        );
    }
}

// A null route (negative segment id) collapses the grouped delta to base-only,
// the batched analogue of the Phase-1 unfed no-op.
#[test]
fn grouped_null_route_is_base_only() {
    let (m, k, n, r) = (2, 4, 3, 2);
    let scale = 0.5f32;
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0, 3.0, 1.0, 0.0, -2.0];
    let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale, &a, &b, k, n)],
    };
    let injection = inject_grouped(&mut graph, &manifest, &adapter).expect("inject_grouped");
    let pool_id = injection.pool_id;

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &HashSet::new(),
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let seg = i32_segments(m, &vec![-1i32; m]);
    let got = exec
        .run(&[("x", &x_t), ("lora.segments", &seg)])
        .unwrap()[0]
        .to_vec_f32();

    LoraPoolRegistry::global().unregister(pool_id);

    let base = reference_base(m, k, n, &x, &w);
    for (g, e) in got.iter().zip(&base) {
        assert!((g - e).abs() < 1e-6, "null route must be base-only");
    }
}

// Multi-adapter grouped injection (design §J.3/§J.4): two adapters live in one
// pool, and the per-row `segments` routing input selects, per token, which
// adapter's delta the shared GroupedLoraDelta op applies. This is the
// end-to-end proof of per-request adapter selection at the injection layer:
// rows routed to adapter A get A's delta, rows routed to adapter B get B's, and
// a base-only row (-1) gets no delta — all in a single mixed batch.
#[test]
fn grouped_two_adapters_route_per_row() {
    let (k, n, r) = (4usize, 3usize, 2usize);
    let (scale_a, scale_b) = (0.5f32, 1.25f32);
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    // Adapter A and adapter B factors differ, so their deltas are distinct.
    let a_a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b_a: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();
    let a_b: Vec<f32> = (0..k * r).map(|i| (i as f32) * -0.2 + 0.4).collect();
    let b_b: Vec<f32> = (0..r * n).map(|i| (i as f32) * 0.15 - 0.1).collect();

    // Four rows: row 0 -> adapter A, row 1 -> adapter B, row 2 -> base (-1),
    // row 3 -> adapter A again (proves grouping is per-row, not contiguous).
    let m = 4usize;
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.13 - 1.0).collect();

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter_a = LoraAdapterSpec {
        name: "alpha".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale_a, &a_a, &b_a, k, n)],
    };
    let adapter_b = LoraAdapterSpec {
        name: "beta".into(),
        modules: vec![f32_module("self_attn.q_proj", r, scale_b, &a_b, &b_b, k, n)],
    };
    let injection = inject_grouped_multi(
        &mut graph,
        &manifest,
        &[(AdapterId(0), &adapter_a), (AdapterId(1), &adapter_b)],
        None,
    )
    .expect("inject_grouped_multi");
    let pool_id = injection.pool_id;

    // Both adapters are name-addressable through the injection's map.
    assert_eq!(
        injection.adapters,
        vec![
            ("alpha".to_string(), AdapterId(0)),
            ("beta".to_string(), AdapterId(1)),
        ]
    );
    assert_eq!(count_op(&graph, "GroupedLoraDelta"), 1);
    assert!(has_input_named(&graph, "lora.segments"));

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &HashSet::new(),
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    // Row routing: adapter A (0), adapter B (1), base (-1), adapter A (0).
    let seg = i32_segments(m, &[0, 1, -1, 0]);
    let got = exec
        .run(&[("x", &x_t), ("lora.segments", &seg)])
        .unwrap()[0]
        .to_vec_f32();

    LoraPoolRegistry::global().unregister(pool_id);

    // Reference: base + the per-row adapter delta (or base only for row 2).
    let base = reference_base(m, k, n, &x, &w);
    let delta_a = reference_delta(m, k, r, n, scale_a, &x, &a_a, &b_a);
    let delta_b = reference_delta(m, k, r, n, scale_b, &x, &a_b, &b_b);
    let mut expected = base.clone();
    for j in 0..n {
        expected[0 * n + j] += delta_a[0 * n + j]; // row 0 -> A
        expected[1 * n + j] += delta_b[1 * n + j]; // row 1 -> B
        // row 2 -> base only (no delta)
        expected[3 * n + j] += delta_a[3 * n + j]; // row 3 -> A
    }
    assert_eq!(got.len(), expected.len());
    for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "row {}/col — grouped multi-adapter got {got:?} expected {expected:?}",
            idx / n
        );
    }
    // Adapter A and B deltas must actually differ, or the test proves nothing.
    assert!(
        (0..n).any(|j| (delta_a[j] - delta_b[j]).abs() > 1e-4),
        "adapter A and B deltas must be distinct for a meaningful selection test"
    );
}

// A divergent module set across adapters is a fail-loud injection error.
#[test]
fn grouped_multi_adapter_module_mismatch_fails_loud() {
    let (k, n, r) = (4usize, 3usize, 2usize);
    let w = vec![0.0f32; k * n];
    let a = vec![0.0f32; k * r];
    let b = vec![0.0f32; r * n];
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, 2, k, n, &w);
    let manifest = LoraManifest {
        entries: vec![direct_entry(node_id, base_v, x_v, k, n)],
    };
    let adapter_a = LoraAdapterSpec {
        name: "alpha".into(),
        modules: vec![f32_module("self_attn.q_proj", r, 1.0, &a, &b, k, n)],
    };
    let adapter_b = LoraAdapterSpec {
        name: "beta".into(),
        modules: vec![f32_module("self_attn.k_proj", r, 1.0, &a, &b, k, n)],
    };
    let error = inject_grouped_multi(
        &mut graph,
        &manifest,
        &[(AdapterId(0), &adapter_a), (AdapterId(1), &adapter_b)],
        None,
    );
    assert!(matches!(
        error,
        Err(LoraInjectError::AdapterModuleSetMismatch { .. })
    ));
}

// Grouped fused-QKV: three GroupedLoraDelta ops share one Concat + Add (§J's
// simpler option). Full-adapter numerics match the per-slice reference.
#[test]
fn grouped_fused_qkv_matches_reference() {
    let (m, k) = (2, 3);
    let fused_n = 8;
    let w: Vec<f32> = (0..k * fused_n).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, fused_n, &w);
    let group = tiny_fused_group(node_id);

    let (rq, rk, rv) = (2usize, 3usize, 1usize);
    let (sq, sk, sv) = (0.5f32, 1.0f32, 2.0f32);
    let aq: Vec<f32> = (0..k * rq).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let bq: Vec<f32> = (0..rq * 4).map(|i| (i as f32) * 0.05).collect();
    let ak: Vec<f32> = (0..k * rk).map(|i| (i as f32) * -0.1 + 0.3).collect();
    let bk: Vec<f32> = (0..rk * 2).map(|i| (i as f32) * 0.07).collect();
    let av: Vec<f32> = (0..k * rv).map(|i| (i as f32) * 0.2).collect();
    let bv: Vec<f32> = (0..rv * 2).map(|i| (i as f32) * -0.1 + 0.4).collect();

    let manifest = LoraManifest {
        entries: vec![
            fused_entry(node_id, base_v, x_v, k, QkvRole::Q, 4, &group),
            fused_entry(node_id, base_v, x_v, k, QkvRole::K, 2, &group),
            fused_entry(node_id, base_v, x_v, k, QkvRole::V, 2, &group),
        ],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![
            f32_module("self_attn.q_proj", rq, sq, &aq, &bq, k, 4),
            f32_module("self_attn.k_proj", rk, sk, &ak, &bk, k, 2),
            f32_module("self_attn.v_proj", rv, sv, &av, &bv, k, 2),
        ],
    };
    let injection = inject_grouped(&mut graph, &manifest, &adapter).expect("inject_grouped fused");
    let pool_id = injection.pool_id;

    // Three ops, one Concat, one Add.
    assert_eq!(count_op(&graph, "GroupedLoraDelta"), 3);
    assert_eq!(count_op(&graph, "Concat"), 1);
    assert_eq!(count_op(&graph, "Add"), 1);

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &HashSet::new(),
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let seg = i32_segments(m, &vec![0i32; m]);
    let got = exec
        .run(&[("x", &x_t), ("lora.segments", &seg)])
        .unwrap()[0]
        .to_vec_f32();

    LoraPoolRegistry::global().unregister(pool_id);

    let base = reference_base(m, k, fused_n, &x, &w);
    let dq = reference_delta(m, k, rq, 4, sq, &x, &aq, &bq);
    let dk = reference_delta(m, k, rk, 2, sk, &x, &ak, &bk);
    let dv = reference_delta(m, k, rv, 2, sv, &x, &av, &bv);
    let mut expected = base.clone();
    for i in 0..m {
        for j in 0..4 {
            expected[i * fused_n + j] += dq[i * 4 + j];
        }
        for j in 0..2 {
            expected[i * fused_n + 4 + j] += dk[i * 2 + j];
        }
        for j in 0..2 {
            expected[i * fused_n + 6 + j] += dv[i * 2 + j];
        }
    }
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-4, "grouped fused got {got:?} expected {expected:?}");
    }
}

// Grouped partial fused-QKV: only q targeted; the untargeted K/V slices admit
// zero-rank pages and contribute nothing, so those columns equal the base.
#[test]
fn grouped_fused_qkv_partial_is_zero_on_untargeted() {
    let (m, k) = (2, 3);
    let fused_n = 8;
    let w: Vec<f32> = (0..k * fused_n).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5];

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let (node_id, x_v, base_v) = add_base_matmul(&mut graph, m, k, fused_n, &w);
    let group = tiny_fused_group(node_id);

    let (rq, sq) = (2usize, 0.5f32);
    let aq: Vec<f32> = (0..k * rq).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let bq: Vec<f32> = (0..rq * 4).map(|i| (i as f32) * 0.05).collect();

    let manifest = LoraManifest {
        entries: vec![fused_entry(node_id, base_v, x_v, k, QkvRole::Q, 4, &group)],
    };
    let adapter = LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", rq, sq, &aq, &bq, k, 4)],
    };
    let injection = inject_grouped(&mut graph, &manifest, &adapter).expect("inject_grouped partial");
    let pool_id = injection.pool_id;

    // Still three ops (q real, k/v zero-rank), one Concat, one Add.
    assert_eq!(count_op(&graph, "GroupedLoraDelta"), 3);

    let mut exec = Executor::build_with_overrides(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
        &HashSet::new(),
    )
    .unwrap();
    let x_t = Tensor::from_f32(&[m, k], &x).unwrap();
    let seg = i32_segments(m, &vec![0i32; m]);
    let got = exec
        .run(&[("x", &x_t), ("lora.segments", &seg)])
        .unwrap()[0]
        .to_vec_f32();

    LoraPoolRegistry::global().unregister(pool_id);

    let base = reference_base(m, k, fused_n, &x, &w);
    let dq = reference_delta(m, k, rq, 4, sq, &x, &aq, &bq);
    let mut expected = base.clone();
    for i in 0..m {
        for j in 0..4 {
            expected[i * fused_n + j] += dq[i * 4 + j];
        }
    }
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-4, "partial fused mismatch");
    }
    // K/V columns are exactly the base (zero-rank contributes nothing).
    for i in 0..m {
        for j in 4..8 {
            assert_eq!(got[i * fused_n + j], base[i * fused_n + j]);
        }
    }
}

// ===========================================================================
// Integration — declared metadata manifest drives grouped injection (§J + §C).
// ===========================================================================

/// A single-target declared manifest for a Direct (non-fused) projection: the
/// authoritative analogue of the graph-discovery path for the split layout.
fn declared_direct_q(node_name: &str, k: usize, n: usize) -> LoraTargetManifest {
    LoraTargetManifest {
        targets: vec![LoraTargetDescriptor {
            module_name: "self_attn.q_proj".to_string(),
            layer_index: 0,
            node_name: node_name.to_string(),
            output_name: format!("{node_name}.out"),
            k,
            n,
            slices: BTreeMap::new(),
            rank: None,
            alpha: None,
        }],
    }
}

/// Build a fresh split-Qwen3-style discovery graph with one targetable q_proj
/// `MatMulNBits` node, resolvable both by graph discovery and by an explicit
/// declared manifest.
fn split_q_proj_graph(node_name: &str, k: usize, n: usize) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = add_activation(&mut graph, k);
    add_matmulnbits(
        &mut graph,
        node_name,
        "model.layers.0.attn.q_proj.MatMulNBits.qweight",
        x,
        k,
        n,
    );
    add_gqa(&mut graph, "/model/layers.0/attn/GroupQueryAttention", 16, 8);
    graph
}

fn q_proj_adapter(k: usize, n: usize) -> LoraAdapterSpec {
    let rank = 4usize;
    let a: Vec<f32> = (0..k * rank).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let b: Vec<f32> = (0..rank * n).map(|i| (i as f32) * -0.02 + 0.05).collect();
    LoraAdapterSpec {
        name: "adapter".into(),
        modules: vec![f32_module("self_attn.q_proj", rank, 0.75, &a, &b, k, n)],
    }
}

/// The critical composition test: the declared-metadata manifest (authoritative)
/// is the SOURCE that feeds Edgemar's grouped `GroupedLoraDelta` emission. Both
/// resolution orders — declared-metadata-manifest (validated against the graph)
/// and graph-derived discovery — must resolve to the SAME manifest and drive the
/// SAME grouped op emission. This proves the two paths are not parallel/dead:
/// `inject_grouped_lora_adapter` threads `declared` straight into `build_manifest`
/// whose single result drives `inject_grouped`.
#[test]
fn grouped_injection_declared_manifest_matches_graph_discovery() {
    let node_name = "/model/layers.0/attn/q_proj/MatMulNBits";
    let (k, n) = (2048usize, 2048usize);

    let mut discovered_graph = split_q_proj_graph(node_name, k, n);
    let discovered = inject_grouped_lora_adapter(&mut discovered_graph, &q_proj_adapter(k, n), None)
        .expect("graph-discovery grouped injection");

    let declaration = declared_direct_q(node_name, k, n);
    let mut declared_graph = split_q_proj_graph(node_name, k, n);
    let declared = inject_grouped_lora_adapter(
        &mut declared_graph,
        &q_proj_adapter(k, n),
        Some(&declaration),
    )
    .expect("declared-manifest grouped injection");

    // 1. The resolved manifest that drives grouped emission is identical.
    assert_eq!(
        declared.manifest.entries.len(),
        discovered.manifest.entries.len(),
        "declared and discovered manifests must have the same arity"
    );
    for (d, g) in declared
        .manifest
        .entries
        .iter()
        .zip(&discovered.manifest.entries)
    {
        assert_eq!(d.semantic, g.semantic);
        assert_eq!(d.node_id, g.node_id);
        assert_eq!(d.base_output, g.base_output);
        assert_eq!(d.activation, g.activation);
        assert_eq!(d.k, g.k);
        assert_eq!(d.n, g.n);
        assert_eq!(d.dtype, g.dtype);
        assert_eq!(d.placement, g.placement);
    }

    // 2. The grouped emission the manifest drove is structurally identical.
    assert_eq!(
        count_op(&declared_graph, "GroupedLoraDelta"),
        count_op(&discovered_graph, "GroupedLoraDelta"),
        "declared path must emit the same GroupedLoraDelta ops as discovery"
    );
    assert_eq!(count_op(&declared_graph, "GroupedLoraDelta"), 1);
    assert!(has_input_named(&declared_graph, "lora.segments"));
    assert_eq!(declared.segments_input, discovered.segments_input);

    LoraPoolRegistry::global().unregister(discovered.pool_id);
    LoraPoolRegistry::global().unregister(declared.pool_id);
}
