//! `com.microsoft::EPContext` **dump / writer path** tests (§55.4).
//!
//! These exercise the loader-owned writer [`onnx_runtime_loader::dump_ep_context`]
//! standalone — no `onnx-runtime-ep-api` / session — via the model-agnostic
//! [`EpContextPartition`] view. They assert the produced `*_ctx.onnx` round-trips
//! back through the load path byte-exact (embed **and** external sidecar), that
//! the compiled subgraph is replaced by a single boundary-wired EPContext node,
//! and that the output path / sidecar naming follow §55.4. Temp files use
//! `CARGO_TARGET_TMPDIR`, never `/tmp`.

use std::path::Path;

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, ValueId, static_shape};
use onnx_runtime_loader::{
    EmbedMode, EpContextDumpConfig, EpContextPartition, Model, ep_context_nodes, load_model,
    resolve_ep_context,
};

/// `X → Relu → H → Relu → Y`; returns the graph and the two Relu node ids.
fn partition_graph() -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("X", DataType::Float32, static_shape([2usize, 4]));
    g.add_input(x);
    let h = g.create_named_value("H", DataType::Float32, static_shape([2usize, 4]));
    let id1 = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![h]));
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2usize, 4]));
    let id2 = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(h)], vec![y]));
    g.add_output(y);
    (g, vec![id1, id2])
}

fn name_of(g: &Graph, v: ValueId) -> String {
    g.value(v).name.clone().unwrap_or_default()
}

fn blob() -> Vec<u8> {
    // Deliberately non-UTF-8 to prove byte-exact preservation.
    vec![0x00, 0x80, 0xFF, 0xC3, 0x28, b'z', 0x01, 0x7F]
}

#[test]
fn embed_dump_replaces_subgraph_and_round_trips() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_embed");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("m.onnx");
    let payload = blob();

    let (g, covered) = partition_graph();
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 1,
    };
    let part = EpContextPartition {
        source: "VendorEP",
        ep_sdk_version: "9.9.9",
        partition_name: "part0",
        main_context: true,
        blob: &payload,
        covered_nodes: &covered,
    };

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &[part], &config).expect("dump");
    assert_eq!(out, dir.join("m_ctx.onnx"));

    let g2 = load_model(&out).expect("reload");
    let nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 1);
    let n = &nodes[0];
    assert_eq!(n.source, Some("VendorEP"));
    assert_eq!(n.sdk_version, Some("9.9.9"));
    assert_eq!(n.partition_name, Some("part0"));
    assert!(n.main_context);
    assert_eq!(n.embed_mode, EmbedMode::Embedded);
    // Boundary wiring: X in, Y out.
    let ins: Vec<String> = n.inputs().iter().map(|s| name_of(&g2, s.unwrap())).collect();
    let outs: Vec<String> = n.outputs().iter().map(|v| name_of(&g2, *v)).collect();
    assert_eq!(ins, vec!["X".to_string()]);
    assert_eq!(outs, vec!["Y".to_string()]);

    // Blob round-trips byte-exact.
    let resolved = resolve_ep_context(&dir, n).expect("resolve blob");
    assert_eq!(resolved.bytes(), &payload[..]);
}

#[test]
fn external_dump_writes_sidecar_and_round_trips() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_external");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("m.onnx");
    let payload = blob();

    let (g, covered) = partition_graph();
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 0,
    };
    // A `source` with a path-unsafe char to prove filename sanitisation.
    let part = EpContextPartition {
        source: "Vendor/EP",
        ep_sdk_version: "",
        partition_name: "p1",
        main_context: true,
        blob: &payload,
        covered_nodes: &covered,
    };

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &[part], &config).expect("dump");
    let sidecar = dir.join("m_ctx_Vendor_EP_p1.bin");
    assert!(sidecar.exists(), "sanitised sidecar written next to ctx model");
    assert_eq!(std::fs::read(&sidecar).unwrap(), payload);

    let g2 = load_model(&out).expect("reload");
    let nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 1);
    let n = &nodes[0];
    assert_eq!(n.embed_mode, EmbedMode::ExternalFile);
    // Empty ep_sdk_version omitted.
    assert_eq!(n.sdk_version, None);

    // External path resolves relative to the model dir → byte-exact blob.
    let resolved = resolve_ep_context(&dir, n).expect("resolve external blob");
    assert_eq!(resolved.bytes(), &payload[..]);
}

#[test]
fn multiple_partitions_each_become_one_node() {
    // Two independent partitions in one graph: X→Relu→A (out) and X→Neg→B (out).
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("X", DataType::Float32, static_shape([4usize]));
    g.add_input(x);
    let a = g.create_named_value("A", DataType::Float32, static_shape([4usize]));
    let n1 = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![a]));
    let b = g.create_named_value("B", DataType::Float32, static_shape([4usize]));
    let n2 = g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(x)], vec![b]));
    g.add_output(a);
    g.add_output(b);

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_multi");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("multi.onnx");
    let p0 = [n1];
    let p1 = [n2];
    let b0 = b"aaa";
    let b1 = b"bbbb";
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 1,
    };
    let parts = [
        EpContextPartition {
            source: "EpA",
            ep_sdk_version: "1",
            partition_name: "p0",
            main_context: true,
            blob: b0,
            covered_nodes: &p0,
        },
        EpContextPartition {
            source: "EpB",
            ep_sdk_version: "2",
            partition_name: "p1",
            main_context: true,
            blob: b1,
            covered_nodes: &p1,
        },
    ];

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &parts, &config).expect("dump");
    let g2 = load_model(&out).expect("reload");
    let mut nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 2);
    nodes.sort_by_key(|n| n.source.map(str::to_owned));
    let blobs: Vec<Vec<u8>> = nodes
        .iter()
        .map(|n| resolve_ep_context(&dir, n).unwrap().bytes().to_vec())
        .collect();
    assert_eq!(nodes[0].source, Some("EpA"));
    assert_eq!(blobs[0], b0);
    assert_eq!(nodes[1].source, Some("EpB"));
    assert_eq!(blobs[1], b1);
}
