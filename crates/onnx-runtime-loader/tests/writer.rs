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
    let ins: Vec<String> = n
        .inputs()
        .iter()
        .map(|s| name_of(&g2, s.unwrap()))
        .collect();
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
    let sidecar = dir.join("m_ctx_p0_Vendor_EP_p1.bin");
    assert!(
        sidecar.exists(),
        "sanitised sidecar written next to ctx model"
    );
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

#[test]
fn multiple_partition_boundaries_keep_documented_order_and_attributes() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("X", DataType::Float32, static_shape([4usize]));
    let z = g.create_named_value("Z", DataType::Float32, static_shape([4usize]));
    g.add_input(x);
    g.add_input(z);

    let a_mid = g.create_named_value("A_mid", DataType::Float32, static_shape([4usize]));
    let a0 = g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(z)], vec![a_mid]));
    let a = g.create_named_value("A", DataType::Float32, static_shape([4usize]));
    let a1 = g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(x), Some(a_mid)],
        vec![a],
    ));
    g.add_output(a);

    let b = g.create_named_value("B", DataType::Float32, static_shape([4usize]));
    let b0 = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]));
    g.add_output(b);

    let c_mid = g.create_named_value("C_mid", DataType::Float32, static_shape([4usize]));
    let c0 = g.insert_node(Node::new(NodeId(0), "Sigmoid", vec![Some(x)], vec![c_mid]));
    let c = g.create_named_value("C", DataType::Float32, static_shape([4usize]));
    g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(c_mid)], vec![c]));
    g.add_output(c);

    // Deliberately reverse partition A's removal list: boundary ordering remains
    // ascending NodeId, independent of covered_nodes order.
    let covered_a = [a1, a0];
    let covered_b = [b0];
    let covered_c = [c0];
    let parts = [
        EpContextPartition {
            source: "EpA",
            ep_sdk_version: "1.2",
            partition_name: "alpha",
            main_context: true,
            blob: b"alpha-blob",
            covered_nodes: &covered_a,
        },
        EpContextPartition {
            source: "EpB",
            ep_sdk_version: "2.3",
            partition_name: "beta",
            main_context: false,
            blob: b"beta-blob",
            covered_nodes: &covered_b,
        },
        EpContextPartition {
            source: "EpC",
            ep_sdk_version: "",
            partition_name: "",
            main_context: true,
            blob: b"gamma-blob",
            covered_nodes: &covered_c,
        },
    ];
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_multi_boundary");
    std::fs::create_dir_all(&dir).unwrap();
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 1,
    };

    let model = Model::new(&g);
    let out =
        onnx_runtime_loader::dump_ep_context(&model, &dir.join("multi.onnx"), &parts, &config)
            .expect("dump");
    let dumped = load_model(&out).expect("reload");
    let nodes: Vec<_> = ep_context_nodes(&dumped).collect();
    assert_eq!(nodes.len(), 3);

    let node = |source| {
        nodes
            .iter()
            .find(|node| node.source == Some(source))
            .expect("EPContext source")
    };
    let a = node("EpA");
    assert_eq!(a.sdk_version, Some("1.2"));
    assert_eq!(a.partition_name, Some("alpha"));
    assert!(a.main_context);
    assert_eq!(a.embed_mode, EmbedMode::Embedded);
    assert_eq!(
        a.inputs()
            .iter()
            .map(|slot| name_of(&dumped, slot.unwrap()))
            .collect::<Vec<_>>(),
        ["Z", "X"]
    );
    assert_eq!(
        a.outputs()
            .iter()
            .map(|&value| name_of(&dumped, value))
            .collect::<Vec<_>>(),
        ["A"]
    );
    assert_eq!(resolve_ep_context(&dir, a).unwrap().bytes(), b"alpha-blob");

    let b = node("EpB");
    assert!(!b.main_context);
    assert_eq!(
        b.inputs()
            .iter()
            .map(|slot| name_of(&dumped, slot.unwrap()))
            .collect::<Vec<_>>(),
        ["A"]
    );
    assert_eq!(
        b.outputs()
            .iter()
            .map(|&value| name_of(&dumped, value))
            .collect::<Vec<_>>(),
        ["B"]
    );

    let c = node("EpC");
    assert_eq!(c.sdk_version, None);
    assert_eq!(c.partition_name, None);
    assert_eq!(
        c.inputs()
            .iter()
            .map(|slot| name_of(&dumped, slot.unwrap()))
            .collect::<Vec<_>>(),
        ["X"]
    );
    assert_eq!(
        c.outputs()
            .iter()
            .map(|&value| name_of(&dumped, value))
            .collect::<Vec<_>>(),
        ["C_mid"]
    );
}

/// Two independent partitions producing `X → Relu → A` and `X → Neg → B` graph
/// outputs; returns the graph plus the two single-node partitions.
fn two_partition_graph() -> (Graph, Vec<NodeId>, Vec<NodeId>) {
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
    (g, vec![n1], vec![n2])
}

/// **B1 regression** — two EXTERNAL-mode partitions whose `source` strings are
/// DISTINCT but SANITISE TO THE SAME component (`Vendor/EP` vs `Vendor_EP`) with
/// identical `partition_name` must NOT alias the same sidecar. The partition
/// index disambiguates the filename, so both `.bin` files exist, are distinct,
/// and each EPContext node reloads ITS OWN blob byte-exact through the consume
/// path. Before the fix both nodes stored `m_ctx_Vendor_EP_p1.bin` and the
/// second write truncated the first — data loss.
#[test]
fn external_dump_colliding_sanitised_sources_do_not_alias() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_external_collide");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("m.onnx");

    // Distinct non-UTF-8 blobs so a byte-exact round-trip is provable.
    let b0: Vec<u8> = vec![0x00, 0x80, 0xFF, 0xC3, 0x28, 0x01];
    let b1: Vec<u8> = vec![0x7F, 0xFE, 0xBE, 0xEF, 0x00, 0x42];
    assert_ne!(b0, b1);

    let (g, p0, p1) = two_partition_graph();
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 0,
    };
    // Both sanitise to `Vendor_EP`; same partition_name `p1`. Only the identity
    // (raw source) differs.
    let parts = [
        EpContextPartition {
            source: "Vendor/EP",
            ep_sdk_version: "",
            partition_name: "p1",
            main_context: true,
            blob: &b0,
            covered_nodes: &p0,
        },
        EpContextPartition {
            source: "Vendor_EP",
            ep_sdk_version: "",
            partition_name: "p1",
            main_context: true,
            blob: &b1,
            covered_nodes: &p1,
        },
    ];

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &parts, &config).expect("dump");

    // Two DISTINCT sidecars written (index-disambiguated), each holding its blob.
    let s0 = dir.join("m_ctx_p0_Vendor_EP_p1.bin");
    let s1 = dir.join("m_ctx_p1_Vendor_EP_p1.bin");
    assert!(s0.exists(), "partition-0 sidecar exists");
    assert!(s1.exists(), "partition-1 sidecar exists");
    assert_ne!(
        s0, s1,
        "filenames are distinct despite identical sanitised parts"
    );
    assert_eq!(std::fs::read(&s0).unwrap(), b0, "sidecar 0 holds blob 0");
    assert_eq!(std::fs::read(&s1).unwrap(), b1, "sidecar 1 holds blob 1");

    // Reload: each EPContext node must resolve to ITS OWN blob byte-exact.
    let g2 = load_model(&out).expect("reload");
    let mut nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 2);
    // `Vendor/EP` ('/'=0x2F) sorts before `Vendor_EP` ('_'=0x5F).
    nodes.sort_by_key(|n| n.source.map(str::to_owned));
    assert_eq!(nodes[0].source, Some("Vendor/EP"));
    assert_eq!(nodes[1].source, Some("Vendor_EP"));
    assert_eq!(nodes[0].embed_mode, EmbedMode::ExternalFile);
    assert_eq!(nodes[1].embed_mode, EmbedMode::ExternalFile);

    let r0 = resolve_ep_context(&dir, &nodes[0]).expect("resolve blob 0");
    let r1 = resolve_ep_context(&dir, &nodes[1]).expect("resolve blob 1");
    assert_eq!(r0.bytes(), &b0[..], "node 0 reloads its own blob");
    assert_eq!(r1.bytes(), &b1[..], "node 1 reloads its own blob");
    assert_ne!(
        r0.bytes(),
        r1.bytes(),
        "no aliasing — distinct blobs preserved"
    );
}

/// **A1 regression** — a disabled config (`enable = false`) must be a no-op: no
/// ctx model and no sidecars are written. The returned path is the *would-be*
/// output location but nothing is created on disk.
#[test]
fn disabled_config_writes_nothing() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_disabled");
    // Start from a clean directory so we can assert emptiness.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("m.onnx");
    let payload = blob();

    let (g, covered) = partition_graph();
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: false,
        file_path: None,
        embed_mode: 0,
    };
    let part = EpContextPartition {
        source: "VendorEP",
        ep_sdk_version: "1.0",
        partition_name: "p0",
        main_context: true,
        blob: &payload,
        covered_nodes: &covered,
    };

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &[part], &config).expect("dump");
    assert_eq!(out, dir.join("m_ctx.onnx"), "returns the would-be path");
    assert!(!out.exists(), "disabled config writes no ctx model");

    // The directory holds nothing (no ctx model, no sidecars).
    let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert!(entries.is_empty(), "disabled config writes no files at all");
}

/// **Deckard re-review regression** — a single EP can legitimately emit MULTIPLE
/// compiled PRIMARY partitions (`main_context=1`) sharing the SAME `source` and
/// the SAME (here empty/omitted) `partition_name`. Such duplicates are NOT an
/// error: the injective per-partition index keeps their sidecars distinct, and
/// the §55.3 consume path loads every `main_context=1` node independently. This
/// asserts both distinct-index sidecars exist and each node reloads ITS OWN blob
/// byte-exact — the case v2's blanket `(source, partition_name)` rejection wrongly
/// blocked.
#[test]
fn duplicate_primary_identity_round_trips_external() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_dup_primary");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("m.onnx");

    // Distinct non-UTF-8 blobs so a byte-exact round-trip is provable.
    let b0: Vec<u8> = vec![0x00, 0x80, 0xFF, 0xC3, 0x28, 0x01];
    let b1: Vec<u8> = vec![0x7F, 0xFE, 0xBE, 0xEF, 0x00, 0x42];
    assert_ne!(b0, b1);

    let (g, p0, p1) = two_partition_graph();
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 0,
    };
    // Same source, same (empty) partition_name, both primary — a legitimate
    // multi-partition EP dump.
    let parts = [
        EpContextPartition {
            source: "EpA",
            ep_sdk_version: "",
            partition_name: "",
            main_context: true,
            blob: &b0,
            covered_nodes: &p0,
        },
        EpContextPartition {
            source: "EpA",
            ep_sdk_version: "",
            partition_name: "",
            main_context: true,
            blob: &b1,
            covered_nodes: &p1,
        },
    ];

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &parts, &config)
        .expect("duplicate primary identities dump without rejection");

    // Two DISTINCT sidecars written, disambiguated purely by partition index.
    let s0 = dir.join("m_ctx_p0_EpA.bin");
    let s1 = dir.join("m_ctx_p1_EpA.bin");
    assert!(s0.exists(), "partition-0 sidecar exists");
    assert!(s1.exists(), "partition-1 sidecar exists");
    assert_ne!(
        s0, s1,
        "distinct p{{index}} filenames despite identical identity"
    );
    assert_eq!(std::fs::read(&s0).unwrap(), b0, "sidecar 0 holds blob 0");
    assert_eq!(std::fs::read(&s1).unwrap(), b1, "sidecar 1 holds blob 1");

    // Reload through the consume path: each primary node resolves to its OWN blob.
    let g2 = load_model(&out).expect("reload");
    let mut nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 2, "both primary nodes present");
    for n in &nodes {
        assert!(n.main_context, "both nodes are primary");
        assert_eq!(n.source, Some("EpA"));
        assert_eq!(n.embed_mode, EmbedMode::ExternalFile);
    }
    // Identify each reloaded node by its boundary output name (partition 0
    // produces "A", partition 1 produces "B") so we can tie each to its own blob.
    nodes.sort_by_key(|n| name_of(&g2, n.outputs()[0]));

    let r0 = resolve_ep_context(&dir, &nodes[0]).expect("resolve blob 0");
    let r1 = resolve_ep_context(&dir, &nodes[1]).expect("resolve blob 1");
    assert_eq!(
        r0.bytes(),
        &b0[..],
        "node 0 reloads its own blob (r0 == b0)"
    );
    assert_eq!(
        r1.bytes(),
        &b1[..],
        "node 1 reloads its own blob (r1 == b1)"
    );
    assert_ne!(
        r0.bytes(),
        r1.bytes(),
        "distinct blobs preserved (r0 != r1)"
    );
}

/// **Gaff advisory B** — hostile `source`/`partition_name` strings (`/`, `\`,
/// `..`, NUL) must never yield a path separator, traversal, or otherwise unsafe
/// sidecar filename. The stored path stays a bare relative filename and still
/// round-trips through the traversal-guarded consume path.
#[test]
fn hostile_source_strings_sanitise_to_safe_bare_filename() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer_hostile");
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
    // Every hostile char (`/`, `\`, `..`, NUL) must be neutralised to `_`.
    let part = EpContextPartition {
        source: "a/b\\c..d\0e",
        ep_sdk_version: "",
        partition_name: "..",
        main_context: true,
        blob: &payload,
        covered_nodes: &covered,
    };

    let out = onnx_runtime_loader::dump_ep_context(&model, &orig, &[part], &config).expect("dump");

    // Exactly one sidecar, and its name is a single safe path component.
    let bins: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".bin"))
        .collect();
    assert_eq!(bins.len(), 1, "one sidecar written");
    let name = &bins[0];
    assert!(!name.contains('/'), "no unix separator: {name}");
    assert!(!name.contains('\\'), "no windows separator: {name}");
    assert!(!name.contains('\0'), "no NUL: {name}");
    assert!(
        !Path::new(name)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir)),
        "no `..` traversal component: {name}"
    );
    assert_eq!(
        Path::new(name).components().count(),
        1,
        "sidecar name is a single bare component: {name}"
    );

    // Still round-trips through the traversal-guarded consume path byte-exact.
    let g2 = load_model(&out).expect("reload");
    let nodes: Vec<_> = ep_context_nodes(&g2).collect();
    assert_eq!(nodes.len(), 1);
    let resolved = resolve_ep_context(&dir, &nodes[0]).expect("resolve blob");
    assert_eq!(resolved.bytes(), &payload[..]);
}
