//! Session-side `com.microsoft::EPContext` **consume path** tests (§55.3).
//!
//! These exercise the session dispatch entry point
//! [`onnx_runtime_session::load_ep_context_nodes`]: building the `source`-keyed
//! registry from the session's EPs, bypassing placement for pre-compiled nodes,
//! mapping the loader blob → runtime `EpContext`, and driving `main_context=1/0`
//! resolution + payload dedup.
//!
//! Synthetic EPContext graphs are hand-built via the IR API (matching the style
//! of `tests/executor.rs`) — no `onnx.helper`. A test-only `MockCompiledEp`
//! (pure Rust, no GPU) records the bytes handed to `load_context` so the tests
//! can assert exact round-trips and dedup. Temp files use `CARGO_TARGET_TMPDIR`,
//! never `/tmp`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpContext, EpError, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult,
};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Graph, Node, NodeId, Shape,
    TensorLayout, ValueId, static_shape,
};
use onnx_runtime_loader::{
    EpContextDumpConfig, Model, ep_context_nodes as loader_ep_context_nodes, load_model,
};
use onnx_runtime_session::{
    CompiledPartition, InferenceSession, SessionError, dump_session_ep_context,
    load_ep_context_nodes,
};

// ── a pure-Rust mock compiled EP ──────────────────────────────────────────────

/// A test-only compiled EP: declares the `"MOCK"` source key and records every
/// blob handed to [`ExecutionProvider::load_context`] so tests can assert the
/// exact bytes were dispatched (and how many times).
struct MockCompiledEp {
    keys: Vec<String>,
    /// Bytes received by each `load_context` call, in order.
    loaded: Mutex<Vec<Vec<u8>>>,
    /// Compiled blob returned by `save_context` (the §55.4 dump path).
    save_blob: Vec<u8>,
    /// SDK/toolchain version returned by `save_context`.
    save_version: String,
}

impl MockCompiledEp {
    fn new() -> Self {
        // Keys come from "config", not hardcoded in dispatch logic (§55.6).
        Self {
            keys: vec!["MOCK".to_string()],
            loaded: Mutex::new(Vec::new()),
            save_blob: Vec::new(),
            save_version: String::new(),
        }
    }

    /// A compiled EP that "compiles" a partition into `blob` (used by the §55.4
    /// dump round-trip). `save_context` returns exactly these bytes.
    fn compiling(blob: &[u8], version: &str) -> Self {
        Self {
            keys: vec!["MOCK".to_string()],
            loaded: Mutex::new(Vec::new()),
            save_blob: blob.to_vec(),
            save_version: version.to_string(),
        }
    }

    /// Snapshot of the payloads restored so far.
    fn loaded(&self) -> Vec<Vec<u8>> {
        self.loaded.lock().unwrap().clone()
    }
}

impl ExecutionProvider for MockCompiledEp {
    fn name(&self) -> &str {
        "mock_compiled_ep"
    }
    fn device_type(&self) -> DeviceType {
        DeviceType::Custom(0)
    }
    fn device_id(&self) -> DeviceId {
        DeviceId::new(DeviceType::Custom(0), 0)
    }
    fn initialize(&mut self, _config: &EpConfig) -> EpResult<()> {
        Ok(())
    }
    fn shutdown(&mut self) -> EpResult<()> {
        Ok(())
    }
    fn supports_op(&self, _op: &Node, _shapes: &[Shape], _layouts: &[TensorLayout]) -> KernelMatch {
        KernelMatch::Unsupported
    }
    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        Err(EpError::NoEpForOp {
            op_type: "<mock>".to_string(),
        })
    }
    fn allocate(&self, _size: usize, _alignment: usize) -> EpResult<DeviceBuffer> {
        Err(EpError::NotInitialized)
    }
    fn deallocate(&self, _buffer: DeviceBuffer) -> EpResult<()> {
        Ok(())
    }
    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> EpResult<()> {
        Ok(())
    }
    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> EpResult<Fence> {
        Ok(Fence::default())
    }
    fn sync(&self) -> EpResult<()> {
        Ok(())
    }

    // --- EPContext contract (§55) ---

    fn context_source_keys(&self) -> Vec<String> {
        self.keys.clone()
    }

    fn save_context(&self) -> EpResult<EpContext> {
        Ok(EpContext::new(
            self.name(),
            self.save_version.clone(),
            self.save_blob.clone(),
            Vec::new(),
            "mock-device",
        ))
    }

    fn load_context(&self, ctx: &EpContext) -> EpResult<()> {
        // Record the exact bytes we were asked to restore.
        self.loaded.lock().unwrap().push(ctx.data.clone());
        Ok(())
    }
}

// ── IR construction helpers ───────────────────────────────────────────────────

/// A `String`-valued attribute (stored as raw bytes in the IR).
fn s_attr(v: &str) -> Attribute {
    Attribute::String(v.as_bytes().to_vec())
}

/// An `Int`-valued attribute.
fn i_attr(v: i64) -> Attribute {
    Attribute::Int(v)
}

/// An embedded `ep_cache_context` payload, stored as a raw-bytes STRING
/// attribute — exactly how the loader's `graph_builder` now preserves the opaque
/// binary blob generically (so arbitrary, non-UTF-8 bytes round-trip byte-exact
/// with no op-specific handling).
fn embedded_blob_attr(bytes: &[u8]) -> Attribute {
    Attribute::String(bytes.to_vec())
}

/// Insert a `com.microsoft::EPContext` node `X → EPContext → Y_<tag>` with the
/// given attributes and return its [`NodeId`]. Each node gets a distinct output
/// value so multiple EPContext nodes can coexist in one graph.
fn add_epctx_node(
    g: &mut Graph,
    input: ValueId,
    tag: &str,
    attrs: Vec<(&str, Attribute)>,
) -> NodeId {
    let out = g.create_named_value(
        format!("Y_{tag}"),
        DataType::Float32,
        static_shape([2usize, 8]),
    );
    let mut node = Node::new(NodeId(0), "EPContext", vec![Some(input)], vec![out]);
    node.domain = "com.microsoft".to_string();
    for (k, v) in attrs {
        node.attributes.insert(k.to_string(), v);
    }
    let id = g.insert_node(node);
    g.add_output(out);
    id
}

/// A fresh graph with a single `X` input `[2,4]`.
fn graph_with_input() -> (Graph, ValueId) {
    let mut g = Graph::new();
    g.opset_imports
        .insert("com.microsoft".to_string(), 1);
    let x = g.create_named_value("X", DataType::Float32, static_shape([2usize, 4]));
    g.add_input(x);
    (g, x)
}

/// The mock as the session's single registered EP, at `EpId(0)`.
fn eps(mock: &MockCompiledEp) -> [(EpId, &dyn ExecutionProvider); 1] {
    [(EpId(0), mock)]
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Embed round-trip: an `embed_mode=1`, `source="MOCK"` node with an inline
/// blob → dispatch calls the mock's `load_context` with the exact bytes.
#[test]
fn embed_mode_round_trip_dispatches_exact_bytes() {
    // Opaque payload including non-UTF-8 bytes to prove lossless preservation.
    let payload: Vec<u8> = vec![0x00, 0x01, 0x80, 0xFE, 0xFF, b'v', b'1', 0xC3, 0x28];

    let (mut g, x) = graph_with_input();
    add_epctx_node(
        &mut g,
        x,
        "a",
        vec![
            ("embed_mode", i_attr(1)),
            ("main_context", i_attr(1)),
            ("source", s_attr("MOCK")),
            ("ep_sdk_version", s_attr("9.9.9")),
            ("ep_cache_context", embedded_blob_attr(&payload)),
        ],
    );

    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g, Path::new("."), &eps(&mock)).expect("dispatch");

    assert_eq!(placement.handled.len(), 1, "one EPContext node handled");
    assert_eq!(
        mock.loaded(),
        vec![payload],
        "load_context received the exact inline blob bytes"
    );
}

/// External round-trip: `embed_mode=0` whose `ep_cache_context` is a `.bin`
/// path relative to the model dir → the correct file bytes are restored.
#[test]
fn external_mode_round_trip_loads_file_bytes() {
    let dir: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_external");
    std::fs::create_dir_all(&dir).unwrap();
    let payload: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x7F, 0x80];
    let bin = dir.join("ctx_blob.bin");
    std::fs::write(&bin, &payload).unwrap();

    let (mut g, x) = graph_with_input();
    add_epctx_node(
        &mut g,
        x,
        "ext",
        vec![
            ("embed_mode", i_attr(0)),
            ("main_context", i_attr(1)),
            ("source", s_attr("MOCK")),
            // Path is relative to the model dir (§55.3), stored as a string.
            ("ep_cache_context", s_attr("ctx_blob.bin")),
        ],
    );

    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g, &dir, &eps(&mock)).expect("dispatch");

    assert_eq!(placement.handled.len(), 1);
    assert_eq!(
        mock.loaded(),
        vec![payload],
        "load_context received the external file's exact bytes"
    );
}

/// Unclaimed: a node whose `source` matches no registered EP surfaces a clear
/// `NoEpForContext` naming the missing key — no guessing.
#[test]
fn unclaimed_source_surfaces_no_ep_for_context() {
    let (mut g, x) = graph_with_input();
    add_epctx_node(
        &mut g,
        x,
        "qnn",
        vec![
            ("embed_mode", i_attr(1)),
            ("source", s_attr("QNN")),
            ("ep_cache_context", embedded_blob_attr(b"blob")),
        ],
    );

    // The mock only claims "MOCK", so "QNN" is unclaimed.
    let mock = MockCompiledEp::new();
    let err = load_ep_context_nodes(&g, Path::new("."), &eps(&mock)).expect_err("must fail");

    match err {
        SessionError::Ep(EpError::NoEpForContext { source_key }) => {
            assert_eq!(source_key.as_deref(), Some("QNN"));
        }
        other => panic!("expected NoEpForContext {{ source_key: QNN }}, got {other:?}"),
    }
    assert!(mock.loaded().is_empty(), "no context restored on failure");
}

/// `main_context` dedup + reference resolution (§55.3):
/// * two primaries with an identical payload → the bytes are loaded once, and
/// * a `main_context=0` reference resolves against its primary by
///   (`source`, `partition_name`) with **no** second blob load.
#[test]
fn main_context_dedup_and_reference_resolution() {
    let shared: Vec<u8> = b"one-packed-binary-holding-two-graphs".to_vec();

    let (mut g, x) = graph_with_input();
    // Primary A (partition "p0") and primary B (partition "p1") share one packed
    // binary → identical `ep_cache_context` payloads.
    add_epctx_node(
        &mut g,
        x,
        "p0",
        vec![
            ("embed_mode", i_attr(1)),
            ("main_context", i_attr(1)),
            ("source", s_attr("MOCK")),
            ("partition_name", s_attr("p0")),
            ("ep_cache_context", embedded_blob_attr(&shared)),
        ],
    );
    add_epctx_node(
        &mut g,
        x,
        "p1",
        vec![
            ("embed_mode", i_attr(1)),
            ("main_context", i_attr(1)),
            ("source", s_attr("MOCK")),
            ("partition_name", s_attr("p1")),
            ("ep_cache_context", embedded_blob_attr(&shared)),
        ],
    );
    // Reference (main_context=0) into primary "p1" — carries no payload.
    add_epctx_node(
        &mut g,
        x,
        "ref",
        vec![
            ("embed_mode", i_attr(1)),
            ("main_context", i_attr(0)),
            ("source", s_attr("MOCK")),
            ("partition_name", s_attr("p1")),
        ],
    );

    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g, Path::new("."), &eps(&mock)).expect("dispatch");

    // All three nodes bypass placement.
    assert_eq!(placement.handled.len(), 3);
    // Identical payload deduped → loaded exactly once; the reference adds no load.
    assert_eq!(
        mock.loaded(),
        vec![shared],
        "identical payload loaded once; reference resolved without a second load"
    );
}

/// A dangling `main_context=0` reference (no matching primary) is a clear error.
#[test]
fn dangling_reference_is_an_error() {
    let (mut g, x) = graph_with_input();
    // A lone reference into partition "missing" — no primary was ever loaded.
    add_epctx_node(
        &mut g,
        x,
        "ref",
        vec![
            ("main_context", i_attr(0)),
            ("source", s_attr("MOCK")),
            ("partition_name", s_attr("missing")),
        ],
    );

    let mock = MockCompiledEp::new();
    let err = load_ep_context_nodes(&g, Path::new("."), &eps(&mock)).expect_err("must fail");
    match err {
        SessionError::DanglingEpContext {
            source_key,
            partition_name,
        } => {
            assert_eq!(source_key.as_deref(), Some("MOCK"));
            assert_eq!(partition_name.as_deref(), Some("missing"));
        }
        other => panic!("expected DanglingEpContext, got {other:?}"),
    }
    assert!(mock.loaded().is_empty());
}

/// Two EPs declaring the same `source` key is a configuration error propagated
/// from the ep-api registry builder (reject-duplicate-key, §55.6).
#[test]
fn duplicate_source_key_across_eps_is_rejected() {
    let (mut g, x) = graph_with_input();
    add_epctx_node(
        &mut g,
        x,
        "a",
        vec![
            ("source", s_attr("MOCK")),
            ("ep_cache_context", embedded_blob_attr(b"x")),
        ],
    );

    let a = MockCompiledEp::new();
    let b = MockCompiledEp::new();
    let eps: [(EpId, &dyn ExecutionProvider); 2] = [(EpId(0), &a), (EpId(1), &b)];
    let err = load_ep_context_nodes(&g, Path::new("."), &eps).expect_err("must fail");
    assert!(
        matches!(
            err,
            SessionError::Ep(EpError::DuplicateContextSource { .. })
        ),
        "expected DuplicateContextSource, got {err:?}"
    );
}

/// End-to-end through the public `InferenceSession`: a graph carrying an
/// EPContext node for a compiled EP that is **not** loaded (Phase-1 CPU-only
/// session) fails with a clear `NoEpForContext` at build — the session refuses
/// to guess rather than trying to run the pre-compiled node as a CPU kernel.
#[test]
fn session_build_rejects_unclaimed_ep_context_node() {
    let (mut g, x) = graph_with_input();
    add_epctx_node(
        &mut g,
        x,
        "a",
        vec![
            ("embed_mode", i_attr(1)),
            ("source", s_attr("SomeCompiledEP")),
            ("ep_cache_context", embedded_blob_attr(b"compiled-blob")),
        ],
    );

    let err = match InferenceSession::from_graph(g) {
        Ok(_) => panic!("CPU-only session must not claim a compiled-EP context node"),
        Err(e) => e,
    };
    assert!(
        matches!(err, SessionError::Ep(EpError::NoEpForContext { .. })),
        "expected NoEpForContext, got {err:?}"
    );
}

// ── §55.4 DUMP / WRITER round-trip (the money test) ───────────────────────────

/// A real two-`Relu` partition `X → Relu → H → Relu → Y` (no EPContext yet), plus
/// the ids of the two nodes that form the partition the writer will replace.
fn build_partition_graph() -> (Graph, Vec<NodeId>) {
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

/// Names of a node's input value slots (skipped optionals become empty strings).
fn input_names(g: &Graph, id: NodeId) -> Vec<String> {
    g.node(id)
        .inputs
        .iter()
        .map(|slot| match slot {
            Some(v) => g.value(*v).name.clone().unwrap_or_default(),
            None => String::new(),
        })
        .collect()
}

/// Names of a node's output values.
fn output_names(g: &Graph, id: NodeId) -> Vec<String> {
    g.node(id)
        .outputs
        .iter()
        .map(|v| g.value(*v).name.clone().unwrap_or_default())
        .collect()
}

/// A non-UTF-8 compiled blob so the round-trip proves byte-exact preservation.
fn compiled_blob() -> Vec<u8> {
    vec![0x00, 0x01, 0x80, 0xFE, 0xFF, b'c', b'x', 0xC3, 0x28, 0x00, 0x7F]
}

/// embed_mode=1 round-trip: dump → reload the `*_ctx.onnx` through the consume
/// path → the exact (non-UTF-8) blob comes back and the partition boundary
/// `X → EPContext → Y` is preserved.
#[test]
fn dump_embed_round_trip_is_byte_exact() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_dump_embed");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("mymodel.onnx");
    let payload = compiled_blob();

    let (g, covered) = build_partition_graph();
    let ep = MockCompiledEp::compiling(&payload, "7.7.7");
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 1,
    };
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "part0".to_string(),
        covered_nodes: covered,
    }];

    let out = dump_session_ep_context(&model, &orig, &parts, &config).expect("dump");
    assert_eq!(out, dir.join("mymodel_ctx.onnx"), "default <stem>_ctx.onnx path");
    assert!(out.exists(), "context model written");

    // Reload the produced ctx model through the production loader.
    let g2 = load_model(&out).expect("reload ctx model");

    // Exactly one EPContext node replaced the two Relus.
    let ids: Vec<NodeId> = loader_ep_context_nodes(&g2).map(|n| n.node).collect();
    assert_eq!(ids.len(), 1, "the partition collapsed to one EPContext node");
    let ep_id = ids[0];
    assert_eq!(input_names(&g2, ep_id), vec!["X".to_string()], "boundary input");
    assert_eq!(output_names(&g2, ep_id), vec!["Y".to_string()], "boundary output");
    // No ordinary Relu survives — the subgraph was fully replaced.
    assert!(
        g2.nodes.values().all(|n| n.op_type == "EPContext"),
        "only the EPContext node remains"
    );

    // Consume the reloaded model — the mock records the restored bytes.
    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g2, &dir, &eps(&mock)).expect("consume");
    assert_eq!(placement.handled.len(), 1);
    assert_eq!(
        mock.loaded(),
        vec![payload],
        "the embedded blob round-tripped byte-exact"
    );
}

/// embed_mode=0 round-trip: dump writes an external sidecar `.bin` next to the
/// ctx model and stores the relative path; reload + consume reads that file back
/// byte-exact.
#[test]
fn dump_external_round_trip_via_sidecar_bin() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_dump_external");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("net.onnx");
    let payload = compiled_blob();

    let (g, covered) = build_partition_graph();
    let ep = MockCompiledEp::compiling(&payload, "3.1.4");
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: None,
        embed_mode: 0,
    };
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "part0".to_string(),
        covered_nodes: covered,
    }];

    let out = dump_session_ep_context(&model, &orig, &parts, &config).expect("dump");
    assert_eq!(out, dir.join("net_ctx.onnx"));

    // A sidecar `.bin` holding the blob was written next to the ctx model.
    let sidecar = dir.join("net_ctx_p0_MOCK_part0.bin");
    assert!(sidecar.exists(), "external sidecar written next to ctx model");
    assert_eq!(std::fs::read(&sidecar).unwrap(), payload, "sidecar holds the blob");

    // Reload + consume: the external path resolves relative to the model dir.
    let g2 = load_model(&out).expect("reload ctx model");
    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g2, &dir, &eps(&mock)).expect("consume");
    assert_eq!(placement.handled.len(), 1);
    assert_eq!(
        mock.loaded(),
        vec![payload],
        "the external blob round-tripped byte-exact"
    );
}

/// An explicit `file_path` overrides the default `<stem>_ctx.onnx` location.
#[test]
fn dump_honours_explicit_output_path() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_dump_explicit");
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("src.onnx");
    let explicit = dir.join("chosen_name.onnx");
    let payload = compiled_blob();

    let (g, covered) = build_partition_graph();
    let ep = MockCompiledEp::compiling(&payload, "1.0.0");
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: true,
        file_path: Some(explicit.clone()),
        embed_mode: 1,
    };
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "p".to_string(),
        covered_nodes: covered,
    }];

    let out = dump_session_ep_context(&model, &orig, &parts, &config).expect("dump");
    assert_eq!(out, explicit);
    assert!(explicit.exists());
}

/// A1 regression: a disabled config (`enable = false`) is a no-op at the session
/// driver level too — no ctx model is written and no EP `save_context` side
/// effect occurs. The returned path is the would-be location but nothing exists.
#[test]
fn dump_disabled_config_is_a_no_op() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_dump_disabled");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orig = dir.join("mymodel.onnx");
    let payload = compiled_blob();

    let (g, covered) = build_partition_graph();
    let ep = MockCompiledEp::compiling(&payload, "7.7.7");
    let model = Model::new(&g);
    let config = EpContextDumpConfig {
        enable: false,
        file_path: None,
        embed_mode: 0,
    };
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "part0".to_string(),
        covered_nodes: covered,
    }];

    let out = dump_session_ep_context(&model, &orig, &parts, &config).expect("dump");
    assert_eq!(out, dir.join("mymodel_ctx.onnx"), "returns the would-be path");
    assert!(!out.exists(), "disabled config writes no ctx model");
    let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert!(entries.is_empty(), "disabled config writes no files at all");
}

// ── §21.4 / §55.5 end-to-end through the public builder ───────────────────────

/// End-to-end via the public `SessionBuilder`: the `ep.context_*` options are
/// parsed into the session's dump config, and `export_ep_context` — driven by a
/// **mock compiling EP** (Phase-1 CPU EP has no compile step) — writes a
/// `*_ctx.onnx` at the configured `ep.context_file_path` whose embedded,
/// non-UTF-8 blob reloads through the consume path byte-exact.
#[test]
fn builder_options_drive_export_byte_exact() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_builder_e2e");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write the original (two-Relu) model to disk.
    let orig = dir.join("orig.onnx");
    let (g, _covered_in_g) = build_partition_graph();
    let bytes = onnx_runtime_loader::encode_model(&Model::new(&g)).expect("encode");
    std::fs::write(&orig, &bytes).unwrap();

    // Configured output path for the context model.
    let ctx_out = dir.join("explicit_ctx.onnx");

    // Build the session through the public builder, setting the ep.context_*
    // options as string key/values (exactly the C-API path forwards).
    let session = InferenceSession::builder()
        .model(&orig)
        .option("ep.context_enable", "1")
        .option("ep.context_file_path", ctx_out.to_str().unwrap())
        .option("ep.context_embed_mode", "1")
        .build()
        .expect("build session");

    // The options parsed into the session's dump config.
    let cfg = session.ep_context_config();
    assert!(cfg.enable);
    assert_eq!(cfg.file_path.as_deref(), Some(ctx_out.as_path()));
    assert_eq!(cfg.embed_mode, 1);

    // Identify the compiled partition's nodes from the session's own graph
    // (the compiler-integration seam). Here: both Relus.
    let covered: Vec<NodeId> = session
        .graph()
        .nodes
        .iter()
        .filter(|(_, n)| n.op_type == "Relu")
        .map(|(id, _)| id)
        .collect();
    assert_eq!(covered.len(), 2, "two Relus form the partition");

    // A mock EP that "compiled" the partition into a known non-UTF-8 blob.
    let payload = compiled_blob();
    let ep = MockCompiledEp::compiling(&payload, "5.5.5");
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "part0".to_string(),
        covered_nodes: covered,
    }];

    // Export honours ep.context_enable + ep.context_file_path.
    let out = session.export_ep_context(&orig, &parts).expect("export");
    assert_eq!(out, ctx_out, "export writes to the configured file_path");
    assert!(out.exists(), "context model written");

    // Reload + consume: the embedded blob round-trips byte-exact.
    let g2 = load_model(&out).expect("reload ctx model");
    let ids: Vec<NodeId> = loader_ep_context_nodes(&g2).map(|n| n.node).collect();
    assert_eq!(ids.len(), 1, "the partition collapsed to one EPContext node");

    let mock = MockCompiledEp::new();
    let placement = load_ep_context_nodes(&g2, &dir, &eps(&mock)).expect("consume");
    assert_eq!(placement.handled.len(), 1);
    assert_eq!(
        mock.loaded(),
        vec![payload],
        "the exported blob round-tripped byte-exact through the builder path"
    );
}

/// A session built with `ep.context_enable=0` (the default) writes **no** file
/// when `export_ep_context` is called — the disabled config is a no-op even
/// though a mock compiling EP is supplied.
#[test]
fn builder_disabled_export_writes_nothing() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_builder_disabled");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let orig = dir.join("orig.onnx");
    let (g, _c) = build_partition_graph();
    let bytes = onnx_runtime_loader::encode_model(&Model::new(&g)).expect("encode");
    std::fs::write(&orig, &bytes).unwrap();

    // No ep.context_* options → disabled by default.
    let session = InferenceSession::builder()
        .model(&orig)
        .build()
        .expect("build session");
    assert!(!session.ep_context_config().enable);

    let covered: Vec<NodeId> = session
        .graph()
        .nodes
        .iter()
        .filter(|(_, n)| n.op_type == "Relu")
        .map(|(id, _)| id)
        .collect();
    let ep = MockCompiledEp::compiling(&compiled_blob(), "0.0.0");
    let parts = [CompiledPartition {
        ep: &ep,
        partition_name: "p".to_string(),
        covered_nodes: covered,
    }];

    let out = session.export_ep_context(&orig, &parts).expect("export no-op");
    assert_eq!(out, dir.join("orig_ctx.onnx"), "returns the would-be path");
    assert!(!out.exists(), "disabled config writes no ctx model");

    // Only the original model exists in the dir.
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec!["orig.onnx".to_string()], "no extra files written");
}
