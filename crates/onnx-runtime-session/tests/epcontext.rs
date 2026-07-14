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
    Attribute, DataType, DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorData,
    TensorLayout, ValueId, static_shape,
};
use onnx_runtime_session::{InferenceSession, SessionError, load_ep_context_nodes};

// ── a pure-Rust mock compiled EP ──────────────────────────────────────────────

/// A test-only compiled EP: declares the `"MOCK"` source key and records every
/// blob handed to [`ExecutionProvider::load_context`] so tests can assert the
/// exact bytes were dispatched (and how many times).
struct MockCompiledEp {
    keys: Vec<String>,
    /// Bytes received by each `load_context` call, in order.
    loaded: Mutex<Vec<Vec<u8>>>,
}

impl MockCompiledEp {
    fn new() -> Self {
        // Keys come from "config", not hardcoded in dispatch logic (§55.6).
        Self {
            keys: vec!["MOCK".to_string()],
            loaded: Mutex::new(Vec::new()),
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

    fn load_context(&self, ctx: &EpContext) -> EpResult<()> {
        // Record the exact bytes we were asked to restore.
        self.loaded.lock().unwrap().push(ctx.data.clone());
        Ok(())
    }
}

// ── IR construction helpers ───────────────────────────────────────────────────

/// A `String`-valued attribute.
fn s_attr(v: &str) -> Attribute {
    Attribute::String(v.to_string())
}

/// An `Int`-valued attribute.
fn i_attr(v: i64) -> Attribute {
    Attribute::Int(v)
}

/// An embedded `ep_cache_context` payload, stored losslessly as a `UINT8`
/// tensor attribute — exactly how the loader's `graph_builder` preserves the
/// opaque binary blob (so arbitrary, non-UTF-8 bytes round-trip).
fn embedded_blob_attr(bytes: &[u8]) -> Attribute {
    Attribute::Tensor(TensorData::from_raw(
        DataType::Uint8,
        vec![bytes.len()],
        bytes.to_vec(),
    ))
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
