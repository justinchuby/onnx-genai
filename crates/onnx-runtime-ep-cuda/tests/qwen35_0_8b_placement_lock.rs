#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! Whole-graph CUDA **placement** regression lock for the Foundry
//! **Qwen3.5-0.8B hybrid** split model (issue #67 / #384).
//!
//! This is the *composition* proof for the per-op CUDA coverage work landed for
//! the Qwen3.5 hybrid recurrent path — `CausalConvWithState` (#480),
//! `LinearAttention` (#484), `GatherBlockQuantized` (#480), and the
//! `com.microsoft` `RotaryEmbedding` / `Bool` `NonZero` declarations (#525). It
//! walks the real `embedding.onnx` + `text.onnx` decode graphs (recursing every
//! control-flow body) and drives each node through the **native CUDA claim
//! gate** (`ExecutionProvider::supports_op`, the same gate the placement pass
//! uses), asserting the whole 1289-node decode graph places on CUDA with
//! **zero declines**.
//!
//! ## Why the claim gate (not an end-to-end decode)
//!
//! `Engine::from_dir` rejects this package (three sibling `.onnx` files, not a
//! single decoder), and `Engine::from_pipeline_dir` currently refuses it during
//! *vision* preprocessing admission (`Resize.attrs.smart_resize=true` is not
//! representable by the runtime's resize spec), so there is no public high-level
//! entry that decodes this split hybrid model end-to-end today. Those are
//! loader/preprocessing-metadata gaps unrelated to CUDA-EP op coverage; see
//! `.squad/decisions/inbox/cohaagen-hybrid-e2e.md`. The claim gate is exactly
//! the surface #67 owns, so this lock guards the coverage guarantee — "every op
//! in the real hybrid decode graph is placeable on CUDA" — directly and cannot
//! silently regress if a covered kernel's claim gate narrows.
//!
//! ```bash
//! QWEN35_0_8B_DIR=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-0.8b-generic-cpu-2/v2 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-runtime-ep-cuda --features cuda \
//!   --test qwen35_0_8b_placement_lock -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use onnx_runtime_ep_api::{ExecutionProvider, KernelMatch};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, TensorLayout};
use onnx_runtime_loader::load_model;

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-0.8b-generic-cpu-2/v2";

#[derive(Default)]
struct Tally {
    total: BTreeMap<String, usize>,
    declined: BTreeMap<String, (usize, String)>,
}

impl Tally {
    fn count(&self, op: &str) -> usize {
        self.total.get(op).copied().unwrap_or(0)
    }
    fn declines(&self, op: &str) -> usize {
        self.declined.get(op).map(|(c, _)| *c).unwrap_or(0)
    }
    fn total_nodes(&self) -> usize {
        self.total.values().sum()
    }
    fn total_declines(&self) -> usize {
        self.declined.values().map(|(c, _)| *c).sum()
    }
}

fn op_key(node: &Node) -> String {
    if node.domain.is_empty() {
        node.op_type.clone()
    } else {
        format!("{}::{}", node.domain, node.op_type)
    }
}

/// Build the per-input `(shapes, dtypes, layouts)` the claim gate inspects,
/// mirroring the placement pass's `node_capability_inputs`.
fn capability_inputs(
    graph: &Graph,
    node: &Node,
) -> (Vec<Vec<Dim>>, Vec<DataType>, Vec<TensorLayout>) {
    let shapes = node
        .inputs
        .iter()
        .map(|i| i.map(|v| graph.value(v).shape.clone()).unwrap_or_default())
        .collect();
    let dtypes = node
        .inputs
        .iter()
        .map(|i| {
            i.map(|v| graph.value(v).dtype)
                .unwrap_or(DataType::Undefined)
        })
        .collect();
    let layouts = node
        .inputs
        .iter()
        .map(|i| {
            i.map(|v| graph.value(v).layout.clone())
                .unwrap_or_else(TensorLayout::contiguous)
        })
        .collect();
    (shapes, dtypes, layouts)
}

fn walk(graph: &Graph, ep: &CudaExecutionProvider, tally: &mut Tally) {
    for (_, node) in graph.nodes.iter() {
        let key = op_key(node);
        *tally.total.entry(key.clone()).or_default() += 1;
        let (shapes, dtypes, layouts) = capability_inputs(graph, node);
        let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
        let m = ep.supports_op(node, opset, &shapes, &dtypes, &layouts);
        if !m.is_supported() {
            let reason = match &m {
                KernelMatch::Unsupported { reason } => reason.to_string(),
                _ => "no registered kernel".to_string(),
            };
            let entry = tally.declined.entry(key).or_insert((0, reason.clone()));
            entry.0 += 1;
            entry.1 = reason;
        }
        for attr in node.attributes.values() {
            match attr {
                Attribute::Graph(g) => walk(g, ep, tally),
                Attribute::Graphs(gs) => gs.iter().for_each(|g| walk(g, ep, tally)),
                _ => {}
            }
        }
    }
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN35_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    for file in ["embedding.onnx", "text.onnx"] {
        if !dir.join(file).is_file() {
            eprintln!(
                "skipping qwen3.5-0.8b placement lock: {} is missing {file}",
                dir.display()
            );
            return None;
        }
    }
    Some(dir)
}

/// Walk one graph file and assert the only declines are within the #525 set.
fn tally_file(dir: &std::path::Path, file: &str, ep: &CudaExecutionProvider) -> Tally {
    let graph = load_model(dir.join(file)).unwrap_or_else(|e| panic!("load {file}: {e}"));
    let mut tally = Tally::default();
    walk(&graph, ep, &mut tally);
    eprintln!(
        "{file}: {} nodes, {} declined",
        tally.total_nodes(),
        tally.total_declines()
    );
    for (op, (count, reason)) in &tally.declined {
        eprintln!(
            "  DECLINE {op}: {count}/{}  reason: {reason}",
            tally.count(op)
        );
    }
    assert!(
        tally.declined.is_empty(),
        "{file}: {} node(s) declined CUDA placement ({:?}) — every op in the qwen3.5 hybrid \
         decode graph must claim on CUDA (CausalConvWithState #480, LinearAttention #484, \
         GatherBlockQuantized #480, com.microsoft RotaryEmbedding / Bool NonZero #525)",
        tally.total_declines(),
        tally.declined.keys().collect::<Vec<_>>()
    );
    tally
}

#[test]
#[ignore = "requires the real qwen3.5-0.8b hybrid model via QWEN35_0_8B_DIR (or the default foundry cache path) and a CUDA device"]
fn qwen35_0_8b_hybrid_graph_places_on_cuda() {
    let Some(dir) = model_dir() else {
        panic!(
            "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
        );
    };
    let ep = match CudaExecutionProvider::new(0) {
        Ok(ep) => ep,
        Err(error) => {
            eprintln!("skipping qwen3.5-0.8b placement lock: CUDA unavailable: {error}");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    // embedding.onnx: block-quantized token embedding + Bool NonZero (#525).
    let embedding = tally_file(&dir, "embedding.onnx", &ep);
    assert_eq!(
        embedding.total_nodes(),
        24,
        "embedding.onnx node count drifted"
    );
    assert!(
        embedding.count("com.microsoft::GatherBlockQuantized") >= 1
            && embedding.declines("com.microsoft::GatherBlockQuantized") == 0,
        "GatherBlockQuantized (#480) must claim on CUDA in embedding.onnx"
    );

    // text.onnx: the hybrid recurrent decode graph.
    let text = tally_file(&dir, "text.onnx", &ep);
    assert_eq!(text.total_nodes(), 1265, "text.onnx node count drifted");
    for (op, expected) in [
        ("com.microsoft::CausalConvWithState", 18usize),
        ("com.microsoft::LinearAttention", 18usize),
    ] {
        assert_eq!(
            text.count(op),
            expected,
            "{op} node count drifted in text.onnx"
        );
        assert_eq!(
            text.declines(op),
            0,
            "{op} must claim on CUDA for every node in text.onnx"
        );
    }

    // Composition guarantee: the whole 1289-node decode graph places 100% on
    // CUDA with zero declines (all coverage — #480/#484/#525 — merged).
    let total = embedding.total_nodes() + text.total_nodes();
    let declines = embedding.total_declines() + text.total_declines();
    eprintln!("qwen3.5-0.8b hybrid decode graph: {total} nodes, {declines} declined");
    assert_eq!(total, 1289, "combined decode-graph node count drifted");
    assert_eq!(
        declines, 0,
        "the qwen3.5-0.8b hybrid decode graph must place 100% on CUDA (zero declines)"
    );
}
