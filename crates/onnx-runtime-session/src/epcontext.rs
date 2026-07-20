//! Session-side `com.microsoft::EPContext` **consume (load) path** (§55.3 /
//! §55.6 / §55.7).
//!
//! This module owns the part of the EPContext contract the session is
//! responsible for (§55.7): **bypassing placement** for pre-compiled EPContext
//! nodes and **driving `main_context=1/0` resolution + dedup**. It bridges the
//! loader's on-disk view (`EpContextNode` / `EpContextBlob`) to the ep-api
//! runtime form (`EpContext`) and hands each node to its owning EP via
//! [`ExecutionProvider::load_context`].
//!
//! ## Dispatch is a pure source-key lookup (hard rule, §55.6)
//!
//! EP selection is **always** by the node's `source` attribute, resolved through
//! the [`EpContextRegistry`] the session builds from its registered EPs
//! ([`build_ep_context_registry`]). There are **no hardcoded EP/vendor names**
//! here — an EP participates iff it declares `source` keys via
//! [`ExecutionProvider::context_source_keys`]. A node whose `source` matches no
//! registered EP surfaces a clear [`EpError::NoEpForContext`] rather than a
//! guess (the model needs an EP that is not loaded).
//!
//! ## `main_context` primary / reference resolution (§55.3)
//!
//! Some EPs pack multiple compiled graphs into one primary context binary.
//! Nodes with `main_context=1` **own** the payload; nodes with `main_context=0`
//! **reference** a sibling primary's already-loaded context, matched by
//! (`source`, `partition_name`). The session therefore:
//!
//! 1. loads every `main_context=1` blob **first**, deduplicating identical
//!    `ep_cache_context` payloads so the same bytes are never handed to
//!    `load_context` twice, then
//! 2. resolves each `main_context=0` node against an already-loaded primary —
//!    **no second blob load**. A reference with no matching primary is a clear
//!    error rather than a silent skip.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use onnx_runtime_ep_api::{EpContext, EpError, EpId, ExecutionProvider, build_ep_context_registry};
use onnx_runtime_ir::{Graph, NodeId};
use onnx_runtime_loader::{
    EpContextDumpConfig, EpContextNode, EpContextPartition, Model, dump_ep_context,
    ep_context_nodes, resolve_ep_context,
};

use crate::error::{Result, SessionError};

/// Outcome of the EPContext consume pass over a graph (§55.3).
///
/// [`handled`](Self::handled) lists the EPContext nodes that bypassed placement
/// and were restored (or resolved as a reference) through their owning EP. The
/// executor skips exactly these nodes — they are pre-compiled and must never be
/// run as ordinary kernels.
#[derive(Clone, Debug, Default)]
pub struct EpContextPlacement {
    /// EPContext node ids that were dispatched to an EP (bypassing placement).
    pub handled: Vec<NodeId>,
}

impl EpContextPlacement {
    /// Whether the graph contained (and this pass handled) any EPContext nodes.
    pub fn is_empty(&self) -> bool {
        self.handled.is_empty()
    }
}

/// Identity of a loaded primary context, used to resolve `main_context=0`
/// references (§55.3). Owned `String`s so the map outlives the borrowed graph.
type PrimaryKey = (Option<String>, Option<String>); // (source, partition_name)

/// Consume every `com.microsoft::EPContext` node in `graph` (§55.3).
///
/// Builds the `source`-keyed [`EpContextRegistry`] from `eps`
/// ([`build_ep_context_registry`] — propagating
/// [`EpError::DuplicateContextSource`] if two EPs claim one key), then for each
/// EPContext node claims the owning EP by its `source` attribute, maps the
/// loader blob + node attributes into the runtime [`EpContext`], and calls
/// [`ExecutionProvider::load_context`]. `main_context=1` primaries load first
/// (deduplicating identical payloads); `main_context=0` references resolve
/// against an already-loaded primary by (`source`, `partition_name`) with no
/// second blob load.
///
/// `model_dir` is the directory of the ONNX model file, needed to resolve
/// `embed_mode=0` external blob paths (identical policy to external weights,
/// §19.2). Returns the set of handled node ids so the executor can bypass them.
///
/// # Errors
///
/// * [`EpError::DuplicateContextSource`] — two EPs declare the same `source` key.
/// * [`EpError::NoEpForContext`] — a node's `source` matches no registered EP.
/// * [`SessionError::DanglingEpContext`] — a `main_context=0` reference has no
///   matching primary.
/// * loader errors from [`resolve_ep_context`] (missing payload, bad external
///   path), and any EP error from [`ExecutionProvider::load_context`].
pub fn load_ep_context_nodes(
    graph: &Graph,
    model_dir: &Path,
    eps: &[(EpId, &dyn ExecutionProvider)],
) -> Result<EpContextPlacement> {
    let nodes: Vec<EpContextNode<'_>> = ep_context_nodes(graph).collect();
    if nodes.is_empty() {
        return Ok(EpContextPlacement::default());
    }

    // Model-agnostic dispatch table: keys come only from each EP's
    // `context_source_keys()`; no vendor name is ever hardcoded (§55.6).
    let registry = build_ep_context_registry(eps.iter().copied())?;
    let ep_by_id: HashMap<EpId, &dyn ExecutionProvider> = eps.iter().copied().collect();

    let mut handled = Vec::with_capacity(nodes.len());
    // Primaries loaded so far, for reference resolution (§55.3).
    let mut primaries: HashSet<PrimaryKey> = HashSet::new();
    // Payloads already handed to `load_context`, to avoid loading identical
    // bytes twice (§55.3 dedup). Keyed by (source, bytes).
    let mut loaded_payloads: HashSet<(Option<String>, Vec<u8>)> = HashSet::new();

    // Phase 1 — primaries (`main_context=1`) load their blobs first.
    for node in nodes.iter().filter(|n| n.main_context) {
        let ep = claim_ep(&registry, &ep_by_id, node)?;

        let blob = resolve_ep_context(model_dir, node)?;
        let dedup_key = (node.source.map(str::to_owned), blob.bytes().to_vec());
        if loaded_payloads.insert(dedup_key) {
            // First time we see these exact bytes for this source → restore.
            let ctx = EpContext {
                ep_name: ep.name().to_string(),
                // `ep_sdk_version` attr → runtime `ep_version` (diagnostics).
                ep_version: node.sdk_version.unwrap_or_default().to_string(),
                // Opaque `ep_cache_context` payload.
                data: blob.bytes().to_vec(),
                // This node's boundary == the partition it replaces.
                covered_nodes: vec![node.node],
                // Filled/validated by the EP.
                device_fingerprint: String::new(),
            };
            ep.load_context(&ctx)?;
        }

        primaries.insert((
            node.source.map(str::to_owned),
            node.partition_name.map(str::to_owned),
        ));
        handled.push(node.node);
    }

    // Phase 2 — references (`main_context=0`) resolve against a loaded primary;
    // no second blob load.
    for node in nodes.iter().filter(|n| !n.main_context) {
        // Still require an EP for this source (model-agnostic; no guessing).
        claim_ep(&registry, &ep_by_id, node)?;

        let key = (
            node.source.map(str::to_owned),
            node.partition_name.map(str::to_owned),
        );
        if !primaries.contains(&key) {
            return Err(SessionError::DanglingEpContext {
                source_key: node.source.map(str::to_owned),
                partition_name: node.partition_name.map(str::to_owned),
            });
        }
        handled.push(node.node);
    }

    Ok(EpContextPlacement { handled })
}

/// Resolve the EP that owns `node` by its `source` key (§55.6). An unclaimed
/// node is a clear [`EpError::NoEpForContext`] naming the missing source, never
/// a guess.
fn claim_ep<'e>(
    registry: &onnx_runtime_ep_api::EpContextRegistry,
    ep_by_id: &HashMap<EpId, &'e dyn ExecutionProvider>,
    node: &EpContextNode<'_>,
) -> Result<&'e dyn ExecutionProvider> {
    let ep_id = registry
        .claim(node.source)
        .ok_or_else(|| EpError::NoEpForContext {
            source_key: node.source.map(str::to_owned),
        })?;
    // The registry only ever holds ids from `eps`, so this lookup is total; a
    // miss would be an internal inconsistency.
    ep_by_id.get(&ep_id).copied().ok_or_else(|| {
        SessionError::Internal(format!(
            "EPContext registry returned unknown Ep id {ep_id:?} for source {:?}",
            node.source
        ))
    })
}

// ── EPContext DUMP / WRITER path (§55.4) ──────────────────────────────────────

/// One compiled partition the session hands to the EPContext writer (§55.4).
///
/// The session owns compilation and partition boundaries; it names the owning
/// EP and the graph nodes that EP compiled. The driver ([`dump_session_ep_context`])
/// pulls the compiled blob + SDK version from the EP via
/// [`ExecutionProvider::save_context`] and the `source` key from
/// [`ExecutionProvider::context_source_keys`] — nothing is hardcoded (§55.6).
pub struct CompiledPartition<'a> {
    /// The EP that compiled this partition (and owns its context).
    pub ep: &'a dyn ExecutionProvider,
    /// The ORT-partition name emitted as the node's `partition_name` attribute.
    pub partition_name: String,
    /// The graph nodes this partition covers (replaced by one EPContext node).
    pub covered_nodes: Vec<NodeId>,
}

/// Drive the §55.4 dump path: serialise `model` to a `*_ctx.onnx` context-cache
/// model, replacing each compiled `partition`'s subgraph with a single
/// `com.microsoft::EPContext` node.
///
/// For each partition the driver calls [`ExecutionProvider::save_context`] to
/// obtain the runtime context (blob + `ep_version`) and takes the EP's first
/// declared [`ExecutionProvider::context_source_keys`] as the node's `source`
/// key (§55.6 — model-agnostic; no vendor name is hardcoded here). It then hands
/// the neutral partition views to the loader-owned writer
/// ([`onnx_runtime_loader::dump_ep_context`]). Returns the written model path.
///
/// If [`EpContextDumpConfig::enable`] is `false` this is a no-op: it does **not**
/// call any EP's `save_context`, writes no files, and returns the path it *would*
/// have written to — so a disabled config has no side effects by construction.
///
/// # Errors
///
/// * [`EpError::UnsupportedContext`] — an EP with no compile step (its
///   `save_context` default).
/// * [`SessionError::Internal`] — an EP that participates in compilation but
///   declares no `source` key (it cannot be dispatched on reload).
/// * loader errors from the writer (I/O, encoding).
pub fn dump_session_ep_context(
    model: &Model,
    orig_path: &Path,
    partitions: &[CompiledPartition],
    config: &EpContextDumpConfig,
) -> Result<PathBuf> {
    // Honour `ep.context_enable` before any side effect (no `save_context`, no
    // I/O): a disabled config is a no-op. Delegate the would-be path to the
    // loader so both drivers agree on the default `<stem>_ctx.onnx` location.
    if !config.enable {
        return Ok(dump_ep_context(model, orig_path, &[], config)?);
    }

    // Materialise each EP's runtime context first so the loader partition views
    // can borrow its bytes / version for the duration of the dump.
    struct Owned {
        source: String,
        ctx: EpContext,
        partition_name: String,
        covered_nodes: Vec<NodeId>,
    }
    let mut owned = Vec::with_capacity(partitions.len());
    for part in partitions {
        let ctx = part.ep.save_context()?;
        let source = part
            .ep
            .context_source_keys()
            .into_iter()
            .next()
            .ok_or_else(|| {
                SessionError::Internal(format!(
                    "EP {:?} produced a context but declares no `source` key — the \
                 EPContext node could not be dispatched on reload (§55.6)",
                    part.ep.name()
                ))
            })?;
        owned.push(Owned {
            source,
            ctx,
            partition_name: part.partition_name.clone(),
            covered_nodes: part.covered_nodes.clone(),
        });
    }

    let loader_parts: Vec<EpContextPartition<'_>> = owned
        .iter()
        .map(|o| EpContextPartition {
            source: &o.source,
            ep_sdk_version: &o.ctx.ep_version,
            partition_name: &o.partition_name,
            // The session emits primaries; multi-graph `main_context=0`
            // referencing is an EP-packing concern handled on the load side.
            main_context: true,
            blob: &o.ctx.data,
            covered_nodes: &o.covered_nodes,
        })
        .collect();

    Ok(dump_ep_context(model, orig_path, &loader_parts, config)?)
}
