//! `com.microsoft::EPContext` **dump / writer path** (§55.4) — the inverse of
//! the load path in [`crate::epcontext`].
//!
//! After the session partitions a graph and an EP compiles a claimed subgraph,
//! it asks that EP for its compiled context ([`save_context`]) and hands the
//! result here. This module:
//!
//! 1. builds one `com.microsoft::EPContext` [`Node`] per compiled partition,
//!    carrying the §55.2 attributes (`source`, `ep_sdk_version`,
//!    `partition_name`, `main_context`, `embed_mode`, `ep_cache_context`);
//! 2. handles `embed_mode`: inline the compiled blob into the `ep_cache_context`
//!    STRING attribute (byte-exact, via the byte-preserving IR string attr), or
//!    write it to an external sidecar `.bin` next to the output model and store
//!    the **relative** path (mirroring the §19.2 external-data convention so the
//!    produced model round-trips through the §55.3 load path);
//! 3. **replaces** each partition's nodes with its single EPContext node, wiring
//!    the partition's boundary inputs/outputs to the node's variadic i/o; and
//! 4. serialises the resulting model to `<orig_stem>_ctx.onnx` (or an explicit
//!    output path) via the [`crate::encoder`] seam.
//!
//! ## Model-agnostic (§55.6 HARD RULE)
//!
//! Nothing here hardcodes a vendor/op/model name. The EPContext op-type/domain
//! come from the shared constants in [`crate::epcontext`]; every `source` key,
//! SDK version, partition name, and blob comes from the caller (ultimately from
//! the EP via its trait). Adding a new compiled EP needs **no** change here.
//!
//! [`save_context`]: https://docs.rs/onnx-runtime-ep-api
//! [`Node`]: onnx_runtime_ir::Node

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, ValueId};

use crate::encoder::{encode_model, Model};
use crate::epcontext::{attr, EP_CONTEXT_OP, MS_DOMAIN};
use crate::LoaderError;

/// Configuration for the EPContext dump path (§55.4).
///
/// A follow-up wires the `SessionBuilder` / C-API options
/// `ep.context_enable` / `ep.context_file_path` / `ep.context_embed_mode` to
/// populate this struct, so the field names/types match those options exactly.
/// It is directly constructible (no option-string parsing) so it can be built
/// and tested standalone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpContextDumpConfig {
    /// `ep.context_enable` — whether to dump a context-cache model at all. When
    /// `false`, [`dump_ep_context`] is a no-op: it writes no sidecars and no ctx
    /// model, so a disabled config is safe by construction.
    pub enable: bool,
    /// `ep.context_file_path` — the output model path. `None` defaults to
    /// `<orig_stem>_ctx.onnx` beside the source model.
    pub file_path: Option<PathBuf>,
    /// `ep.context_embed_mode` — `1` embeds the blob inline (default), `0`
    /// writes it to an external sidecar `.bin` and stores the relative path.
    pub embed_mode: u8,
}

impl Default for EpContextDumpConfig {
    fn default() -> Self {
        Self {
            enable: false,
            file_path: None,
            // §55.2 default: embed the payload inline.
            embed_mode: 1,
        }
    }
}

impl EpContextDumpConfig {
    /// Whether `embed_mode` selects the external-file form (`0`). Every other
    /// value (including the §55.2 default `1`) embeds inline — the same
    /// fail-closed decode the load path uses ([`crate::EmbedMode`]).
    fn is_external(&self) -> bool {
        self.embed_mode == 0
    }
}

/// One EP-compiled partition to serialise into an `EPContext` node (§55.4).
///
/// This is the **model-agnostic** input to the writer: the loader does not
/// depend on `onnx-runtime-ep-api`, so the session maps its runtime `EpContext`
/// (+ the EP's `context_source_keys`) into this neutral view. Every field is
/// supplied by the caller — no vendor name is baked in.
#[derive(Clone, Copy, Debug)]
pub struct EpContextPartition<'a> {
    /// `source` attribute — the EP's own dispatch key (§55.6). Never hardcoded.
    pub source: &'a str,
    /// `ep_sdk_version` attribute — the SDK/toolchain version (from the runtime
    /// context's `ep_version`). Empty ⇒ the attribute is omitted.
    pub ep_sdk_version: &'a str,
    /// `partition_name` attribute — the ORT-partitioned graph name. Empty ⇒ the
    /// attribute is omitted.
    pub partition_name: &'a str,
    /// `main_context` attribute — `true` for a primary node owning the payload.
    pub main_context: bool,
    /// The compiled vendor blob (the runtime context's `data`) → the node's
    /// `ep_cache_context` (inline or sidecar, per `embed_mode`).
    pub blob: &'a [u8],
    /// The partition's nodes (the runtime context's `covered_nodes`) that this
    /// single EPContext node replaces. Their boundary tensors become the node's
    /// variadic i/o.
    pub covered_nodes: &'a [NodeId],
}

/// Dump `model` to a `*_ctx.onnx` context-cache model, replacing each compiled
/// `partition`'s subgraph with a single `com.microsoft::EPContext` node (§55.4).
///
/// Returns the path the context model was written to.
///
/// * The output path is `config.file_path` if set, else `<orig_stem>_ctx.onnx`
///   beside `orig_path`.
/// * For `embed_mode = 0` each partition's blob is written to a sidecar
///   `<ctx_stem>_p{index}_<source>_<partition>.bin` **next to the output model**
///   and the node stores that filename as a **relative** path (resolved back
///   relative to the model dir by the §55.3 load path). The partition **index**
///   makes the filename injective: two partitions with different identities can
///   never alias the same file even when their sanitised components collide.
///
/// # Errors
///
/// Beyond the loader I/O / encoding errors, this returns
/// [`LoaderError::EpContext`] if two partitions share the **same** `source`
/// **and** `partition_name` (an ambiguous request), or if a partition has no
/// covered nodes.
///
/// If [`EpContextDumpConfig::enable`] is `false` this writes **nothing** (no
/// sidecars, no ctx model) and returns the path it *would* have written to, so a
/// disabled config is a no-op by construction.
pub fn dump_ep_context(
    model: &Model,
    orig_path: &Path,
    partitions: &[EpContextPartition],
    config: &EpContextDumpConfig,
) -> Result<PathBuf, LoaderError> {
    let out_path = resolve_output_path(orig_path, config);

    // Honour `ep.context_enable`: a disabled config produces no files and no ctx
    // model. Return early *before* any side effect so the dump is safe by
    // construction regardless of how the caller gates.
    if !config.enable {
        return Ok(out_path);
    }

    // Reject an ambiguous request: the same (source, partition_name) identity
    // appearing twice cannot be disambiguated on reload (both nodes would carry
    // identical dispatch keys), so fail loudly rather than silently alias.
    let mut identities: HashSet<(&str, &str)> = HashSet::with_capacity(partitions.len());
    for part in partitions {
        if !identities.insert((part.source, part.partition_name)) {
            return Err(LoaderError::EpContext(format!(
                "EPContext dump: duplicate partition identity (source {:?}, \
                 partition_name {:?}) — distinct partitions must have distinct \
                 (source, partition_name) keys",
                part.source, part.partition_name
            )));
        }
    }

    let out_dir = out_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    // Mutate a clone so the caller's graph is untouched.
    let mut graph = model.graph.clone();

    // EPContext lives in the `com.microsoft` opset; ensure the produced model
    // declares it so it is a valid ONNX model and round-trips through ORT.
    graph
        .opset_imports
        .entry(MS_DOMAIN.to_string())
        .or_insert(1);

    for (index, part) in partitions.iter().enumerate() {
        splice_partition(&mut graph, &out_dir, &out_path, config, index, part)?;
    }

    let out_model = Model {
        graph: &graph,
        metadata: model.metadata.clone(),
        weights: model.weights,
    };
    let bytes = encode_model(&out_model)?;
    std::fs::write(&out_path, bytes).map_err(|source| LoaderError::Io {
        path: out_path.clone(),
        source,
    })?;
    Ok(out_path)
}

/// Replace one partition's nodes with a single EPContext node wired to the
/// partition's boundary i/o.
fn splice_partition(
    graph: &mut Graph,
    out_dir: &Path,
    out_path: &Path,
    config: &EpContextDumpConfig,
    index: usize,
    part: &EpContextPartition,
) -> Result<(), LoaderError> {
    let covered: HashSet<NodeId> = part.covered_nodes.iter().copied().collect();
    if covered.is_empty() {
        return Err(LoaderError::EpContext(
            "EPContext dump: partition has no covered nodes".to_string(),
        ));
    }
    let (inputs, outputs) = partition_boundary(graph, &covered);

    // Build the ep_cache_context payload (inline bytes or a relative sidecar
    // path), writing the external `.bin` next to the output model when needed.
    let ep_cache_context = if config.is_external() {
        let rel = sidecar_filename(out_path, index, part.source, part.partition_name);
        let sidecar = out_dir.join(&rel);
        std::fs::write(&sidecar, part.blob).map_err(|source| LoaderError::Io {
            path: sidecar,
            source,
        })?;
        Attribute::String(rel.into_bytes())
    } else {
        Attribute::String(part.blob.to_vec())
    };

    // Remove the compiled subgraph. Interior values (consumed only within the
    // partition) are GC'd; boundary values survive (graph I/O or consumed
    // outside), their producer cleared — the new node re-produces them.
    for &nid in part.covered_nodes {
        graph.remove_node(nid);
    }

    let mut node = Node::new(NodeId(0), EP_CONTEXT_OP, inputs, outputs);
    node.domain = MS_DOMAIN.to_string();
    let attrs = &mut node.attributes;
    attrs.insert(
        attr::MAIN_CONTEXT.to_string(),
        Attribute::Int(part.main_context as i64),
    );
    attrs.insert(
        attr::EMBED_MODE.to_string(),
        Attribute::Int(if config.is_external() { 0 } else { 1 }),
    );
    attrs.insert(
        attr::SOURCE.to_string(),
        Attribute::String(part.source.as_bytes().to_vec()),
    );
    if !part.ep_sdk_version.is_empty() {
        attrs.insert(
            attr::EP_SDK_VERSION.to_string(),
            Attribute::String(part.ep_sdk_version.as_bytes().to_vec()),
        );
    }
    if !part.partition_name.is_empty() {
        attrs.insert(
            attr::PARTITION_NAME.to_string(),
            Attribute::String(part.partition_name.as_bytes().to_vec()),
        );
    }
    attrs.insert(attr::EP_CACHE_CONTEXT.to_string(), ep_cache_context);

    graph.insert_node(node);
    Ok(())
}

/// Compute a partition's boundary tensors (§55.4): the variadic inputs/outputs
/// the replacement EPContext node must carry.
///
/// * **Inputs** — values a covered node consumes that are produced *outside* the
///   partition (or are graph inputs / initializers / sources). Positional order
///   and skipped-optional (`None`) slots are preserved.
/// * **Outputs** — values a covered node produces that are graph outputs *or*
///   consumed by a node outside the partition.
///
/// Both are deduped preserving first-seen order, iterating covered nodes in
/// ascending [`NodeId`] for deterministic output.
///
/// # Boundary-ordering ABI (compiler-integration seam)
///
/// The variadic input/output order of the emitted EPContext node is reconstructed
/// here purely from **ascending `NodeId`** over the covered set. This is
/// deterministic and correct for the current caller, but node-id order is **not**
/// an explicit, versioned ABI — a future compiler that emits its own canonical
/// boundary ordering (or relies on a specific i/o slot layout) must not silently
/// assume this ordering. If/when compiler integration lands, make the intended
/// boundary order an explicit contract between the partitioner and this seam
/// rather than depending on `NodeId` allocation order.
fn partition_boundary(
    graph: &Graph,
    covered: &HashSet<NodeId>,
) -> (Vec<Option<ValueId>>, Vec<ValueId>) {
    let mut ordered: Vec<NodeId> = covered.iter().copied().collect();
    ordered.sort_by_key(|n| n.0);

    let mut inputs: Vec<Option<ValueId>> = Vec::new();
    let mut seen_in: HashSet<ValueId> = HashSet::new();
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut seen_out: HashSet<ValueId> = HashSet::new();
    let graph_outputs: HashSet<ValueId> = graph.outputs.iter().copied().collect();

    for nid in &ordered {
        let node = graph.node(*nid);
        for slot in &node.inputs {
            let Some(vid) = *slot else { continue };
            let external = match graph.value(vid).producer {
                Some(prod) => !covered.contains(&prod),
                None => true, // graph input / initializer / source
            };
            if external && seen_in.insert(vid) {
                inputs.push(Some(vid));
            }
        }
    }
    for nid in &ordered {
        let node = graph.node(*nid);
        for &vid in &node.outputs {
            let escapes = graph_outputs.contains(&vid)
                || graph
                    .value(vid)
                    .consumers
                    .iter()
                    .any(|c| !covered.contains(c));
            if escapes && seen_out.insert(vid) {
                outputs.push(vid);
            }
        }
    }
    (inputs, outputs)
}

/// The output context-model path: `config.file_path`, else `<orig_stem>_ctx.onnx`
/// beside `orig_path`.
fn resolve_output_path(orig_path: &Path, config: &EpContextDumpConfig) -> PathBuf {
    if let Some(p) = &config.file_path {
        return p.clone();
    }
    let dir = orig_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = orig_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    dir.join(format!("{stem}_ctx.onnx"))
}

/// Sidecar `.bin` filename for an external blob (§55.4):
/// `<ctx_stem>_p{index}_<source>_<partition>.bin`, with `source`/`partition`
/// sanitised to filesystem-safe characters. Returned as a bare filename
/// (relative to the model dir, per the §55.3 external-path policy).
///
/// The `index` (the partition's position within this dump call) is an
/// **injective** disambiguator: [`sanitize_component`] is non-injective (it maps
/// every disallowed char to `_`), so two partitions with different identities —
/// e.g. sources `Vendor/EP` and `Vendor_EP` — sanitise to the same components
/// and would otherwise collide onto one file, silently overwriting each other's
/// blob. Prefixing the distinct partition index guarantees a distinct filename
/// per partition, so every EPContext node stores the path of *its own* blob.
fn sidecar_filename(out_path: &Path, index: usize, source: &str, partition: &str) -> String {
    let stem = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let src = sanitize_component(source);
    let part = sanitize_component(partition);
    if part.is_empty() {
        format!("{stem}_p{index}_{src}.bin")
    } else {
        format!("{stem}_p{index}_{src}_{part}.bin")
    }
}

/// Replace any character that is not alphanumeric, `-`, `_`, or `.` with `_`,
/// so an EP-supplied `source`/`partition` never yields a path separator or an
/// otherwise unsafe filename component.
fn sanitize_component(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
