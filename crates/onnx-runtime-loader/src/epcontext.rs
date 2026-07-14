//! `com.microsoft::EPContext` node support — the load path (§55.3).
//!
//! An `EPContext` node is the **on-disk / interchange form** of a compiled-EP
//! context (§55.1): an ordinary ONNX contrib node (`op_type = "EPContext"`,
//! `domain = "com.microsoft"`) that either embeds a pre-compiled vendor blob
//! inline or references it in an external file. This module provides:
//!
//! * [`EpContextNode`] — a typed *view* over an ordinary IR [`Node`], parsing
//!   the §55.2 attributes. No new IR node kind is introduced.
//! * [`ep_context_nodes`] — enumerate the `EPContext` nodes in a [`Graph`].
//! * [`resolve_ep_context`] — turn the `ep_cache_context` attribute into an
//!   [`EpContextBlob`] (inline bytes, or a read-only `mmap` of an external file
//!   resolved relative to the model directory, §55.3).
//!
//! The loader **never interprets** the blob bytes — they are opaque vendor
//! payload handed to whichever EP claims the node's `source` key (§55.6),
//! which is out of scope here (owned by the session + ep-api).

use std::fs::File;
use std::path::{Component, Path, PathBuf};

use memmap2::Mmap;
use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, ValueId};

use crate::LoaderError;

/// The op type of an EPContext node.
const EP_CONTEXT_OP: &str = "EPContext";
/// The (single, non-aliased) domain EPContext nodes live in. The shape-inference
/// registry only normalises the default ONNX domain (`ai.onnx` ⇄ `""`); there is
/// no `com.microsoft` alias to normalise, so an exact match is correct (§55.3).
const MS_DOMAIN: &str = "com.microsoft";

/// Attribute names (§55.2).
mod attr {
    pub const MAIN_CONTEXT: &str = "main_context";
    pub const EP_CACHE_CONTEXT: &str = "ep_cache_context";
    pub const EMBED_MODE: &str = "embed_mode";
    pub const EP_SDK_VERSION: &str = "ep_sdk_version";
    pub const SOURCE: &str = "source";
    pub const PARTITION_NAME: &str = "partition_name";
}

/// Whether the payload lives inline in the node or in an external file (§55.2).
///
/// Mirrors the `embed_mode` attribute: `1` (or absent, the §55.2 default) means
/// [`Embedded`](EmbedMode::Embedded); `0` means [`ExternalFile`](EmbedMode::ExternalFile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedMode {
    /// `embed_mode = 1`: `ep_cache_context` holds the compiled blob inline.
    Embedded,
    /// `embed_mode = 0`: `ep_cache_context` holds an external file path
    /// (relative to the ONNX model file).
    ExternalFile,
}

impl EmbedMode {
    /// Decode the `embed_mode` attribute integer. Only `0` selects the external
    /// file; every other value (including the §55.2 default `1`) is treated as
    /// [`Embedded`] — the fail-closed choice, since it never touches the
    /// filesystem.
    fn from_attr_int(v: i64) -> Self {
        match v {
            0 => EmbedMode::ExternalFile,
            _ => EmbedMode::Embedded,
        }
    }
}

/// Where the compiled blob physically lives after load-time resolution (§55.3).
///
/// The loader treats the bytes as opaque; it never parses them.
#[derive(Debug)]
pub enum EpContextBlob {
    /// `embed_mode = 1`: bytes owned inline (copied out of `ep_cache_context`).
    Embedded(Vec<u8>),
    /// `embed_mode = 0`: a read-only `mmap` of an external file resolved
    /// *relative to the model directory*. Never eagerly copied into the graph.
    External {
        /// The resolved absolute path to the external context file.
        path: PathBuf,
        /// Read-only memory map of the file's bytes.
        map: Mmap,
    },
}

impl EpContextBlob {
    /// Borrow the raw blob bytes regardless of where they live.
    pub fn bytes(&self) -> &[u8] {
        match self {
            EpContextBlob::Embedded(b) => b,
            EpContextBlob::External { map, .. } => &map[..],
        }
    }
}

/// A typed view over a `com.microsoft::EPContext` node in the [`Graph`] IR
/// (§55.3). Backed by an ordinary [`Node`] plus its attributes — no separate IR
/// node kind. Constructed via [`EpContextNode::from_node`] or enumerated with
/// [`ep_context_nodes`].
///
/// Variadic boundary tensors are read straight from the underlying node via
/// [`inputs`](EpContextNode::inputs) / [`outputs`](EpContextNode::outputs); no
/// fixed arity is assumed.
#[derive(Debug)]
pub struct EpContextNode<'g> {
    /// The id of the backing IR node.
    pub node: NodeId,
    /// `source` attribute — the EP dispatch key (§55.6). `None` if absent.
    pub source: Option<&'g str>,
    /// `main_context != 0` (§55.2 default `true` when the attribute is absent).
    pub main_context: bool,
    /// Embedded vs. external payload (§55.2 default [`EmbedMode::Embedded`]).
    pub embed_mode: EmbedMode,
    /// `ep_sdk_version` attribute — SDK/toolchain version (diagnostics).
    pub sdk_version: Option<&'g str>,
    /// `partition_name` attribute — the ORT-partitioned graph name.
    pub partition_name: Option<&'g str>,
    /// The backing node, used to reach the variadic i/o and the raw
    /// `ep_cache_context` payload.
    inner: &'g Node,
}

impl<'g> EpContextNode<'g> {
    /// Build the typed view if `node` is a `com.microsoft::EPContext` node,
    /// otherwise `None`. Attribute parsing is defensive: missing or wrong-typed
    /// attributes fall back to the §55.2 defaults rather than erroring.
    pub fn from_node(node: &'g Node) -> Option<Self> {
        if !is_ep_context_op(&node.op_type, &node.domain) {
            return None;
        }
        let a = &node.attributes;
        let main_context = a
            .get(attr::MAIN_CONTEXT)
            .and_then(Attribute::as_int)
            .map(|v| v != 0)
            .unwrap_or(true);
        let embed_mode = a
            .get(attr::EMBED_MODE)
            .and_then(Attribute::as_int)
            .map(EmbedMode::from_attr_int)
            .unwrap_or(EmbedMode::Embedded);
        Some(EpContextNode {
            node: node.id,
            source: str_attr(node, attr::SOURCE),
            main_context,
            embed_mode,
            sdk_version: str_attr(node, attr::EP_SDK_VERSION),
            partition_name: str_attr(node, attr::PARTITION_NAME),
            inner: node,
        })
    }

    /// The node's variadic input value slots (partition boundary inputs, in
    /// order). `None` slots preserve positional arity for skipped optionals.
    pub fn inputs(&self) -> &'g [Option<ValueId>] {
        &self.inner.inputs
    }

    /// The node's variadic output values (partition boundary outputs, in order).
    pub fn outputs(&self) -> &'g [ValueId] {
        &self.inner.outputs
    }

    /// The backing [`Node`].
    pub fn inner(&self) -> &'g Node {
        self.inner
    }

    /// Raw `ep_cache_context` bytes, if present. STRING attributes are stored in
    /// the IR as raw bytes (see `Attribute::String`), so a binary payload is
    /// never mangled by UTF-8 decoding. A `UINT8`/opaque tensor is accepted as a
    /// fallback (e.g. a hand-built graph) (§55.3).
    fn ep_cache_context_bytes(&self) -> Option<&'g [u8]> {
        match self.inner.attributes.get(attr::EP_CACHE_CONTEXT)? {
            Attribute::String(s) => Some(s),
            Attribute::Tensor(t) => Some(&t.data),
            _ => None,
        }
    }
}

/// Whether `(op_type, domain)` identifies a `com.microsoft::EPContext` node.
pub fn is_ep_context_op(op_type: &str, domain: &str) -> bool {
    op_type == EP_CONTEXT_OP && domain == MS_DOMAIN
}

/// Read a non-empty string attribute, returning `None` when absent, empty, or
/// not a string.
fn str_attr<'g>(node: &'g Node, name: &str) -> Option<&'g str> {
    node.attributes
        .get(name)
        .and_then(Attribute::as_str)
        .filter(|s| !s.is_empty())
}

/// Enumerate the `com.microsoft::EPContext` nodes in `graph` as typed views
/// (§55.3, recognition helper). The session dispatches each one on its `source`
/// key via the `EpContextRegistry` (§55.6, owned by ep-api/session).
pub fn ep_context_nodes(graph: &Graph) -> impl Iterator<Item = EpContextNode<'_>> {
    graph.nodes.values().filter_map(EpContextNode::from_node)
}

/// The [`NodeId`]s of the `EPContext` nodes in `graph`, in node-arena order.
pub fn ep_context_node_ids(graph: &Graph) -> Vec<NodeId> {
    ep_context_nodes(graph).map(|n| n.node).collect()
}

/// Resolve the payload for one `EPContext` node (§55.3).
///
/// * `embed_mode = 1` → copy the inline `ep_cache_context` bytes into
///   [`EpContextBlob::Embedded`].
/// * `embed_mode = 0` → treat `ep_cache_context` as a path **relative to the
///   model directory** (identical policy to external-weight resolution, §19.2),
///   guard it against traversal/absolute escapes, then open + `mmap` it
///   read-only into [`EpContextBlob::External`]. The bytes are never eagerly
///   copied into the graph.
pub fn resolve_ep_context(
    model_dir: &Path,
    n: &EpContextNode,
) -> Result<EpContextBlob, LoaderError> {
    let raw = n.ep_cache_context_bytes().ok_or_else(|| {
        LoaderError::EpContext(format!(
            "EPContext node {:?} is missing the 'ep_cache_context' attribute",
            n.node
        ))
    })?;

    match n.embed_mode {
        EmbedMode::Embedded => Ok(EpContextBlob::Embedded(raw.to_vec())),
        EmbedMode::ExternalFile => {
            let rel = std::str::from_utf8(raw).map_err(|_| {
                LoaderError::EpContext(format!(
                    "EPContext node {:?}: external 'ep_cache_context' path is not valid UTF-8",
                    n.node
                ))
            })?;
            let path = resolve_external_path(model_dir, rel)?;
            let file = File::open(&path).map_err(|_| LoaderError::ExternalDataNotFound {
                path: path.clone(),
            })?;
            // SAFETY: identical idiom to `weights.rs`'s external-data mmap — the
            // `File` is held open for the duration of the map and the bytes are
            // only ever read immutably (opaque vendor blob). This is the same,
            // and only, `unsafe` pattern the loader already relies on.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::Mmap(e.to_string()))?;
            Ok(EpContextBlob::External { path, map })
        }
    }
}

/// Join a relative external-context path onto `model_dir`, rejecting anything
/// that escapes the model directory (§55.3 path-safety).
///
/// External-weight resolution (`weights.rs` §19.2) blindly `join`s the stored
/// `location`; it does **not** guard traversal. For untrusted `*_ctx.onnx`
/// interchange models we add a minimal guard here: absolute paths, filesystem
/// roots/prefixes, and any `..` component are rejected. (This guard is noted in
/// the decision record so the equivalent guard can be added to `weights.rs`.)
fn resolve_external_path(model_dir: &Path, rel: &str) -> Result<PathBuf, LoaderError> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err(LoaderError::EpContextPath {
            path: rel.to_string(),
            reason: "absolute paths are not allowed",
        });
    }
    for comp in rel_path.components() {
        match comp {
            Component::ParentDir => {
                return Err(LoaderError::EpContextPath {
                    path: rel.to_string(),
                    reason: "parent-directory (`..`) traversal is not allowed",
                });
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(LoaderError::EpContextPath {
                    path: rel.to_string(),
                    reason: "absolute / rooted paths are not allowed",
                });
            }
            // CurDir (`.`) and Normal segments are safe.
            _ => {}
        }
    }
    Ok(model_dir.join(rel_path))
}
