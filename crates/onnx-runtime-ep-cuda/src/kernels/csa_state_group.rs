//! Single property-typed accounting authority for Compressed-Sparse-Attention
//! (CSA / HCA) device state groups — the C1 closure of design blocker **B6**
//! (`docs/memory/...` DeepSeek-V4 CSA/HCA slice, §7).
//!
//! Before this module the CUDA CSA runner reserved its fixed-capacity device
//! buffers (compressed records, carry, dense sliding-window ring, index streams
//! and pooled scratch) straight through [`CudaRuntime::alloc_raw`]
//! (`csa_device_state.rs`) — a *second, unaccounted* device allocator sitting
//! beside every other governed pool. Those bytes were invisible to any
//! accounting authority, could not fail closed against a budget, and left no
//! per-request/per-device residency an operator could inspect.
//!
//! [`CsaStateGroupLedger`] is the single authority the CSA reservation now
//! routes through. It does **not** allocate device memory and it is **not** a
//! second cache/page manager: it is the ownership/accounting ledger the runner
//! consults *before* it reserves, so that
//!
//! * every CSA byte is charged to one place, per `(request, device)` for
//!   multi-request / multi-device isolation,
//! * a reservation **fails closed** against an optional managed byte limit with
//!   a typed refusal *before* any physical allocation (no partial, no leak),
//! * residency is exposed per state class (compressed / carry / dense-ring /
//!   index / scratch) so a test or operator can assert "compressed < dense" and
//!   "returns to baseline on teardown",
//! * and the backend keeps its logical cursors (`csa_checkpoint::CsaCursors`) —
//!   the ledger owns *bytes*, never cursor semantics.
//!
//! [`CsaStateGroupDescriptor`] is the property-typed gate in front of it: it
//! decides support from the graph's declared *properties* (compression ratio,
//! cache format, head geometry, device fan-out, presence of the state edges) —
//! never a model name or a shape allowlist — and carries the exact reason a
//! case was refused ([`CsaStateGroupRefusal`]).
//!
//! [`CudaRuntime::alloc_raw`]: crate::runtime::CudaRuntime::alloc_raw

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::EpError;
use onnx_runtime_ir::Node;

use crate::kernels::csa_device_state::CsaBufferLayout;

/// Record/cache encoding a graph declares for a CSA state group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CsaCacheFormat {
    F32,
    Fp8E4m3Block64,
    Fp4E2m1Block32,
}

impl CsaCacheFormat {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "f32" => Some(Self::F32),
            "fp8_e4m3_block64" => Some(Self::Fp8E4m3Block64),
            "fp4_e2m1_block32" => Some(Self::Fp4E2m1Block32),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Fp8E4m3Block64 => "fp8_e4m3_block64",
            Self::Fp4E2m1Block32 => "fp4_e2m1_block32",
        }
    }
}

/// Why a CSA state group was refused. Every variant carries the property that
/// decided it (design-discipline: "carry the reason"), so the next case is
/// diagnosable and a test can match the exact cause rather than a string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CsaStateGroupRefusal {
    /// The declared compression ratio is neither CSA (4) nor HCA (128).
    UnsupportedRatio { ratio: usize },
    /// The graph declares a `cache_format` string this build does not know.
    UnknownCacheFormat { raw: String },
    /// Ratio-4 (CSA) records are hybrid FP8/BF16; any other cache format cannot
    /// host the learned-index compressor.
    Ratio4RequiresFp8Cache { cache_format: &'static str },
    /// Ratio-4 selection reads a 128-wide FP4 index key; a different index head
    /// dim is a different, unsupported contract.
    Ratio4RequiresIndexHeadDim128 { index_head_dim: usize },
    /// Ratio-128 (HCA) attention-compressor records are f32 or hybrid FP8/BF16,
    /// never FP4.
    Ratio128RejectsFp4,
    /// A head geometry with a zero axis cannot describe a real attention op.
    InvalidHeadGeometry { num_heads: usize, head_dim: usize },
    /// The rotary sub-dimension cannot exceed the head dimension.
    RopeExceedsHeadDim {
        qk_rope_head_dim: usize,
        head_dim: usize,
    },
    /// v1 threads one device per state group; a batch that fans out across
    /// devices is ambiguous and fails closed rather than guessing.
    MultiDeviceAmbiguity { device_count: u32 },
    /// A required `past_* -> present_*` state edge is absent from the node, so
    /// there is nothing to thread — fail closed rather than fabricate state.
    MissingStateEdge { which: &'static str },
    /// The op supports this ratio but the C1 runtime slice threads only
    /// ratio-128 (HCA). Ratio-4 (CSA) and MTP recurrence are follow-up slices
    /// and are typed-refused here rather than silently threaded.
    UnsupportedC1Ratio { ratio: usize },
    /// The reservation would exceed the ledger's managed byte limit. Fails
    /// closed *before* any physical allocation.
    OutOfMemory {
        request: u64,
        device_ordinal: u32,
        requested: u64,
        resident: u64,
        limit: u64,
    },
}

impl std::fmt::Display for CsaStateGroupRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRatio { ratio } => {
                write!(
                    f,
                    "compression_ratio {ratio} is neither CSA (4) nor HCA (128)"
                )
            }
            Self::UnknownCacheFormat { raw } => write!(f, "unknown cache_format {raw:?}"),
            Self::Ratio4RequiresFp8Cache { cache_format } => {
                write!(f, "ratio-4 records are hybrid FP8/BF16, not {cache_format}")
            }
            Self::Ratio4RequiresIndexHeadDim128 { index_head_dim } => write!(
                f,
                "ratio-4 selection needs a 128-wide index key, not {index_head_dim}"
            ),
            Self::Ratio128RejectsFp4 => {
                write!(
                    f,
                    "ratio-128 attention-compressor records are f32 or FP8/BF16, not FP4"
                )
            }
            Self::InvalidHeadGeometry {
                num_heads,
                head_dim,
            } => write!(
                f,
                "invalid head geometry num_heads={num_heads} head_dim={head_dim}"
            ),
            Self::RopeExceedsHeadDim {
                qk_rope_head_dim,
                head_dim,
            } => write!(
                f,
                "qk_rope_head_dim {qk_rope_head_dim} exceeds head_dim {head_dim}"
            ),
            Self::MultiDeviceAmbiguity { device_count } => write!(
                f,
                "a CSA state group spans {device_count} devices; v1 requires exactly one"
            ),
            Self::MissingStateEdge { which } => {
                write!(f, "required state edge {which} is missing from the node")
            }
            Self::UnsupportedC1Ratio { ratio } => write!(
                f,
                "the C1 runtime slice threads only ratio-128 (HCA); ratio-{ratio} is a follow-up slice"
            ),
            Self::OutOfMemory {
                request,
                device_ordinal,
                requested,
                resident,
                limit,
            } => write!(
                f,
                "CSA state group (request {request}, device {device_ordinal}) needs {requested} B \
                 but {resident} B are resident against a {limit} B limit"
            ),
        }
    }
}

impl From<CsaStateGroupRefusal> for EpError {
    fn from(refusal: CsaStateGroupRefusal) -> Self {
        match refusal {
            CsaStateGroupRefusal::OutOfMemory {
                requested,
                resident,
                limit,
                ..
            } => EpError::OutOfMemory {
                requested: usize::try_from(requested).unwrap_or(usize::MAX),
                available: usize::try_from(limit.saturating_sub(resident)).unwrap_or(usize::MAX),
            },
            other => EpError::KernelFailed(format!(
                "CompressedSparseAttention state group refused: {other}"
            )),
        }
    }
}

/// Property-typed identity of one CSA state group, derived from the graph node.
///
/// Support is decided from these properties alone — no op/model name, no shape
/// allowlist — so an unseen but property-compatible layer is accepted and an
/// unsupported one is refused with its exact reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CsaStateGroupDescriptor {
    pub ratio: usize,
    pub cache_format: CsaCacheFormat,
    pub num_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub index_head_dim: usize,
    pub device_ordinal: u32,
    pub device_count: u32,
    pub request_id: u64,
    pub has_compressed_state: bool,
    pub has_carry_state: bool,
}

impl CsaStateGroupDescriptor {
    /// Parse the property set from a `pkg.nxrt::CompressedSparseAttention` node.
    ///
    /// Only the presence/values of declared attributes and the state input
    /// edges are read; numerics stay in the kernel. Returns a typed refusal if
    /// the declared `cache_format` string is unknown to this build.
    pub(crate) fn from_node(
        node: &Node,
        device_ordinal: u32,
        device_count: u32,
        request_id: u64,
    ) -> Result<Self, CsaStateGroupRefusal> {
        let ratio = attr_usize(node, "compression_ratio").unwrap_or(0);
        let raw_format = node
            .attr("cache_format")
            .and_then(|attribute| attribute.as_str())
            .unwrap_or("f32");
        let cache_format = CsaCacheFormat::parse(raw_format).ok_or_else(|| {
            CsaStateGroupRefusal::UnknownCacheFormat {
                raw: raw_format.to_string(),
            }
        })?;
        Ok(Self {
            ratio,
            cache_format,
            num_heads: attr_usize(node, "num_heads").unwrap_or(0),
            head_dim: attr_usize(node, "head_dim").unwrap_or(0),
            qk_rope_head_dim: attr_usize(node, "qk_rope_head_dim").unwrap_or(0),
            index_head_dim: attr_usize(node, "index_head_dim").unwrap_or(0),
            device_ordinal,
            device_count,
            request_id,
            // Frozen v1 threads the compressed records as input 6 and the carry
            // as input 7; a node missing either has no state edge to thread.
            has_compressed_state: node.inputs.get(6).map(Option::is_some).unwrap_or(false),
            has_carry_state: node.inputs.get(7).map(Option::is_some).unwrap_or(false),
        })
    }

    /// Decide support from the declared properties, carrying the exact reason on
    /// refusal. Accepts iff the layer is a property-compatible CSA (ratio-4) or
    /// HCA (ratio-128) group this build can thread on a single device.
    pub(crate) fn validate(&self) -> Result<(), CsaStateGroupRefusal> {
        if self.ratio != 4 && self.ratio != 128 {
            return Err(CsaStateGroupRefusal::UnsupportedRatio { ratio: self.ratio });
        }
        if self.num_heads == 0 || self.head_dim == 0 {
            return Err(CsaStateGroupRefusal::InvalidHeadGeometry {
                num_heads: self.num_heads,
                head_dim: self.head_dim,
            });
        }
        if self.qk_rope_head_dim > self.head_dim {
            return Err(CsaStateGroupRefusal::RopeExceedsHeadDim {
                qk_rope_head_dim: self.qk_rope_head_dim,
                head_dim: self.head_dim,
            });
        }
        if self.device_count != 1 {
            return Err(CsaStateGroupRefusal::MultiDeviceAmbiguity {
                device_count: self.device_count,
            });
        }
        if !self.has_compressed_state {
            return Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compressed_kv",
            });
        }
        if !self.has_carry_state {
            return Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compression_carry",
            });
        }
        match self.ratio {
            4 => {
                if self.cache_format != CsaCacheFormat::Fp8E4m3Block64 {
                    return Err(CsaStateGroupRefusal::Ratio4RequiresFp8Cache {
                        cache_format: self.cache_format.as_str(),
                    });
                }
                if self.index_head_dim != 128 {
                    return Err(CsaStateGroupRefusal::Ratio4RequiresIndexHeadDim128 {
                        index_head_dim: self.index_head_dim,
                    });
                }
            }
            128 => {
                if self.cache_format == CsaCacheFormat::Fp4E2m1Block32 {
                    return Err(CsaStateGroupRefusal::Ratio128RejectsFp4);
                }
            }
            _ => unreachable!("ratio guarded above"),
        }
        Ok(())
    }

    /// The ledger isolation key: one CSA state group per `(request, device)`.
    pub(crate) fn charge_key(&self) -> (u64, u32) {
        (self.request_id, self.device_ordinal)
    }

    /// Stricter than [`validate`](Self::validate): the C1 runtime slice threads
    /// exactly one ratio-128 (HCA) state group end-to-end. Ratio-4 (CSA) and
    /// any other ratio are op-supported but out of C1 scope, so they are
    /// typed-refused here rather than silently threaded through native decode.
    /// This is the gate the native-decode integration calls before admitting a
    /// CSA layer to the compressed state-threading path; that caller lands in
    /// the follow-up slice, so it is exercised by tests today.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn validate_c1_runtime(&self) -> Result<(), CsaStateGroupRefusal> {
        self.validate()?;
        if self.ratio != 128 {
            return Err(CsaStateGroupRefusal::UnsupportedC1Ratio { ratio: self.ratio });
        }
        Ok(())
    }
}

/// Bytes reserved for a CSA state group, split by state class so residency is
/// inspectable (e.g. "compressed < dense" for the HCA memory advantage).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CsaStateGroupBytes {
    pub compressed: u64,
    pub carry: u64,
    pub dense_ring: u64,
    pub index: u64,
    pub index_carry: u64,
    pub scratch: u64,
}

impl CsaStateGroupBytes {
    /// Exactly the bytes [`CsaDeviceBufferManager::reserve`] physically
    /// reserves for this layout: the seven fixed buffers folded into classes
    /// plus the pooled workspaces as scratch. Keeping this in lockstep with the
    /// reservation is what lets a test assert `ledger.resident == sum(reserved)`.
    ///
    /// [`CsaDeviceBufferManager::reserve`]: crate::kernels::csa_device_state::CsaDeviceBufferManager::reserve
    pub(crate) fn from_layout(layout: &CsaBufferLayout, workspace_bytes: &[usize]) -> Self {
        let scratch = workspace_bytes.iter().map(|&size| size.max(1) as u64).sum();
        Self {
            compressed: layout.attention_r4_bytes as u64 + layout.attention_r128_bytes as u64,
            carry: layout.attention_r4_carry_bytes as u64
                + layout.attention_r128_carry_bytes as u64,
            dense_ring: layout.dense_ring_bytes as u64,
            index: layout.index_r4_bytes as u64,
            index_carry: layout.index_r4_carry_bytes as u64,
            scratch,
        }
    }

    pub(crate) fn total(&self) -> u64 {
        self.compressed
            .saturating_add(self.carry)
            .saturating_add(self.dense_ring)
            .saturating_add(self.index)
            .saturating_add(self.index_carry)
            .saturating_add(self.scratch)
    }
}

/// The single ownership/accounting authority for CSA device state groups.
///
/// Owns *bytes*, never cursors. Charges are per `(request, device)` for
/// isolation, checked against an optional managed limit (0 = unlimited) that
/// fails closed *before* physical allocation, and released on
/// [`CsaStateGroupCharge`] drop so teardown returns to baseline.
#[derive(Debug, Default)]
pub(crate) struct CsaStateGroupLedger {
    limit: AtomicU64,
    resident: AtomicU64,
    peak: AtomicU64,
    compressed: AtomicU64,
    carry: AtomicU64,
    dense_ring: AtomicU64,
    index: AtomicU64,
    index_carry: AtomicU64,
    scratch: AtomicU64,
    charge_failures: AtomicU64,
    request_seq: AtomicU64,
    /// Resident bytes per `(request, device)` — the isolation ledger, and the
    /// serialization point that makes the check-then-charge atomic.
    charges: Mutex<HashMap<(u64, u32), u64>>,
}

impl CsaStateGroupLedger {
    /// Set the managed byte limit (0 = unlimited). Steady state leaves it 0 so
    /// the accounting is observable without changing behaviour; a test or an
    /// operator sets it to prove fail-closed admission.
    pub(crate) fn set_limit(&self, bytes: u64) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    pub(crate) fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }

    /// A fresh, process-unique request id for one runner instance, so distinct
    /// CSA runners get isolated `(request, device)` keys by default.
    pub(crate) fn next_request_id(&self) -> u64 {
        self.request_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Charge `bytes` for `(request, device)`, failing closed if it would cross
    /// the managed limit. The whole check-then-charge runs under the isolation
    /// lock, so a concurrent charge can never race past the limit, and no state
    /// is mutated on refusal (no partial reservation).
    pub(crate) fn try_charge(
        self: &Arc<Self>,
        key: (u64, u32),
        bytes: CsaStateGroupBytes,
    ) -> Result<CsaStateGroupCharge, CsaStateGroupRefusal> {
        let total = bytes.total();
        let mut charges = self
            .charges
            .lock()
            .expect("CSA state-group ledger poisoned");
        let limit = self.limit.load(Ordering::Relaxed);
        let resident = self.resident.load(Ordering::Relaxed);
        let next = resident.saturating_add(total);
        if limit != 0 && next > limit {
            self.charge_failures.fetch_add(1, Ordering::Relaxed);
            return Err(CsaStateGroupRefusal::OutOfMemory {
                request: key.0,
                device_ordinal: key.1,
                requested: total,
                resident,
                limit,
            });
        }
        self.resident.store(next, Ordering::Relaxed);
        bump_peak(&self.peak, next);
        self.compressed
            .fetch_add(bytes.compressed, Ordering::Relaxed);
        self.carry.fetch_add(bytes.carry, Ordering::Relaxed);
        self.dense_ring
            .fetch_add(bytes.dense_ring, Ordering::Relaxed);
        self.index.fetch_add(bytes.index, Ordering::Relaxed);
        self.index_carry
            .fetch_add(bytes.index_carry, Ordering::Relaxed);
        self.scratch.fetch_add(bytes.scratch, Ordering::Relaxed);
        *charges.entry(key).or_insert(0) += total;
        drop(charges);
        Ok(CsaStateGroupCharge {
            ledger: Arc::clone(self),
            key,
            bytes,
        })
    }

    fn release(&self, key: (u64, u32), bytes: CsaStateGroupBytes) {
        let total = bytes.total();
        let mut charges = self
            .charges
            .lock()
            .expect("CSA state-group ledger poisoned");
        self.resident.fetch_sub(
            total.min(self.resident.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
        sub_saturating(&self.compressed, bytes.compressed);
        sub_saturating(&self.carry, bytes.carry);
        sub_saturating(&self.dense_ring, bytes.dense_ring);
        sub_saturating(&self.index, bytes.index);
        sub_saturating(&self.index_carry, bytes.index_carry);
        sub_saturating(&self.scratch, bytes.scratch);
        if let Some(entry) = charges.get_mut(&key) {
            *entry = entry.saturating_sub(total);
            if *entry == 0 {
                charges.remove(&key);
            }
        }
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident.load(Ordering::Relaxed)
    }

    pub(crate) fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    pub(crate) fn compressed_bytes(&self) -> u64 {
        self.compressed.load(Ordering::Relaxed)
    }

    pub(crate) fn dense_ring_bytes(&self) -> u64 {
        self.dense_ring.load(Ordering::Relaxed)
    }

    pub(crate) fn charge_failures(&self) -> u64 {
        self.charge_failures.load(Ordering::Relaxed)
    }

    /// Resident bytes held by exactly one `(request, device)` — proves that one
    /// request/device's residency is isolated from another's.
    pub(crate) fn resident_for(&self, request: u64, device_ordinal: u32) -> u64 {
        self.charges
            .lock()
            .expect("CSA state-group ledger poisoned")
            .get(&(request, device_ordinal))
            .copied()
            .unwrap_or(0)
    }

    /// Number of distinct live `(request, device)` groups.
    pub(crate) fn active_group_count(&self) -> usize {
        self.charges
            .lock()
            .expect("CSA state-group ledger poisoned")
            .len()
    }
}

/// RAII charge against a [`CsaStateGroupLedger`]. Dropping it returns every
/// byte to the ledger, so a runner's teardown restores baseline residency with
/// no explicit accounting call.
pub(crate) struct CsaStateGroupCharge {
    ledger: Arc<CsaStateGroupLedger>,
    key: (u64, u32),
    bytes: CsaStateGroupBytes,
}

impl CsaStateGroupCharge {
    #[cfg(test)]
    pub(crate) fn total(&self) -> u64 {
        self.bytes.total()
    }
}

impl std::fmt::Debug for CsaStateGroupCharge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsaStateGroupCharge")
            .field("key", &self.key)
            .field("total", &self.bytes.total())
            .finish()
    }
}

impl Drop for CsaStateGroupCharge {
    fn drop(&mut self) {
        self.ledger.release(self.key, self.bytes);
    }
}

fn attr_usize(node: &Node, name: &str) -> Option<usize> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
}

fn bump_peak(peak: &AtomicU64, candidate: u64) {
    let mut current = peak.load(Ordering::Relaxed);
    while candidate > current {
        match peak.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn sub_saturating(counter: &AtomicU64, delta: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(delta);
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Node, NodeId};

    fn hca_node(cache_format: &str, with_state: bool) -> Node {
        // Inputs 0..=7 present means the frozen-v1 compressed (6) and carry (7)
        // state edges exist; `with_state=false` truncates them.
        let input_count = if with_state { 8 } else { 6 };
        let inputs = (0..input_count)
            .map(|_| Some(onnx_runtime_ir::ValueId(0)))
            .collect();
        let mut node = Node::new(NodeId(0), "CompressedSparseAttention", inputs, vec![]);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));
        node.attributes
            .insert("num_heads".into(), Attribute::Int(64));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(512));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes.insert(
            "cache_format".into(),
            Attribute::String(cache_format.into()),
        );
        node
    }

    #[test]
    fn hca_descriptor_accepts_property_compatible_ratio128() {
        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert_eq!(descriptor.ratio, 128);
        assert_eq!(descriptor.cache_format, CsaCacheFormat::F32);
        assert!(descriptor.validate().is_ok());
    }

    #[test]
    fn ratio128_rejects_fp4_by_property() {
        let node = hca_node("fp4_e2m1_block32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio128RejectsFp4)
        );
    }

    #[test]
    fn ratio4_requires_fp8_and_index_head_dim_128() {
        let mut node = hca_node("f32", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio4RequiresFp8Cache {
                cache_format: "f32"
            })
        );

        node.attributes.insert(
            "cache_format".into(),
            Attribute::String("fp8_e4m3_block64".into()),
        );
        // index_head_dim missing (0) -> refused for the index-key contract.
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::Ratio4RequiresIndexHeadDim128 { index_head_dim: 0 })
        );

        node.attributes
            .insert("index_head_dim".into(), Attribute::Int(128));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert!(descriptor.validate().is_ok());
    }

    #[test]
    fn unsupported_ratio_and_geometry_and_multidevice_and_missing_edges_refuse() {
        let mut node = hca_node("f32", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(2));
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 0, 1, 7)
                .unwrap()
                .validate(),
            Err(CsaStateGroupRefusal::UnsupportedRatio { ratio: 2 })
        );

        let mut node = hca_node("f32", true);
        node.attributes.insert("head_dim".into(), Attribute::Int(0));
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 0, 1, 7)
                .unwrap()
                .validate(),
            Err(CsaStateGroupRefusal::InvalidHeadGeometry {
                num_heads: 64,
                head_dim: 0
            })
        );

        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 2, 7).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::MultiDeviceAmbiguity { device_count: 2 })
        );

        let node = hca_node("f32", false);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert_eq!(
            descriptor.validate(),
            Err(CsaStateGroupRefusal::MissingStateEdge {
                which: "past_compressed_kv"
            })
        );
    }

    #[test]
    fn unknown_cache_format_is_typed() {
        let node = hca_node("bf16_block16", true);
        assert_eq!(
            CsaStateGroupDescriptor::from_node(&node, 0, 1, 7),
            Err(CsaStateGroupRefusal::UnknownCacheFormat {
                raw: "bf16_block16".into()
            })
        );
    }

    fn bytes(total_split: [u64; 6]) -> CsaStateGroupBytes {
        CsaStateGroupBytes {
            compressed: total_split[0],
            carry: total_split[1],
            dense_ring: total_split[2],
            index: total_split[3],
            index_carry: total_split[4],
            scratch: total_split[5],
        }
    }

    #[test]
    fn ledger_charges_release_and_return_to_baseline() {
        let ledger = Arc::new(CsaStateGroupLedger::default());
        assert_eq!(ledger.resident_bytes(), 0);
        let charge = ledger
            .try_charge((1, 0), bytes([100, 20, 40, 0, 0, 8]))
            .unwrap();
        assert_eq!(ledger.resident_bytes(), 168);
        assert_eq!(ledger.compressed_bytes(), 100);
        assert_eq!(ledger.dense_ring_bytes(), 40);
        assert_eq!(ledger.peak_bytes(), 168);
        assert_eq!(ledger.active_group_count(), 1);
        drop(charge);
        assert_eq!(ledger.resident_bytes(), 0);
        assert_eq!(ledger.active_group_count(), 0);
        // Peak is a high-water mark: it does not fall on release.
        assert_eq!(ledger.peak_bytes(), 168);
    }

    #[test]
    fn ledger_fails_closed_over_limit_without_mutating() {
        let ledger = Arc::new(CsaStateGroupLedger::default());
        ledger.set_limit(150);
        let _first = ledger
            .try_charge((1, 0), bytes([100, 0, 0, 0, 0, 0]))
            .unwrap();
        let refusal = ledger
            .try_charge((2, 0), bytes([100, 0, 0, 0, 0, 0]))
            .unwrap_err();
        assert!(matches!(refusal, CsaStateGroupRefusal::OutOfMemory { .. }));
        // Fail-closed: residency and the second group are untouched.
        assert_eq!(ledger.resident_bytes(), 100);
        assert_eq!(ledger.resident_for(2, 0), 0);
        assert_eq!(ledger.charge_failures(), 1);
    }

    #[test]
    fn ledger_isolates_requests_and_devices() {
        let ledger = Arc::new(CsaStateGroupLedger::default());
        let a = ledger
            .try_charge((1, 0), bytes([100, 0, 0, 0, 0, 0]))
            .unwrap();
        let b = ledger
            .try_charge((2, 0), bytes([50, 0, 0, 0, 0, 0]))
            .unwrap();
        let c = ledger
            .try_charge((1, 1), bytes([25, 0, 0, 0, 0, 0]))
            .unwrap();
        assert_eq!(ledger.resident_for(1, 0), 100);
        assert_eq!(ledger.resident_for(2, 0), 50);
        assert_eq!(ledger.resident_for(1, 1), 25);
        assert_eq!(ledger.resident_bytes(), 175);
        assert_eq!(ledger.active_group_count(), 3);
        drop(b);
        // Dropping one group leaves the others exactly as they were.
        assert_eq!(ledger.resident_for(1, 0), 100);
        assert_eq!(ledger.resident_for(2, 0), 0);
        assert_eq!(ledger.resident_for(1, 1), 25);
        assert_eq!(ledger.resident_bytes(), 125);
        drop(a);
        drop(c);
        assert_eq!(ledger.resident_bytes(), 0);
    }

    #[test]
    fn next_request_id_is_monotonic() {
        let ledger = Arc::new(CsaStateGroupLedger::default());
        let first = ledger.next_request_id();
        let second = ledger.next_request_id();
        assert!(second > first);
    }

    #[test]
    fn c1_runtime_admits_only_ratio128() {
        // Ratio-128 (HCA) is the one C1 slice group and is admitted.
        let node = hca_node("f32", true);
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert!(descriptor.validate_c1_runtime().is_ok());

        // Ratio-4 (CSA) is a valid op config but out of C1 scope: it validates
        // at the op level yet is typed-refused by the C1 runtime gate.
        let mut node = hca_node("fp8_e4m3_block64", true);
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        node.attributes
            .insert("index_head_dim".into(), Attribute::Int(128));
        let descriptor = CsaStateGroupDescriptor::from_node(&node, 0, 1, 7).unwrap();
        assert!(
            descriptor.validate().is_ok(),
            "ratio-4 is a valid op config"
        );
        assert_eq!(
            descriptor.validate_c1_runtime(),
            Err(CsaStateGroupRefusal::UnsupportedC1Ratio { ratio: 4 }),
            "ratio-4 is out of C1 scope"
        );
    }

    #[test]
    fn ledger_limit_round_trips() {
        let ledger = Arc::new(CsaStateGroupLedger::default());
        assert_eq!(
            ledger.limit(),
            0,
            "default ledger is unlimited (0 sentinel)"
        );
        ledger.set_limit(4096);
        assert_eq!(ledger.limit(), 4096);
    }
}
