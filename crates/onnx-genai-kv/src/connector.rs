//! Pluggable external KV-cache connector abstraction (DESIGN §38).
//!
//! This module defines the [`KvCacheConnector`] trait and its associated key /
//! value types so external KV stores (LMCache, Mooncake, InfiniStore, Redis, or
//! our own tiered backend) can plug into the engine. It is the **abstraction
//! foundation** ("K1"): it ships the trait, the key/value types, the
//! chunk-hashing helper used for prefix-cache keying, and a [`NullConnector`]
//! default implementation.
//!
//! Deliberately *not* included here (by milestone design):
//! - concrete backends such as `LocalTieredConnector` / `LMCacheConnector`
//!   (milestone "K2");
//! - scheduler / engine wiring (milestone "K3").
//!
//! ## Model-agnostic by construction
//!
//! [`KvCacheKey::model_id`] is an **opaque** identity string. Different models
//! produce incompatible KV, so keys from different `model_id`s never collide,
//! but no code here (or in any backend) is allowed to branch on specific model
//! names. Chunk hashing operates on raw token ids generically, and `chunk_size`,
//! compression, and capabilities are configuration/parameters — never
//! hardcoded per model.

use std::ops::Range;
use std::time::Duration;

use crate::{Device, PageId, TokenId};

/// Default token-chunk size (tokens per cached chunk), matching the design's
/// recommended `page_size == chunk_size` alignment (DESIGN §38.8).
///
/// This is only a *default*; the chunk size is always a parameter to
/// [`chunk_tokens`] and is carried in [`ConnectorCapabilities::max_chunk_tokens`].
pub const DEFAULT_CHUNK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`KvCacheConnector`] operations.
///
/// Kept separate from [`crate::KvError`] because connector failures are
/// dominated by I/O / transport concerns (network, disk, remote nodes) rather
/// than paged-cache invariants.
#[derive(Debug, thiserror::Error)]
pub enum ConnectorError {
    /// The requested key is not cached anywhere, so there is nothing to fetch.
    #[error("KV not found for the requested key")]
    NotFound,
    /// The backend transport (network / disk / daemon) failed.
    #[error("connector backend error: {0}")]
    Backend(String),
    /// The operation is not supported by this connector.
    #[error("operation not supported by this connector: {0}")]
    Unsupported(&'static str),
}

/// Convenience result alias for connector operations.
pub type ConnectorResult<T> = Result<T, ConnectorError>;

// ---------------------------------------------------------------------------
// Key types (DESIGN §38.4)
// ---------------------------------------------------------------------------

/// Identifies a cached KV segment by token content hash.
///
/// Uses chunked hashing: tokens are split into fixed-size chunks (see
/// [`chunk_tokens`]), and each chunk is hashed **cumulatively** over every
/// token from position 0 through the end of that chunk. This makes the key
/// *prefix-dependent*: two sequences produce the same `chunk_hash` for chunk N
/// only when they share an identical token prefix `[0..=end_of_chunk_N]`.
/// Prefix sharing still works at chunk granularity — genuinely shared prefixes
/// (e.g. a common system prompt) collide as before — but sequences that differ
/// *earlier* now correctly diverge, so a matching key guarantees identical
/// preceding context. The hash is deterministic and process-independent
/// (see [`hash_tokens`]), so keys are stable *across processes and nodes*.
///
/// This prefix-dependence is a **correctness requirement**: a chunk's real KV
/// depends on all earlier tokens (causal attention), so keying it on the chunk
/// tokens alone would let KV computed under one prefix be reused under a
/// different prefix — silently corrupting output once fetched KV is
/// materialized into the paged cache.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct KvCacheKey {
    /// Opaque model identity. Different models have incompatible KV; this is
    /// never interpreted or branched on — it only namespaces the key.
    pub model_id: String,
    /// Layer range this KV covers (for layer-parallel / layer-partial storage).
    pub layer_range: Range<usize>,
    /// Cumulative token hash: `hash_tokens(token_ids[0..=chunk_end])`, i.e. the
    /// running hash of every token up to and including this chunk. Encodes the
    /// full preceding context, not just this chunk's tokens.
    pub chunk_hash: u64,
    /// Chunk index within the sequence (for ordering).
    pub chunk_index: u32,
    /// Number of tokens in this chunk.
    pub num_tokens: u32,
}

/// Where the KV for a key currently lives (DESIGN §38.4).
#[derive(Clone, Debug, PartialEq)]
pub enum KvCacheLocation {
    /// On this node's GPU — can be used immediately, zero load cost.
    LocalGpu { page_ids: Vec<PageId> },
    /// On this node's CPU (pinned memory) — needs a GPU upload.
    LocalCpu {
        estimated_load_ms: f64,
        size_bytes: usize,
    },
    /// On this node's disk/NVMe — needs a disk read + GPU upload.
    LocalDisk {
        estimated_load_ms: f64,
        size_bytes: usize,
    },
    /// On a remote node — needs a network transfer.
    Remote {
        node_id: String,
        estimated_load_ms: f64,
        size_bytes: usize,
    },
    /// Not cached anywhere — must recompute (full prefill).
    NotFound,
}

/// Element type of the f32-materialized KV payload carried by the connector.
///
/// Only [`KvPayloadDtype::F32`] is produced today; the field makes the on-wire
/// contract explicit and lets a future revision carry compressed/quantized host
/// data (e.g. FP8) without changing the trait surface. The `kv` crate treats the
/// payload purely as typed host data — it never interprets its *meaning*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvPayloadDtype {
    /// 32-bit IEEE-754 host floats (the only format materialized today).
    F32,
}

/// Real, owned host KV bytes for one cached chunk (DESIGN §38.4, milestone K4).
///
/// This replaces the K1–K3 size-only placeholder: a connector now carries the
/// *actual* KV so it can be reused across sessions/nodes. It is deliberately
/// **opaque in meaning** — the `kv` crate never interprets these floats as
/// attention state; it only stores, sizes, and round-trips them. The engine
/// owns the semantics (which model, which layers, RoPE positions, etc.).
///
/// ## Byte-layout contract (must match [`crate::MaterializedLayerKv`])
///
/// - `layers[l].key` / `layers[l].value` are each a contiguous `f32` buffer of
///   exactly `num_kv_heads * num_tokens * head_dim` elements.
/// - The element for head `h`, chunk-relative token `t`, dim `d` lives at
///   flat index `(h * num_tokens + t) * head_dim + d` — i.e. **head-major**
///   `[num_kv_heads, num_tokens, head_dim]`, identical to
///   [`crate::MaterializedLayerKv`] sliced to this chunk's token range. Token
///   `t` corresponds to the chunk's `chunk_index * chunk_size + t`-th absolute
///   sequence position.
/// - `layers.len() == num_layers`; every layer shares the same dims.
///
/// This exact layout is what makes an extract→store→fetch→inject round-trip
/// byte-identical to a full recompute (see the engine's K4 wiring), so it is a
/// **correctness contract**, not just a memory convention.
#[derive(Clone, Debug, PartialEq)]
pub struct KvPayload {
    /// Number of tokens this chunk's KV covers.
    pub num_tokens: usize,
    /// Number of transformer layers represented (`== layers.len()`).
    pub num_layers: usize,
    /// KV attention heads per layer.
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Element type of every `key`/`value` buffer.
    pub dtype: KvPayloadDtype,
    /// Per-layer key/value host data (see the byte-layout contract above).
    pub layers: Vec<KvLayerPayload>,
}

/// Per-layer key/value host buffers for a [`KvPayload`].
///
/// Both buffers use the head-major `[num_kv_heads, num_tokens, head_dim]` layout
/// documented on [`KvPayload`].
#[derive(Clone, Debug, PartialEq)]
pub struct KvLayerPayload {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

impl KvPayload {
    /// Number of `f32` elements expected in each per-layer key (or value)
    /// buffer under the layout contract.
    pub fn elements_per_layer(&self) -> usize {
        self.num_kv_heads * self.num_tokens * self.head_dim
    }

    /// Total host bytes occupied by all key/value data. Used for honest size
    /// accounting / eviction math in place of the old placeholder `size_bytes`.
    pub fn byte_size(&self) -> usize {
        let elem = match self.dtype {
            KvPayloadDtype::F32 => std::mem::size_of::<f32>(),
        };
        self.layers
            .iter()
            .map(|layer| (layer.key.len() + layer.value.len()) * elem)
            .sum()
    }

    /// Validate that the payload's buffers match its declared dimensions.
    ///
    /// Returns `false` when any layer buffer has the wrong length or the layer
    /// count disagrees with `num_layers`, so backends can reject malformed data
    /// instead of silently storing something that will corrupt output on fetch.
    pub fn is_well_formed(&self) -> bool {
        if self.layers.len() != self.num_layers {
            return false;
        }
        let expected = self.elements_per_layer();
        self.layers
            .iter()
            .all(|layer| layer.key.len() == expected && layer.value.len() == expected)
    }
}

/// Data handed to a connector to store externally (DESIGN §38.4).
#[derive(Clone, Debug)]
pub struct KvStoreEntry {
    pub key: KvCacheKey,
    /// Real, owned KV host data for this chunk. See [`KvPayload`].
    pub kv_data: KvPayload,
    /// Storage priority hint.
    pub priority: CachePriority,
    /// Optional time-to-live.
    pub ttl: Option<Duration>,
}

/// KV data retrieved from external storage, ready to be injected by the engine
/// (DESIGN §38.4).
#[derive(Clone, Debug, PartialEq)]
pub struct FetchedKv {
    pub key: KvCacheKey,
    /// Pages allocated and filled on the target device, when the backend keeps
    /// a device-resident mirror (empty for pure host backends).
    pub pages: Vec<PageId>,
    /// The real KV host data for this chunk, in the [`KvPayload`] layout. The
    /// engine injects this into its own KV state at the chunk's absolute
    /// positions.
    pub payload: KvPayload,
    /// Actual transfer time (for metrics).
    pub transfer_time: Duration,
}

/// What a connector supports, so the scheduler knows what is possible
/// (DESIGN §38.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectorCapabilities {
    /// Supports cross-node sharing (not just local offload).
    pub distributed: bool,
    /// Supports async prefetch.
    pub prefetch: bool,
    /// Supports pinning entries as non-evictable.
    pub pinnable: bool,
    /// Maximum chunk size in tokens the backend accepts.
    pub max_chunk_tokens: usize,
    /// Supported compression formats.
    pub compression: Vec<CompressionFormat>,
}

/// Storage priority for a stored entry (DESIGN §38.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePriority {
    /// System prompt, shared by many sessions — keep as long as possible.
    SystemPrompt,
    /// Active session — keep until the session ends.
    Session,
    /// Speculative — might be reused, low priority.
    Opportunistic,
}

/// Compression format for stored KV (DESIGN §38.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompressionFormat {
    /// No compression.
    None,
    /// FP16 → FP8 quantization (~2× compression, minimal quality loss).
    Fp8,
    /// CacheGen-style learned compression.
    CacheGen,
    /// zstd byte-level compression (for CPU/disk tier).
    Zstd,
}

/// Connector health, reported by [`KvCacheConnector::health`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectorHealth {
    /// Fully operational.
    Healthy,
    /// Operational but impaired (e.g. a tier is unavailable or slow).
    Degraded { detail: String },
    /// Not usable right now.
    Unavailable { detail: String },
}

// ---------------------------------------------------------------------------
// Token chunking + stable hashing (DESIGN §38.8)
// ---------------------------------------------------------------------------

/// A fixed-size (except possibly the last) chunk of tokens plus its cumulative
/// prefix hash, the unit of external caching (DESIGN §38.8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenChunk {
    /// Chunk index within the sequence (for ordering).
    pub index: u32,
    /// The tokens in this chunk (last chunk may be shorter than `chunk_size`).
    pub tokens: Vec<TokenId>,
    /// Cumulative content hash of every token from position 0 through the end
    /// of this chunk — equal to `hash_tokens(&all_tokens[0..=chunk_end])` (see
    /// [`chunk_tokens`] / [`hash_tokens`]). This makes the hash — and therefore
    /// the [`KvCacheKey`] — prefix-dependent, not a hash of this chunk alone.
    pub hash: u64,
}

impl TokenChunk {
    /// Build the [`KvCacheKey`] for this chunk under a given model identity and
    /// layer range. `model_id` is opaque (see the module docs). The chunk's
    /// cumulative prefix `hash` is carried into [`KvCacheKey::chunk_hash`], so
    /// the resulting key encodes the full preceding token context.
    pub fn to_key(&self, model_id: impl Into<String>, layer_range: Range<usize>) -> KvCacheKey {
        KvCacheKey {
            model_id: model_id.into(),
            layer_range,
            chunk_hash: self.hash,
            chunk_index: self.index,
            num_tokens: self.tokens.len() as u32,
        }
    }
}

/// FNV-1a (64-bit) parameters. Fixed here so the same tokens hash to the same
/// value on every process and node (required for cross-node prefix sharing).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Absorb the little-endian bytes of each token id into an existing FNV-1a
/// state and return the updated state. Threading the state across chunk
/// boundaries is what yields the cumulative, prefix-dependent chunk hashes.
fn fnv1a_absorb_tokens(mut hash: u64, tokens: &[TokenId]) -> u64 {
    for &token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Stable, deterministic, process-independent hash of a token slice.
///
/// Implemented as **FNV-1a (64-bit)** over the little-endian bytes of each
/// token id. FNV-1a is chosen deliberately over Rust's default hasher
/// (`DefaultHasher`/SipHash is randomly seeded per process, so it could not be
/// used for cross-node prefix sharing). The constants and byte order are fixed
/// here, so the same tokens hash to the same value on every process and node.
///
/// This is the pure, position-independent hash of exactly the tokens passed in.
/// Chunk keys are *not* built from this directly on a per-chunk window; instead
/// [`chunk_tokens`] threads the FNV state across chunk boundaries so each
/// chunk's hash is cumulative over all preceding tokens (prefix-dependent) —
/// which is what makes a matching [`KvCacheKey`] guarantee identical preceding
/// context under causal attention.
pub fn hash_tokens(tokens: &[TokenId]) -> u64 {
    fnv1a_absorb_tokens(FNV_OFFSET_BASIS, tokens)
}

/// Split a token sequence into fixed-size chunks for caching (DESIGN §38.8).
///
/// The last chunk may be smaller than `chunk_size` and is stored as-is.
/// `chunk_size` is always a parameter (default [`DEFAULT_CHUNK_SIZE`]); it is
/// never derived from the model.
///
/// Each chunk's `hash` is **cumulative**: the running FNV-1a state is threaded
/// across chunks, so `chunks[i].hash == hash_tokens(&token_ids[0..=chunk_end])`
/// covers every token up to and including that chunk. This makes chunk keys
/// prefix-dependent — two sequences yield the same hash for chunk `i` iff they
/// share an identical prefix through the end of chunk `i` — which preserves
/// genuine prefix sharing while preventing reuse of KV across differing
/// prefixes.
///
/// # Panics
///
/// Panics if `chunk_size == 0`.
pub fn chunk_tokens(token_ids: &[TokenId], chunk_size: usize) -> Vec<TokenChunk> {
    assert!(chunk_size > 0, "chunk_size must be greater than zero");
    let mut rolling = FNV_OFFSET_BASIS;
    token_ids
        .chunks(chunk_size)
        .enumerate()
        .map(|(idx, chunk)| {
            rolling = fnv1a_absorb_tokens(rolling, chunk);
            TokenChunk {
                index: idx as u32,
                tokens: chunk.to_vec(),
                hash: rolling,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core trait (DESIGN §38.3)
// ---------------------------------------------------------------------------

/// External KV cache storage interface.
///
/// Implementations (in later milestones): `LocalTiered`, `LMCache`, `Mooncake`,
/// `InfiniStore`, `Redis`, etc. This K1 milestone only ships [`NullConnector`].
#[async_trait::async_trait]
pub trait KvCacheConnector: Send + Sync {
    /// Query: is KV for this token chunk already cached externally? Returns
    /// location info so the scheduler can estimate load cost.
    async fn lookup(&self, key: &KvCacheKey) -> ConnectorResult<KvCacheLocation>;

    /// Batch lookup: check multiple chunks at once (amortize network RTT).
    ///
    /// The default implementation loops over [`lookup`](Self::lookup);
    /// distributed backends should override it to issue a single round-trip.
    async fn lookup_batch(&self, keys: &[KvCacheKey]) -> ConnectorResult<Vec<KvCacheLocation>> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            out.push(self.lookup(key).await?);
        }
        Ok(out)
    }

    /// Store: push newly computed KV to external storage. Called asynchronously
    /// after prefill — must NOT block inference.
    async fn store(&self, entry: KvStoreEntry) -> ConnectorResult<()>;

    /// Fetch: load KV from external storage into local device memory. Returns
    /// pages ready to be used by the model. Errors with
    /// [`ConnectorError::NotFound`] if the key is not cached.
    async fn fetch(&self, key: &KvCacheKey, target: Device) -> ConnectorResult<FetchedKv>;

    /// Prefetch: hint that this KV will be needed soon. Non-blocking; an
    /// implementation may start a background transfer.
    fn prefetch(&self, key: &KvCacheKey, target: Device);

    /// Pin: mark an entry as non-evictable (hot system prompts, etc.).
    async fn pin(&self, key: &KvCacheKey) -> ConnectorResult<()>;

    /// Unpin: allow eviction again.
    async fn unpin(&self, key: &KvCacheKey) -> ConnectorResult<()>;

    /// Evict: explicitly remove an entry from external storage.
    async fn evict(&self, key: &KvCacheKey) -> ConnectorResult<()>;

    /// Health check.
    async fn health(&self) -> ConnectorHealth;

    /// Connector capabilities (so the scheduler knows what is possible).
    fn capabilities(&self) -> ConnectorCapabilities;
}

// ---------------------------------------------------------------------------
// NullConnector (DESIGN §38.5.3)
// ---------------------------------------------------------------------------

/// No external storage: KV lives only in the local GPU paged cache.
///
/// The simplest mode — single node, no offload. Every lookup reports
/// [`KvCacheLocation::NotFound`], stores/evicts/pins are successful no-ops, and
/// `fetch` errors with [`ConnectorError::NotFound`] (there is nothing to
/// fetch).
#[derive(Debug, Clone, Copy, Default)]
pub struct NullConnector;

#[async_trait::async_trait]
impl KvCacheConnector for NullConnector {
    async fn lookup(&self, _key: &KvCacheKey) -> ConnectorResult<KvCacheLocation> {
        Ok(KvCacheLocation::NotFound)
    }

    async fn lookup_batch(&self, keys: &[KvCacheKey]) -> ConnectorResult<Vec<KvCacheLocation>> {
        Ok(vec![KvCacheLocation::NotFound; keys.len()])
    }

    async fn store(&self, _entry: KvStoreEntry) -> ConnectorResult<()> {
        Ok(())
    }

    async fn fetch(&self, _key: &KvCacheKey, _target: Device) -> ConnectorResult<FetchedKv> {
        Err(ConnectorError::NotFound)
    }

    fn prefetch(&self, _key: &KvCacheKey, _target: Device) {}

    async fn pin(&self, _key: &KvCacheKey) -> ConnectorResult<()> {
        Ok(())
    }

    async fn unpin(&self, _key: &KvCacheKey) -> ConnectorResult<()> {
        Ok(())
    }

    async fn evict(&self, _key: &KvCacheKey) -> ConnectorResult<()> {
        Ok(())
    }

    async fn health(&self) -> ConnectorHealth {
        ConnectorHealth::Healthy
    }

    fn capabilities(&self) -> ConnectorCapabilities {
        ConnectorCapabilities {
            distributed: false,
            prefetch: false,
            pinnable: false,
            max_chunk_tokens: usize::MAX,
            compression: vec![CompressionFormat::None],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- chunk_tokens ---------------------------------------------------

    #[test]
    fn chunk_tokens_splits_with_final_partial_chunk() {
        let tokens: Vec<TokenId> = (0..10).collect();
        let chunks = chunk_tokens(&tokens, 4);
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].tokens, vec![0, 1, 2, 3]);
        assert_eq!(chunks[1].index, 1);
        assert_eq!(chunks[1].tokens, vec![4, 5, 6, 7]);
        // Final partial chunk kept as-is.
        assert_eq!(chunks[2].index, 2);
        assert_eq!(chunks[2].tokens, vec![8, 9]);
    }

    #[test]
    fn chunk_tokens_num_tokens_via_to_key() {
        let tokens: Vec<TokenId> = (0..10).collect();
        let chunks = chunk_tokens(&tokens, 4);
        let full = chunks[0].to_key("m", 0..1);
        let partial = chunks[2].to_key("m", 0..1);
        assert_eq!(full.num_tokens, 4);
        assert_eq!(partial.num_tokens, 2);
        assert_eq!(partial.chunk_index, 2);
    }

    #[test]
    fn chunk_tokens_empty_input_is_empty() {
        let chunks = chunk_tokens(&[], 4);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_tokens_index_ordering_is_monotonic() {
        let tokens: Vec<TokenId> = (0..100).collect();
        let chunks = chunk_tokens(&tokens, 7);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index as usize, i);
        }
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than zero")]
    fn chunk_tokens_zero_chunk_size_panics() {
        let _ = chunk_tokens(&[1, 2, 3], 0);
    }

    // --- hash stability -------------------------------------------------

    #[test]
    fn hash_tokens_is_stable_against_hardcoded_values() {
        // Hardcoded FNV-1a(64) over little-endian u32 bytes. If the hashing
        // scheme ever changes, these assertions fail and force a review, since
        // cross-node prefix sharing depends on hash stability.
        assert_eq!(hash_tokens(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash_tokens(&[1, 2, 3]), 0xfd1f_0f43_81eb_0395);
        assert_eq!(hash_tokens(&[1, 2, 3, 4]), 0x84c3_9a07_9fc0_8121);
        assert_eq!(hash_tokens(&[4, 5, 6]), 0x9c54_9f63_2639_e712);
    }

    #[test]
    fn hash_tokens_same_tokens_same_hash() {
        assert_eq!(hash_tokens(&[7, 8, 9, 10]), hash_tokens(&[7, 8, 9, 10]));
    }

    #[test]
    fn hash_tokens_different_tokens_differ() {
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[3, 2, 1]));
    }

    #[test]
    fn chunk_tokens_cumulative_hash_is_stable_against_hardcoded_values() {
        // Guards the *rolling* chunk-hash scheme: each chunk's hash is the FNV
        // state threaded across all preceding tokens, so it equals the pure
        // hash of the sequence up to and including that chunk. If the rolling
        // scheme ever changes, these fail and force a review, since cross-node
        // prefix sharing (and materialization correctness) depend on stability.
        let tokens: Vec<TokenId> = (0..10).collect();
        let chunks = chunk_tokens(&tokens, 4);
        assert_eq!(chunks[0].hash, 0x30d7_7e22_c5da_0365);
        assert_eq!(chunks[1].hash, 0x66b0_4c33_23ce_3f25);
        assert_eq!(chunks[2].hash, 0x4363_3e3f_f0f8_85b4);
        // Cumulative: chunk i's hash == pure hash of every token through it.
        assert_eq!(chunks[0].hash, hash_tokens(&tokens[0..4]));
        assert_eq!(chunks[1].hash, hash_tokens(&tokens[0..8]));
        assert_eq!(chunks[2].hash, hash_tokens(&tokens[0..10]));
    }

    #[test]
    fn chunk_hash_is_prefix_dependent() {
        // Two sequences that share an identical chunk-N *window* but differ in
        // an EARLIER chunk must now produce DIFFERENT hashes/keys for chunk N —
        // because chunk N's real KV depends on all preceding tokens (causal
        // attention). This is the correctness landmine fix.
        let a: Vec<TokenId> = vec![100, 101, 102, 103, 200, 201, 202, 203];
        let b: Vec<TokenId> = vec![900, 901, 902, 903, 200, 201, 202, 203];
        let ca = chunk_tokens(&a, 4);
        let cb = chunk_tokens(&b, 4);
        // Identical chunk-1 window ([200,201,202,203]) but different prefix.
        assert_eq!(ca[1].tokens, cb[1].tokens);
        assert_ne!(ca[1].hash, cb[1].hash);
        assert_ne!(ca[1].to_key("m", 0..1), cb[1].to_key("m", 0..1));
    }

    #[test]
    fn chunk_hash_shared_prefix_still_collides() {
        // Genuinely-shared prefixes (e.g. a common system prompt) must still
        // dedupe: two sequences with the SAME tokens through the end of chunk N
        // get the SAME hash/key for every chunk up to N, even if they diverge
        // afterwards.
        let a: Vec<TokenId> = vec![10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33];
        let b: Vec<TokenId> = vec![10, 11, 12, 13, 20, 21, 22, 23, 77, 78, 79, 80];
        let ca = chunk_tokens(&a, 4);
        let cb = chunk_tokens(&b, 4);
        // Shared prefix through chunk 1 → identical keys for chunks 0 and 1.
        assert_eq!(ca[0].to_key("m", 0..1), cb[0].to_key("m", 0..1));
        assert_eq!(ca[1].to_key("m", 0..1), cb[1].to_key("m", 0..1));
        // Diverge at chunk 2 → different keys.
        assert_ne!(ca[2].to_key("m", 0..1), cb[2].to_key("m", 0..1));
    }

    // --- KvCacheKey Hash/Eq --------------------------------------------

    fn key(model: &str, layers: Range<usize>, hash: u64) -> KvCacheKey {
        KvCacheKey {
            model_id: model.to_string(),
            layer_range: layers,
            chunk_hash: hash,
            chunk_index: 0,
            num_tokens: 4,
        }
    }

    #[test]
    fn kvcachekey_equal_keys_collide() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(key("model-a", 0..8, 42));
        assert!(set.contains(&key("model-a", 0..8, 42)));
        // Re-inserting an equal key does not grow the set.
        set.insert(key("model-a", 0..8, 42));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn kvcachekey_differing_fields_differ() {
        let base = key("model-a", 0..8, 42);
        assert_ne!(base, key("model-b", 0..8, 42)); // model_id
        assert_ne!(base, key("model-a", 0..8, 43)); // chunk_hash
        assert_ne!(base, key("model-a", 8..16, 42)); // layer_range
    }

    // --- NullConnector --------------------------------------------------

    fn sample_key() -> KvCacheKey {
        key("model-a", 0..1, 12345)
    }

    /// A deterministic, well-formed payload for round-trip / accounting tests.
    fn sample_payload(
        num_tokens: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> KvPayload {
        let per_layer = num_kv_heads * num_tokens * head_dim;
        let layers = (0..num_layers)
            .map(|l| KvLayerPayload {
                key: (0..per_layer).map(|i| (l * 1000 + i) as f32).collect(),
                value: (0..per_layer).map(|i| -((l * 1000 + i) as f32)).collect(),
            })
            .collect();
        KvPayload {
            num_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype: KvPayloadDtype::F32,
            layers,
        }
    }

    #[test]
    fn kv_payload_byte_size_and_well_formed() {
        let payload = sample_payload(2, 3, 2, 4);
        // Each layer holds key+value = 2 * (2*2*4) = 32 f32 = 128 bytes; 3 layers.
        assert_eq!(payload.elements_per_layer(), 2 * 2 * 4);
        assert_eq!(payload.byte_size(), 3 * 2 * (2 * 2 * 4) * 4);
        assert!(payload.is_well_formed());

        let mut broken = payload.clone();
        broken.layers[0].key.pop();
        assert!(!broken.is_well_formed());

        let mut wrong_layers = payload;
        wrong_layers.layers.pop();
        assert!(!wrong_layers.is_well_formed());
    }

    #[tokio::test]
    async fn null_connector_lookup_is_not_found() {
        let c = NullConnector;
        assert_eq!(
            c.lookup(&sample_key()).await.unwrap(),
            KvCacheLocation::NotFound
        );
    }

    #[tokio::test]
    async fn null_connector_lookup_batch_all_not_found() {
        let c = NullConnector;
        let keys = vec![sample_key(), sample_key(), sample_key()];
        let locs = c.lookup_batch(&keys).await.unwrap();
        assert_eq!(locs.len(), 3);
        assert!(locs.iter().all(|l| *l == KvCacheLocation::NotFound));
    }

    #[tokio::test]
    async fn null_connector_store_is_ok() {
        let c = NullConnector;
        let entry = KvStoreEntry {
            key: sample_key(),
            kv_data: sample_payload(1, 1, 1, 1),
            priority: CachePriority::Opportunistic,
            ttl: None,
        };
        assert!(c.store(entry).await.is_ok());
    }

    #[tokio::test]
    async fn null_connector_fetch_errors_not_found() {
        let c = NullConnector;
        let err = c.fetch(&sample_key(), Device::Gpu(0)).await.unwrap_err();
        assert!(matches!(err, ConnectorError::NotFound));
    }

    #[tokio::test]
    async fn null_connector_pin_unpin_evict_are_ok() {
        let c = NullConnector;
        assert!(c.pin(&sample_key()).await.is_ok());
        assert!(c.unpin(&sample_key()).await.is_ok());
        assert!(c.evict(&sample_key()).await.is_ok());
    }

    #[tokio::test]
    async fn null_connector_prefetch_is_noop() {
        let c = NullConnector;
        c.prefetch(&sample_key(), Device::Gpu(0));
    }

    #[tokio::test]
    async fn null_connector_health_is_healthy() {
        let c = NullConnector;
        assert_eq!(c.health().await, ConnectorHealth::Healthy);
    }

    #[test]
    fn null_connector_capabilities_as_specified() {
        let caps = NullConnector.capabilities();
        assert!(!caps.distributed);
        assert!(!caps.prefetch);
        assert!(!caps.pinnable);
        assert_eq!(caps.max_chunk_tokens, usize::MAX);
        assert_eq!(caps.compression, vec![CompressionFormat::None]);
    }

    #[test]
    fn null_connector_is_object_safe() {
        // Ensure the trait is usable as a trait object (needed for the engine
        // to hold `Arc<dyn KvCacheConnector>` in K3).
        let _c: std::sync::Arc<dyn KvCacheConnector> = std::sync::Arc::new(NullConnector);
    }
}
