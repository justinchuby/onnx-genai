//! Engine-side bridge to a pluggable [`KvCacheConnector`] (DESIGN §38, K3).
//!
//! This module wires the connector abstraction (K1) and its concrete backends
//! (K2, e.g. [`onnx_genai_kv::LocalTieredConnector`]) into the engine's existing
//! prefix-cache-hit path. It is deliberately **model-agnostic**: the only
//! per-model input is the opaque [`KvCacheKey::model_id`] namespace string —
//! nothing here branches on a specific model.
//!
//! ## What is LIVE vs DEFERRED in K3
//!
//! - **STORE (live):** after prefill computes fresh KV, complete prompt chunks
//!   are pushed to the connector via [`ConnectorBridge::store_prefix`] so future
//!   cross-session / cross-node requests can reuse them.
//! - **LOOKUP (live, reporting only):** for the tokens beyond the in-process
//!   prefix hit, [`ConnectorBridge::lookup_extension`] chunks them at the
//!   configured chunk size, builds keys, calls
//!   [`KvCacheConnector::lookup_batch`], and computes the contiguous would-be
//!   extension using each location's `estimated_load_ms` as the
//!   fetch-vs-recompute signal.
//! - **FETCH → MATERIALIZE (deferred):** actually copying fetched KV bytes into
//!   the engine's paged KV cache is **not** wired yet, because the K1
//!   [`onnx_genai_kv::KvTensorRef`] / [`onnx_genai_kv::FetchedKv`] carry only a
//!   size placeholder and page ids in the connector's *own* page table — there
//!   is no real device-tensor handle to copy from. Materializing would require a
//!   real tensor handle on those types. Until then the bridge never claims a hit
//!   it cannot serve: [`lookup_extension`](ConnectorBridge::lookup_extension)
//!   reports `would_extend_tokens` but does **not** shorten prefill, so output
//!   stays correct. See the `TODO(K3-materialize)` note on `lookup_extension`.
//!
//! ## Async on a sync engine
//!
//! [`KvCacheConnector`] is an `#[async_trait]`. The engine API is synchronous,
//! so an inactive ([`NullConnector`]) bridge never touches any runtime and an
//! active bridge owns a private current-thread Tokio runtime it `block_on`s.
//! (The shipped backends complete synchronously; they never yield across an
//! `.await`.)

use std::ops::Range;
use std::sync::Arc;

use onnx_genai_kv::{
    CachePriority, KvCacheConnector, KvCacheKey, KvCacheLocation, KvStoreEntry, KvTensorRef,
    NullConnector, TokenChunk, chunk_tokens,
};

use crate::logits::TokenId;

/// Connector activity accumulated across a generation, for metrics and tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectorStats {
    /// Chunk keys submitted to `lookup_batch`.
    pub lookups: usize,
    /// Chunks the connector reported resident and cheaper-to-fetch, counted
    /// contiguously from the in-process prefix boundary.
    pub chunk_hits: usize,
    /// Tokens that *could* be skipped in prefill once fetch→materialize lands.
    /// Reported only; not applied to prefill in K3 (see module docs).
    pub would_extend_tokens: usize,
    /// Tokens actually materialized from the connector into the engine KV cache.
    /// Always `0` in K3 (materialization deferred).
    pub fetched_tokens: usize,
    /// Chunks pushed to the connector via `store`.
    pub stores: usize,
}

/// Outcome of a single [`ConnectorBridge::lookup_extension`] call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectorLookupOutcome {
    /// Contiguous chunk hits beyond the in-process prefix boundary.
    pub chunk_hits: usize,
    /// Tokens the connector could serve contiguously (reporting only in K3).
    pub would_extend_tokens: usize,
    /// Tokens actually materialized into the engine KV cache (0 in K3).
    pub fetched_tokens: usize,
}

/// Bridges the engine's prefix-cache path to an optional [`KvCacheConnector`].
///
/// A [`null`](ConnectorBridge::null) bridge is fully inert: every method is an
/// early-return no-op, so `NullConnector` reproduces the pre-connector engine
/// behavior exactly.
pub(crate) struct ConnectorBridge {
    connector: Arc<dyn KvCacheConnector>,
    /// Private runtime used to drive the async trait; `None` when inactive.
    runtime: Option<tokio::runtime::Runtime>,
    active: bool,
    /// Opaque model namespace for keys (never branched on).
    model_id: String,
    /// Tokens per cached chunk.
    chunk_size: usize,
    /// Layer span covered by stored KV (full model: `0..num_layers`).
    layer_range: Range<usize>,
    /// Priority applied to stored chunks.
    store_priority: CachePriority,
    /// Estimated recompute cost per token (ms), the fetch-vs-recompute baseline.
    recompute_ms_per_token: f64,
    stats: ConnectorStats,
}

impl ConnectorBridge {
    /// An inert bridge backed by [`NullConnector`]. No runtime is created and
    /// every operation is a no-op.
    pub(crate) fn null() -> Self {
        Self {
            connector: Arc::new(NullConnector),
            runtime: None,
            active: false,
            model_id: String::new(),
            chunk_size: onnx_genai_kv::DEFAULT_CHUNK_SIZE,
            layer_range: 0..0,
            store_priority: CachePriority::Session,
            recompute_ms_per_token: 0.0,
            stats: ConnectorStats::default(),
        }
    }

    /// An active bridge over `connector`. Builds a private current-thread
    /// runtime to drive the async trait from the synchronous engine.
    pub(crate) fn new(
        connector: Arc<dyn KvCacheConnector>,
        model_id: String,
        chunk_size: usize,
        layer_range: Range<usize>,
        store_priority: CachePriority,
        recompute_ms_per_token: f64,
    ) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build KV connector runtime: {e}"))?;
        Ok(Self {
            connector,
            runtime: Some(runtime),
            active: true,
            model_id,
            chunk_size: chunk_size.max(1),
            layer_range,
            store_priority,
            recompute_ms_per_token: recompute_ms_per_token.max(0.0),
            stats: ConnectorStats::default(),
        })
    }

    /// Whether a real (non-null) connector is configured.
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Accumulated connector activity since the last [`reset_stats`](Self::reset_stats).
    pub(crate) fn stats(&self) -> &ConnectorStats {
        &self.stats
    }

    /// Clear the accumulated stats (called at the start of each generation).
    pub(crate) fn reset_stats(&mut self) {
        self.stats = ConnectorStats::default();
    }

    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.runtime
            .as_ref()
            .expect("active connector bridge always has a runtime")
            .block_on(fut)
    }

    fn key_for(&self, chunk: &TokenChunk) -> KvCacheKey {
        chunk.to_key(self.model_id.clone(), self.layer_range.clone())
    }

    /// Report how far the connector could extend the prefix hit beyond the
    /// in-process boundary `in_process_hit`.
    ///
    /// Chunks `prompt_tokens` at the configured chunk size, looks up the
    /// complete chunks that begin at or after the boundary, and walks the
    /// contiguous run of resident chunks. A chunk only counts while fetching it
    /// is no more expensive than recomputing it — `estimated_load_ms` vs
    /// `num_tokens * recompute_ms_per_token` — so the connector never forces a
    /// slower path than plain prefill.
    ///
    /// TODO(K3-materialize): once [`onnx_genai_kv::FetchedKv`] / `KvTensorRef`
    /// carry a real device-tensor handle, `fetch` the hit chunks and copy them
    /// into the engine's paged KV cache here, then return `fetched_tokens > 0`
    /// so the caller can lengthen `prefix_cache_hit_len` and skip that prefill.
    /// Until then this reports the opportunity but does not alter prefill, so
    /// generation output is unaffected.
    pub(crate) fn lookup_extension(
        &mut self,
        prompt_tokens: &[TokenId],
        in_process_hit: usize,
    ) -> ConnectorLookupOutcome {
        if !self.active || in_process_hit >= prompt_tokens.len() {
            return ConnectorLookupOutcome::default();
        }
        // Contiguous extension is only possible from a chunk boundary; a hit
        // that starts mid-chunk would leave an un-served gap.
        if !in_process_hit.is_multiple_of(self.chunk_size) {
            return ConnectorLookupOutcome::default();
        }

        let chunks = chunk_tokens(prompt_tokens, self.chunk_size);
        let start_index = in_process_hit / self.chunk_size;
        if start_index >= chunks.len() {
            return ConnectorLookupOutcome::default();
        }
        // Only complete chunks are cacheable/servable; a trailing partial chunk
        // is always recomputed.
        let candidate_chunks: Vec<&TokenChunk> = chunks[start_index..]
            .iter()
            .take_while(|c| c.tokens.len() == self.chunk_size)
            .collect();
        if candidate_chunks.is_empty() {
            return ConnectorLookupOutcome::default();
        }
        let keys: Vec<KvCacheKey> = candidate_chunks.iter().map(|c| self.key_for(c)).collect();

        let connector = Arc::clone(&self.connector);
        let locations = match self.block_on(connector.lookup_batch(&keys)) {
            Ok(locations) => locations,
            Err(error) => {
                tracing::debug!(%error, "KV connector lookup_batch failed; recomputing prefix");
                return ConnectorLookupOutcome::default();
            }
        };

        self.stats.lookups += keys.len();

        let mut outcome = ConnectorLookupOutcome::default();
        for (location, key) in locations.iter().zip(&keys) {
            let Some(load_ms) = location_load_ms(location) else {
                break; // NotFound → prefix broken, stop extending.
            };
            let recompute_ms = key.num_tokens as f64 * self.recompute_ms_per_token;
            if load_ms > recompute_ms {
                break; // Cheaper to recompute this chunk than to fetch it.
            }
            outcome.chunk_hits += 1;
            outcome.would_extend_tokens += key.num_tokens as usize;
        }

        self.stats.chunk_hits += outcome.chunk_hits;
        self.stats.would_extend_tokens += outcome.would_extend_tokens;
        if outcome.would_extend_tokens > 0 {
            tracing::debug!(
                chunk_hits = outcome.chunk_hits,
                would_extend_tokens = outcome.would_extend_tokens,
                "KV connector could extend prefix reuse (materialization deferred in K3)"
            );
        }
        outcome
    }

    /// Push the complete chunks covering `tokens[..resident_len]` to the
    /// connector for future cross-session/cross-node reuse.
    ///
    /// `resident_len` must not exceed the number of tokens whose KV is actually
    /// resident. A trailing partial chunk is skipped (only whole chunks are
    /// cacheable). Storage is best-effort: a failing chunk is logged and
    /// skipped, never surfaced to inference.
    pub(crate) fn store_prefix(&mut self, tokens: &[TokenId], resident_len: usize) {
        if !self.active {
            return;
        }
        let resident_len = resident_len.min(tokens.len());
        if resident_len < self.chunk_size {
            return;
        }
        let chunks = chunk_tokens(&tokens[..resident_len], self.chunk_size);
        let connector = Arc::clone(&self.connector);
        let priority = self.store_priority;
        for chunk in &chunks {
            if chunk.tokens.len() != self.chunk_size {
                continue; // Skip the trailing partial chunk.
            }
            let entry = KvStoreEntry {
                key: self.key_for(chunk),
                kv_data: KvTensorRef::default(),
                priority,
                ttl: None,
            };
            match self.block_on(connector.store(entry)) {
                Ok(()) => self.stats.stores += 1,
                Err(error) => {
                    tracing::debug!(%error, "KV connector store failed; chunk not cached externally");
                }
            }
        }
    }
}

/// Load-cost of a location in ms, or `None` when the chunk is not cached.
fn location_load_ms(location: &KvCacheLocation) -> Option<f64> {
    match location {
        KvCacheLocation::LocalGpu { .. } => Some(0.0),
        KvCacheLocation::LocalCpu {
            estimated_load_ms, ..
        }
        | KvCacheLocation::LocalDisk {
            estimated_load_ms, ..
        }
        | KvCacheLocation::Remote {
            estimated_load_ms, ..
        } => Some(*estimated_load_ms),
        KvCacheLocation::NotFound => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_kv::{
        ConnectorCapabilities, ConnectorHealth, ConnectorResult, Device, FetchedKv, LocalTieredConfig,
        LocalTieredConnector,
    };
    use std::sync::Mutex;

    /// Records every call and answers lookups from a fixed resident set.
    #[derive(Default)]
    struct SpyConnector {
        /// Chunk hashes reported resident, with their load-cost in ms.
        resident: Mutex<std::collections::HashMap<u64, f64>>,
        stored: Mutex<Vec<KvCacheKey>>,
        lookups: Mutex<Vec<KvCacheKey>>,
    }

    impl SpyConnector {
        fn resident(entries: &[(u64, f64)]) -> Self {
            let spy = SpyConnector::default();
            let mut guard = spy.resident.lock().unwrap();
            for (hash, ms) in entries {
                guard.insert(*hash, *ms);
            }
            drop(guard);
            spy
        }
    }

    #[async_trait::async_trait]
    impl KvCacheConnector for SpyConnector {
        async fn lookup(&self, key: &KvCacheKey) -> ConnectorResult<KvCacheLocation> {
            self.lookups.lock().unwrap().push(key.clone());
            Ok(match self.resident.lock().unwrap().get(&key.chunk_hash) {
                Some(ms) if *ms == 0.0 => KvCacheLocation::LocalGpu { page_ids: vec![] },
                Some(ms) => KvCacheLocation::LocalCpu {
                    estimated_load_ms: *ms,
                    size_bytes: 0,
                },
                None => KvCacheLocation::NotFound,
            })
        }

        async fn store(&self, entry: KvStoreEntry) -> ConnectorResult<()> {
            self.stored.lock().unwrap().push(entry.key);
            Ok(())
        }

        async fn fetch(&self, _key: &KvCacheKey, _target: Device) -> ConnectorResult<FetchedKv> {
            Err(onnx_genai_kv::ConnectorError::NotFound)
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
                compression: vec![],
            }
        }
    }

    fn bridge_over(connector: Arc<dyn KvCacheConnector>, chunk_size: usize) -> ConnectorBridge {
        ConnectorBridge::new(
            connector,
            "test-model".to_string(),
            chunk_size,
            0..2,
            CachePriority::Session,
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn null_bridge_is_inert() {
        let mut bridge = ConnectorBridge::null();
        assert!(!bridge.is_active());
        let tokens: Vec<TokenId> = (0..1000).collect();
        let outcome = bridge.lookup_extension(&tokens, 0);
        assert_eq!(outcome, ConnectorLookupOutcome::default());
        bridge.store_prefix(&tokens, tokens.len());
        assert_eq!(bridge.stats(), &ConnectorStats::default());
    }

    #[test]
    fn store_prefix_pushes_only_complete_chunks() {
        let spy = Arc::new(SpyConnector::default());
        let mut bridge = bridge_over(spy.clone(), 4);
        // 10 tokens, chunk_size 4 => 2 complete chunks + 1 partial (skipped).
        let tokens: Vec<TokenId> = (0..10).collect();
        bridge.store_prefix(&tokens, tokens.len());
        assert_eq!(bridge.stats().stores, 2);
        assert_eq!(spy.stored.lock().unwrap().len(), 2);
    }

    #[test]
    fn store_prefix_respects_resident_len() {
        let spy = Arc::new(SpyConnector::default());
        let mut bridge = bridge_over(spy.clone(), 4);
        let tokens: Vec<TokenId> = (0..10).collect();
        // Only 4 tokens resident => exactly one full chunk.
        bridge.store_prefix(&tokens, 4);
        assert_eq!(bridge.stats().stores, 1);
    }

    #[test]
    fn lookup_extension_walks_contiguous_hits_from_boundary() {
        let tokens: Vec<TokenId> = (0..12).collect();
        let chunks = chunk_tokens(&tokens, 4);
        // Mark chunk 1 and chunk 2 resident (cheap), leave chunk 0 (already in
        // process) irrelevant. Boundary at 4 tokens => start at chunk index 1.
        let spy = Arc::new(SpyConnector::resident(&[
            (chunks[1].hash, 0.0),
            (chunks[2].hash, 1.0),
        ]));
        let mut bridge = bridge_over(spy, 4);
        let outcome = bridge.lookup_extension(&tokens, 4);
        assert_eq!(outcome.chunk_hits, 2);
        assert_eq!(outcome.would_extend_tokens, 8);
        // Materialization is deferred: no tokens actually fetched.
        assert_eq!(outcome.fetched_tokens, 0);
    }

    #[test]
    fn lookup_extension_stops_at_first_miss() {
        let tokens: Vec<TokenId> = (0..12).collect();
        let chunks = chunk_tokens(&tokens, 4);
        // Chunk 1 resident, chunk 2 missing => extension stops after one chunk.
        let spy = Arc::new(SpyConnector::resident(&[(chunks[1].hash, 0.0)]));
        let mut bridge = bridge_over(spy, 4);
        let outcome = bridge.lookup_extension(&tokens, 4);
        assert_eq!(outcome.chunk_hits, 1);
        assert_eq!(outcome.would_extend_tokens, 4);
    }

    #[test]
    fn lookup_extension_prefers_recompute_when_fetch_is_costlier() {
        let tokens: Vec<TokenId> = (0..8).collect();
        let chunks = chunk_tokens(&tokens, 4);
        // Load estimate 100ms/chunk but recompute is 4 tokens * 1.0 = 4ms.
        let spy = Arc::new(SpyConnector::resident(&[(chunks[1].hash, 100.0)]));
        let mut bridge = bridge_over(spy, 4);
        let outcome = bridge.lookup_extension(&tokens, 4);
        assert_eq!(outcome.chunk_hits, 0);
        assert_eq!(outcome.would_extend_tokens, 0);
    }

    #[test]
    fn lookup_extension_requires_chunk_aligned_boundary() {
        let tokens: Vec<TokenId> = (0..12).collect();
        let chunks = chunk_tokens(&tokens, 4);
        let spy = Arc::new(SpyConnector::resident(&[(chunks[1].hash, 0.0)]));
        let mut bridge = bridge_over(spy, 4);
        // Boundary 3 is not a multiple of chunk_size => no extension attempted.
        let outcome = bridge.lookup_extension(&tokens, 3);
        assert_eq!(outcome, ConnectorLookupOutcome::default());
        assert_eq!(bridge.stats().lookups, 0);
    }

    #[test]
    fn store_then_lookup_roundtrips_through_local_tiered() {
        let config = LocalTieredConfig {
            chunk_size: 4,
            page_size: 4,
            ..LocalTieredConfig::default()
        };
        let connector = Arc::new(LocalTieredConnector::new(config).unwrap());
        let mut bridge = bridge_over(connector, 4);
        let tokens: Vec<TokenId> = (0..12).collect();
        // Store the whole prefix, then a fresh request reuses tokens[4..].
        bridge.store_prefix(&tokens, tokens.len());
        assert_eq!(bridge.stats().stores, 3);

        let outcome = bridge.lookup_extension(&tokens, 4);
        // Chunks 1 and 2 are resident (hot => 0ms load) so both extend.
        assert_eq!(outcome.chunk_hits, 2);
        assert_eq!(outcome.would_extend_tokens, 8);
    }
}
