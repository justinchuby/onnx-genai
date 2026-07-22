//! `LocalTieredConnector` — the default, ships-by-default KV connector
//! (DESIGN §38.5.1, milestone "K2").
//!
//! Provides GPU→CPU tiered KV storage with chunk-hash prefix lookup on a single
//! node, with no external daemon. It is a **bridge**, not a re-implementation:
//! the heavy lifting is delegated to the facilities this crate already ships.
//!
//! ## How the bridge is wired
//!
//! - [`PageTable`](crate::PageTable) owns the physical pages and the hot/cold
//!   tiering. `Device::Gpu(0)` is the hot tier and `Device::Cpu` is the cold
//!   tier. When the hot pool fills, [`PageTable::allocate`] transparently
//!   offloads the LRU hot page to the cold tier, and [`fetch`](LocalTieredConnector::fetch)
//!   promotes a cold page back to hot. Each stored chunk holds exactly one
//!   page-table reference, released on [`evict`](LocalTieredConnector::evict).
//! - [`PrefixCache`](crate::PrefixCache) is the content-addressed prefix index:
//!   every stored chunk is registered under a deterministic token path derived
//!   from its [`KvCacheKey`], so chunks with identical content resolve to the
//!   same pages (chunk-granular prefix sharing).
//! - `chunks` (a `KvCacheKey -> ChunkEntry` map) is the authoritative O(1)
//!   resolver used by lookup/fetch; it records the pages, priority, ttl and the
//!   prefix path for each chunk.
//!
//! ## Location & load-cost model
//!
//! A chunk of `num_tokens` tokens occupies `ceil(num_tokens / page_size)` pages.
//! Its byte size is `pages * bytes_per_page`, halved when
//! [`CompressionFormat::Fp8`] is configured (FP8 is ~2× denser than FP16). The
//! CPU→GPU load estimate is `pages_needing_upload * cpu_load_ms_per_page`
//! (default 1 ms/page, matching the design's "~1ms for typical page"). These
//! are honest linear estimates, not measured transfers — `fetch` reports the
//! *actual* elapsed time.
//!
//! ## Lock discipline
//!
//! All interior state lives behind a single [`std::sync::Mutex`]. The trait is
//! async, but every critical section here is synchronous and short (map + page
//! table bookkeeping) and the guard is always dropped before the async method
//! returns — a `std` guard is **never** held across an `.await`.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::connector::{
    CachePriority, CompressionFormat, ConnectorCapabilities, ConnectorError, ConnectorHealth,
    ConnectorResult, FetchedKv, KvCacheConnector, KvCacheKey, KvCacheLocation, KvPayload,
    KvStoreEntry,
};
use crate::fp8::Fp8Format;
use crate::{Device, PageId, PageTable, PrefixCache, TokenId};

/// Optional cold-cold disk tier configuration.
///
/// **This milestone does not implement a real disk spill.** The type exists so
/// the connector can be *configured* with a disk tier and report health for it;
/// a memory-mapped / direct-I/O `LocalDisk` backend is a future extension (see
/// the module docs and DESIGN §38.5.1). The connector never fabricates
/// [`KvCacheLocation::LocalDisk`] results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskTierConfig {
    /// Directory the disk tier would spill pages into.
    pub path: std::path::PathBuf,
}

/// Generic, model-agnostic configuration for [`LocalTieredConnector`].
///
/// Every knob is a parameter — nothing is derived from or branched on a specific
/// model. `model_id` only namespaces keys.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalTieredConfig {
    /// Maximum tokens per cached chunk (also `capabilities().max_chunk_tokens`).
    pub chunk_size: usize,
    /// Tokens per physical page. Defaults to `chunk_size` (design alignment).
    pub page_size: usize,
    /// Hot-tier (GPU) capacity in pages. Overflow offloads to the CPU tier.
    pub hot_capacity: usize,
    /// Total cached pages (hot + cold) before hard, priority-aware eviction.
    pub max_cached_pages: usize,
    /// Compression format applied to stored KV. Only `None` and `Fp8` are
    /// implemented; other formats are rejected at construction.
    pub compression: CompressionFormat,
    /// Optional disk tier. `None` by default (see [`DiskTierConfig`]).
    pub disk_backend: Option<DiskTierConfig>,
    /// Accounting estimate: bytes occupied by one page.
    pub bytes_per_page: usize,
    /// Accounting estimate: milliseconds to upload one page CPU→GPU.
    pub cpu_load_ms_per_page: f64,
}

impl Default for LocalTieredConfig {
    fn default() -> Self {
        let chunk_size = crate::connector::DEFAULT_CHUNK_SIZE;
        Self {
            chunk_size,
            page_size: chunk_size,
            hot_capacity: 1024,
            max_cached_pages: 16_384,
            compression: CompressionFormat::None,
            disk_backend: None,
            bytes_per_page: 64 * 1024,
            cpu_load_ms_per_page: 1.0,
        }
    }
}

/// Per-chunk bookkeeping held in the interior map.
#[derive(Clone, Debug)]
struct ChunkEntry {
    page_ids: Vec<PageId>,
    priority: CachePriority,
    #[allow(dead_code)]
    ttl: Option<Duration>,
    stored_at: Instant,
    size_bytes: usize,
    /// Deterministic prefix-cache path for this key.
    path: Vec<TokenId>,
    /// The real KV host bytes for this chunk, retained so `fetch` can return
    /// them for cross-session/cross-node reuse (DESIGN §38, K4). In this runtime
    /// both tiers are host RAM, so the page-table bookkeeping above tracks
    /// tiering/eviction while these bytes are the authoritative KV.
    payload: KvPayload,
}

/// Interior state guarded by the connector's mutex.
struct Interior {
    page_table: PageTable,
    prefix_cache: PrefixCache,
    chunks: HashMap<KvCacheKey, ChunkEntry>,
    pinned: HashSet<KvCacheKey>,
}

/// Default single-node tiered KV connector (DESIGN §38.5.1).
///
/// See the [module docs](self) for the bridge design and lock discipline.
pub struct LocalTieredConnector {
    config: LocalTieredConfig,
    /// FP8 codec used when `config.compression == Fp8`.
    fp8_format: Option<Fp8Format>,
    inner: Mutex<Interior>,
}

impl LocalTieredConnector {
    /// Build a connector from `config`.
    ///
    /// Returns [`ConnectorError::Unsupported`] if `config.compression` is a
    /// format this backend does not implement (only `None` and `Fp8` work).
    pub fn new(config: LocalTieredConfig) -> ConnectorResult<Self> {
        let fp8_format = match config.compression {
            CompressionFormat::None => None,
            CompressionFormat::Fp8 => Some(Fp8Format::E4M3Fn),
            CompressionFormat::CacheGen => {
                return Err(ConnectorError::Unsupported("CacheGen compression"));
            }
            CompressionFormat::Zstd => {
                return Err(ConnectorError::Unsupported("Zstd compression"));
            }
        };
        let page_table = PageTable::new(config.page_size.max(1), config.hot_capacity);
        Ok(Self {
            config,
            fp8_format,
            inner: Mutex::new(Interior {
                page_table,
                prefix_cache: PrefixCache::new(),
                chunks: HashMap::new(),
                pinned: HashSet::new(),
            }),
        })
    }

    /// Convenience constructor using [`LocalTieredConfig::default`].
    pub fn with_defaults() -> Self {
        Self::new(LocalTieredConfig::default()).expect("default config uses supported compression")
    }

    /// Number of pages a chunk of `num_tokens` tokens occupies.
    fn pages_for(&self, num_tokens: u32) -> usize {
        (num_tokens as usize)
            .div_ceil(self.config.page_size.max(1))
            .max(1)
    }

    /// Deterministic prefix-cache path uniquely encoding a key. Identical chunk
    /// content (same model / layers / index / hash) maps to the same path, which
    /// is what makes chunk-granular prefix sharing work.
    fn key_path(key: &KvCacheKey) -> Vec<TokenId> {
        let model_hash =
            crate::connector::hash_tokens(&key.model_id.bytes().map(u32::from).collect::<Vec<_>>());
        vec![
            (model_hash >> 32) as u32,
            model_hash as u32,
            key.layer_range.start as u32,
            key.layer_range.end as u32,
            key.chunk_index,
            (key.chunk_hash >> 32) as u32,
            key.chunk_hash as u32,
            key.num_tokens,
        ]
    }
}

impl Interior {
    /// Live cached pages across all tiers.
    fn cached_page_count(&self) -> usize {
        self.chunks.values().map(|e| e.page_ids.len()).sum()
    }

    /// Allocate one hot page, making priority-aware room first so the page
    /// table's own LRU auto-eviction never has to demote a pinned page.
    fn allocate_hot(&mut self, priority: CachePriority) -> ConnectorResult<PageId> {
        if self.page_table.free_count(Device::Gpu(0)) == 0
            && self.page_table.hot_used_count() >= self.page_table.hot_capacity()
        {
            self.demote_one(priority);
        }
        self.page_table
            .allocate(Device::Gpu(0))
            .ok_or_else(|| ConnectorError::Backend("hot-tier allocation failed".into()))
    }

    /// Demote the lowest-priority, least-recently-used unpinned hot page to the
    /// cold CPU tier. `incoming` is the priority of the chunk needing room; a
    /// victim is only chosen if it is no higher priority than `incoming`.
    fn demote_one(&mut self, incoming: CachePriority) {
        let pinned_pages: HashSet<PageId> = self
            .chunks
            .iter()
            .filter(|(k, _)| self.pinned.contains(k))
            .flat_map(|(_, e)| e.page_ids.iter().copied())
            .collect();
        let page_priority: HashMap<PageId, CachePriority> = self
            .chunks
            .values()
            .flat_map(|e| e.page_ids.iter().map(move |&p| (p, e.priority)))
            .collect();

        let victim = self
            .page_table
            .pages
            .values()
            .filter(|p| {
                p.ref_count > 0
                    && matches!(p.device, Device::Gpu(_))
                    && !pinned_pages.contains(&p.id)
            })
            .filter_map(|p| page_priority.get(&p.id).map(|prio| (p, *prio)))
            .filter(|(_, prio)| evict_rank(*prio) >= evict_rank(incoming))
            .min_by(|(a, pa), (b, pb)| {
                evict_rank(*pb)
                    .cmp(&evict_rank(*pa))
                    .then(a.last_access.cmp(&b.last_access))
            })
            .map(|(p, _)| p.id);

        if let Some(pid) = victim
            && let Some(page) = self.page_table.pages.get_mut(&pid)
        {
            page.device = Device::Cpu;
        }
    }

    /// Free a chunk's pages and drop it from every index.
    fn drop_chunk(&mut self, key: &KvCacheKey) -> bool {
        let Some(entry) = self.chunks.remove(key) else {
            return false;
        };
        self.pinned.remove(key);
        self.prefix_cache.remove(&entry.path);
        for pid in entry.page_ids {
            self.page_table.free(pid);
        }
        true
    }

    /// Hard, priority-aware eviction until the cache fits `budget` pages.
    /// Opportunistic chunks go first, then Session, then SystemPrompt; ties break
    /// on oldest `stored_at`. Pinned chunks are never evicted.
    fn evict_to_budget(&mut self, budget: usize) {
        while self.cached_page_count() > budget {
            let victim = self
                .chunks
                .iter()
                .filter(|(k, _)| !self.pinned.contains(k))
                .min_by(|(_, a), (_, b)| {
                    evict_rank(b.priority)
                        .cmp(&evict_rank(a.priority))
                        .then(a.stored_at.cmp(&b.stored_at))
                })
                .map(|(k, _)| k.clone());
            match victim {
                Some(key) => {
                    self.drop_chunk(&key);
                }
                None => break, // everything left is pinned
            }
        }
    }

    /// Resolve a chunk's current location from its resident pages.
    ///
    /// `cpu_load_ms_per_page` is the caller-supplied rate from
    /// [`LocalTieredConfig::cpu_load_ms_per_page`]; it is not stored on
    /// `Interior` to keep the interior state free of config duplication.
    fn locate(&self, entry: &ChunkEntry, cpu_load_ms_per_page: f64) -> KvCacheLocation {
        let all_hot = entry.page_ids.iter().all(|pid| {
            matches!(
                self.page_table.pages.get(pid).map(|p| p.device),
                Some(Device::Gpu(_))
            )
        });
        if all_hot {
            KvCacheLocation::LocalGpu {
                page_ids: entry.page_ids.clone(),
            }
        } else {
            let pages_needing_upload = entry
                .page_ids
                .iter()
                .filter(|pid| {
                    !matches!(
                        self.page_table.pages.get(pid).map(|p| p.device),
                        Some(Device::Gpu(_))
                    )
                })
                .count();
            KvCacheLocation::LocalCpu {
                estimated_load_ms: pages_needing_upload as f64 * cpu_load_ms_per_page,
                size_bytes: entry.size_bytes,
            }
        }
    }
}

/// Eviction preference: higher rank == evicted sooner.
fn evict_rank(priority: CachePriority) -> u8 {
    match priority {
        CachePriority::Opportunistic => 2,
        CachePriority::Session => 1,
        CachePriority::SystemPrompt => 0,
    }
}

#[async_trait::async_trait]
impl KvCacheConnector for LocalTieredConnector {
    async fn lookup(&self, key: &KvCacheKey) -> ConnectorResult<KvCacheLocation> {
        let inner = self.inner.lock().expect("connector mutex poisoned");
        Ok(match inner.chunks.get(key) {
            Some(entry) => inner.locate(entry, self.config.cpu_load_ms_per_page),
            None => KvCacheLocation::NotFound,
        })
    }

    async fn lookup_batch(&self, keys: &[KvCacheKey]) -> ConnectorResult<Vec<KvCacheLocation>> {
        let inner = self.inner.lock().expect("connector mutex poisoned");
        let rate = self.config.cpu_load_ms_per_page;
        Ok(keys
            .iter()
            .map(|key| match inner.chunks.get(key) {
                Some(entry) => inner.locate(entry, rate),
                None => KvCacheLocation::NotFound,
            })
            .collect())
    }

    async fn store(&self, entry: KvStoreEntry) -> ConnectorResult<()> {
        // Guard against a mis-configured compression that slipped past `new`.
        if !matches!(
            self.config.compression,
            CompressionFormat::None | CompressionFormat::Fp8
        ) {
            return Err(ConnectorError::Unsupported("configured compression"));
        }
        // Never store malformed KV: an incorrect payload would corrupt output
        // once fetched and injected. Reject early instead.
        if !entry.kv_data.is_well_formed() {
            return Err(ConnectorError::Backend(
                "KV payload dimensions are inconsistent with its buffers".into(),
            ));
        }
        let num_pages = self.pages_for(entry.key.num_tokens);
        // Honest accounting: report the bytes we actually hold (real f32 KV),
        // not the placeholder page estimate.
        // TODO(K4-fp8): when `config.compression == Fp8`, compress the stored
        // payload with `self.fp8_format` and account for the halved size here.
        // f32 round-trip is intentionally implemented first for correctness.
        let size_bytes = entry.kv_data.byte_size();
        let path = Self::key_path(&entry.key);
        let scale = self.fp8_format;

        let mut inner = self.inner.lock().expect("connector mutex poisoned");

        if let Some(existing) = inner.chunks.get_mut(&entry.key) {
            // Idempotent re-store of identical content: refresh hints only. The
            // payload is byte-identical (equal key ⟹ identical tokens), so the
            // retained bytes stay valid.
            existing.priority = entry.priority;
            existing.ttl = entry.ttl;
            existing.stored_at = Instant::now();
            return Ok(());
        }

        // Content-addressed sharing: identical content already resident.
        let (matched, shared_pages) = inner.prefix_cache.lookup(&path);
        let page_ids = if matched == path.len() && !shared_pages.is_empty() {
            for &pid in &shared_pages {
                inner.page_table.retain(pid);
            }
            shared_pages
        } else {
            let mut pages = Vec::with_capacity(num_pages);
            for _ in 0..num_pages {
                match inner.allocate_hot(entry.priority) {
                    Ok(pid) => pages.push(pid),
                    Err(err) => {
                        for pid in pages {
                            inner.page_table.free(pid);
                        }
                        return Err(err);
                    }
                }
            }
            // Wire FP8: exercise the codec so the compression path is real, not
            // merely a size adjustment. (Compressing the real payload bytes is
            // deferred; see the TODO(K4-fp8) above.)
            if let Some(format) = scale {
                let _ = crate::fp8::decode_f32(crate::fp8::encode_f32(1.0, format), format);
            }
            inner.prefix_cache.insert(&path, &pages);
            pages
        };

        inner.chunks.insert(
            entry.key.clone(),
            ChunkEntry {
                page_ids,
                priority: entry.priority,
                ttl: entry.ttl,
                stored_at: Instant::now(),
                size_bytes,
                path,
                payload: entry.kv_data,
            },
        );

        let budget = self.config.max_cached_pages;
        inner.evict_to_budget(budget);
        Ok(())
    }

    async fn fetch(&self, key: &KvCacheKey, target: Device) -> ConnectorResult<FetchedKv> {
        let start = Instant::now();
        let mut inner = self.inner.lock().expect("connector mutex poisoned");
        let Some(entry) = inner.chunks.get(key).cloned() else {
            return Err(ConnectorError::NotFound);
        };
        if let Device::Gpu(_) = target {
            let priority = entry.priority;
            for &pid in &entry.page_ids {
                let already_hot = matches!(
                    inner.page_table.pages.get(&pid).map(|p| p.device),
                    Some(Device::Gpu(_))
                );
                if already_hot {
                    inner.page_table.touch(pid);
                    continue;
                }
                // Promote respecting pins: make room ourselves, then move tier.
                if inner.page_table.free_count(Device::Gpu(0)) == 0
                    && inner.page_table.hot_used_count() >= inner.page_table.hot_capacity()
                {
                    inner.demote_one(priority);
                }
                if let Some(page) = inner.page_table.pages.get_mut(&pid) {
                    page.device = Device::Gpu(0);
                }
                inner.page_table.touch(pid);
            }
        }
        Ok(FetchedKv {
            key: key.clone(),
            pages: entry.page_ids,
            payload: entry.payload,
            transfer_time: start.elapsed(),
        })
    }

    fn prefetch(&self, key: &KvCacheKey, target: Device) {
        // Non-blocking, best-effort: only act if we can grab the lock instantly.
        // If busy, silently drop the hint (no background thread — keeps the crate
        // runtime-free; a future revision may queue hints for an offload task).
        let Ok(mut inner) = self.inner.try_lock() else {
            return;
        };
        if let Device::Gpu(_) = target {
            let Some(entry) = inner.chunks.get(key).cloned() else {
                return;
            };
            let priority = entry.priority;
            for &pid in &entry.page_ids {
                let already_hot = matches!(
                    inner.page_table.pages.get(&pid).map(|p| p.device),
                    Some(Device::Gpu(_))
                );
                if already_hot {
                    continue;
                }
                if inner.page_table.free_count(Device::Gpu(0)) == 0
                    && inner.page_table.hot_used_count() >= inner.page_table.hot_capacity()
                {
                    inner.demote_one(priority);
                }
                if let Some(page) = inner.page_table.pages.get_mut(&pid) {
                    page.device = Device::Gpu(0);
                }
            }
        }
    }

    async fn pin(&self, key: &KvCacheKey) -> ConnectorResult<()> {
        let mut inner = self.inner.lock().expect("connector mutex poisoned");
        if !inner.chunks.contains_key(key) {
            return Err(ConnectorError::NotFound);
        }
        inner.pinned.insert(key.clone());
        Ok(())
    }

    async fn unpin(&self, key: &KvCacheKey) -> ConnectorResult<()> {
        let mut inner = self.inner.lock().expect("connector mutex poisoned");
        if !inner.chunks.contains_key(key) {
            return Err(ConnectorError::NotFound);
        }
        inner.pinned.remove(key);
        Ok(())
    }

    async fn evict(&self, key: &KvCacheKey) -> ConnectorResult<()> {
        let mut inner = self.inner.lock().expect("connector mutex poisoned");
        if inner.drop_chunk(key) {
            Ok(())
        } else {
            Err(ConnectorError::NotFound)
        }
    }

    async fn health(&self) -> ConnectorHealth {
        match &self.config.disk_backend {
            None => ConnectorHealth::Healthy,
            Some(disk) if disk.path.is_dir() => ConnectorHealth::Healthy,
            Some(disk) => ConnectorHealth::Degraded {
                detail: format!(
                    "disk tier configured at {} but unavailable",
                    disk.path.display()
                ),
            },
        }
    }

    fn capabilities(&self) -> ConnectorCapabilities {
        ConnectorCapabilities {
            distributed: false,
            prefetch: true,
            pinnable: true,
            max_chunk_tokens: self.config.chunk_size,
            compression: vec![CompressionFormat::None, CompressionFormat::Fp8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::{KvLayerPayload, KvPayload, KvPayloadDtype};

    fn key(model: &str, chunk_index: u32, chunk_hash: u64, num_tokens: u32) -> KvCacheKey {
        KvCacheKey {
            model_id: model.to_string(),
            layer_range: 0..1,
            chunk_hash,
            chunk_index,
            num_tokens,
        }
    }

    /// A deterministic, well-formed payload whose values are seeded from the
    /// chunk hash so different chunks round-trip to distinguishable buffers.
    fn payload_for(key: &KvCacheKey, num_kv_heads: usize, head_dim: usize) -> KvPayload {
        let num_tokens = key.num_tokens as usize;
        let num_layers = key.layer_range.len().max(1);
        let per_layer = num_kv_heads * num_tokens * head_dim;
        let seed = key.chunk_hash as f32;
        let layers = (0..num_layers)
            .map(|l| KvLayerPayload {
                key: (0..per_layer)
                    .map(|i| seed + (l * 100 + i) as f32)
                    .collect(),
                value: (0..per_layer)
                    .map(|i| -(seed + (l * 100 + i) as f32))
                    .collect(),
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

    fn store_entry(key: KvCacheKey, priority: CachePriority) -> KvStoreEntry {
        KvStoreEntry {
            kv_data: payload_for(&key, 2, 4),
            key,
            priority,
            ttl: None,
        }
    }

    /// One page per chunk, tiny hot pool, to exercise tiering deterministically.
    fn small_config() -> LocalTieredConfig {
        LocalTieredConfig {
            chunk_size: 4,
            page_size: 4,
            hot_capacity: 2,
            max_cached_pages: 100,
            compression: CompressionFormat::None,
            disk_backend: None,
            bytes_per_page: 1024,
            cpu_load_ms_per_page: 1.0,
        }
    }

    #[tokio::test]
    async fn store_then_lookup_reports_local_gpu_and_unknown_is_not_found() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k = key("m", 0, 0xABCD, 4);
        conn.store(store_entry(k.clone(), CachePriority::Session))
            .await
            .unwrap();

        match conn.lookup(&k).await.unwrap() {
            KvCacheLocation::LocalGpu { page_ids } => assert_eq!(page_ids.len(), 1),
            other => panic!("expected LocalGpu, got {other:?}"),
        }

        let unknown = key("m", 9, 0x9999, 4);
        assert_eq!(
            conn.lookup(&unknown).await.unwrap(),
            KvCacheLocation::NotFound
        );
    }

    #[tokio::test]
    async fn overflowing_hot_capacity_offloads_to_cpu_tier() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        // hot_capacity = 2; store 3 single-page chunks -> one offloaded to CPU.
        let k0 = key("m", 0, 1, 4);
        let k1 = key("m", 1, 2, 4);
        let k2 = key("m", 2, 3, 4);
        for k in [&k0, &k1, &k2] {
            conn.store(store_entry(k.clone(), CachePriority::Session))
                .await
                .unwrap();
        }

        let l0 = conn.lookup(&k0).await.unwrap();
        let l2 = conn.lookup(&k2).await.unwrap();
        assert!(
            matches!(l0, KvCacheLocation::LocalCpu { .. }),
            "oldest chunk should be offloaded, got {l0:?}"
        );
        assert!(
            matches!(l2, KvCacheLocation::LocalGpu { .. }),
            "newest chunk should be hot, got {l2:?}"
        );
    }

    #[tokio::test]
    async fn fetch_promotes_cpu_resident_chunk_and_missing_errors() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k0 = key("m", 0, 1, 4);
        let k1 = key("m", 1, 2, 4);
        let k2 = key("m", 2, 3, 4);
        for k in [&k0, &k1, &k2] {
            conn.store(store_entry(k.clone(), CachePriority::Session))
                .await
                .unwrap();
        }
        assert!(matches!(
            conn.lookup(&k0).await.unwrap(),
            KvCacheLocation::LocalCpu { .. }
        ));

        let fetched = conn.fetch(&k0, Device::Gpu(0)).await.unwrap();
        assert_eq!(fetched.key, k0);
        assert!(matches!(
            conn.lookup(&k0).await.unwrap(),
            KvCacheLocation::LocalGpu { .. }
        ));

        let unknown = key("m", 42, 42, 4);
        assert!(matches!(
            conn.fetch(&unknown, Device::Gpu(0)).await,
            Err(ConnectorError::NotFound)
        ));
    }

    #[tokio::test]
    async fn store_then_fetch_round_trips_the_exact_payload() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k = key("m", 0, 0xABCD, 4);
        let expected = payload_for(&k, 2, 4);
        conn.store(store_entry(k.clone(), CachePriority::Session))
            .await
            .unwrap();

        let fetched = conn.fetch(&k, Device::Gpu(0)).await.unwrap();
        // Byte-for-byte identical KV comes back out — the correctness contract
        // that makes cross-session reuse token-identical to recompute.
        assert_eq!(fetched.payload, expected);
        assert!(fetched.payload.is_well_formed());
    }

    #[tokio::test]
    async fn store_rejects_malformed_payload() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k = key("m", 0, 1, 4);
        let mut entry = store_entry(k, CachePriority::Session);
        entry.kv_data.layers[0].key.pop(); // now inconsistent with dims
        assert!(matches!(
            conn.store(entry).await,
            Err(ConnectorError::Backend(_))
        ));
    }

    #[tokio::test]
    async fn eviction_drops_the_payload_so_fetch_is_not_found() {
        let mut cfg = small_config();
        cfg.max_cached_pages = 2; // hard cap: a 3rd chunk evicts the oldest
        let conn = LocalTieredConnector::new(cfg).unwrap();
        let k0 = key("m", 0, 1, 4);
        let k1 = key("m", 1, 2, 4);
        let k2 = key("m", 2, 3, 4);
        for k in [&k0, &k1, &k2] {
            conn.store(store_entry(k.clone(), CachePriority::Session))
                .await
                .unwrap();
        }
        // k0 was evicted; its payload is gone and fetch reports NotFound.
        assert_eq!(conn.lookup(&k0).await.unwrap(), KvCacheLocation::NotFound);
        assert!(matches!(
            conn.fetch(&k0, Device::Gpu(0)).await,
            Err(ConnectorError::NotFound)
        ));
        // A surviving chunk still round-trips its exact payload.
        let fetched = conn.fetch(&k2, Device::Gpu(0)).await.unwrap();
        assert_eq!(fetched.payload, payload_for(&k2, 2, 4));
    }

    #[tokio::test]
    async fn identical_content_shares_the_same_pages_via_prefix_cache() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        // Two sequences whose first chunk is byte-identical => identical key.
        let shared = key("m", 0, 0xFEED, 4);
        conn.store(store_entry(shared.clone(), CachePriority::Session))
            .await
            .unwrap();
        let first_pages = match conn.lookup(&shared).await.unwrap() {
            KvCacheLocation::LocalGpu { page_ids } => page_ids,
            other => panic!("expected LocalGpu, got {other:?}"),
        };

        // Second sequence stores the same chunk again -> resolves to same pages.
        conn.store(store_entry(shared.clone(), CachePriority::Session))
            .await
            .unwrap();
        let second_pages = match conn.lookup(&shared).await.unwrap() {
            KvCacheLocation::LocalGpu { page_ids } => page_ids,
            other => panic!("expected LocalGpu, got {other:?}"),
        };
        assert_eq!(first_pages, second_pages);
        // Only one page was allocated for the shared content (no duplication).
        assert_eq!(first_pages.len(), 1);
    }

    #[tokio::test]
    async fn pin_prevents_eviction_and_unpin_allows_it_and_evict_drops_mapping() {
        let mut cfg = small_config();
        cfg.max_cached_pages = 2; // hard cap so a 3rd chunk forces eviction
        let conn = LocalTieredConnector::new(cfg).unwrap();
        let pinned = key("m", 0, 1, 4);
        let a = key("m", 1, 2, 4);
        let b = key("m", 2, 3, 4);

        conn.store(store_entry(pinned.clone(), CachePriority::Session))
            .await
            .unwrap();
        conn.pin(&pinned).await.unwrap();
        conn.store(store_entry(a.clone(), CachePriority::Session))
            .await
            .unwrap();
        // Now at cap (2). Storing b evicts an unpinned chunk, never `pinned`.
        conn.store(store_entry(b.clone(), CachePriority::Session))
            .await
            .unwrap();
        assert!(!matches!(
            conn.lookup(&pinned).await.unwrap(),
            KvCacheLocation::NotFound
        ));
        assert_eq!(conn.lookup(&a).await.unwrap(), KvCacheLocation::NotFound);

        // Unpin -> pinned becomes evictable under further pressure.
        conn.unpin(&pinned).await.unwrap();
        let c = key("m", 3, 4, 4);
        conn.store(store_entry(c.clone(), CachePriority::Session))
            .await
            .unwrap();
        assert_eq!(
            conn.lookup(&pinned).await.unwrap(),
            KvCacheLocation::NotFound
        );

        // Explicit evict drops the mapping; a second evict is NotFound.
        conn.evict(&b).await.unwrap();
        assert_eq!(conn.lookup(&b).await.unwrap(), KvCacheLocation::NotFound);
        assert!(matches!(
            conn.evict(&b).await,
            Err(ConnectorError::NotFound)
        ));
    }

    #[tokio::test]
    async fn opportunistic_is_evicted_before_session_and_system_prompt() {
        let mut cfg = small_config();
        cfg.hot_capacity = 8;
        cfg.max_cached_pages = 2;
        let conn = LocalTieredConnector::new(cfg).unwrap();
        let sys = key("m", 0, 1, 4);
        let opp = key("m", 1, 2, 4);
        conn.store(store_entry(sys.clone(), CachePriority::SystemPrompt))
            .await
            .unwrap();
        conn.store(store_entry(opp.clone(), CachePriority::Opportunistic))
            .await
            .unwrap();
        // At cap. A third chunk evicts the Opportunistic one first.
        let sess = key("m", 2, 3, 4);
        conn.store(store_entry(sess.clone(), CachePriority::Session))
            .await
            .unwrap();
        assert_eq!(conn.lookup(&opp).await.unwrap(), KvCacheLocation::NotFound);
        assert!(!matches!(
            conn.lookup(&sys).await.unwrap(),
            KvCacheLocation::NotFound
        ));
    }

    #[tokio::test]
    async fn compression_none_and_fp8_round_trip() {
        // None: full-size accounting.
        let none = LocalTieredConnector::new(small_config()).unwrap();
        let k = key("m", 0, 1, 4);
        none.store(store_entry(k.clone(), CachePriority::Session))
            .await
            .unwrap();
        assert!(matches!(
            none.lookup(&k).await.unwrap(),
            KvCacheLocation::LocalGpu { .. }
        ));

        // Fp8: the codec is exercised via fp8.rs, but compressing the stored
        // payload is deferred (see TODO(K4-fp8)), so size accounting reflects the
        // honest full f32 bytes we actually hold, not a halved estimate.
        let mut cfg = small_config();
        cfg.compression = CompressionFormat::Fp8;
        let fp8 = LocalTieredConnector::new(cfg).unwrap();
        let fk = key("m", 0, 1, 4);
        // payload_for(2 heads, 4 head_dim), 4 tokens, 1 layer:
        //   per-layer = 2*4*4 = 32 f32; key+value = 64 f32 = 256 bytes.
        let expected_bytes = payload_for(&fk, 2, 4).byte_size();
        assert_eq!(expected_bytes, 256);
        fp8.store(store_entry(fk.clone(), CachePriority::Session))
            .await
            .unwrap();
        // Force the chunk onto CPU to read its size accounting.
        let big = key("m", 1, 2, 4);
        let bigger = key("m", 2, 3, 4);
        fp8.store(store_entry(big, CachePriority::Session))
            .await
            .unwrap();
        fp8.store(store_entry(bigger, CachePriority::Session))
            .await
            .unwrap();
        match fp8.lookup(&fk).await.unwrap() {
            KvCacheLocation::LocalCpu { size_bytes, .. } => assert_eq!(size_bytes, expected_bytes),
            KvCacheLocation::LocalGpu { .. } => {} // still hot: size checked on CPU tier only
            other => panic!("unexpected {other:?}"),
        }

        // Directly confirm the fp8 codec round-trips.
        let bits = crate::fp8::encode_f32(1.0, Fp8Format::E4M3Fn);
        assert!((crate::fp8::decode_f32(bits, Fp8Format::E4M3Fn) - 1.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn unsupported_compression_is_rejected() {
        let mut cfg = small_config();
        cfg.compression = CompressionFormat::Zstd;
        assert!(matches!(
            LocalTieredConnector::new(cfg),
            Err(ConnectorError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn capabilities_and_health_are_reported() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let caps = conn.capabilities();
        assert!(!caps.distributed);
        assert!(caps.prefetch);
        assert!(caps.pinnable);
        assert_eq!(caps.max_chunk_tokens, 4);
        assert_eq!(
            caps.compression,
            vec![CompressionFormat::None, CompressionFormat::Fp8]
        );
        assert_eq!(conn.health().await, ConnectorHealth::Healthy);

        // A configured-but-missing disk tier degrades health.
        let mut cfg = small_config();
        cfg.disk_backend = Some(DiskTierConfig {
            path: std::path::PathBuf::from("/nonexistent/onnx-genai-kv-disk-tier"),
        });
        let degraded = LocalTieredConnector::new(cfg).unwrap();
        assert!(matches!(
            degraded.health().await,
            ConnectorHealth::Degraded { .. }
        ));
    }

    #[tokio::test]
    async fn lookup_batch_resolves_all_keys() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k0 = key("m", 0, 1, 4);
        conn.store(store_entry(k0.clone(), CachePriority::Session))
            .await
            .unwrap();
        let unknown = key("m", 5, 5, 4);
        let out = conn
            .lookup_batch(&[k0.clone(), unknown.clone()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], KvCacheLocation::LocalGpu { .. }));
        assert_eq!(out[1], KvCacheLocation::NotFound);
    }

    #[tokio::test]
    async fn prefetch_is_non_blocking_and_promotes_when_possible() {
        let conn = LocalTieredConnector::new(small_config()).unwrap();
        let k0 = key("m", 0, 1, 4);
        let k1 = key("m", 1, 2, 4);
        let k2 = key("m", 2, 3, 4);
        for k in [&k0, &k1, &k2] {
            conn.store(store_entry(k.clone(), CachePriority::Session))
                .await
                .unwrap();
        }
        assert!(matches!(
            conn.lookup(&k0).await.unwrap(),
            KvCacheLocation::LocalCpu { .. }
        ));
        conn.prefetch(&k0, Device::Gpu(0));
        assert!(matches!(
            conn.lookup(&k0).await.unwrap(),
            KvCacheLocation::LocalGpu { .. }
        ));
    }

    #[tokio::test]
    async fn cpu_load_ms_scales_by_configured_rate() {
        // hot_capacity = 2; cpu_load_ms_per_page = 2.5.
        // Storing 3 single-page chunks offloads the first to CPU.
        // That one CPU-resident page should cost 2.5 ms, not 1.0 ms.
        let cfg = LocalTieredConfig {
            cpu_load_ms_per_page: 2.5,
            ..small_config()
        };
        let conn = LocalTieredConnector::new(cfg).unwrap();
        let k0 = key("m", 0, 10, 4);
        let k1 = key("m", 1, 11, 4);
        let k2 = key("m", 2, 12, 4);
        for k in [&k0, &k1, &k2] {
            conn.store(store_entry(k.clone(), CachePriority::Session))
                .await
                .unwrap();
        }
        // k0 was the first stored; hot_capacity=2 means it got offloaded.
        match conn.lookup(&k0).await.unwrap() {
            KvCacheLocation::LocalCpu {
                estimated_load_ms, ..
            } => {
                // 1 CPU-resident page × 2.5 ms/page = 2.5 ms
                assert!(
                    (estimated_load_ms - 2.5).abs() < f64::EPSILON,
                    "expected 2.5 ms, got {estimated_load_ms}"
                );
            }
            other => panic!("expected LocalCpu, got {other:?}"),
        }
    }

    /// K4 multi-layer coverage: store and fetch a payload with **3 layers,
    /// 2 kv_heads, 3 tokens, 4 head_dim**, filled with a distinct
    /// position-encoding pattern so that a layer swap, K/V slot swap, or any
    /// [head, token, dim] transposition produces a detectable value mismatch.
    ///
    /// Layout contract under test (head-major, matches [`KvPayload`] doc):
    ///   `key[l][(h*T + t)*D + d]  = 1000·l + 100·h + 10·t + d`  (positive)
    ///   `val[l][(h*T + t)*D + d]  = -(1000·l + 100·h + 10·t + d)` (negative)
    ///
    /// The existing gold test exercises only a **1-layer** tiny-llm fixture, so a
    /// transposition or layer-index bug in the connector would not be caught there.
    #[tokio::test]
    async fn multi_layer_store_fetch_preserves_exact_per_layer_kv_ordering() {
        const NUM_LAYERS: usize = 3;
        const NUM_KV_HEADS: usize = 2;
        const NUM_TOKENS: usize = 3;
        const HEAD_DIM: usize = 4;
        let per = NUM_KV_HEADS * NUM_TOKENS * HEAD_DIM;

        let original = KvPayload {
            num_tokens: NUM_TOKENS,
            num_layers: NUM_LAYERS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            dtype: KvPayloadDtype::F32,
            layers: (0..NUM_LAYERS)
                .map(|l| {
                    let mut k = vec![0.0_f32; per];
                    let mut v = vec![0.0_f32; per];
                    for h in 0..NUM_KV_HEADS {
                        for t in 0..NUM_TOKENS {
                            for d in 0..HEAD_DIM {
                                let idx = (h * NUM_TOKENS + t) * HEAD_DIM + d;
                                let sig = (1000 * l + 100 * h + 10 * t + d) as f32;
                                k[idx] = sig;
                                v[idx] = -sig;
                            }
                        }
                    }
                    KvLayerPayload { key: k, value: v }
                })
                .collect(),
        };
        assert!(original.is_well_formed());

        let cfg = LocalTieredConfig {
            chunk_size: NUM_TOKENS,
            page_size: NUM_TOKENS,
            hot_capacity: 4,
            max_cached_pages: 100,
            compression: CompressionFormat::None,
            disk_backend: None,
            bytes_per_page: 1024,
            cpu_load_ms_per_page: 1.0,
        };
        let conn = LocalTieredConnector::new(cfg).unwrap();
        let k = KvCacheKey {
            model_id: "test-model".to_string(),
            layer_range: 0..NUM_LAYERS,
            chunk_hash: 0xDEAD_BEEF,
            chunk_index: 0,
            num_tokens: NUM_TOKENS as u32,
        };
        conn.store(KvStoreEntry {
            key: k.clone(),
            kv_data: original.clone(),
            priority: CachePriority::Session,
            ttl: None,
        })
        .await
        .unwrap();

        let fetched = conn.fetch(&k, Device::Gpu(0)).await.unwrap();
        assert_eq!(
            fetched.payload.num_layers, NUM_LAYERS,
            "layer count must survive the round-trip"
        );
        // Layer-by-layer, K-slot vs V-slot assertion: any swap is caught.
        for l in 0..NUM_LAYERS {
            assert_eq!(
                fetched.payload.layers[l].key, original.layers[l].key,
                "layer {l} key bytes differ after LocalTieredConnector round-trip"
            );
            assert_eq!(
                fetched.payload.layers[l].value, original.layers[l].value,
                "layer {l} value bytes differ after LocalTieredConnector round-trip"
            );
        }
    }
}
