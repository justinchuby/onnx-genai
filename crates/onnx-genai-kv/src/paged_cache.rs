//! Paged KV cache implementation.

use crate::{
    CacheCheckpoint, Device, EvictionPolicy, KvCacheOps, KvError, SequenceId,
    page_table::{KvQuantConfig, LayerTensorConfig, PageId, PageTable, PageTensorConfig},
};
use onnx_genai_metadata::KvCacheSpec;

/// Borrowed per-layer K/V tensors for one token.
///
/// `key` and `value` must each contain `num_kv_heads * head_dim` f32 values,
/// laid out as `[num_kv_heads, head_dim]`.
pub struct LayerKv<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
}

/// Materialized K/V tensors for one layer over a sequence.
///
/// `key` and `value` are contiguous f32 buffers with shape
/// `[num_kv_heads, sequence_len, head_dim]` using this layer's own geometry.
/// Different layers may declare different `num_kv_heads`/`head_dim` (e.g.
/// Gemma-4 E2B sliding vs full layers), so the geometry is carried per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedLayerKv {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    /// This layer's KV head count.
    pub num_kv_heads: usize,
    /// This layer's head dimension.
    pub head_dim: usize,
}

/// Materialized K/V tensors for all layers over a sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedKv {
    /// Absolute position of the first *window* token in these tensors.
    ///
    /// With attention sinks (`sink_len > 0`) the buffer holds `sink_len`
    /// pinned tokens at absolute positions `[0, sink_len)` followed by the
    /// window tokens starting at `start_position`; the absolute positions are
    /// therefore discontinuous. Without sinks the buffer is contiguous from
    /// `start_position`.
    pub start_position: usize,
    /// Number of leading attention-sink tokens in the buffer (0 if contiguous).
    pub sink_len: usize,
    pub sequence_len: usize,
    /// Per-layer materialized K/V. Each layer carries its own
    /// `num_kv_heads`/`head_dim`, so heterogeneous geometry (e.g. Gemma-4 E2B
    /// sliding vs full head_dim) round-trips without a uniform assumption.
    pub layers: Vec<MaterializedLayerKv>,
}

/// Paged KV cache manager.
#[derive(Clone)]
pub struct PagedKvCache {
    pub page_table: PageTable,
    next_seq_id: SequenceId,
}

impl PagedKvCache {
    pub fn new(page_size: usize, num_gpu_pages: usize) -> Self {
        Self {
            page_table: PageTable::new(page_size, num_gpu_pages),
            next_seq_id: 0,
        }
    }

    pub fn new_with_tensor_config(config: PageTensorConfig, num_gpu_pages: usize) -> Self {
        Self {
            page_table: PageTable::new_with_tensor_config(
                config.page_size,
                num_gpu_pages,
                Some(config),
            ),
            next_seq_id: 0,
        }
    }

    /// Create a tensor cache with a per-layer key/value precision policy.
    pub fn new_with_quant_config(
        config: PageTensorConfig,
        quant_config: KvQuantConfig,
        num_gpu_pages: usize,
    ) -> Result<Self, KvError> {
        Ok(Self {
            page_table: PageTable::new_with_quant_config(
                config.page_size,
                num_gpu_pages,
                config,
                quant_config,
            )?,
            next_seq_id: 0,
        })
    }

    /// Create a tensor cache with **heterogeneous** per-layer KV geometry.
    ///
    /// Each entry in `layer_configs` declares that layer's `num_kv_heads`/
    /// `head_dim`; layers may differ (e.g. Gemma-4 E2B sliding vs full
    /// layers). `page_size` is tokens-per-page and `dtype` the KV precision.
    pub fn new_with_layer_tensor_configs(
        page_size: usize,
        dtype: crate::KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        num_gpu_pages: usize,
    ) -> Self {
        Self {
            page_table: PageTable::new_with_layer_configs(
                page_size,
                num_gpu_pages,
                dtype,
                layer_configs,
            ),
            next_seq_id: 0,
        }
    }

    /// Heterogeneous per-layer geometry whose storage is granted by a memory
    /// governor.
    ///
    /// The pool is planned, leased, and only then allocated, so it can never
    /// occupy more than it was granted. A tier with insufficient room refuses
    /// rather than returning a smaller pool: a pool that quietly came back
    /// short would fail later, mid-generation, when mirroring ran dry and pages
    /// claimed KV that was never written.
    pub fn new_leased(
        page_size: usize,
        dtype: crate::KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        num_gpu_pages: usize,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<Self, KvError> {
        Ok(Self {
            page_table: PageTable::new_leased(
                page_size,
                num_gpu_pages,
                dtype,
                layer_configs,
                governor,
                tier,
                holder,
            )?,
            next_seq_id: 0,
        })
    }

    /// Heterogeneous per-layer geometry with an explicit KV precision policy.
    pub fn new_with_layer_quant_config(
        page_size: usize,
        dtype: crate::KvDType,
        layer_configs: Vec<LayerTensorConfig>,
        quant_config: KvQuantConfig,
        num_gpu_pages: usize,
    ) -> Result<Self, KvError> {
        Ok(Self {
            page_table: PageTable::new_with_layer_quant_config(
                page_size,
                num_gpu_pages,
                dtype,
                layer_configs,
                quant_config,
            )?,
            next_seq_id: 0,
        })
    }

    /// Create a tensor cache using the KV precision policy in model metadata.
    pub fn new_with_metadata(
        config: PageTensorConfig,
        spec: &KvCacheSpec,
        num_gpu_pages: usize,
    ) -> Result<Self, KvError> {
        let quant_config = KvQuantConfig::from_metadata(spec, config.num_layers)?;
        Self::new_with_quant_config(config, quant_config, num_gpu_pages)
    }

    /// Create a new sequence, returns its ID.
    /// Bytes this pool holds a governor grant for, or `None` if it was built
    /// without one.
    ///
    /// The pool leases its whole capacity at construction, so this is a
    /// constant for the life of the cache rather than a running total. It is
    /// what admission must add back when it asks the ledger how much room is
    /// left for KV: the pool's own grant is not competition for the sequences
    /// that will be served out of it.
    pub fn pool_lease_bytes(&self) -> Option<u64> {
        self.page_table.leased_bytes()
    }

    pub fn create_sequence(&mut self) -> SequenceId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        self.page_table.create_sequence(id);
        id
    }

    /// Append one token of per-layer K/V tensors at the sequence tail.
    pub fn append_token_kv(
        &mut self,
        seq: SequenceId,
        layers: &[LayerKv<'_>],
    ) -> Result<usize, KvError> {
        let position = self.len(seq)?;
        self.write_token_kv(seq, position, layers)?;
        Ok(position)
    }

    /// Write one token of per-layer K/V tensors at `position`.
    ///
    /// `position` may be exactly the current sequence length (append) or may
    /// rewrite an existing token. Rewriting a shared page performs page-level
    /// Copy-on-Write before mutation.
    pub fn write_token_kv(
        &mut self,
        seq: SequenceId,
        position: usize,
        layers: &[LayerKv<'_>],
    ) -> Result<(), KvError> {
        if self.page_table.tensor_config.is_none() {
            return Err(KvError::TensorStorageNotConfigured);
        }
        // Per-layer geometry: each layer may declare its own num_kv_heads/head_dim.
        let layer_configs = self.page_table.layer_configs.clone();
        let page_size = self.page_table.page_size;
        self.validate_layers(&layer_configs, layers)?;

        let len = self.len(seq)?;
        let start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        if position >= sink && position < start {
            return Err(KvError::PositionEvicted {
                position,
                retained_start: start,
            });
        }
        if position > len {
            return Err(KvError::InvalidPosition {
                position,
                length: len,
            });
        }

        let retained_position = self.buffer_index(seq, position)?;
        let page_index = retained_position / self.page_table.page_size;
        let token_offset = retained_position % self.page_table.page_size;
        let page_id = self.ensure_page_for_write(seq, page_index)?;
        self.page_table.promote_to_hot(page_id)?;

        {
            let page = self
                .page_table
                .pages
                .get_mut(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?;
            for (layer_idx, layer) in layers.iter().enumerate() {
                let geom = layer_configs[layer_idx];
                for head in 0..geom.num_kv_heads {
                    let src = head * geom.head_dim;
                    let key = &layer.key[src..src + geom.head_dim];
                    let value = &layer.value[src..src + geom.head_dim];
                    page.write_head_token(
                        page_size,
                        geom.head_dim,
                        layer_idx * 2,
                        head,
                        token_offset,
                        key,
                    )?;
                    page.write_head_token(
                        page_size,
                        geom.head_dim,
                        layer_idx * 2 + 1,
                        head,
                        token_offset,
                        value,
                    )?;
                }
            }
            page.filled = page.filled.max(token_offset + 1);
        }
        self.page_table.touch(page_id);

        if position == len {
            self.page_table.set_sequence_len(seq, len + 1);
        }
        Ok(())
    }

    /// Materialize a sequence's paged K/V data into contiguous per-layer buffers.
    pub fn materialize_sequence(&self, seq: SequenceId) -> Result<MaterializedKv, KvError> {
        self.materialize_sequence_to(seq, self.len(seq)?)
    }

    /// Materialize a sequence's paged K/V as it would look after rewinding to
    /// `end`, without touching the sequence.
    ///
    /// Exists so a caller wanting "rewind, then read" does not have to mutate
    /// first and hope the read succeeds. The alternative was cloning the entire
    /// page pool to rewind a copy, which duplicates every page's storage —
    /// transiently doubling KV memory to answer a question about one sequence.
    ///
    /// **Mirrors only rewinds that stay at or above the pinned sink prefix.**
    /// A real [`KvCacheOps::rewind_to`] below the sink resets the sequence's
    /// window bookkeeping and returns tokens `[0, end)`; this function refuses
    /// that case rather than reporting a view it would not produce. Refusing is
    /// the safe direction — it never green-lights a position a rewind would
    /// reject — but it is a narrower contract than the name suggests, so it is
    /// stated rather than left to be discovered.
    pub fn materialize_sequence_to(
        &self,
        seq: SequenceId,
        end: usize,
    ) -> Result<MaterializedKv, KvError> {
        if self.page_table.tensor_config.is_none() {
            return Err(KvError::TensorStorageNotConfigured);
        }
        let layer_configs = &self.page_table.layer_configs;
        let page_size = self.page_table.page_size;
        let start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        let current = self.len(seq)?;
        if end > current {
            return Err(KvError::InvalidPosition {
                position: end,
                length: current,
            });
        }
        // Rewinding below the pinned sink prefix resets the sequence's window
        // bookkeeping, which this read cannot reproduce without mutating. Say so
        // rather than returning a view that does not match what the rewind would
        // leave behind.
        if sink > 0 && end < sink {
            return Err(KvError::RewindBelowSinkNotMaterializable {
                position: end,
                sink_len: sink,
            });
        }
        // Reading below the retained window would silently produce zeros for
        // tokens that were evicted, so it is refused the same way a rewind to
        // that position is.
        if end < start {
            return Err(KvError::PositionEvicted {
                position: end,
                retained_start: start,
            });
        }
        // Contiguous buffer holds the pinned sink prefix followed by the window.
        let len = sink + (end - start);
        let pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        let mut layers = layer_configs
            .iter()
            .map(|geom| {
                let per_layer_len = geom.num_kv_heads * len * geom.head_dim;
                MaterializedLayerKv {
                    key: vec![0.0; per_layer_len],
                    value: vec![0.0; per_layer_len],
                    num_kv_heads: geom.num_kv_heads,
                    head_dim: geom.head_dim,
                }
            })
            .collect::<Vec<_>>();

        for token_pos in 0..len {
            let page_index = token_pos / page_size;
            let token_offset = token_pos % page_size;
            let page_id = pages[page_index];
            let page = self
                .page_table
                .pages
                .get(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?;
            for (layer_idx, layer_out) in layers.iter_mut().enumerate() {
                let head_dim = layer_out.head_dim;
                for head in 0..layer_out.num_kv_heads {
                    for dim in 0..head_dim {
                        let dst = (head * len + token_pos) * head_dim + dim;
                        layer_out.key[dst] = page.value_at_slot(
                            page_size,
                            head_dim,
                            layer_idx * 2,
                            head,
                            token_offset,
                            dim,
                        )?;
                        layer_out.value[dst] = page.value_at_slot(
                            page_size,
                            head_dim,
                            layer_idx * 2 + 1,
                            head,
                            token_offset,
                            dim,
                        )?;
                    }
                }
            }
        }

        // Per-layer geometry lives on each `MaterializedLayerKv`; the
        // materialized cache no longer carries a uniform num_kv_heads/head_dim.
        Ok(MaterializedKv {
            start_position: start,
            sink_len: sink,
            sequence_len: len,
            layers,
        })
    }

    /// Promote the sequence's pages to HOT, then materialize K/V data.
    pub fn materialize_sequence_promoting(
        &mut self,
        seq: SequenceId,
    ) -> Result<MaterializedKv, KvError> {
        let start = self.retained_start(seq)?;
        let len = self.len(seq)?;
        self.prefetch(seq, start, len)?;
        self.materialize_sequence(seq)
    }

    /// Absolute range of KV positions currently retained for `seq`.
    pub fn retained_range(&self, seq: SequenceId) -> Result<std::ops::Range<usize>, KvError> {
        Ok(self.retained_start(seq)?..self.len(seq)?)
    }

    /// Number of KV tokens currently retained for `seq`.
    pub fn retained_len(&self, seq: SequenceId) -> Result<usize, KvError> {
        let range = self.retained_range(seq)?;
        Ok(range.end - range.start)
    }

    /// Free complete leading pages that are older than the sliding window.
    ///
    /// The logical sequence length remains absolute, while the retained start
    /// advances by page-sized increments. At most `window_size + page_size - 1`
    /// tokens remain because a partially overlapping page is preserved.
    pub fn apply_sliding_window(
        &mut self,
        seq: SequenceId,
        window_size: usize,
    ) -> Result<usize, KvError> {
        if window_size == 0 {
            return Err(KvError::InvalidWindowSize);
        }
        let start = self.retained_start(seq)?;
        let end = self.len(seq)?;
        let keep_from = end.saturating_sub(window_size);
        let pages_to_free = keep_from
            .saturating_sub(start)
            .checked_div(self.page_table.page_size)
            .unwrap_or(0);
        if pages_to_free == 0 {
            return Ok(0);
        }

        let removed = {
            let pages = self
                .page_table
                .sequences
                .get_mut(&seq)
                .ok_or(KvError::SequenceNotFound(seq))?;
            pages
                .drain(..pages_to_free.min(pages.len()))
                .collect::<Vec<_>>()
        };
        for page_id in &removed {
            self.page_table.free(*page_id);
        }
        self.page_table
            .set_sequence_start(seq, start + removed.len() * self.page_table.page_size);
        Ok(removed.len())
    }

    /// Sink-aware sliding window: retain a pinned prefix of attention-sink
    /// tokens *and* the most recent `window_size` tokens, evicting the pages in
    /// between (StreamingLLM, DESIGN §40.4).
    ///
    /// Sink retention is page-granular: the first `ceil(sink_tokens/page_size)`
    /// pages are pinned and never evicted. The retained set becomes the disjoint
    /// union `[0, sink_len) ∪ [window_start, len)`, stored contiguously as
    /// `[sink pages | window pages]`. With `sink_tokens == 0` this is exactly
    /// [`apply_sliding_window`]. Returns the number of pages freed.
    ///
    /// The absolute positions of the two runs are discontinuous; RoPE models
    /// remain correct because each token's positional embedding derives from its
    /// absolute position, not its buffer index (DESIGN §40.8). Feeding these
    /// discontinuous positions into a contiguous ORT past/present graph requires
    /// explicit `position_ids` support and is out of scope here — see the crate
    /// docs for the runtime boundary.
    pub fn apply_sliding_window_with_sinks(
        &mut self,
        seq: SequenceId,
        window_size: usize,
        sink_tokens: usize,
    ) -> Result<usize, KvError> {
        if window_size == 0 {
            return Err(KvError::InvalidWindowSize);
        }
        if sink_tokens == 0 {
            return self.apply_sliding_window(seq, window_size);
        }

        let page_size = self.page_table.page_size;
        let sink_pages = sink_tokens.div_ceil(page_size);
        let sink_len_target = sink_pages * page_size;
        let end = self.len(seq)?;
        let keep_from = end.saturating_sub(window_size);

        // Window abuts or overlaps the sink prefix: everything is retained
        // contiguously, so there is no gap to open.
        if keep_from <= sink_len_target {
            return Ok(0);
        }

        let sink_active = self.sink_len(seq)? > 0;
        let cur_window_start = if sink_active {
            self.retained_start(seq)?
        } else {
            // First activation: window pages currently begin right after the
            // soon-to-be-pinned sink pages.
            //
            // Validate the first-activation invariant (debug builds only):
            //  1. The sequence must already hold at least `sink_pages` allocated
            //     pages so the sink prefix can be pinned without additional
            //     allocation.
            //  2. The candidate window start must not regress into the sink
            //     region (already guaranteed by the `keep_from <= sink_len_target`
            //     guard above, but made explicit here for auditing).
            let page_count = self.page_table.get_sequence(seq).map_or(0, |p| p.len());
            debug_assert!(
                page_count >= sink_pages,
                "SWA sink first-activation: sequence has only {page_count} page(s) \
                 but sink_tokens={sink_tokens} requires {sink_pages} sink page(s) \
                 (page_size={page_size}); the sequence must have advanced past the \
                 sink boundary before sinks can activate"
            );
            debug_assert!(
                keep_from >= sink_len_target,
                "SWA sink first-activation: window keep_from ({keep_from}) precedes \
                 the pinned sink boundary ({sink_len_target}); this case must be \
                 caught by the no-gap guard above \
                 (sink_tokens={sink_tokens}, page_size={page_size})"
            );
            sink_len_target
        };
        let new_window_start = (keep_from / page_size) * page_size;
        if new_window_start <= cur_window_start {
            // Ensure sink bookkeeping is set even when nothing new is evicted.
            self.page_table.set_sequence_sink_len(seq, sink_len_target);
            self.page_table.set_sequence_start(seq, cur_window_start);
            return Ok(0);
        }

        let evict_pages = (new_window_start - cur_window_start) / page_size;
        let removed = {
            let pages = self
                .page_table
                .sequences
                .get_mut(&seq)
                .ok_or(KvError::SequenceNotFound(seq))?;
            let window_page_count = pages.len().saturating_sub(sink_pages);
            // Always keep at least the final window page.
            let evict = evict_pages.min(window_page_count.saturating_sub(1));
            pages
                .drain(sink_pages..sink_pages + evict)
                .collect::<Vec<_>>()
        };
        for page_id in &removed {
            self.page_table.free(*page_id);
        }
        self.page_table.set_sequence_sink_len(seq, sink_len_target);
        self.page_table
            .set_sequence_start(seq, cur_window_start + removed.len() * page_size);
        Ok(removed.len())
    }

    /// Evict pages to free memory. Returns number of pages freed.
    pub fn evict(&mut self, _policy: EvictionPolicy, _target: usize) -> usize {
        match _policy {
            EvictionPolicy::Lru | EvictionPolicy::Priority | EvictionPolicy::LayerAware => {
                let mut evicted = 0;
                for _ in 0.._target {
                    if self.page_table.evict_lru_hot(None).is_ok() {
                        evicted += 1;
                    } else {
                        break;
                    }
                }
                evicted
            }
        }
    }

    /// Promote all pages backing a sequence range to the hot tier.
    pub fn prefetch(
        &mut self,
        seq: SequenceId,
        start: usize,
        end: usize,
    ) -> Result<usize, KvError> {
        let retained_start = self.retained_start(seq)?;
        let len = self.len(seq)?;
        if start < retained_start {
            return Err(KvError::PositionEvicted {
                position: start,
                retained_start,
            });
        }
        if start > end || end > len {
            return Err(KvError::InvalidPosition {
                position: end,
                length: len,
            });
        }
        if start == end {
            return Ok(0);
        }
        let page_size = self.page_table.page_size;
        let first_page = self.buffer_index(seq, start)? / page_size;
        let last_page = self.buffer_index(seq, end - 1)? / page_size;
        let page_ids = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?[first_page..=last_page]
            .to_vec();
        let mut promoted = 0;
        for page_id in page_ids {
            let was_cold = self
                .page_table
                .pages
                .get(&page_id)
                .is_some_and(|page| !matches!(page.residency(), Device::Gpu(_)));
            self.page_table.promote_to_hot(page_id)?;
            if was_cold {
                promoted += 1;
            }
        }
        Ok(promoted)
    }

    /// Preempt `seq`: evict all pages it exclusively owns from the hot tier to
    /// the cold CPU tier, releasing hot residency without dropping any KV.
    ///
    /// This is the engine-side execution of a scheduler
    /// `ScheduleDecision::preempt` entry. Each page is copied into a cold-tier
    /// store transactionally, so [`restore_sequence`](Self::restore_sequence)
    /// brings back byte-identical KV and a preempted-then-restored sequence
    /// decodes the same tokens as if it had never been preempted. Both stores
    /// remain host-backed emulation until Stage 3. Returns pages demoted.
    pub fn preempt_sequence(&mut self, seq: SequenceId) -> Result<usize, KvError> {
        // Validate the sequence exists so a bogus id surfaces as an error
        // rather than a silent no-op.
        self.len(seq)?;
        self.page_table.evict_sequence_to_cold(seq)
    }

    /// Restore `seq`: promote every page backing its retained range back to the
    /// hot tier so decoding can resume. Inverse of
    /// [`preempt_sequence`](Self::preempt_sequence) and the engine-side
    /// execution of a scheduler `ScheduleDecision::swap_in` entry. Returns the
    /// number of pages promoted.
    pub fn restore_sequence(&mut self, seq: SequenceId) -> Result<usize, KvError> {
        let start = self.retained_start(seq)?;
        let end = self.len(seq)?;
        self.prefetch(seq, start, end)
    }

    /// Number of pages backing `seq` currently resident on the hot tier.
    pub fn sequence_hot_pages(&self, seq: SequenceId) -> usize {
        self.page_table.sequence_hot_pages(seq)
    }
    fn validate_layers(
        &self,
        layer_configs: &[LayerTensorConfig],
        layers: &[LayerKv<'_>],
    ) -> Result<(), KvError> {
        if layers.len() != layer_configs.len() {
            return Err(KvError::InvalidTensorShape("wrong number of layers"));
        }
        for (layer, geom) in layers.iter().zip(layer_configs) {
            let expected = geom.f32_len_per_token();
            if layer.key.len() != expected || layer.value.len() != expected {
                return Err(KvError::InvalidTensorShape(
                    "layer key/value length must be num_kv_heads * head_dim",
                ));
            }
        }
        Ok(())
    }

    fn ensure_page_for_write(
        &mut self,
        seq: SequenceId,
        page_index: usize,
    ) -> Result<PageId, KvError> {
        let current_pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?
            .to_vec();

        if let Some(&page_id) = current_pages.get(page_index) {
            let is_shared = self
                .page_table
                .pages
                .get(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?
                .ref_count
                > 1;
            if !is_shared {
                self.page_table.promote_to_hot(page_id)?;
                return Ok(page_id);
            }

            let mut checkpoint = self
                .page_table
                .allocation_checkpoint(page_id, Device::Gpu(0))?;
            let new_page_id =
                self.page_table
                    .allocate(Device::Gpu(0))
                    .ok_or_else(|| KvError::OutOfMemory {
                        needed: 1,
                        available: self.page_table.free_count(Device::Gpu(0)),
                    })?;
            let old_storage = {
                let old = self
                    .page_table
                    .pages
                    .get(&page_id)
                    .ok_or(KvError::PageNotFound(page_id))?;
                (old.clone_physical_store(), old.filled)
            };
            let copy_result = {
                let new_page = self
                    .page_table
                    .pages
                    .get_mut(&new_page_id)
                    .ok_or(KvError::PageNotFound(new_page_id))?;
                let result = new_page.copy_physical_store_from(old_storage.0.as_ref());
                if result.is_ok() {
                    new_page.filled = old_storage.1;
                }
                result
            };
            if let Err(copy_error) = copy_result {
                self.page_table
                    .rollback_allocation(&mut checkpoint, new_page_id)?;
                return Err(copy_error);
            }
            if let Err(commit_error) = self.page_table.commit_allocation(&mut checkpoint) {
                self.page_table
                    .rollback_allocation(&mut checkpoint, new_page_id)?;
                return Err(commit_error);
            }
            self.page_table.replace_page(seq, page_index, new_page_id);
            self.page_table.free(page_id);
            return Ok(new_page_id);
        }

        if page_index != current_pages.len() {
            return Err(KvError::InvalidPosition {
                position: page_index * self.page_table.page_size,
                length: current_pages.len() * self.page_table.page_size,
            });
        }

        let page_id =
            self.page_table
                .allocate(Device::Gpu(0))
                .ok_or_else(|| KvError::OutOfMemory {
                    needed: 1,
                    available: self.page_table.free_count(Device::Gpu(0)),
                })?;
        self.page_table.push_page(seq, page_id);
        Ok(page_id)
    }

    fn retained_start(&self, seq: SequenceId) -> Result<usize, KvError> {
        self.page_table
            .sequence_start(seq)
            .ok_or(KvError::SequenceNotFound(seq))
    }

    /// Number of pinned leading attention-sink tokens for `seq` (0 if none).
    fn sink_len(&self, seq: SequenceId) -> Result<usize, KvError> {
        self.page_table
            .sequence_sink_len(seq)
            .ok_or(KvError::SequenceNotFound(seq))
    }

    /// Map an absolute token position (in the retained set) to its index in the
    /// contiguous `[sink pages | window pages]` buffer.
    ///
    /// Positions inside the pinned sink prefix map to themselves; positions in
    /// the window run are shifted so the window follows the sinks. Callers must
    /// ensure `position` is not in the evicted gap `[sink_len, window_start)`.
    fn buffer_index(&self, seq: SequenceId, position: usize) -> Result<usize, KvError> {
        let sink = self.sink_len(seq)?;
        let window_start = self.retained_start(seq)?;
        if position < sink {
            Ok(position)
        } else {
            Ok(sink + (position - window_start))
        }
    }

    /// Number of tokens physically stored in `seq`'s contiguous page buffer:
    /// `sink_len + (len - window_start)`. This is the length of the tensors
    /// returned by [`materialize_sequence`](Self::materialize_sequence).
    pub fn retained_buffer_len(&self, seq: SequenceId) -> Result<usize, KvError> {
        let sink = self.sink_len(seq)?;
        let window_start = self.retained_start(seq)?;
        Ok(sink + (self.len(seq)? - window_start))
    }

    /// Borrow one layer's contiguous f32 K/V row for a single head at absolute
    /// token `position`, reading it **in place** from the owning page.
    ///
    /// This is the runtime-managed (paged) attention read primitive: instead of
    /// materializing the whole sequence into a fresh `present` buffer every
    /// decode step, an attention kernel calls this per attended token to borrow
    /// the persistent page row directly (`num_kv_heads * head_dim` values are
    /// stored as `[head, head_dim]`, so one head's row is `head_dim` contiguous
    /// f32). Returns `None` for a quantized layer component (no contiguous f32
    /// row exists); the caller must then dequantize via
    /// [`materialize_sequence`](Self::materialize_sequence).
    ///
    /// `position` must be a retained, filled token (`retained_start(seq) <=
    /// position < len(seq)`, or a pinned sink); an evicted or out-of-range
    /// position is an error.
    pub fn head_token_row(
        &self,
        seq: SequenceId,
        layer_idx: usize,
        kind: crate::KvKind,
        head: usize,
        position: usize,
    ) -> Result<Option<&[f32]>, KvError> {
        if self.page_table.tensor_config.is_none() {
            return Err(KvError::TensorStorageNotConfigured);
        }
        let len = self.len(seq)?;
        if position >= len {
            return Err(KvError::InvalidPosition {
                position,
                length: len,
            });
        }
        let start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        if position >= sink && position < start {
            return Err(KvError::PositionEvicted {
                position,
                retained_start: start,
            });
        }
        let geom = *self
            .page_table
            .layer_configs
            .get(layer_idx)
            .ok_or(KvError::SequenceNotFound(seq))?;
        let page_size = self.page_table.page_size;
        let buffer_index = self.buffer_index(seq, position)?;
        let page_index = buffer_index / page_size;
        let token_offset = buffer_index % page_size;
        let pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        let page_id = *pages.get(page_index).ok_or(KvError::InvalidPosition {
            position,
            length: len,
        })?;
        let page = self
            .page_table
            .pages
            .get(&page_id)
            .ok_or(KvError::PageNotFound(page_id))?;
        let component = layer_idx * 2 + if kind == crate::KvKind::Key { 0 } else { 1 };
        Ok(page.head_token_f32(page_size, geom.head_dim, component, head, token_offset))
    }

    /// Validate that `seq` can rewind to `position` without mutating page state.
    pub fn validate_rewind_to(&self, seq: SequenceId, position: usize) -> Result<(), KvError> {
        let retained_start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        let length = self.len(seq)?;
        if position < retained_start && (sink == 0 || position >= sink) {
            return Err(KvError::PositionEvicted {
                position,
                retained_start,
            });
        }
        if position > length {
            return Err(KvError::InvalidPosition { position, length });
        }
        self.page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        Ok(())
    }
}

impl KvCacheOps for PagedKvCache {
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError> {
        self.validate_rewind_to(seq, position)?;
        let sink = self.sink_len(seq)?;

        let page_size = self.page_table.page_size;
        let retained_position = self.buffer_index(seq, position)?;
        let pages_needed = retained_position.div_ceil(page_size);

        let current_pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?
            .to_vec();
        for &page_id in current_pages.iter().skip(pages_needed) {
            self.page_table.free(page_id);
        }

        if let Some(seq_pages) = self.page_table.sequences.get_mut(&seq) {
            seq_pages.truncate(pages_needed);
        }
        if retained_position > 0 {
            let last_offset = (retained_position - 1) % page_size + 1;
            if let Some(&last_page_id) = self.page_table.sequences.get(&seq).and_then(|p| p.last())
                && let Some(page) = self.page_table.pages.get_mut(&last_page_id)
            {
                page.filled = last_offset;
            }
        }
        self.page_table.set_sequence_len(seq, position);

        // Rewinding into the pinned sink prefix discards the entire window and
        // any remaining sink pages beyond `position`. Reset gap bookkeeping so
        // the truncated sequence is treated as a plain contiguous prefix.
        if position < sink {
            self.page_table.set_sequence_sink_len(seq, 0);
            self.page_table.set_sequence_start(seq, 0);
        }

        Ok(())
    }

    fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId, KvError> {
        let retained_start = self.retained_start(source)?;
        let length = self.len(source)?;
        if position < retained_start {
            return Err(KvError::PositionEvicted {
                position,
                retained_start,
            });
        }
        if position > length {
            return Err(KvError::InvalidPosition { position, length });
        }

        let page_size = self.page_table.page_size;
        let sink = self.sink_len(source)?;
        let pages_needed = self.buffer_index(source, position)?.div_ceil(page_size);
        let source_pages = self
            .page_table
            .get_sequence(source)
            .ok_or(KvError::SequenceNotFound(source))?
            .iter()
            .copied()
            .take(pages_needed)
            .collect::<Vec<_>>();

        let new_seq = self.create_sequence();
        self.page_table.set_sequence_start(new_seq, retained_start);
        self.page_table.set_sequence_sink_len(new_seq, sink);
        for page_id in &source_pages {
            self.page_table.retain(*page_id);
            self.page_table.push_page(new_seq, *page_id);
        }
        self.page_table.set_sequence_len(new_seq, position);

        Ok(new_seq)
    }

    fn checkpoint(&self, seq: SequenceId) -> Result<CacheCheckpoint, KvError> {
        let pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;

        Ok(CacheCheckpoint {
            seq,
            position: self.len(seq)?,
            page_ids: pages.to_vec(),
        })
    }

    fn restore(&mut self, seq: SequenceId, checkpoint: CacheCheckpoint) -> Result<(), KvError> {
        let retained_start = self.retained_start(seq)?;
        if checkpoint.position < retained_start {
            return Err(KvError::PositionEvicted {
                position: checkpoint.position,
                retained_start,
            });
        }
        self.rewind_to(seq, checkpoint.position)
    }

    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError> {
        let length = self.len(seq)?;
        let page_size = self.page_table.page_size;
        for position in length..length + num_tokens {
            let retained_position = self.buffer_index(seq, position)?;
            let page_index = retained_position / page_size;
            let token_offset = retained_position % page_size;
            let page_id = self.ensure_page_for_write(seq, page_index)?;
            self.page_table.promote_to_hot(page_id)?;
            if let Some(page) = self.page_table.pages.get_mut(&page_id) {
                page.filled = page.filled.max(token_offset + 1);
            }
        }
        self.page_table.set_sequence_len(seq, length + num_tokens);
        Ok(())
    }

    fn len(&self, seq: SequenceId) -> Result<usize, KvError> {
        self.page_table
            .sequence_len(seq)
            .ok_or(KvError::SequenceNotFound(seq))
    }

    fn sequence_bytes(&self, seq: SequenceId) -> Result<u64, KvError> {
        let pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        Ok((pages.len() as u64).saturating_mul(self.page_table.bytes_per_page()))
    }

    fn resident_bytes(&self) -> u64 {
        self.page_table.resident_bytes()
    }

    fn leased_bytes(&self) -> Option<u64> {
        self.page_table.leased_bytes()
    }

    fn view(&self) -> crate::KvViewKind {
        // Pages are scattered host buffers today. Virtual contiguity, which
        // would let a backend that needs one flat range read them without a
        // copy, is not built yet -- so claiming it here would be a promise the
        // store cannot keep.
        crate::KvViewKind::Paged
    }

    fn remove(&mut self, seq: SequenceId) -> Result<(), KvError> {
        let pages = self.page_table.remove_sequence(seq);
        for page_id in pages {
            self.page_table.free(page_id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DevicePageSpan, HostPageStoreView, HostPageStoreViewMut, KvPageStore, KvPageStoreFactory,
        PageStoreLayout,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    #[derive(Debug)]
    struct ToggleCopyFactory {
        fail_copy: Arc<AtomicBool>,
    }

    impl KvPageStoreFactory for ToggleCopyFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            residency: Device,
            layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            Ok(Box::new(ToggleCopyStore {
                residency,
                data: vec![0.0; layout.f32_len],
                quantized_data: vec![0; layout.int8_len],
                fp8_data: vec![0; layout.fp8_len],
                quant_scales: vec![1.0; layout.scale_len],
                fail_copy: Arc::clone(&self.fail_copy),
            }))
        }
    }

    #[derive(Debug, Clone)]
    struct ToggleCopyStore {
        residency: Device,
        data: Vec<f32>,
        quantized_data: Vec<i8>,
        fp8_data: Vec<u8>,
        quant_scales: Vec<f32>,
        fail_copy: Arc<AtomicBool>,
    }

    impl KvPageStore for ToggleCopyStore {
        fn residency(&self) -> Device {
            self.residency
        }

        fn allocated_bytes(&self) -> u64 {
            (self.data.len() * std::mem::size_of::<f32>()
                + self.quantized_data.len()
                + self.fp8_data.len()
                + self.quant_scales.len() * std::mem::size_of::<f32>()) as u64
        }

        fn reset_storage(&mut self) {
            self.data.fill(0.0);
            self.quantized_data.fill(0);
            self.fp8_data.fill(0);
            self.quant_scales.fill(1.0);
        }

        fn host_view(&self) -> Option<HostPageStoreView<'_>> {
            Some(HostPageStoreView {
                data: &self.data,
                quantized_data: &self.quantized_data,
                fp8_data: &self.fp8_data,
                quant_scales: &self.quant_scales,
            })
        }

        fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>> {
            Some(HostPageStoreViewMut {
                data: &mut self.data,
                quantized_data: &mut self.quantized_data,
                fp8_data: &mut self.fp8_data,
                quant_scales: &mut self.quant_scales,
            })
        }

        fn device_span(&self) -> Option<DevicePageSpan> {
            None
        }

        fn copy_to(&self, target: &mut dyn KvPageStore) -> Result<u64, KvError> {
            if self.fail_copy.load(Ordering::Relaxed) {
                return Err(KvError::PageStoreCopyUnsupported {
                    from: self.residency,
                    to: target.residency(),
                });
            }
            target.copy_from_host(self.host_view().unwrap())?;
            Ok(self.allocated_bytes())
        }

        fn copy_from_host(&mut self, source: HostPageStoreView<'_>) -> Result<(), KvError> {
            self.data.copy_from_slice(source.data);
            self.quantized_data.copy_from_slice(source.quantized_data);
            self.fp8_data.copy_from_slice(source.fp8_data);
            self.quant_scales.copy_from_slice(source.quant_scales);
            Ok(())
        }

        fn clone_store(&self) -> Box<dyn KvPageStore> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct LeaseTrackedFactory {
        ledger: Arc<onnx_runtime_memory_governor::LeaseLedger>,
        observed_copy_used: Arc<AtomicU64>,
        observed_snapshot_drop_used: Arc<AtomicU64>,
        fail_copy: Arc<AtomicBool>,
    }

    impl KvPageStoreFactory for LeaseTrackedFactory {
        fn allocation_bytes(&self, _residency: Device, layout: PageStoreLayout) -> u64 {
            layout.host_allocated_bytes()
        }

        fn create(
            &self,
            residency: Device,
            layout: PageStoreLayout,
        ) -> Result<Box<dyn KvPageStore>, KvError> {
            Ok(Box::new(LeaseTrackedStore {
                residency,
                data: vec![0.0; layout.f32_len],
                quantized_data: vec![0; layout.int8_len],
                fp8_data: vec![0; layout.fp8_len],
                quant_scales: vec![1.0; layout.scale_len],
                ledger: Arc::clone(&self.ledger),
                observed_copy_used: Arc::clone(&self.observed_copy_used),
                observed_snapshot_drop_used: Arc::clone(&self.observed_snapshot_drop_used),
                fail_copy: Arc::clone(&self.fail_copy),
                snapshot: false,
            }))
        }
    }

    #[derive(Debug)]
    struct LeaseTrackedStore {
        residency: Device,
        data: Vec<f32>,
        quantized_data: Vec<i8>,
        fp8_data: Vec<u8>,
        quant_scales: Vec<f32>,
        ledger: Arc<onnx_runtime_memory_governor::LeaseLedger>,
        observed_copy_used: Arc<AtomicU64>,
        observed_snapshot_drop_used: Arc<AtomicU64>,
        fail_copy: Arc<AtomicBool>,
        snapshot: bool,
    }

    impl Drop for LeaseTrackedStore {
        fn drop(&mut self) {
            if self.snapshot {
                self.observed_snapshot_drop_used.store(
                    self.ledger.used(onnx_runtime_memory_governor::Tier::Host),
                    Ordering::Relaxed,
                );
            }
        }
    }

    impl KvPageStore for LeaseTrackedStore {
        fn residency(&self) -> Device {
            self.residency
        }

        fn allocated_bytes(&self) -> u64 {
            (self.data.len() * std::mem::size_of::<f32>()
                + self.quantized_data.len()
                + self.fp8_data.len()
                + self.quant_scales.len() * std::mem::size_of::<f32>()) as u64
        }

        fn reset_storage(&mut self) {
            self.data.fill(0.0);
            self.quantized_data.fill(0);
            self.fp8_data.fill(0);
            self.quant_scales.fill(1.0);
        }

        fn host_view(&self) -> Option<HostPageStoreView<'_>> {
            Some(HostPageStoreView {
                data: &self.data,
                quantized_data: &self.quantized_data,
                fp8_data: &self.fp8_data,
                quant_scales: &self.quant_scales,
            })
        }

        fn host_view_mut(&mut self) -> Option<HostPageStoreViewMut<'_>> {
            Some(HostPageStoreViewMut {
                data: &mut self.data,
                quantized_data: &mut self.quantized_data,
                fp8_data: &mut self.fp8_data,
                quant_scales: &mut self.quant_scales,
            })
        }

        fn device_span(&self) -> Option<DevicePageSpan> {
            None
        }

        fn copy_to(&self, target: &mut dyn KvPageStore) -> Result<u64, KvError> {
            self.observed_copy_used.store(
                self.ledger.used(onnx_runtime_memory_governor::Tier::Host),
                Ordering::Relaxed,
            );
            if self.fail_copy.load(Ordering::Relaxed) {
                return Err(KvError::PageStoreCopyUnsupported {
                    from: self.residency,
                    to: target.residency(),
                });
            }
            target.copy_from_host(self.host_view().unwrap())?;
            Ok(self.allocated_bytes())
        }

        fn copy_from_host(&mut self, source: HostPageStoreView<'_>) -> Result<(), KvError> {
            self.data.copy_from_slice(source.data);
            self.quantized_data.copy_from_slice(source.quantized_data);
            self.fp8_data.copy_from_slice(source.fp8_data);
            self.quant_scales.copy_from_slice(source.quant_scales);
            Ok(())
        }

        fn clone_store(&self) -> Box<dyn KvPageStore> {
            Box::new(Self {
                residency: self.residency,
                data: self.data.clone(),
                quantized_data: self.quantized_data.clone(),
                fp8_data: self.fp8_data.clone(),
                quant_scales: self.quant_scales.clone(),
                ledger: Arc::clone(&self.ledger),
                observed_copy_used: Arc::clone(&self.observed_copy_used),
                observed_snapshot_drop_used: Arc::clone(&self.observed_snapshot_drop_used),
                fail_copy: Arc::clone(&self.fail_copy),
                snapshot: true,
            })
        }
    }
    use crate::{KvDType, KvKind, PageTensorConfig};
    use onnx_genai_metadata::{
        KvCacheSpec, KvComponentTolerance, KvQuantTolerance, LayerPrecisionOverride,
    };

    fn config() -> PageTensorConfig {
        PageTensorConfig {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 3,
            page_size: 2,
            dtype: KvDType::F32,
        }
    }

    fn layers(base: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..2)
            .map(|layer| {
                let key = (0..6)
                    .map(|i| base + layer as f32 * 100.0 + i as f32)
                    .collect();
                let value = (0..6)
                    .map(|i| base + layer as f32 * 100.0 + 50.0 + i as f32)
                    .collect();
                (key, value)
            })
            .collect()
    }

    fn borrowed_layers(data: &[(Vec<f32>, Vec<f32>)]) -> Vec<LayerKv<'_>> {
        data.iter()
            .map(|(key, value)| LayerKv { key, value })
            .collect()
    }

    fn small_config(dtype: KvDType) -> PageTensorConfig {
        PageTensorConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            page_size: 1,
            dtype,
        }
    }

    fn two_head_config(dtype: KvDType) -> PageTensorConfig {
        PageTensorConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            page_size: 1,
            dtype,
        }
    }

    fn small_layers(values: [f32; 4]) -> Vec<(Vec<f32>, Vec<f32>)> {
        vec![(values.to_vec(), values.map(|value| value + 10.0).to_vec())]
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= tolerance,
                "idx {idx}: actual {actual}, expected {expected}, diff {diff}, tolerance {tolerance}"
            );
        }
    }

    /// Two layers with different geometry: sliding-style (head_dim 8) and
    /// full-style (head_dim 16), also with different `num_kv_heads`.
    fn heterogeneous_layer_configs() -> Vec<LayerTensorConfig> {
        vec![LayerTensorConfig::new(2, 8), LayerTensorConfig::new(3, 16)]
    }

    /// Build one token of per-layer K/V for `heterogeneous_layer_configs`,
    /// filling each scalar with a deterministic, per-(token, layer, index) value.
    fn hetero_token(configs: &[LayerTensorConfig], token: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        configs
            .iter()
            .enumerate()
            .map(|(layer, geom)| {
                let per_layer = geom.f32_len_per_token();
                let key = (0..per_layer)
                    .map(|i| (token * 1000 + layer * 100 + i) as f32)
                    .collect::<Vec<f32>>();
                let value = key.iter().map(|k| k + 500.0).collect();
                (key, value)
            })
            .collect()
    }

    /// Expected head-major `[num_kv_heads, num_tokens, head_dim]` per-layer
    /// buffer for a set of input tokens (key selects .0, value selects .1).
    fn expected_head_major(
        configs: &[LayerTensorConfig],
        tokens: &[Vec<(Vec<f32>, Vec<f32>)>],
        layer: usize,
        want_value: bool,
    ) -> Vec<f32> {
        let geom = configs[layer];
        let num_tokens = tokens.len();
        let mut out = vec![0.0; geom.num_kv_heads * num_tokens * geom.head_dim];
        for (t, token) in tokens.iter().enumerate() {
            let (key, value) = &token[layer];
            let src = if want_value { value } else { key };
            for head in 0..geom.num_kv_heads {
                for dim in 0..geom.head_dim {
                    let dst = (head * num_tokens + t) * geom.head_dim + dim;
                    out[dst] = src[head * geom.head_dim + dim];
                }
            }
        }
        out
    }

    #[test]
    fn heterogeneous_per_layer_geometry_round_trips_within_a_page() {
        let configs = heterogeneous_layer_configs();
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(4, KvDType::F32, configs.clone(), 8);
        let seq = cache.create_sequence();
        let token = hetero_token(&configs, 0);
        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.sequence_len, 1);
        assert_eq!(materialized.layers.len(), 2);

        // Per-layer geometry is carried on each layer, not uniformly.
        assert_eq!(materialized.layers[0].num_kv_heads, 2);
        assert_eq!(materialized.layers[0].head_dim, 8);
        assert_eq!(materialized.layers[1].num_kv_heads, 3);
        assert_eq!(materialized.layers[1].head_dim, 16);

        // Per-layer byte correctness: buffer lengths follow each layer's own
        // geometry (num_kv_heads * seq_len * head_dim), not a uniform value.
        assert_eq!(materialized.layers[0].key.len(), 2 * 8);
        assert_eq!(materialized.layers[1].key.len(), 3 * 16);
        assert_eq!(&materialized.layers[0].key, &token[0].0);
        assert_eq!(&materialized.layers[0].value, &token[0].1);
        assert_eq!(&materialized.layers[1].key, &token[1].0);
        assert_eq!(&materialized.layers[1].value, &token[1].1);
    }

    #[test]
    fn heterogeneous_per_layer_geometry_round_trips_across_page_boundaries() {
        let configs = heterogeneous_layer_configs();
        // page_size 2 with 3 tokens forces a second page.
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(2, KvDType::F32, configs.clone(), 8);
        let seq = cache.create_sequence();
        let tokens = (0..3)
            .map(|t| hetero_token(&configs, t))
            .collect::<Vec<_>>();
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 2);

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.sequence_len, 3);
        for layer in 0..configs.len() {
            let geom = configs[layer];
            assert_eq!(materialized.layers[layer].num_kv_heads, geom.num_kv_heads);
            assert_eq!(materialized.layers[layer].head_dim, geom.head_dim);
            assert_eq!(
                materialized.layers[layer].key.len(),
                geom.num_kv_heads * 3 * geom.head_dim
            );
            assert_eq!(
                materialized.layers[layer].key,
                expected_head_major(&configs, &tokens, layer, false)
            );
            assert_eq!(
                materialized.layers[layer].value,
                expected_head_major(&configs, &tokens, layer, true)
            );
        }
    }

    #[test]
    fn heterogeneous_quantized_per_layer_geometry_round_trips_within_tolerance() {
        let configs = heterogeneous_layer_configs();
        let mut cache = PagedKvCache::new_with_layer_quant_config(
            2,
            KvDType::Fp8E4M3Fn,
            configs.clone(),
            crate::KvQuantConfig::homogeneous(KvDType::Fp8E4M3Fn, configs.len()),
            8,
        )
        .unwrap();
        let seq = cache.create_sequence();
        let tokens = (0..3)
            .map(|t| hetero_token(&configs, t))
            .collect::<Vec<_>>();
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }

        let materialized = cache.materialize_sequence(seq).unwrap();
        for layer in 0..configs.len() {
            let expected_k = expected_head_major(&configs, &tokens, layer, false);
            let expected_v = expected_head_major(&configs, &tokens, layer, true);
            // FP8 is lossy; assert within a relative tolerance of the max value.
            let tolerance = expected_k.iter().cloned().fold(0.0_f32, f32::max) * 0.1;
            assert_close(&materialized.layers[layer].key, &expected_k, tolerance);
            assert_close(&materialized.layers[layer].value, &expected_v, tolerance);
        }
    }

    #[test]
    fn heterogeneous_write_rejects_wrong_per_layer_length() {
        let configs = heterogeneous_layer_configs();
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(2, KvDType::F32, configs.clone(), 8);
        let seq = cache.create_sequence();
        // Layer 1 expects 48 scalars (3 heads * 16); give it a layer-0-sized
        // buffer (16) to prove per-layer validation, not a uniform check.
        let mut token = hetero_token(&configs, 0);
        token[1].0.truncate(16);
        token[1].1.truncate(16);
        assert!(matches!(
            cache.append_token_kv(seq, &borrowed_layers(&token)),
            Err(KvError::InvalidTensorShape(_))
        ));
    }

    #[test]
    fn page_tensor_write_read_round_trip() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 4);
        let seq = cache.create_sequence();
        let token = layers(10.0);

        assert_eq!(
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap(),
            0
        );

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.sequence_len, 1);
        assert_eq!(materialized.layers.len(), 2);
        for (layer_idx, (expected_k, expected_v)) in token.iter().enumerate() {
            assert_eq!(&materialized.layers[layer_idx].key, expected_k);
            assert_eq!(&materialized.layers[layer_idx].value, expected_v);
        }
    }

    #[test]
    fn append_across_page_boundaries_materializes_in_order() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 4);
        let seq = cache.create_sequence();
        let all = [layers(0.0), layers(1000.0), layers(2000.0)];
        for token in &all {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }

        assert_eq!(cache.len(seq).unwrap(), 3);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 2);
        let materialized = cache.materialize_sequence(seq).unwrap();
        for layer_idx in 0..2 {
            let mut expected_k = Vec::new();
            let mut expected_v = Vec::new();
            for head in 0..2 {
                for token in &all {
                    expected_k.extend_from_slice(&token[layer_idx].0[head * 3..head * 3 + 3]);
                    expected_v.extend_from_slice(&token[layer_idx].1[head * 3..head * 3 + 3]);
                }
            }

            assert_eq!(materialized.layers[layer_idx].key, expected_k);
            assert_eq!(materialized.layers[layer_idx].value, expected_v);
        }
    }

    #[test]
    fn sliding_window_evicts_leading_pages_and_preserves_absolute_positions() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 8);
        let seq = cache.create_sequence();
        for position in 0..9 {
            let token = layers(position as f32 * 1000.0);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }

        assert_eq!(cache.apply_sliding_window(seq, 3).unwrap(), 3);
        assert_eq!(cache.len(seq).unwrap(), 9);
        assert_eq!(cache.retained_range(seq).unwrap(), 6..9);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 2);

        for position in 9..11 {
            let token = layers(position as f32 * 1000.0);
            assert_eq!(
                cache
                    .append_token_kv(seq, &borrowed_layers(&token))
                    .unwrap(),
                position
            );
            cache.apply_sliding_window(seq, 3).unwrap();
        }

        assert_eq!(cache.len(seq).unwrap(), 11);
        assert_eq!(cache.retained_range(seq).unwrap(), 8..11);
        assert!(matches!(
            cache.rewind_to(seq, 7),
            Err(KvError::PositionEvicted {
                position: 7,
                retained_start: 8
            })
        ));
        cache.rewind_to(seq, 10).unwrap();
        assert_eq!(cache.retained_range(seq).unwrap(), 8..10);

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.start_position, 8);
        assert_eq!(materialized.sequence_len, 2);
        for layer_idx in 0..2 {
            let expected = [layers(8000.0), layers(9000.0)];
            let mut expected_k = Vec::new();
            let mut expected_v = Vec::new();
            for head in 0..2 {
                for token in &expected {
                    expected_k.extend_from_slice(&token[layer_idx].0[head * 3..head * 3 + 3]);
                    expected_v.extend_from_slice(&token[layer_idx].1[head * 3..head * 3 + 3]);
                }
            }
            assert_eq!(materialized.layers[layer_idx].key, expected_k);
            assert_eq!(materialized.layers[layer_idx].value, expected_v);
        }
    }

    fn assert_materialized_order(cache: &PagedKvCache, seq: SequenceId, order: &[f32]) {
        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.sequence_len, order.len());
        for layer_idx in 0..2 {
            let expected = order.iter().map(|base| layers(*base)).collect::<Vec<_>>();
            let mut expected_k = Vec::new();
            let mut expected_v = Vec::new();
            for head in 0..2 {
                for token in &expected {
                    expected_k.extend_from_slice(&token[layer_idx].0[head * 3..head * 3 + 3]);
                    expected_v.extend_from_slice(&token[layer_idx].1[head * 3..head * 3 + 3]);
                }
            }
            assert_eq!(materialized.layers[layer_idx].key, expected_k);
            assert_eq!(materialized.layers[layer_idx].value, expected_v);
        }
    }

    #[test]
    fn sliding_window_with_sinks_pins_prefix_and_evicts_middle() {
        // page_size = 2; sink_tokens = 2 (one pinned sink page); window = 3.
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 16);
        let seq = cache.create_sequence();
        for position in 0..9 {
            let token = layers(position as f32 * 1000.0);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }

        // keep_from = 6, sink pinned = [0,2), window = [6,9): evict pages [2,4),[4,6).
        assert_eq!(cache.apply_sliding_window_with_sinks(seq, 3, 2).unwrap(), 2);
        assert_eq!(cache.len(seq).unwrap(), 9);
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(2));
        assert_eq!(cache.retained_start(seq).unwrap(), 6);
        assert_eq!(cache.retained_buffer_len(seq).unwrap(), 5);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 3);
        let m = cache.materialize_sequence(seq).unwrap();
        assert_eq!(m.sink_len, 2);
        assert_eq!(m.start_position, 6);
        // Contiguous buffer holds sinks [0,1] followed by window [6,7,8].
        assert_materialized_order(&cache, seq, &[0.0, 1000.0, 6000.0, 7000.0, 8000.0]);

        // Roll forward: sinks stay pinned, window slides.
        for position in 9..13 {
            let token = layers(position as f32 * 1000.0);
            assert_eq!(
                cache
                    .append_token_kv(seq, &borrowed_layers(&token))
                    .unwrap(),
                position
            );
            cache.apply_sliding_window_with_sinks(seq, 3, 2).unwrap();
        }
        // len = 13, keep_from = 10 -> window_start = 10, sinks still [0,2).
        assert_eq!(cache.len(seq).unwrap(), 13);
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(2));
        assert_eq!(cache.retained_start(seq).unwrap(), 10);
        assert_materialized_order(&cache, seq, &[0.0, 1000.0, 10000.0, 11000.0, 12000.0]);

        // Rewind inside the window preserves the pinned sinks.
        cache.rewind_to(seq, 12).unwrap();
        assert_eq!(cache.len(seq).unwrap(), 12);
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(2));
        assert_materialized_order(&cache, seq, &[0.0, 1000.0, 10000.0, 11000.0]);

        // Positions in the evicted gap are rejected.
        assert!(matches!(
            cache.rewind_to(seq, 5),
            Err(KvError::PositionEvicted { position: 5, .. })
        ));
    }

    #[test]
    fn rewind_into_sink_discards_window_and_resets_gap_bookkeeping() {
        // page_size=2; sink_tokens=2 (1 pinned sink page); window=3.
        // After sinks activate the retained set is [0,2) ∪ [keep_from, len).
        // Rewinding to a position inside the sink prefix (<2) must:
        //   - discard all window pages,
        //   - truncate the sink pages to what is needed,
        //   - reset sink_len and retained_start to 0 (plain contiguous prefix).
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 16);
        let seq = cache.create_sequence();
        for position in 0..10 {
            let token = layers(position as f32 * 1000.0);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }
        // len=10, keep_from=7 → sinks=[0,2), window=[8,10) (page-aligned).
        cache.apply_sliding_window_with_sinks(seq, 3, 2).unwrap();
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(2));
        let retained_start = cache.retained_start(seq).unwrap();
        assert!(
            retained_start > 2,
            "gap must be open for the test to be meaningful"
        );

        // Positions in the evicted gap are still rejected.
        assert!(matches!(
            cache.rewind_to(seq, 4),
            Err(KvError::PositionEvicted { position: 4, .. })
        ));

        // Rewind to position 1 (inside sink prefix).
        cache.rewind_to(seq, 1).unwrap();
        // Length is now 1.
        assert_eq!(cache.len(seq).unwrap(), 1);
        // sink_len and retained_start reset: no gap, plain contiguous prefix.
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(0));
        assert_eq!(cache.retained_start(seq).unwrap(), 0);
        // Only the first page (which covers positions 0 and 1) remains; the
        // window pages were freed.
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 1);
        // Materialized buffer holds exactly token 0.
        let m = cache.materialize_sequence(seq).unwrap();
        assert_eq!(m.sink_len, 0);
        assert_eq!(m.start_position, 0);
        assert_eq!(m.sequence_len, 1);
        assert_materialized_order(&cache, seq, &[0.0]);

        // After rewind the sequence is usable as a normal contiguous prefix:
        // appending a token and materializing produces two tokens.
        let token = layers(99000.0);
        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();
        assert_eq!(cache.len(seq).unwrap(), 2);
        assert_materialized_order(&cache, seq, &[0.0, 99000.0]);
    }

    #[test]
    fn sliding_window_with_zero_sinks_matches_plain_window() {
        let mut plain = PagedKvCache::new_with_tensor_config(config(), 16);
        let mut sunk = PagedKvCache::new_with_tensor_config(config(), 16);
        let a = plain.create_sequence();
        let b = sunk.create_sequence();
        for position in 0..9 {
            let token = layers(position as f32 * 1000.0);
            plain.append_token_kv(a, &borrowed_layers(&token)).unwrap();
            sunk.append_token_kv(b, &borrowed_layers(&token)).unwrap();
        }
        assert_eq!(
            plain.apply_sliding_window(a, 3).unwrap(),
            sunk.apply_sliding_window_with_sinks(b, 3, 0).unwrap(),
        );
        assert_eq!(sunk.page_table.sequence_sink_len(b), Some(0));
        assert_eq!(
            plain.retained_range(a).unwrap(),
            sunk.retained_range(b).unwrap()
        );
        assert_eq!(
            plain.materialize_sequence(a).unwrap(),
            sunk.materialize_sequence(b).unwrap()
        );
    }

    #[test]
    fn sliding_window_with_sinks_no_gap_when_window_covers_sequence() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 16);
        let seq = cache.create_sequence();
        for position in 0..4 {
            let token = layers(position as f32 * 1000.0);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }
        // window (4) + sinks cover the whole 4-token sequence: no eviction, no gap.
        assert_eq!(cache.apply_sliding_window_with_sinks(seq, 4, 2).unwrap(), 0);
        assert_eq!(cache.page_table.sequence_sink_len(seq), Some(0));
        assert_eq!(cache.retained_start(seq).unwrap(), 0);
        assert_materialized_order(&cache, seq, &[0.0, 1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn sliding_window_with_zero_window_is_rejected() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 16);
        let seq = cache.create_sequence();
        cache
            .append_token_kv(seq, &borrowed_layers(&layers(0.0)))
            .unwrap();
        assert!(matches!(
            cache.apply_sliding_window_with_sinks(seq, 0, 2),
            Err(KvError::InvalidWindowSize)
        ));
    }

    #[test]
    fn cache_without_sliding_window_retains_full_sequence() {
        let mut cache = PagedKvCache::new(2, 4);
        let seq = cache.create_sequence();
        cache.append(seq, 7).unwrap();

        assert_eq!(cache.len(seq).unwrap(), 7);
        assert_eq!(cache.retained_range(seq).unwrap(), 0..7);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 4);
    }

    #[test]
    fn rewind_truncates_pages_and_sequence_length() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 4);
        let seq = cache.create_sequence();
        for i in 0..3 {
            let token = layers(i as f32 * 10.0);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }

        cache.rewind_to(seq, 1).unwrap();

        assert_eq!(cache.len(seq).unwrap(), 1);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 1);
        assert_eq!(cache.page_table.free_count(Device::Gpu(0)), 3);
        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.sequence_len, 1);
    }

    #[test]
    fn count_append_len_and_rewind_still_work() {
        let mut cache = PagedKvCache::new(4, 4);
        let seq = cache.create_sequence();
        cache.append(seq, 5).unwrap();
        assert_eq!(cache.len(seq).unwrap(), 5);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 2);
        cache.rewind_to(seq, 4).unwrap();
        assert_eq!(cache.len(seq).unwrap(), 4);
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 1);
    }

    #[test]
    fn append_after_fork_copies_shared_partial_page() {
        let mut cache = PagedKvCache::new(4, 4);
        let seq = cache.create_sequence();
        cache.append(seq, 2).unwrap();
        let original_page = cache.page_table.get_sequence(seq).unwrap()[0];

        let forked = cache.fork(seq, 2).unwrap();
        cache.append(forked, 1).unwrap();

        let forked_page = cache.page_table.get_sequence(forked).unwrap()[0];
        assert_ne!(original_page, forked_page);
        assert_eq!(cache.page_table.pages[&original_page].ref_count, 1);
        assert_eq!(cache.page_table.pages[&forked_page].ref_count, 1);
        assert_eq!(cache.len(seq).unwrap(), 2);
        assert_eq!(cache.len(forked).unwrap(), 3);
    }

    #[test]
    fn failed_cow_copy_rolls_back_allocation_without_capacity_loss() {
        let mut cache = PagedKvCache::new_with_tensor_config(
            PageTensorConfig {
                page_size: 2,
                ..small_config(KvDType::F32)
            },
            3,
        );
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let shared_page = cache.page_table.get_sequence(source).unwrap()[0];
        let fail_copy = Arc::new(AtomicBool::new(false));
        cache
            .page_table
            .set_migration_factory(Arc::new(ToggleCopyFactory {
                fail_copy: Arc::clone(&fail_copy),
            }));
        cache
            .page_table
            .migrate_page(shared_page, Device::Gpu(1))
            .unwrap();
        for values in [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]] {
            let unrelated = cache.create_sequence();
            cache
                .append_token_kv(unrelated, &borrowed_layers(&small_layers(values)))
                .unwrap();
        }

        cache.page_table.touch(shared_page);
        fail_copy.store(true, Ordering::Relaxed);

        let free_before = cache.page_table.free_count(Device::Gpu(0));
        let refs_before = cache.page_table.pages[&shared_page].ref_count;
        let resident_before = cache.resident_bytes();
        let stats_before = cache.page_table.stats();
        let hot_before = cache.page_table.hot_used_count();
        let residency_before = cache
            .page_table
            .pages
            .iter()
            .map(|(page_id, page)| (*page_id, page.residency()))
            .collect::<std::collections::HashMap<_, _>>();
        for _ in 0..4 {
            assert!(
                cache
                    .append_token_kv(
                        forked,
                        &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
                    )
                    .is_err()
            );
            assert_eq!(cache.page_table.free_count(Device::Gpu(0)), free_before);
            assert_eq!(cache.page_table.pages[&shared_page].ref_count, refs_before);
            assert_eq!(cache.resident_bytes(), resident_before);
            assert_eq!(cache.page_table.stats(), stats_before);
            assert_eq!(cache.page_table.hot_used_count(), hot_before);
            assert_eq!(
                cache
                    .page_table
                    .pages
                    .iter()
                    .map(|(page_id, page)| (*page_id, page.residency()))
                    .collect::<std::collections::HashMap<_, _>>(),
                residency_before
            );
            assert_eq!(cache.len(forked).unwrap(), 1);
        }

        fail_copy.store(false, Ordering::Relaxed);
        cache
            .append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            )
            .unwrap();
        assert_eq!(cache.len(forked).unwrap(), 2);
        assert_eq!(cache.page_table.pages[&shared_page].ref_count, 1);
    }

    #[test]
    fn full_budget_refuses_cow_snapshot_before_allocation() {
        use onnx_runtime_memory_governor::{
            HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
        };

        let config = PageTensorConfig {
            page_size: 2,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let pool_bytes = PageTable::planned_pool_bytes(2, 1, &layer_configs, Some(&quant));
        let ledger = LeaseLedger::new(0, pool_bytes, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            2,
            KvDType::F32,
            layer_configs,
            1,
            &governor,
            Tier::Host,
            HolderId::new(99),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let shared = cache.page_table.get_sequence(source).unwrap()[0];
        let free_before = cache.page_table.free_count(Device::Gpu(0));
        let stats_before = cache.page_table.stats();
        let resident_before = cache.resident_bytes();

        assert!(matches!(
            cache.append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            ),
            Err(KvError::MigrationPressure(_))
        ));
        assert_eq!(governor.used(Tier::Host), pool_bytes);
        assert_eq!(cache.page_table.free_count(Device::Gpu(0)), free_before);
        assert_eq!(cache.page_table.stats(), stats_before);
        assert_eq!(cache.resident_bytes(), resident_before);
        assert_eq!(cache.page_table.pages[&shared].ref_count, 2);
        assert_eq!(cache.len(forked).unwrap(), 1);
    }

    #[test]
    fn full_budget_with_free_page_refuses_cow_before_snapshot_clone() {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

        let config = PageTensorConfig {
            page_size: 2,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let pool_bytes = PageTable::planned_pool_bytes(2, 2, &layer_configs, Some(&quant));
        let ledger = LeaseLedger::new(0, pool_bytes, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            2,
            KvDType::F32,
            layer_configs,
            2,
            &governor,
            Tier::Host,
            HolderId::new(100),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let shared = cache.page_table.get_sequence(source).unwrap()[0];
        let free_before = cache.page_table.free_count(Device::Gpu(0));
        let stats_before = cache.page_table.stats();
        let resident_before = cache.resident_bytes();

        assert!(matches!(
            cache.append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            ),
            Err(KvError::MigrationPressure(_))
        ));
        assert_eq!(ledger.used(Tier::Host), pool_bytes);
        assert_eq!(cache.page_table.free_count(Device::Gpu(0)), free_before);
        assert_eq!(cache.page_table.stats(), stats_before);
        assert_eq!(cache.resident_bytes(), resident_before);
        assert_eq!(cache.page_table.pages[&shared].ref_count, 2);
        assert_eq!(cache.len(forked).unwrap(), 1);
    }

    fn governed_tracked_cow(
        fail_copy: bool,
    ) -> (
        PagedKvCache,
        SequenceId,
        Arc<onnx_runtime_memory_governor::LeaseLedger>,
        u64,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
    ) {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

        let config = PageTensorConfig {
            page_size: 2,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let pool_bytes = PageTable::planned_pool_bytes(2, 2, &layer_configs, Some(&quant));
        let page_bytes = pool_bytes / 2;
        let ledger = LeaseLedger::new(0, pool_bytes + page_bytes * 2, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            2,
            KvDType::F32,
            layer_configs,
            2,
            &governor,
            Tier::Host,
            HolderId::new(101),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let shared = cache.page_table.get_sequence(source).unwrap()[0];
        let observed_copy = Arc::new(AtomicU64::new(0));
        let observed_drop = Arc::new(AtomicU64::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        cache
            .page_table
            .set_migration_factory(Arc::new(LeaseTrackedFactory {
                ledger: Arc::clone(&ledger),
                observed_copy_used: Arc::clone(&observed_copy),
                observed_snapshot_drop_used: Arc::clone(&observed_drop),
                fail_copy: Arc::clone(&fail),
            }));
        cache
            .page_table
            .migrate_page(shared, Device::Gpu(1))
            .unwrap();
        observed_copy.store(0, Ordering::Relaxed);
        observed_drop.store(0, Ordering::Relaxed);
        fail.store(fail_copy, Ordering::Relaxed);
        (
            cache,
            forked,
            ledger,
            pool_bytes,
            observed_copy,
            observed_drop,
        )
    }

    #[test]
    fn cow_lease_covers_both_snapshots_and_outlives_them_on_success() {
        use onnx_runtime_memory_governor::Tier;

        let (mut cache, forked, ledger, baseline, observed_copy, observed_drop) =
            governed_tracked_cow(false);
        cache
            .append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            )
            .unwrap();
        assert_eq!(observed_copy.load(Ordering::Relaxed), baseline * 2);
        assert_eq!(observed_drop.load(Ordering::Relaxed), baseline * 2);
        assert_eq!(cache.pool_lease_bytes(), Some(baseline));
        assert_eq!(ledger.used(Tier::Host), baseline);
    }

    #[test]
    fn cow_failure_keeps_lease_until_snapshots_drop_then_returns_to_baseline() {
        use onnx_runtime_memory_governor::Tier;

        let (mut cache, forked, ledger, baseline, observed_copy, observed_drop) =
            governed_tracked_cow(true);
        assert!(
            cache
                .append_token_kv(
                    forked,
                    &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
                )
                .is_err()
        );
        assert_eq!(observed_copy.load(Ordering::Relaxed), baseline * 2);
        assert_eq!(observed_drop.load(Ordering::Relaxed), baseline * 2);
        assert_eq!(ledger.used(Tier::Host), baseline);
        assert_eq!(cache.len(forked).unwrap(), 1);
    }

    #[test]
    fn cow_pressure_refusal_happens_before_tracked_snapshot_clone() {
        use onnx_runtime_memory_governor::Tier;

        let (mut cache, forked, ledger, baseline, observed_copy, observed_drop) =
            governed_tracked_cow(false);
        ledger.set_limit(Tier::Host, baseline);

        assert!(matches!(
            cache.append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            ),
            Err(KvError::MigrationPressure(_))
        ));
        assert_eq!(observed_copy.load(Ordering::Relaxed), 0);
        assert_eq!(observed_drop.load(Ordering::Relaxed), 0);
        assert_eq!(ledger.used(Tier::Host), baseline);
        assert_eq!(cache.len(forked).unwrap(), 1);
    }

    #[test]
    fn cow_persistent_growth_is_absorbed_into_pool_lease() {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

        let config = PageTensorConfig {
            page_size: 2,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let page_bytes = PageTable::planned_pool_bytes(2, 1, &layer_configs, Some(&quant));
        let ledger = LeaseLedger::new(0, page_bytes * 5, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            2,
            KvDType::F32,
            layer_configs,
            1,
            &governor,
            Tier::Host,
            HolderId::new(102),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();

        cache
            .append_token_kv(
                forked,
                &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
            )
            .unwrap();

        assert_eq!(cache.pool_lease_bytes(), Some(page_bytes * 2));
        assert_eq!(ledger.used(Tier::Host), page_bytes * 2);
        assert_eq!(cache.page_table.total_pages(), 2);
    }

    #[test]
    fn repeated_cow_growth_accumulates_exact_persistent_pool_bytes() {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

        const GROWTHS: usize = 3;
        let config = PageTensorConfig {
            page_size: 4,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let page_bytes = PageTable::planned_pool_bytes(4, 1, &layer_configs, Some(&quant));
        let ledger = LeaseLedger::new(0, page_bytes * (GROWTHS as u64 + 4), 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            4,
            KvDType::F32,
            layer_configs,
            1,
            &governor,
            Tier::Host,
            HolderId::new(103),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forks = (0..GROWTHS)
            .map(|_| cache.fork(source, 1).unwrap())
            .collect::<Vec<_>>();

        for (index, forked) in forks.into_iter().enumerate() {
            cache
                .append_token_kv(
                    forked,
                    &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
                )
                .unwrap();
            let expected = page_bytes * (index as u64 + 2);
            assert_eq!(cache.pool_lease_bytes(), Some(expected));
            assert_eq!(ledger.used(Tier::Host), expected);
        }
    }

    #[test]
    fn failed_cow_growth_releases_persistent_preflight_after_page_drop() {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

        let config = PageTensorConfig {
            page_size: 2,
            ..small_config(KvDType::F32)
        };
        let layer_configs = vec![LayerTensorConfig::new(config.num_kv_heads, config.head_dim)];
        let quant = KvQuantConfig::homogeneous(KvDType::F32, 1);
        let page_bytes = PageTable::planned_pool_bytes(2, 1, &layer_configs, Some(&quant));
        let ledger = LeaseLedger::new(0, page_bytes * 5, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let mut cache = PagedKvCache::new_leased(
            2,
            KvDType::F32,
            layer_configs,
            1,
            &governor,
            Tier::Host,
            HolderId::new(104),
        )
        .unwrap();
        let source = cache.create_sequence();
        cache
            .append_token_kv(
                source,
                &borrowed_layers(&small_layers([1.0, 2.0, 3.0, 4.0])),
            )
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let shared = cache.page_table.get_sequence(source).unwrap()[0];
        let observed_copy = Arc::new(AtomicU64::new(0));
        let observed_drop = Arc::new(AtomicU64::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        cache
            .page_table
            .set_migration_factory(Arc::new(LeaseTrackedFactory {
                ledger: Arc::clone(&ledger),
                observed_copy_used: observed_copy,
                observed_snapshot_drop_used: observed_drop,
                fail_copy: Arc::clone(&fail),
            }));
        cache
            .page_table
            .migrate_page(shared, Device::Gpu(1))
            .unwrap();
        fail.store(true, Ordering::Relaxed);

        assert!(
            cache
                .append_token_kv(
                    forked,
                    &borrowed_layers(&small_layers([5.0, 6.0, 7.0, 8.0])),
                )
                .is_err()
        );
        assert_eq!(cache.pool_lease_bytes(), Some(page_bytes));
        assert_eq!(ledger.used(Tier::Host), page_bytes);
        assert_eq!(cache.page_table.total_pages(), 1);
        assert_eq!(cache.len(forked).unwrap(), 1);
    }

    #[test]
    fn tiered_eviction_moves_lru_hot_page_to_cold_and_preserves_f32_data() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::F32), 2);
        let seq = cache.create_sequence();
        let t0 = small_layers([1.0, 2.0, 3.0, 4.0]);
        let t1 = small_layers([5.0, 6.0, 7.0, 8.0]);
        let t2 = small_layers([9.0, 10.0, 11.0, 12.0]);

        cache.append_token_kv(seq, &borrowed_layers(&t0)).unwrap();
        cache.append_token_kv(seq, &borrowed_layers(&t1)).unwrap();
        let first_page = cache.page_table.get_sequence(seq).unwrap()[0];
        assert_eq!(cache.page_table.hot_used_count(), 2);

        cache.append_token_kv(seq, &borrowed_layers(&t2)).unwrap();

        assert_eq!(cache.page_table.pages[&first_page].residency(), Device::Cpu);
        assert_eq!(cache.page_table.hot_used_count(), 2);
        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(
            materialized.layers[0].key,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
        assert_eq!(
            materialized.layers[0].value,
            [
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0
            ]
        );
    }

    #[test]
    fn tiered_prefetch_promotes_cold_page_and_evicts_another_lru_hot_page() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::F32), 2);
        let seq = cache.create_sequence();
        for base in [1.0, 5.0, 9.0] {
            let token = small_layers([base, base + 1.0, base + 2.0, base + 3.0]);
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }
        let pages = cache.page_table.get_sequence(seq).unwrap().to_vec();
        assert_eq!(cache.page_table.pages[&pages[0]].residency(), Device::Cpu);
        assert_eq!(cache.prefetch(seq, 0, 1).unwrap(), 1);

        assert_eq!(
            cache.page_table.pages[&pages[0]].residency(),
            Device::Gpu(0)
        );
        assert_eq!(cache.page_table.hot_used_count(), 2);
        assert!(
            pages[1..]
                .iter()
                .any(|page_id| cache.page_table.pages[page_id].residency() == Device::Cpu)
        );
    }

    #[test]
    fn tiered_lru_evicts_least_recently_accessed_hot_page() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::F32), 2);
        let seq = cache.create_sequence();
        let t0 = small_layers([1.0, 1.1, 1.2, 1.3]);
        let t1 = small_layers([2.0, 2.1, 2.2, 2.3]);
        let t2 = small_layers([3.0, 3.1, 3.2, 3.3]);
        cache.append_token_kv(seq, &borrowed_layers(&t0)).unwrap();
        cache.append_token_kv(seq, &borrowed_layers(&t1)).unwrap();
        let pages = cache.page_table.get_sequence(seq).unwrap().to_vec();

        cache.write_token_kv(seq, 0, &borrowed_layers(&t0)).unwrap();
        cache.append_token_kv(seq, &borrowed_layers(&t2)).unwrap();

        assert_eq!(
            cache.page_table.pages[&pages[0]].residency(),
            Device::Gpu(0)
        );
        assert_eq!(cache.page_table.pages[&pages[1]].residency(), Device::Cpu);
    }

    #[test]
    fn int8_quantize_dequantize_round_trip_is_within_tolerance() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::Int8), 2);
        let seq = cache.create_sequence();
        let token = small_layers([-1.0, -0.25, 0.25, 1.0]);

        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();

        let page_id = cache.page_table.get_sequence(seq).unwrap()[0];
        let page = &cache.page_table.pages[&page_id];
        let storage = page.host_view().expect("host store");
        assert!(storage.data.is_empty());
        assert_eq!(
            storage.quantized_data.len(),
            small_config(KvDType::Int8).f32_len_per_page()
        );
        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_close(&materialized.layers[0].key, &token[0].0, 0.05);
        assert_close(&materialized.layers[0].value, &token[0].1, 0.05);
    }

    #[test]
    fn fp8_e4m3fn_round_trip_uses_per_component_head_scales() {
        let config = two_head_config(KvDType::Fp8E4M3Fn);
        let mut cache = PagedKvCache::new_with_tensor_config(config, 1);
        let seq = cache.create_sequence();
        let token = vec![(
            vec![-1.0, -0.3, 0.3, 1.0, -100.0, -30.0, 30.0, 100.0],
            vec![-2.0, -0.6, 0.6, 2.0, -200.0, -60.0, 60.0, 200.0],
        )];

        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();

        let page_id = cache.page_table.get_sequence(seq).unwrap()[0];
        let page = &cache.page_table.pages[&page_id];
        let storage = page.host_view().expect("host store");
        assert!(storage.data.is_empty());
        assert!(storage.quantized_data.is_empty());
        assert_eq!(storage.fp8_data.len(), config.f32_len_per_page());
        assert_eq!(storage.quant_scales.len(), 4);
        assert_close(
            storage.quant_scales,
            &[1.0 / 448.0, 100.0 / 448.0, 2.0 / 448.0, 200.0 / 448.0],
            1.0e-7,
        );

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_close(
            &materialized.layers[0].key,
            &[
                -1.0,
                -0.285_714_3,
                0.285_714_3,
                1.0,
                -100.0,
                -28.571_43,
                28.571_43,
                100.0,
            ],
            1.0e-5,
        );
        assert_close(
            &materialized.layers[0].value,
            &[
                -2.0,
                -0.571_428_6,
                0.571_428_6,
                2.0,
                -200.0,
                -57.142_86,
                57.142_86,
                200.0,
            ],
            1.0e-5,
        );
    }

    #[test]
    fn fp8_e5m2_round_trip_is_within_format_error_bound() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::Fp8E5M2), 1);
        let seq = cache.create_sequence();
        let token = small_layers([-1.0, -0.3, 0.3, 1.0]);

        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();

        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_close(
            &materialized.layers[0].key,
            &[-1.0, -0.285_714_3, 0.285_714_3, 1.0],
            1.0e-6,
        );
        assert_close(
            &materialized.layers[0].value,
            &[9.428_572, 9.428_572, 11.0, 11.0],
            1.0e-5,
        );
    }

    #[test]
    fn metadata_precision_policy_honors_overrides_and_sensitive_layers() {
        let spec = KvCacheSpec {
            native_dtype: Some("float8_e4m3fn".to_owned()),
            quantization_tolerance: Some(KvQuantTolerance {
                key: Some(KvComponentTolerance {
                    default: Some("float8_e5m2".to_owned()),
                    per_layer: Some(vec![LayerPrecisionOverride {
                        layers: vec![1],
                        min_precision: "fp16".to_owned(),
                    }]),
                    quantization_axis: Some("per_token".to_owned()),
                }),
                value: Some(KvComponentTolerance {
                    default: None,
                    per_layer: None,
                    quantization_axis: Some("per_token".to_owned()),
                }),
            }),
            sensitive_layers: Some(vec![0, -1]),
            operations: None,
        };

        let quant = KvQuantConfig::from_metadata(&spec, 4).unwrap();
        assert_eq!(
            quant.layer(0),
            Some(crate::LayerKvDType {
                key: KvDType::F32,
                value: KvDType::F32,
            })
        );
        assert_eq!(
            quant.layer(1),
            Some(crate::LayerKvDType {
                key: KvDType::F32,
                value: KvDType::Fp8E4M3Fn,
            })
        );
        assert_eq!(
            quant.layer(2),
            Some(crate::LayerKvDType {
                key: KvDType::Fp8E5M2,
                value: KvDType::Fp8E4M3Fn,
            })
        );
        assert_eq!(
            quant.layer(3),
            Some(crate::LayerKvDType {
                key: KvDType::F32,
                value: KvDType::F32,
            })
        );
        assert_eq!(
            KvDType::from_metadata_name("float16").unwrap(),
            KvDType::F32
        );
    }

    #[test]
    fn sensitive_layer_storage_bypasses_fp8_quantization() {
        let config = PageTensorConfig {
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 4,
            page_size: 1,
            dtype: KvDType::Fp8E4M3Fn,
        };
        let spec = KvCacheSpec {
            native_dtype: Some("float8_e4m3fn".to_owned()),
            quantization_tolerance: None,
            sensitive_layers: Some(vec![1]),
            operations: None,
        };
        let mut cache = PagedKvCache::new_with_metadata(config, &spec, 1).unwrap();
        let seq = cache.create_sequence();
        let token = vec![
            (vec![0.1, 0.2, 0.3, 0.4], vec![1.1, 1.2, 1.3, 1.4]),
            (vec![10.1, 10.2, 10.3, 10.4], vec![11.1, 11.2, 11.3, 11.4]),
        ];

        cache
            .append_token_kv(seq, &borrowed_layers(&token))
            .unwrap();

        let page_id = cache.page_table.get_sequence(seq).unwrap()[0];
        let page = &cache.page_table.pages[&page_id];
        let storage = page.host_view().expect("host store");
        assert_eq!(storage.data.len(), 8);
        assert_eq!(storage.fp8_data.len(), 8);
        let materialized = cache.materialize_sequence(seq).unwrap();
        assert_eq!(materialized.layers[1].key, token[1].0);
        assert_eq!(materialized.layers[1].value, token[1].1);
        assert_close(&materialized.layers[0].key, &token[0].0, 0.025);
        assert_close(&materialized.layers[0].value, &token[0].1, 0.1);
    }

    #[test]
    fn int8_quantized_append_materialize_across_pages() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::Int8), 1);
        let seq = cache.create_sequence();
        let tokens = [
            small_layers([0.0, 0.2, 0.4, 0.6]),
            small_layers([0.8, 1.0, 1.2, 1.4]),
        ];
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }

        let pages = cache.page_table.get_sequence(seq).unwrap();
        assert_eq!(pages.len(), 2);
        assert!(
            pages
                .iter()
                .any(|id| cache.page_table.pages[id].residency() == Device::Cpu)
        );
        let materialized = cache.materialize_sequence(seq).unwrap();
        let expected_key = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];
        let expected_value = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4];
        assert_close(&materialized.layers[0].key, &expected_key, 0.05);
        assert_close(&materialized.layers[0].value, &expected_value, 0.05);
    }

    fn full_page_config(dtype: KvDType, page_size: usize) -> PageTensorConfig {
        PageTensorConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            page_size,
            dtype,
        }
    }

    /// Sixteen tokens whose per-token magnitude spans six orders of magnitude.
    /// Token 0 deliberately carries `1.061` in its first channel — the value Chew
    /// showed drifting under the old dequantize-whole-page / requantize-whole-page
    /// append. With a single page-wide scale driven by the largest token, token 0
    /// collapses toward zero (~100% error); per-token scales keep it exact.
    fn spread_magnitude_tokens() -> Vec<Vec<(Vec<f32>, Vec<f32>)>> {
        (0..16)
            .map(|i| {
                let magnitude = if i == 0 { 1.061 } else { 2.0_f32.powi(i + 6) };
                let key = vec![
                    magnitude,
                    magnitude * 0.9,
                    -magnitude * 0.8,
                    magnitude * 0.95,
                ];
                let value = vec![
                    magnitude * 1.1,
                    -magnitude,
                    magnitude * 0.85,
                    magnitude * 0.7,
                ];
                vec![(key, value)]
            })
            .collect()
    }

    fn assert_relative_error_bounded(actual: &[f32], expected: &[f32], max_relative: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let relative = if *expected == 0.0 {
                actual.abs()
            } else {
                (actual - expected).abs() / expected.abs()
            };
            assert!(
                relative <= max_relative,
                "idx {idx}: actual {actual}, expected {expected}, relative {relative}, bound {max_relative}"
            );
        }
    }

    /// Regression test for the fp8 page-scaling bug: filling a whole multi-token
    /// page must not requantize tokens that were already stored. This is the test
    /// the previous `page_size = 1` fp8 tests could never exercise.
    #[test]
    fn fp8_full_page_never_requantizes_previously_stored_tokens() {
        let config = full_page_config(KvDType::Fp8E4M3Fn, 16);
        let mut cache = PagedKvCache::new_with_tensor_config(config, 2);
        let seq = cache.create_sequence();
        let tokens = spread_magnitude_tokens();
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }
        // A single physical page holds the entire sequence.
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 1);
        let materialized = cache.materialize_sequence(seq).unwrap();

        // Every entry stays within a single E4M3 round-trip (3 mantissa bits ->
        // <=6.25% per normal value); the old code drove token 0 to ~33% error.
        let mut expected_key = Vec::new();
        let mut expected_value = Vec::new();
        for token in &tokens {
            expected_key.extend_from_slice(&token[0].0);
            expected_value.extend_from_slice(&token[0].1);
        }
        assert_relative_error_bounded(&materialized.layers[0].key, &expected_key, 0.07);
        assert_relative_error_bounded(&materialized.layers[0].value, &expected_value, 0.07);

        // Token 0's `1.061` is preserved, not the 1.41143 the old design produced.
        assert!((materialized.layers[0].key[0] - 1.061).abs() < 0.07);

        // Stronger invariant: a stored token is byte-identical whether or not
        // later tokens were appended, proving stored data is never touched again.
        let mut isolated = PagedKvCache::new_with_tensor_config(config, 2);
        let iso_seq = isolated.create_sequence();
        isolated
            .append_token_kv(iso_seq, &borrowed_layers(&tokens[0]))
            .unwrap();
        let iso = isolated.materialize_sequence(iso_seq).unwrap();
        assert_eq!(
            &materialized.layers[0].key[0..4],
            iso.layers[0].key.as_slice()
        );
        assert_eq!(
            &materialized.layers[0].value[0..4],
            iso.layers[0].value.as_slice()
        );
    }

    /// Same invariant for the int8 path: a full multi-token page keeps every
    /// entry within its per-token error bound and never rewrites stored tokens.
    #[test]
    fn int8_full_page_error_is_bounded_and_stable() {
        let config = full_page_config(KvDType::Int8, 8);
        let mut cache = PagedKvCache::new_with_tensor_config(config, 2);
        let seq = cache.create_sequence();
        let tokens: Vec<_> = (0..8)
            .map(|i| {
                let magnitude = if i == 0 { 1.0 } else { 2.0_f32.powi(i + 8) };
                vec![(
                    vec![
                        magnitude,
                        magnitude * 0.9,
                        -magnitude * 0.8,
                        magnitude * 0.95,
                    ],
                    vec![
                        magnitude * 1.1,
                        -magnitude,
                        magnitude * 0.85,
                        magnitude * 0.7,
                    ],
                )]
            })
            .collect();
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }
        assert_eq!(cache.page_table.get_sequence(seq).unwrap().len(), 1);
        let materialized = cache.materialize_sequence(seq).unwrap();

        let mut expected_key = Vec::new();
        for token in &tokens {
            expected_key.extend_from_slice(&token[0].0);
        }
        // int8 keeps ~7 bits of precision per token -> <1% per entry.
        assert_relative_error_bounded(&materialized.layers[0].key, &expected_key, 0.01);

        let mut isolated = PagedKvCache::new_with_tensor_config(config, 2);
        let iso_seq = isolated.create_sequence();
        isolated
            .append_token_kv(iso_seq, &borrowed_layers(&tokens[0]))
            .unwrap();
        let iso = isolated.materialize_sequence(iso_seq).unwrap();
        assert_eq!(
            &materialized.layers[0].key[0..4],
            iso.layers[0].key.as_slice()
        );
    }

    #[test]
    fn metadata_rejects_per_channel_quantization_axis() {
        let spec = KvCacheSpec {
            native_dtype: Some("float8_e4m3fn".to_owned()),
            quantization_tolerance: Some(KvQuantTolerance {
                key: Some(KvComponentTolerance {
                    default: None,
                    per_layer: None,
                    quantization_axis: Some("per_channel".to_owned()),
                }),
                value: None,
            }),
            sensitive_layers: None,
            operations: None,
        };
        assert!(matches!(
            KvQuantConfig::from_metadata(&spec, 2),
            Err(KvError::UnsupportedQuantizationAxis(axis)) if axis == "per_channel"
        ));

        // per_token (and an unspecified axis) remain accepted.
        let per_token = KvCacheSpec {
            quantization_tolerance: Some(KvQuantTolerance {
                key: Some(KvComponentTolerance {
                    default: None,
                    per_layer: None,
                    quantization_axis: Some("per_token".to_owned()),
                }),
                value: None,
            }),
            ..spec
        };
        assert!(KvQuantConfig::from_metadata(&per_token, 2).is_ok());
    }

    #[test]
    fn tensor_write_rejects_unconfigured_invalid_shape_and_position() {
        let mut unconfigured = PagedKvCache::new(2, 1);
        let seq = unconfigured.create_sequence();
        let token = layers(0.0);
        assert!(matches!(
            unconfigured.append_token_kv(seq, &borrowed_layers(&token)),
            Err(KvError::TensorStorageNotConfigured)
        ));

        let mut cache = PagedKvCache::new_with_tensor_config(config(), 2);
        let seq = cache.create_sequence();
        let missing_layer = &borrowed_layers(&token)[..1];
        assert!(matches!(
            cache.append_token_kv(seq, missing_layer),
            Err(KvError::InvalidTensorShape("wrong number of layers"))
        ));

        let malformed = vec![
            LayerKv {
                key: &[1.0],
                value: &[1.0],
            },
            LayerKv {
                key: &[1.0],
                value: &[1.0],
            },
        ];
        assert!(matches!(
            cache.append_token_kv(seq, &malformed),
            Err(KvError::InvalidTensorShape(_))
        ));
        assert!(matches!(
            cache.write_token_kv(seq, 1, &borrowed_layers(&token)),
            Err(KvError::InvalidPosition {
                position: 1,
                length: 0
            })
        ));
    }

    #[test]
    fn int8_rewrite_after_fork_is_copy_on_write() {
        let mut cache = PagedKvCache::new_with_tensor_config(small_config(KvDType::Int8), 2);
        let source = cache.create_sequence();
        let original = small_layers([-1.0, -0.5, 0.5, 1.0]);
        cache
            .append_token_kv(source, &borrowed_layers(&original))
            .unwrap();
        let forked = cache.fork(source, 1).unwrap();
        let replacement = small_layers([2.0, 3.0, 4.0, 5.0]);

        cache
            .write_token_kv(forked, 0, &borrowed_layers(&replacement))
            .unwrap();

        let source_page = cache.page_table.get_sequence(source).unwrap()[0];
        let forked_page = cache.page_table.get_sequence(forked).unwrap()[0];
        assert_ne!(source_page, forked_page);
        assert_close(
            &cache.materialize_sequence(source).unwrap().layers[0].key,
            &original[0].0,
            0.05,
        );
        assert_close(
            &cache.materialize_sequence(forked).unwrap().layers[0].key,
            &replacement[0].0,
            0.05,
        );
    }

    #[test]
    fn eviction_and_prefetch_cover_empty_and_invalid_ranges() {
        for policy in [
            EvictionPolicy::Lru,
            EvictionPolicy::Priority,
            EvictionPolicy::LayerAware,
        ] {
            let mut cache = PagedKvCache::new(1, 2);
            let seq = cache.create_sequence();
            cache.append(seq, 2).unwrap();
            assert_eq!(cache.evict(policy, 3), 2);
        }

        let mut cache = PagedKvCache::new(2, 1);
        let seq = cache.create_sequence();
        cache.append(seq, 1).unwrap();
        assert_eq!(cache.prefetch(seq, 1, 1).unwrap(), 0);
        assert!(matches!(
            cache.prefetch(seq, 1, 0),
            Err(KvError::InvalidPosition {
                position: 0,
                length: 1
            })
        ));
        assert!(matches!(
            cache.prefetch(seq, 0, 2),
            Err(KvError::InvalidPosition {
                position: 2,
                length: 1
            })
        ));
    }

    #[test]
    fn preempt_then_restore_keeps_kv_bit_identical() {
        let configs = heterogeneous_layer_configs();
        // page_size 2 with 5 tokens spans multiple pages, so preemption must
        // move every backing page — not just the tail.
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(2, KvDType::F32, configs.clone(), 8);
        let seq = cache.create_sequence();
        let tokens = (0..5)
            .map(|t| hetero_token(&configs, t))
            .collect::<Vec<_>>();
        for token in &tokens {
            cache.append_token_kv(seq, &borrowed_layers(token)).unwrap();
        }
        let before = cache.materialize_sequence(seq).unwrap();
        let page_count = cache.page_table.get_sequence(seq).unwrap().len();
        assert!(page_count > 1, "test should exercise multiple pages");
        assert_eq!(cache.sequence_hot_pages(seq), page_count);

        // Preempt: every exclusively-owned page leaves the hot tier.
        let demoted = cache.preempt_sequence(seq).unwrap();
        assert_eq!(demoted, page_count);
        assert_eq!(cache.sequence_hot_pages(seq), 0);

        // KV is fully readable while cold and byte-identical to before.
        let while_cold = cache.materialize_sequence(seq).unwrap();
        assert_eq!(while_cold, before);

        // Restore: pages return to the hot tier, KV still byte-identical.
        let promoted = cache.restore_sequence(seq).unwrap();
        assert_eq!(promoted, page_count);
        assert_eq!(cache.sequence_hot_pages(seq), page_count);
        let after = cache.materialize_sequence(seq).unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn preempt_leaves_shared_prefix_pages_resident() {
        let configs = heterogeneous_layer_configs();
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(2, KvDType::F32, configs.clone(), 8);
        let seq = cache.create_sequence();
        for token in (0..4).map(|t| hetero_token(&configs, t)) {
            cache
                .append_token_kv(seq, &borrowed_layers(&token))
                .unwrap();
        }
        // Pin the first page as if it were a retained shared prefix.
        let shared_page = cache.page_table.get_sequence(seq).unwrap()[0];
        assert!(cache.page_table.retain(shared_page));

        let hot_before = cache.sequence_hot_pages(seq);
        let demoted = cache.preempt_sequence(seq).unwrap();
        // The shared page stays resident; only the exclusively-owned pages move.
        assert_eq!(demoted, hot_before - 1);
        assert!(matches!(
            cache
                .page_table
                .pages
                .get(&shared_page)
                .unwrap()
                .residency(),
            Device::Gpu(_)
        ));
    }

    #[test]
    fn preempt_and_restore_are_noops_for_empty_sequence() {
        let mut cache = PagedKvCache::new(2, 4);
        let seq = cache.create_sequence();
        assert_eq!(cache.preempt_sequence(seq).unwrap(), 0);
        assert_eq!(cache.restore_sequence(seq).unwrap(), 0);
    }

    #[test]
    fn preempt_unknown_sequence_errors() {
        let mut cache = PagedKvCache::new(2, 4);
        assert!(matches!(
            cache.preempt_sequence(999),
            Err(KvError::SequenceNotFound(999))
        ));
    }

    #[test]
    fn head_token_row_borrows_pages_in_place_across_boundaries() {
        // config(): 2 layers, 2 kv heads, head_dim 3, page_size 2. Appending 5
        // tokens spans 3 pages, so this crosses page boundaries.
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 8);
        let seq = cache.create_sequence();
        for token in 0..5u32 {
            let data = layers(token as f32);
            cache.append_token_kv(seq, &borrowed_layers(&data)).unwrap();
        }

        // Every (layer, kind, head, position) row must equal the value that was
        // written, and must match the materialized dense tensors element for
        // element (the accessor is the zero-copy equivalent of materialize).
        let materialized = cache.materialize_sequence(seq).unwrap();
        for layer_idx in 0..2 {
            let geom = &materialized.layers[layer_idx];
            for (kind, dense) in [(KvKind::Key, &geom.key), (KvKind::Value, &geom.value)] {
                for head in 0..geom.num_kv_heads {
                    for position in 0..5 {
                        let row = cache
                            .head_token_row(seq, layer_idx, kind, head, position)
                            .unwrap()
                            .expect("f32 cache yields a contiguous row");
                        let base = (head * 5 + position) * geom.head_dim;
                        assert_eq!(
                            row,
                            &dense[base..base + geom.head_dim],
                            "layer {layer_idx} {kind:?} head {head} pos {position}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn head_token_row_rejects_out_of_range_and_quantized() {
        let mut cache = PagedKvCache::new_with_tensor_config(config(), 8);
        let seq = cache.create_sequence();
        let data = layers(1.0);
        cache.append_token_kv(seq, &borrowed_layers(&data)).unwrap();

        // Position at/after the sequence length is out of range.
        assert!(matches!(
            cache.head_token_row(seq, 0, KvKind::Key, 0, 1),
            Err(KvError::InvalidPosition { position: 1, .. })
        ));

        // A quantized (fp8) component has no contiguous f32 row to borrow.
        let mut quant = PagedKvCache::new_with_layer_quant_config(
            2,
            KvDType::F32,
            vec![LayerTensorConfig {
                num_kv_heads: 1,
                head_dim: 4,
            }],
            crate::page_table::KvQuantConfig::homogeneous(KvDType::Fp8E4M3Fn, 1),
            8,
        )
        .unwrap();
        let qseq = quant.create_sequence();
        quant
            .append_token_kv(
                qseq,
                &[LayerKv {
                    key: &[1.0, 2.0, 3.0, 4.0],
                    value: &[5.0, 6.0, 7.0, 8.0],
                }],
            )
            .unwrap();
        assert!(
            quant
                .head_token_row(qseq, 0, KvKind::Key, 0, 0)
                .unwrap()
                .is_none(),
            "quantized component must report no borrowable f32 row"
        );
    }
}

#[cfg(test)]
mod accounting_contract_tests {
    use super::*;
    use crate::{KvCacheOps, KvViewKind};

    fn configs(layers: usize) -> Vec<crate::LayerTensorConfig> {
        (0..layers)
            .map(|_| crate::LayerTensorConfig {
                num_kv_heads: 2,
                head_dim: 32,
            })
            .collect()
    }

    fn cache() -> PagedKvCache {
        PagedKvCache::new_with_layer_tensor_configs(4, crate::KvDType::F32, configs(2), 32)
    }

    /// Summing per-sequence bytes is **not** the store's footprint.
    ///
    /// A forked sequence shares its prefix pages, and `sequence_bytes` counts a
    /// shared page in full for each holder because that is what the sequence
    /// would need on its own. Anything that adds those numbers up to decide how
    /// much memory is in use will over-count -- which is how a budget starts
    /// describing memory that was never allocated. `resident_bytes` counts each
    /// page once and is the number to compare against a lease.
    #[test]
    fn sharing_makes_attributed_bytes_exceed_what_is_actually_resident() {
        let mut cache = cache();
        let base = cache.create_sequence();
        cache.append(base, 8).expect("pool has capacity");
        let resident_before = cache.resident_bytes();

        let forked = cache.fork(base, 8).expect("fork at the full length");

        let attributed = cache.sequence_bytes(base).expect("base exists")
            + cache.sequence_bytes(forked).expect("fork exists");
        let resident_after = cache.resident_bytes();

        assert_eq!(
            resident_after, resident_before,
            "a copy-on-write fork allocated new pages before anything diverged"
        );
        assert!(
            attributed > resident_after,
            "expected attributed bytes ({attributed}) to exceed resident bytes \
             ({resident_after}) once two sequences share pages; if these are equal the \
             fork did not actually share and the test is no longer testing sharing"
        );
        assert_eq!(
            attributed,
            resident_after * 2,
            "both sequences reference the same pages, so attribution should be exactly double"
        );
    }

    /// Resident bytes never exceed what the pool allocated, or the lease is a lie.
    #[test]
    fn resident_bytes_never_exceed_the_pool() {
        let mut cache = cache();
        let mut sequences = Vec::new();
        for _ in 0..6 {
            let seq = cache.create_sequence();
            cache.append(seq, 8).expect("pool has capacity");
            sequences.push(seq);
        }
        assert!(
            cache.resident_bytes() <= cache.page_table.pool_bytes(),
            "resident {} exceeds pool {}",
            cache.resident_bytes(),
            cache.page_table.pool_bytes()
        );
    }

    /// Removing a sequence returns its pages to the pool's free bytes.
    #[test]
    fn removing_a_sequence_frees_its_resident_bytes() {
        let mut cache = cache();
        let empty = cache.resident_bytes();
        let seq = cache.create_sequence();
        cache.append(seq, 8).expect("pool has capacity");
        assert!(cache.resident_bytes() > empty, "append occupied no pages");

        cache.remove(seq).expect("sequence exists");
        assert_eq!(
            cache.resident_bytes(),
            empty,
            "removing a sequence left its pages referenced"
        );
        assert!(
            cache.sequence_bytes(seq).is_err(),
            "a removed sequence still reports bytes"
        );
    }

    /// The store reports the view it can actually provide.
    ///
    /// Claiming `VirtuallyContiguous` before the mapping exists would let a
    /// backend that needs one flat range bind scattered pages.
    #[test]
    fn a_paged_store_does_not_claim_contiguity_it_cannot_provide() {
        assert_eq!(cache().view(), KvViewKind::Paged);
    }
}

#[cfg(test)]
mod materialize_to_tests {
    use super::*;
    use crate::KvCacheOps;

    fn configs() -> Vec<crate::LayerTensorConfig> {
        vec![crate::LayerTensorConfig {
            num_kv_heads: 2,
            head_dim: 4,
        }]
    }

    fn filled_cache(tokens: usize) -> (PagedKvCache, SequenceId) {
        let mut cache =
            PagedKvCache::new_with_layer_tensor_configs(4, crate::KvDType::F32, configs(), 16);
        let seq = cache.create_sequence();
        for token in 0..tokens {
            let key: Vec<f32> = (0..8).map(|i| (token * 100 + i) as f32).collect();
            let value: Vec<f32> = (0..8).map(|i| (token * 100 + i) as f32 + 0.5).collect();
            cache
                .append_token_kv(
                    seq,
                    &[LayerKv {
                        key: &key,
                        value: &value,
                    }],
                )
                .expect("pool has capacity");
        }
        (cache, seq)
    }

    /// Reading at a rewind target must equal rewinding and then reading.
    ///
    /// This is what lets the ORT rewind path drop its whole-pool clone: the
    /// clone existed only so the rewind could be applied to a copy before the
    /// read. If these two ever disagree, that swap silently changes what the
    /// decoder is handed after a rewind.
    #[test]
    fn materializing_at_a_target_equals_rewinding_first_then_materializing() {
        for target in [1usize, 3, 5, 8] {
            let (ahead, seq) = filled_cache(8);
            let read_first = ahead
                .materialize_sequence_to(seq, target)
                .expect("target is within the sequence");

            let (mut rewound, seq2) = filled_cache(8);
            rewound.rewind_to(seq2, target).expect("target is valid");
            let rewind_first = rewound
                .materialize_sequence(seq2)
                .expect("rewound sequence materializes");

            assert_eq!(
                read_first.sequence_len, rewind_first.sequence_len,
                "target {target}: lengths differ"
            );
            assert_eq!(
                read_first.layers[0].key, rewind_first.layers[0].key,
                "target {target}: key data differs"
            );
            assert_eq!(
                read_first.layers[0].value, rewind_first.layers[0].value,
                "target {target}: value data differs"
            );
        }
    }

    /// Reading past the end is refused rather than returning zeros.
    #[test]
    fn materializing_beyond_the_sequence_is_refused() {
        let (cache, seq) = filled_cache(4);
        let error = cache
            .materialize_sequence_to(seq, 5)
            .expect_err("5 exceeds a 4 token sequence");
        assert!(
            matches!(
                error,
                KvError::InvalidPosition {
                    position: 5,
                    length: 4
                }
            ),
            "unexpected error: {error}"
        );
    }

    /// A refused read leaves the sequence exactly as it was.
    ///
    /// This is the guarantee the clone was providing, so it has to survive the
    /// clone's removal.
    #[test]
    fn a_refused_read_does_not_disturb_the_sequence() {
        let (cache, seq) = filled_cache(4);
        let before = cache.len(seq).expect("sequence exists");
        let _ = cache.materialize_sequence_to(seq, 99);
        assert_eq!(
            cache.len(seq).expect("sequence exists"),
            before,
            "a refused read changed the sequence length"
        );
    }
    /// A read below the pinned sink prefix is refused with a reason.
    ///
    /// The rewind itself would be legal, but it resets the window bookkeeping,
    /// so a non-mutating read cannot reproduce its result. Refusing is the safe
    /// direction -- this function never green-lights a position a rewind would
    /// reject -- but a generic "evicted" error would misdescribe it, since
    /// nothing was evicted.
    #[test]
    fn a_read_below_the_pinned_sink_prefix_says_why_it_cannot_be_answered() {
        // Sinks only activate once the window has moved clear of the pinned
        // prefix, so the sequence has to be long enough for a gap to open.
        let (mut cache, seq) = filled_cache(16);
        cache
            .apply_sliding_window_with_sinks(seq, 4, 4)
            .expect("window applies to a 16 token sequence");
        let sink = cache.sink_len(seq).expect("sequence exists");
        assert!(sink > 0, "test needs a sequence with a pinned sink prefix");

        let error = cache
            .materialize_sequence_to(seq, sink - 1)
            .expect_err("a read inside the sink prefix must be refused");
        assert!(
            matches!(error, KvError::RewindBelowSinkNotMaterializable { .. }),
            "expected a sink-specific refusal, got {error}"
        );
        assert!(
            !error.to_string().contains("evicted"),
            "the message calls it an eviction, but nothing was evicted: {error}"
        );
    }

    /// Delegating at the sequence's own length is never refused.
    ///
    /// `materialize_sequence` forwards with `end = len(seq)`, so the new bounds
    /// checks must be unreachable there or existing callers would start failing.
    #[test]
    fn materializing_a_whole_sequence_is_never_refused_by_the_new_bounds() {
        let (mut cache, seq) = filled_cache(16);
        cache
            .apply_sliding_window_with_sinks(seq, 4, 4)
            .expect("window applies");
        cache
            .materialize_sequence(seq)
            .expect("a whole-sequence read must still work with a window applied");
    }
}
