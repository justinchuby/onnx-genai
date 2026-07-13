//! Paged KV cache implementation.

use crate::{
    CacheCheckpoint, Device, EvictionPolicy, KvCacheOps, KvError, SequenceId,
    page_table::{KvKind, KvQuantConfig, PageId, PageTable, PageTensorConfig},
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
/// `[num_kv_heads, sequence_len, head_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedLayerKv {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
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
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub layers: Vec<MaterializedLayerKv>,
}

/// Paged KV cache manager.
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
        let config = self
            .page_table
            .tensor_config
            .ok_or(KvError::TensorStorageNotConfigured)?;
        self.validate_layers(config, layers)?;

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
                for head in 0..config.num_kv_heads {
                    let src = head * config.head_dim;
                    let key = &layer.key[src..src + config.head_dim];
                    let value = &layer.value[src..src + config.head_dim];
                    page.write_head_token(config, layer_idx * 2, head, token_offset, key);
                    page.write_head_token(config, layer_idx * 2 + 1, head, token_offset, value);
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
        let config = self
            .page_table
            .tensor_config
            .ok_or(KvError::TensorStorageNotConfigured)?;
        let start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        let end = self.len(seq)?;
        // Contiguous buffer holds the pinned sink prefix followed by the window.
        let len = sink + (end - start);
        let pages = self
            .page_table
            .get_sequence(seq)
            .ok_or(KvError::SequenceNotFound(seq))?;
        let per_layer_len = config.num_kv_heads * len * config.head_dim;
        let mut layers = (0..config.num_layers)
            .map(|_| MaterializedLayerKv {
                key: vec![0.0; per_layer_len],
                value: vec![0.0; per_layer_len],
            })
            .collect::<Vec<_>>();

        for token_pos in 0..len {
            let page_index = token_pos / config.page_size;
            let token_offset = token_pos % config.page_size;
            let page_id = pages[page_index];
            let page = self
                .page_table
                .pages
                .get(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?;
            for (layer_idx, layer_out) in layers.iter_mut().enumerate() {
                for head in 0..config.num_kv_heads {
                    for dim in 0..config.head_dim {
                        let dst = (head * len + token_pos) * config.head_dim + dim;
                        let key_src = self
                            .page_table
                            .tensor_offset(layer_idx, KvKind::Key, head, token_offset, dim)
                            .expect("validated offset");
                        let value_src = self
                            .page_table
                            .tensor_offset(layer_idx, KvKind::Value, head, token_offset, dim)
                            .expect("validated offset");
                        layer_out.key[dst] = page.value_at(config, key_src);
                        layer_out.value[dst] = page.value_at(config, value_src);
                    }
                }
            }
        }

        Ok(MaterializedKv {
            start_position: start,
            sink_len: sink,
            sequence_len: len,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
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
            let page_count = self
                .page_table
                .get_sequence(seq)
                .map_or(0, |p| p.len());
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
                .is_some_and(|page| !matches!(page.device, Device::Gpu(_)));
            self.page_table.promote_to_hot(page_id)?;
            if was_cold {
                promoted += 1;
            }
        }
        Ok(promoted)
    }

    fn validate_layers(
        &self,
        config: PageTensorConfig,
        layers: &[LayerKv<'_>],
    ) -> Result<(), KvError> {
        if layers.len() != config.num_layers {
            return Err(KvError::InvalidTensorShape("wrong number of layers"));
        }
        let expected = config.f32_len_per_token_per_layer();
        if layers
            .iter()
            .any(|layer| layer.key.len() != expected || layer.value.len() != expected)
        {
            return Err(KvError::InvalidTensorShape(
                "layer key/value length must be num_kv_heads * head_dim",
            ));
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
                (
                    old.data.clone(),
                    old.quantized_data.clone(),
                    old.fp8_data.clone(),
                    old.quant_scales.clone(),
                    old.filled,
                )
            };
            if let Some(new_page) = self.page_table.pages.get_mut(&new_page_id) {
                new_page.data = old_storage.0;
                new_page.quantized_data = old_storage.1;
                new_page.fp8_data = old_storage.2;
                new_page.quant_scales = old_storage.3;
                new_page.filled = old_storage.4;
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
}

impl KvCacheOps for PagedKvCache {
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError> {
        let retained_start = self.retained_start(seq)?;
        let sink = self.sink_len(seq)?;
        let length = self.len(seq)?;
        // Positions in the pinned sink prefix [0, sink) are physically retained
        // and are valid rewind targets. Only the evicted gap
        // [sink, retained_start) must be rejected.
        if position < retained_start && (sink == 0 || position >= sink) {
            return Err(KvError::PositionEvicted {
                position,
                retained_start,
            });
        }
        if position > length {
            return Err(KvError::InvalidPosition { position, length });
        }

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
    use crate::{KvDType, PageTensorConfig};
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
        assert!(retained_start > 2, "gap must be open for the test to be meaningful");

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

        assert_eq!(cache.page_table.pages[&first_page].device, Device::Cpu);
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
        assert_eq!(cache.page_table.pages[&pages[0]].device, Device::Cpu);
        assert_eq!(cache.prefetch(seq, 0, 1).unwrap(), 1);

        assert_eq!(cache.page_table.pages[&pages[0]].device, Device::Gpu(0));
        assert_eq!(cache.page_table.hot_used_count(), 2);
        assert!(
            pages[1..]
                .iter()
                .any(|page_id| cache.page_table.pages[page_id].device == Device::Cpu)
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

        assert_eq!(cache.page_table.pages[&pages[0]].device, Device::Gpu(0));
        assert_eq!(cache.page_table.pages[&pages[1]].device, Device::Cpu);
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
        assert!(page.data.is_empty());
        assert_eq!(
            page.quantized_data.len(),
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
        assert!(page.data.is_empty());
        assert!(page.quantized_data.is_empty());
        assert_eq!(page.fp8_data.len(), config.f32_len_per_page());
        assert_eq!(page.quant_scales.len(), 4);
        assert_close(
            &page.quant_scales,
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
        assert_eq!(page.data.len(), 8);
        assert_eq!(page.fp8_data.len(), 8);
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
                .any(|id| cache.page_table.pages[id].device == Device::Cpu)
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
}
