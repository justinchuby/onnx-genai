//! Paged KV cache implementation.

use crate::{
    CacheCheckpoint, Device, EvictionPolicy, KvCacheOps, KvError, SequenceId,
    page_table::{KvKind, PageId, PageTable, PageTensorConfig},
};

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
        if position > len {
            return Err(KvError::InvalidPosition {
                position,
                length: len,
            });
        }

        let page_index = position / self.page_table.page_size;
        let token_offset = position % self.page_table.page_size;
        let page_id = self.ensure_page_for_write(seq, page_index)?;

        {
            let page = self
                .page_table
                .pages
                .get_mut(&page_id)
                .ok_or(KvError::PageNotFound(page_id))?;
            for (layer_idx, layer) in layers.iter().enumerate() {
                for head in 0..config.num_kv_heads {
                    for dim in 0..config.head_dim {
                        let src = head * config.head_dim + dim;
                        let k_offset = (((layer_idx * 2) * config.num_kv_heads + head)
                            * config.page_size
                            + token_offset)
                            * config.head_dim
                            + dim;
                        let v_offset = (((layer_idx * 2 + 1) * config.num_kv_heads + head)
                            * config.page_size
                            + token_offset)
                            * config.head_dim
                            + dim;
                        page.data[k_offset] = layer.key[src];
                        page.data[v_offset] = layer.value[src];
                    }
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
        let len = self.len(seq)?;
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
                        layer_out.key[dst] = page.data[key_src];
                        layer_out.value[dst] = page.data[value_src];
                    }
                }
            }
        }

        Ok(MaterializedKv {
            sequence_len: len,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            layers,
        })
    }

    /// Evict pages to free memory. Returns number of pages freed.
    pub fn evict(&mut self, _policy: EvictionPolicy, _target: usize) -> usize {
        // TODO: implement eviction strategies
        0
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
                return Ok(page_id);
            }

            let new_page_id =
                self.page_table
                    .allocate(Device::Gpu(0))
                    .ok_or_else(|| KvError::OutOfMemory {
                        needed: 1,
                        available: self.page_table.free_count(Device::Gpu(0)),
                    })?;
            let (old_data, old_filled) = {
                let old = self
                    .page_table
                    .pages
                    .get(&page_id)
                    .ok_or(KvError::PageNotFound(page_id))?;
                (old.data.clone(), old.filled)
            };
            if let Some(new_page) = self.page_table.pages.get_mut(&new_page_id) {
                new_page.data = old_data;
                new_page.filled = old_filled;
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
}

impl KvCacheOps for PagedKvCache {
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError> {
        let length = self.len(seq)?;
        if position > length {
            return Err(KvError::InvalidPosition { position, length });
        }

        let page_size = self.page_table.page_size;
        let pages_needed = position.div_ceil(page_size);

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
        if position > 0 {
            let last_offset = (position - 1) % page_size + 1;
            if let Some(&last_page_id) = self.page_table.sequences.get(&seq).and_then(|p| p.last())
            {
                if let Some(page) = self.page_table.pages.get_mut(&last_page_id) {
                    page.filled = last_offset;
                }
            }
        }
        self.page_table.set_sequence_len(seq, position);

        Ok(())
    }

    fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId, KvError> {
        let length = self.len(source)?;
        if position > length {
            return Err(KvError::InvalidPosition { position, length });
        }

        let page_size = self.page_table.page_size;
        let pages_needed = position.div_ceil(page_size);
        let source_pages = self
            .page_table
            .get_sequence(source)
            .ok_or(KvError::SequenceNotFound(source))?
            .iter()
            .copied()
            .take(pages_needed)
            .collect::<Vec<_>>();

        let new_seq = self.create_sequence();
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
        self.rewind_to(seq, checkpoint.position)
    }

    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError> {
        let length = self.len(seq)?;
        let page_size = self.page_table.page_size;
        for position in length..length + num_tokens {
            let page_index = position / page_size;
            let token_offset = position % page_size;
            let page_id = self.ensure_page_for_write(seq, page_index)?;
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
}
