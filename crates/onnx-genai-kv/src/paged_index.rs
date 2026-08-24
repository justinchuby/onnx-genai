//! Token-major / LATENT paged-attention index emission (additive).
//!
//! ONNX Runtime's `com.microsoft::PagedAttention` v1 addresses a *token-major*
//! (block-major) KV cache through three integer index tensors, exactly as
//! `paged_attention_helper.h` describes:
//!
//! - `block_table[batch_size, max_num_blocks_per_seq]` (int32): the physical
//!   block id backing each logical block of each sequence.
//! - `slot_mapping[token_count]` (int32): for each *query* token being written
//!   this step, the physical slot `phys_block * block_size + offset_in_block`.
//!   A slot of [`PAGED_SLOT_EMPTY`] means "do not write" (skipped by the op).
//! - `cumulative_sequence_length[batch_size + 1]` (int32): the running sum of
//!   per-sequence query-token counts (`cu_seqlens_q`).
//!
//! plus `past_seqlens[batch_size]` (int32), the number of tokens already
//! resident in each sequence's cache before this step.
//!
//! ## What this module is (and is not)
//!
//! This is a strictly **read-only view over the existing page authority**
//! ([`PageTable`]). It never allocates, frees, retains, or mutates pages, and it
//! never touches the head-major byte layout of a [`Page`](crate::Page). It only
//! translates the authority's *existing* per-sequence page lists and lengths
//! into the integer index tensors the op consumes, so `onnx-genai-kv` stays the
//! single owner of page allocation and lifetime. There is no second manager and
//! no op-side allocation: a native kernel binds these indices as device inputs
//! and reads/writes the caller-owned cache tensors in place.
//!
//! The physical block id is the [`PageId`] itself, because pages are physically
//! scattered — the token-major cache is *virtually contiguous*
//! (`KvViewKind::VirtuallyContiguous`): `slot = page_id * block_size + offset`.
//!
//! LATENT (absorbed multi-head latent attention) is described by
//! [`LatentCacheGeometry`]. The absorbed latent cache content is produced by the
//! caller's MLA export — it is *not* repacked from head-major separate K/V pages
//! (a separate per-head K/V cache cannot be reinterpreted as an absorbed latent).
//! This module only pins the **addressing contract** (block/slot/element offset
//! math) so that a CUDA kernel and the CPU oracle index the token-major/LATENT
//! cache identically.

use crate::{KvError, SequenceId, page_table::PageId, page_table::PageTable};

/// Minimum block size the op accepts (`block_size >= 16`, power of two).
pub const MIN_PAGED_BLOCK_SIZE: usize = 16;

/// Sentinel slot meaning "no physical slot; skip this token" (ORT skips
/// `slot_mapping` entries equal to `-1`).
pub const PAGED_SLOT_EMPTY: i32 = -1;

/// Value written into unused `block_table` entries. These are never
/// dereferenced by a correct kernel, which bounds its logical-block loop by the
/// sequence's context length; the fill value only makes the tensor rectangular.
pub const PAGED_BLOCK_TABLE_PAD: i32 = 0;

/// True when `block_size` is a power of two and at least [`MIN_PAGED_BLOCK_SIZE`]
/// — the exact constraint `check_kv_cache` enforces on `key_cache.dims()[1]`.
#[must_use]
pub fn is_valid_paged_block_size(block_size: usize) -> bool {
    block_size >= MIN_PAGED_BLOCK_SIZE && block_size.is_power_of_two()
}

/// KV cache addressing mode for the emitted paged op inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedKvLayout {
    /// Separate K and V token-major caches, each
    /// `[num_blocks, block_size, kv_num_heads, head_size]`.
    SeparateTokenMajor,
    /// Absorbed MLA: a single cache `[num_blocks, block_size, 1, latent_dim]`
    /// (`kv_num_heads == 1`). K reads the full `latent_dim`; V reads the leading
    /// `v_head_size` channels of the same latent row.
    Latent,
}

/// Geometry of a single absorbed-MLA LATENT cache row.
///
/// Field semantics mirror the `com.microsoft::PagedAttention` v1 schema:
/// `latent_dim` is the cache's `head_size` (the compressed latent width), and V
/// is the leading `v_head_size` channels of that same row. Partial RoPE is
/// applied to the `rotary_dim`-wide suffix beginning at channel `rotary_offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentCacheGeometry {
    /// Tokens per block; must satisfy [`is_valid_paged_block_size`].
    pub block_size: usize,
    /// Latent cache width (`head_size` of the single cache).
    pub latent_dim: usize,
    /// Value head size; `v_head_size <= latent_dim`, V uses leading channels.
    pub v_head_size: usize,
    /// Width of the partial-RoPE suffix (0 = no rotary). Must be even.
    pub rotary_dim: usize,
    /// First channel of the RoPE suffix; `rotary_offset + rotary_dim <= latent_dim`.
    pub rotary_offset: usize,
}

impl LatentCacheGeometry {
    /// Validate the LATENT geometry, returning a typed error rather than
    /// silently addressing out of range.
    pub fn validate(&self) -> Result<(), KvError> {
        if !is_valid_paged_block_size(self.block_size) {
            return Err(KvError::PagedInvalidBlockSize {
                block_size: self.block_size,
            });
        }
        if self.latent_dim == 0 {
            return Err(KvError::LatentGeometryInvalid(
                "latent_dim must be greater than zero".to_owned(),
            ));
        }
        if self.v_head_size == 0 || self.v_head_size > self.latent_dim {
            return Err(KvError::LatentGeometryInvalid(format!(
                "v_head_size ({}) must be in 1..=latent_dim ({})",
                self.v_head_size, self.latent_dim
            )));
        }
        if !self.rotary_dim.is_multiple_of(2) {
            return Err(KvError::LatentGeometryInvalid(format!(
                "rotary_dim ({}) must be even",
                self.rotary_dim
            )));
        }
        // Mirror the op's rotary-cache contract: `rotary_dim = cos_cache.dims[1]
        // * 2` with `cos_cache.dims[1] % 8 == 0` (see `check_rotary_caches`), so
        // a non-zero rotary suffix is a multiple of 16. Reject a geometry the op
        // would reject downstream rather than deferring the failure.
        if self.rotary_dim != 0 && !self.rotary_dim.is_multiple_of(16) {
            return Err(KvError::LatentGeometryInvalid(format!(
                "rotary_dim ({}) must be a multiple of 16 (cos/sin cache last dim is rotary_dim/2 \
                 and must be a multiple of 8)",
                self.rotary_dim
            )));
        }
        if self.rotary_offset + self.rotary_dim > self.latent_dim {
            return Err(KvError::LatentGeometryInvalid(format!(
                "rotary suffix (offset {} + dim {}) exceeds latent_dim {}",
                self.rotary_offset, self.rotary_dim, self.latent_dim
            )));
        }
        Ok(())
    }
}

/// Flat element offset of `channel` for the token at `(block_id, offset_in_block)`
/// and head `kv_head` in a token-major cache
/// `[num_blocks, block_size, kv_num_heads, head_size]`.
///
/// This is the single canonical addressing formula shared by the emitter, the
/// CPU oracle, and a native kernel, so all three index the same cache byte.
#[must_use]
pub fn token_major_element_offset(
    block_size: usize,
    kv_num_heads: usize,
    head_size: usize,
    block_id: usize,
    offset_in_block: usize,
    kv_head: usize,
    channel: usize,
) -> usize {
    (((block_id * block_size + offset_in_block) * kv_num_heads + kv_head) * head_size) + channel
}

/// Flat element offset of `channel` for the LATENT token at
/// `(block_id, offset_in_block)`. LATENT is the `kv_num_heads == 1` special case
/// of [`token_major_element_offset`].
#[must_use]
pub fn latent_element_offset(
    geom: &LatentCacheGeometry,
    block_id: usize,
    offset_in_block: usize,
    channel: usize,
) -> usize {
    token_major_element_offset(
        geom.block_size,
        1,
        geom.latent_dim,
        block_id,
        offset_in_block,
        0,
        channel,
    )
}

/// One request in a paged batch: `query_len` new tokens are being processed for
/// sequence `seq` this step. The tokens are assumed already appended to the
/// page authority (their pages allocated), so their slots are backed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedRequest {
    pub seq: SequenceId,
    pub query_len: usize,
}

/// The emitted paged index tensors for one batched attention step.
///
/// All buffers are host-side `i32`, matching the op's int32 index inputs. The
/// native EP uploads them once to stable device buffers and reuses those across
/// CUDA-graph replay; re-emit only when the page assignment changes. Producing
/// them here keeps `onnx-genai-kv` the sole page authority while leaving device
/// residency to the EP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedIndexPlan {
    /// Row-major `[num_seqs, max_num_blocks_per_seq]`.
    block_table: Vec<i32>,
    /// `[token_count]`, one physical slot per query token (row-major over the
    /// batch in request order).
    slot_mapping: Vec<i32>,
    /// `[num_seqs + 1]` cumulative query-token counts (`cu_seqlens_q`).
    cumulative_sequence_length: Vec<i32>,
    /// `[num_seqs]` tokens already resident before this step.
    past_seqlens: Vec<i32>,
    /// `[num_seqs]` total attended length (`past + query`). Convenience derived
    /// value; the op recomputes it from `past_seqlens` + `cu_seqlens`.
    context_lens: Vec<i32>,
    num_seqs: usize,
    block_size: usize,
    max_num_blocks_per_seq: usize,
    token_count: usize,
}

impl PagedIndexPlan {
    /// Build the paged index tensors for `requests` from the page authority.
    ///
    /// Read-only: it consults [`PageTable::get_sequence`],
    /// [`PageTable::sequence_len`], [`PageTable::sequence_start`], and
    /// [`PageTable::sequence_sink_len`] and never mutates the table.
    ///
    /// Rejections are typed rather than silent:
    /// - non-power-of-two or `< 16` `page_size` → [`KvError::PagedInvalidBlockSize`];
    /// - a windowed / attention-sink sequence (non-contiguous positions) →
    ///   [`KvError::PagedNonContiguousSequence`] (unsupported in this slice);
    /// - `query_len` greater than the sequence's current length →
    ///   [`KvError::PagedQueryExceedsContext`];
    /// - fewer allocated pages than the context needs →
    ///   [`KvError::PagedMissingPages`];
    /// - a page id or slot that does not fit in `i32` →
    ///   [`KvError::PagedBlockIdOverflow`] / [`KvError::PagedSlotOverflow`].
    pub fn build(table: &PageTable, requests: &[PagedRequest]) -> Result<Self, KvError> {
        let block_size = table.page_size;
        if !is_valid_paged_block_size(block_size) {
            return Err(KvError::PagedInvalidBlockSize { block_size });
        }

        let num_seqs = requests.len();
        let mut past_seqlens = Vec::with_capacity(num_seqs);
        let mut context_lens = Vec::with_capacity(num_seqs);
        let mut cumulative_sequence_length = Vec::with_capacity(num_seqs + 1);
        cumulative_sequence_length.push(0i32);

        // First pass: validate, gather per-seq context/pages, size the tensors.
        let mut max_num_blocks_per_seq = 0usize;
        let mut token_count: usize = 0;
        let mut per_seq: Vec<(&[PageId], usize, usize)> = Vec::with_capacity(num_seqs);
        for req in requests {
            let pages = table
                .get_sequence(req.seq)
                .ok_or(KvError::SequenceNotFound(req.seq))?;
            let context_len = table
                .sequence_len(req.seq)
                .ok_or(KvError::SequenceNotFound(req.seq))?;
            let start = table.sequence_start(req.seq).unwrap_or(0);
            let sink = table.sequence_sink_len(req.seq).unwrap_or(0);
            if start != 0 || sink != 0 {
                return Err(KvError::PagedNonContiguousSequence {
                    seq: req.seq,
                    start,
                    sink_len: sink,
                });
            }
            if req.query_len > context_len {
                return Err(KvError::PagedQueryExceedsContext {
                    seq: req.seq,
                    query_len: req.query_len,
                    context_len,
                });
            }
            let blocks_needed = context_len.div_ceil(block_size);
            if pages.len() < blocks_needed {
                return Err(KvError::PagedMissingPages {
                    seq: req.seq,
                    need_pages: blocks_needed,
                    have_pages: pages.len(),
                });
            }

            let past = context_len - req.query_len;
            past_seqlens.push(i32_from_usize(past, || KvError::PagedSlotOverflow {
                slot: past as i64,
            })?);
            context_lens.push(i32_from_usize(context_len, || {
                KvError::PagedSlotOverflow {
                    slot: context_len as i64,
                }
            })?);
            token_count += req.query_len;
            cumulative_sequence_length.push(i32_from_usize(token_count, || {
                KvError::PagedSlotOverflow {
                    slot: token_count as i64,
                }
            })?);
            max_num_blocks_per_seq = max_num_blocks_per_seq.max(blocks_needed);
            per_seq.push((pages, context_len, past));
        }

        // Second pass: fill block_table and slot_mapping now that widths are known.
        let mut block_table = vec![PAGED_BLOCK_TABLE_PAD; num_seqs * max_num_blocks_per_seq];
        let mut slot_mapping = Vec::with_capacity(token_count);
        for (seq_idx, (pages, context_len, past)) in per_seq.iter().enumerate() {
            let blocks_needed = context_len.div_ceil(block_size);
            let row = seq_idx * max_num_blocks_per_seq;
            for logical_block in 0..blocks_needed {
                let page_id = pages[logical_block];
                block_table[row + logical_block] = i32::try_from(page_id)
                    .map_err(|_| KvError::PagedBlockIdOverflow { page_id })?;
            }
            for pos in *past..*context_len {
                let logical_block = pos / block_size;
                let offset_in_block = pos % block_size;
                let page_id = pages[logical_block];
                let slot = page_id as i64 * block_size as i64 + offset_in_block as i64;
                let slot = i32::try_from(slot).map_err(|_| KvError::PagedSlotOverflow { slot })?;
                slot_mapping.push(slot);
            }
        }

        Ok(Self {
            block_table,
            slot_mapping,
            cumulative_sequence_length,
            past_seqlens,
            context_lens,
            num_seqs,
            block_size,
            max_num_blocks_per_seq,
            token_count,
        })
    }

    /// Row-major `[num_seqs, max_num_blocks_per_seq]` block table.
    #[must_use]
    pub fn block_table(&self) -> &[i32] {
        &self.block_table
    }

    /// The block-table row for one sequence in the batch.
    #[must_use]
    pub fn block_table_row(&self, seq_index: usize) -> &[i32] {
        let start = seq_index * self.max_num_blocks_per_seq;
        &self.block_table[start..start + self.max_num_blocks_per_seq]
    }

    /// `[token_count]` physical slots for the query tokens, in batch order.
    #[must_use]
    pub fn slot_mapping(&self) -> &[i32] {
        &self.slot_mapping
    }

    /// `[num_seqs + 1]` cumulative query-token counts (`cu_seqlens_q`).
    #[must_use]
    pub fn cumulative_sequence_length(&self) -> &[i32] {
        &self.cumulative_sequence_length
    }

    /// `[num_seqs]` per-sequence past lengths (tokens already cached).
    #[must_use]
    pub fn past_seqlens(&self) -> &[i32] {
        &self.past_seqlens
    }

    /// `[num_seqs]` per-sequence total attended lengths (`past + query`).
    #[must_use]
    pub fn context_lens(&self) -> &[i32] {
        &self.context_lens
    }

    #[must_use]
    pub fn num_seqs(&self) -> usize {
        self.num_seqs
    }

    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[must_use]
    pub fn max_num_blocks_per_seq(&self) -> usize {
        self.max_num_blocks_per_seq
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }
}

fn i32_from_usize(v: usize, err: impl FnOnce() -> KvError) -> Result<i32, KvError> {
    i32::try_from(v).map_err(|_| err())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_validation_matches_op_contract() {
        assert!(is_valid_paged_block_size(16));
        assert!(is_valid_paged_block_size(32));
        assert!(is_valid_paged_block_size(256));
        assert!(!is_valid_paged_block_size(8)); // below minimum
        assert!(!is_valid_paged_block_size(24)); // not power of two
        assert!(!is_valid_paged_block_size(0));
    }

    #[test]
    fn latent_offset_is_token_major_special_case() {
        let geom = LatentCacheGeometry {
            block_size: 16,
            latent_dim: 192,
            v_head_size: 128,
            rotary_dim: 64,
            rotary_offset: 128,
        };
        geom.validate().unwrap();
        // slot = block 2, offset 3 => (2*16+3)=35; channel 7 => 35*192 + 7.
        assert_eq!(latent_element_offset(&geom, 2, 3, 7), 35 * 192 + 7);
        assert_eq!(
            latent_element_offset(&geom, 2, 3, 7),
            token_major_element_offset(16, 1, 192, 2, 3, 0, 7)
        );
    }

    #[test]
    fn latent_geometry_rejects_bad_shapes() {
        let base = LatentCacheGeometry {
            block_size: 16,
            latent_dim: 192,
            v_head_size: 128,
            rotary_dim: 64,
            rotary_offset: 128,
        };
        assert!(matches!(
            LatentCacheGeometry {
                block_size: 24,
                ..base
            }
            .validate(),
            Err(KvError::PagedInvalidBlockSize { .. })
        ));
        assert!(matches!(
            LatentCacheGeometry {
                v_head_size: 256,
                ..base
            }
            .validate(),
            Err(KvError::LatentGeometryInvalid(_))
        ));
        assert!(matches!(
            LatentCacheGeometry {
                rotary_dim: 63,
                ..base
            }
            .validate(),
            Err(KvError::LatentGeometryInvalid(_))
        ));
        // Even but not a multiple of 16 → rejected (mirrors the op's cos/sin
        // cache contract).
        assert!(matches!(
            LatentCacheGeometry {
                rotary_dim: 8,
                ..base
            }
            .validate(),
            Err(KvError::LatentGeometryInvalid(_))
        ));
        // A multiple of 16, and zero (no rotary), are both accepted.
        assert!(
            LatentCacheGeometry {
                rotary_dim: 32,
                ..base
            }
            .validate()
            .is_ok()
        );
        assert!(
            LatentCacheGeometry {
                rotary_dim: 0,
                rotary_offset: 0,
                ..base
            }
            .validate()
            .is_ok()
        );
        assert!(matches!(
            LatentCacheGeometry {
                rotary_offset: 160,
                rotary_dim: 64,
                ..base
            }
            .validate(),
            Err(KvError::LatentGeometryInvalid(_))
        ));
    }
}
