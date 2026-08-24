//! Integration tests for token-major PagedAttention index emission
//! (`onnx-genai-kv` slice 3A.1).
//!
//! These exercise the emitter against a real `PagedKvCache` whose pages are
//! allocated by ordinary appends, so the block ids and slots are genuine
//! physical page assignments — not fixtures. The emitter is a read-only view
//! over the page authority; every test that mutates does so through the normal
//! append path only.

use onnx_genai_kv::{
    KvCacheOps, KvDType, KvError, LayerKv, PAGED_BLOCK_TABLE_PAD, PageTensorConfig, PagedKvCache,
    PagedRequest,
};

fn cache(page_size: usize, num_pages: usize) -> PagedKvCache {
    let config = PageTensorConfig {
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 16,
        page_size,
        dtype: KvDType::F32,
    };
    PagedKvCache::new_with_tensor_config(config, num_pages)
}

/// Append `count` tokens whose head_dim values encode `(position)` so the data
/// can be checked after read-only emission.
fn append_tokens(cache: &mut PagedKvCache, seq: u64, count: usize) {
    for _ in 0..count {
        let pos = cache.len(seq).unwrap();
        let key: Vec<f32> = (0..16).map(|d| (pos * 100 + d) as f32).collect();
        let value: Vec<f32> = (0..16).map(|d| (pos * 100 + d + 50) as f32).collect();
        let layers = [LayerKv {
            key: &key,
            value: &value,
        }];
        cache.append_token_kv(seq, &layers).unwrap();
    }
}

#[test]
fn prefill_single_request_slots_match_physical_pages() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 20); // 20 tokens -> 2 pages (16 + 4)

    let pages = c.page_table.get_sequence(seq).unwrap().to_vec();
    assert_eq!(pages.len(), 2, "20 tokens over block_size 16 needs 2 pages");

    let plan = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 20 }])
        .unwrap();

    assert_eq!(plan.num_seqs(), 1);
    assert_eq!(plan.block_size(), 16);
    assert_eq!(plan.token_count(), 20);
    assert_eq!(plan.max_num_blocks_per_seq(), 2);
    assert_eq!(plan.past_seqlens(), &[0]);
    assert_eq!(plan.context_lens(), &[20]);
    assert_eq!(plan.cumulative_sequence_length(), &[0, 20]);

    // block_table row = physical page ids in logical order.
    assert_eq!(plan.block_table_row(0), &[pages[0] as i32, pages[1] as i32]);

    // slot = phys_block * block_size + offset for every query position.
    let expected: Vec<i32> = (0..20)
        .map(|pos| pages[pos / 16] as i32 * 16 + (pos % 16) as i32)
        .collect();
    assert_eq!(plan.slot_mapping(), expected.as_slice());
}

#[test]
fn decode_step_writes_single_tail_slot() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 17); // crosses a block boundary: pos 16 -> block 1

    let pages = c.page_table.get_sequence(seq).unwrap().to_vec();
    assert_eq!(pages.len(), 2);

    // Decode step: only the last appended token is the query token.
    let plan = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 1 }])
        .unwrap();

    assert_eq!(plan.past_seqlens(), &[16]);
    assert_eq!(plan.context_lens(), &[17]);
    assert_eq!(plan.token_count(), 1);
    // Token at absolute position 16 -> logical block 1, offset 0.
    assert_eq!(plan.slot_mapping(), &[pages[1] as i32 * 16]);
    assert_eq!(plan.max_num_blocks_per_seq(), 2);
}

#[test]
fn multi_request_batch_is_row_major_and_padded() {
    let mut c = cache(16, 64);
    let a = c.create_sequence();
    let b = c.create_sequence();
    append_tokens(&mut c, a, 5); // 1 block
    append_tokens(&mut c, b, 33); // 3 blocks

    let pa = c.page_table.get_sequence(a).unwrap().to_vec();
    let pb = c.page_table.get_sequence(b).unwrap().to_vec();

    let plan = c
        .emit_paged_index_plan(&[
            PagedRequest {
                seq: a,
                query_len: 5,
            },
            PagedRequest {
                seq: b,
                query_len: 33,
            },
        ])
        .unwrap();

    assert_eq!(plan.num_seqs(), 2);
    assert_eq!(plan.max_num_blocks_per_seq(), 3); // widened to the longer seq
    assert_eq!(plan.token_count(), 38);
    assert_eq!(plan.cumulative_sequence_length(), &[0, 5, 38]);
    assert_eq!(plan.past_seqlens(), &[0, 0]);

    // Row A: 1 real block then padding to width 3.
    assert_eq!(
        plan.block_table_row(0),
        &[pa[0] as i32, PAGED_BLOCK_TABLE_PAD, PAGED_BLOCK_TABLE_PAD]
    );
    // Row B: 3 real blocks, no padding.
    assert_eq!(
        plan.block_table_row(1),
        &[pb[0] as i32, pb[1] as i32, pb[2] as i32]
    );

    // slot_mapping is A's 5 slots followed by B's 33 slots, in request order.
    assert_eq!(plan.slot_mapping().len(), 38);
    let expected_a: Vec<i32> = (0..5)
        .map(|p| pa[p / 16] as i32 * 16 + (p % 16) as i32)
        .collect();
    let expected_b: Vec<i32> = (0..33)
        .map(|p| pb[p / 16] as i32 * 16 + (p % 16) as i32)
        .collect();
    assert_eq!(&plan.slot_mapping()[0..5], expected_a.as_slice());
    assert_eq!(&plan.slot_mapping()[5..38], expected_b.as_slice());
}

#[test]
fn block_boundaries_are_exact_for_power_of_two_sizes() {
    for &block_size in &[16usize, 32, 64] {
        let mut c = cache(block_size, 64);
        let seq = c.create_sequence();
        let tokens = block_size * 2 + 1; // exactly 3 blocks, last block 1 token
        append_tokens(&mut c, seq, tokens);
        let pages = c.page_table.get_sequence(seq).unwrap().to_vec();
        assert_eq!(pages.len(), 3);

        let plan = c
            .emit_paged_index_plan(&[PagedRequest {
                seq,
                query_len: tokens,
            }])
            .unwrap();
        assert_eq!(plan.block_size(), block_size);
        assert_eq!(plan.max_num_blocks_per_seq(), 3);
        // Last token sits alone in block 2 at offset 0.
        let last = *plan.slot_mapping().last().unwrap();
        assert_eq!(last, pages[2] as i32 * block_size as i32);
    }
}

#[test]
fn rejects_non_power_of_two_block_size() {
    let mut c = cache(24, 64); // valid page_size, invalid op block_size
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 4);
    let err = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 4 }])
        .unwrap_err();
    assert!(matches!(
        err,
        KvError::PagedInvalidBlockSize { block_size: 24 }
    ));
}

#[test]
fn rejects_block_size_below_minimum() {
    let mut c = cache(8, 64); // power of two but < 16
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 4);
    let err = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 4 }])
        .unwrap_err();
    assert!(matches!(
        err,
        KvError::PagedInvalidBlockSize { block_size: 8 }
    ));
}

#[test]
fn rejects_windowed_sequence() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 40);
    // Slide the window so the retained start moves off zero.
    c.apply_sliding_window(seq, 16).unwrap();
    assert!(c.page_table.sequence_start(seq).unwrap() > 0);

    let err = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 1 }])
        .unwrap_err();
    assert!(matches!(err, KvError::PagedNonContiguousSequence { .. }));
}

#[test]
fn rejects_query_longer_than_context() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 3);
    let err = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 5 }])
        .unwrap_err();
    assert!(matches!(
        err,
        KvError::PagedQueryExceedsContext {
            query_len: 5,
            context_len: 3,
            ..
        }
    ));
}

#[test]
fn rejects_missing_pages() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    // Declare a length with no backing pages appended.
    c.page_table.set_sequence_len(seq, 100);
    let err = c
        .emit_paged_index_plan(&[PagedRequest { seq, query_len: 1 }])
        .unwrap_err();
    assert!(matches!(
        err,
        KvError::PagedMissingPages {
            need_pages: 7,
            have_pages: 0,
            ..
        }
    ));
}

#[test]
fn rejects_unknown_sequence() {
    let c = cache(16, 64);
    let err = c
        .emit_paged_index_plan(&[PagedRequest {
            seq: 4242,
            query_len: 1,
        }])
        .unwrap_err();
    assert!(matches!(err, KvError::SequenceNotFound(4242)));
}

#[test]
fn emission_is_read_only_and_leak_free() {
    let mut c = cache(16, 64);
    let seq = c.create_sequence();
    append_tokens(&mut c, seq, 20);

    let usage_before = c.page_table.usage();
    let stats_before = c.page_table.stats();
    let materialized_before = c.materialize_sequence(seq).unwrap();

    // Emit many times; a read-only view must not allocate, free, or mutate.
    for _ in 0..8 {
        let _ = c
            .emit_paged_index_plan(&[PagedRequest { seq, query_len: 20 }])
            .unwrap();
    }

    assert_eq!(
        c.page_table.usage(),
        usage_before,
        "page usage must not change"
    );
    assert_eq!(
        c.page_table.stats(),
        stats_before,
        "pool stats must not change"
    );
    assert_eq!(
        c.materialize_sequence(seq).unwrap(),
        materialized_before,
        "head-major KV data must be byte-identical after emission"
    );
}

#[test]
fn reused_pages_after_free_still_emit_valid_slots() {
    let mut c = cache(16, 32);
    let a = c.create_sequence();
    append_tokens(&mut c, a, 20);
    let free_before = {
        let u = c.page_table.usage();
        u.free
    };
    // Drop the sequence, returning its pages to the free list.
    c.remove(a).unwrap();
    let u_after_remove = c.page_table.usage();
    assert!(u_after_remove.free > free_before);

    // A fresh sequence reuses freed pages; slots must still be well-formed.
    let b = c.create_sequence();
    append_tokens(&mut c, b, 18);
    let pages = c.page_table.get_sequence(b).unwrap().to_vec();
    let plan = c
        .emit_paged_index_plan(&[PagedRequest {
            seq: b,
            query_len: 18,
        }])
        .unwrap();
    let expected: Vec<i32> = (0..18)
        .map(|p| pages[p / 16] as i32 * 16 + (p % 16) as i32)
        .collect();
    assert_eq!(plan.slot_mapping(), expected.as_slice());
}
