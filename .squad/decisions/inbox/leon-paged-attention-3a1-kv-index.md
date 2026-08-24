# Decision: PagedAttention slice 3A.1 — token-major/LATENT paged index emission in `onnx-genai-kv`

- **Author:** Leon (Engine Dev, KV & Buffers)
- **Date:** 2026-08-24
- **Follows:** #1940 (merged: corrected audit + typed validator/oracle crate `onnx-genai-paged-attention`).
- **Branch:** `squad/paged-attention-3a-cuda-latent` (based on `origin/main` @ `387f840b0`).
- **Status:** NOT merged. Draft PR; independent review pending (reviewer excludes Leon; Gaff or Roy final approval).

## What this slice lands (the KV-authority half of one-authority integration)

Slice 3A as briefed (native CUDA `com.microsoft::PagedAttention` LATENT kernel +
KV APIs + full parity/capture test matrix + A100 measurement) is larger than one
safe, verifiable increment. Per the "land in order / no dead code / each slice
independently reviewed" directive, this commit lands **Requirement 1 only**: the
additive **token-major / LATENT page-index emission** in `onnx-genai-kv`. It is a
strict prerequisite for the kernel and is **fully validated on host** (no GPU).

New, strictly additive, read-only surface (`crates/onnx-genai-kv/src/paged_index.rs`):

- `PagedIndexPlan::build(&PageTable, &[PagedRequest])` emits the exact ORT v1
  index tensors from the existing page authority:
  - `block_table[num_seqs, max_num_blocks_per_seq]` (i32) — physical `PageId`
    per logical block, padded with `PAGED_BLOCK_TABLE_PAD`;
  - `slot_mapping[token_count]` (i32) — `page_id * block_size + offset` per query
    token; `PAGED_SLOT_EMPTY == -1` is the documented skip sentinel;
  - `cumulative_sequence_length[num_seqs + 1]` (i32, `cu_seqlens_q`);
  - `past_seqlens[num_seqs]` (i32) and derived `context_lens[num_seqs]`.
- `LatentCacheGeometry { block_size, latent_dim, v_head_size, rotary_dim,
  rotary_offset }` + `validate()`, and the canonical addressing functions
  `token_major_element_offset(...)` / `latent_element_offset(...)` that a CUDA
  kernel and the CPU oracle will both index through (one formula, no drift).
- `is_valid_paged_block_size` = power-of-two **≥ 16** (matches `check_kv_cache`).
- Convenience: `PagedKvCache::emit_paged_index_plan(&self, &[PagedRequest])`.

**Byte-identity:** nothing in `Page`/`PageTable` storage changed. The existing
head-major layout `head*(page_size*head_dim) + token*head_dim + dim` is untouched;
a read-only-and-leak-free test asserts `materialize_sequence` is identical before
and after emission and that pool `usage()`/`stats()` are unchanged.

**Typed rejections (never silent miscompute):** non-pow2/`<16` block size,
windowed/attention-sink sequences (non-contiguous positions — deferred), query
longer than context, missing backing pages, and i32 block-id/slot overflow each
return a distinct `KvError` variant, all covered by tests.

## One-authority invariant (unchanged, now enforced in code)

> `onnx-genai-kv` remains the **sole** owner of page allocation and lifetime.
> `paged_index` is a read-only *view*: it allocates/frees/mutates nothing and
> introduces no second manager. A native kernel binds these host-emitted indices
> as stable device inputs and updates the caller-owned cache tensors in place
> (`key_cache_out` aliases `key_cache`). The physical block id **is** the
> `PageId` (`KvViewKind::VirtuallyContiguous`): `slot = page_id*block_size + off`.

## "No dead code" note (pre-empting the review objection)

There is no production consumer of the emitter *yet* — the native CUDA kernel
(slice 3A.2) is its first caller. This mirrors the crate's existing
`kv_capacity_bucket` / `ensure_kv_capacity` / `KvCapacityGrowthBackend` seam,
which is likewise a policy/authority surface exercised by tests and consumed by
backends. The emitter is the KV-authority half of the one-authority contract; its
consumer lands in the immediately-following, independently-reviewed slice.

## Tests / validation

- 3 unit + 12 integration tests (all green); full crate suite 147+12+... passes;
  `cargo clippy -p onnx-genai-kv --tests` clean.
- Covered: prefill/decode slot math, exact pow2 block boundaries (16/32/64),
  multi-request row-major + padding, page reuse after free, read-only/leak-free,
  and all typed rejections.

## Remaining gates (next slices, each independently reviewed)

1. **3A.2 — native CUDA LATENT kernel + claim wiring** (design below). Consumes
   `PagedIndexPlan`; from equations/oracle, capture+replay safe, stable buffers,
   no host sync/alloc in capture, in-place cache alias enforced by pointer
   equality (as `kv_cache_capacity_append.rs` does). Quantized cache modes:
   typed `NotImplemented` + tests. Then A100 measurement (n≥3).
2. **3B — Mobius `--paged-attention`** opt-in export (only after 3A approval).

### 3A.2 CUDA kernel design (grounded in the EP survey; not yet implemented)

- Register `OpKey::new("PagedAttention", "com.microsoft", 1)` in
  `kernels/mod.rs`; `supports_op` returns `KernelMatch::unsupported(reason)` for
  every mode outside the GLM LATENT fp16/bf16 subset (reusing the merged
  `onnx-genai-paged-attention` validator), `Supported` only when it proves out.
- Kernel reads session-provided device tensors via `data_ptr()`→`cuptr`; enforces
  `key_cache_out` aliases `key_cache` by pointer equality; partial RoPE on the
  `rotary_dim` suffix at `rotary_offset` (extends `rotary_embedding.rs`, which
  today rotates from channel 0 only); NVRTC decode kernel mirroring
  `gqa_decode_{fp16,bf16}.rs` (online softmax, `__half`/`__nv_bfloat16`), single
  latent cache, `kv_num_heads==1`, V = leading `v_head_size` channels.
- Capture-safety via warmed fixed signature + sticky device-error latch; no
  host allocation during capture; index tensors uploaded once to stable buffers.
- Model-agnostic: all dims from attributes/inputs (EP §15.1 / `mod.rs` hard rule).
