# Rachael — DeepSeek MLA capture: (c1) device valid-length ABI (prefill+decode) landed & verified

**Branch:** `perf/deepseek-mla-capture` @ `9ab1c6e` (stacked on merged main `53afab0`; commits `e14d7df` (a′) + `9ab1c6e` (c1)). Do NOT merge — fresh reviewer verifies the whole device-valid-length ABI at once.

## What the ABI does
The default-domain `('','Attention')` MLA path (DeepSeek-V2-Lite; 27 Attention nodes, no `is_causal` attr, additive mask `Unsqueeze_18` fully encodes causal+padding) now derives its valid **key length from device memory** — mirroring how GQA reads `seqlens_k` — instead of from the KV tensor extent. This is the prerequisite that makes the path capture-eligible (the growing logical extent no longer needs to be the source of truth for length).

- `derive_len` NVRTC kernel scans the additive mask frontier (first fully-masked key) into a device `i32`.
- Optional `dev_len` pointer plumbed `native_decode → executor → StandardAttentionKernel` (`build_kv` + `attention_row` read length from device when non-null).
- **Null-pointer-safe:** when `dev_len` is null, the host path is bit-for-bit unchanged (eager/dense/GQA untouched).

## (c1) fix: robust for BOTH prefill and decode
The original (a′) `derive_len` scanned **row 0** of the mask → correct for single-token decode (returns `total_seq`) but **wrong for multi-token causal prefill** (row 0 is valid only for key 0 → returns 1, not `prompt_len`).

**Fix:** scan the **LAST query row** `i = q_seq-1`. At the final query position the causal+padding frontier equals the total valid key length → returns `total_seq` for decode AND `prompt_len` for prefill.
- `derive_len(mask, mask_kind, key_len, row_base, out_len)` scans `[row_base, row_base+key_len)`.
- `execute()` computes `row_base = last_row * key_len`, `last_row = mask_q.saturating_sub(1)`, `key_len = mask.dims[3]`, `mask_q = mask.dims[2]`. Eligibility relaxed from decode-only to prefill+decode.
- **General**, not DeepSeek-special-cased; gated to the default-domain Attention path via the null `dev_len` fallback.

**Why eager e2e stays bit-identical:** with growing bindings the mask last dim equals `total` (no padding region), so last-row frontier = total = host `total_seq`. Last-row only *differs* from row-0 under a padding region (future fixed-cap, c2/d) or a causal prefill mask — the latter is locked by the new unit test.

## Verification (all pass)
- **Prefill+decode correctness (unit test):** `derive_len_reads_valid_length_from_device_for_prefill_and_decode` — decode→`total`; prefill last-row→`prompt_len`; prefill row-0→1 (guard: fails if it reverts to decode-only / host-shape derivation).
- **Determinism, both exports (greedy, 12 tok, warmups=0):**
  - blk32 ×3 → identical `[8913,13,185,549,19305,280,7239,317,254,28071,13,185]`
  - blk128 ×2 → identical, same seq. Coherent: pos0=8913 ' Paris' — " Paris.\nThe currency of France is the Euro.\n"
- **Dense non-regression:** Qwen2.5-0.5b GQA → `cuda_graph enabled=true captures=4 replays=244 fallbacks=0`, coherent, 358.9 tok/s (GQA path untouched, null `dev_len`).
- **Eager DeepSeek tok/s (64 tok, runs=3, warmups=1, idle GPU 6):** blk32 21.96, blk128 22.39 — unchanged vs ~23 baseline (ABI-only pass; capture OFF as expected).
- **Gate:** `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` = **206/0** (+1 new test over 205); `onnx-runtime-session --features cuda --lib` = **65/0**; `clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` clean.

## Capture status: OFF (expected this pass)
Deferred to the next passes. Remaining plan pinned to file:line:

**(c2) prune the `Unsqueeze_18` growing-mask island + synthesize causal+padding bias in-kernel at fixed capacity** (proven engine-side, no export/Mobius dependency):
- Island: `attention_mask → Shape/Unsqueeze/CumSum → GreaterOrEqual causal → And key-validity → Where{0,-65504}`; `Unsqueeze_18` consumed ONLY by the 27 Attention nodes (re-confirm on both exports before pruning).
- Synthesize additive causal+padding bias in `standard_attention.rs attention_row` from (c1)'s device valid-length + fixed-cap `attention_mask [1,max_len]`. Kernel already does causal masking + valid-length bounding internally, so this makes the redundant growing bias unnecessary.
- IR ops available: `create_named_value`/`add_input`/`remove_nodes`/`node_mut`.

**(d) flip `capture_support() → Supported` + remove per-op scratch alloc + entry/exit `synchronize()`** (mirror `gqa_decode.rs` capture-safe pattern: device seqlens ptr, fixed module-global scratch, no sync):
- `standard_attention.rs`: entry sync ~`836`, exit sync ~`1250`; `capture_support()` Unsupported ~`1267`; replace per-op `alloc_raw`/`free_raw` with fixed module-global scratch.
- `native_decode.rs`: capture gate ~`1919` (`!bindings.any(has_dynamic_logical_input_shape())`) requires fixed-cap bindings — needs prefill-eager + post-prefill fixed-cap binding switch + deferred capture gate; host prologue rework so `total_seq`/`capacity_key` still hold at fixed cap (`past_key.seq`=capacity).
- Then VERIFY capture engages: captures≥1, replays=N, fallbacks=0 both exports; headline eager ~24-26 → captured ~57-61 tok/s (Pris: 40.5ms/tok, only 15.5ms real GPU compute; ~60% orchestration removed by capture).

**Files:** `crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs` (ABI); reference `crates/onnx-runtime-ep-cuda/src/kernels/gqa_decode.rs` (capture-safe pattern); `crates/onnx-genai-engine/src/native_decode.rs` (bindings/gate for c2/d).
