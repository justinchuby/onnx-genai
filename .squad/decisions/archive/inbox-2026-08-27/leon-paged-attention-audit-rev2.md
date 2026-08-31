# Decision: Microsoft PagedAttention — corrected audit (rev2) + typed validator/oracle

- **Author:** Leon (Engine Dev, KV & Buffers)
- **Date:** 2026-08-24
- **Supersedes:** Sapper's rejected audit (`squad/paged-attention-audit` @ `0a4726200`,
  REJECTED by Gaff). Rewritten independently from authoritative ORT v1.29.0 source
  under the reviewer-protocol lockout (no Sapper input).
- **Branch:** `squad/paged-attention-audit-rev2` (based on `main`).
- **Reviewer:** independent code review — **APPROVE** (both findings fixed; 16/16 tests, clippy clean).
- **Status:** NOT merged (task requires explicit approval).

## Core correction

The rejected audit STOPed on a **stale/wrong schema**. The authoritative ORT
**1.29.0** `com.microsoft::PagedAttention` v1 schema (`bert_defs.cc:1385-1754`,
`paged_attention_helper.h`) is **15 attributes / 17 inputs / 3 outputs** and
supports **`kv_cache_layout="LATENT"` absorbed MLA**. That flips the verdict:
dense MLA (GLM-5.2 `--glm-full-attention`, DeepSeek V2/V3) **is expressible** as
PagedAttention. (The brief's "16 attrs" is off by one; source is authoritative.)

Key schema facts (all cited in the audit and ported in `validate.rs`):
- `block_size` power-of-two **≥ 16** (NOT `%256`).
- `v_head_size < head_size` allowed **only** in LATENT (DSV3 576/512); explicit
  `scale` **required** when `v_head_size != head_size` (the "scale trap").
- `rotary_offset` for a partial-RoPE suffix; `head_sink`; `slot_mapping` (`-1`
  suppresses a write); quantized cache storage `int8`/`float8e4m3fn` (`int4`/
  `float4` reserved but rejected by shipped kernels).
- Providers: **CUDA** (6 typed kernels) + **WebGPU** (fp16, SEPARATE-only, typed
  `NOT_IMPLEMENTED` for every other mode). **No CPU kernel.**

## One-authority invariant (the constraint that matters to KV & Buffers)

> `onnx-genai-kv` remains the **sole** allocator/owner of KV pages, block tables,
> slot mapping, sequence lengths and lifetimes. PagedAttention v1 **allocates no
> KV cache** — it aliases the caller's buffers and mutates them in place
> (`key_cache_out` must alias `key_cache`). Integration is therefore **additive**
> on `onnx-genai-kv`: add a **token-major** page layout
> `[block_size, kv_num_heads, head_size]` (and a **LATENT single-cache** variant
> of width `head_size`), set `page_size == block_size` (pow2 ≥ 16), and emit
> device `block_table` / `past_seqlens` / `cumulative_sequence_length` /
> `slot_mapping` from the page table. **No second KV manager is created.**

A hard blocker would exist only if the op required its **own** cache
allocation/resize — v1 does not. (CUDA-EP transient compute scratch is
EP-managed workspace, not a KV-cache authority.)

## Applicability (first-principles, not model-name allowlist)

- **Expressible now:** GLM-5.2 dense MLA (`--glm-full-attention`), DeepSeek V2/V3
  MLA — via LATENT + narrower `v_head_size` + partial-RoPE `rotary_offset`.
- **NOT expressible:** GLM DSA / IndexShare and DeepSeek-V4 CSA/HCA
  sparse-selection — **no query-dependent sparse-index input** exists in the v1
  schema; `block_table` is per-sequence and the kernel reads the whole prefix.
  (Custom `pkg.nxrt::{IndexShare,CompressedSparseAttention,SparseKvGather}` ops
  do exist in this tree, but they are a different, non-PagedAttention path.)

## Delivered (this branch)

1. **Corrected audit** — `docs/models/PAGED_ATTENTION_AUDIT.md` (commit `f4c26c53b`).
2. **`onnx-genai-paged-attention`** crate (commits `3107beadc`, `3bd5f4b53`),
   test/reference-only (upstream has no CPU kernel — this is **not** upstream CPU support):
   - `validate.rs` — faithful `CheckInputs` port → typed `INVALID_ARGUMENT`.
   - `backend.rs` — capability gate → typed `NOT_IMPLEMENTED` (mirrors the WebGPU
     EP precedent); presets `webgpu_separate()`, `glm_dense_mla_latent()`.
   - `oracle.rs` — f32 dense paged attention (SEPARATE GQA + LATENT absorbed MLA).
   - `tests/equivalence.rs` — 16 tests incl. **absorbed-LATENT == decomposed-MLA**
     knockout at tiny **and** DeepSeek-V3 576/512/64 dims.

## Proven

- Absorbed MLA folds `W_UK` into the query (`W_UK^T q_nope`) and `W_UV`/`W_O`
  after the op; `scale = 1/sqrt(qk_nope+qk_rope)`; latent row `[c_kv ; k_rope]`,
  V = leading `v_head_size` channels. The knockout compares two structurally
  different code paths and matches — the applicability claim is executable, not
  asserted.

## Remaining gates (deferred, each an independently reviewed follow-on slice)

- **Slice 3 — CUDA LATENT kernel:** capture/replay-safe, stable buffers, no host
  sync/alloc in capture; validated against this CPU oracle at GLM dims
  (qk=192, v=128, partial RoPE), prefill + decode.
- **Slice 4 — Mobius `--paged-attention` export:** opt-in, GLM-5.2 dense MLA only,
  default stays DSA/decomposed; gate on **MLA layout properties** (latent width,
  `v_head_size`, rotary suffix, cache dtype, pow2≥16 block), not a model-name
  allowlist; keep a decomposed oracle path.
- **onnx-genai-kv additive change:** token-major (+LATENT) page layout +
  block_table/slot_mapping view. **No second KV manager.**
- **Full-size + measurement:** coordinate with GLM GGUF/safetensors work; **no
  full-size perf claim** before official checkpoint runs on idle A100 (CUDA
  events, capture/replay, page/VRAM accounting, n≥3). Tiny correctness is only a gate.
- **Merge:** blocked pending explicit approval.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
