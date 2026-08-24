# `com.microsoft::PagedAttention` Applicability Audit (Revision 2)

> **Status:** Phase-1 audit **complete**, corrected from authoritative source.
> **Verdict: GO for a bounded first slice** — `com.microsoft::PagedAttention`
> **is** semantically applicable to **dense MLA** (GLM‑5.2 `--glm-full-attention`
> fallback and DeepSeek‑V2/V3) via its `kv_cache_layout="LATENT"` absorbed‑MLA
> mode. The query‑dependent sparse families (GLM DSA/IndexShare, DeepSeek‑V4
> CSA/HCA) remain **not expressible** — for a first‑principles reason, not a
> model‑name reason. One‑authority integration on `onnx-genai-kv` is **possible**
> (additive), not blocked.
>
> **Auditor:** Leon (Engine Dev — KV & Buffers). **Date:** 2026‑08‑24.
> **Supersedes:** the Revision‑1 audit (Sapper), which was **rejected** by
> reviewer Gaff. This revision was produced independently from ORT v1.29.0
> source per the reviewer‑protocol lockout; the prior author did not contribute.
> **Scope authorized:** `onnx-genai` (native runtime) and `mobius` (exporter).
> **Requested by:** Justin Chu.

## 0. Why Revision 1 was wrong (the correction, up front)

Revision 1 extracted a **stale/incomplete schema** and built its STOP verdict on
invariants that **do not exist** in ORT v1.29.0. Cross‑checked against the pinned
source (`ort-sys` links ORT **1.29.0**, `crates/onnx-genai-ort/ort-sys/build.rs`
`ORT_VERSION = "1.29.0"`; the tree's `VERSION_NUMBER` is `1.29.0`):

| Revision‑1 claim | ORT v1.29.0 reality | Source |
|---|---|---|
| 7 attributes | **15 attributes** | `bert_defs.cc:1545‑1620` |
| 8–10 inputs | **17 inputs** | `bert_defs.cc:1622‑1729` |
| `block_size % 256 == 0` | **power of two and ≥ 16** | `paged_attention_helper.h:157‑165` (`CheckKVCache`) |
| symmetric K/V head dims required | **`v_head_size` may be narrower than `head_size`** (LATENT) | `bert_defs.cc:1607`; helper `:453‑466` |
| no latent/compressed cache | **`kv_cache_layout="LATENT"` absorbed MLA** | `bert_defs.cc:1599‑1606`; helper `:477‑509` |
| head‑level `do_rotary` only | **`rotary_offset` partial‑RoPE suffix** | `bert_defs.cc:1615`; helper `:551‑560` |
| no learned per‑head sink | **`head_sink` input** | `bert_defs.cc:1685‑1690`; helper `:276‑287` |
| no quantized cache | **int8 / fp8e4m3fn cache + int4/float4 dtype attrs + PER_TENSOR/PER_CHANNEL scales** | `bert_defs.cc:1572‑1598,1697‑1712`, `T_CACHE`; helper `:310‑385` |
| CUDA only | **CUDA + WebGPU** registrations; **no CPU kernel** | see §6 |

The consequence: dense **MLA** — asymmetric `v_head_dim < qk_head_dim`, partial
RoPE, and a compressed/latent cache — is precisely the case the op's **LATENT**
mode was designed for (the schema doc literally cites **DeepSeek‑V3 `head_size=576`,
`v_head_size=512`**, `bert_defs.cc:1609`). Revision 1 declared that case
inexpressible; it is the op's headline feature. Revision 1 also treated the
"second cache authority" concern as proof of **semantic** incompatibility; it is
an **integration** constraint, and it is solvable additively (§4).

## 1. Authoritative schema (ORT 1.29.0, `com.microsoft::PagedAttention` v1)

- **Domain / opset:** `com.microsoft`, operator‑set version **1**
  (`ONNX_MS_OPERATOR_SET_SCHEMA(PagedAttention, 1, ...)`, `bert_defs.cc:1542`).
- **Providers:** **CUDA** and **WebGPU** execution providers. **No CPU kernel**
  exists in ORT (see §6).
- **Type constraints** (`bert_defs.cc:1747‑1751`):
  - `T ∈ {tensor(float16), tensor(bfloat16)}` — query/key/value, rotary caches,
    `head_sink`, q/k norm, output.
  - `T_CACHE ∈ {tensor(float16), tensor(bfloat16), tensor(int8), tensor(float8e4m3fn)}`
    — the KV cache (quantized cache is first‑class).
  - `T_KV_SCALE = tensor(float)` — the dequant scales.
  - `S = tensor(int32)` — all positional/index tensors.

### 1.1 Attributes (15) — `bert_defs.cc:1545‑1620`

| # | Attr | Type | Default | Meaning |
|---|---|---|---|---|
| 1 | `num_heads` | INT | — (req) | Q heads |
| 2 | `kv_num_heads` | INT | — (req) | K/V heads (GQA); **must be 1 in LATENT** |
| 3 | `scale` | FLOAT | `1/sqrt(head_size)` | QK scale; **required** when `v_head_size ≠ head_size` |
| 4 | `softcap` | FLOAT | `0` | logit softcap |
| 5 | `local_window_size` | INT | `-1` | left window (Mistral‑style) |
| 6 | `do_rotary` | INT | `0` | apply RoPE in‑op |
| 7 | `rotary_interleaved` | INT | `0` | interleaved RoPE |
| 8 | `qk_norm_epsilon` | FLOAT | `1e-6` | ε for Q/K RMSNorm when `q_norm_weight`/`k_norm_weight` present |
| 9 | `k_quant_type` | STRING | `"NONE"` | `NONE`/`PER_TENSOR`/`PER_CHANNEL` key‑cache quant granularity |
| 10 | `v_quant_type` | STRING | `"NONE"` | value‑cache quant granularity |
| 11 | `k_cache_dtype` | STRING | `""` | logical key‑cache elem type: `""`,`float16`,`bfloat16`,`int8`,`float8e4m3fn`,`int4`,`float4e2m1` (sub‑byte packed 2/byte into uint8, last dim `(head_size+1)/2` bytes) |
| 12 | `v_cache_dtype` | STRING | `""` | logical value‑cache elem type (same values as `k_cache_dtype`) |
| 13 | `kv_cache_layout` | STRING | `"SEPARATE"` | `SEPARATE` (distinct K/V caches) or `LATENT` (single absorbed‑MLA cache) |
| 14 | `v_head_size` | INT | `0` (= `head_size`) | value‑head width; may be `< head_size` **only** in LATENT |
| 15 | `rotary_offset` | INT | `0` | first channel covered by RoPE: RoPE over `[rotary_offset, rotary_offset+rotary_dim)`, multiple of 8; MLA sets it to `kv_lora_rank` |

> The task brief said "16 attrs"; the pinned v1.29.0 source registers **15**
> `.Attr(...)` calls (`bert_defs.cc:1545‑1620`). Source is authoritative.

### 1.2 Inputs (17) — `bert_defs.cc:1622‑1729`

| # | Name | Type | Opt | Shape / role |
|---|---|---|---|---|
| 0 | `query` | T | | `(num_tokens, hidden)` or packed `(num_tokens, (num_heads+2·kv_num_heads)·head_size)` |
| 1 | `key` | T | opt | `(num_tokens, kv_hidden)`; the **latent row** in LATENT; absent ⇒ packed‑QKV |
| 2 | `value` | T | opt | `(num_tokens, kv_hidden)`; **absent in LATENT** |
| 3 | `key_cache` | T_CACHE | | `(num_blocks, block_size, kv_num_heads, head_size)`; **updated in place**; sole cache in LATENT |
| 4 | `value_cache` | T_CACHE | opt | same shape as `key_cache`; **absent in LATENT** |
| 5 | `cumulative_sequence_length` | S | | `(batch+1)` new‑token boundaries |
| 6 | `past_seqlens` | S | | `(batch)` cached length per sequence |
| 7 | `block_table` | S | | `(batch, max_blocks_per_seq)` seq→block map (read path; always required) |
| 8 | `cos_cache` | T | opt | `(max_seqlen, head_size/2)` |
| 9 | `sin_cache` | T | opt | `(max_seqlen, head_size/2)` |
| 10 | `slot_mapping` | S | opt | `(num_tokens)` flat write slot `block*block_size+off`; **`-1` skips the write** (prefix hits, rejected speculative tokens) |
| 11 | `head_sink` | T | opt | `(num_heads)` learnable per‑head sink logit in the softmax denominator |
| 12 | `q_norm_weight` | T | opt | `(head_size)` RMSNorm gain on Q before RoPE (with `k_norm_weight`) |
| 13 | `k_norm_weight` | T | opt | `(head_size)` RMSNorm gain on K before RoPE and before cache write |
| 14 | `k_scale` | T_KV_SCALE | opt | `(1)` PER_TENSOR or `(kv_num_heads,1,head_size)` PER_CHANNEL; symmetric (no zero point) |
| 15 | `v_scale` | T_KV_SCALE | opt | as `k_scale` for the value cache |
| 16 | `attention_metadata` | S | opt | `(2)` CPU `[max_query_len_bound, max_kv_len_bound]`; **replay‑wide upper bounds** — removes device readback and makes the node **CUDA‑Graph‑capturable** |

Outputs (3) — `bert_defs.cc:1730‑1746`: `output (num_tokens, num_heads·v_head_size)`;
optional `key_cache_out`/`value_cache_out` that **must alias** inputs 3/4
(`value_cache_out` absent in LATENT).

### 1.3 Invariants & typed rejections (from `paged_attention_helper.h::CheckInputs`)

The op **rejects with `INVALID_ARGUMENT`** unless all hold (helper line refs):

1. `head_size % 8 == 0` (`:26‑30`).
2. **SEPARATE:** `kv_hidden/kv_num_heads == head_size` and V head/hidden == K's —
   symmetric K/V (`:44‑63`). Asymmetric V is impossible here.
3. **`block_size` is a power of two and `≥ 16`** (`:157‑165`) — **not** `%256`.
4. `cumulative_sequence_length (batch+1)`, `past_seqlens (batch)`, `block_table`
   rank‑2 dim0==batch, `slot_mapping (num_tokens)` (`:224‑274`).
5. `key_cache_out`/`value_cache_out`, if present, **alias** the inputs — the op
   mutates the KV cache in place and **allocates no KV cache of its own**
   (`bert_defs.cc:1735‑1746` output docstrings). (The CUDA EP allocates transient
   compute scratch — densified Q/K, gathered K/V, softmax‑LSE, decode partials —
   but that is EP‑managed workspace, never a KV‑cache authority.)
6. **LATENT** (`:477‑509`): `value` and `value_cache` **absent**; `kv_num_heads == 1`;
   V is the leading `v_head_size` channels of `key_cache`; `head_sink`, q/k‑norm,
   and `v_scale`/`v_quant_type`/`v_cache_dtype` **not supported** (typed rejects).
7. `v_head_size ∈ {0} ∪ [1, head_size]`, may differ from `head_size` **only** in
   LATENT, and then an explicit `scale` is **required** (the `1/sqrt(head_size)`
   default is the wrong scale for absorbed MLA — the "softmax‑scale trap",
   `:453‑474`).
8. `rotary_offset ≥ 0`, multiple of 8, and `rotary_offset + rotary_dim ≤ head_size`
   (`:551‑560`); cos/sin both‑present‑or‑both‑absent (`:540‑549`).
9. Quant: a non‑`NONE` `k/v_quant_type` **iff** the cache elem type is quantized,
   and then the matching scale is **required**; PER_TENSOR scale shape `(1)`,
   PER_CHANNEL `(kv_num_heads,1,head_size)` (`:310‑385`).
10. `head_sink (num_heads)`; `q_norm_weight`/`k_norm_weight` `(head_size)` and
    provided **together**, with `qk_norm_epsilon > 0` (`:276‑308,533‑537`).
11. `attention_metadata` shape `(2)`; entries are **trusted upper bounds**, never
    exact — over‑estimating only costs empty work; under‑sizing violates the
    contract (`:565‑583`).

Backend is **dense** (Flash / MemoryEfficient / XQA), selected over the whole
`block_table`. There is **no query‑dependent sparse‑index input**, no learned
top‑k selection, and no temporal compression. This single fact is the dividing
line in §5.

## 2. First‑principles invariant (carry the reason — `design-discipline`)

> `com.microsoft::PagedAttention` v1 expresses a layer **iff** every query
> attends **densely** to the **whole** cached KV prefix (optionally a left window
> / softcap / per‑head sink), and the per‑token KV can be written into a
> block cache as **either**
> (a) full per‑head K and V with `head_size(K)=head_size(V)` — `SEPARATE`; **or**
> (b) a **single shared latent row** `[c^{KV}; k^R]` of width `head_size`, with V
>     taken as its leading `v_head_size` channels and RoPE confined to the
>     `[rotary_offset, head_size)` suffix — `LATENT` (absorbed MLA).
> A layer is **outside** the op — by construction, not by name — iff it selects a
> **query‑dependent subset** of the cache (top‑k / learned index), because the
> schema has no per‑query index input and the kernel reads the entire prefix.

Everything below falls out of this one sentence. Note what is **not** in it:
"MLA", "compressed cache", and "asymmetric V" are all inside case (b); only
*query‑dependent selection* is outside.

## 3. Mapping dense MLA onto `LATENT` (the mathematics Revision 1 missed)

Absorbed MLA is exact linear algebra, and it is what `LATENT` computes:

- Cache one latent row per token `L_s = [c_s^{KV}; k_s^{R}]`, width
  `head_size = kv_lora_rank + qk_rope_head_dim`, shared by all heads
  (`kv_num_heads = 1`).
- Fold `W_UK^i` into the query: the nope score
  `q^{C,i}·(W_UK^i c^{KV}) = (W_UK^{iT} q^{C,i})·c^{KV}`, so the per‑head absorbed
  query is `Q^i = [W_UK^{iT} q^{C,i}; q^{R,i}]`, width `head_size`. RoPE touches
  only the `q^{R,i}` suffix ⇒ `rotary_offset = kv_lora_rank`.
- `v_head_size = kv_lora_rank`: attention over `L` produces the context
  `ctx^i = Σ_s p_{s} c_s^{KV}` (the leading `v_head_size` channels), and `W_UV^i`
  is folded into the output projection `W_O` **after** the op.
- `scale = 1/sqrt(qk_head_dim)` (the pre‑absorption width) is supplied explicitly,
  which the schema **requires** whenever `v_head_size ≠ head_size`.

This is the DeepSeek‑V3 configuration named in the schema (`576/512`). GLM‑5.2's
`--glm-full-attention` fallback is dense MLA (`DeepSeekV3TextModel`) with
decomposed head dims `qk_head_dim = 192` (`qk_nope 128 + qk_rope 64`),
`v_head_dim = 128`; after absorption it presents to the op as a single latent row
with `rotary_offset = kv_lora_rank`, `v_head_size = kv_lora_rank`. The op computes
the identical attention probabilities and, after the folded `W_O`, the identical
output. §7's slice‑2 oracle proves this numerically (absorbed‑`LATENT` == decomposed
dense MLA), which is the direct refutation of Revision 1.

## 4. Exact one‑authority invariant (`onnx-genai-kv` stays the sole manager)

**Facts about the two ownership models:**

- `onnx-genai-kv` **owns** page allocation, page tables, sequence→page maps,
  lengths and lifetimes (`crates/onnx-genai-kv/src/page_table.rs`,
  `paged_cache.rs`; `lib.rs:1‑8`). Its physical page slab per `(layer, K|V)`
  component is **head‑major**: `offset = head·(page_size·head_dim) + token·head_dim + dim`
  (`page_table.rs::value_at_slot`), K and V in **separate** component slabs, with a
  configurable **token `page_size`** not constrained to any quantum.
- The op's cache is **token‑major within a block**:
  `(num_blocks, block_size, kv_num_heads, head_size)` ⇒ `[block_size, head, dim]`,
  `block_size` a power of two `≥ 16`; and the op **allocates no KV cache** — it
  only reads/writes KV buffers the caller owns (`key_cache_out` must alias
  `key_cache`; any compute scratch is transient EP workspace, not a cache).

**The op is one‑authority‑compatible.** Because the op is a pure
consumer/mutator with **no KV‑cache allocator of its own**, the single authority
remains `onnx-genai-kv`. What integration requires is **additive** on that
existing authority, never a second manager:

> **One‑authority invariant.** `onnx-genai-kv` remains the sole allocator/owner of
> KV pages, block tables, slot mapping, sequence lengths and lifetimes.
> `PagedAttention` is admitted **iff** `onnx-genai-kv` can present its owned pages
> in the op's exact physical view and emit the op's device‑side index tensors —
> concretely: (i) a **token‑major page layout** `[block_size, kv_num_heads, head_size]`
> (and a **`LATENT` single‑cache** variant of width `head_size`), (ii)
> `page_size == block_size` chosen a power of two `≥ 16`, and (iii) device
> `block_table`, `past_seqlens`, `cumulative_sequence_length`, and `slot_mapping`
> (with `-1` for suppressed writes) derived from the page table. The op then
> mutates those buffers in place. No `PagedAttention`‑owned allocator is created.

**Blocker status: none that forbids the design.** The head‑major↔token‑major
layout difference and the absence of a `block_table`/`slot_mapping` emitter today
are *missing additive capabilities on the existing authority*, not an ownership
conflict. The precise, bounded work is: add a token‑major (and LATENT) page‑store
layout option and a block‑table/slot‑mapping view to `onnx-genai-kv`. If a future
kernel demanded that the op **allocate or resize** the KV cache itself, *that*
would be a hard blocker (it would create a second authority) — but v1 explicitly
does not: it aliases the caller's KV buffers and never allocates a cache.

## 5. Applicability matrix (property, not model name)

| Target path | Fits v1? | Reason (first‑principles) |
|---|---|---|
| **GLM‑5.2 `--glm-full-attention` (dense MLA)** | **YES — `LATENT`** | Dense attention over the full prefix; absorbed MLA maps to a single latent row with `v_head_size = kv_lora_rank < head_size`, `rotary_offset = kv_lora_rank`, explicit `scale`, `kv_num_heads=1` (§3). This is the op's headline mode. |
| **DeepSeek‑V2 / V3 (dense MLA)** | **YES — `LATENT`** | Same mapping; the schema names DSV3 `head_size=576, v_head_size=512`. |
| **Genuine dense GQA/MHA** (any) | **YES — `SEPARATE`** | Full per‑head symmetric K/V is the default mode. |
| **GLM‑5.2 DSA / IndexShare** (default) | **NO** | Per‑query learned **shared‑index top‑k** selection; the schema has **no** per‑query sparse‑index input — the kernel attends to the entire `block_table`. Served by `pkg.nxrt::IndexShare` v1. |
| **DeepSeek‑V4 CSA / HCA** | **NO** | Temporal **compression** (ratios 4/128; 128 = HCA) **plus** learned FP4 index **top‑k** selection; query‑dependent selection is not expressible. `head_sink` covers the learned sink term but not the selection. Served by `pkg.nxrt::CompressedSparseAttention` + `SparseKvGather` v1. |

The rejections are decided by the **query‑dependent‑selection** property in §2, so
they generalize to any future model with the same shape and do not depend on a
name allowlist. (The `pkg.nxrt::*` custom ops referenced above exist in‑tree:
`crates/onnx-runtime-ep-cpu/src/kernels/{index_share,compressed_sparse_attention,sparse_kv_gather}.rs`,
registered under domain `pkg.nxrt` in `provider.rs`.)

## 6. Providers: who executes it

- **CUDA EP** — 6 typed kernels: `T ∈ {fp16,bf16}` × `T_CACHE ∈ {fp16/bf16,int8,float8e4m3fn}`
  (`contrib_ops/cuda/bert/paged_attention.cc:28‑49`; registered in
  `cuda_contrib_kernels.cc:124‑130,407‑413`). Full LATENT/quant/sink/qk‑norm/
  rotary‑offset support.
- **WebGPU EP** — one kernel, `T = float16`, `S = int32`
  (`contrib_ops/webgpu/bert/paged_attention.cc:34‑37`; registered in
  `webgpu_contrib_kernels.cc:40`). It implements **`SEPARATE` only** and **rejects
  every optional mode with a typed `NOT_IMPLEMENTED`** — `softcap`,
  `local_window_size`, `kv_cache_layout!=SEPARATE`, `v_head_size!=head_size`,
  `rotary_offset!=0`, `head_sink`, quant, `slot_mapping`, q/k‑norm, `k/v_scale`,
  `attention_metadata` (`paged_attention.cc:500‑560`). **This is the exact
  precedent our native backend follows** (§7, step 5): implement one subset,
  reject the rest with typed reasons.
- **No CPU kernel** — `contrib_ops/cpu/bert/paged_attention.cc` does not exist and
  `cpu_contrib_kernels.cc` registers no `PagedAttention`. A native CPU reference
  is therefore permitted **for tests only**; it must not be claimed as upstream
  CPU support.
- **Native `onnx-runtime-ep-cuda`** currently claims `GroupQueryAttention`, not
  `PagedAttention`. The engine's decode classifier already treats `PagedAttention`
  as explicit‑valid‑length attention (`decode/metadata.rs::graph_uses_explicit_kv_length_attention`)
  — classification only; it does not execute the op.

## 7. Decision & staged plan (no dead code; each slice independently reviewed)

**GO** for a bounded first implementation, landed in order; **do not merge any
slice without explicit reviewer approval** (reviewer ≠ Sapper, ≠ Leon).

- **Slice 1 — this corrected audit + the exact one‑authority invariant.** (This
  document + the Leon decision record.)
- **Slice 2 — typed schema validator + CPU reference oracle** (crate
  `onnx-genai-paged-attention`). A faithful, typed port of the v1.29.0 helper
  invariants that **rejects every unsupported optional mode with a typed reason**
  (WebGPU‑style), plus a CPU oracle for `LATENT` dense MLA and `SEPARATE` dense
  GQA over the block cache / `block_table` / `slot_mapping`. Tests: schema
  accept/reject matrix; **paged == contiguous** equivalence; **absorbed‑`LATENT`
  == decomposed dense MLA** for GLM dims (`qk_head_dim=192, v_head_dim=128`,
  partial RoPE); block boundaries (pow2 ≥ 16); `slot_mapping = -1` skip; invalid
  tables/lengths; `head_sink` / q/k‑norm; explicit rejection of unimplemented
  modes. **This slice is implemented and green in this revision.**
- **Slice 3 — CUDA `LATENT` subset** on `onnx-runtime-ep-cuda`, reusing
  `onnx-genai-kv` page/block/slot authorities per §4 (token‑major + LATENT page
  layout, block‑table/slot‑mapping emitter). Capture/replay‑safe: stable buffers,
  no host sync or allocation inside capture; `attention_metadata` supplies the
  host bounds so the node is capturable. Schema validation rejects every
  unsupported optional mode with a typed reason. CUDA‑native vs CPU‑oracle parity
  for GLM dims across first‑token / prefill / decode.
- **Slice 4 — Mobius opt‑in `--paged-attention` export** for GLM‑5.2 dense‑MLA
  only; default stays DSA/decomposed. Emit the exact v1 attrs/inputs; **gate on
  MLA layout properties** (latent width, `v_head_size`, rotary suffix, cache
  dtype, page/block constraints), **never on model name**; preserve a decomposed
  oracle path.
- **ORT‑backend path** — verify the emitted model executes under ORT CUDA v1.29.0
  with no custom‑schema collision; **do not** re‑register `com.microsoft` under
  another domain.

**Remaining full‑size gate (unchanged):** no full‑size performance claim before
official GLM‑5.2 / DeepSeek checkpoint runs on an idle A100 (CUDA events, capture/
replay/fallbacks, page/VRAM accounting, n ≥ 3; no wall‑clock‑only claim). Tiny
correctness is a gate, not a performance result. Coordinate with the ongoing GLM
GGUF/safetensors work — no conflict.

## 8. What changed vs Revision 1 (auditable summary)

1. Schema corrected to the real 15 attrs / 17 inputs / 3 outputs, incl.
   `LATENT`, `v_head_size`, `rotary_offset`, `head_sink`, quant cache, and the
   **pow2 ≥ 16** block rule (not `%256`).
2. Verdict flipped for dense MLA: **applicable via `LATENT`** (GLM‑5.2 full‑attn,
   DeepSeek‑V2/V3), with the absorption math (§3) and a numerical proof (§7 slice 2).
3. Sparse families (DSA/IndexShare, CSA/HCA) remain **not expressible**, but for
   the precise reason (no query‑dependent sparse‑index input), not by name.
4. "Second cache authority" reframed as an **additive integration** task on
   `onnx-genai-kv`, with an exact one‑authority invariant and the true hard‑blocker
   condition stated (op‑owned allocation/resize — which v1 does not do).
5. Providers corrected: **CUDA + WebGPU**, no CPU kernel; WebGPU is the typed‑
   rejection precedent for the native subset.
