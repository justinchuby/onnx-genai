# DsaIndexSelect — GLM-5.2 DSA query-dependent index selection — frozen v1

**Status:** frozen v1 (CPU oracle + native/plugin CUDA implementation)
**Operator:** `pkg.nxrt::DsaIndexSelect`, version 1
**Pairs with:** `pkg.nxrt::IndexShare` (see `INDEXSHARE_DESIGN.md`)

## Why this op exists

GLM-5.2 / DeepSeek Sparse Attention (DSA) is *query-dependent* sparse attention:
each query position scores every candidate key with a small learned "indexer"
and attends only to its own top-`k` keys. That query-dependent selection is why
DSA **cannot** be expressed by `com.microsoft::PagedAttention` — PagedAttention
has no query-dependent sparse-index input (see `docs/models/PAGED_ATTENTION_AUDIT.md`).

DSA decomposes cleanly into two halves along one data boundary — the
`selected_indices` tensor:

1. **index selection** — score + causal-mask + top-`k` + sort → `selected_indices`.
   This op, `DsaIndexSelect`.
2. **sparse attention** — gather the selected KV and attend → `output`.
   The existing `pkg.nxrt::IndexShare`.

`onnx-genai-kv` remains the sole page/cache authority: `DsaIndexSelect` allocates
no pages, owns no cache, and invents no PagedAttention inputs. It is a pure
function producing indices that `IndexShare` (which owns the KV I/O) consumes.

### The capture-safety value-add

The Mobius decomposition ends the indexer with `k = Min(key_length, index_topk)`
— a **data-dependent** TopK width read from a device tensor every decode step. A
CUDA graph cannot capture a kernel whose output width is a run-time value, so the
decomposed indexer is capture-hostile. `DsaIndexSelect` implements scoring +
mask + top-k + ascending sort as one operator with a **stable output width**
(always `top_k`), right-padding short rows with the `-1` sentinel `IndexShare`
already understands. The CUDA implementation records a fixed two-kernel
pipeline into the graph. Stable width and stable kernel-owned scratch make
capture/replay safe.

## Boundary and schema

`DsaIndexSelect` computes only the index selection. Projections (`wq_b`, `wk`,
`weights_proj`), norms and RoPE stay as separate generic native ops upstream; the
scoring MatMul is fused *inside* this op so the `(B, S, H, T)` per-head scores
are never materialized. CUDA stores only the final masked scalar score and an
allowed byte per `(B,S,T)` cell in private persistent scratch between its two
kernels; neither tensor crosses the operator boundary.

### Inputs

Positional, all required:

| # | Name | Type and shape | Notes |
|---|---|---|---|
| 0 | `query` | f16/bf16/f32 `[B, S, H, D]` | indexer query, RoPE-split. `H`=`index_n_heads`, `D`=`index_head_dim` |
| 1 | `key` | same dtype as query `[B, T, D]` | indexer key cache (`all_index_keys`), grown or fixed-capacity |
| 2 | `weights` | same dtype as query `[B, S, H]` | per-head indexer weights (`weights_proj` output) |
| 3 | `attention_bias` | f32 `[B, 1, S, T]` | additive causal mask; the source of truth for which key positions are allowed |

`query`, `key`, `weights` share one float element type. `attention_bias` is
always f32 (matching `IndexShare`'s widened bias) with a head-broadcast axis of 1.

### Output

| # | Name | Type and shape | Notes |
|---|---|---|---|
| 0 | `selected_indices` | int64 `[B, 1, S, top_k]` | consumed directly by `IndexShare` |

Index-head dimension is `1`: the head-weighted reduction collapses `H`, so all
query heads share one selection (matches GLM's shared-index reuse). Output is
int64 (the wide, unambiguous choice; `IndexShare` accepts int32 or int64).

### Attributes (frozen v1 ABI)

- `top_k` (required int, `> 0`) — output selection width `K` (= `index_topk`).
- `scale` (required float, finite `> 0`) — `softmax_scale` = `index_head_dim ** -0.5`, applied to the dot product before ReLU.
- `weights_scale` (optional float, default `1.0`, finite `> 0`) — the `n_heads ** -0.5` factor applied to `weights`.

Any other attribute is rejected as "not part of the frozen v1 ABI".

## Frozen semantics

For every batch `b` and query row `s`, over candidate key positions `t ∈ [0, T)`:

```
dot(h, t)   = Σ_d query[b,s,h,d] · key[b,t,d]
score(h, t) = relu(scale · dot(h, t))
weighted(t) = Σ_h score(h, t) · (weights[b,s,h] · weights_scale)
masked(t)   = weighted(t) + attention_bias[b,0,s,t]
ordered(t)  = canonicalize_nan(masked(t))
```

A position `t` is **allowed** iff `attention_bias[b,0,s,t] > -1e30` (both `-inf`
and `finfo.min ≈ -3.4e38` causal fills, and `NaN`, count as *not allowed*).
Select the top `min(#allowed, top_k)` allowed positions by
`(ordered(t) descending, t ascending)`, sort them **ascending by `t`**, and
right-pad with `-1` to width `top_k`. A row with zero allowed positions yields an
all-`-1` row (the caller's causal construction guarantees at least the self
position; `IndexShare` rejects all-`-1` rows if that guarantee is violated).

`canonicalize_nan` runs after the final additive score and before ranking. Every
NaN becomes the positive quiet-NaN bit pattern `0x7fc00000`. This is the
canonical v1 contract, not an implementation detail: finite overflow such as
`infinity * 0` may otherwise produce backend-dependent signs/payloads. CPU and
CUDA then apply the same Rust `f32::total_cmp` order, descending by score and
ascending by index. The overflow regression produces NaNs non-vacuously and
selects `[0, 2]` for f32/f16/bf16 native execution and f32/f16 through the real
CUDA plugin.

### Equivalence to the decomposed path

The decomposed indexer selects a **uniform** `k = min(T, index_topk)` per step and
relies on `IndexShare` re-applying the causal bias to discard any masked-future
slots it selected for early prefill rows. `DsaIndexSelect` instead pads with `-1`.
Both produce the **same valid (finite-score) selections** — a masked-future slot
(decomposed) and a `-1` slot (fused) each contribute exactly zero under
`IndexShare`'s causal re-masking, so the attention output is identical. Finite
scores always outrank `-inf`, so the top-`k` finite sets coincide.

### Tie-breaking

Among equal canonical total-order keys, the **lower position index** wins. This
mirrors ONNX Runtime `TopK` (`total_cmp` order, ties by index) and is a hard part
of the op contract: CPU and CUDA must agree bit-for-bit on ties and NaNs.

## Claim-time contract

The typed validator (`unsupported_reason`, shared with the CUDA EP so both
backends' `supports_op` stay in lockstep) rejects, with typed reasons and never a
silent miscompute:

- unknown attributes, missing/`≤0` `top_k` or `scale`, non-finite/`≤0` `weights_scale`;
- wrong input arity (`≠ 4`) or output arity (`≠ 1`);
- unsupported/mismatched dtypes (non-float inputs, `key`/`weights` not matching
  `query`, `attention_bias` not exactly f32);
- wrong ranks (`query` 4, `key` 3, `weights` 3, `bias` 4) and static cross-input
  dimension conflicts (batch, seq, heads, head_dim, key seq, bias head-broadcast `= 1`).

`attention_bias` is pinned to f32 because the mask decision `bias > -1e30` depends
on the fill magnitude: an f16 `finfo.min` (−65504) would be misread as "allowed".

Output shape/dtype are not part of claim metadata (which is inputs-only, matching
the sibling `IndexShare` gate), so the `int64` + `[B,1,S,top_k]` output contract is
enforced at **execute** time rather than claim time.

Index *values* are produced by this op (never read from an input), so there is no
run-time index-validation step here — that lives in the consuming `IndexShare`.

## Typed refusals — what DSA index selection deliberately does NOT cover

- **GLM DSA / IndexShare is the *sparse* path**, but sparse selection that needs
  a *second* query-dependent granularity (block/tile indices, learned routing
  beyond a single top-k over a flat key axis) is out of scope: this op selects a
  single flat top-`k` per row. Coarser/hierarchical selection is a later version.
- **DeepSeek-V4 CSA/HCA** (compressed / hierarchical sparse attention) is a
  different operator (`CompressedSparseAttention`) and is explicitly avoided here.
- **PagedAttention** cannot host this selection at all (no query-dependent index
  input); do not route DSA through it.
- **Quantized index-key caches** are not in v1: `query`/`key`/`weights` are
  f16/bf16/f32 only. A quantized indexer cache is a typed-reject until implemented.

## Native runtime integration handoff (full-size GLM)

- **One-authority invariant.** `DsaIndexSelect` produces indices; `IndexShare`
  consumes them and owns KV I/O; `onnx-genai-kv` owns page/slot/lifetime. No
  second allocator, no op-side page allocation, no new PagedAttention inputs.
- **Buffers.** All inputs are caller-owned and stable across decode steps; the
  fused op writes a fixed-width `[B,1,S,top_k]` int64 output. This is the shape
  contract the CUDA kernel must honor for whole-step CUDA-graph capture.
- **Exporter reconciliation (separate Mobius follow-up).** The tiny fixture
  `tests/fixtures/tiny-glm52-qmoe-indexshare` currently emits the *decomposed*
  indexer (2× `TopK` + `Min` + `Relu`/`ReduceSum`). Replacing that subgraph with a
  single `DsaIndexSelect` node is a Mobius export change gated behind runtime
  approval and is **not** part of this runtime slice.

## CUDA implementation — parallel score + radix selection

The native/plugin CUDA path uses two NVRTC kernels, both captured as graph nodes.
Integer output parity with the CPU oracle is byte-exact.

### Kernel 1: parallel `(row, T)` scoring

`dsa_index_select_score` flattens all `B·S·T` score cells. It launches 256-thread
blocks with `ceil(B·S·T/256)` CTAs, capped by the device maximum grid X, and
grid-strides over any remaining cells. A realistic single decode row therefore
uses 8 score CTAs at T=2048 and 32 at T=8192 instead of leaving one CTA to do the
whole row.

One thread owns one `(b,s,t)` cell and loops over `H` then `D`. f16/bf16 storage
is widened with `__half2float` / `__bfloat162float`. `__fmul_rn` and
`__fadd_rn` preserve the CPU oracle's separate-rounding, ascending-D then
ascending-H accumulation order rather than allowing FMA contraction. The thread
writes the canonicalized final f32 score plus one u8 allowed flag. Disallowed
cells receive `-inf` and allowed=0.

### Kernel 2: 32-pass radix threshold and stable compaction

`dsa_index_select_select` launches one 256-thread block per row, capped and
grid-strided over `B·S`. It first fills the fixed-width output with `-1`. Each
thread owns a contiguous index interval, which is essential for deterministic
lower-index tie selection and ascending output compaction.

For each row the block:

1. counts allowed cells and sets `keep = min(allowed_count, top_k)`;
2. performs exactly **32** most-significant-bit-first radix count passes over
   the unsigned transform of Rust's signed `f32::total_cmp` key;
3. obtains the exact kth descending threshold key;
4. counts keys greater than and equal to the threshold;
5. assigns the required threshold ties to the lowest source indices; and
6. uses block scans over the contiguous thread chunks to compact every winner
   in stable ascending index order.

No thread performs a `top_k`-length sequence of full-T scans. Aggregate work is
`O(B·S·T·H·D)` for scoring plus `O(32·B·S·T)` for selection; parallel depth per
row is approximately `O(H·D + 32·T/256 + T/256)`. Output initialization adds
`O(B·S·top_k/256)`. The launch structure is always two kernels after warmup.

### Workspace, capture, and lifetime

The kernel owns persistent device scratch: one f32 score and one u8 allowed flag
per cell, with each segment and the total allocation aligned to 256 bytes. For
the measured decode shapes this is 10,240 B at T=2048 and 40,960 B at T=8192.
The slot grows only outside capture; growth drains prior users before replacing
the allocation. It is shared by native and ORT-plugin execution, stays
pointer-stable through warmup/capture/replay, and is released with the compiled
kernel/session (teardown evidence: 0 B live). The kernel declares no external
workspace, allocates no KV pages, and does not compete with `onnx-genai-kv` for
page/slot/lifetime authority.

There is no D2H, stream synchronization, allocation, free, or NVRTC compilation
on the captured path. `capture_support()` remains unsupported until one eager
execution has compiled both kernels and sized the fixed-shape persistent slot.
Capture then contains exactly two graph nodes and replay changes only device
buffer contents.

### Schema and capability

`pkg.nxrt::DsaIndexSelect` has one schema with `since_version = 1`; imports 1,
2, and later resolve to that sole v1 registration until a newer schema exists,
while import 0 rejects. Native and plugin capability validation delegates to the
shared CPU contract. CUDA projects only query/key/weights f16/bf16 metadata to
f32 for that structural validation and leaves `attention_bias` untouched, so
the strict f32 bias contract remains enforced.

### Consumer-boundary decision (why no extra glue op)

`DsaIndexSelect` fuses the **index-selection** half of DSA (the scoring MatMul +
top-k). Its `[B,1,S,top_k]` int64 output is consumed directly by the existing
native CUDA `pkg.nxrt::IndexShare` kernel, which already owns the sparse KV
gather + attention and the run-time index validation. Real GLM-5.2 DSA is
therefore fully covered by **`DsaIndexSelect` (CUDA) + `IndexShare` (CUDA)** with
no additional boundary op: the "smallest sparse gather/attention consumer" the
slice asked about already exists and is unchanged. This keeps the one-authority
invariant intact (indices are data flowing between two ops; neither allocates KV
pages).

### A100 measurement — real GLM indexer dimensions

Validated 2026-08-27 on physical GPU 7, A100-SXM4-80GB, UUID
`GPU-ef3c1a70-d297-933c-0c37-dbaad2136a57`, with an idle witness before every
process (0 MiB, 0%, 210 MHz and empty `nvidia-smi pmon`). Each counter-ordered
process ran an 8 s continuous captured-replay ramp, held 1410 MHz at 100%
utilization (~95.4–95.8 W for T=2048; ~103.4–103.9 W for T=8192), and timed
seven batches of four graph replays with CUDA events. Host-enqueue was measured
separately over drained small batches and is not the performance claim.

The fixture input type excludes `top_k`; K=4 and K=2048 share the exact same
query/key/weights/bias bytes. A host-side regression, run after the CUDA suite's
fail-loud device gate, asserts all four tensors byte-for-byte. The release probe
prints stable FNV-1a identity digests:

- T=2048: `9b0c18ee6979f3a7`
- T=8192: `92e37c6e547702b8`

| T | K | K=2048→4 process median (min–max), µs | K=4→2048 process median (min–max), µs | pooled n=14 median (min–max), µs |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 2048 | 853.76 (852.74–856.83) | 860.42 (858.88–862.21) | 857.86 (852.74–862.21) |
| 2048 | 4 | 857.09 (856.83–857.86) | 851.46 (850.43–853.50) | 855.17 (850.43–857.86) |
| 8192 | 2048 | 1089.79 (1088.51–1091.07) | 1096.70 (1096.45–1098.50) | 1093.76 (1088.51–1098.50) |
| 8192 | 4 | 1095.68 (1093.12–1099.01) | 1088.77 (1087.74–1091.58) | 1092.35 (1087.74–1099.01) |

The pooled K=2048 overhead versus byte-identical K=4 is **0.31% at T=2048**
and **0.13% at T=8192**, smaller than the 0.87–1.10% pooled min/max spreads.
This falsifies a hidden `top_k·T` scan; cost is parallel scoring plus the fixed
32-pass radix pipeline. Ordered first/last drift within individual configs was
0.00–0.40%.

For historical comparison, the exact serial head `e69e21ea9` measured
147,456.25 µs (147,446.28–147,476.23) at T=2048 and 593,648.62 µs
(593,475.12–593,678.56) at T=8192 under the same captured CUDA-event protocol
(PR #2076 evidence comment `5434343333`). Against the pooled parallel medians,
the current implementation is **171.89x** and **542.76x** faster respectively.
The T=8192 operator-only ceiling is about **914 calls/s**. These are per-operator
device-time results, not wall-clock or full-model tok/s claims.
