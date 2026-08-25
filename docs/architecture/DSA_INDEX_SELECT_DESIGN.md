# DsaIndexSelect — GLM-5.2 DSA query-dependent index selection — frozen v1

**Status:** frozen v1 (CPU oracle + typed validator landed; CUDA is the next slice)
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
decomposed indexer is capture-hostile. `DsaIndexSelect` fuses scoring + mask +
top-k + ascending sort into one kernel with a **stable output width** (always
`top_k`), right-padding short rows with the `-1` sentinel `IndexShare` already
understands. Stable width ⇒ CUDA-graph capture/replay safe.

## Boundary and schema

`DsaIndexSelect` computes only the index selection. Projections (`wq_b`, `wk`,
`weights_proj`), norms and RoPE stay as separate generic native ops upstream; the
scoring MatMul is fused *inside* this op so the `(B, S, H, T)` scores and
`(B, S, T)` weighted tensors are never materialized.

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
```

A position `t` is **allowed** iff `attention_bias[b,0,s,t] > -1e30` (both `-inf`
and `finfo.min ≈ -3.4e38` causal fills, and `NaN`, count as *not allowed*).
Select the top `min(#allowed, top_k)` allowed positions by
`(masked(t) descending, t ascending)`, sort them **ascending by `t`**, and
right-pad with `-1` to width `top_k`. A row with zero allowed positions yields an
all-`-1` row (the caller's causal construction guarantees at least the self
position; `IndexShare` rejects all-`-1` rows if that guarantee is violated).

### Equivalence to the decomposed path

The decomposed indexer selects a **uniform** `k = min(T, index_topk)` per step and
relies on `IndexShare` re-applying the causal bias to discard any masked-future
slots it selected for early prefill rows. `DsaIndexSelect` instead pads with `-1`.
Both produce the **same valid (finite-score) selections** — a masked-future slot
(decomposed) and a `-1` slot (fused) each contribute exactly zero under
`IndexShare`'s causal re-masking, so the attention output is identical. Finite
scores always outrank `-inf`, so the top-`k` finite sets coincide.

### Tie-breaking

Among equal scores, the **lower position index** wins. This mirrors ONNX Runtime
`TopK` (`total_cmp` order, ties by index) and is a hard part of the op contract:
the CPU oracle and the CUDA kernel must agree bit-for-bit on ties.

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

## CUDA slice — landed (native NVRTC kernel)

The CUDA NVRTC kernel `dsa_index_select_row` reproduces the frozen semantics
above **bit-for-bit** against the CPU oracle (integer `selected_indices`, so
parity is byte-exact, not tolerance-based). Design as landed:

- **One launch, one block per `(b, s)` row.** Phase 1 computes the head-weighted
  ReLU'd scaled score for every allowed key in parallel (threads widen f16/bf16
  storage to f32 with `__half2float` / `__bfloat162float`, matching the CPU
  oracle's `half::to_f32`, so scores are identical). Phase 2 runs the
  deterministic keep-round argmax on the lead thread using an integer
  `total_order_key` transform that reproduces Rust `f32::total_cmp` exactly
  (score desc, index asc), then emits the kept indices ascending and right-pads
  with the `-1` sentinel.
- **Un-fused reduction for exact rounding parity.** The score reduction uses
  `__fmul_rn` / `__fadd_rn` for the `dot += q·k` and `weighted += scored·wprod`
  accumulations so every multiply-add stays separately rounded. NVRTC compiles
  with the NVCC default `--fmad=true`, which would otherwise contract those
  `a + b·c` patterns into single-rounding `fma.rn.f32` and diverge from the
  CPU oracle's non-FMA order by a few ULPs — enough to flip an integer top-k
  selection at a cutoff tie. This matches the sibling `CompressedSparseAttention`
  kernel's convention. Single-operand sites (`scale·dot`, `weights·weights_scale`,
  `weighted + bias`) have no fusible partner and already match CPU rounding.
- **Fixed scratch, no page allocation.** The only device memory the op adds is a
  `B·S·T` f32 `scores` buffer plus a `B·S·T` u8 `state` buffer in the
  executor-owned **SessionPersistent** workspace (256-aligned, overwritten fresh
  each call). `onnx-genai-kv` remains the sole page/slot/lifetime authority — the
  op allocates and owns no pages.
- **Capture-safe.** The kernel produces indices (no index *input* to validate),
  so there is **no capture-error latch, no D2H, and no host sync on any path**.
  `warmed` flips true only after a non-capturing eager pass has NVRTC-compiled
  the kernel and sized the workspace; `capture_support()` reports `Supported`
  only once warmed, so a captured region performs no host alloc / sync / compile.
- **Claim gate.** `supports_op` delegates to the shared CPU
  `unsupported_reason`, first projecting the query/key/weights float dtypes to f32
  (CUDA supports f16/bf16 storage) while leaving `attention_bias` untouched, so
  the oracle's strict f32-only bias reject still fires. Quantized cache modes,
  head_sink, and q/k norm are out of this subset and typed-reject.

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

### A100 measurement (tiny + real GLM indexer dims)

Idle A100-80GB, pinned device, 8 s clock ramp (device 210 MHz → 1410 MHz across
the run), CUDA events over batched captured-graph launches for device time and a
drained small-batch wall-clock for pure host-enqueue, medians of n = 7 with
first-vs-last drift re-measure. **No full-size tok/s claim — tiny correctness is
the gate; this is a per-launch kernel-cost witness only.**

| Geometry (B,S,H,D,T,top_k) | device (µs) | host-enqueue (µs) | scratch (B) |
| --- | --- | --- | --- |
| tiny-decode (1,1,2,8,16,4)   | 7.6  | 2.2 | 512   |
| glm-prefill (1,64,2,8,64,4)  | 18.4 | 2.2 | 20480 |
| glm-decode  (1,1,2,8,512,4)  | 119  | 2.2 | 2560  |

Host-enqueue for a captured-graph launch is a flat ~2.2 µs independent of kernel
size (device time and submit cost are cleanly separated). First-vs-last drift on
the tiny geometry was **0.5 %**, and the min/max spread within each config was
under 0.5 %, so the device held still. Scratch VRAM is exactly `B·S·T·5` bytes,
confirming no page allocation. The wide-`T` decode row (T=512) is dominated by
the single-threaded Phase-2 selection (`O(keep·T)`); parallelizing it is future
perf work and is out of scope for this correctness slice (no full-size claim).
