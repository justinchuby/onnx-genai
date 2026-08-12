# Decision: pin GQA fixed-capacity KV seq symbol so the capture classifier admits GQA — and the bf16 GQA-capture blocker it exposes

**Author:** Leon (Engine Dev — KV & Buffers)
**Branch:** `squad/gqa-kv-capture-pin`
**Date:** 2026-08-12
**Status:** Landed (classifier pin); flags a follow-on CUDA-EP blocker (bf16 GQA capture) for Sebastian.

## Context

Native CUDA decode of Muse-Glimmer-30B engages CUDA-graph capture but gets ZERO
speedup (14.52→14.58 tok/s, GPU 0–2% idle): the captured decode step fragments
into 54 segments / 53 eager seams — all 52 GroupQueryAttention (GQA) nodes plus 1
SkipSimplifiedLayerNormalization. Sebastian's root cause: the build-time capture
classifier force-declines every GQA node because the decoder graph's GQA
present/past KV boundary tensors carry a **symbolic (growing) penultimate seq
axis**, so the classifier seeds it as a growing symbol and vetoes each GQA node —
a **false positive** for the fixed-capacity, device-valid-length KV the runtime
actually binds (`DecodeCudaState`, physical `[1, kv_heads, max_len, head_dim]`).

## What I changed (the KV pin — my assigned scope)

At the point the engine binds fixed-capacity device KV, pin the GQA KV seq-axis
symbols CONSTANT so `collect_structural_growing_symbols` no longer seeds them and
the classifier admits GQA. Implemented as an **engine-gated symbol exclusion**
(the engine knows the binding is fixed-capacity; `Executor::build` does not):

- `crates/onnx-runtime-session/src/executor/kernel_cache.rs`
  - New `collect_capacity_pinned_kv_symbols(graph)`: the KV seq symbols of any
    attention node whose EVERY past-KV input is read as physical capacity
    (`geometry::kernel_input_uses_physical_capacity`). Growing-concat / paged /
    mask-less `Attention` and `CompressedSparseAttention` (no past inputs) do NOT
    qualify, so a genuinely growing KV path stays vetoed.
  - `_excluding` variants of the seed/compute fns that drop the pinned symbols
    from the structural SEED (so lineage closure never re-introduces them) and
    force-remove them from the final disqualifying set.
- `crates/onnx-runtime-session/src/executor/capture.rs`
  - New `Executor::pin_fixed_capacity_kv_capture_symbols()` recomputes
    `capture_growing_symbols` with the pinned KV axes excluded; records them in
    the new `capacity_pinned_kv_symbols` field.
- `crates/onnx-runtime-session/src/lib.rs`
  - `InferenceSession::pin_fixed_capacity_kv_capture_symbols()` pins both `exec`
    and the `decode_inline_exec` sibling.
- `crates/onnx-genai-engine/src/native_decode/cuda.rs`
  - `DecodeCudaState::new` calls the pin **only when `graph_enabled`** (a
    growing/paged decoder clears it and never pins — the veto is preserved for
    those paths).
- `state.rs` / `build.rs`: the `capacity_pinned_kv_symbols` field + init.
- `tests.rs`: 3 regression tests (see below).

## Why this is correct (and safe to land now)

1. On the fixed-capacity device-valid-length path the GQA launch grid is
   **capacity-sized** (`max_len`, constant within a capture); valid length is read
   on-device (`seqlens_k`); present = `past_capacity.max(total)` = `max_len`;
   overflow caught by `total_len > max_len`. Every replayed step has an identical
   grid — a captured replay is shape-static. This is the same reasoning the
   default-domain `Attention` present-shape widening already relies on
   (`dispatch.rs:755-800`), extended to GQA.
2. **Two-gate AND, unchanged:** a node captures only if (classifier says
   seq-independent) **AND** (kernel `capture_support()` says Supported)
   (`capture.rs` ~L660-668). My pin removes only the *classifier* gate for
   fixed-capacity GQA; the kernel gate remains an **independent authoritative
   backstop**. So the pin can never, on its own, cause a stale-grid capture.
3. **Growth/rebucket** still invalidates + recaptures via the untouched
   `ensure_capacity → invalidate_graph → reset_device_graph` machinery, plus the
   replay binding-signature guard.
4. **Byte-exact greedy parity preserved** — generated ids match Sebastian's
   reference exactly: `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511,
   917, 4921, 768, 328, 2885, 262, ...]`.

### Regression tests (all pass)
- `gqa_fixed_capacity_kv_seq_symbol_is_pinned_and_admits_the_node` — fixed-cap GQA
  KV seq symbol is pinned constant and the node (and a KV-sized consumer) become
  capture-eligible; without the pin both are vetoed.
- `growing_kv_paths_are_not_pinned_and_stay_vetoed` — a causal (growing-concat)
  `Attention` and a `CompressedSparseAttention` are NOT pinned and stay
  disqualifying (guard not blanket-disabled).
- `executor_pin_fixed_capacity_kv_admits_gqa` — end-to-end via the executor entry
  point.

## Measured result — pin works; a SECOND blocker (bf16) remains

`ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1
ONNX_GENAI_LOG_GROWING_SYMBOLS=1`, staged Muse-Glimmer, H200:

- **Classifier veto: GONE.** Log: build-time disqualifying set **53 symbols →
  after pin `disqualifying set now 0 symbol(s)`** ("pinned 53 fixed-capacity KV
  seq symbol(s)"). This is exactly the assigned fix, proven.
- **Segments still 54 / 53 eager seams; tok/s still ~14.6.** The 52 GQA nodes now
  decline at the **CUDA-EP kernel** gate, not the classifier: they are
  `[eager-device-seam]` with reason *"requires a warmed f32/fp16 q_seq==1
  k_seq<=1 fixed-capacity device-KV decode path; the current signature was not
  warmed as capture-safe"*.

### Root cause of the remaining blocker — bf16 (OUT OF MY SCOPE, flagged)

Muse-Glimmer's decoder is **bfloat16** (`model_config.json: "dtype":
"bfloat16"`; all 52 GQA nodes have BF16 past/present KV, verified in
`decoder/model.onnx`). The GQA kernel's capture-safe signature
(`group_query_attention.rs:1932`) admits only **Float32** or **Float16 +
`gqa_decode_fp16::supported`**. There is **no `gqa_decode_bf16`** device-length
split-K decode kernel (only `gqa_decode` (f32) and `gqa_decode_fp16` (f16) exist
in `kernels/mod.rs`). So `last_capture_safe_signature` is never set for bf16 →
`capture_support()` declines every bf16 GQA node.

**This is the bf16 effort the task pre-flagged as "a genuinely separate large
effort — flag it rather than expanding scope."** Collapsing Muse-Glimmer to 1
segment / 40+ tok/s requires a **bf16 capture-safe GQA decode kernel**
(`gqa_decode_bf16` mirroring `gqa_decode_fp16`, plus its numerical oracle/accuracy
gate), which is CUDA-EP/perf-domain work (Sebastian). My KV pin is a **necessary
prerequisite** — without it the classifier would still veto GQA even after a bf16
kernel lands — but it is **not sufficient** on its own for this bf16 model.

## Recommendation / handoff

- **Land the pin.** It is correct, tested, safe (kernel backstop intact), and a
  hard prerequisite for GQA capture on ANY model.
- **Sebastian (Perf / CUDA-EP):** add a bf16 capture-safe GQA device-length decode
  path (`gqa_decode_bf16`) so `capture_candidate`/`capture_support()` admit bf16
  q_seq==1 fixed-capacity aliased device-KV. Once that lands, this pin should
  collapse Muse-Glimmer decode 54→~2 segments (the residual
  SkipSimplifiedLayerNormalization warmup-signature seam is minor and separate).
- On an **f32 or f16** GQA model, this pin alone should already admit GQA into
  capture (no bf16 kernel gap) — a good validation target.
