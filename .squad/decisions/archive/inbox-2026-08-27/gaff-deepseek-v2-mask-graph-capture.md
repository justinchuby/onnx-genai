# Decision: recover whole-graph CUDA capture for DeepSeek-V2-Lite by pinning the decode-freeze-safe mask/bias length symbol

**Author:** Gaff (native-CUDA-EP kernel/executor specialist)
**Branch:** `squad/deepseek-v2-mask-graph-capture`
**Base:** commit `5a12882bd` (PR #1681, the coherence fix)
**Scope:** CUDA-graph capture classifier + decode mask freeze plumbing

## Context

PR #1681 made DeepSeek-V2-Lite int4 decode coherent on the native CUDA EP, but
had to force CUDA-graph capture **off** for that model (~75 tok/s) because
exposing the attention-mask binding's *logical* valid length makes it dynamic
per decode step (10, 11, 12, …), so it cannot be frozen for static capture.
Qwen/GQA avoids this: GQA takes the true length via explicit graph params
(`seqlens_k` / `total_sequence_length`), not from `Shape(mask)`.

## Root cause of the graph-off

The capture classifier disqualifies any node whose shape carries a "growing"
(KV / total-sequence-length) symbol, forcing it eager.
`pin_fixed_capacity_kv_capture_symbols` pins the KV-cache seq-axis symbols
(present/past slots) so attention is admitted for dense/GQA models. **But
DeepSeek's HF-style causal-mask/bias length axis is a SEPARATE symbol that does
NOT live on any KV slot**, so `collect_capacity_pinned_kv_symbols` never
captured it. Left unpinned it kept the entire additive causal-mask builder cone
(`CumSum`/`Unsqueeze`/`GreaterOrEqual`/`And`/`Where`/`Cast`) **plus all 27
`Attention` nodes** that consume the derived bias as forced eager seams.
Diagnostics (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`) showed **30 captured segments
/ 29 eager seams**; that fragile interleaving replayed incoherently /
non-deterministically once capture engaged.

## Decision

When `geometry::mask_binding_feeds_additive_causal_builder` holds — the SAME
predicate that drives the runtime decode mask-freeze
(`DeviceIoBinding::mask_decode_freeze_safe`) — the mask/bias length symbol is a
fixed-capacity constant on the **single-token decode** path: the frozen width
saturates the `CumSum` prefix to the true valid length and forces the padded
suffix to `-inf`, so a captured replay over the per-step-updated mask buffer is
byte-identical to the eager mask. It is therefore safe to pin exactly like a KV
seq symbol.

New `kernel_cache::collect_freeze_safe_mask_symbols(graph)` walks the additive
mask-builder cone forward from each freeze-safe mask input and collects **every**
symbolic dim on **every** cone value, including the capacity-form `Attention`
bias leaf. `pin_fixed_capacity_kv_capture_symbols` unions these with the KV
pins. (Symbol IDs are inference-minted and vary per run; the bias axis carries a
*derived* symbol distinct from the mask input's raw axis and not always recorded
as a derivation of it — so pinning only the input symbol is insufficient, hence
the full cone walk.)

Pinning empties the disqualifying set → **whole-graph capture (1 segment, 0
seams)**.

## Why this is safe for Qwen / dense / GLM (no regression)

`mask_binding_feeds_additive_causal_builder` returns **false** for GQA+`seqlens_k`
masks (no `CumSum` causal cone / no capacity-form `Attention` leaf) and for
GLM-5.2's indexer `Add` (excluded from `is_additive_mask_builder_op`). So
`collect_freeze_safe_mask_symbols` returns ∅ → identical pinned set →
bit-identical behavior. Confirmed empirically: Qwen pinned set = 17 symbols (all
KV; the freeze-safe collector adds 0). Capture engages ONLY on single-token
decode (prefill runs eager and keeps the logical length exposed), so the frozen
mask symbol never leaks `max_len` into multi-token prefill.

## Validation (before → after)

| Model | Base #1681 | This branch |
|---|---|---|
| **DeepSeek-V2-Lite int4** (CUDA greedy) | graph **off**, 75.4 tok/s, oracle-coherent | graph **on**, **165.3 tok/s**, oracle-coherent, `enabled=true captures=4 fallbacks=0`, whole-graph (1 segment / 0 seams), deterministic |
| **Qwen3.8-27B int4** (CUDA greedy) | 60.9 tok/s, `enabled=true captures=4 fallbacks=0` | **bit-identical tokens**, 61.3 tok/s, `enabled=true captures=4 fallbacks=0` |

- DeepSeek CUDA greedy tokens == CPU oracle for all 24 tokens:
  `[11, 304, 608, 245, 207, 16, 24, 1012, 1712, 5075, 13, 304, 608, 245, 1079, 37844, 1491, 13, 304, 608, 245, 1079, 2074, 18891]`.
- **2.2× decode throughput** for DeepSeek with no loss of coherence.
- `cargo test -p onnx-runtime-session --lib`: 190 passed (incl. two new
  geometry/symbol unit tests for `collect_freeze_safe_mask_symbols`).
- `cargo test -p onnx-runtime-ep-cuda --features cuda,cuda-13000`: 473 passed;
  only the pre-existing `a_module_restored_from_cached_ptx_computes_what_a_compiled_one_does`
  fails (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, an environmental cached-PTX
  toolchain mismatch present on all branches).

## Files

- `crates/onnx-runtime-session/src/executor/kernel_cache.rs` —
  `collect_freeze_safe_mask_symbols`.
- `crates/onnx-runtime-session/src/executor/capture.rs` — union freeze-safe mask
  symbols into `pin_fixed_capacity_kv_capture_symbols`.
- `crates/onnx-runtime-session/src/executor/geometry.rs` —
  `mask_binding_feeds_additive_causal_builder` (freeze-safe predicate) +
  `ShapeConsumptionPolicy`.
- `crates/onnx-runtime-session/src/tensor.rs`,
  `crates/onnx-runtime-session/src/executor/build.rs` — `decode_freeze_safe_mask`
  binding spec + wiring.
- `crates/onnx-genai-engine/src/native_decode/cuda.rs` — decode mask expose-len /
  capture-decline guard reads the freeze-safe flag.
- `crates/onnx-runtime-session/src/executor/tests.rs` — two new unit tests.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
