# Decision: DeepSeek-V2-Lite int4 CUDA-EP coherence — mask capacity classification fix

**Author:** Gaff (native-CUDA-EP kernel/executor specialist)
**Branch:** `squad/deepseek-v2-cuda-coherence`
**Scope:** `crates/onnx-runtime-session/src/executor/geometry.rs` (+ regression test)

## Context

The native runtime runs `deepseek_v2` on the GPU (ORT-genai 0.14.1 cannot even
load this model). On the native **CPU EP** the exported ONNX graph decodes
coherently; on the native **CUDA EP** decode was incoherent garbage
(`", I amigo.\nsame, and the SMPS, and the SMPS, ..."`). Because CPU and CUDA
run the *identical* graph, the export is correct — the bug was in the CUDA-EP
executor wiring, specific to a DeepSeek-V2 op path that Qwen-dense never
exercises.

## Root cause (first divergent op → classification bug)

Per-op CPU-vs-CUDA activation bisection isolated the first real divergence at
**layer-0 `Attention` output** (d≈1.34) while every Q/K/V/RoPE input matched CPU
exactly. The `Attention` op has **no `is_causal` attribute** (defaults to 0), so
causality relies entirely on the in-graph *additive* causal mask. On CUDA that
mask's row 0 was `[0,0,0,…]` (non-causal) vs CPU `[0,-65504,…]` (causal) — the
model attended to future tokens ⇒ garbage.

The additive mask is built HF-style from `attention_mask`:

```
Slice(CumSum(attention_mask), start = Shape(attention_mask) - q_seq,
                              end   = Shape(attention_mask))
```

The CUDA mask input binding was **frozen to physical padded capacity (256)**
instead of the logical valid length (10). `Shape(attention_mask)` therefore
returned **256**, so `Sub` = 256-10 = 246 and `Slice` selected query positions
`[246..256]` → `[10,10,…]` instead of `[1..10]`. Wrong query positions →
`GreaterOrEqual` produced a non-causal mask.

The classification lives in
`mask_binding_feeds_capacity_form_attention` (geometry.rs). It treated the
default-domain `Shape` and `ReduceSum` at input 0 as "padded-capacity-safe
leaves" and blessed the mask binding to freeze to physical width. That premise
holds for `ReduceSum` (0-padding sums to nothing ⇒ logical length) but is
**false for `Shape`**, which returns the physical width and here feeds
width-sensitive `Sub`/`Slice` index arithmetic.

## Fix

In `mask_binding_feeds_capacity_form_attention`, keep `ReduceSum(mask)` as an
unconditional safe leaf, but treat `Shape(mask)` as safe **only when its output
is a dead end**. If anything consumes the `Shape` output, the padded `max_len`
leaks into the mask geometry, so the binding must expose its logical valid
length (forfeits CUDA-graph capture for that mask on DeepSeek — eager but
correct; op fallbacks stay 0).

## Why Qwen is unaffected (no regression)

Qwen uses `GroupQueryAttention` (com.microsoft) with explicit
`seqlens_k`/`total_sequence_length` and internal causal masking; its mask goes
only to `ReduceSum`→seqlens_k and `Shape`, hitting the fast path
(`all_direct_padded`) that returns *before* the modified function. The change is
literally unreachable for Qwen — confirmed at runtime: Qwen keeps
`cuda_graph: enabled=true captures>0 fallbacks=0` after the fix.

## Validation

- **CUDA greedy tokens now equal the CPU oracle for all 24 tokens:**
  `[11, 304, 608, 245, 207, 16, 24, 1012, 1712, 5075, 13, 304, 608, 245, 1079,
   37844, 1491, 13, 304, 608, 245, 1079, 2074, 18891]` — coherent text.
- **DeepSeek fallbacks = 0** (QMoE + Attention stay on GPU; `access=MoeRouted`,
  FullResident).
- **Qwen3.8-27B no regression:** coherent, `cuda_graph enabled=true`,
  fallbacks=0, ~61 tok/s (shared-box variance vs the ~64.8 baseline).
- `cargo test -p onnx-runtime-ep-cuda --features cuda,cuda-13000`: 473 passed,
  1 failed = the pre-existing environmental
  `a_module_restored_from_cached_ptx_computes_what_a_compiled_one_does`
  (CUDA_ERROR_UNSUPPORTED_PTX_VERSION, fails on main too).
- `cargo test -p onnx-runtime-session --lib`: 188 passed. Added regression test
  `deepseek_shape_feeding_slice_window_keeps_logical_width`; existing mask tests
  (including `vestigial_window_mask_builder_routes_to_padded_capacity` and
  `glm_indexer_add_mask_keeps_logical_width`) still pass.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
