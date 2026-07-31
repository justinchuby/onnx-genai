# Rank-3 mrope native decode positions — the last mile for qwen3.5-0.8b hybrid native-CUDA

**Author:** Mary (native_decode / Inc3) · **Status:** design → implement · refs #384, #67
**Branch:** `squad/native-rank3-mrope-positions` (off origin/main) · handoff: `cohaagen-hybrid-loader.md`

## Problem

The native step driver hardcodes **rank-2** `position_ids`:

- `native_decode/cuda.rs:295` (eager token/inputs_embeds path, `run_cuda_eager_rows`) → `Tensor::from_i64(&[1, token_ids.len()], &positions)`
- `native_decode/cuda.rs:625` (eager step-inputs path, `prepare_cuda_owned_step_inputs`) → same rank-2 shape
- `native_decode/cpu.rs:203` (CPU step-input build) → same rank-2 shape
- `native_decode/cuda.rs:1284` (Inc3c captured device binding) → allocates `[1,1]`; `write_decode_inputs`/`write_captured_step_inputs` write a single i64 at offset 0

The Qwen3.5-0.8B hybrid decoder (`text.onnx`) declares **rank-3 mrope** `position_ids`
`[3, batch_size, sequence_length]` (verified via onnx: leading dim **static 3**, Int64).
So the native forward fails: `input position_ids: rank mismatch (graph declares rank 3, got 2)`.
The loader + text-only pipeline synthesis + hybrid state-init are all on main (#535) and PROVEN
via the ORT reference decode (`qwen35_0_8b_hybrid_text_decode_e2e`, coherent "Paris", 16-token lock).
Native-CUDA decode is the ONLY remaining gap.

## The value-type / rank seam (verdict)

The ORT path already builds rank-aware positions in `decode/step.rs::build_position_step`:
it reads `rank` from the pipeline `PositionProgram` (or falls back to physical shape), then for
`linear_increment` sets `starts = vec![absolute_start; rank]` and emits, per axis, the linear
sequence `[start, start+1, …]` — physical shape `[1, S]` for rank 1, `[rank, 1, S]` for rank > 1.
For **pure-text** linear_increment mrope every coordinate axis advances with the sequence position,
so a one-token step at absolute position `t` is `[t, t, t]` shaped `[3, 1, 1]`.

The native drivers must build the **identical** tensor — the seam is purely the physical `i64`
position layout (rank + per-axis linear values); the value TYPE is a plain `Tensor::from_i64`
(nxrt) vs `Value::from_vec_i64` (ORT) wrapping the SAME `Vec<i64>` + shape. No dtype conversion,
no device round-trip. So the fix is: **compute the shared `(Vec<i64> data, Vec<i64> shape)` once,
wrap it in each backend's tensor type.**

## Fix (general, metadata-driven — NOT hardcode-to-3, NOT a model-name gate)

1. **Shared helper** `decode::position_ids_from_starts(starts: &[i64], input_len) -> (data, shape)`
   factored out of `build_position_step`'s existing inline loop (lines 394–427), re-exported
   `pub(crate)` from `decode/mod.rs`. Both the ORT step driver AND the native drivers call it, so
   they build byte-identical positions (no fork). ORT stays byte-identical (mechanical extraction;
   the rank-1 `linear_increment` legacy-copy override is preserved).

2. **Rank derivation** `native_decode` reads the coordinate rank from the **graph's declared
   `position_ids` input shape** (the same source that raises the rank-mismatch error) — physical
   rank 2 → `position_rank = 1` (legacy `[1, S]`); physical rank 3 → `position_rank =` the
   declared **static** leading dim (3 for mrope); a symbolic leading dim or any other rank is a
   loud error (we cannot invent the axis count). This is metadata-driven and needs **zero** new
   plumbing through `ModelIoSpec`/constructors. Continuation for the native text-decode path is
   `linear_increment` (every axis advances with the position) — the only continuation the native
   text path supports; `from_grid`/`carry_max` need routed processor coordinates (vision), which is
   scoped OUT of this increment and already refused upstream.

3. `position_rank` is computed once at construction and stored on both `NativeDecodeSession`
   (eager + CPU builds) and `DecodeCudaState` (captured device binding). Both use the SAME
   `declared_position_rank` helper, so they cannot drift.

4. **Both cuda.rs AND cpu.rs** build identical positions (parity). The captured device binding is
   allocated `[position_rank, 1, 1]` and each step writes `position_rank` copies of the position
   (rank-1 → byte-identical to today).

## Increment split

This is a single reviewable slice: **Inc3d = rank-aware native positions**. Sub-parts land together
because they are one seam:
- (a) shared helper + rank derivation,
- (b) eager cuda + cpu builds,
- (c) captured device binding (kept rank-correct for generality even though the hybrid uses the
  eager path — its hybrid conv/recurrent state is not GQA-capacity KV, so capture stays dormant;
  see validation).

## Deliverable / validation

1. `qwen35_0_8b_hybrid_native_cuda_e2e` (auto-activating harness on main, #535/#529) now RUNS native
   (no longer skips the rank-3 gap) and enforces native-CUDA ↔ ORT token-for-token parity on the
   real qwen3.5-0.8b hybrid. This is the first real-weights `inputs_embeds` split-package native-CUDA
   decode == ORT proof. Device 0/4.
2. **Regression:** a standard rank-2 `position_ids` decoder still builds `[1, S]` and decodes
   byte-identically (existing synthetic pipeline fixtures + qwen3-0.6b). The cuda,native-backend
   failing-set stays 17 (byte-identical to base).
3. **Bonus:** whether the capture-step-inputs flag now ENGAGES on the hybrid (inputs_embeds) — expected
   to DECLINE (hybrid conv/recurrent state is not GQA capacity-aware KV → `graph_enabled = false`),
   reported honestly with the counter. If it declines, the hybrid still proves native decode CORRECTNESS
   on real weights; capture engagement remains a GQA-decoder property (35B-A3B).

## Result — it RUNS and is TOKEN-IDENTICAL to ORT (real weights, on-GPU)

Implemented and PROVEN on device 4 (release, ORT 1.27.0 CUDA). The auto-activating harness
`qwen35_0_8b_hybrid_native_cuda_e2e` now RUNS native (no longer skips the rank-3 gap) and locks
native-CUDA ↔ ORT token-for-token:

```
qwen3.5-0.8b hybrid ORT reference : [11751, 11, 321, 279, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369]
qwen3.5-0.8b hybrid native CUDA    : [11751, 11, 321, 279, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369]  ✅ IDENTICAL
```

This is the first real-weights `inputs_embeds` split-package native-CUDA decode == ORT proof.

### Second gap found + fixed (native-CUDA `Range` kernel, NOT position-driver)

With rank-3 positions correct, the native forward reached a **deeper** native-CUDA op gap: the
mrope rotary `Range` (`/model/layers.*/attn/k_mrope/range/Range`) supplies its start/limit/delta
as single-element `[1]` tensors, but `onnx-runtime-ep-cuda/src/kernels/range.rs` required strict
rank-0 scalars and errored: *"cuda_ep Range: inputs must be same-dtype contiguous scalars…"*.
ONNX `Range` is scalar-valued, and real exports commonly emit `[1]`-shaped scalars (ORT's CPU/CUDA
accept both); the kernel already reads only the first element. Fix: accept any contiguous
**single-element** (`numel() == 1`) input — rank-0 or `[1]` — a small, general, spec-correct
relaxation. This is the last-mile op gap; without it the composition claim ("100% native placement")
placed but could not execute the mrope range.

### Regression — rank-2 unchanged

`position_rank` defaults to `1`; the shared helper's rank-1 branch emits exactly `[1, S]` with the
same linear values, and the captured binding stays `[1, 1]` with one write — byte-identical to
before. Proof: 349 engine lib tests green (incl. every existing ORT multi-axis / rank-2 position
test — the shared-helper extraction kept the ORT path byte-identical); new deterministic unit tests
(`position_helper_tests::*`, `declared_position_rank_maps_graph_shape`) lock rank-1 → `[1, S]` and
rank-3 → `[3, 1, S]`; the full `cuda,native-backend` failing-set stayed **17** (documented
pre-existing: native_engine 10 / gemma4_assistant 3 / glm_tiny_qmoe 2 / native_prompt_lookup 2),
byte-identical to base.

### Bonus — capture-step-inputs on the hybrid: DECLINES (no-op), token-identical

With `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS=1` the hybrid decodes to the **identical** 16
tokens (flag ON == OFF == ORT). Expected: the hybrid's `conv_state`/`recurrent_state` are
wholesale-replaced fixed-state inputs, not GQA capacity-aware appendable KV, so `graph_enabled =
false` and the capture-step-inputs path stays dormant — the flag is a no-op. Native decode
CORRECTNESS on real weights is proven regardless; capture engagement remains a GQA-decoder property
(the 35B-A3B path), consistent with the qwen3-0.6b decline finding.

## Out of scope (unchanged rank-2)

The speculative proposer draft path (`native_decode/proposer.rs`) still builds rank-2 positions —
it is a distinct `NativeProposerSession` and no rank-3 proposer exists; deferred with the rest of
speculative-native work.

