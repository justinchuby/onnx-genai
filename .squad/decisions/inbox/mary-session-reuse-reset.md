# Decision: native-CUDA session-reuse recurrent-state reset (LinearAttention corruption)

- **Author:** Mary (Lead, large-model memory-offload workstream)
- **Date:** 2026-07-31T08:11:36+0000
- **Requested by:** Justin (@justinchuby)
- **Branch:** `squad/native-session-reuse-reset`
- **References:** #384 (native-CUDA large-model decode); discovered during the
  27B weight-offload A/B (see `mary-27b-native-offload-ab.md`). Possible link to
  Justin's "repeated / degenerate sentence" report on multi-turn native decode.

## Scope (decided EARLY, per Step 0)

**LinearAttention-only — NOT a general session-reuse bug.**

Step-0 blast-radius test on a plain GQA transformer (KV-only, no conv/recurrent
state), `qwen2.5-0.5b-instruct-cuda`, `profile_native --runs 2`:

- GQA: **clean / deterministic** — gen#2 byte-identical to gen#1.
- 27B hybrid LinearAttention (`conv_state`/`recurrent_state`): **corrupts** gen#2+.

So the general KV/position/decode-step reset path is fine for everyone. The bug
is confined to models that carry fixed-size recurrent/conv state.

## Root cause

`DecodeCudaState` (`crates/onnx-genai-engine/src/native_decode/cuda.rs`) appends
the fixed recurrent/conv-state device bindings after the growable KV bindings.
Those fixed-state bindings are `memset_zero`'d **once, in `new()`**, and never
again. `reset()` → `rewind(0)` (called at the start of every `generate()`) only
re-zeroed `bindings[0]` (the attention mask) and reset `logical_len`.

Growable KV is safe on reuse because it is masked and length-tracked (stale
slots are inert). Fixed recurrent state is a **wholesale, unmasked rolling
cache**: on the 2nd generation it inherited generation #1's *terminal* recurrent
state, so decode #2 started from garbage → non-deterministic degenerate output.

This reproduces both with CUDA-graph capture ON and with `ONNX_GENAI_CUDA_GRAPH=0`
(eager), confirming it is a state-reset bug, not a capture bug.

## Fix (general, no model-specific special-casing)

In `cuda.rs`:
- Added `fixed_state_binding_range: Range<usize>` to `DecodeCudaState`,
  populated in `new()` as `kv_end..fixed_state_end` (empty for pure-KV models).
- In `rewind()`, when `target_len == 0`, re-zero every binding in
  `fixed_state_binding_range` via `native_cuda_memset_zero` (same zero-init the
  constructor applies; `state_pairs` declare `init: zeros`). Only at the reset
  boundary — speculative recurrent rewind to a non-zero length is intentionally
  unsupported, mirroring the CPU path.

No model names, no architecture switches: any decoder that declares fixed
recurrent/conv state pairs is reset correctly. Pure-KV decoders have an empty
range and are entirely unaffected (verified: GQA path unchanged).

CPU native was checked and is clean here — its full reset clears the recurrent
state via the `past` map on `rewind(0)` — so the regression test is CUDA-gated.

## Regression test (non-vacuous)

`native_cuda_reused_session_rezeros_recurrent_state`
(`crates/onnx-genai-engine/src/native_decode/tests.rs`, `#[cfg(feature="cuda")]`,
gated by `ONNX_GENAI_RUN_CUDA_SMOKE=1`).

Builds a synthetic recurrent decoder whose **`logits` are a direct function of
the incoming `conv_state`** (`logits = ReduceSum(Cast(conv_state))`), and whose
next state accumulates the decoded token id. Decodes a fixed 3-token sequence,
`reset()`s, decodes it again, and asserts the two logits sequences are identical.
It also asserts the per-step logits strictly grow, proving the state actually
feeds the logits (guards against a vacuous pass).

**Non-vacuity proof** — with the re-zero disabled:

```
gen#1 [0.0, 60.0, 144.0] != gen#2 [252.0, 312.0, 396.0]   → FAIL
```

With the fix, gen#2 == gen#1 → PASS. (The existing `profile_native --runs 2`
determinism check is the natural end-to-end CI guard on real models.)

## End-to-end verification (real models, device 4/6, greedy)

Prompt `"The capital of France is"`, `--tokens 16 --warmups 0 --runs 2`:

| model | capture | gen#1==gen#2 | output |
|---|---|---|---|
| 27B int4 LinearAttention | ON (captures=2) | ✅ deterministic | `" Paris.\n\n<think>\n\n</think>\n\nThat is correct. Paris is the capital and"` |
| 27B int4 LinearAttention | OFF (`ONNX_GENAI_CUDA_GRAPH=0`) | ✅ deterministic | identical to above |
| qwen2.5-0.5b GQA (regression) | ON | ✅ deterministic | `" Paris. It is the largest city in France and the second largest in the European"` |

Before the fix the 27B failed: `native greedy decode was not deterministic:
first=[11751,...] rerun=[279,6511,...]`.

## Exact repro command

```bash
source .cudaenv.sh
export ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0
export ONNX_GENAI_REQUIRE_CUDA=1 CUDA_VISIBLE_DEVICES=4
./target/release/profile_native \
  --model <27b-int4-native-pkg> --ep cuda --backend native \
  --tokens 16 --warmups 0 --runs 2 --prompt "The capital of France is"
```

(27B package: `model.onnx`/`model.onnx.data`/`tokenizer.json` from
`~/mary-models/qwen3.6-27b-int4-cuda/` + `inference_metadata.yaml` carrying the
`io:` block copied from the `qwen36-ortref` sibling, as in the offload A/B.)

Unit regression:
```bash
ONNX_GENAI_RUN_CUDA_SMOKE=1 CUDA_VISIBLE_DEVICES=6 \
  cargo test --release -p onnx-genai-engine --lib --features "cuda native-backend" \
  native_cuda_reused_session_rezeros_recurrent_state
```

## Guardrails honored

- Did not touch `weight_paging.rs` / `provider.rs` (harry-6, #544 / squad/87-async-pagein).
- Did not switch the main checkout branch; all work in worktree `wt-mary-session-reset` off `origin/main` (fa1afed3).
- This is the sync page-in / native-decode session-state domain only.
