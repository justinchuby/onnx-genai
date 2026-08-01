# Lane A perf — native multi-component pipeline CUDA-graph capture — READ-ONLY scope

**Author:** Cohaagen (EP/runtime)  ·  **Mode:** READ-ONLY scope, no production edits
**Worktree:** `/home/justinchu/onnx-genai/wt-cohaagen-perfcap` (detached @ origin/main `f3c6e796`)
**GPU:** CUDA_VISIBLE_DEVICES=2  ·  **Authority:** Justin — "确保高性能", full perf authority

## VERDICT: WIRING GAP — in fact a *default-flag flip*. LOW RISK.

The multi-component pipeline decode step **already has a complete CUDA-graph capture
path** (`Inc3c`, issue #384). It reuses the exact monolithic `run_one_token`
warmup→capture→replay state machine, and already ships **parity + engagement + decline
tests**. It is simply **gated OFF by default** behind an env flag. The 6.5× gemma4-e2b
slowdown is the *eager* fallback that runs because that flag is unset. This is NOT missing
infra; it is a conservative default left from bring-up.

Empirical proof gathered this session (GPU 2):
- `native_cuda_captured_step_inputs_parity` (tiny-gqa-embeds-cuda, real GroupQueryAttention
  KV) → **PASS**: `captured_decodes=3`, tokens `[0,5,6,7]` byte-identical to the eager path.
  The captured multi-component path engages and is token-exact.
- Real gemma4-e2b (`/home/justinchu/mobius/.scratch/gemma4-e2b-native`) capture test →
  **graceful-skip**: blocked by `vision_encoder` `OneHot ... Depth is negative` even for a
  text-only prompt (the pipeline forces the vision component to run). **This is the same
  orthogonal stale vision/audio export blocker — not a capture problem.**

---

## Q1 — Capture path for MONOLITHIC native decode (baseline, works today)

The capture engine is `DecodeCudaState` in
`crates/onnx-genai-engine/src/native_decode/cuda.rs`:
- `run_one_token` (cuda.rs:1725) is the state machine: `DecodeCudaGraphPhase`
  **NeedsWarmup** (eager warm run) → **Armed** (`try_capture_with_device_bindings`) →
  **Ready** (`replay_device_graph` every subsequent token). Counters `graph_captures` /
  `graph_replays` / `graph_fallbacks` (cuda.rs:1426-1428).
- Under the hood it calls the session-level capture API
  (`onnx-runtime-session/src/lib.rs:1118 try_capture_with_device_bindings`,
  `:1131 replay_device_graph`), which drives the EP-level
  `begin_device_graph_capture`/`replay` → `CudaGraphLifecycle`
  (`onnx-runtime-ep-cuda/src/graph.rs:114`, `runtime.rs:407/429`).
- A monolithic token-id decode (qwen2.5/qwen3) reaches `run_one_token` via
  `decode_cuda` cuda.rs:450-455 for the `token_ids.len()==1` step, so it **does**
  capture+replay today. Proof: the tiny parity test above shows `graph_captures>0` on a
  GQA decoder; the parity-test module doc records the captured path is "the Part A
  612/220/443 lever" and that monolithic native beats ORT-CUDA once captured.

## Q2 — MULTI-COMPONENT pipeline decode loop, and the EXACT reason capture is off

Pipeline decode drives the decoder component through:
`pipeline/decoder_component.rs:239 step()` → `session.decode_with_step_inputs(...)`
(`native_decode/mod.rs:438`) → `decode_cuda` (`native_decode/cuda.rs`).

The decoder component's decoder declares an `inputs_embeds` port (embedding output) plus
routed ports (e.g. `per_layer_inputs`), so `has_eager_step_inputs()` (cuda.rs:490) is
**true**. In `decode_cuda`, cuda.rs:394-414:

```rust
if self.has_eager_step_inputs() {
    let capture = token_ids.len() == 1
        && self.cuda.as_ref().is_some_and(|state| state.capture_step_inputs); // <-- gate
    if capture { return self.decode_cuda_captured_step_inputs(...); }         // Inc3c: CAPTURES
    return self.decode_cuda_eager_step_inputs(...);                           // default: NO capture
}
```

**Exact reason capture does not happen:** `state.capture_step_inputs` is **false by
default**. It is set at cuda.rs:1403-1404:

```rust
let capture_step_inputs =
    graph_enabled && !captured_step_inputs.is_empty() && capture_step_inputs_enabled();
```

and `capture_step_inputs_enabled()` (cuda.rs:250-258) reads env
`ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS`, **defaulting to `false`**. The in-code
comment (cuda.rs:1398-1402) is explicit: *"enable the captured per-step-input path only
when (a) the operator opted in … Off by default keeps the eager owned path
byte-identical."*

So: the pipeline decoder falls to `decode_cuda_eager_step_inputs` — it binds the one-token
embedding/routed tensors into fresh **owned** device inputs each step and runs
`run_with_device_bindings` **eagerly** (no capture, full per-op launch cost every token).
`decode_cuda_captured_step_inputs` (cuda.rs:508) instead writes those one-token tensors
into the **persistent** bindings and reuses `run_one_token` — identical mechanism to the
monolithic path. Both exist; only the default gate differs.

## Q3 — Is the plain-transformer decoder step capture-ELIGIBLE?  YES.

`graph_enabled` is only cleared for structural reasons (cuda.rs:1358-1396):
- a persistent binding exposing a **growing logical prefix** (`dynamic_logical`), or
- the **attention-mask binding exposing its logical length** to a non-capacity-aware
  consumer (`mask_exposes_logical`, e.g. GLM-5.2's indexer `Add`).

A plain-transformer GQA decode step (gemma4-e2b, qwen, glm-GQA) at seq_len=1 is
straight-line with capacity-aware GroupQueryAttention KV → **no dynamic-logical binding, no
control-flow node** → `graph_enabled=true`. Empirically confirmed: `tiny-gqa-embeds-cuda`
(inputs_embeds + real GQA) engages whole-graph capture (`captured_decodes=3`). This is
**independent of the parked 27B Scan-capture lane**: the prefill/decode shared-plan Scan
trap only affects Scan/LinearAttention hybrids (Qwen3.5 27B / 35B-A3B), which have
recurrent `conv_state`/`recurrent_state` — a *follow-up*, not this increment. Capture-
ineligible decoders (GLM indexer / mask-exposes-logical) **auto-decline** to the eager path
via `graph_enabled=false` — no silent-wrong risk.

## Q4 — Regression-guard plan (Justin is regression-sensitive)

The guards **already exist** and pass; the increment extends them to assert the new
default:
1. **Byte-identical capture-on == eager-off**, non-vacuous:
   `tests/native_cuda_captured_step_inputs_parity.rs` runs the pipeline with the flag on and
   off and asserts (a) identical token IDs and (b) the process-global counter
   `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` is **>0 with capture, 0 without** — so a
   silent decline to eager fails the test. (Verified: PASS, `captured_decodes=3`.)
2. **Real-model engagement** (token-for-token free):
   `tests/gemma3n_native_cuda_capture_realmodel.rs` (`--ignored`) asserts the captured path
   engages on the real gemma4-e2b decoder. (Currently skips on the orthogonal vision-export
   blocker — see below.)
3. **Honest decline:** `tests/qwen3_0_6b_capture_step_inputs_decline.rs` proves a
   single-component `input_ids` model leaves the counter at 0 (uses the token-id capture
   mechanism, not step-inputs) — the flag doesn't change unrelated paths.

Increment adds: flip these to assert **default-on** (counter >0 *without* setting env), and
one guard that a `mask_exposes_logical` (GLM-like) decoder still declines to eager and stays
token-exact with the default on.

---

## Ranked increment plan

**Increment 1 (RECOMMENDED — the flip).** Make the captured step-input path the default for
capture-eligible multi-component decoders.
- Change: invert `capture_step_inputs_enabled()` to **default-on** (env becomes an
  **opt-out**, e.g. `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS=0` to disable);
  `capture_step_inputs` stays `graph_enabled && !captured_step_inputs.is_empty() && …`, so
  ineligible decoders auto-decline. File: `native_decode/cuda.rs:250-258` (+ the doc
  comment at 1398-1402). ~10 lines.
- Tests: update the three tests above to assert default-on + add the GLM-decline guard.
- Correctness bar: `native_cuda_captured_step_inputs_parity` byte-identical with default-on;
  counter non-vacuous; `--test-threads=1`. `cargo fmt --all --check`.
- Blast radius: **LOW.** Engine-level `native_decode` only. Does **not** touch the parked
  capture core, `plan_capture_region`, `standard_attention`, or the GAP-3 KV path. Cannot
  regress the monolithic token-id path (different branch) or single-component models
  (decline proven). Eager path stays as the structural fallback.
- Unblocks / upside: every capture-eligible multi-component decode (gemma4-e2b text, any
  2-component GQA text model, and the 35B-A3B GQA layers). Estimated **~2.5–6× decode
  throughput**: the parity-test module records eager is ~2.8× below the captured ceiling and
  ~2× below ORT-CUDA on qwen3-0.6b; the gemma4-e2b investigation measured native 0.6 vs ORT
  3.9 tok/s (6.5×) dominated by per-step launch overhead. Capturing recovers most of that —
  expect native to reach **ORT-parity-or-better** (~3.9+ tok/s), matching the monolithic
  story where captured native beats ORT-CUDA.

**Increment 2 (observability, small).** Surface the decoder component's
`graph.captures/replays/fallbacks` and `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` through
`profile_native --pipeline` (today `run_pipeline` at `profile_native.rs:422-562` prints only
tok/s; the single-model path at :766-772 already prints capture stats). Lets the bench prove
capture-hit + tok/s on/off directly. LOW risk, bench-only.

**Increment 3 (follow-up, NOT this lane).** Recurrent-hybrid (Qwen3.5 27B / 35B-A3B)
step-input capture — depends on the parked Scan/`recurrent_state` capture work (Mary's lane).
Gate separately; the mask/`dynamic_logical` decline already keeps it safe (eager) until then.

**Orthogonal blocker (flag, not our lane).** A real gemma4-e2b *end-to-end* capture
benchmark is blocked by the stale vision/audio export: package admission rejects
`audio_encoder.audio_features` rank 3 vs embedding rank 2 (`/home/justinchu/gemma4-e2b-onnx`),
and the non-stale `.scratch` export fails `vision_encoder OneHot Depth is negative` even
text-only because the pipeline forces the vision component to load. Fixing that (re-export
via mobius, or a native/optional-modality text-only prologue = Inc-B) is a **separate
lane** (model-package / vision). Increment 1's win is provable at the fixture level today and
lands the moment a clean gemma4-e2b export or text-only skip exists.

## RECOMMENDED FIRST INCREMENT

**Increment 1 — default-on the captured step-input path** (invert
`ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` to opt-out).
- **Correctness bar:** `native_cuda_captured_step_inputs_parity` byte-identical tokens with
  default-on + non-vacuous capture counter; GLM-like decoder still declines to eager and
  stays token-exact; `--test-threads=1`; `cargo fmt --all --check`.
- **Est. upside:** ~2.5–6× native decode throughput on multi-component GQA models; brings
  gemma4-e2b native from ~0.6 tok/s toward/above ORT's ~3.9 tok/s.
- **Risk:** LOW — a proven, parity-tested path already merged behind a conservative default;
  no capture-core / KV-core / GAP-3 contact; eager path retained as fallback.

---
*Evidence: capture-path trace (file:line above) + on-GPU runs (`captured_decodes=3` parity
PASS; real-model test graceful-skip on the vision-export blocker). No production edits, no
PR; git tree clean.*
