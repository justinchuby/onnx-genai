### GAP-3 native pipeline decode — scope, current state, and remaining decomposition

- **By:** Cohaagen (EP/runtime perf & engine)
- **Date:** 2026-08-03
- **Branch:** `squad/gap3-native-pipeline-decode-1` (off `origin/main` @ `be6d4e34`)
- **Status:** design + honest state audit. Increment 1 = a bounded, zero-behavior docs
  truth-up (PR open for Harry). The substantive GAP-3 increments are already merged.

## TL;DR — the task premise is stale; GAP-3's core is DONE on `origin/main`

The task cites `pipeline/mod.rs:303-338` (`build_native_pipeline_and_report_gap`,
*"replacing `DecodeState` and `PipelineDecodeLoopBackend` ownership of ORT `Value`/
`Session` with backend-neutral tensors/component sessions…"*). That function and its
error **no longer exist on `origin/main`** — it was renamed to
`native_pipeline_plan_unsupported` and its scope narrowed to *non-flat-autoregressive*
plans only. The task was authored against the local checkout `1ba215ee`
(`fix(session): expose bound inputs to control flow #443`), which is **~35+ merged PRs
behind** `origin/main` (`be6d4e34`, `#612`).

Between those two points the entire GAP-3 pipeline-decode-loop de-coupling landed:

| Concern | Where | PR |
|---|---|---|
| Backend-neutral component ownership seam | `pipeline/decoder_component.rs`, `ComponentSession` | #546 |
| Native device-KV decoder via `PipelineDecoderComponent` (Inc2b) | `NativePipelineDecoder` | #479 |
| Pure-native multi-component decode wiring (Inc-A) | `flat_autoregressive.rs`, `native_component_selection` | #565 |
| Native present-KV mirroring for paged native decode (Inc-C) | `decoder_component.rs`, `native_decode/mod.rs` | #566 |
| Device-resident present-KV read-out for paged native CUDA (Inc-D / D.1) | `native_decode/{mod,cuda}.rs` | #567/#568 |
| rank-3 mrope native positions (unblocks the hybrid) | `decode/step.rs`, `resolved_io.rs` | #543 |
| Text-only decode pipeline synthesis (loader unblock) | loader | #535 |
| fp16 TopK (dense_fallback MoE routers on CUDA) | ep-cuda | #612 |

**The decode loop is already backend-neutral.** `PipelineDecodeLoopBackend`
(`pipeline/paged_decode.rs`) holds `decoder: Box<dyn PipelineDecoderComponent>` and
`step_components: Vec<(_, Box<dyn ComponentSession>)>` — no ORT `Session` is owned by
the loop. `DecodeState`/ORT `Value`/`Session` are confined to the **ORT impl**
(`OrtPipelineDecoder`); the native impl (`NativePipelineDecoder`) owns a
`NativeDecodeSession` with device-resident KV and never crosses a concrete ORT type
except the host tensor container (see "residual coupling" below).

**The 35B-A3B shape already decodes natively and is conformance-locked.** Its
architecture (gated LinearAttention + causal short-conv + GQA + rank-3 mrope +
dense_fallback MoE router) is exercised in miniature by
`tests/qwen35_0_8b_hybrid_native_cuda_e2e.rs` (Qwen3.5-0.8B hybrid), which enforces
**token-for-token parity vs the ORT reference** the instant native decode runs, plus
`native_pipeline_backend_selection_parity.rs`, `native_pipeline_decoder_parity.rs`,
`native_cuda_routed_pipeline_decoder_parity.rs`, and `native_full_pipeline_parity.rs`.

**Conclusion:** there is **no monolithic ORT-coupling left to break apart** for the
flat-autoregressive native path, and **no large new zero-behavior increment** is
available — the substantive increments are merged and tested. What remains is a set of
*feature-sized* extensions (each needs new op/attention support and/or fixtures, i.e.
NOT zero-behavior) plus one genuinely bounded docs truth-up.

## The ORT-typed fields/methods (already de-coupled — for the record)

For completeness, the ORT-typed state the original task wanted made neutral, and the
neutral replacements that already exist (reuse, no parallel abstractions):

- `DecodeState { past/loop_state: HashMap<String, Value>, io: ResolvedIo, runner:
  Option<DecodeRunner>, … }` (`decode/state.rs`) and `run_decode_step_with_extra(session:
  &Session, …)` → confined behind **`trait PipelineDecoderComponent`**
  (`decoder_component.rs:35`). ORT keeps them in `OrtPipelineDecoder`; native uses
  `NativeDecodeSession` (`native_decode/mod.rs:84`).
- `PipelineDecodeLoopBackend.decoder` → `Box<dyn PipelineDecoderComponent>` (neutral).
- `PipelineDecodeLoopBackend.step_components` → `Box<dyn ComponentSession>` (neutral;
  `onnx_genai_metadata::ComponentSession`, native impl `NativeComponentSession`).
- Per-step KV mirror/logits: `mirror_last_present_kv` / `next_token_logits` on the
  trait (native reads via `NativeDecodeSession::present_kv` / `seed_kv`).

**Residual, non-blocking coupling (purity only):** the shared tensor pool is still
`PipelineTensors = HashMap<String, Value>` (`pipeline/mod.rs:51`) and `decoder_extras()`
returns `Vec<(String, Value)>` (`paged_decode.rs:183`); the native decoder converts each
extra `Value → ComponentTensor → native Tensor` per step (`decoder_component.rs:250-256`).
This is one small per-step host round-trip for routed inputs (e.g. one token's
`inputs_embeds`); the expensive KV stays device-resident. It does **not** block native
decode — it is a candidate cleanup (R5), not a gap.

## Remaining decomposition (ordered; each is feature-sized unless noted)

Ordering is by value-for-the-35B-native-number and independence.

1. **R1 — Docs truth-up (increment 1, this PR). Size S. Zero-behavior.**
   `native_component.rs:12-15` still asserts *"wiring these neutral sessions and tensors
   into the ORT-owned pipeline decode loop is the remaining GAP 3 work"* — false since
   #546/#565. Correct it to describe the now-neutral loop and the merged Inc-A/C/D. No
   code path changes; guards against a future contributor re-doing finished work.
   *Test strategy:* compile + `cargo clippy -D warnings` (comment-only; no behavioral
   test possible or warranted).

2. **R2 — Native sliding-window paged mirror. Size M. NOT zero-behavior.**
   `NativePipelineDecoder::{sliding_window→None, sink_tokens→0, retained_kv_len→past_len}`
   are hardcoded (`decoder_component.rs:351-364`). A windowed native decoder that reports
   `supports_paged_kv()==true` would mirror pages in the wrong (absolute vs retained)
   index space. Today unreachable (native attention has no windowing yet + no fixture),
   so it is a latent hole, not an active bug. Two sub-steps: (a) a **defensive gate** —
   thread the model's declared window (`sliding_window_from_metadata`, `decode/metadata.rs:175`)
   into `NativeDecodeSession` and make `supports_paged_kv()` return `false` when a window
   is present (keeps such models on the non-paged fresh-decode path, no regression);
   (b) the real windowed mirror. *Test:* a tiny windowed native fixture; parity vs ORT
   paged mirror; and a unit test that a windowed native decoder stays off the paged path.

3. **R3 — Inc-D.2 discontinuous attention-sink prefix reuse. Size M. NOT zero-behavior.**
   `NativePipelineDecoder::load_paged_prefix` bails when `start_position != 0 || sink_len
   != 0` (`decoder_component.rs:380-387`), matching the current ORT restriction. Enabling
   offset re-seed (host + device) unlocks cross-request KV reuse for windowed/sink
   prefixes. *Test:* parity vs ORT for a shared-prefix reuse that begins at a non-zero
   absolute position.

4. **R4 — Native cross-attention / vision KV (Inc3). Size L. NOT zero-behavior.**
   Encoder-decoder / VLM `past_*_cross_%d` binding on the native decoder. Not needed for
   the text-only 35B-A3B number, but required to bring VLM/codec pipelines to native
   parity. *Test:* extend `vlm_pipeline_e2e` / `codec_pipeline_e2e` native parity.

5. **R5 — Non-flat plans native (nested-AR TTS, iterative diffusion, composite). Size L.**
   Currently rejected up front by `native_pipeline_plan_unsupported`
   (`pipeline/mod.rs:318`) with an actionable error. Each plan drives its own loop and
   needs its decoder(s) routed through `PipelineDecoderComponent`. *Test:* per-plan
   native parity harnesses (`tts_nested_*`, `iterative_pipeline_e2e`).

6. **R6 — (optional purity) Backend-neutral host tensor in the shared pool. Size L.**
   Replace `PipelineTensors = HashMap<String, Value>` with a neutral host tensor so the
   native path drops the per-step `Value` conversion. Large, cross-cutting, and **not a
   blocker** — do last, only if profiling shows the per-step conversion matters.

## First native pipeline model to light up, and how to verify it

**Already lit:** Qwen3.5-0.8B hybrid (same class as 35B-A3B) via
`qwen35_0_8b_hybrid_native_cuda_e2e.rs`. Verification method (the fairness-first pattern
Justin requires, already in place): **byte/token-exact differential vs an ORT-backend
decode of the SAME artifact.** ORT drives the embedding front-end for both runs so the
comparison isolates the decoder EP, and greedy decode must match the trusted ORT
reference token-for-token; the harness auto-activates and fails the instant native
diverges.

**Next real number — the 35B-A3B itself:** run the same differential on the full model
(8×H200 available). Plan:
- Front-end (embedding/router) on ORT for both arms; decoder on native CUDA for the test
  arm, ORT for the reference arm — isolate the decoder EP exactly as the 0.8B harness.
- Greedy, fixed prompt, assert token-for-token equality against the ORT reference;
  additionally spot-check top-1 logit agreement at a few positions to localize any
  divergence to a layer class (LinearAttention vs GQA vs MoE router).
- If any divergence appears, bisect by layer type using the existing per-op locks
  (LinearAttention #484, CausalConvWithState #480, GatherBlockQuantized gate #525, fp16
  TopK #612) rather than trusting an end-to-end pass alone.
- Only after token parity holds, report a decode throughput number.

## Honest effort estimate

- Flat-autoregressive native text decode (the 35B-A3B path): **DONE + locked.** Remaining
  work to *report a 35B number* is a test/bench harness on the real artifact (S–M), not
  engine changes.
- R2–R5 are independent feature increments (M/M/L/L), each gated behind its own fixture
  and parity lock; none is required for the text-only 35B-A3B native decode number.
- No monolithic mega-PR is warranted or attempted.
