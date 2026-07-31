### 2026-07-30: Native multi-component pipeline decode — Inc2 design (stateful decoder seam, issue #384)
**By:** Mary
**What:** Design for driving the pipeline **decoder** backend-agnostically, and the
Inc2a/Inc2b split. Follows Inc1 (#450, merged), which routed the stateless
`every_step` components through `ComponentSession`. The decoder is the hard crux
flagged in the Inc1 plan because it is **stateful** (KV cache across steps).

#### 1. How the decoder is driven today (two distinct mechanisms)

There are **two** decode mechanisms in the engine, and they do not share a path:

**(A) Single-model `Engine` hot loop — already backend-generic and stateful.**
`DecodeState.runner: Option<DecodeRunner>` (decode/mod.rs:112) is an enum over
`StaticCache(StaticCacheDecodeSession)`, `PastPresent(DecodeSession)`, and
`Native(NativeDecodeSession)`, unified by `trait DecodeBackend` (decode/mod.rs:78)
with `decode/decode_argmax/decode_sampled(token_ids, past_len)`. `NativeDecodeSession`
owns a **device-resident KV cache**, advancing/rewinding its own cursor
(`current_len`, `rewind`). This is the stateful decoder abstraction the single-model
path already uses. **But `DecodeBackend::decode` takes only `(token_ids, past_len)`
and returns logits** — it has no channel for per-step routed/extra inputs and does
not expose present-KV tensors for paged mirroring.

**(B) Pipeline decoder — ORT-only, host-KV, supports extras + paging.**
`PipelineDecodeLoopBackend` (pipeline/paged_decode.rs) holds `decoder: &'a Session`
and `decoder_state: &'a mut DecodeState` and per step calls, in `next_logits`:
  1. `run_decode_step_with_extra(decoder, decoder_state, input_tokens, past_len,
     &extras) -> Vec<Value>` (decode/step.rs:119) — binds routed `extra_inputs`
     (every_step outputs, `inputs_embeds`, routed positions, static cross-KV),
     holds **KV as host ORT `Value`s in `decode_state.past`**, cloned each step.
  2. `mirror_present_kv_to_pages(decoder, kv_model, cache, seq, &outputs,
     retained_past_len, input_len)` (kv_bridge.rs:306) — reads `present.*` outputs
     for prefix reuse.
  3. `apply_paged_sliding_window(cache, seq, decoder_state.sliding_window(),
     decoder_state.sink_tokens())`.
  4. `extract_next_token_logits_with_io(decoder, outputs, io.logits_output)`
     (decode/logits.rs:7).
`PipelineDecodeLoopBackend` is the **only** user of these per-step calls for the flat
AR pipeline (nested_autoregressive drives its own inner loop; iterative is diffusion).

**The gap:** mechanism (B) is concrete-ORT and host-KV; mechanism (A) is
backend-generic with device-KV but cannot accept the pipeline's per-step routed
inputs nor expose KV for paging. Inc2 must give the **pipeline** decoder a stateful,
backend-agnostic seam.

#### 2. Why Inc1's stateless `ComponentSession` seam does NOT fit the decoder

`ComponentSession::run` is stateless: host `ComponentTensor` in, host
`ComponentTensor` out, no retained state. Driving the decoder through it would
(a) drop the native device-KV continuity `NativeDecodeSession` maintains, and
(b) re-stage the whole KV cache across the host seam every step — destroying decode
throughput (KV is the large, per-layer, per-token-growing tensor). So the decoder
needs a **stateful** seam that keeps KV inside the backend across steps.

#### 3. Proposed stateful decoder abstraction

Introduce (pipeline-scoped) `trait PipelineDecoderComponent`:

```
trait PipelineDecoderComponent {
    // Run one decoder step over input_tokens at past_len, binding routed extras
    // (every_step outputs, inputs_embeds, routed positions, static cross-KV),
    // advancing internal KV. The step's outputs are retained internally.
    fn step(&mut self, input_tokens: &[TokenId], past_len: usize,
            extras: &[(String, Value)]) -> Result<()>;
    // Next-token logits from the most recent step.
    fn next_token_logits(&self) -> Result<Vec<f32>>;
    // Mirror the most recent step's present KV into the paged cache.
    fn mirror_last_present_kv(&self, kv_model: &KvModelInfo, cache: &mut PagedKvCache,
        seq: SequenceId, retained_past_len: usize, input_len: usize) -> Result<()>;
    // KV/window queries the loop needs (were direct decoder_state reads).
    fn use_kv(&self) -> bool;
    fn retained_kv_len(&self, past_len: usize) -> usize;
    fn sliding_window(&self) -> Option<usize>;
    fn sink_tokens(&self) -> usize;
}
```

Key design choice: **the decoder impl owns its per-step outputs internally**
(`last_outputs`), so the pipeline loop never touches ORT `Value`/nxrt tensors — it
calls `step()` then `next_token_logits()`/`mirror_last_present_kv()`. This keeps the
loop truly backend-agnostic (the DRY principle from Inc1, but stateful).

`PipelineDecodeLoopBackend` replaces its two fields `decoder: &Session` +
`decoder_state: &mut DecodeState` with one `decoder: Box<dyn PipelineDecoderComponent
+ 'a>`. `next_logits` restructures to call the trait; the paged-mirror and
sliding-window bookkeeping stay in the loop but source their queries from the trait.

**ORT impl `OrtPipelineDecoder<'a>`** wraps `session: &'a Session`, `state: &'a mut
DecodeState`, `last_outputs: Option<Vec<Value>>` and forwards to the existing
`run_decode_step_with_extra` / `mirror_present_kv_to_pages` /
`extract_next_token_logits_with_io` — **behaviour-identical to today**.

**Native impl (Inc2b) `NativePipelineDecoder`** wraps `NativeDecodeSession`, keeping
its KV **device-resident** across steps. Per-step routed inputs reach the device
each step; see §4.

#### 4. Per-step input seam for the native decoder

Each step the decoder's non-token inputs are: the `every_step` outputs (e.g.
`inputs_embeds`), cached `prompt_only` conditioning, routed positions, and the
static cross-attention KV. All but the static cross-KV are **one token's worth per
decode step** (embedding/hidden of shape `[1,1,hidden]`) — uploading that to the
device each step is cheap; the **KV cache stays device-resident** (the expensive
invariant). The static cross-KV is resolved once and is invariant across the loop,
so it is uploaded once, not per step. `NativeDecodeSession::step` today takes only
`(input_ids, attention_mask, position_ids)`; Inc2b must extend the native step to
accept **routed named tensors / an `inputs_embeds` sequence** (embeds-driven
decoders have no token input) — that native-side extension is the bulk of Inc2b and
is non-trivial. It also must expose present-KV for paged mirroring, or Inc2b ships
first **without** paged-reuse for the native decoder (paging is an optimization, not
a correctness requirement) and adds device-KV mirroring in Inc3.

#### 5. KV ownership, lifetime, rewind, device placement

- The ORT impl borrows `&'a mut DecodeState` (host KV lives in `decode_state.past`),
  exactly as today. Lifetime: the backend is dropped before the paged sequence is
  retired (the `drop(backend)` added in Inc1), releasing the `DecodeState` borrow.
- The native impl **owns** its `NativeDecodeSession` (device KV). Rewind/reset on a
  restarted loop maps to `NativeDecodeSession::rewind`/cursor reset. A pipeline that
  restarts a sequence resets the native cursor rather than replaying host KV.
- Device placement: routed per-step inputs are staged host→device inside the native
  step; the pipeline pool stays host-resident (`ort::Value`), so the seam is a small
  per-step upload, decoupled from the resident KV.

#### 6. Increment split (honest)

- **Inc2a (this task, pure refactor):** introduce `PipelineDecoderComponent` + the
  ORT impl `OrtPipelineDecoder`; make `PipelineDecodeLoopBackend` drive the decoder
  through the trait. **No native decoder.** Proven **behaviour-identical** by the
  existing pipeline e2e suite (token output unchanged) plus an explicit
  ORT-decoder-through-trait equivalence assertion. This de-risks Inc2b by
  establishing the seam without touching native semantics.
- **Inc2b (next):** native impl wrapping `NativeDecodeSession` with device-resident
  KV; extend the native step to accept routed/`inputs_embeds` per-step inputs; prove
  native-decoder-in-pipeline token parity vs ORT on a small CPU model. Paged reuse
  for the native decoder may defer to Inc3.
- **Inc3:** device-KV paged mirroring + cross-component/vision handoff; full
  35B-A3B embedding+decoder+vision on native.

**Verdict:** Inc2a is a clean, provably-behaviour-identical refactor and is the
right first slice. Inc2b's native step-input extension is substantial (native
decoders today take only token/mask/position; the pipeline needs `inputs_embeds`
and routed tensors device-side) and must be its own reviewed increment — do not
half-wire it.

**Why:** Establishing the stateful decoder seam (owning KV internally, loop stays
backend-agnostic) up front lets the native decoder land later without touching the
loop, mirrors the Inc1 DRY outcome, and prevents a stateless-seam mistake that would
have destroyed native decode throughput by re-staging KV each step.
