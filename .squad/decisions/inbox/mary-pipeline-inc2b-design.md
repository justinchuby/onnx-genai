# Inc2b design — the native device-KV decoder in the pipeline (GAP 3, text decoder)

Author: Mary (native-pipeline). Refs #384. Stacks on #478 (Inc2a: the stateful
`PipelineDecoderComponent` trait + `OrtPipelineDecoder`).

## Goal

A `NativePipelineDecoder` that implements the Inc2a `PipelineDecoderComponent`
trait by wrapping `NativeDecodeSession`, keeping the native decoder's KV cache
**live across `.step()` calls** (never re-staging KV through the host pipeline
pool), and proving native-decoder-in-pipeline **token parity** vs the ORT decoder
on the `tiny-gemma4-vlm` fixture (embedding every_step -> `inputs_embeds` decoder).

## What already exists (so this increment is small)

Grepping `NativeDecodeSession`:

- **GAP 2 is already done.** `NativeDecodeSession::decode_with_step_inputs(token_ids,
  past_len, step_inputs: &[(String, Tensor)])` already accepts routed / `inputs_embeds`
  per-step inputs (`native_decode/mod.rs:260`, `native_decode/cpu.rs:170`
  `prepare_cpu_step_inputs`). `NativeStepInputSource::InputsEmbeds` and `Routed`
  are resolved from the supplied map by exact graph-port name; token ids /
  attention mask / position ids are generated internally.
- **KV is owned by the session across steps.** `self.past` (host) or the CPU
  in-place `DecodeCpuKvState` / CUDA `DecodeCudaState` bindings hold KV; the
  session tracks `current_len` and appends each step. `decode_with_step_inputs`
  asserts `past_len == self.current_len`, i.e. it is inherently stateful — exactly
  the Inc2a "stateful seam" contract, not the Inc1 stateless round-trip.
- **The load path builds the right bindings.** `NativeDecodeSession::load` reads
  `io.sequence_source` / `io.inputs_embeds_input` and constructs the `InputsEmbeds`
  step-input binding + the `present_to_past` KV pairs (`native_decode/load.rs:244`,
  `:363`). The `tiny-gemma4-vlm` decoder declares exactly this
  (`sequence_source: inputs_embeds`, `inputs_embeds_input: inputs_embeds`, one KV
  pair), so it loads with no new metadata.
- **The value-type seam is solved (Inc1).** `value_to_component_tensor` (ort::Value
  -> ComponentTensor) already exists; `native_component::to_native_tensor`
  (ComponentTensor -> onnx_runtime_session::Tensor) is the inverse used by the
  every_step native path. Chaining the two converts a pool `ort::Value` extra to a
  native `Tensor`.

So GAP 3 (this increment) is only the **adapter + wiring + parity proof**; the
native decode kernel and step-input plumbing are already in tree.

## The per-step seam (quantified)

Each step the every_step embedding component publishes `inputs_embeds` into the
host pool as an `ort::Value` (routed edge `embedding.inputs_embeds ->
decoder.inputs_embeds`). The native decoder receives it as its one per-step input:
that is **one token's embedding**, shape `[1, 1, hidden]` on decode steps
(`[1, prompt_len, hidden]` on prefill) — `hidden * 4` bytes, negligible. The
**KV cache stays inside the native session** and is never uploaded from or
downloaded to the pipeline pool — the expensive part never round-trips. This is
the whole point of a stateful decoder seam vs the Inc1 stateless one.

`decoder_in_edges` maps cleanly: each `(source_endpoint, decoder_port)` becomes a
`(decoder_port, native Tensor)` step input keyed by the exact graph-port name that
`prepare_cpu_step_inputs` resolves. `static_cross_kv` is **empty** for this text
decoder; cross-attention KV (encoder→decoder, vision) is **scoped OUT** to Inc3.

## Trait method mapping (`NativePipelineDecoder`)

- `step(input_tokens, past_len, extras)` — convert each `extras` `ort::Value` to a
  native `Tensor` (value -> ComponentTensor -> Tensor), call
  `decode_with_step_inputs(input_tokens, past_len, &step_inputs)`, keep the last
  logits row.
- `next_token_logits()` — return the retained last row (native decode already
  returns the final-position logits row).
- `use_kv()` — `true` (the native session owns KV across steps).
- `retained_kv_len(past_len)` / `sliding_window()` / `sink_tokens()` — `past_len` /
  `None` / `0`. These only feed **paged mirroring**, which this increment does not
  support (see below).
- `mirror_last_present_kv(...)` — **unsupported in Inc2b**: the native KV is
  session-resident and not exposed as host present tensors to the paged cache.
  Returns a clear error. Never called, because native selection disables paging.

## KV lifetime, rewind, device

The wrapper **owns** its `NativeDecodeSession` for the whole generation: created
once in `flat_autoregressive`, it lives across every `.step()` and is dropped at
request end — KV resident the entire time. `NativeDecodeSession::reset()`/`rewind()`
exist for loop restart; Inc2b loads a fresh session per request and starts from
`past_len == 0`, so no cross-request carry-over is attempted. Device: **CPU** for
the deterministic parity fixture (CUDA target + routed host step inputs is
explicitly refused today — `native_decode/mod.rs:276` — so CPU is the correct and
honest target for this slice; GPU device 4 remains available but is not required).

## Paging / cross-request reuse — deferred to Inc3

The ORT pipeline mirrors each step's present KV into a shared paged cache for
cross-request prefix reuse. The native session does not expose present KV as host
tensors, so **native selection forces the non-paged, fresh-decode path**:
`paged_enabled = false` and `reused = 0` when the native decoder is selected, which
makes `paged_session = None` (hence `paged_mirror = None`) through the existing
match. This changes **cross-request KV reuse only**, never the tokens produced
within a generation, so ORT (paged) vs native (non-paged) token IDs must still be
identical. Native present-KV exposure + paged mirroring is Inc3, alongside vision
cross-KV.

## Selection flag

`ONNX_GENAI_PIPELINE_NATIVE_DECODER` — when it names the decoder component (or is a
truthy value), the flat AR pipeline builds `NativePipelineDecoder` instead of
`OrtPipelineDecoder`. Unset/empty keeps the ORT decoder (default, unchanged),
mirroring Inc1's `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS`. Requesting it in a
build without `--features native-backend` is a clear error.

## Split

- **Inc2b-i (already in tree):** `NativeDecodeSession::decode_with_step_inputs`
  accepting `inputs_embeds`/routed inputs + session-resident KV. Proven in
  isolation by the existing `native_decode` tests. No new work.
- **Inc2b-ii (THIS PR):** the `NativePipelineDecoder` adapter + flat-AR wiring +
  env selection + **native-decoder-in-pipeline token parity** on `tiny-gemma4-vlm`
  (native tokens == ORT tokens == `[0, 5, 6, 7]`). This is the smallest slice that
  ends on a GREEN token-parity proof.

## Risks

- **fp divergence:** nxrt vs ORT numerics could shift an argmax. Mitigated by the
  tiny synthetic fixture with well-separated logits; parity asserts exact token IDs.
- **dtype at the seam:** `inputs_embeds` is f32 here; `to_native_tensor` handles the
  declared dtype. Unsupported dtypes surface as a clear conversion error.
- **paging off:** documented above — affects reuse, not tokens.

## Out of scope (Inc3)

Cross-attention/vision KV into the native decoder, native present-KV exposure +
paged mirroring, cross-request retained/paged reuse with the native decoder, CUDA
target with routed host step inputs.
