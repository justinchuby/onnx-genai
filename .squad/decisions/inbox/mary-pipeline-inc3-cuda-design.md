# Inc3 design — CUDA-target native decoder with `inputs_embeds` / routed step inputs

Author: Mary (native-pipeline). Refs #384. Stacks on #479 (Inc2b native device-KV
decoder). Scope: **text decoder on the CUDA EP only** — paged cross-request reuse
and vision cross-KV stay deferred (Inc4+).

## Why this is the critical-path unblock

Inc2b proved the native device-KV decoder end-to-end **on CPU**. The whole point of
the native path is to run the 35B-A3B decoder **on-GPU** with a device-resident KV
cache. Today the native decoder *refuses* a CUDA target the moment the decoder is
driven by `inputs_embeds` (a fused VLM decoder) or any routed host step input — so
the GPU native-decode path for a multi-component / VLM model is blocked at load.

## The refusal — exact sites and root cause

Two guards, both hit for a CUDA + `inputs_embeds` decoder:

1. **Load-time (the load-bearing one):**
   `native_decode/load.rs:434-446` — when `device == Cuda` and
   (`sequence_source != TokenIds` **or** any `step_input.source` is
   `InputsEmbeds | Routed`), it `bail!`s before constructing `DecodeCudaState`.
2. **Runtime:** `native_decode/mod.rs:275-280` — `decode_with_step_inputs` bails if
   `self.cuda.is_some()` and `step_inputs` is non-empty.

**Root cause (verified by reading `cuda.rs`): an unimplemented on-device binding
path, NOT a correctness/architecture barrier.** `DecodeCudaState`
(`cuda.rs:731-1046`) hardwires the *sequence* input as an `Int64 [1,1]` `input_ids`
device binding (`input_ids_binding`, `cuda.rs:944-951`) and writes it each step via
`write_decode_inputs(token_id, position)` (`cuda.rs:1118-1125`). There is:
  - no allocation of a **float `inputs_embeds [1, 1, hidden]`** device binding, and
  - no path to **upload a routed host tensor** into a device binding each step.

`DecodeCudaIo` (`cuda.rs:174-179`) only carries `input_ids / attention_mask /
position_ids / logits`. Everything else (KV device-residency, mask growth,
CUDA-graph capture, logits readback) is **already generic** over the bindings vector
and does not care whether the sequence input is a token id or an embedding — only the
*sequence-input binding + its per-step write* are token-specific.

### What must change to accept `inputs_embeds` on CUDA
- `DecodeCudaIo`: carry the sequence source (token vs `inputs_embeds`) + the embed
  port name, dtype and hidden width from `ModelIoSpec`.
- `DecodeCudaState::new`: when the source is `InputsEmbeds`, allocate the sequence
  binding as the graph's `inputs_embeds` dtype/shape `[1, 1, hidden]` **instead of**
  the `Int64 [1,1]` token binding. KV / mask / position / logits bindings are
  unchanged.
- A per-step **`write_decode_embeds(&host_embed_bytes)`** that copies the one-token
  embedding into the (fixed-address, stable) device binding — exactly parallel to
  `write_decode_inputs` writing a token id. Because the binding *address* is stable
  and only the *contents* change per step, this stays **CUDA-graph-capture eligible**
  (same property as the token-id write).
- `decode_cuda` / `run_cuda_eager_rows`: resolve the sequence binding by source; when
  `InputsEmbeds`, pull the embedding tensor from `step_inputs` (validated exactly like
  the CPU path in `cpu.rs:205-213`) and upload it, rather than building a token tensor.
- Lift the two refusals **for the `inputs_embeds` case**; keep bailing for
  *arbitrary* `Routed` ports (defer generic routing to a later slice).

### The device-residency guarantee (the whole point)
Per step, exactly **one token's embedding** crosses host→device: `[1, 1, hidden]`
(e.g. `hidden * 4` bytes for f32, `* 2` for f16) — negligible. The **KV cache never
leaves the device**: it lives in the persistent `kv_binding_range` device bindings
and grows in place across steps (`extend_mask` + `set_logical_len`), identical to the
existing token-id CUDA path. This is the same guarantee Inc2b gave on CPU, now on the
CUDA EP (device 4).

## Honest split

- **Inc3a (this PR — the smallest PROVEN slice):** implement the CUDA `inputs_embeds`
  sequence-input device binding + per-step upload; lift the refusals for the
  `inputs_embeds` case; keep KV device-resident. **Prove native-CUDA decode token IDs
  == CPU-native decode token IDs** on a tiny CUDA-capable `inputs_embeds` fixture
  (`inputs_embeds` + `attention_mask` + `position_ids` + one KV pair, closed-form
  deterministic head) driven directly through `decode_with_step_inputs` on both
  devices. This isolates the new binding path from pipeline wiring.
- **Inc3b (next):** wire `NativePipelineDecoder` to load the decoder on the CUDA
  device (via the existing `ONNX_GENAI_PIPELINE_NATIVE_DECODER` flag + a device
  selector) and prove **native-CUDA-decoder-in-pipeline** token parity vs the ORT
  pipeline on GPU. Then generic `Routed` ports.
- **Deferred (Inc4+):** native present-KV exposure for paged cross-request reuse,
  vision cross-KV, non-`inputs_embeds` arbitrary routed device bindings.

## Risks
- **KV op support on CUDA persistent bindings.** The fixture's KV must grow correctly
  under the CUDA persistent-device-KV contract. Mitigation: model the fixture KV on a
  pattern the CUDA path already handles; if the toy `Concat` KV does not bind cleanly
  on CUDA, the fixture uses the graph shape the CUDA persistent-KV path expects, or the
  slice is reported as blocked with the exact failure rather than shipped half-wired.
- **dtype at the seam.** The every_step embedding is produced on host as the pool
  dtype (f32/f16); the device binding dtype must match the decoder's declared
  `inputs_embeds` dtype. Reuse the Inc1 `coerce_value_to_dtype` seam; upload bytes
  only after coercion.
- **Graph capture.** The per-step embedding write keeps a stable device address, so
  capture eligibility is unchanged; verified by the parity run (capture on vs eager).
- **CUDA + routed generic ports** remain refused — explicitly out of Inc3a.

## Verification plan
- New parity test (feature-gated `cuda,native-backend`, skips when no GPU):
  native-CUDA `decode_with_step_inputs` tokens **==** CPU-native tokens on the tiny
  fixture, ≥2 decode steps (real growing KV).
- ORT-path goldens unchanged; `cargo test -p onnx-genai-engine` default +
  `native-backend`; `fmt --check`; clippy default / native-backend / cuda /
  cuda,native-backend (cfg-correct: all CUDA-embeds code behind `cfg(feature =
  "cuda")`, all native behind `cfg(feature = "native-backend")`).
