# Inc3b — generic routed ports on the native CUDA decoder

Lift the remaining CUDA refusal (deferred in Inc3a): the native CUDA decoder
currently accepts only the `inputs_embeds` sequence source and refuses **generic
`Routed`** step-input ports. Inc3b makes arbitrary declared non-KV routed ports
bind on-device per step, so cross-component handoffs (e.g. a routed hidden/state
edge, and eventually `static_cross_kv`) work on the CUDA EP — the same DRY
mechanism Inc3a used for `inputs_embeds`, generalized.

**Scope IN:** generic routed *text* ports (arbitrary declared decoder inputs fed
by a pipeline dataflow edge), bound on-device per step, native-CUDA token parity.
**Scope OUT (defer):** vision cross-KV (needs the vision Attention float-mask
fixes — a separate #384 blocker); `static_cross_kv` *upload-once* optimization
(re-uploading a routed port each step is correct, just not yet optimal — Inc3c).

## What "generic routed ports" needs beyond Inc3a's inputs_embeds path

Inc3a's `decode_cuda_inputs_embeds` builds an eager `owned: Vec<(String,
Tensor)>` of exactly `[inputs_embeds, position_ids]` and refuses any other
routed port, then calls `run_cuda_eager_rows_owned(owned, ...)`, which uploads
those owned host inputs each step and binds them alongside the **persistent**
device mask + KV bindings. The persistent set (`bindings[..base_binding_count]`
= mask(0) + KV + fixed-state + aux) never round-trips.

The CPU path already does the general thing: `prepare_cpu_step_inputs`
(cpu.rs:170) iterates **all** declared `self.step_inputs` and builds `owned` for
every source — `TokenIds`/`AttentionMask`/`PositionIds` generated, `InputsEmbeds`
/`Routed` pulled from the supplied `step_inputs` by exact graph-port name,
erroring on any unmapped or unknown port.

So the CUDA generalization is: build the eager `owned` set the **same generic
way** as CPU, with one CUDA-specific exclusion — **`AttentionMask` is a
persistent device binding on CUDA** (bindings[0], filled by `extend_mask`), so it
must NOT also be pushed as an owned input. Everything else (token ids or embeds
sequence, position ids, and every `Routed` port) is an owned per-step upload.

### Precise changes
1. **cuda.rs — generalize the eager owned build.** Replace the embeds-only
   `decode_cuda_inputs_embeds` with a generic `decode_cuda_eager_step_inputs`
   that iterates `self.step_inputs` (mirroring `prepare_cpu_step_inputs`),
   generating `TokenIds`/`PositionIds`, pulling `InputsEmbeds`/`Routed` from the
   supplied tensors, and **skipping `AttentionMask`** (persistent). Reuses
   `run_cuda_eager_rows_owned` unchanged. The inputs_embeds case becomes one
   instance of this general path — no forked code.
2. **cuda.rs — routing decision in `decode_cuda`.** Take the eager generic path
   when the decoder declares **any** `InputsEmbeds` *or* `Routed` step input.
   The pure token-id decode (no routed ports) keeps its existing captured
   single-token fast path + eager prefill **byte-identical** — routed ports force
   eager (per-step upload) because the captured graph writes only the fixed
   token/mask/KV bindings.
3. **load.rs — lift the refusal.** Remove the `NativeStepInputSource::Routed`
   bail in the CUDA construction block (load.rs:439); routed ports are now bound
   generically. Keep the `inputs_embeds` metadata resolution from Inc3a. (The
   sequence binding stays token-id vs embeds as in Inc3a; routed ports are owned
   inputs, not the sequence binding, so no new persistent binding is needed.)

### dtype/shape generality
Owned eager inputs are uploaded from host `Tensor`s (converted from the pool's
`ort::Value`) and bound by name via `run_with_device_bindings`; any dtype/shape a
routed port declares is honored — no per-port persistent device binding is
allocated, so there is no `DecodeCudaState::new` binding-table change (unlike the
Inc3a sequence binding). Device placement: the upload targets the session's CUDA
device (device 4); KV/mask stay device-resident. Per step, only the small routed
tensors cross host→device — same guarantee as inputs_embeds.

### decoder_in_edges / static_cross_kv mapping
Pipeline `decoder_in_edges` map each upstream component output to a decoder input
port by name; those non-generated, non-KV ports resolve to `NativeStepInputSource
::Routed` and are supplied in `step_inputs` each step. `static_cross_kv` is a
routed port that happens to be constant across steps; Inc3b binds it correctly
(re-uploaded per step). The upload-once optimization is deferred (Inc3c) and is a
perf change, not a correctness one.

## Split
Single reviewable slice (no further split needed): generalize the eager owned
build + lift the load refusal + a routed-port fixture + native-CUDA parity test.
No `DecodeCudaState` binding-table surgery (routed ports are owned inputs), which
keeps the risk low.

## Validation bar
New fixture `tiny-gemma4-vlm-cuda-routed`: the embedding every_step component
emits a **second** output `router_state` routed to a decoder `router_state`
input (a real cross-component handoff, source `Routed`, not `inputs_embeds`),
consumed by the decoder via a zero contribution so tokens stay `[0,5,6,7]`. Test:
native decoder CPU vs CUDA (device 4) through the pipeline, both `[0,5,6,7]`,
proving the routed port binds on-device and the KV stays resident. Gate behind
the existing `ONNX_GENAI_PIPELINE_NATIVE_DECODER` +
`ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` flags. ORT goldens unchanged;
clippy ×4 cfg-correct.
