### 2026-07-31: Pipeline backend-neutral component ownership — assessment, and locking the full-native keystone

**By:** Mary

**What (headline):** I was tasked to introduce a backend-neutral ownership seam so
`PipelineDecodeLoopBackend` can drive EITHER ORT or native component sessions per
step, then route the existing ORT path through it byte-identically as increment-1.
On inspection, **that seam already fully exists on `origin/main`** — it landed
across the inc1→inc3c chain (#450, #478, #479, #485, #487, #533) and was hardened
by #543. Re-introducing it would duplicate merged work. So increment-1 here is the
piece the chain had NOT yet locked: a parity test proving the seam's **keystone**
end-state — *every declared component running natively at once* (native every_step
embedding + native device-KV decoder in the same decode loop) — matches the ORT
baseline token-for-token. No production code changed; the ORT path is byte-identical
by construction.

**Assessment — where ORT ownership lives today (ownership map):**

`PipelineDecodeLoopBackend` (`crates/onnx-genai-engine/src/pipeline/paged_decode.rs:81`)
no longer hard-owns ORT `Session`/decode state. Its backend-facing fields are
already neutral:
- `decoder: Box<dyn PipelineDecoderComponent + 'a>` — stateful decoder seam
  (`pipeline/decoder_component.rs:31`). Impls: `OrtPipelineDecoder` (borrows
  `&Session` + `&mut DecodeState`) and `NativePipelineDecoder` (owns a
  `NativeDecodeSession`, KV session-resident). Selected in
  `flat_autoregressive.rs:198` via `native_decoder_selected` /
  `build_native_pipeline_decoder` (`pipeline/mod.rs:400,472`).
- `step_components: Vec<(StepComponentBinding, Box<dyn ComponentSession + 'a>)>` —
  stateless every_step seam (`onnx-genai-metadata/src/component.rs:260`). Impls:
  `OrtComponentSessionRef` and native `NativeComponentSession`, selected in
  `flat_autoregressive.rs:169` via `build_step_component_session` (`mod.rs:422`).

What REMAINS ORT-typed is only the **host tensor currency**, not ownership:
- `pool: &mut PipelineTensors`, `static_cross_kv: Vec<(String, Arc<Value>)>`, and the
  per-step `extras: Vec<(String, Value)>` handed to `decoder.step` are `ort::Value`.
  The native decoder converts each extra `Value → ComponentTensor → native Tensor`
  per step (`decoder_component.rs:218`), so `ort::Value` is a neutral carrier here,
  not a backend coupling. Making the pool currency a first-class neutral tensor type
  would be a whole-engine, high-blast-radius change and is explicitly NOT this work.

Per-step invocation needs (already satisfied through the seam): embedding/every_step
step over the running token → outputs into pool; decoder step over `input_tokens` at
absolute `past_len` with routed `extras` (inputs_embeds / routed ports / positions /
static cross-KV); KV/state handoff kept internal to the decoder; logits out via
`next_token_logits()`; optional paged present-KV mirror via `mirror_last_present_kv()`.

**Design (the seam, as it stands and as I affirm it):** two traits, no enum —
`PipelineDecoderComponent` (stateful: `step` / `next_token_logits` /
`mirror_last_present_kv` / `use_kv` / `retained_kv_len` / `sliding_window` /
`sink_tokens`) for the decoder, and `ComponentSession` (stateless `run`) for
every_step components. Both are DRY and model-agnostic: native decoder selection and
native step-component selection are env-driven injection points, not per-model code.
This is the right shape; I did not add a parallel abstraction.

**What increment-1 landed here (test-only, zero behavior change):**
- New CPU parity test `native_full_pipeline_parity` (registered in
  `crates/onnx-genai-engine/Cargo.toml`, `required-features = ["native-backend"]`):
  drives the `tiny-gemma4-vlm` composite with BOTH
  `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS=embedding` and
  `ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder`, asserting the fully-native run equals
  the ORT baseline `[0, 5, 6, 7]`. The prior increments proved each slice in
  isolation (inc1: native embedding + ORT decoder; inc2b: ORT embedding + native
  decoder); nothing locked the two natively at once — the exact composite shape the
  35B-A3B package decodes through. This closes that gap.

**ORT byte-identical proof:** no production source touched (only a new test file +
its Cargo `[[test]]` entry + this note), so the ORT decode path is byte-identical to
`origin/main` by construction. Empirically: ORT baseline `[0,5,6,7]` and fully-native
`[0,5,6,7]` match; `native_step_component_parity`, `native_pipeline_decoder_parity`,
and the 343 engine lib unit tests all pass; `cargo fmt --all --check` clean.

**Deferred to the next increment (native wiring completion):** the one genuine
remaining hard limitation the code itself flags (`decoder_component.rs:244-260`):
`NativePipelineDecoder::mirror_last_present_kv` bails — the native decoder keeps KV
session-resident and does not expose host present tensors, so native selection runs
the non-paged, fresh-decode path with no cross-request KV reuse. Wiring native
present-KV exposure + paged mirroring (so the native decoder participates in the
prefix cache) is the next slice; it needs device present tensors surfaced and is
higher blast radius, so it is intentionally NOT bundled here. Its
`retained_kv_len`/`sliding_window`/`sink_tokens` are currently constant stubs that
only feed paged mirroring, so they come along with that increment.

**Why:** The keystone for large multi-component native decode (incl. 35B-A3B) is
"drive all native component sessions through one loop." The seam that enables it is
already merged; the missing safety rail was a regression lock proving the full-native
composite is token-identical to ORT. Landing that lock now — with zero behavior
change — protects the seam for the 3-component package while the remaining
native-paged-reuse wiring is designed separately.
