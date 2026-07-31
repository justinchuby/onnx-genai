### 2026-07-30: Native multi-component pipeline decode — refactor plan (issue #384)
**By:** Mary
**What:** Scoped plan to make `PipelineDecodeLoopBackend` drive **native** component
sessions (nxrt custom-EP), unblocking native decode of pipelined multi-component
models (Qwen3.6-35B-A3B = embedding + decoder + vision, and any multimodal pipeline).
This is the GAP 3 work referenced in `mary-35b-a3b-blocker.md`.

#### 1. Where the pipeline loop hardcodes the concrete ORT `Session` / ORT `Value`

Grepped `&'a Session`, `.run(`, `Session,`, `Value` across the pipeline module.
The decode loop that must become backend-neutral is **`flat_autoregressive.rs` +
`paged_decode.rs`** (only these two construct/use `PipelineDecodeLoopBackend`;
`nested_autoregressive.rs` and `iterative.rs` drive their own loops and are out of
scope for the AR-decoder path).

Concrete-ORT couplings in `paged_decode.rs::PipelineDecodeLoopBackend`:

| Field / call | Coupling |
|---|---|
| `decoder: &'a Session` | concrete ORT decoder session (GAP: Inc2) |
| `step_components: Vec<(StepComponentBinding, &'a Session)>` | concrete ORT **every_step** sessions (GAP: **Inc1**) |
| `pool: &'a mut PipelineTensors` = `HashMap<String, ort::Value>` | pool holds ORT `Value` (the value-type seam) |
| `static_cross_kv: Vec<(String, Arc<Value>)>` | ORT `Value` cross-attn KV (Inc3) |
| `run_step_components`: builds `Vec<(String, Value)>`, calls `session.run(&refs)`, inserts `Value` | ORT value + ORT session run (Inc1) |
| `decoder_extras`: `clone_value`, `Value::alias_with_shape` | ORT `Value` decoder binding (Inc2) |
| `next_logits`: `run_decode_step_with_extra(self.decoder, ...)`, `mirror_present_kv_to_pages`, `extract_next_token_logits_with_io` | ORT decoder step + ORT KV mirror (Inc2/Inc3) |

`flat_autoregressive.rs` pairs each binding with `self.models.session(name) -> &Session`
(lines ~152–169) and constructs the backend (line 187). `pipeline/mod.rs` imports the
concrete `Session, Value, DataType` from `onnx_genai_ort` and defines
`PipelineTensors = HashMap<String, Value>` (mod.rs:31–51).

#### 2. Trait boundary — THE VALUE-TYPE SEAM (verdict)

**Does the ORT `Session` implement `ComponentSession`?** No — but an adapter already
exists: `onnx_genai_ort::OrtComponentSession` wraps an **owned** `Session` and
implements `ComponentSession` (ort/src/component.rs:130). `NativeComponentSession`
implements the same trait (engine/src/native_component.rs:164). Both are object-safe
(`Box<dyn ComponentSession>`).

**Do `ComponentSession::run` inputs/outputs use a backend-neutral `Value` or ORT
`Value`?** **VERDICT: the seam is a backend-neutral, host-resident `ComponentTensor`
(raw little-endian element bytes + dtype + static shape), NOT ORT `Value` and NOT an
nxrt tensor.** `ComponentSession::run(&mut self, &[(&str, &ComponentTensor)]) ->
Vec<(String, ComponentTensor)>` (metadata/src/component.rs:283). Each backend adapter
translates at its own boundary: ORT via `Value::to_raw_bytes`/`from_raw_bytes`
(ort/src/component.rs `to_value`/`from_value`); native via `Tensor::from_raw`/`as_bytes`
(native_component.rs `to_native_tensor`/`from_native_tensor`).

Consequence: the pipeline **pool holds ORT `Value`**, but the trait speaks
`ComponentTensor`. So routing step components through `dyn ComponentSession` requires a
**pool-`Value` ⇄ `ComponentTensor` conversion at the loop boundary** — this host
round-trip **is the crux of the work**. `DataType ⇄ ComponentDataType` `From` impls
already exist in `onnx_genai_ort`. The conversion is a host copy (`numel * dtype.size`);
for the decoder (Inc2) this is the KV-cache-sized cost and must eventually be avoided by
keeping tensors backend-native across the seam, but for small every_step embedding
outputs it is negligible (same order as the existing per-step `clone_value` the decoder
already pays).

#### 3. Increment breakdown

- **Inc1 (THIS task):** Route the `every_step` (step) components through
  `dyn ComponentSession`. Change `step_components` to
  `Vec<(StepComponentBinding, Box<dyn ComponentSession + 'a>)>`. Rewrite
  `run_step_components` to convert pool `Value → ComponentTensor`, call the trait, and
  convert `ComponentTensor → Value` back into the pool. ORT default path wraps
  `&Session` in a **borrowing** ORT adapter (`OrtComponentSessionRef`), behaviour
  unchanged. Native path loads the component via `NativeComponentSession`. **Prove one
  NATIVE every_step component (the gemma4-vlm `embedding`) runs inside the loop with
  ORT-vs-native token parity** while the decoder stays ORT (hybrid). Deliverable =
  solve the value seam for step components + wire embedding + parity test.
- **Inc2:** The decoder itself (`decoder: &Session`, `run_decode_step_with_extra`,
  `decoder_extras`, logits extraction) becomes trait-driven. This forces the KV-cache
  ownership question (see risks) and the decoder-sized value-seam copy — the real perf
  seam. Reconcile with `NativeDecodeSession` (the single-graph native decode path).
- **Inc3:** Cross-component value handoff — `static_cross_kv` (encoder cross-attn KV),
  device placement/transfer between components, and the vision `prompt_only` stage; the
  full 35B-A3B embedding+decoder+vision chain end-to-end on native.

#### 4. Risks

- **Value-seam copies:** every seam crossing is a host copy. Fine for embeddings (Inc1),
  costly for the decoder KV (Inc2) — Inc2 should keep tensors backend-native across the
  seam or add a zero-copy fast path behind the trait rather than always round-tripping
  through host bytes.
- **KV-cache ownership (Inc2):** the single-graph native path is `NativeDecodeSession`,
  which owns its own persistent KV state. A pipeline-driven decoder needs the loop
  (`DecodeState`, paged mirror) to own/advance KV. Reconciling these two ownership
  models is the central Inc2 design decision.
- **Device placement:** native EP tensors may be device-resident; the `ComponentTensor`
  seam is host-only (`as_raw_bytes`/`to_raw_bytes` reject device tensors). Cross-device
  step→decoder handoff (Inc3) needs explicit host staging or a device-aware seam.
- **Feature gating (#436/#441 class):** native paths are `#[cfg(feature =
  "native-backend")]`; imports must be cfg-correct so both cuda and non-cuda,
  native and non-native builds compile without unused-import warnings.

**Why:** `PipelineDecodeLoopBackend` owning ORT `Value`/`Session` is the last blocker to
native pipelined decode. Establishing the value-type seam verdict (neutral
`ComponentTensor`, not ORT `Value`) up front prevents Inc2/Inc3 from re-litigating the
boundary, and the Inc1 every_step slice proves the seam end-to-end with token parity on a
tiny CPU fixture before the heavier decoder/KV work.
