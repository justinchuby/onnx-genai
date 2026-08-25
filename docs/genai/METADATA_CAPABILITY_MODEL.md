# Metadata capability model

Status: **normative proposal**. This document audits the current tree at
`f55520de9ec6160962a6a7532729119ff1279c1a` and proposes a simplification. It
does not change the metadata schema, parser, validator, generated schema, or
runtime.

The central rule is:

> A package describes what its authored graphs and workflow require. A runtime
> proves what a concrete backend deployment can execute. An operator chooses
> policy. Measurements justify performance claims. None of those four answers
> substitutes for another.

## 1. Terms and conceptual model

### 1.1 Four kinds of statement

| Kind | Meaning | Authority | Examples |
| --- | --- | --- | --- |
| **Derived structural fact** | A fact recoverable from authored graphs, workflow dataflow, tensor contracts, operator attributes, initializers, or state contracts. | Package artifacts plus deterministic derivation. | Decoder ports, position-input presence and rank, append versus indexed-scatter cache updates, graph KV dtype, image/video/audio workflow shape, control flow. |
| **Authored model requirement** | A semantic fact required for correct execution but not safely inferable from tensor shape or graph topology. | Typed package contract. | Aliasing legality, prefix-reuse legality, rollback bounds, component grouping equivalence, effect retry/speculation semantics, equivalence class, session mutation semantics. |
| **Deployment policy** | A choice about cost, risk, placement, or service behavior. | Operator/runtime configuration, never portable model metadata. | Backend/provider selection, page size, KV mirror dtype, cache/tier budgets, graph capture, fallback permission, batch size, worker count and placement. |
| **Runtime evidence** | A result observed for one artifact and one runtime environment. | Generated admission, conformance, profile, or benchmark record. | ORT session loaded with no fallback, native operator coverage passed, capture replayed, concurrent `Run` was admitted, throughput met a threshold. |

### 1.2 Capability, support, readiness, and performance

- A **requirement** is what correct execution needs.
- **Implementation capability** means code for a typed requirement exists in a
  particular runtime build.
- **Readiness** means that implementation admitted and executed a particular
  artifact under a particular backend/provider/platform configuration.
- **Functional support** means accepted outputs satisfied a correctness test.
- **Performance-proven** means a controlled measurement met a stated gate.

Schema representability is not runtime execution. Runtime execution is not
performance evidence. A model-family label is evidence of none of them.

### 1.3 Normative invariants

1. A structural fact **MUST** have one serialized authority. Any optimized view
   of it **MUST** be derived and non-serialized.
2. An authored requirement **MUST** be typed, versioned where extensible, and
   narrow enough to test. A generic optimistic boolean is not sufficient.
3. Deployment policy **MUST NOT** be interpreted as proof that a model or
   backend supports the requested behavior.
4. Backend readiness **MUST** be keyed to artifact identity and runtime
   fingerprint, and **MUST** fail closed when evidence is absent or stale.
5. A runtime **MUST NOT** upgrade represented or validated status into an
   execution claim, nor execution into a performance claim.
6. Declining an optional optimization is valid. Executing a requirement
   incorrectly, silently changing backend, or returning base-model output for
   an adapter package is not.

### 1.4 “Policy graph” means executable semantics

A workflow **policy graph** is a tensor program that computes observable model
semantics: sampling, termination, accepted-prefix selection, cache-length or
position updates, classifier-free guidance, or a diffusion solver step. It is
usually a small ONNX component and is invoked through the same typed SSA
workflow as a learned decoder. It is not deployment, scheduling, or QoS policy.

The distinction is ownership, not naming:

- `temperature`, logits filtering, RNG counter advancement, and the selected
  token change the mathematical result, so the package authors them and the
  workflow runtime executes them;
- provider choice, queue priority, batching delay, capture, cache capacity, and
  worker placement change where or when the same semantics run, so deployment
  configuration owns them.

Compact decoder example:

```yaml
# Request tensors carry semantic controls; they are not server configuration.
inputs:
  request.temperature:
    contract: {dtype: float32, rank: 1, shape: [batch]}
    role: {kind: runtime, version: "1.0", role: sampling_temperature}
    source: {kind: request}
    required: false
    default: 1.0  # Package default for the sampling equation.

# This ONNX artifact computes a token and the next explicit RNG counter.
components:
  sampler:
    implementation: {kind: onnx, artifact: policies/token_sampler.onnx}
    contract:
      id: onnx-genai.token-sampler
      version: "2"
      bindings:  # Stable meanings are independent of concrete graph port names.
        logits: logits
        temperature: temperature
        seed: seed
        counter: counter
        token: token
        next_counter: next_counter
      parameters:
        batching: per_row  # Semantic row independence, not a batching target.
        inactive_rows: preserve

# The workflow executes the equation; no host-side model-family sampler is implied.
steps:
  - kind: invoke
    component: sampler
    inputs:
      logits: decoder.last_logits
      temperature: request.temperature
      seed: request.seed
      counter: sampler.counter
    outputs:
      token: sample.token
      next_counter: sample.next_counter
```

The example is intentionally annotated rather than serializer output. Production
generators that cannot preserve YAML comments should publish an adjacent
annotated reference and validate that both documents parse to the same typed
metadata; they must not claim comments survive generation.

## 2. Current inventory

The anchors below name symbols and fields, not transient line numbers.

### 2.1 Serialized package surfaces

| Current surface | What it currently mixes | Classification |
| --- | --- | --- |
| [`InferenceMetadata::required_capabilities`](../../crates/onnx-genai-metadata/src/schema/mod.rs) | Open-ended manually authored strings used as load requirements. | Redundant when the same requirement is visible in typed metadata; otherwise too weakly typed. |
| [`WorkflowManifest::{capabilities, adapter_abis}`](../../crates/onnx-genai-metadata/src/schema/ir.rs) | A second generic capability list plus versioned adapter ABIs. | `capabilities` is redundant; versioned adapter ABI selection is an authored requirement. |
| [`ModelCapabilities`](../../crates/onnx-genai-metadata/src/schema/decoder_abi.rs) | Graph facts (`attention`, `vocab_size`, MoE, sharding), a semantic maximum, and runtime configurability. | Split by field; do not treat the object as one capability class. |
| [`RuntimeConfigurable`](../../crates/onnx-genai-metadata/src/schema/decoder_abi.rs) | `prefix_cache` and `chunked_prefill`. The former `continuous_batching` field is already rejected by the parser. | Prefix caching is an obsolete boolean; chunk size is currently consumed as a runtime preference. |
| [`QuantizationIntent`](../../crates/onnx-genai-metadata/src/schema/decoder_abi.rs) | Desired weight recipe, not the graph's actual packed representation. | Distribution/provenance hint, not execution readiness. |
| [`HardwareRequirements`](../../crates/onnx-genai-metadata/src/schema/hardware.rs) | Required and beneficial dtypes, memory estimates, and TP hints. | Graph-derived facts, distribution hints, and policy are mixed. |
| [`WorkflowSpec`, `TensorContract`, `BatchLayout`, and `ComponentBatchCapacity`](../../crates/onnx-genai-metadata/src/schema/ir.rs) | Typed dataflow, padding/packing, and artifact invocation bounds. | Structural facts plus the component-scoped authored assertion that grouped and solo execution are equivalent. The former profile-wide `batch_invariance` field is rejected. |
| [`WorkflowStateCell` and `StateGroupContract`](../../crates/onnx-genai-metadata/src/schema/ir.rs) | Lifetime, ownership, graph update discipline, semantic reuse, rollback/fork/snapshot, and checkpoint ABI. | Correctly contains both derived graph facts and authored semantic requirements; physical storage policy is explicitly excluded. |
| [`PreprocessingSpec`](../../crates/onnx-genai-metadata/src/schema/pipeline.rs) | Typed image, video, and audio transform programs. | Authored executable contract whose presence and bindings are structurally validated. |
| [`TaskProfile`](../../crates/onnx-genai-metadata/src/schema/package.rs) | Task kind, output roles, and pooling/decoding semantics. | Typed authored task contract. Recognition by validation is not proof of an engine executor or grouping safety. |
| [`SpeculativeContract`](../../crates/onnx-genai-metadata/src/schema/package.rs) | Proposer/target compatibility, shared state/weights, vocabulary, rollback, and equivalence. | Typed authored correctness requirements. Proposal width and enablement remain runtime policy. |
| [`SpeculatorConfig`](../../crates/onnx-genai-metadata/src/schema/generation.rs) | Legacy/sidecar proposer description for EAGLE, MTP, D-Flash, and shared-KV forms. | Typed artifact contract, but overlaps the workflow-native speculative contract. |
| [`AdapterServiceContract`](../../crates/onnx-genai-metadata/src/schema/ir.rs) | Target ABI, artifact loading, fallback permission, cache size/eviction, bucketing, stable buffers, and capture invalidation. | Model ABI, deployment policy, and implementation choice are mixed. |
| [`GenerationContract`](../../crates/onnx-genai-metadata/src/schema/package.rs) | Package defaults and permitted request overrides. | Authored request semantics, not backend readiness. |

The positive precedent is
[`InferenceMetadata::decoder_io`](../../crates/onnx-genai-metadata/src/schema/mod.rs):
the optimized [`DecoderAbi`](../../crates/onnx-genai-metadata/src/decoder_abi.rs)
is cached from the canonical workflow and is never serialized. The cache cannot
contradict its source.

### 2.2 Runtime and deployment surfaces

| Current surface | Current role | Classification |
| --- | --- | --- |
| [`RuntimeCapabilities::supported`](../../crates/onnx-genai-metadata/src/validation.rs) | One global list of capability strings. | Implementation catalogue without backend, artifact, provider, or environment evidence. |
| [`EngineConfig`](../../crates/onnx-genai-engine/src/config.rs) | Backend/device, batch extent, scheduler, speculation, KV page/dtype/connector, budgets, placement, and caches. | Deployment policy; this is the proper home for most of these choices. |
| [`RuntimeConfig`](../../crates/onnx-genai-runtime-config/src/lib.rs) | EP priority, attention mode, threads, capture, fallback, device KV, provider/plugin options. | Deployment policy and environment selection. |
| [`ServeArgs`](../../crates/onnx-genai-server/src/cli.rs) | Session/queue/batch limits and KV mirror dtype. | Service deployment policy. |
| [`BatchingCapability`](../../crates/onnx-genai-engine/src/batched.rs) | Result derived from the resolved backend and decode path. | Runtime admission evidence; stronger than a package boolean. |
| [`ConcurrentRunSupport`](../../crates/onnx-genai-ort/src/session/mod.rs) | Result derived from resolved providers plus capture state. | Runtime admission evidence. |
| [`MemoryStrategyPlan`](../../crates/onnx-genai-engine/src/config.rs) | Inferred and effective memory strategy, source, evidence, and advisory status. | Useful shape for generated evidence, though ORT plans may be advisory only. |
| [`WorkerPool` and `SessionPlacement`](../../crates/onnx-genai-server/src/worker.rs) | Thread ownership and routing of session state. | Runtime topology and placement policy. |
| [`SessionLeases`](../../crates/onnx-genai-server/src/lease.rs) | Same-session single-flight enforcement. | Runtime correctness enforcement of authored mutation semantics. |

### 2.3 Hosted Hugging Face examples

The 2026-08-25 audit searched current documentation, fixtures, scripts, tests,
Git history, pull-request publication records, and every model owned by
`justinchuby` whose exact Hub revision contains `inference_metadata.yaml`.
Upstream Hugging Face sources that contain no inference metadata are provenance,
not hosted metadata examples. The resulting fleet is 28 packages.

Every revision below was downloaded by immutable SHA and checked with the
`f55520de9` `validate_metadata --metadata-only --shape` binary. **Valid** means
the metadata parses and passes semantic validation. Direct ORT evidence means a
model card or evidence file exercises graphs directly; workflow ORT evidence
means the generic workflow engine executed the package. Neither is native
readiness or performance proof unless stated separately.

| Hosted package and exact metadata revision | Model form | Schema | Validation and evidence status |
| --- | --- | --- | --- |
| [`qwen2.5-0.5b-instruct-onnx-genai`](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/blob/a61ca2e7e7a41db4c310b6a24479d768d6ab20ae/inference_metadata.yaml) `a61ca2e7e7a41db4c310b6a24479d768d6ab20ae` | Decoder plus ten semantic policy graphs | v1.0 | Valid; card records generic workflow CPU output. No accepted performance record. |
| [`qwen3-0.6b-onnx-genai`](https://huggingface.co/justinchuby/qwen3-0.6b-onnx-genai/blob/38714511f57e01df01808b930168459a8e7aa9a3/inference_metadata.yaml) `38714511f57e01df01808b930168459a8e7aa9a3` | Decoder plus policy graphs and session conversation state | v1.0 | Valid; card records generic workflow CPU output and multi-turn use. |
| [`deepseek-r1-distill-qwen-1.5b-onnx-genai`](https://huggingface.co/justinchuby/deepseek-r1-distill-qwen-1.5b-onnx-genai/blob/1427c4896f798893e58ffec91aef65c34de4503a/inference_metadata.yaml) `1427c4896f798893e58ffec91aef65c34de4503a` | Decoder plus policy graphs | v1.0 | Valid; card records generic workflow CPU output. |
| [`Muse-Glimmer-30B-ONNX-INT4-CUDA`](https://huggingface.co/justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA/blob/85a1f4b4ac24f1076be51a52e6c934aff4b9e40c/inference_metadata.yaml) `85a1f4b4ac24f1076be51a52e6c934aff4b9e40c` | Image multimodal decoder with full/sliding KV and policy graphs | v1.0 | Valid; card records generic workflow ORT CUDA execution on H200. It does not prove this repository's native backend. |
| [`qwen2.5-14b-instruct-int4-zp-onnx`](https://huggingface.co/justinchuby/qwen2.5-14b-instruct-int4-zp-onnx/blob/753817320d232b0205a7971e8ea25068453fb393/inference_metadata.yaml) `753817320d232b0205a7971e8ea25068453fb393` | INT4 decoder plus policy graphs | v1.0 | Valid; card gives a run recipe and explicitly declines to treat one run as evidence. |
| [`onnx-genai-example-gemma4-e2b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b/blob/79ca25afe326719e4daab79430c90195dfd28f3b/inference_metadata.yaml) `79ca25afe326719e4daab79430c90195dfd28f3b` | Dense hybrid full/sliding decoder | v1.0 | Valid; direct ORT CUDA parity/timing evidence, and this exact target revision participates in the recorded generic speculative run. No exact native-package record. |
| [`onnx-genai-example-qwen-image-edit-2509`](https://huggingface.co/justinchuby/onnx-genai-example-qwen-image-edit-2509/blob/e859aef2289ad02e64812c43fd5e73b5e1c36a2f/inference_metadata.yaml) `e859aef2289ad02e64812c43fd5e73b5e1c36a2f` | Image-conditioned flow-matching workflow | v1.0 | Valid; generic workflow ORT CUDA and HTTP API image-edit evidence. Timings are artifact records, not a general performance proof. |
| [`pangu-weather-1h-onnx-catalogue`](https://huggingface.co/justinchuby/pangu-weather-1h-onnx-catalogue/blob/82beb24f24169b88bb0f108e40fc35840d4a8d57/inference_metadata.yaml) `82beb24f24169b88bb0f108e40fc35840d4a8d57` | Stateless weather forecast graph | v1.0 | Valid; card records a direct ORT CUDA deterministic request. Generic workflow and native execution are not recorded. |
| [`onnx-genai-example-whisper-tiny`](https://huggingface.co/justinchuby/onnx-genai-example-whisper-tiny/blob/a37efd017b049d697d690824618c0cded5cffa78/inference_metadata.yaml) `a37efd017b049d697d690824618c0cded5cffa78` | Audio encoder-decoder with cross-attention state and policy graphs | v1.0 | Valid; direct ORT load/output evidence. General profile-driven or native execution is not established. |
| [`onnx-genai-example-wav2vec2-base-960h-ctc`](https://huggingface.co/justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc/blob/28480e393ad1b8fa2e0bb6939e5daded02f24014/inference_metadata.yaml) `28480e393ad1b8fa2e0bb6939e5daded02f24014` | Audio preprocessing plus encoder-only CTC | v1.0 | **Refreshed and valid**; direct ORT evidence. No `batch_capacity`, so grouped execution is not claimed; no engine CTC-profile dispatcher is proved. |
| [`onnx-genai-example-esm2-t6-8m`](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/d9f1947f4973056fe287e498b0b0e9b675f1bb92/inference_metadata.yaml) `d9f1947f4973056fe287e498b0b0e9b675f1bb92` | Protein encoder with embedding profile | v1.0 | **Refreshed and valid**; direct ORT evidence. The retired profile batching hint was removed; profile execution and grouping remain unproved. |
| [`onnx-genai-example-prot-bert`](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/blob/916d8290b9108a3a6487ff4381ea9a66c32588c0/inference_metadata.yaml) `916d8290b9108a3a6487ff4381ea9a66c32588c0` | Protein encoder with embedding profile | v1.0 | **Refreshed and valid**; direct ORT evidence. The retired profile batching hint was removed; profile execution and grouping remain unproved. |
| [`onnx-genai-example-qwen2-5-0-5b-portable-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-portable-f32/blob/65ef8d35466d402f5bfa5330bb48477e0b330415/inference_metadata.yaml) `65ef8d35466d402f5bfa5330bb48477e0b330415` | Portable f32 decoder plus policy graphs | v1.0 | Valid; direct ORT smoke/output records. Exact native readiness is not recorded. |
| [`onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16/blob/ec8046b051a8f11e6d339a7d9d85dd1235053989/inference_metadata.yaml) `ec8046b051a8f11e6d339a7d9d85dd1235053989` | CUDA GQA f16 decoder plus policy graphs | v1.0 | Valid; direct ORT CUDA smoke/output records. Provider-specific execution is not interchangeability evidence. |
| [`onnx-genai-example-qwen2-5-0-5b-static-cache-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32/blob/bc6b427cbba8db42a1ec3616002a5efab87f6fd0/inference_metadata.yaml) `bc6b427cbba8db42a1ec3616002a5efab87f6fd0` | Fixed-capacity indexed-scatter decoder plus policy graphs | v1.0 | Valid; direct ORT static-cache records. No portable performance claim follows. |
| [`onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32/blob/88352734dc2d8c352c58be5450cc5c2dd7521aef/inference_metadata.yaml) `88352734dc2d8c352c58be5450cc5c2dd7521aef` | Image VLM with hybrid decoder and policy graphs | v1.0 | Valid; direct ORT graph evidence. A full generic multimodal or native package run is not recorded. |
| [`onnx-genai-example-qwen2-5-1-5b-lora-selection`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection/blob/5ca5336b04f0c778f83c0083ee41203dd36961d2/inference_metadata.yaml) `5ca5336b04f0c778f83c0083ee41203dd36961d2` | Decoder with LoRA selection and policy graphs | v1.0 | Valid; direct ORT CUDA adapter evidence. Heterogeneous logical rows ran separately; accelerated heterogeneous workflow batching is not proved. |
| [`onnx-genai-example-mistral-7b-v0-1-sliding-window`](https://huggingface.co/justinchuby/onnx-genai-example-mistral-7b-v0-1-sliding-window/blob/9d1e328848ab57e665d29ad4acb1182621775143/inference_metadata.yaml) `9d1e328848ab57e665d29ad4acb1182621775143` | Sliding-window decoder plus policy graphs | v1.0 | Valid; direct ORT CUDA boundary-crossing evidence. |
| [`onnx-genai-example-qwen3-0-6b-eagle3`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-0-6b-eagle3/blob/0341cd47a8882c1fcbae0840613972321a007371/inference_metadata.yaml) `0341cd47a8882c1fcbae0840613972321a007371` | EAGLE-3 proposer/target workflow | v1.0 | Valid; generic workflow ORT CUDA correctness/acceptance evidence. Native request admission rejects EAGLE-3. |
| [`onnx-genai-stable-diffusion-bk-sdm-small`](https://huggingface.co/justinchuby/onnx-genai-stable-diffusion-bk-sdm-small/blob/2d30ae2ebfacf5c071693836d70ebd14d8fd84d3/inference_metadata.yaml) `2d30ae2ebfacf5c071693836d70ebd14d8fd84d3` | Text-to-image diffusion workflow | v1.0 | Valid; generic workflow ORT CUDA and A1111 API evidence. No accepted end-to-end performance gate. |
| [`onnx-genai-cogvideox-2b`](https://huggingface.co/justinchuby/onnx-genai-cogvideox-2b/blob/29da9103c4517f8026155c0d97e195c26ee56758/inference_metadata.yaml) `29da9103c4517f8026155c0d97e195c26ee56758` | Text-to-video diffusion workflow with temporal recurrence | v1.0 | Valid; generic workflow ORT CUDA video evidence. No native video proof. |
| [`act-aloha-policy-onnx-catalogue`](https://huggingface.co/justinchuby/act-aloha-policy-onnx-catalogue/blob/ebe2b9485d9f2e4ae9d0b181654e2d6d844fda57/inference_metadata.yaml) `ebe2b9485d9f2e4ae9d0b181654e2d6d844fda57` | Action-rollout policy graph | v1.0 | Valid; direct ORT CUDA/PyTorch parity and timing record. Generic workflow/native readiness is not separately keyed. |
| [`moshiko-full-duplex-onnx-catalogue`](https://huggingface.co/justinchuby/moshiko-full-duplex-onnx-catalogue/blob/426253a5a5822eb405e4ac214d6895427c64ef0c/inference_metadata.yaml) `426253a5a5822eb405e4ac214d6895427c64ef0c` | Full-duplex audio workflow with temporal and codec state | v1.0 | Valid; card records direct CUDA duplex output/timing. Generic workflow/native evidence is not separately keyed. |
| [`sensenova-u1.5-8b-mot-onnx-canonical`](https://huggingface.co/justinchuby/sensenova-u1.5-8b-mot-onnx-canonical/blob/541afaea12e85222766b694cccc30153ea6dd3c1/inference_metadata.yaml) `541afaea12e85222766b694cccc30153ea6dd3c1` | Shared-prefix multimodal pixel-flow workflow | v1.0 | Valid; generic workflow ORT CUDA text, image generation, and image-edit evidence. Full autoregressive text decode remains absent. |
| [`onnx-genai-example-gemma4-e2b-assistant`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant/blob/4b6f1533fec1475ade9e3fa3d401ae00a2d7be67/inference_metadata.yaml) `4b6f1533fec1475ade9e3fa3d401ae00a2d7be67` | Target plus cacheless shared-KV assistant | v1.0 | Valid; direct ORT CUDA drafter parity/assisted evidence. The separately packaged composite carries the current generic real-package run. |
| [`onnx-genai-example-minimax-music3`](https://huggingface.co/justinchuby/onnx-genai-example-minimax-music3/blob/5f95fbbfa01956626fc3170fc90a467666aebdd6/inference_metadata.yaml) `5f95fbbfa01956626fc3170fc90a467666aebdd6` | Hierarchical text-to-music/audio workflow | v1.0 | Valid; card records component L4 and audio L5 artifacts. Backend and performance claims are not keyed as project conformance records. |
| [`onnx-genai-example-gemma4-26b-a4b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-26b-a4b/blob/63e02e455bd835f75b096694ec31d5ad91800299/inference_metadata.yaml) `63e02e455bd835f75b096694ec31d5ad91800299` | MoE hybrid full/sliding decoder | v1.0 | Valid; direct ORT CUDA parity/generation/timing evidence. The card describes an ORT/native deterministic fixture, not an exact keyed portability certificate. |
| [`onnx-genai-example-gemma4-e2b-speculative`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-speculative/blob/77a8161bc2a2c9de478dae50307f60e2a0c6beff/inference_metadata.yaml) `77a8161bc2a2c9de478dae50307f60e2a0c6beff` | Self-contained chained shared-KV speculative workflow | v1.0 | Valid; exact generic workflow ORT CUDA target-equivalence, acceptance, rejection, rollback, and residency evidence. No accepted speedup or exact native-package record. |

Fleet-wide findings:

- all 28 packages are workflow-only documents; none relies on retired
  `model.io`;
- all normalize to schema v1.0, which is correct because none authors a v1.1
  `batch_capacity`, padding provenance, or packed ownership contract;
- none uses top-level `required_capabilities`, but every workflow repeats
  derived strings in `manifest.capabilities`;
- none claims encoder/component grouping. Request-aligned tensor layouts alone
  do not permit coalescing;
- cache semantics are represented where applicable, including append,
  replacement, sliding/full attention, and one indexed-scatter static-cache
  package; physical paging/tiering remains runtime-owned;
- hosted model cards contain useful direct and workflow execution records, but
  those prose claims are not backend-readiness fields in the metadata and do
  not become performance proof.

The first audit found exactly three files rejected by merged main:
Wav2Vec2 CTC, ESM-2, and ProtBERT still authored the retired
`profiles.*.batch_invariance`. They were republished metadata-only at the
revisions in the table. Their YAML now contains inline review comments and
conservatively omits `batch_capacity`; graphs, weights, and unrelated files
were not changed.

#### Hosted-example governance

1. The typed parser, semantic validator, generated schema, and canonical design
   documents are authoritative. A Hub file is a deployed instance, never a
   second schema or capability catalogue.
2. Repository tests, evidence, and documentation **MUST** pin immutable Hub
   revisions. `main` or an unpinned `resolve` URL is not review evidence.
3. Publication **MUST** run metadata-only validation against the intended
   runtime revision; artifact-backed execution evidence is a separate gate.
4. A metadata revision change invalidates prior readiness evidence unless the
   evidence key proves the semantic identity is unchanged. Model-card prose
   alone is not such a key.
5. Schema upgrades are need-driven. A v1.0 package does not become stale merely
   because v1.1 exists; it becomes stale when it uses a retired field or needs a
   v1.1 contract it cannot express.
6. A publisher **MUST NOT** add `batch_capacity` merely to replace a removed
   boolean. It must author exact component ports, padding/length provenance,
   ownership, uniform dimensions, and budgets needed for safe grouping.
7. Human review examples SHOULD retain explanatory YAML comments. When a
   production serializer strips them, publish an adjacent annotated reference
   and validate typed semantic equality instead of claiming comments survived.
8. When generic capability lists are removed, the hosted fleet should be
   regenerated from the canonical workflow source in one migration, validated,
   and repinned here; hosted copies must not preserve obsolete fields as a
   compatibility authority.

Ordered annotated references:

1. the compact decoder policy-graph example in
   [§1.4](#14-policy-graph-means-executable-semantics);
2. the in-tree [catalogue](../../examples/inference_metadata/catalogue/README.md),
   especially decoder/static cache examples 1 and 18, multimodal/diffusion
   examples 3 and 7–9, adapters/speculation examples 10–11 and 20–24, and
   encoder/task examples 4–5 and 12–13;
3. the refreshed, inline-annotated
   [Wav2Vec2](https://huggingface.co/justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc/blob/28480e393ad1b8fa2e0bb6939e5daded02f24014/inference_metadata.yaml),
   [ESM-2](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/d9f1947f4973056fe287e498b0b0e9b675f1bb92/inference_metadata.yaml),
   and
   [ProtBERT](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/blob/916d8290b9108a3a6487ff4381ea9a66c32588c0/inference_metadata.yaml)
   production files.

## 3. Classification rules

| Question | Classification | Required representation |
| --- | --- | --- |
| Can it be recovered deterministically from the authored graph/workflow/contracts? | Derived | Do not serialize a second answer. Cache the derivation if necessary. |
| Does a wrong answer change model semantics, state correctness, or request interpretation? | Authored requirement | Typed field or versioned contract with validation. Silence means the conservative behavior. |
| Does an operator choose it differently for cost, latency, memory, trust, or risk? | Deployment policy | Runtime/server configuration outside portable inference metadata. |
| Does the answer depend on artifact bytes, runtime build, EP, device, driver, options, or measurement? | Runtime evidence | Generated record with identity, environment, result, and timestamp. |

Examples:

- `StateUpdate::IndexedScatter`, decoder port roles, graph KV dtype, a rank-3
  position input, and an authored RoPE initializer are derived facts.
- `StateAliasing`, `StateReuse`, rollback bounds, `ComponentBatchCapacity`,
  `EquivalenceClass`, and `SessionMutationPolicy` are authored requirements.
- physical paging, prefix-cache capacity, tiering, connector choice, graph
  capture, EP fallback, batch width, and worker placement are deployment policy.
- “ORT CUDA 1.29 executes this FP8 cache,” “native supports this operator set,”
  “this session may run concurrently,” and “speculation is faster” are evidence.

## 4. Redundancy and contradictions in the current tree

### 4.1 The same structural capability is authored twice

[`workflow_required_capabilities`](../../crates/onnx-genai-metadata/src/validation.rs)
derives capabilities from workflow structure. Validation then requires those
same strings to appear in `pipeline.workflow.manifest.capabilities`.
`derived_capabilities` additionally unions the manually authored manifest list.
The document can therefore omit a fact the validator already proved, or assert
an unrelated string the structure does not justify.

`InferenceMetadata.required_capabilities` is a second generic list above the
manifest list. Neither list is needed for built-in structural features.

### 4.2 Runtime capability strings are neither backend nor artifact readiness

`RuntimeCapabilities::default()` advertises `prefix_cache` and broad workflow
behavior without selecting a decoder path, backend, provider, build feature,
graph, or deployment option. In
contrast, `Engine::batching_capability()` derives an answer from the actual
native batch extent or ORT cache path, and
`Session::concurrent_run_support()` derives an answer from resolved EPs and
capture state. The latter two are the correct pattern.

### 4.3 Admission is not uniformly fail closed

[`admit_inference_metadata`](../../crates/onnx-genai-engine/src/engine/load.rs)
rejects unsupported adapter capabilities because continuing would return
base-model output. It only warns for other unsupported “descriptive”
capabilities and continues through a path that does not interpret the workflow.
That may preserve a known bare-decoder fast path, but it means the generic
capability contract does not consistently mean “support or refuse.”

### 4.4 Obsolete runtime-configurable booleans

Repository consumers of `model.runtime_configurable` read only
`chunked_prefill.chunk_size` during native loading.
`runtime_configurable.prefix_cache` has no execution consumer. The former
`runtime_configurable.continuous_batching`, profile `batch_invariance`, and
`continuous_batching` capability spellings are already rejected with migration
diagnostics. Actual prefix reuse and batching are selected from state/decode
structure, component capacity, provider behavior, and runtime policy.

### 4.5 Hardware and quantization hints can contradict artifacts

- `hardware_requirements.supports_tensor_parallel` duplicates the typed legal
  sharding contract.
- `required_dtypes` and `kv_cache_memory_per_1k_tokens_mb` are derivable from
  graph/state geometry when enough information exists; otherwise a manually
  authored estimate is not readiness evidence.
- `QuantizationIntent` describes intent, while the graph and initializers
  describe what will execute. Admission must use the latter.

### 4.6 Position capability strings have no current serialized program

[`PositionProgram`](../../crates/onnx-genai-metadata/src/schema/pipeline.rs) is
explicitly internal and is not referenced by `PipelineSpec`. Workflow packages
express position tensors as ordinary typed values and port roles. Therefore the
built-in strings `position_program` and `multi_axis_positions` do not identify a
separate serialized workflow position-program surface.

The current native path already derives coordinate rank from the graph's
physical position-input shape in
[`declared_position_rank`](../../crates/onnx-genai-engine/src/native_decode/io.rs).
An absent position input is supported. Authored `cos_cache`/`sin_cache` tensors
are graph values or initializers consumed by RoPE operators and should remain
graph-derived, not mirrored by metadata booleans.

### 4.7 Represented preprocessing is not necessarily executable

Validation recognizes `onnx-genai.image-preprocess`,
`onnx-genai.video-preprocess`, and `onnx-genai.audio-preprocess` in
[`validate_preprocessing_workflow`](../../crates/onnx-genai-metadata/src/validation.rs).
The workflow adapter registry now registers all three in
[`workflow_adapter_registry`](../../crates/onnx-genai-engine/src/pipeline/workflow.rs).
Grouped image and ordered-frame preprocessing execute, but encoded
video-container decode, temporal sampling, and `pad_frames` still fail closed
with a diagnostic directing callers to the grouped frame-sequence API. A
registered boundary therefore still does not imply complete execution of every
program the schema can represent.

### 4.8 Recognized task profiles are not profile-driven execution

Validation recognizes generation, embedding, reranking, classification, reward,
and transcription profiles. The engine has a separate ORT-only
[`Engine::embed_with_options`](../../crates/onnx-genai-engine/src/embedding.rs),
but no general dispatcher that executes `TaskProfile` pooling or CTC decoding.
“Known to the validator” must not be reported as “implemented by the engine.”

### 4.9 Adapter metadata mixes four layers

`application_capability` and `loader_capability` duplicate version/format
information. `portable_fallback` chooses fallback behavior.
`cache.max_entries`, `cache.eviction`, `bucket_by_adapter_set`,
`stable_buffers`, and `invalidate_capture_on_eviction` are deployment/runtime
planning policy. The portable fallback that actually executes requires host
float32 input and float32 JSON weights in
[`pipeline/adapters.rs`](../../crates/onnx-genai-engine/src/pipeline/adapters.rs);
schema support for PEFT, safetensors, and ORT bundles is not proof of accelerated
ORT or native execution.

### 4.10 Cache representation and storage are partially separated, but claims leak

The state service correctly separates graph update discipline from physical
storage. Runtime paging, prefix tries, tiering, connectors, and host-mirror
quantization live outside package metadata.

Gaps remain:

- Graph-authored FP8 KV validates, but the recorded ORT CUDA 1.29
  `GroupQueryAttention` run is rejected before execution; see
  [the measured FP8 section](../benchmarks/2026-08-21-mobius-workflow-conformance.md#fp8-kv-cache--not-executable-with-the-shipped-kernel-and-not-an-ima).
- [`ConnectorBridge`](../../crates/onnx-genai-engine/src/connector_bridge.rs)
  materially injects fetched state only for a byte-exact f32
  `ZeroCopyRebind` path. Other paths may report an opportunity without reducing
  prefill.
- Native loading currently rejects a non-null external KV connector.
- `LocalTieredConfig::compression = Fp8` is accepted, but
  [`local_tiered.rs`](../../crates/onnx-genai-kv/src/local_tiered.rs) records that
  stored-payload compression is still deferred.

### 4.11 Backend auto-selection is not interchangeability evidence

[`resolve_decode_backend`](../../crates/onnx-genai-engine/src/engine/decode_backend.rs)
detects one native-only custom operator and otherwise prefers ORT. Native
operator compatibility is generally discovered during load or execution.
There is no durable evidence record establishing that both backends execute the
same artifact under named conditions.

### 4.12 Graph capture changes correctness and concurrency constraints

Capture is runtime policy, not a model capability.

- ORT capture withdraws concurrent `Run` support for the session.
- Shared-buffer batched decode explicitly runs uncaptured because mask width
  changes per step.
- top-level `If`/`Loop`/`Scan` detection is conservative, but an explicit ORT
  capture request currently emits a warning rather than refusing or requiring
  an explicit fallback policy.
- Native capture has its own structural predicates and decline reasons.

A field that says only “graph capture supported” would conceal all four facts.

### 4.13 Functional workflow execution is not a performance claim

The universal interpreter and execution islands have broad functional tests.
The current published CUDA synthetic measurements fail at least one acceptance
criterion: decoder-policy throughput is `0.903` of the direct composite and
min-p warm TTFT regresses. The record explicitly does not cover production KV,
per-row serving, or the overridable sampler path; see
[Current measured baseline](../benchmarks/2026-08-21-mobius-workflow-conformance.md#current-measured-baseline).

Speculative decoding similarly has strong correctness evidence, while the
speedup test is ignored and environment-gated. `distribution_preserving` is a
correctness/equivalence assertion, never a speed claim.

## 5. Proposed minimal representation

### 5.1 Portable package metadata

Keep only typed package facts and requirements:

1. **Workflow and graph bindings:** components, typed ports, roles, SSA
   dataflow, control flow, effects, outputs, and preprocessing.
2. **Semantic state contracts:** state kind, update discipline, logical length,
   aliasing legality, reuse/eviction legality, rollback/snapshot/fork bounds,
   checkpoint ABI, scope, management, and session continuation.
3. **Request/task contracts:** package facts, generation defaults and override
   surface, task profiles, pooling/decoding semantics, batch invariance.
4. **Composition contracts:** speculative proposer/target compatibility and
   adapter target/artifact bindings.
5. **Non-derivable hard bounds:** for example a semantic maximum context length
   when the graph itself does not encode it.

Do not add a replacement generic capability list. Extension requirements should
enter through the narrow typed extension points that already carry identity and
version: component contracts, adapter ABIs, checkpoint adapters, task profiles,
and artifact formats.

### 5.2 Deterministic derived view

At load, produce a non-serialized `DerivedRequirements` report. Illustrative
fields are:

```text
workload:
  profiles, input modalities, output modalities, control-flow form
decoder:
  sequence source, ports, state groups, cache update disciplines
positions:
  input absent/present, coordinate rank, graph-internal/opaque, RoPE cache edges
state:
  dtypes, geometry, ownership, aliasing, reuse, rollback, checkpoint adapters
composition:
  required component contracts, adapters, proposer/verifier relationships
batching:
  row layouts, padding/packing, invocation bounds, compaction obligations
```

Each item should include a source path such as
`workflow.components.decoder.ports.roles.position_ids` or
`workflow.serving.state_service.groups.full.update`, so diagnostics explain
which authored fact caused a requirement.

### 5.3 Deployment policy

Keep a separate runtime/server configuration for:

- backend, provider, device, precision override, and fallback permission;
- batch sizes, queueing, scheduler, preemption, and worker placement;
- KV physical form, page size, mirror dtype, tiering, connector, budgets, and
  eviction;
- prefix reuse enablement and capacity;
- graph capture, execution islands, and stable-buffer strategy;
- adapter caching, normalization, bucketing, and fallback permission;
- session limits, TTL policy, and observability.

Policy may choose less than the package permits. It may not choose more.

### 5.4 Evidence record

Generate, rather than author, an `AdmissionEvidence` record:

```text
artifact:
  semantic identity, graph/weight digests, metadata version
runtime:
  commit/build features, backend, EP and version, provider options
platform:
  OS, architecture, device, driver
policy:
  execution-affecting resolved options
requirements:
  hash of DerivedRequirements
results:
  parse, structural validation, load, provider placement, execution,
  output parity, capture, concurrency, copies/synchronizations
performance:
  benchmark protocol id, raw-record digest, verdict
```

Readiness is a positive record for the exact key. A family name, a generic
runtime flag, or a result from another provider version does not match.
Performance evidence is optional and separate; absence means “not proven,” not
“slow” and not “unsupported.”

## 6. Derivation rules

1. **Workload form:** derive from required task profiles, workflow inputs and
   outputs, component implementations/contracts, and control flow. Do not infer
   from model names.
2. **Decoder ABI:** continue the existing `decoder_io()` lowering from port
   roles and state groups. Remove any remaining serialized duplicate.
3. **Cache discipline:** derive dynamic append, fixed replacement, and static
   indexed scatter from `StateUpdate`, recurrence, ports, and tensor geometry.
4. **Cache storage:** never derive physical paging, tiering, prefix-cache size,
   or connector placement as a package requirement. Admit a deployment choice
   only if the semantic state contract permits it.
5. **KV precision:** derive graph-visible KV dtype and scale inputs from graph
   values and attention operator inputs. Treat host-mirror quantization as
   deployment policy.
6. **Positions:** derive external position-input presence, dtype, rank, and
   coordinate count from port roles and graph I/O. If no input exists, report
   `graph_internal_or_unused`; do not guess the internal algorithm. Derive
   authored sin/cos cache use from operator edges and initializers.
7. **Batching:** derive whether values can be permuted, packed, padded, or
   compacted from `BatchLayout`, padding companions, serving controls, state
   row scope, and update discipline. `ComponentBatchCapacity` states an upper
   structural bound, not that the runtime should group.
8. **Speculation:** derive all state/effect rollback obligations from the
   speculative region. Keep equivalence and vocabulary compatibility authored.
   Enablement, proposal width, and tree/scheduling strategy are policy.
9. **Adapters:** derive required loader from artifact format and required
   application ABI from the invoked typed contract. Target mapping and scaling
   semantics remain authored. Caching, bucketing, capture invalidation, and
   fallback are policy.
10. **Backend readiness:** run backend-specific admission against the derived
    requirements and artifact. A static implementation registry can explain
    what code is present, but only admission/execution produces readiness.
11. **Concurrency:** derive model/session mutation obligations from metadata.
    Derive actual concurrent-run safety from the concrete session/provider and
    capture state. Choose worker count and placement as policy.

## 7. Current feature-status matrix

Legend:

- **Yes**: the current tree contains the representation/check or at least one
  non-vacuous execution test.
- **Artifact**: proven only for named artifacts/environments, not generally.
- **Partial**: a narrower path exists, or execution bypasses part of the
  represented contract.
- **No**: absent, rejected, or not demonstrated.
- **N/A**: deliberately not a portable package field.

“Executed” does not mean every operator/provider combination works.
“Performance-proven” requires an accepted controlled record, not a benchmark
harness or functional test.

### 7.1 Workloads and composition

| Form | Represented | Validated | Executed ORT | Executed native | Performance-proven | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- |
| Standard autoregressive decoder | Yes | Yes | Yes | Yes | Artifact | Canonical decoder lowering and extensive direct decode tests; performance records are artifact/hardware specific. |
| Encoder-only embedding | Yes | Yes | Yes | No | No | Profiles and hidden-output roles exist; `Engine::embed_with_options` is explicitly ORT-only and does not dispatch from `TaskProfile`. |
| Encoder-only reranking/classification/reward | Yes | Yes | No | No | No | Profile kinds are accepted by validation; no profile executor was found. |
| Encoder-only CTC transcription | Yes | Yes | No | No | No | CTC decoding and vocabulary are validated, but no engine profile-driven CTC interpreter was found. |
| Encoder-decoder / cross-attention | Yes | Yes | Artifact | No | No | Cross-attention and audio ports/state are represented; Whisper ORT evidence is environment-gated. Native end-to-end parity is not established. |
| Image multimodal preprocessing + VLM | Yes | Yes | Yes | Partial | No | Image adapter and ORT workflow tests execute. Native component execution exists, but general real image-path readiness and native grouped vision remain unproven. |
| Encoded-video preprocessing | Yes | Yes | No | No | No | The adapter is registered but encoded container decode/temporal sampling fails closed. Ordered encoded-frame grouping executes in preprocessing, not as an ORT/native encoder invocation. |
| Video-producing workflow/diffusion | Yes | Yes | Yes | No | No | Hermetic and real-package-gated ORT tests exist; no native video workflow proof was found. |
| Audio preprocessing + multimodal/audio workflow | Yes | Yes | Yes | Partial | No | Audio preprocessing and buffered WAV conformance execute; no general native real-audio record was found. |
| Image diffusion/workflow | Yes | Yes | Yes | Yes | No | Hermetic ORT/native diffusion parity and smoke tests exist; no accepted end-to-end performance record. |
| Nested/general workflow IR | Yes | Yes | Yes | Partial | No | ORT has broad interpreter conformance; native has component parity and selected end-to-end workflows, not a universal corpus proof. |
| Legacy external draft-model speculation | Partial | Partial | Yes | No | No | ORT direct path exists. Native rejects per-request draft-model speculation. Speedup test is ignored/env-gated. |
| Workflow-native chained/shared-KV speculation | Yes | Yes | Artifact | Yes | No | Real H200 ORT correctness/acceptance record plus native fixture parity; no accepted speedup record. |
| MTP | Yes | Yes | Yes | Partial | No | ORT MTP executes. Native target execution can use an ORT MTP head; that is not pure-native proposer readiness. |
| EAGLE-3 | Yes | Yes | Yes | No | No | ORT implementation exists; native request admission explicitly rejects EAGLE-3. |
| LoRA/adapters | Yes | Yes | Partial | No | No | Portable host float32 JSON overlay and heterogeneous-row tests exist. Represented PEFT/safetensors/ORT formats do not prove accelerated backend execution. |

### 7.2 Batching and cache forms

| Form | Represented | Validated | Executed ORT | Executed native | Performance-proven | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- |
| ORT continuous batching, static cache | Yes | Yes | Yes | N/A | No | `ContinuousBatchManager` and static-cache functional evidence include divergent row cursors. |
| ORT continuous batching, shared past/present buffer | Yes | Yes | Yes | N/A | No | Requires package aliasing permission, max length, and provider fixed-capacity binding. Capture replay is disabled because mask width changes. |
| Native continuous batching | Yes | Yes | N/A | Yes | Artifact | Requires a persistent session pinned to batch N; recorded native batch measurements are artifact/hardware specific. |
| Encoder/component batching | Yes | Yes | No | No | No | Component grouping contracts validate and grouped image/ordered-frame preprocessing executes. Scheduler-driven grouped component invocation, backend parity/readiness, and performance evidence remain incomplete in [ENCODER_BATCHING.md](ENCODER_BATCHING.md). |
| Dynamic/growing KV | Yes | Yes | Yes | Yes | Artifact | Append discipline and past/present paths execute. Performance is model/backend specific. |
| Fixed/static indexed-scatter KV | Yes | Yes | Artifact | Yes | No | Real ORT Qwen2 evidence and native static-cache fixture execution exist. Ragged prefill is not claimed. |
| Physical paged KV storage | N/A | Runtime | Partial | Partial | No | `PagedKvCache` exists; physical paging is correctly runtime-owned. Backend handoff and device residency remain path-specific. |
| Prefix reuse | Semantic legality only | Yes | Yes | Yes | Artifact | `StateReuse` states legality; runtime prefix tries/snapshots execute. A small benchmark is not a portable claim. |
| Quantized host paged-KV mirror | N/A | Runtime | Partial | Partial | No | `EngineConfig::kv_cache_dtype` and paged-store codecs exist; this is not graph-native quantized KV execution. |
| Graph-authored FP8 KV | Yes | Yes | No on recorded ORT CUDA 1.29 | No evidence | No | Valid package, unavailable shipped CUDA GQA type support. |
| Local hot/warm/cold tiering | N/A | Runtime | Partial | No | No | Local tiered connector executes, but materialized engine reuse is limited to compatible ORT paths; native rejects connectors. |
| External/distributed KV connector | N/A | Runtime | Partial | No | No | Lookup/store/fetch APIs exist; actual prefill shortening is byte-exact f32 `ZeroCopyRebind` only, otherwise reporting-only. |

### 7.3 Positions, backends, capture, and concurrency

| Form | Represented | Validated | Executed ORT | Executed native | Performance-proven | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- |
| No external position input | Derived | Yes | Yes | Yes | N/A | Absence is supported; it means graph-internal or unused, not a guessed position algorithm. |
| Rank-2 linear position input | Derived | Yes | Yes | Yes | N/A | Port role and graph shape determine the contract. |
| Rank-3 multi-axis position input | Derived | Yes | Artifact | Artifact | No | Native derives the static coordinate rank from graph I/O; readiness remains artifact/operator specific. |
| Authored internal position generation | Graph only | Graph load | Artifact | Artifact | No | Execute the graph as authored; no generic metadata flag proves the internal algorithm. |
| Authored sin/cos RoPE caches | Graph only | Operator/load | Artifact | Artifact | No | Initializers/edges are the authority; operator coverage is backend specific. |
| ORT/native interchangeability | No evidence model | No | Artifact | Artifact | No | Both backends execute overlapping corpora, but no durable paired readiness record exists. Auto-selection is not proof. |
| ORT graph capture | N/A | Runtime | Yes | N/A | Artifact | Policy plus provider capability; capture withdraws concurrent `Run`, and some paths deliberately run uncaptured. |
| Native whole-step graph capture | N/A | Runtime | N/A | Yes | Artifact | Structural predicates and decline reasons exist; results are shape/artifact/device specific. |
| Concurrent `Run` on one ORT session | N/A | Runtime | Yes when admitted | N/A | No | Allowed only when every resolved EP declares it and capture is off. |
| Same-session concurrent turns | Authored mutation rule | Yes | Refused | Refused | N/A | Routing-layer leases enforce single-flight for exclusive session mutation. |
| Worker placement and multi-worker serving | N/A | Runtime | One worker | One worker | No | Session placement includes worker identity, but the production pool contains exactly one worker. |
| Workflow sampler/interpreter optimization | Yes | Yes | Yes | Partial | No | Functional execution exists. Published CUDA policy-chain results do not meet all acceptance gates. |

## 8. Fail-closed rules and examples

1. **Unknown required typed contract:** reject and name its ID/version. Only a
   profile explicitly marked `ignorable` may be skipped.
2. **Executor absent:** a package using
   `onnx-genai.video-preprocess@1` must be rejected until the selected runtime
   registers and admits that executor. Validation success is insufficient.
3. **Backend unavailable:** an explicit native request must not silently run
   ORT, and an explicit ORT request must not silently run native. Any operator-
   authorized fallback must be policy, observable, and part of the evidence key.
4. **Provider cannot execute a graph dtype/operator:** the FP8 GQA package
   remains represented and valid but is “not ready on ORT CUDA 1.29,” not
   “FP8-capable.”
5. **Optional optimization unproven:** encoder grouping, graph capture, prefix
   fetch, adapter acceleration, or concurrent `Run` may decline to a correct
   ordinary path only when policy permits that fallback and the fallback still
   honors all model requirements.
6. **Fallback changes semantics:** never continue with base weights when an
   adapter is required, never omit state rollback, and never replace a required
   preprocessing program with guessed model-family logic.
7. **Capture safety uncertain:** disable capture or reject according to explicit
   policy. A warning followed by attempting a known-unsafe capture is not a
   fail-closed admission result.
8. **Concurrency uncertain:** use one session per worker or reject overlapping
   runs. No provider declaration means no concurrent sharing.
9. **External KV incompatible:** report zero materialized tokens and recompute.
   Do not count a lookup opportunity as a cache hit that shortened prefill.
10. **No performance record:** report “functional; performance unproven.” Do not
    advertise “optimized,” “fast,” or a speedup from capability presence.

## 9. Migration and removal table

There is no development-time backward-compatibility requirement. Staging below
exists to keep active PRs independently landable, not to preserve obsolete
fields indefinitely.

| Current field/surface | Action | Replacement/source of truth |
| --- | --- | --- |
| `InferenceMetadata.required_capabilities` | Remove | Deterministic derived requirements plus typed extension contracts. |
| `WorkflowManifest.capabilities` | Remove | Workflow structure, contracts, effects, state, batch layouts, and emits. |
| `RuntimeCapabilities.supported` global strings | Replace | Typed implementation registry plus artifact/backend admission report. |
| `model.runtime_configurable.prefix_cache` | Remove | `StateReuse::prefix_reusable` states legality; runtime policy enables/caps reuse. |
| `model.runtime_configurable.continuous_batching` | Keep removed | Merged main rejects it; derive batchability from component/decode/state ABI and concrete backend session, while policy selects width. |
| `profiles.*.batch_invariance` | Keep removed | Merged main rejects it; `ComponentBatchCapacity` is the sole authored grouping-equivalence assertion. |
| Built-in `continuous_batching` capability string | Keep removed | Merged main rejects it as an optimization, not a correctness requirement. |
| `model.runtime_configurable.chunked_prefill.chunk_size` | Move | Runtime policy/default profile. Keep only a typed hard graph bound if one exists. |
| `hardware_requirements.supports_tensor_parallel` | Remove | `model.sharding.tensor_parallel` legal shard facts. |
| `hardware_requirements.min_tp_degree` | Move | Distribution/performance recommendation with evidence. |
| `hardware_requirements.required_dtypes` | Derive/remove | Graph/operator/input/initializer requirements plus backend admission. |
| `hardware_requirements.beneficial_dtypes` | Move | Distribution/performance recommendation. |
| `hardware_requirements.kv_cache_memory_per_1k_tokens_mb` | Derive/remove | State geometry, graph KV dtype, and selected runtime storage policy. |
| `hardware_requirements.min_memory_gb` | Move | Distribution placement hint; admission uses measured/resolved budgets. |
| `quantization.default/overrides` intent | Move | Provenance/distribution manifest; execution reads actual graph representation. |
| `model.vocab_size` when logits/tokenizer determine it | Derive | Graph output and package tokenizer facts; reject disagreements. |
| `adapters.application_capability` | Remove or type | Versioned invoked adapter/component contract. |
| `adapters.*.weights[].loader_capability` | Derive | `AdapterWeightFormat` plus a versioned format decoder. |
| `adapters.portable_fallback` | Move | Operator fallback policy; absence of an executable required adapter rejects. |
| `adapters.cache.*` | Move | Runtime adapter-cache policy. |
| `adapters.planning.*` | Move | Runtime scheduler/buffer/capture policy and generated diagnostics. |
| Built-in `position_program` / `multi_axis_positions` strings | Remove | Workflow port roles and graph I/O rank; internal position logic stays in graph. |
| Serialized decoder ABI duplicates | Remove | Existing `InferenceMetadata::decoder_io()` derivation. |

### Stages

1. **Document and instrument:** land this proposal alone. Add a generated
   `DerivedRequirements`/admission diagnostic in a later runtime PR without
   changing schema behavior.
2. **Make derivation authoritative:** make all loaders consume the same derived
   view. Add contradiction tests against old fields while they still exist.
3. **Separate policy:** move runtime-configurable, hardware-placement, adapter
   cache/planning, and fallback choices to runtime/server configuration.
4. **Add evidence keys and gates:** persist backend admission and conformance
   results; make auto-selection consume exact matching evidence or probe by
   loading.
5. **Delete redundant schema fields and strings:** now that the batching schema
   work is merged, update parser, validation, generated schema, fixtures, and
   canonical docs in one dedicated follow-up PR. No compatibility aliases.
6. **Republish hosted instances:** regenerate the pinned Hub fleet from the
   authoritative typed source, validate exact uploaded bytes, and update this
   inventory. Do not preserve obsolete hosted fields as aliases.
7. **Tighten claims:** make status APIs and documentation report represented,
   validated, executed, and performance-proven separately.

Stages 1–4 can land without editing the batching schema or its canonical design
document. Stage 5 remains intentionally isolated from this proposal and from
the already-merged batching work.

## 10. Recommended acceptance tests

1. **Single-authority derivation:** mutate one workflow fact at a time and prove
   the derived view changes; there is no second writable answer.
2. **Contradiction removal:** after field deletion, old spellings fail as
   unknown fields rather than being silently translated.
3. **Backend admission corpus:** for every committed artifact, record ORT and
   native parse/load/run verdicts keyed by exact build/provider/device.
4. **Workload corpus:** standard decoder; encoder-only embedding and CTC;
   encoder-decoder; image, video, and audio input paths; image/video/audio
   outputs; diffusion; nested control flow.
5. **Speculation corpus:** block, chained/shared-KV, prompt lookup, MTP, and
   EAGLE-3; verify proposal activity, acceptance, rejection, rollback, state
   correction, and target-equivalent output.
6. **Adapter corpus:** each represented format, heterogeneous adapter sets,
   compaction/release, cache eviction, backend path, and base-output prevention.
7. **Cache corpus:** dynamic append, replacement, indexed scatter, paged
   storage, prefix reuse, quantized mirror, graph FP8, tiered and external
   fetch. Separately assert `would_extend_tokens` and `fetched_tokens`.
8. **Position corpus:** absent input, rank-2, rank-3, graph-internal positions,
   and authored sin/cos caches under every claimed backend.
9. **Batching corpus:** ORT static/shared-buffer and native pinned batch, plus
   encoder grouping only after the component/backend/artifact triple passes.
10. **Capture/concurrency matrix:** capture on/off × provider concurrent-run
    declaration × control flow × stable/dynamic shape. Every decline names the
    exact predicate.
11. **Session placement:** exclusive and copy-on-write mutation policies,
    same-session racing turns, multiple workers, failover, and state continuity.
12. **Performance gates:** use
    [WORKFLOW_PERFORMANCE_CONFORMANCE.md](../WORKFLOW_PERFORMANCE_CONFORMANCE.md);
    retain raw samples and identities, and never accept an env-gated harness as
    a published pass without a recorded run.
13. **Hosted fleet:** discover Hub files from repository references and the
    publisher inventory, download immutable revisions, validate exact bytes,
    reject retired fields, verify uploads byte-for-byte, and prove any
    comment-free production document has a parse-equivalent annotated
    reference.

## 11. Unresolved questions

1. Should readiness key the whole package semantic identity, every graph and
   weight digest, or both? The answer must detect a changed component without
   invalidating unrelated evidence unnecessarily.
2. Where should evidence live, who may sign it, and when does a provider/driver
   update expire it?
3. Which custom-operator semantics need typed authored requirements because
   graph inspection cannot recover them?
4. Should `Auto` backend selection probe both backends at load, consume a
   trusted evidence cache, or use both in that order?
5. What is the exact policy vocabulary for correctness-preserving fallback
   versus forbidden semantic fallback?
6. Who owns execution of task-profile pooling and CTC decoding: the universal
   workflow, a profile interpreter, or API-specific code?
7. Which runtime/service owns encoded video-container decode, frame sampling,
   and `pad_frames`, and how is that implementation admitted independently from
   the already-executable ordered-frame grouping path?
8. What constitutes pure-native MTP readiness when the target is native but the
   proposer remains ORT?
9. What compatibility identity is required before external KV bytes may be
   materialized across processes or builds?
10. Which session state forms can actually snapshot/fork/copy-on-write on each
    backend, rather than merely represent those legal semantics?
11. How should sharding legality, collective implementation, distributed
    placement, and measured scaling be recorded without recreating one
    optimistic “distributed supported” flag?
12. Which performance evidence may be generalized across devices, and which
    must remain exact-machine evidence?

## 12. Reader guide

Read in this order:

1. [`RULES.md`](../../RULES.md) — especially fail-closed behavior and avoiding
   model/provider identity conditionals.
2. [`schema/mod.rs`](../../crates/onnx-genai-metadata/src/schema/mod.rs) — the
   top-level package contract and the non-serialized decoder ABI cache.
3. [`schema/ir.rs`](../../crates/onnx-genai-metadata/src/schema/ir.rs) — typed
   workflow, batching, adapters, state service, session semantics, and outputs.
4. [`schema/package.rs`](../../crates/onnx-genai-metadata/src/schema/package.rs),
   [`schema/pipeline.rs`](../../crates/onnx-genai-metadata/src/schema/pipeline.rs),
   and
   [`schema/decoder_abi.rs`](../../crates/onnx-genai-metadata/src/schema/decoder_abi.rs)
   — task/speculative/preprocessing/model surfaces.
5. [`decoder_abi.rs`](../../crates/onnx-genai-metadata/src/decoder_abi.rs) and
   [`validation.rs`](../../crates/onnx-genai-metadata/src/validation.rs) —
   derivation versus duplicated capability enforcement.
6. [`engine/load.rs`](../../crates/onnx-genai-engine/src/engine/load.rs) and
   [`engine/decode_backend.rs`](../../crates/onnx-genai-engine/src/engine/decode_backend.rs)
   — actual admission, fallback, and backend selection.
7. [`pipeline/workflow.rs`](../../crates/onnx-genai-engine/src/pipeline/workflow.rs),
   [`pipeline/native_component.rs`](../../crates/onnx-genai-engine/src/pipeline/native_component.rs),
   and
   [`pipeline/islands.rs`](../../crates/onnx-genai-engine/src/pipeline/islands.rs)
   — what the interpreter and both component backends actually execute.
8. [`batched.rs`](../../crates/onnx-genai-engine/src/batched.rs),
   [`connector_bridge.rs`](../../crates/onnx-genai-engine/src/connector_bridge.rs),
   and [`onnx-genai-kv`](../../crates/onnx-genai-kv/src/lib.rs) — batching and
   cache reality.
9. [`onnx-genai-ort/session/mod.rs`](../../crates/onnx-genai-ort/src/session/mod.rs)
   — provider resolution, capture, concurrent-run evidence, and thread safety.
10. [`server/worker.rs`](../../crates/onnx-genai-server/src/worker.rs),
    [`server/lease.rs`](../../crates/onnx-genai-server/src/lease.rs), and
    [`server/driver.rs`](../../crates/onnx-genai-server/src/driver.rs) — current
    single-worker placement and session single-flight.
11. [ENCODER_BATCHING.md](ENCODER_BATCHING.md),
    [CHAINED_SPECULATIVE_EVIDENCE.md](CHAINED_SPECULATIVE_EVIDENCE.md),
    [WORKFLOW_PERFORMANCE_CONFORMANCE.md](../WORKFLOW_PERFORMANCE_CONFORMANCE.md),
    and
    [the dated workflow evidence](../benchmarks/2026-08-21-mobius-workflow-conformance.md)
    — represented versus executed versus measured status.
12. The pinned [hosted-example inventory](#23-hosted-hugging-face-examples),
    followed by the compact annotated policy graph and the three refreshed
    annotated production files — deployed instances versus their authoritative
    schema and evidence.

### Invariants to scrutinize

- Is every structural answer derived from one authored source?
- Does every remaining authored field change correctness rather than merely
  deployment preference?
- Can an unsupported requirement ever degrade to a warning or silent fallback?
- Is readiness attached to exact artifact/runtime/provider identity?
- Does a status distinguish validation, execution, and performance?
- Does any cache field confuse model semantics with storage policy?
- Does capture or concurrent execution change the admission result?
- Does a batching claim name the exact artifact/backend/component combination?
- Could an adapter or speculative fallback silently change output semantics?
- Is every hosted example pinned, validated at that revision, and prevented from
  becoming a schema or readiness authority?

### Short review checklist

- [ ] No generic capability boolean/list duplicates typed structure.
- [ ] No deployment choice is stored as portable model truth.
- [ ] Unsupported required behavior fails closed with an actionable source path.
- [ ] ORT and native readiness are independently evidenced.
- [ ] Functional and performance claims are separately labeled.
- [ ] Cache, position, batching, adapter, speculation, and session forms are all covered.
- [ ] Migration deletions have one replacement authority and no compatibility alias.
- [ ] Acceptance tests include negative and decline paths, not only successful execution.
- [ ] Hosted revisions, comments/annotated references, and evidence labels match exact bytes.
