# Metadata capability model

Status: **normative proposal and implementation reader guide**. This document
audits the current tree and explains the schema, validation, runtime, examples,
and evidence changes carried by this proposal.

The central rule is:

> A package describes what its authored graphs and workflow require. A runtime
> proves what a concrete backend deployment can execute. An operator chooses
> policy. A backend derives optimization plans. Measurements justify performance
> claims. None of those five answers substitutes for another.

## 1. Terms and conceptual model

### 1.1 Five kinds of statement

| Kind | Meaning | Authority | Examples |
| --- | --- | --- | --- |
| **Derived structural fact** | A fact recoverable from authored graphs, workflow dataflow, tensor contracts, operator attributes, initializers, or state contracts. | Package artifacts plus deterministic derivation. | Decoder ports, position-input presence and rank, append versus indexed-scatter cache updates, graph KV dtype, image/video/audio workflow shape, control flow. |
| **Authored model requirement** | A semantic fact required for correct execution but not safely inferable from tensor shape or graph topology. | Typed package contract. | Aliasing legality, prefix-reuse legality, rollback bounds, component grouping equivalence, effect retry/speculation semantics, equivalence class, session mutation semantics. |
| **Deployment/QoS policy** | A choice about cost, risk, placement, or service behavior. | Operator/runtime configuration, never portable model metadata. | Backend/provider selection, page size, KV mirror dtype, cache/tier budgets, graph-capture enablement/fallback, batch size, worker count and placement. |
| **Backend-derived execution plan** | An automatic optimizer plan computed after artifact inspection and resource admission. | Backend compiler/planner; never authored package metadata. | ORT execution islands, `IoBinding` groups, capture segments, buffer-reuse schedules. |
| **Runtime evidence** | A result observed for one artifact and one runtime environment. | Generated admission, conformance, profile, or benchmark record. | ORT session loaded with no fallback, native operator coverage passed, capture replayed, concurrent `Run` was admitted, throughput met a threshold. |

#### ONNX port ABI versus workflow contracts

An ONNX artifact is the sole authority for its physical port ABI: port names,
dtype, rank, shape, and opset imports. An ONNX-backed workflow component does
not author a second copy of those port contracts. The workflow instead owns the
facts the graph cannot state: semantic roles, request/package/upstream sources,
cross-component bindings, state lifetime and recurrence, batching alignment,
control flow, and effects.

Workflow SSA values still carry contracts where the interpreter must validate a
request boundary, connect several components, or type a native adapter/policy
component that has no ONNX graph. For an ONNX-backed value those contracts are
generated from the artifact and checked against it; they are not an independent
human-authored answer. This division provides static whole-workflow validation
without allowing metadata to contradict the graph.

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
7. Package-authored policy graphs, deployment/QoS knobs, and backend-derived
   execution plans **MUST NOT** share a generic `capabilities` bucket. A
   real enable/disable or fallback knob is policy; the generated plan itself
   remains derived.

Core field names are reserved and fail closed when unknown. Package-local names,
artifact port names, and registered extension identifiers are separate naming
classes with different validation rules; see
[reserved fields, local names, and extension identifiers](INFERENCE_METADATA_DECISIONS.md#42a-reserved-fields-local-names-and-extension-identifiers).

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
- provider choice, queue priority, batching delay, capture
  enablement/fallback, cache capacity, and worker placement change where or
  when the same semantics run, so deployment configuration owns them.

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
metadata; they must not claim comments survive generation. A backend may fuse
the `decoder` and `sampler` invocations into an ORT `IoBinding`/CUDA-graph
execution island. That island is derived optimizer output: it changes neither
the package-authored sampling equation nor the operator's deployment policy.

### 1.5 Token authority and the generation boundary

Numeric token ids belong to the tokenizer vocabulary namespace. Their sole
package authority is therefore `package.tokenizer.special_tokens`:

```yaml
package:
  tokenizer:
    algorithm: bpe       # Optional fact: how tokenizer.json segments text.
    vocab_size: 256000   # Optional fact: the id namespace used by model tensors.
    artifacts:
      - location: tokenizer.json
    special_tokens:      # Numeric package defaults, not token spellings.
      bos_token_id: 2
      eos_token_id: [1, 106]  # Ordered set: either id ends default generation.
      pad_token_id: 0
      image_token_id: 255036
      video_token_id: 255037
      audio_token_id: 255038
```

Token strings, added-token maps, and chat templates remain in tokenizer assets.
Repeating their ids in workflow literals, components, or generation defaults
would create conflicting authorities. A request EOS tensor is an explicit
override: when present it replaces the package EOS set for that request.

A termination component consumes the resolved EOS set and computes active/done
state; it does not own the ids. Consequently:

- a **complete-generation package** declares a generation loop, non-empty
  package EOS facts, and an invoked `onnx-genai.termination-predicate` or
  `onnx-genai.token-policy` implementation;
- a **logits-only decoder package** may declare tokenizer facts and a decoder
  output without sampling or termination, but must not claim complete
  generation;
- encoder, embedding, diffusion, and other non-autoregressive packages need no
  EOS declaration merely because a tokenizer artifact is present.

This explains the apparent difference among hosted examples. Gemma decoder-only
examples intentionally stop at logits and therefore have no sampler or EOS
termination component. Examples that claim end-to-end text generation include
those components and the package token facts they consume.

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
| [`plan_execution_islands`](../../crates/onnx-genai-engine/src/pipeline/islands.rs) and [`ExecutionIslandDiagnostic`](../../crates/onnx-genai-engine/src/pipeline/islands.rs) | Automatically group compatible workflow components for ORT `IoBinding`/CUDA-graph optimization and report the resulting plan. | Backend-derived execution plan, not package semantics or deployment policy. |
| [`BatchingCapability`](../../crates/onnx-genai-engine/src/batched.rs) | Result derived from the resolved backend and decode path. | Runtime admission evidence; stronger than a package boolean. |
| [`ConcurrentRunSupport`](../../crates/onnx-genai-ort/src/session/mod.rs) | Result derived from resolved providers plus capture state. | Runtime admission evidence. |
| [`MemoryStrategyPlan`](../../crates/onnx-genai-engine/src/config.rs) | Inferred and effective memory strategy, source, evidence, and advisory status. | Useful shape for generated evidence, though ORT plans may be advisory only. |
| [`WorkerPool` and `SessionPlacement`](../../crates/onnx-genai-server/src/worker.rs) | One or more owner-thread engines, deterministic new-session/stateless placement, and routing back to the worker that owns session state. | Implemented runtime topology; worker count is deployment policy, while the saved placement is runtime state. |
| [`SessionLeases`](../../crates/onnx-genai-server/src/lease.rs) | Same-session single-flight enforcement. | Runtime correctness enforcement of authored mutation semantics. |

### 2.3 Hosted Hugging Face examples

The 2026-08-26 audit searched current documentation, fixtures, scripts, tests,
Git history, pull-request publication records, and the live Hugging Face
collection API. It also checked every model owned by `justinchuby` whose exact
Hub revision contains `inference_metadata.yaml`.
Upstream Hugging Face sources that contain no inference metadata are provenance,
not hosted metadata examples. The resulting fleet is 28 packages.

Every revision below was downloaded by immutable SHA and checked with the
`edc42d2cb` `validate_metadata --metadata-only --shape` binary. **Valid** means
the metadata parses and passes semantic validation. Direct ORT evidence means a
model card or evidence file exercises graphs directly; workflow ORT evidence
means the generic workflow engine executed the package. Neither is native
readiness or performance proof unless stated separately. A card-reported run is
accepted here only when it names the artifact/runtime/result; a maintained test
that can silently skip does not independently upgrade that claim.

| Hosted package and exact metadata revision | Model form | Normalized schema | Validation and evidence status |
| --- | --- | --- | --- |
| [`qwen2.5-0.5b-instruct-onnx-genai`](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/blob/1eabeec267303a75170ae1b43acf59cb01b47a63/inference_metadata.yaml) `1eabeec267303a75170ae1b43acf59cb01b47a63` | Decoder plus ten semantic policy graphs | v1.2 | Valid; card records generic workflow CPU output. No accepted performance record. |
| [`qwen3-0.6b-onnx-genai`](https://huggingface.co/justinchuby/qwen3-0.6b-onnx-genai/blob/e6fdc5eb2ba34f2163c95255279303b09360b702/inference_metadata.yaml) `e6fdc5eb2ba34f2163c95255279303b09360b702` | Decoder plus policy graphs and session conversation state | v1.2 | Valid; card records generic workflow CPU output and multi-turn use. |
| [`deepseek-r1-distill-qwen-1.5b-onnx-genai`](https://huggingface.co/justinchuby/deepseek-r1-distill-qwen-1.5b-onnx-genai/blob/b30d31a08983f5acc013f7f2df95d9f9444b69d2/inference_metadata.yaml) `b30d31a08983f5acc013f7f2df95d9f9444b69d2` | Decoder plus policy graphs | v1.2 | Valid; card records generic workflow CPU output. |
| [`Muse-Glimmer-30B-ONNX-INT4-CUDA`](https://huggingface.co/justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA/blob/76c9896978f6c3f36d75ab7f627521f168cc7010/inference_metadata.yaml) `76c9896978f6c3f36d75ab7f627521f168cc7010` | Image multimodal decoder with full/sliding KV and policy graphs | v1.2 | Valid; card records generic workflow ORT CUDA execution on H200. It does not prove this repository's native backend. |
| [`qwen2.5-14b-instruct-int4-zp-onnx`](https://huggingface.co/justinchuby/qwen2.5-14b-instruct-int4-zp-onnx/blob/b25c589c213ec1efe51eabff1bd35c3cc38fbc4f/inference_metadata.yaml) `b25c589c213ec1efe51eabff1bd35c3cc38fbc4f` | INT4 decoder plus policy graphs | v1.2 | Valid; card gives a run recipe and explicitly declines to treat one run as evidence. |
| [`onnx-genai-example-gemma4-e2b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b/blob/a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f/inference_metadata.yaml) `a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f` | Dense hybrid full/sliding decoder | v1.0 | Valid; direct ORT CUDA parity/timing evidence. Its canonical package bytes are unchanged from the target revision in the recorded generic speculative run. No exact native-package record. |
| [`onnx-genai-example-qwen-image-edit-2509`](https://huggingface.co/justinchuby/onnx-genai-example-qwen-image-edit-2509/blob/69544173de85fe785b1dfd7e1f5f1795c23bafb1/inference_metadata.yaml) `69544173de85fe785b1dfd7e1f5f1795c23bafb1` | Image-conditioned flow-matching workflow | v1.0 | Valid; generic workflow ORT CUDA and HTTP API image-edit evidence. Timings are artifact records, not a general performance proof. |
| [`pangu-weather-1h-onnx-catalogue`](https://huggingface.co/justinchuby/pangu-weather-1h-onnx-catalogue/blob/36baa7a9b345c3accf6f9e5a0303d9b6960dea34/inference_metadata.yaml) `36baa7a9b345c3accf6f9e5a0303d9b6960dea34` | Stateless weather forecast graph | v1.0 | Valid; card records a direct ORT CUDA deterministic request. Generic workflow and native execution are not recorded. |
| [`onnx-genai-example-whisper-tiny`](https://huggingface.co/justinchuby/onnx-genai-example-whisper-tiny/blob/4a80120cbe62c10d25ae89d6430896e726565569/inference_metadata.yaml) `4a80120cbe62c10d25ae89d6430896e726565569` | Audio encoder-decoder with cross-attention state and policy graphs | v1.2 | Valid; direct ORT load/output evidence. General profile-driven or native execution is not established. |
| [`onnx-genai-example-wav2vec2-base-960h-ctc`](https://huggingface.co/justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc/blob/820a7c59ad73e088858230d567395d625c3fac04/inference_metadata.yaml) `820a7c59ad73e088858230d567395d625c3fac04` | Audio preprocessing plus encoder-only CTC | v1.0 | **Refreshed and valid**; direct ORT evidence. No `batch_capacity`, so grouped execution is not claimed; no engine CTC-profile dispatcher is proved. |
| [`onnx-genai-example-esm2-t6-8m`](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/1954e4b60aa939a8220c884c3408d06ffa34e494/inference_metadata.yaml) `1954e4b60aa939a8220c884c3408d06ffa34e494` | Protein encoder with embedding profile | v1.0 | **Canonical-producer parity restored and valid.** [`provenance.json`](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/1954e4b60aa939a8220c884c3408d06ffa34e494/provenance.json) pins `onnxruntime/mobius@8e3ab921a`; request-aligned layouts are preserved and `batch_capacity` is correctly absent. Direct ORT evidence remains; profile-driven execution and cross-request grouping remain unproved. |
| [`onnx-genai-example-prot-bert`](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/blob/83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d/inference_metadata.yaml) `83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d` | Protein encoder with embedding profile | v1.0 | **Canonical-producer parity restored and valid.** [`provenance.json`](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/blob/83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d/provenance.json) pins `onnxruntime/mobius@8e3ab921a`; request-aligned layouts are preserved and `batch_capacity` is correctly absent. Direct ORT evidence remains; profile-driven execution and cross-request grouping remain unproved. |
| [`onnx-genai-example-qwen2-5-0-5b-portable-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-portable-f32/blob/329b806feac8e219970124825d37348f7196152a/inference_metadata.yaml) `329b806feac8e219970124825d37348f7196152a` | Portable f32 decoder plus policy graphs | v1.2 | Valid; direct ORT smoke/output records. Exact native readiness is not recorded. |
| [`onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16/blob/96fa048518005bc0eb9df7ad939fb9f0dd172911/inference_metadata.yaml) `96fa048518005bc0eb9df7ad939fb9f0dd172911` | CUDA GQA f16 decoder plus policy graphs | v1.2 | Valid; direct ORT CUDA smoke/output records. Provider-specific execution is not interchangeability evidence. |
| [`onnx-genai-example-qwen2-5-0-5b-static-cache-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32/blob/5e94f4ee439af803b990a59094c2902f7bf03a4f/inference_metadata.yaml) `5e94f4ee439af803b990a59094c2902f7bf03a4f` | Fixed-capacity indexed-scatter decoder plus policy graphs | v1.2 | Valid; direct ORT static-cache records. No portable performance claim follows. |
| [`onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32/blob/a5f905097a0316eb918f71e94fc55084d34f09ca/inference_metadata.yaml) `a5f905097a0316eb918f71e94fc55084d34f09ca` | Image VLM with hybrid decoder and policy graphs | v1.2 | Valid; direct ORT graph evidence. A full generic multimodal or native package run is not recorded. |
| [`onnx-genai-example-qwen2-5-1-5b-lora-selection`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection/blob/5f0878d8f72cb54b284f052b5ce016717898872b/inference_metadata.yaml) `5f0878d8f72cb54b284f052b5ce016717898872b` | Decoder with LoRA selection and policy graphs | v1.2 | Valid; direct ORT CUDA adapter evidence. Heterogeneous logical rows ran separately; accelerated heterogeneous workflow batching is not proved. |
| [`onnx-genai-example-mistral-7b-v0-1-sliding-window`](https://huggingface.co/justinchuby/onnx-genai-example-mistral-7b-v0-1-sliding-window/blob/e6c5b87a25883e2d1560584137bea340e3a8fba2/inference_metadata.yaml) `e6c5b87a25883e2d1560584137bea340e3a8fba2` | Sliding-window decoder plus policy graphs | v1.2 | Valid; direct ORT CUDA boundary-crossing evidence. |
| [`onnx-genai-example-qwen3-0-6b-eagle3`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-0-6b-eagle3/blob/10385b7b8f1a3066d4ff15a72ec2194cce324f19/inference_metadata.yaml) `10385b7b8f1a3066d4ff15a72ec2194cce324f19` | EAGLE-3 proposer/target workflow | v1.0 | Valid. The card reports ORT CUDA correctness/acceptance, but the maintained real-package gate returns success without executing when its environment variable is absent; this proposal therefore records implementation but no accepted project execution evidence. Native request admission rejects EAGLE-3. |
| [`onnx-genai-stable-diffusion-bk-sdm-small`](https://huggingface.co/justinchuby/onnx-genai-stable-diffusion-bk-sdm-small/blob/dd7ecd9d50a2210aa796a2efedb5489125f8be37/inference_metadata.yaml) `dd7ecd9d50a2210aa796a2efedb5489125f8be37` | Text-to-image diffusion workflow | v1.0 | Valid; generic workflow ORT CUDA and A1111 API evidence. No accepted end-to-end performance gate. |
| [`onnx-genai-cogvideox-2b`](https://huggingface.co/justinchuby/onnx-genai-cogvideox-2b/blob/27e85b66e91a2be33be53ec44d0247a9b232220d/inference_metadata.yaml) `27e85b66e91a2be33be53ec44d0247a9b232220d` | Text-to-video diffusion workflow with temporal recurrence | v1.0 | Valid; generic workflow ORT CUDA video evidence. No native video proof. |
| [`act-aloha-policy-onnx-catalogue`](https://huggingface.co/justinchuby/act-aloha-policy-onnx-catalogue/blob/8428f02d75fb029407e3b699dd81842d7e9bb3ff/inference_metadata.yaml) `8428f02d75fb029407e3b699dd81842d7e9bb3ff` | Action-rollout policy graph | v1.0 | Valid; direct ORT CUDA/PyTorch parity and timing record. Generic workflow/native readiness is not separately keyed. |
| [`moshiko-full-duplex-onnx-catalogue`](https://huggingface.co/justinchuby/moshiko-full-duplex-onnx-catalogue/blob/4114e4103f0e9458a2917ecb3048e33b06577891/inference_metadata.yaml) `4114e4103f0e9458a2917ecb3048e33b06577891` | Full-duplex audio workflow with temporal and codec state | v1.0 | Valid; card records direct CUDA duplex output/timing. Generic workflow/native evidence is not separately keyed. |
| [`sensenova-u1.5-8b-mot-onnx-canonical`](https://huggingface.co/justinchuby/sensenova-u1.5-8b-mot-onnx-canonical/blob/a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5/inference_metadata.yaml) `a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5` | Shared-prefix multimodal pixel-flow workflow | v1.0 | Valid; generic workflow ORT CUDA text, image generation, and image-edit evidence. Full autoregressive text decode remains absent. |
| [`onnx-genai-example-gemma4-e2b-assistant`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant/blob/0778733e00713fad71c858553939817a273b7114/inference_metadata.yaml) `0778733e00713fad71c858553939817a273b7114` | Target plus cacheless shared-KV assistant | v1.0 | Valid; direct ORT CUDA drafter parity/assisted evidence. The separately packaged composite carries the current generic real-package run. |
| [`onnx-genai-example-minimax-music3`](https://huggingface.co/justinchuby/onnx-genai-example-minimax-music3/blob/2c7c9f57c42eb7953a01750d51adb724b3181223/inference_metadata.yaml) `2c7c9f57c42eb7953a01750d51adb724b3181223` | Hierarchical text-to-music/audio workflow | v1.0 | Valid; card records component L4 and audio L5 artifacts. Backend and performance claims are not keyed as project conformance records. |
| [`onnx-genai-example-gemma4-26b-a4b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-26b-a4b/blob/e3336a2baea76d6a759fd32347927ca6ec85fbd1/inference_metadata.yaml) `e3336a2baea76d6a759fd32347927ca6ec85fbd1` | MoE hybrid full/sliding decoder | v1.0 | Valid; direct ORT CUDA parity/generation/timing evidence. The card describes an ORT/native deterministic fixture, not an exact keyed portability certificate. |
| [`onnx-genai-example-gemma4-e2b-speculative`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-speculative/blob/6a6d111c877c0b395aff022efa7374de77be2e00/inference_metadata.yaml) `6a6d111c877c0b395aff022efa7374de77be2e00` | Self-contained chained shared-KV speculative workflow | v1.0 | Valid; canonical package bytes remain identical to the exact generic workflow ORT CUDA target-equivalence, acceptance, rejection, rollback, and residency record. No accepted speedup or exact native-package record. |

Fleet-wide findings:

- all 28 packages are workflow-only documents; none relies on retired
  `model.io`;
- all normalize to schema v1.0 (18 declare `v1`; 10 declare `1.0`), which is
  correct because none authors a v1.1 `batch_capacity`, padding provenance, or
  packed ownership contract;
- none uses top-level `required_capabilities`, but every workflow repeats
  derived strings in `manifest.capabilities`;
- none claims encoder/component grouping. Request-aligned tensor layouts alone
  do not permit coalescing;
- ESM-2 and ProtBERT now carry exact producer provenance to the merged Mobius
  source, so their hosted metadata is a generated deployment instance rather
  than an independently maintained correction;
- cache semantics are represented where applicable, including append,
  replacement, sliding/full attention, and one indexed-scatter static-cache
  package; physical paging/tiering remains runtime-owned;
- hosted model cards contain useful direct and workflow execution records, but
  those prose claims are not backend-readiness fields in the metadata and do
  not become performance proof.

The first audit found exactly three files rejected by merged main:
Wav2Vec2 CTC, ESM-2, and ProtBERT still authored the retired
`profiles.*.batch_invariance`. Wav2Vec2 remains the metadata-only,
inline-commented correction in the table. ESM-2 and ProtBERT were subsequently
regenerated after
[Mobius PR #636](https://github.com/onnxruntime/mobius/pull/636) merged as
[`8e3ab921a`](https://github.com/onnxruntime/mobius/commit/8e3ab921a48c0f57eb0b6d24782335c32da3ea4f).
Their canonical republish changed exactly `inference_metadata.yaml` and
`provenance.json`. The provenance records the producer repository/commit and
file hashes, removing the prior divergence between corrected deployed metadata
and its canonical source. The generated canonical YAML remains comment-free,
while the adjacent annotated companion carries review prose. Both preserve
request-aligned layouts, contain no retired batching fields, and omit
`batch_capacity`, so current fail-closed behavior is per item. The cited Mobius
fixtures are deliberately tiny contract examples
(hidden widths 64 rather than the hosted models' 320 and 1024), so producer
parity does not mean those fixture bytes equal the real-model hosted bytes;
provenance instead hashes the exact generated deployment files.

#### Published collection annotations

The collection API resolved
[`justinchuby/onnx-genai-inference-metadata-examples`](https://huggingface.co/collections/justinchuby/onnx-genai-inference-metadata-examples)
to 29 items at publication time: 28 model repositories plus the
[`onnx-genai-inference-metadata-catalogue`](https://huggingface.co/datasets/justinchuby/onnx-genai-inference-metadata-catalogue/tree/6df78ce20485fbe41c807186ec554c18f7575554)
dataset. Every model now publishes `inference_metadata.annotated.yaml` beside
an unchanged canonical `inference_metadata.yaml`, and its README links the
companion. Comments are tailored to the actual model form and cover top-level
sections, policy graphs, profiles, preprocessing, tensor contracts,
cache/state/batching semantics, adapter/speculative contracts, and the absence
of backend-readiness or performance evidence.

The publication inserted 26,657 comment lines across the 28 companions. Parsed
canonical and annotated YAML objects compare equal for all 28 repositories, and
all 56 exact-revision files pass current metadata-only shape validation. The
automated check is
[`scripts/validate_hf_metadata_annotations.py`](../../scripts/validate_hf_metadata_annotations.py).
The catalogue dataset publishes the same check plus a machine-readable
[`annotation_inventory.json`](https://huggingface.co/datasets/justinchuby/onnx-genai-inference-metadata-catalogue/blob/6df78ce20485fbe41c807186ec554c18f7575554/annotation_inventory.json).

Canonical producer provenance remains authoritative:

- eight exhaustive distribution manifests were refreshed for the changed
  README hash and new annotated-file hash, and their canonical metadata entries
  were reverified; producer/source identity and canonical bytes were preserved;
- eight scoped asset provenance files did not cover README/metadata and were
  left unchanged;
- twelve repositories had no `provenance.json`, so publication did not fabricate
  producer identity;
- Wav2Vec2's exhaustive manifest exposed a pre-existing stale canonical
  metadata hash from its earlier correction. The annotation publication fixed
  that hash while leaving source identity and canonical bytes unchanged.

Because every annotation commit left canonical metadata, graphs, and weights
byte-identical, it does not manufacture new runtime evidence. Existing
artifact-backed records remain applicable only to those unchanged bytes and
their originally recorded runtime/provider/result.

| Collection model repo | Pre-annotation revision | Annotated revision | Uploaded files | Verification / provenance |
| --- | --- | --- | --- | --- |
| [`qwen2.5-0.5b-instruct-onnx-genai`](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/blob/56f6964bea748533cd544f6c451a9527307d4e79/inference_metadata.annotated.yaml) | `a61ca2e7e7a41db4c310b6a24479d768d6ab20ae` | `56f6964bea748533cd544f6c451a9527307d4e79` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`qwen3-0.6b-onnx-genai`](https://huggingface.co/justinchuby/qwen3-0.6b-onnx-genai/blob/4913b5e5b485bf62c5254fe41acd91d01db5a21c/inference_metadata.annotated.yaml) | `38714511f57e01df01808b930168459a8e7aa9a3` | `4913b5e5b485bf62c5254fe41acd91d01db5a21c` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`deepseek-r1-distill-qwen-1.5b-onnx-genai`](https://huggingface.co/justinchuby/deepseek-r1-distill-qwen-1.5b-onnx-genai/blob/f48fcc47c9b75bfab8369075855c1d8f5e1a9428/inference_metadata.annotated.yaml) | `1427c4896f798893e58ffec91aef65c34de4503a` | `f48fcc47c9b75bfab8369075855c1d8f5e1a9428` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-example-qwen2-5-0-5b-portable-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-portable-f32/blob/c9958924d55d71965e811e38c9123e39cbae4adc/inference_metadata.annotated.yaml) | `65ef8d35466d402f5bfa5330bb48477e0b330415` | `c9958924d55d71965e811e38c9123e39cbae4adc` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-cuda-gqa-f16/blob/874f622bcce62d9beb01b10a1211a287eedbf3c2/inference_metadata.annotated.yaml) | `ec8046b051a8f11e6d339a7d9d85dd1235053989` | `874f622bcce62d9beb01b10a1211a287eedbf3c2` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-qwen2-5-0-5b-static-cache-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32/blob/33724b837d9ddd2eb5fc078bf5185c847df6df7d/inference_metadata.annotated.yaml) | `bc6b427cbba8db42a1ec3616002a5efab87f6fd0` | `33724b837d9ddd2eb5fc078bf5185c847df6df7d` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32/blob/b180af9a81e249895ff900368fdc0df29d83c516/inference_metadata.annotated.yaml) | `88352734dc2d8c352c58be5450cc5c2dd7521aef` | `b180af9a81e249895ff900368fdc0df29d83c516` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-esm2-t6-8m`](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/1954e4b60aa939a8220c884c3408d06ffa34e494/inference_metadata.annotated.yaml) | `d1e2ada5086f6ef0d1bfffb4099a5292104dbc1b` | `1954e4b60aa939a8220c884c3408d06ffa34e494` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-prot-bert`](https://huggingface.co/justinchuby/onnx-genai-example-prot-bert/blob/83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d/inference_metadata.annotated.yaml) | `17942612a34372dc4191251455ebcb9f854a9db3` | `83d5acc54a7b0eb9f3cd33b668fa8b1fe80e701d` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-whisper-tiny`](https://huggingface.co/justinchuby/onnx-genai-example-whisper-tiny/blob/f1a67225928c21b926bd2ee87940aed3d582b8ef/inference_metadata.annotated.yaml) | `a37efd017b049d697d690824618c0cded5cffa78` | `f1a67225928c21b926bd2ee87940aed3d582b8ef` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md. |
| [`onnx-genai-example-wav2vec2-base-960h-ctc`](https://huggingface.co/justinchuby/onnx-genai-example-wav2vec2-base-960h-ctc/blob/820a7c59ad73e088858230d567395d625c3fac04/inference_metadata.annotated.yaml) | `28480e393ad1b8fa2e0bb6939e5daded02f24014` | `820a7c59ad73e088858230d567395d625c3fac04` | `inference_metadata.annotated.yaml`, `README.md`, `provenance.json` | Parsed equality and current-schema validation passed; hashes verified: inference_metadata.yaml,inference_metadata.annotated.yaml,README.md; stale pre-existing canonical hash corrected. |
| [`onnx-genai-example-qwen2-5-1-5b-lora-selection`](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection/blob/e0f35d893a21fd1d188ea2f5d38d51a0bbeffa43/inference_metadata.annotated.yaml) | `5ca5336b04f0c778f83c0083ee41203dd36961d2` | `e0f35d893a21fd1d188ea2f5d38d51a0bbeffa43` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`onnx-genai-example-mistral-7b-v0-1-sliding-window`](https://huggingface.co/justinchuby/onnx-genai-example-mistral-7b-v0-1-sliding-window/blob/02ea2ae40ef5abad7872c4777a883306bec307e5/inference_metadata.annotated.yaml) | `9d1e328848ab57e665d29ad4acb1182621775143` | `02ea2ae40ef5abad7872c4777a883306bec307e5` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`onnx-genai-example-qwen3-0-6b-eagle3`](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-0-6b-eagle3/blob/10385b7b8f1a3066d4ff15a72ec2194cce324f19/inference_metadata.annotated.yaml) | `0341cd47a8882c1fcbae0840613972321a007371` | `10385b7b8f1a3066d4ff15a72ec2194cce324f19` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-stable-diffusion-bk-sdm-small`](https://huggingface.co/justinchuby/onnx-genai-stable-diffusion-bk-sdm-small/blob/dd7ecd9d50a2210aa796a2efedb5489125f8be37/inference_metadata.annotated.yaml) | `2d30ae2ebfacf5c071693836d70ebd14d8fd84d3` | `dd7ecd9d50a2210aa796a2efedb5489125f8be37` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`onnx-genai-example-qwen-image-edit-2509`](https://huggingface.co/justinchuby/onnx-genai-example-qwen-image-edit-2509/blob/69544173de85fe785b1dfd7e1f5f1795c23bafb1/inference_metadata.annotated.yaml) | `e859aef2289ad02e64812c43fd5e73b5e1c36a2f` | `69544173de85fe785b1dfd7e1f5f1795c23bafb1` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`onnx-genai-cogvideox-2b`](https://huggingface.co/justinchuby/onnx-genai-cogvideox-2b/blob/27e85b66e91a2be33be53ec44d0247a9b232220d/inference_metadata.annotated.yaml) | `29da9103c4517f8026155c0d97e195c26ee56758` | `27e85b66e91a2be33be53ec44d0247a9b232220d` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`pangu-weather-1h-onnx-catalogue`](https://huggingface.co/justinchuby/pangu-weather-1h-onnx-catalogue/blob/36baa7a9b345c3accf6f9e5a0303d9b6960dea34/inference_metadata.annotated.yaml) | `82beb24f24169b88bb0f108e40fc35840d4a8d57` | `36baa7a9b345c3accf6f9e5a0303d9b6960dea34` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`act-aloha-policy-onnx-catalogue`](https://huggingface.co/justinchuby/act-aloha-policy-onnx-catalogue/blob/8428f02d75fb029407e3b699dd81842d7e9bb3ff/inference_metadata.annotated.yaml) | `ebe2b9485d9f2e4ae9d0b181654e2d6d844fda57` | `8428f02d75fb029407e3b699dd81842d7e9bb3ff` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`moshiko-full-duplex-onnx-catalogue`](https://huggingface.co/justinchuby/moshiko-full-duplex-onnx-catalogue/blob/4114e4103f0e9458a2917ecb3048e33b06577891/inference_metadata.annotated.yaml) | `426253a5a5822eb405e4ac214d6895427c64ef0c` | `4114e4103f0e9458a2917ecb3048e33b06577891` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; present; scoped asset manifest unchanged. |
| [`onnx-genai-example-gemma4-e2b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b/blob/a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f/inference_metadata.annotated.yaml) | `79ca25afe326719e4daab79430c90195dfd28f3b` | `a74c3ad0209c4f04251f0c1d48a3796fc63a4a8f` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-example-gemma4-e2b-assistant`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant/blob/0778733e00713fad71c858553939817a273b7114/inference_metadata.annotated.yaml) | `4b6f1533fec1475ade9e3fa3d401ae00a2d7be67` | `0778733e00713fad71c858553939817a273b7114` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-example-gemma4-26b-a4b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-26b-a4b/blob/e3336a2baea76d6a759fd32347927ca6ec85fbd1/inference_metadata.annotated.yaml) | `63e02e455bd835f75b096694ec31d5ad91800299` | `e3336a2baea76d6a759fd32347927ca6ec85fbd1` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-example-minimax-music3`](https://huggingface.co/justinchuby/onnx-genai-example-minimax-music3/blob/2c7c9f57c42eb7953a01750d51adb724b3181223/inference_metadata.annotated.yaml) | `5f95fbbfa01956626fc3170fc90a467666aebdd6` | `2c7c9f57c42eb7953a01750d51adb724b3181223` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`onnx-genai-example-gemma4-e2b-speculative`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-speculative/blob/6a6d111c877c0b395aff022efa7374de77be2e00/inference_metadata.annotated.yaml) | `77a8161bc2a2c9de478dae50307f60e2a0c6beff` | `6a6d111c877c0b395aff022efa7374de77be2e00` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`qwen2.5-14b-instruct-int4-zp-onnx`](https://huggingface.co/justinchuby/qwen2.5-14b-instruct-int4-zp-onnx/blob/2c4311ae4ee87bb7c5976fe26268cb28986fd898/inference_metadata.annotated.yaml) | `753817320d232b0205a7971e8ea25068453fb393` | `2c4311ae4ee87bb7c5976fe26268cb28986fd898` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`sensenova-u1.5-8b-mot-onnx-canonical`](https://huggingface.co/justinchuby/sensenova-u1.5-8b-mot-onnx-canonical/blob/a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5/inference_metadata.annotated.yaml) | `541afaea12e85222766b694cccc30153ea6dd3c1` | `a57ae0a765ac6ec55ddefaa8af12fdf3c9e670d5` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |
| [`Muse-Glimmer-30B-ONNX-INT4-CUDA`](https://huggingface.co/justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA/blob/ddbb787663ca0c5dd05eec93d0514284db7dd787/inference_metadata.annotated.yaml) | `85a1f4b4ac24f1076be51a52e6c934aff4b9e40c` | `ddbb787663ca0c5dd05eec93d0514284db7dd787` | `inference_metadata.annotated.yaml`, `README.md` | Parsed equality and current-schema validation passed; no `provenance.json` existed; none was fabricated. |

#### Hosted-example governance

1. The typed parser, semantic validator, generated schema, and canonical design
   documents are authoritative. A Hub file is a deployed instance, never a
   second schema or capability catalogue.
2. Repository tests, evidence, and documentation **MUST** pin immutable Hub
   revisions. `main` or an unpinned `resolve` URL is not review evidence.
3. A generated hosted example **MUST** carry adjacent provenance that pins the
   canonical producer repository/commit and identifies the exact metadata
   bytes. Hand-edited deployed metadata without a corresponding producer change
   is a source/deployment divergence to remove, not a new source of truth.
4. Publication **MUST** run metadata-only validation against the intended
   runtime revision; artifact-backed execution evidence is a separate gate.
5. A metadata revision change invalidates prior readiness evidence unless the
   evidence key proves the semantic identity is unchanged. Model-card prose
   alone is not such a key.
6. Schema upgrades are need-driven. A v1.0 package does not become stale merely
   because v1.1 exists; it becomes stale when it uses a retired field or needs a
   v1.1 contract it cannot express.
7. A publisher **MUST NOT** add `batch_capacity` merely to replace a removed
   boolean. It must author exact component ports, padding/length provenance,
   ownership, uniform dimensions, and budgets needed for safe grouping.
8. Every hosted example **MUST** publish an adjacent
   `inference_metadata.annotated.yaml` with useful model-specific comments and a
   README link. Automation must prove parsed equality with the canonical YAML
   and validate both files. A serializer-stripped canonical file remains the
   authority; comments never create a second semantic contract.
9. When generic capability lists are removed, the hosted fleet should be
   regenerated from the canonical workflow source in one migration, validated,
   and repinned here; hosted copies must not preserve obsolete fields as a
   compatibility authority.

#### Annotated generated encoder companion

The following compact companion illustrates the common ESM-2/ProtBERT contract.
The hosted repositories now contain their full model-specific annotated files;
this shorter snippet remains a review pattern:

```yaml
# v1 is sufficient because this package deliberately makes no grouping claim.
schema_version: v1

# The profile tells API consumers how to interpret the emitted hidden states.
profiles:
  embedding:
    kind: embedding
    version: "1.0"
    requirement: required
    outputs:
      last_hidden_state: last_hidden_state
    pooling:
      kind: mean  # Pool tokens along the sequence axis after execution.
      axis: 1
      normalize: false

pipeline:
  workflow:
    # These legacy manifest strings mirror structure today; they are not
    # backend-readiness or performance evidence.
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit]

    # Encoding is pure, so retrying it cannot duplicate an external effect.
    effects:
      encode:
        retry: pure
        speculation_safety: {kind: clonable}

    inputs:
      request.input_ids:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, sequence_len]
          # Row identity follows the request. This does not authorize grouping.
          batch_layout: {kind: request_aligned, axis: 0}
        role: {kind: runtime, version: "1.0", role: prompt_tokens}
        source: {kind: request}
        required: true
      request.attention_mask:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, sequence_len]
          batch_layout: {kind: request_aligned, axis: 0}
        # The application supplies authored mask values; the runtime must not
        # infer them from a model-family name.
        role: {kind: opaque}
        source: {kind: application, name: request.attention_mask}
        required: true

    outputs:
      last_hidden_state:
        contract:
          dtype: float32
          rank: 3
          shape: [batch, sequence_len, hidden]
          batch_layout: {kind: request_aligned, axis: 0}
        role: tensor
        stage: post_adapter

    components:
      encoder:
        implementation: {kind: onnx, artifact: model.onnx}
        # batch_capacity is intentionally absent: execute each request alone.
        ports:
          roles:
            input_ids: token_ids
            attention_mask: attention_mask
            last_hidden_state: hidden_states

    # The workflow invokes the authored encoder and publishes its semantic output.
    steps:
      - kind: invoke
        component: encoder
        inputs:
          input_ids: request.input_ids
          attention_mask: request.attention_mask
        outputs:
          last_hidden_state: encoder.last_hidden_state
      - kind: emit
        value: encoder.last_hidden_state
        output: last_hidden_state
        mode: replace
```

Ordered annotated references:

1. the compact decoder policy-graph example in
   [§1.4](#14-policy-graph-means-executable-semantics);
2. the complete [published annotation inventory](#published-collection-annotations)
   and its machine-readable
   [catalogue record](https://huggingface.co/datasets/justinchuby/onnx-genai-inference-metadata-catalogue/blob/6df78ce20485fbe41c807186ec554c18f7575554/annotation_inventory.json);
3. the in-tree [catalogue](../../examples/inference_metadata/catalogue/README.md),
   especially decoder/static cache examples 1 and 18, multimodal/diffusion
   examples 3 and 7–9, adapters/speculation examples 10–11 and 20–24, and
   encoder/task examples 4–5 and 12–13;
4. hosted annotated representatives in this order: baseline
   [decoder/policy graphs](https://huggingface.co/justinchuby/qwen2.5-0.5b-instruct-onnx-genai/blob/56f6964bea748533cd544f6c451a9527307d4e79/inference_metadata.annotated.yaml),
   [static KV](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-0-5b-static-cache-f32/blob/33724b837d9ddd2eb5fc078bf5185c847df6df7d/inference_metadata.annotated.yaml),
   [vision-language](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-5-0-8b-hybrid-vlm-f32/blob/b180af9a81e249895ff900368fdc0df29d83c516/inference_metadata.annotated.yaml),
   [audio encoder-decoder](https://huggingface.co/justinchuby/onnx-genai-example-whisper-tiny/blob/f1a67225928c21b926bd2ee87940aed3d582b8ef/inference_metadata.annotated.yaml),
   [encoder/profile](https://huggingface.co/justinchuby/onnx-genai-example-esm2-t6-8m/blob/1954e4b60aa939a8220c884c3408d06ffa34e494/inference_metadata.annotated.yaml),
   [LoRA](https://huggingface.co/justinchuby/onnx-genai-example-qwen2-5-1-5b-lora-selection/blob/e0f35d893a21fd1d188ea2f5d38d51a0bbeffa43/inference_metadata.annotated.yaml),
   [EAGLE-3](https://huggingface.co/justinchuby/onnx-genai-example-qwen3-0-6b-eagle3/blob/10385b7b8f1a3066d4ff15a72ec2194cce324f19/inference_metadata.annotated.yaml),
   [diffusion](https://huggingface.co/justinchuby/onnx-genai-stable-diffusion-bk-sdm-small/blob/dd7ecd9d50a2210aa796a2efedb5489125f8be37/inference_metadata.annotated.yaml),
   and
   [full-duplex audio](https://huggingface.co/justinchuby/moshiko-full-duplex-onnx-catalogue/blob/4114e4103f0e9458a2917ecb3048e33b06577891/inference_metadata.annotated.yaml);
5. the canonical Mobius
   [ESM-2](https://github.com/onnxruntime/mobius/blob/8e3ab921a48c0f57eb0b6d24782335c32da3ea4f/tests/fixtures/onnx_genai_workflows/esm2_protein_embeddings/inference_metadata.yaml)
   and
   [ProtBERT](https://github.com/onnxruntime/mobius/blob/8e3ab921a48c0f57eb0b6d24782335c32da3ea4f/tests/fixtures/onnx_genai_workflows/protbert_protein_embeddings/inference_metadata.yaml)
   producer fixtures.

## 3. Classification rules

| Question | Classification | Required representation |
| --- | --- | --- |
| Can it be recovered deterministically from the authored graph/workflow/contracts? | Derived | Do not serialize a second answer. Cache the derivation if necessary. |
| Does a wrong answer change model semantics, state correctness, or request interpretation? | Authored requirement | Typed field or versioned contract with validation. Silence means the conservative behavior. |
| Does an operator choose it differently for cost, latency, memory, trust, or risk? | Deployment/QoS policy | Runtime/server configuration outside portable inference metadata. |
| Does a backend planner compute it from the admitted artifact, aliases, provider, shapes, and resources? | Backend-derived execution plan | Generated optimizer plan and diagnostics; do not serialize it as package semantics or label the generated topology as policy. |
| Does the answer depend on artifact bytes, runtime build, EP, device, driver, options, or measurement? | Runtime evidence | Generated record with identity, environment, result, and timestamp. |

Examples:

- `StateUpdate::IndexedScatter`, decoder port roles, graph KV dtype, a rank-3
  position input, and an authored RoPE initializer are derived facts.
- `StateAliasing`, `StateReuse`, rollback bounds, `ComponentBatchCapacity`,
  `EquivalenceClass`, and `SessionMutationPolicy` are authored requirements.
- physical paging, prefix-cache capacity, tiering, connector choice,
  graph-capture enablement/fallback, EP fallback, batch width, and worker
  placement are deployment/QoS policy.
- execution islands, capture segments, and backend buffer-reuse schedules are
  backend-derived execution plans.
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

Capture enablement/fallback is runtime policy, not a model capability. The
capture regions or segments selected after admission are backend-derived plans.

- ORT capture withdraws concurrent `Run` support for the session.
- Shared-buffer batched decode explicitly runs uncaptured because mask width
  changes per step.
- top-level `If`/`Loop`/`Scan` detection is conservative, but an explicit ORT
  capture request currently emits a warning rather than refusing or requiring
  an explicit fallback policy.
- Native capture has its own structural predicates and decline reasons.

A field that says only “graph capture supported” would conceal all four facts.

### 4.13 Worker placement is implemented, policy-selected, and fail closed

[`OrtSessionWorkerCount`](../../crates/onnx-genai-server/src/state.rs) is the
operator's bounded `--ort-session-workers` choice and defaults to one.
[`WorkerPool::reserve_session_placement`](../../crates/onnx-genai-server/src/worker.rs)
places each new session on the healthy worker with the fewest live plus pending
sessions, while stateless turns use the fewest active turns. The resulting
`SessionPlacement { worker, engine_session_id }` is runtime state: later turns
return to that owner, continuous batches never span workers, and a failed
worker's sessions are not silently migrated.

Multiple workers are not a portable model capability. They are admitted only
for a contracted single-decoder ORT engine whose resolved session reports
concurrent `Run`; native, composite, speculative, external-KV, capture, and
single-flight configurations fail closed rather than being silently reduced to
one worker. Same-session turns remain protected by the routing lease even when
distinct sessions execute on different workers. See
[`two_ort_workers_run_distinct_sessions_concurrently_with_colliding_local_ids`](../../crates/onnx-genai-server/src/driver.rs)
and
[`multiple_ort_workers_fail_closed_for_native_decode`](../../crates/onnx-genai-server/src/tests.rs).

### 4.14 Functional workflow execution is not a performance claim

The universal interpreter has recorded functional execution for named
fixtures/artifacts.
[`pure_policy_chain_lowers_to_one_execution_island`](../../crates/onnx-genai-engine/tests/workflow_policy_e2e.rs)
records two CPU ORT island runs and their diagnostics. That does not establish
CUDA island readiness: the CUDA allocator test may return without executing
when CUDA is unavailable. Merged #2137 additionally records that inferred
dynamic batch axes fail before island or backend execution when
`batch_capacity` is absent; see
[`inferred_dynamic_axis_zero_fails_before_island_or_backend_execution`](../../crates/onnx-genai-engine/src/pipeline/islands.rs).
The current published CUDA synthetic measurements fail at least one acceptance
criterion: decoder-policy throughput is `0.903` of the direct composite and
min-p warm TTFT regresses. The record explicitly does not cover production KV,
per-row serving, or the overridable sampler path; see
[Current measured baseline](../benchmarks/2026-08-21-mobius-workflow-conformance.md#current-measured-baseline).

Selected chained/shared-KV speculation has exact and fixture correctness
records. MTP's full generation test is ignored, and the EAGLE-3 real-package
gate can return success without execution when its artifact variable is absent.
Neither receives a generic execution claim. The speedup test is also
ignored/environment-gated. `distribution_preserving` is a
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
   surface, task profiles, pooling/decoding semantics, and component-scoped
   grouping equivalence through `ComponentBatchCapacity`.
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
- graph-capture enablement/fallback and stable-buffer strategy;
- adapter caching, normalization, bucketing, and fallback permission;
- session limits, TTL policy, and observability.

Policy may choose less than the package permits. It may not choose more.

### 5.4 Backend-derived execution plans

Keep automatic optimizer output out of both package metadata and deployment
policy:

- ORT execution islands and their `IoBinding` groups;
- CUDA-graph capture segmentation;
- buffer alias/reuse schedules chosen after backend inspection; and
- native or ORT fallback boundaries chosen from actual operator coverage.

[`plan_execution_islands`](../../crates/onnx-genai-engine/src/pipeline/islands.rs)
derives islands from the compiled workflow, model sessions, aliasable outputs,
and speculative live values.
[`WorkflowRuntime`](../../crates/onnx-genai-engine/src/pipeline/mod.rs) describes
the result as the ORT `IoBinding`/CUDA-graph optimization and leaves native
components unfused.
[`ExecutionIslandDiagnostic`](../../crates/onnx-genai-engine/src/pipeline/islands.rs)
reports the generated result; it is not an authored enablement claim. Resource
admission may decline fusion and a deployment knob may permit fallback, but
neither fact turns the generated island topology into deployment policy.

### 5.5 Evidence record

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
   row scope, and update discipline. `ComponentBatchCapacity` is the sole
   authored grouping-permission and semantic-invariance authority. Its
   presence asserts grouped component execution is equivalent to solo
   execution within the declared bounds; absence requires per-item execution.
   Whether a service actually groups requests, and its target/delay below the
   bound, remains deployment policy.
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

The columns intentionally do not overload “support”:

- **Represented/validated:** `Yes`, `Partial`, `No`, or `N/A`.
- **Implementation:** `Yes` means backend code exists for the stated scope;
  `Partial` means a narrower path exists; `No` means it does not. Source code or
  a test body proves only implementation.
- **Recorded execution:** `Recorded artifact` names an exact package/runtime
  result; `Recorded fixture` names a non-skipped backend fixture and a recorded
  passing run; `Recorded refusal` is a negative execution/admission result;
  `No record` means no accepted non-skipped execution record was found; `N/A`
  means the question is not backend execution.
- **Performance evidence:** `Pass (scope)` or `Fail (scope)` requires a
  controlled record with a stated gate. `No accepted record` is neither a
  performance failure nor a functional failure.

`Yes` is therefore never an execution-evidence status, and an ignored,
environment-gated, or early-returning test cannot create one. Every positive
recorded-execution cell below names its evidence and is intentionally narrower
than the implementation column.

### 7.1 Workloads and composition

| Form | Rep. | Val. | ORT impl. | ORT recorded execution | Native impl. | Native recorded execution | Performance evidence | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard autoregressive decoder | Yes | Yes | Yes | Recorded artifact | Yes | Recorded artifact | Pass (named decoder/hardware) | Canonical `decoder_io()` lowering plus the exact ORT/native runs in [the CUDA comparison record](../benchmarks/2026-07-23-ort-vs-native-cuda.md). No family-wide claim. |
| Encoder-only embedding | Yes | Yes | Yes | Recorded artifact | No | No record | No accepted record | Exact hosted CPU workflow results are inventoried in [§2.3](#23-hosted-hugging-face-examples). `Engine::embed_with_options` is ORT-only and does not dispatch from every `TaskProfile`. |
| Encoder-only reranking/classification/reward | Yes | Yes | No | No record | No | No record | No accepted record | Profile kinds validate, but no profile executor or non-skipped backend result was found. |
| Encoder-only CTC transcription | Yes | Yes | No | No record | No | No record | No accepted record | CTC decoding/vocabulary validate; the refreshed Wav2Vec2 package proves schema validity, not an engine CTC interpreter. |
| Encoder-decoder / cross-attention | Yes | Yes | Yes | Recorded artifact | Partial | No record | No accepted record | Exact hosted Whisper/translation CPU workflow results are listed in [§2.3](#23-hosted-hugging-face-examples). No accepted native end-to-end record was found. |
| Image multimodal preprocessing + VLM | Yes | Yes | Yes | Recorded artifact | Partial | No record | No accepted record | Hosted Qwen vision packages record ORT execution; generic native component coverage is not a real image-package result. |
| Encoded-video preprocessing | Yes | Yes | Partial | No record | Partial | No record | No accepted record | Container decode/temporal sampling still fails closed. Ordered frame preprocessing is not an ORT/native encoder execution record. |
| Video-producing workflow/diffusion | Yes | Yes | Yes | Recorded fixture | Partial | No record | No accepted record | [`mobius_video_diffusion_workflow_publishes_causal_temporal_chunks`](../../crates/onnx-genai-engine/tests/onnx_genai_workflow_conformance.rs) is the maintained ORT fixture. No native video record was found. |
| Audio preprocessing + multimodal/audio workflow | Yes | Yes | Yes | Recorded artifact | Partial | No record | No accepted record | Exact hosted audio workflow results are listed in [§2.3](#23-hosted-hugging-face-examples); no general native real-audio record was found. |
| Image diffusion/workflow | Yes | Yes | Yes | Recorded fixture | Yes | Recorded fixture | No accepted record | ORT: [`mobius_euler_diffusion_workflow_executes_complete_path`](../../crates/onnx-genai-engine/tests/onnx_genai_workflow_conformance.rs). Native: [`native_runs_diffusion_loop_package`](../../crates/onnx-genai-engine/tests/native_workflow_smoke.rs) and [`diffusion_loop_parity`](../../crates/onnx-genai-engine/tests/native_workflow_parity.rs). |
| Nested/general workflow IR | Yes | Yes | Yes | Recorded fixture | Partial | Recorded fixture (subset) | No accepted record | ORT interpreter conformance covers nested workflow fixtures; native records selected workflows, not a universal corpus. |
| Legacy external draft-model speculation | Partial | Partial | Yes | No record | No | No record | No accepted record | ORT code exists and native request admission rejects this mode. No accepted non-skipped artifact result or speedup record was found. |
| Workflow-native chained/shared-KV speculation | Yes | Yes | Yes | Recorded artifact | Yes | Recorded fixture | No accepted record | The exact Gemma4 composite in [§2.3](#23-hosted-hugging-face-examples) records ORT equivalence/acceptance/rejection. [`chained_speculative_proposal_parity`](../../crates/onnx-genai-engine/tests/native_workflow_parity.rs) records only the hermetic native scope. |
| MTP | Yes | Yes | Yes | No record | Partial | No record | No accepted record | Implementation exists, but [`mtp_speculative_generation_matches_plain_greedy`](../../crates/onnx-genai-engine/tests/mtp_full.rs) is `#[ignore]`; the normally run test checks only a configuration literal. A native target with an ORT MTP head is not pure-native execution. |
| EAGLE-3 | Yes | Yes | Yes | No record | No | No record | No accepted record | Implementation exists, but [`real_chained_proposer_matches_target_and_accepts_and_rejects`](../../crates/onnx-genai-engine/tests/chained_proposer_real.rs) returns success without executing when its package variable is absent. Native request admission rejects EAGLE-3. |
| LoRA/adapters | Yes | Yes | Partial | Recorded artifact (portable overlay) | No | No record | No accepted record | The exact LoRA package in [§2.3](#23-hosted-hugging-face-examples) records ORT CUDA adapter output for the portable overlay path. Heterogeneous logical rows ran separately, and represented PEFT/safetensors/ORT formats are not accelerated batching evidence. |

### 7.2 Batching and cache forms

| Form | Rep. | Val. | ORT impl. | ORT recorded execution | Native impl. | Native recorded execution | Performance evidence | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-item decoder | Yes | Yes | Yes | Recorded artifact | Yes | Recorded artifact | Pass (named decoder/hardware) | Baseline evidence is the same exact decoder record cited in §7.1. |
| Continuous/dynamic decoder batching | Yes | Yes | Yes | Recorded fixture | Yes | Recorded artifact | Pass (named native batches) | [`a_continuous_batch_row_stops_on_a_non_first_declared_end_token`](../../crates/onnx-genai-engine/tests/authored_iteration_e2e.rs) executes ORT width 2. Native grouping/capture measurements are recorded in [batch decode segmentation](../benchmarks/2026-08-19-batch-decode-mge2-capture-segmentation.md). `batch_capacity` permits grouping; it does not mandate it. |
| Component/encoder grouping capacity | Yes | Yes | Partial | Recorded fixture (request-aligned component only) | Partial | No record | No accepted record | `ComponentBatchCapacity` is the only authored grouping-equivalence authority. Merged #2137 adds packed/request-aligned admission and [`group_workflow_component_inputs`](../../crates/onnx-genai-engine/src/engine/workflow_api.rs), but that API explicitly leaves grouped-payload materialization to a backend packer. [`workflow_executes_masked_and_speculative_policy_artifacts`](../../crates/onnx-genai-engine/tests/workflow_policy_e2e.rs) records a width-2 request-aligned ONNX component; no packed encoder artifact, cross-request scheduler, or generalized solo-versus-grouped parity record was found. |
| Dynamic append KV | Yes | Yes | Yes | Recorded artifact | Yes | Recorded artifact | Pass (named decoder/hardware) | Exact standard-decoder records exercise dynamic KV for their named packages only. |
| Static indexed KV | Yes | Yes | Yes | Recorded artifact | Yes | Recorded fixture | No accepted record | Exact static hosted packages are listed in [§2.3](#23-hosted-hugging-face-examples); [`native_runs_static_cache_ar_package`](../../crates/onnx-genai-engine/tests/native_workflow_smoke.rs) records the hermetic native path. |
| Physical paged KV | N/A | N/A | Partial | No record | Partial | No record | No accepted record | Physical paging is policy. [`GQA_KV_MATERIALIZATION_DESIGN.md`](../memory/GQA_KV_MATERIALIZATION_DESIGN.md) explicitly defers live-session wiring and CUDA GQA Tier B; page-pool primitives/unit tests are not live decoder execution. |
| Prefix reuse | Yes | Yes | Yes | No record | Yes | No record | No accepted record | Semantic legality and stores exist, but no accepted backend-specific live artifact/result was found. Capacity and eviction remain policy. |
| Quantized host/paged KV mirror | N/A | N/A | Partial | No record | Partial | No record | No accepted record | Codecs and host page-store tests do not prove live decoder integration. [`local_tiered.rs`](../../crates/onnx-genai-kv/src/local_tiered.rs) says actual stored-payload FP8 compression is deferred. |
| Graph-visible FP8 KV | Yes | Yes | No | Recorded refusal | No | No record | N/A | Metadata can validate while ORT CUDA rejects the required GQA FP8 path; this is the canonical representable-but-not-ready case. |
| Tiered KV | N/A | N/A | Partial | No record | Partial | No record | No accepted record | Tiering is runtime policy. Connector/local-store code exists, but live end-to-end tiered decoder execution is not established; stored-payload FP8 remains deferred. |
| External KV connector | N/A | N/A | Partial | No record | No | No record | No accepted record | Connector selection is runtime policy. [`connector_bridge.rs`](../../crates/onnx-genai-engine/src/connector_bridge.rs) materially injects only compatible byte-exact f32 `ZeroCopyRebind`; other paths can report opportunity without shortening prefill. |

### 7.3 Positions, backends, capture, and concurrency

| Form | Rep. | Val. | ORT impl. | ORT recorded execution | Native impl. | Native recorded execution | Performance evidence | Evidence and limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Position input absent / graph-internal | Yes | Yes | Yes | No record | Yes | No record | No accepted record | Absence is derivable; internal semantics remain opaque. A generic decoder run does not prove which internal position algorithm was used. |
| Rank-2 position IDs | Yes | Yes | Yes | Recorded artifact | Yes | Recorded artifact | No accepted position-specific record | Exact standard-decoder records cover named rank-2 artifacts. |
| Rank-3 multi-coordinate position IDs | Yes | Yes | Yes | Recorded artifact | Partial | No record | No accepted record | Exact Qwen3.5-VL hosted metadata/execution is in [§2.3](#23-hosted-hugging-face-examples). The real native gate can skip on a known model mismatch, so it is not accepted evidence. |
| Authored sin/cos RoPE cache inputs | Yes | Yes | Yes | No record | Yes | No record | No accepted record | Role-derived graph edges are implementation capability; no specific non-skipped artifact result was found. |
| ORT/native interchangeability | Partial | Yes | Partial | No paired record | Partial | No paired record | No accepted record | The backend domains overlap but are unequal. Readiness requires a paired certificate for the exact artifact, not two optimistic implementation flags. |
| ORT CUDA graph capture | N/A | N/A | Yes | Recorded artifact | N/A | N/A | Pass (named Qwen/CUDA) | [The ORT/native CUDA record](../benchmarks/2026-07-23-ort-vs-native-cuda.md) names capture success and rejection by artifact. Eligibility is provider/shape/allocation specific. |
| Native graph capture / single-flight | N/A | N/A | N/A | N/A | Yes | Recorded artifact | Pass (named native decoders) | The same record measures native capture; mutable captured buffers require one in-flight owner. |
| Backend-derived execution islands | N/A | N/A | Yes | Recorded fixture (CPU ORT) | No | No record | No accepted record | [`pure_policy_chain_lowers_to_one_execution_island`](../../crates/onnx-genai-engine/tests/workflow_policy_e2e.rs) records two CPU ORT island runs. [`inferred_dynamic_axis_zero_fails_before_island_or_backend_execution`](../../crates/onnx-genai-engine/src/pipeline/islands.rs) records fail-closed batched admission before island/backend execution. CUDA tests may still return without executing, so no CUDA island record is claimed. |
| Concurrent ORT `Session::Run` | N/A | N/A | Yes | Recorded fixture | N/A | N/A | No accepted performance record | [`a_concurrently_runnable_session_can_actually_be_run_from_two_threads`](../../crates/onnx-genai-ort/tests/session_thread_contract.rs) is non-skipped and executes one admitted session from two threads. |
| Same-session overlapping turns | Yes | Yes | N/A | N/A | N/A | N/A | N/A | Shared routing-layer [`SessionLeases`](../../crates/onnx-genai-server/src/lease.rs) serialize or reject overlap before backend execution unless isolated state is proven; a backend concurrency flag cannot override this. |
| Worker placement / cross-session parallelism | N/A | N/A | Yes | Recorded fixture (W=2 CPU ORT) | No | Recorded refusal | No accepted performance record | Worker count is deployment policy and defaults to one. [`two_ort_workers_run_distinct_sessions_concurrently_with_colliding_local_ids`](../../crates/onnx-genai-server/src/driver.rs) records deterministic placement and simultaneous execution; [`multiple_ort_workers_fail_closed_for_native_decode`](../../crates/onnx-genai-server/src/tests.rs) records native refusal. Sessions never migrate and same-session turns remain single-flight. |
| Workflow policy interpreter/sampler | Yes | Yes | Yes | Recorded artifact | Partial | Recorded fixture (subset) | Fail (current scoped gate) | Functional policy-graph execution does not establish an optimized sampler. [The current measured baseline](../benchmarks/2026-08-21-mobius-workflow-conformance.md#current-measured-baseline) records `0.903x` direct-composite throughput and min-p TTFT regression, and excludes production KV/per-row serving. |

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
| `model.runtime_configurable.continuous_batching` | Already retired; preserve targeted rejection | [`reject_retired_batching_hints`](../../crates/onnx-genai-metadata/src/parser.rs) explains that `ComponentBatchCapacity` is the sole authored grouping permission/equivalence assertion and runtime policy decides whether to group. |
| `profiles.*.batch_invariance` | Already retired; preserve targeted rejection | [`reject_retired_batching_hints`](../../crates/onnx-genai-metadata/src/parser.rs) directs authors to component-scoped `batch_capacity`; absence means per-item execution. |
| Built-in `continuous_batching` capability string | Already retired; preserve targeted rejection | Merged main rejects it as an optimization, not a correctness requirement; the parser names the typed migration instead of emitting a generic unknown-capability error. |
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
   view. Preserve targeted migration diagnostics for fields already retired;
   do not regress them to generic unknown-field errors.
3. **Separate policy:** move runtime-configurable, hardware-placement, adapter
   cache/planning, and fallback choices to runtime/server configuration.
4. **Add evidence keys and gates:** persist backend admission and conformance
   results; make auto-selection consume exact matching evidence or probe by
   loading.
5. **Delete remaining redundant schema fields and strings:** update parser,
   validation, generated schema, fixtures, and canonical docs in one dedicated
   follow-up PR. The batching fields above are already retired and are not work
   for this stage. No compatibility aliases.
6. **Republish hosted instances:** regenerate the pinned Hub fleet from the
   authoritative typed source, validate exact uploaded bytes, and update this
   inventory. Do not preserve obsolete hosted fields as aliases.
7. **Tighten claims:** make status APIs and documentation report represented,
   validated, implemented, recorded execution, and performance evidence
   separately.

Stages 1–4 can land without editing the batching schema or its canonical design
document. Stage 5 remains intentionally isolated from this proposal and from
the already-merged batching work.

## 10. Recommended acceptance tests

1. **Single-authority derivation:** mutate one workflow fact at a time and prove
   the derived view changes; there is no second writable answer.
2. **Actionable retired-field diagnostics:** preserve
   [`retired_profile_batch_invariance_explains_the_component_contract_migration`,
   `retired_model_batching_hint_explains_derived_feasibility_and_policy`, and
   `retired_continuous_batching_capability_is_not_a_correctness_requirement`](../../crates/onnx-genai-metadata/tests/encoder_batching.rs).
   Old batching spellings must fail with the exact migration authority, not a
   generic unknown-field error or silent translation.
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
9. **Batching corpus:** prove absence of `batch_capacity` forces per-item
   execution; prove presence permits, but never mandates, grouped execution
   within its bounds; then cover ORT static/shared-buffer, native pinned batch,
   and encoder grouping only after the component/backend/artifact triple
   passes.
10. **Capture/concurrency matrix:** capture on/off × provider concurrent-run
    declaration × control flow × stable/dynamic shape. Every decline names the
    exact predicate.
11. **Session placement:** exclusive and copy-on-write mutation policies,
    same-session racing turns, default `W = 1`, opt-in ORT `W > 1`,
    deterministic least-loaded placement, unsupported-backend refusal, worker
    failure without silent migration, and state continuity.
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
13. What durable run record turns a non-skipped repository test or hosted model
    card into accepted execution evidence, and how must environment-gated tests
    report “not run” so they cannot appear as passes?
14. Which execution-island enable/fallback controls, if any, should be exposed
    as deployment policy without exposing or serializing the generated island
    topology?
15. What aggregate-throughput and memory-accounting gate must an opt-in
    `W > 1` ORT worker deployment pass before it may claim performance benefit,
    separately from the recorded two-worker functional result?

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
5. [`decoder_abi.rs`](../../crates/onnx-genai-metadata/src/decoder_abi.rs),
   [`validation.rs`](../../crates/onnx-genai-metadata/src/validation.rs),
   [`parser.rs`](../../crates/onnx-genai-metadata/src/parser.rs), and
   [`encoder_batching.rs`](../../crates/onnx-genai-metadata/tests/encoder_batching.rs)
   — derivation, validation, and targeted diagnostics for retired batching
   spellings.
6. [`engine/load.rs`](../../crates/onnx-genai-engine/src/engine/load.rs) and
   [`engine/decode_backend.rs`](../../crates/onnx-genai-engine/src/engine/decode_backend.rs)
   — actual admission, fallback, and backend selection.
7. [`pipeline/mod.rs`](../../crates/onnx-genai-engine/src/pipeline/mod.rs),
   [`pipeline/workflow.rs`](../../crates/onnx-genai-engine/src/pipeline/workflow.rs),
   [`pipeline/native_component.rs`](../../crates/onnx-genai-engine/src/pipeline/native_component.rs),
   and
   [`pipeline/islands.rs`](../../crates/onnx-genai-engine/src/pipeline/islands.rs)
   — what the interpreter and both component backends implement, and which ORT
   execution-island plan is derived automatically.
8. [`batched.rs`](../../crates/onnx-genai-engine/src/batched.rs),
   [`pipeline/batching.rs`](../../crates/onnx-genai-engine/src/pipeline/batching.rs),
   [`engine/workflow_api.rs`](../../crates/onnx-genai-engine/src/engine/workflow_api.rs),
   [`connector_bridge.rs`](../../crates/onnx-genai-engine/src/connector_bridge.rs),
   and [`onnx-genai-kv`](../../crates/onnx-genai-kv/src/lib.rs) — batching and
   cache reality, including the boundary between admission/grouping plans and
   backend payload materialization.
9. [`onnx-genai-ort/session/mod.rs`](../../crates/onnx-genai-ort/src/session/mod.rs)
   — provider resolution, capture, concurrent-run evidence, and thread safety.
10. [`server/worker.rs`](../../crates/onnx-genai-server/src/worker.rs),
    [`server/lease.rs`](../../crates/onnx-genai-server/src/lease.rs), and
    [`server/driver.rs`](../../crates/onnx-genai-server/src/driver.rs) — current
    opt-in ORT worker pool, deterministic placement, owner-thread routing,
    fail-closed backend gate, and same-session single-flight.
11. [ENCODER_BATCHING.md](ENCODER_BATCHING.md),
    [CHAINED_SPECULATIVE_EVIDENCE.md](CHAINED_SPECULATIVE_EVIDENCE.md),
    [WORKFLOW_PERFORMANCE_CONFORMANCE.md](../WORKFLOW_PERFORMANCE_CONFORMANCE.md),
    and
    [the dated workflow evidence](../benchmarks/2026-08-21-mobius-workflow-conformance.md)
    — represented versus executed versus measured status.
12. The pinned [hosted-example inventory](#23-hosted-hugging-face-examples),
    [publication revisions](#published-collection-annotations), collection
    validator, model-specific annotated sidecars, exhaustive/scoped/absent
    provenance cases, and Mobius source fixtures — deployed instances versus
    their authoritative producer, schema, and evidence.

### Invariants to scrutinize

- Is every structural answer derived from one authored source?
- Does every remaining authored field change correctness rather than merely
  deployment preference?
- Can an unsupported requirement ever degrade to a warning or silent fallback?
- Is readiness attached to exact artifact/runtime/provider identity?
- Does a status distinguish representation, validation, implementation,
  recorded execution, and performance?
- Does any cache field confuse model semantics with storage policy?
- Does capture or concurrent execution change the admission result?
- Is `batch_capacity` the only grouping-permission/equivalence authority, with
  absence forcing per-item execution?
- Does a batching claim name the exact artifact/backend/component combination?
- Is direct execution of an already grouped request distinguished from
  automatic cross-request scheduling and backend payload packing?
- Is an execution island treated as backend-derived optimizer output rather
  than package semantics or deployment policy?
- Can any ignored, environment-gated, or early-returning test be mistaken for
  recorded execution?
- Could an adapter or speculative fallback silently change output semantics?
- Is every hosted example pinned, validated at that revision, and prevented from
  becoming a schema or readiness authority?
- Does every collection model publish a README-linked annotated sidecar whose
  parsed object equals the unchanged canonical metadata?

### Short review checklist

- [ ] No generic capability boolean/list duplicates typed structure.
- [ ] No deployment choice is stored as portable model truth.
- [ ] Package policy graphs, deployment/QoS knobs, and backend-derived plans are distinct.
- [ ] Unsupported required behavior fails closed with an actionable source path.
- [ ] ORT/native implementation and recorded execution are independently labeled.
- [ ] Functional and performance evidence are separately labeled.
- [ ] Retired batching spellings retain targeted migration diagnostics.
- [ ] Cache, position, batching, adapter, speculation, and session forms are all covered.
- [ ] Migration deletions have one replacement authority and no compatibility alias.
- [ ] Acceptance tests include negative and decline paths, not only successful execution.
- [ ] Hosted revisions, producer provenance, comments/annotated references, and evidence labels match exact bytes.
- [ ] Canonical and annotated hosted YAML parse equally and both pass the current validator.
