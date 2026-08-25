# Inference metadata specification

Status: **Normative**

This is the normative specification of the portable inference-metadata contract.
It defines what a model package states, what a runtime decides, what a caller
supplies, and what a validator must reject. It supersedes contrary statements
elsewhere in this repository.

Requirement keywords (**MUST**, **MUST NOT**, **SHOULD**, **MAY**) are used in
the RFC 2119 sense. Every **MUST** in this document is either enforced by the
semantic validator in `crates/onnx-genai-metadata/src/validation.rs` or is a
structural property of the schema in `crates/onnx-genai-metadata/src/schema/`.
Section [Conformance](#20-conformance) maps each requirement to its test.

---

## 1. Goals and non-goals

### 1.1 Goals

1. **One executable description.** A package states everything a runtime needs
   to execute it correctly, and nothing about how to deploy it.
2. **Structural, not nominal.** Behavior follows from typed structure — values,
   ports, control flow, state kinds — never from a model-family name, an
   architecture string, or a filename convention.
3. **Fail-closed.** A document that a reader does not fully understand is
   rejected, not approximated. Silence never grants a capability.
4. **Correct under batching.** A package remains correct when the runtime
   batches, reorders, compacts, forks, or rewinds requests, without the package
   ever naming a request, a row, or a slot.
5. **Stable under substitution.** A runtime may replace a component with an
   equivalent implementation. The package states which equivalences it accepts,
   so a substitution can never silently change what a caller agreed to.

### 1.2 Non-goals

1. **Deployment policy.** Memory budgets, device placement, execution providers,
   allocators, paging, tiering, quality of service, and deadlines are not
   metadata. See [§5](#5-ownership-layers-in-the-schema).
2. **Backward compatibility.** There is no export path to `genai_config.json`
   and no reverse synthesizer. See [§17](#17-legacy-import).
3. **Integrity and trust.** Signing, provenance, and artifact attestation belong
   to the distribution layer. Metadata carries a semantic *identity* hash, which
   is not an integrity claim. See [§4.4](#44-semantic-identity).
4. **Benchmark-derived cost models.** Metadata exposes static geometry, never
   measured throughput, admission predictions, or tuned heuristics.
5. **A portable cross-runtime cache wire format.** See [§16](#16-distributed-execution).
6. **Standardized diagnostics.** Determinism levels, profiling surfaces, and
   implementation-selection reports are runtime and API concerns.

---

## 2. Terminology

| Term | Meaning |
| --- | --- |
| **Package** | The distributed artifact set plus one metadata document. |
| **Metadata document** | One `inference_metadata.yaml` (or `.json`) validated against `schema/inference_metadata.schema.json`. |
| **Workflow** | The typed structural IR under `pipeline.workflow` that describes execution. |
| **Component** | A named unit of computation with typed ports: an ONNX graph, a native implementation, or a runtime binding. |
| **Contract** | A versioned semantic ABI a component claims to implement. |
| **Value** | A named SSA tensor produced once and read many times. |
| **State cell** | A named mutable location with declared scope, recurrence, and lifetime. |
| **State group** | A runtime-owned collection of state ports sharing one semantic kind and geometry. |
| **Row** | One request's position along a batched tensor's request axis. Rows are positional and unnamed. |
| **Request** | A caller-visible unit of work. Requests are runtime-owned; metadata never names one. |
| **Effect domain** | A named linear resource threaded through mutating operations. |
| **Profile** | One executable task view sharing the package's common facts. |
| **Runtime** | The implementation that loads, plans, and executes a package. |
| **Caller** | The application issuing requests to the runtime. |

---

## 3. Ownership layers

Every fact belongs to exactly one layer. A fact in the wrong layer is a defect,
not a convenience.

### 3.1 Package (metadata)

The package states **what must be true for execution to be correct**:

- component semantics, typed ports, and declared equivalence classes;
- workflow structure and control flow;
- state kinds, geometry, graph ABI constraints, and reuse semantics;
- tokenizer, vocabulary, special-token, and constraint-language facts;
- adapter target manifests and artifact bindings;
- generation defaults and the structural override surface;
- legal sharding and replication facts;
- correctness dependencies for cache reuse.

### 3.2 Runtime

The runtime decides **how to execute it here**:

- memory budget, placement, execution provider, transfers;
- state allocator, paging, storage layout, compaction algorithm;
- execution islands, graph capture, memory plans;
- request and sequence tables, row identity, slot reuse, epochs;
- cache key construction, hashing, salting, tenant namespacing;
- adapter lifecycle, caching, budgeting, eviction;
- speculative proposal width, tree shape, scheduling, and enablement;
- distributed transfer protocols, QoS, tiering, deadlines;
- diagnostics, determinism levels, and trust policy.

### 3.3 Caller

The caller supplies **request data**: prompts, images, audio, grammars, JSON
schemas, adapter selections, and overrides of generation fields the package
structurally exposes ([§14](#14-generation)).

### 3.4 Distribution layer

Distribution manifests own artifact hashes, signatures, provenance,
attestation, mirroring, and byte-level integrity. Runtime inference metadata
contains no artifact hashes or fingerprints: replacing compatible component,
tokenizer, or adapter bytes must not require rewriting the semantic contract.

---

## 4. Schema model

### 4.1 Document shape

```yaml
schema_version: v1          # required
required_capabilities: []   # capabilities the reader must implement
model: {...}                # package-wide baked facts (attention geometry, vocab, MoE, sharding)
quantization: {...}         # model-weight quantization intent
pipeline: {workflow: {...}} # the executable workflow: the only graph ABI
adapters: {...}             # LoRA target manifest and artifact bindings
preprocessing: {...}        # typed preprocessing contracts
package: {...}              # tokenizer and constraint-language facts
generation: {...}           # authoritative defaults + override surface
profiles: {...}             # task profiles
speculative: {...}          # proposer/target compatibility facts
hardware_requirements: {...}
```

`schema_version` versions the workflow syntax together with the rest of the
document. The workflow manifest therefore carries only facts that are not
already authoritative elsewhere, such as adapter ABI versions and declared
capabilities. It does not repeat a workflow `ir_version`, and it does not copy
ONNX opset imports: every ONNX artifact carries its own exact domain/version
map, including the case where different components use different opsets.

`pipeline.workflow` is the **sole serialized expression of a package's
executable graph ABI**, for every package, including one that ships a single
ONNX file. `model:` carries facts that are true of the package rather than of
any one graph — attention geometry, vocabulary and length limits, MoE structure,
sharding. It does **not** carry port names. See §4.1a.

### 4.1a One canonical graph ABI

A package must not be able to say what its decode step looks like twice.

Component ports, `Invoke` input/output bindings, workflow state cells and
`state_service` groups already describe every port a runtime touches, because
they are what the workflow engine executes. A second serialized block beside them
would be a second writable answer to the same question, and nothing forces the
two to agree: a runtime reading one never learns that the other said something
else. Two accepted representations is a defect independent of their contents —
which is why there is now exactly one, for every package shape including a bare
decoder.

So the ABI is written once, in the workflow, and **derived** where a runtime
needs it flattened:

| Resolved fact | Canonical source |
| --- | --- |
| token / embeds / mask / position inputs | `components.<c>.ports.roles` |
| logits / hidden outputs | `components.<c>.ports.roles` |
| encoder-hidden / audio inputs | `components.<c>.ports.roles` |
| KV input/output pairs, per layer | self-attention `state_service` groups |
| cross-attention KV pairs | `cross_attention` groups |
| fixed loop-carried state pairs | `recurrent` groups |
| `aliasing`, KV layout | the owning group |
| fixed-capacity scatter ABI | `StateUpdate::IndexedScatter` |
| KV ownership | presence of an owning group |

`InferenceMetadata::decoder_io()` performs that derivation. It is a *lowering*,
not a second source: the result is computed, never serialized, and there is no
key a producer can write to make it disagree with the workflow. The optimized
single-graph decode path is therefore a **recognizer** over the canonical
workflow rather than a path with its own serialized truth, and a bare single
ONNX decoder and a composite package are driven from one representation.

Only one fact could not be recovered from an existing field: what a component
*does* with a value an invoke binds to it. A binding records which value reaches
a port, not whether that port is tokens, a mask, or logits, and guessing from a
port's spelling is exactly the name-matching this schema refuses everywhere
else. `ports.roles` states it, with an architecture-neutral vocabulary that
describes the port and never the model family exposing it.

State ports use the same separation of graph facts from semantic facts. The
artifact owns every individual tensor's geometry. The state group supplies
`role: key|value|combined` and numeric `layer`, because neither semantic half
nor layer ordering is recoverable from shape or a producer-chosen label.
`layer` pairs and orders aliases; it never asserts equal geometry. Different
layers may expose different KV head counts, and the key and value aliases of
one layer may themselves have different head counts or head dimensions without
requiring a different metadata vocabulary.

**`model.io` is removed.** There is no schema field for it and no code path
that reads one. A single decoder declares its graph ABI exactly where every
other package does: `pipeline.workflow.components.<c>.ports` with `ports.roles`,
and the `state_service` group that owns its cache. A document still carrying the
retired block is refused at load with an error naming the offline conversion, so
the failure is actionable rather than a puzzled "declares no workflow". §18.1
records the completed removal.

### 4.2 Strictness

The core is strict. Every core structure uses `deny_unknown_fields`; an unknown
core field **MUST** fail validation. There is no "ignore what you do not know"
mode for the core.

### 4.3 Versioning and evolution

The document carries `schema_version`. Each entry in `profiles` additionally
carries its own `version` and a `requirement`:

| `requirement` | Reader behavior |
| --- | --- |
| `required` | A reader that does not understand the profile **MUST** fail. |
| `ignorable` | A reader that does not understand the profile **MUST** skip it and **MUST NOT** fail. |

This is the only ignorable surface. Unknown *core* fields still fail. A strict
reader can therefore load a package that carries a newer optional profile
without either guessing or refusing.

### 4.3a Capability admission and complete built-in catalogue

A **capability identifier** is a load-time promise by the reader. It is not a
model name, component contract, deployment preference, or statement that a
feature should be enabled. The package says "correct execution requires this
semantic behavior"; the runtime either advertises that exact serialized string
or rejects the package before execution.

There are two declaration surfaces:

- top-level `required_capabilities` states requirements not recoverable from the
  workflow, including implementation and legacy model-package requirements;
- `pipeline.workflow.manifest.capabilities` states requirements of the workflow
  IR. The validator also derives requirements from structures such as loops,
  emits, serving, state recurrence, and known adapter contracts, and rejects a
  manifest that omits a capability its structure uses.

Both fields intentionally accept extension-defined strings. Therefore no
repository can enumerate every future vendor extension. The table below is the
complete **built-in** vocabulary defined or advertised by this repository:
**30 identifiers** — syntax/IR 10, state/serving 8, media 5, adapters 4,
speculative 1, implementation 2, distributed 0. The source of truth is
`onnx_genai_metadata::capabilities::BUILTIN`; a test compares that constant
directly with this table. Adding a built-in identifier without documenting it
fails CI. Extension-defined identifiers remain enumerable only by the runtime
or producer that defines them and MUST still fail closed when unavailable.

<!-- capability-catalogue:start -->
| Serialized identifier | Category | Semantic feature admitted | Metadata that requires it | Runtime obligation and fail-closed rule | Representative examples and dependencies |
| --- | --- | --- | --- | --- | --- |
| `kv_cache` | state/serving | Runtime-owned autoregressive key/value state. | Explicit top-level `required_capabilities`; related state is described by workflow state-service groups. | Preserve the declared cache semantics and graph ABI; reject if no compatible cache implementation exists. | Decoder packages; commonly combined with an attention implementation capability. |
| `grouped_query_attention` | implementation | Execution of grouped-query attention. | Explicit top-level `required_capabilities`; attention geometry remains a package fact, not an inferred grant. | Execute the declared GQA semantics or reject; never silently substitute MHA. | Gemma/Qwen-style decoders; often paired with `kv_cache`. |
| `multi_head_attention` | implementation | Execution of ordinary multi-head attention. | Explicit top-level `required_capabilities`. | Execute MHA with the artifact's exact ABI or reject. | DeepSeek/GLM test fixtures; independent of GQA. |
| `prefix_cache` | state/serving | Reuse of a compatible cached prefix. | Explicit top-level `required_capabilities`; reuse dependencies and state-group reuse facts constrain correctness. | Reuse only when all declared dependencies match; otherwise reject or do not admit the package if the capability is required. | Repeated-prefix text serving; builds on cache-state support. |
| `continuous_batching` | state/serving | Interleaving, row compaction, and release while requests advance independently. | Explicit top-level `required_capabilities`; workflow `batch_layout`, serving values, and row-scoped ABIs provide the structural contract. | Apply one consistent row permutation to every request-aligned value and compact/release row-scoped components; reject if this cannot be guaranteed. | Batched decoder serving; depends on correct row semantics, not serialized row IDs. |
| `control_flow_loop` | syntax/IR | Legacy admission of a runtime-controlled generation loop. | Explicit top-level `required_capabilities`; retained for bare/legacy packages. | Honor the loop contract or reject; do not guess termination defaults. | Legacy decoder metadata; new workflows use `workflow_ssa` plus structural loop capabilities. |
| `image_preprocessing_program` | media | Typed, declarative image transform execution. | Explicit top-level `required_capabilities` with `preprocessing.image`. | Execute the declared transforms and tensor contracts exactly or reject. | Vision-language preprocessing; may combine with `packed_image_outputs`. |
| `packed_image_outputs` | media | More than one packed image tensor output and its packing metadata. | Explicit top-level `required_capabilities` when the image program emits multiple packed tensors. | Preserve offsets/ownership and output contracts; reject readers that only understand one dense image tensor. | Multi-view or grid-aware VLM preprocessing; depends on `image_preprocessing_program`. |
| `position_program` | media | Declarative construction of position IDs. | Explicit top-level `required_capabilities` with a preprocessing position program. | Produce the declared coordinates rather than inventing model-family position logic. | Vision-language token expansion and routed position construction. |
| `multi_axis_positions` | media | Position coordinates whose position axis has rank greater than one. | Explicit top-level `required_capabilities` when the position program emits multi-axis coordinates. | Preserve every coordinate axis and its ordering; reject scalar-only position implementations. | Spatial/temporal multimodal positions; normally depends on `position_program`. |
| `loop_carried_state` | state/serving | Legacy fixed-shape recurrent state with replace semantics. | Explicit top-level `required_capabilities`. | Carry the exact state tensor across iterations or reject. | Older recurrent decoder packages; structural workflows normally use state cells and `bounded_state_recurrence`. |
| `dual_sequence_inputs` | media | One decoder invocation consumes raw-token and routed-sequence inputs together. | Explicit top-level `required_capabilities` when the decoder ABI exposes both. | Bind both inputs without dropping or conflating either sequence. | Multimodal embedding/routing pipelines; independent of model-family identity. |
| `workflow_ssa` | syntax/IR | Typed SSA values and structural workflow execution. | Every `pipeline.workflow`; validator-derived and required in the manifest. | Compile/bind values without name guessing, enforce single assignment and lexical control-flow scope, or reject. | Every canonical catalogue workflow; foundation for the other workflow capabilities. |
| `linear_effects` | syntax/IR | Explicit, linearly threaded external-effect domains. | Manifest declaration for workflows/components with external mutation semantics. | Preserve effect order, branch joins, retry class, and speculation safety; reject unsupported effect semantics. | Grammar state, telemetry, session mutation, and streams; pure tensor state does not need an effect. |
| `serving_service_contract` | state/serving | Generic serving controls and runtime-owned state-service groups. | Presence of `pipeline.workflow.serving`; validator-derived and manifest-required. | Own active/done/accepted-length handling, state-group lifecycle, and declared rollback/fork bounds; reject incomplete service support. | Autoregressive, recurrent, weather-rollout, and hybrid speculative workflows. |
| `parameter_adapters` | adapters | Parameter overlay/application against resolved targets. | Top-level `adapters`; validator-derived. | Apply only validated target bindings and the declared application ABI, or use an allowed portable fallback; otherwise reject. | LoRA adapter selection; coupled to `onnx-genai.adapters@1`, which is an ABI identifier rather than this capability string. |
| `heterogeneous_adapter_batching` | adapters | Different adapter sets and scales for different request rows in one batch. | Top-level `adapters.selection`; validator-derived with `parameter_adapters`. | Keep adapter selection request-aligned through batching/compaction and never leak a row's adapter set to another row. | Batched LoRA serving; depends on `parameter_adapters` and row-safe batching. |
| `session_state_lease` | state/serving | State whose lifetime extends across invocations under a runtime lease. | Any workflow state cell with `scope: session`; validator-derived and manifest-required. | Create, restore, isolate, expire, and release leased state according to its contract; reject if only invocation state is supported. | Full-duplex/audio sessions and long-running rollouts. |
| `bounded_state_recurrence` | state/serving | Loop-carried state with metadata-declared bounded growth. | Any state cell with `recurrence.kind: bounded`; validator-derived and manifest-required. | Enforce the bound and recurrence axis without unbounded allocation or silent truncation. | Decoder masks/history, video schedules, nested audio loops. |
| `advisory_state` | state/serving | Droppable/resettable state that cannot change semantic correctness or output distribution. | Any state cell with `class: advisory`; validator-derived and manifest-required. | Keep advisory state out of semantic checkpoints and permit reset only because correctness is invariant; reject if that distinction cannot be honored. | Adaptive speculative estimates; often paired with `adaptive_proposal_budget`. |
| `adaptive_proposal_budget` | speculative | Runtime-visible adaptive choice of speculative proposal width. | A component contract with ID `onnx-genai.adaptive-proposal-budget`; validator-derived. | Execute the contract while treating estimate state as advisory; fixed or unsupported behavior must not masquerade as adaptive support. | Speculative proposer/verifier workflows; may use telemetry and advisory state. |
| `grammar_guidance_adapter` | adapters | Stateful grammar clone/lookahead/commit ABI. | A component contract/adapter ABI `onnx-genai.grammar-guidance@1`; validator-derived. | Enforce the action-specific binding set, linear effect ordering, and speculation-safety declaration or reject. | JSON/grammar-constrained speculative decoding; combines with policy sampling rather than replacing it. |
| `telemetry_adapter` | adapters | Versioned timestamp/elapsed adapter ABI. | A component contract/adapter ABI `onnx-genai.telemetry@1`; validator-derived. | Enforce exact action bindings and effect semantics; reject unknown actions instead of approximating timing. | Adaptive proposal metrics; optional unless the workflow invokes it. |
| `nested_control_flow` | syntax/IR | Loops or branches nested inside structural workflow nodes. | Any compiled loop or branch; validator-derived and manifest-required. | Preserve lexical SSA, zero-trip loops, branch phi results, and effect joins at every nesting level or reject. | Diffusion, autoregressive, weather, and nested audio/music workflows. |
| `loop_induction_values` | syntax/IR | A typed zero-based iteration value visible to policy components. | A loop with `iteration`; validator-derived and manifest-required. | Produce the declared scalar/per-row induction tensor and keep it scoped to the loop. | Diffusion scheduler lookup, termination equations, nested codec loops; depends on nested control-flow support. |
| `typed_emit` | syntax/IR | Publication of a typed SSA value to a declared workflow output. | Every compiled `emit`; validator-derived and manifest-required. | Enforce output contract, mode, row semantics, and effect ordering; reject undeclared or incompatible output publication. | All canonical workflows that return or stream values. |
| `streaming_emit` | syntax/IR | Incremental event publication during execution. | An `emit` with `mode: event`; validator-derived. | Preserve event order and effect semantics without converting it silently to only a final result. | Token/audio chunk streaming; depends on `typed_emit`. |
| `emit_valid_length` | syntax/IR | Ragged valid-prefix publication from a fixed-capacity tensor. | An `emit` with `valid_length`; validator-derived and manifest-required. | Slice each request row to its valid prefix, maintain positional row ownership through compaction, and reject incompatible output layout. | Speculative accepted-token prefixes and masked generation; depends on `typed_emit`. |
| `input_presence` | syntax/IR | An observable boolean for whether an optional caller/application tensor was supplied. | Any workflow input with `present_as`; validator requires the manifest entry. | Set the presence SSA value from actual caller presence, require control flow before optional use, and never fabricate sentinel tensors. | Text-only requests to a VLM and optional audio/image inputs; typically combines with `nested_control_flow`. |
| `explicit_transfer` | syntax/IR | A device-transfer node in the planner's lowered internal IR. | Derived only from internal `WorkflowNode::Transfer`; authored workflow steps MUST NOT serialize transfers. | Execute the transfer with correct ordering/device semantics or reject the lowered plan. | Heterogeneous placement; no authored model example because placement owns transfer insertion. |
<!-- capability-catalogue:end -->

The following similar-looking strings are **not entries in this capability
table**:

- component contract IDs plus versions, such as
  `onnx-genai.token-sampler@1`, `onnx-genai.token-sampler@2`,
  `onnx-genai.termination-predicate@1`,
  `onnx-genai.termination-predicate@2`, `onnx-genai.state-update@1`,
  `onnx-genai.state-update@2`,
  `onnx-genai.solver-step@1`, `onnx-genai.speculative-verifier@1`,
  `onnx-genai.counter-rng@1`, and `onnx-genai.guidance-combine@1`;
- adapter ABIs such as `onnx-genai.grammar-guidance@1`,
  `onnx-genai.telemetry@1`, `onnx-genai.image-preprocess@1`, and
  `onnx-genai.audio-preprocess@1`;
- adapter application/loader identifiers such as `onnx-genai.adapters@1`,
  `onnx-genai.adapters.hf-peft@1`,
  `onnx-genai.adapters.safetensors@1`,
  `onnx-genai.adapters.json@1`, and `onnxruntime.lora-adapter@1`;
- bounded facts inside a state group, such as `rollback_positions`, `snapshot`,
  `fork`, and `cascade`.

Those strings select a versioned semantic ABI or state bound and are validated
at the structure that owns them. They do not become negotiated capabilities
merely because older comments used the word "capability" broadly.

### 4.4 Semantic identity

`onnx_genai_metadata::identity::semantic_identity` computes a canonical hash
over the *normalized* document:

- keys sorted;
- `null` values dropped;
- containers that normalize to empty treated as absent;
- `requirement: ignorable` profiles dropped.

The scheme is `onnx-genai-metadata-identity-v1:sha256`.

Two documents with the same identity mean the same thing to a conforming
reader, so a disposable artifact bound to one — an execution-island plan, a
memory plan, a captured graph, a state checkpoint — remains valid for the other.
Skipping an ignorable profile does not change the identity, which is exactly why
a strict and a permissive reader can share a plan.

This hash is **identity, not integrity**. It says two documents are semantically
the same. It says nothing about who produced either one. Trust belongs to the
distribution layer ([§3.4](#34-distribution-layer)).

---

## 5. Ownership layers in the schema

The following fields **MUST NOT** exist in metadata. A document containing one
**MUST** be rejected:

| Removed field | Owner | Why |
| --- | --- | --- |
| `serving.slot_ids`, `package.slot_ids` | runtime | scheduler row identity |
| `emit.row_ids` | runtime | ragged rows are associated by the output API |
| `adapters.selection.slot_ids`, `.request_epochs` | runtime | exposes the same identity indirectly |
| `kv_service.storage`, `.paging`, `.slot_allocation` | runtime | allocator policy |
| `kv_service.shared_buffer` | runtime | replaced by the graph-ABI fact `aliasing` |
| KV quantization mode/tolerance | runtime | runtime-private cache representation |
| `transfer` as a serialized node | runtime | planner-lowered internal IR only |

Removal is fail-closed: because every core structure denies unknown fields, a
document that still carries one of these is rejected with a message naming it.

---

## 6. Workflow semantics

### 6.1 Structural IR

The graph is typed SSA over five node kinds. There are no phases, strategies, or
model-family dispatch:

| Node | Meaning |
| --- | --- |
| `sequence` | orders child nodes |
| `invoke` | binds named values to a component's ports |
| `loop` | setup, body, condition, `max_iterations`, induction value, carried state |
| `branch` | selects one case; exports only declared phi results and effect joins |
| `emit` | publishes a streaming or final package output |

Control-flow *location* defines lifecycle. A value produced inside a loop body
is loop-scoped; a value carried across iterations is a declared carried cell.

`transfer` is **internal lowered IR only**. The planner introduces transfers when
it assigns placement. A metadata document **MUST NOT** serialize one.

### 6.2 Values and effects

Values are SSA: produced once, read many times, freed by liveness. Mutating
operations thread a linear **effect token** through a named effect domain, so
ordering between mutations is explicit in the graph rather than implied by
document order.

Each effect domain is declared under `pipeline.workflow.effects`:

```yaml
effects:
  grammar:
    retry: idempotent          # pure | idempotent | transactional | non_retryable
    speculation_safety: rewindable
```

`retry` and `speculation_safety` are independent axes ([§13.3](#133-effects-and-speculation)).

### 6.3 Branch and phi

A branch exports only values its declared phi mapping names. Every case **MUST**
produce every exported value, and effect joins **MUST** be declared. This keeps
liveness and effect ordering decidable without executing the branch.

---

## 7. Components and implementation substitution

### 7.1 Implementation kinds

```yaml
components:
  splice:
    implementation: { kind: onnx, location: splice.onnx }
  grammar:
    implementation: { kind: binding }
  overlay:
    implementation: { kind: adapter, abi: onnx-genai.parameter-overlay, version: "1" }
```

ONNX is preferred where it enables portable composition and graph-level
optimization. It is not required. Grammar engines, tokenizer parsers, and other
algorithms with mature native implementations **MAY** be native without
apology — a native grammar component is a first-class, valid declaration.

### 7.2 Equivalence classes

A runtime **MAY** replace a component with a different implementation of the same
contract. What it **MAY NOT** do is change what the caller agreed to. Each
contract therefore declares its equivalence class:

| `equivalence` | Meaning | Automatic substitution |
| --- | --- | --- |
| `bitwise` | identical outputs, bit for bit | permitted |
| `distribution_preserving` | may differ per sample; the output distribution is unchanged | permitted |
| `semantic` | same meaning; the distribution may change | **caller opt-in only** |

`semantic` is the **default**. A package that says nothing gets the strictest
treatment, so silence never buys an optimization.

A component that declares **no contract at all** is read as `semantic`. It is
counted, not skipped: a reader that filtered undeclared components out of the
check would find the remaining set vacuously permissive, and a package whose
components declare nothing would be granted exactly the consent it never gave. By
the same rule, a workflow with no components permits nothing.

This is what makes speculative decoding safe: the runtime auto-enables
speculation only when **every component** in the workflow permits automatic
substitution. Otherwise the caller must ask for it explicitly. See
[§13](#13-speculative-execution).

There is no capability negotiation and no standardized implementation-selection
diagnostic. The equivalence class is the whole contract.

### 7.3 No custom-op admission

A component declares a *contract*, not an operator. Metadata never admits an
arbitrary custom operator into the graph by name.

---

## 8. Batching, varlen, and paged attention

### 8.1 Three layers

Continuous batching is layered, and each layer is invisible to the one above:

1. **Runtime-private request/sequence table.** Requests, rows, slots, epochs,
   and their lifetimes. Never serialized.
2. **Kernel/runtime varlen ABI.** `block_tables`, sequence lengths, slot
   mapping, `cu_seqlens`. A private contract between the runtime and its
   kernels. Never serialized.
3. **Metadata batching layout facts.** What a package declares.

### 8.2 Declared layout facts

Every `TensorContract` carries a `batch_layout`:

```yaml
batch_layout: { kind: shared }                                  # invariant across requests
batch_layout: { kind: request_aligned, axis: 0 }                # one row per request
batch_layout: { kind: token_packed, axis: 0,                    # packed items
                levels: [ { offsets: cu_seqlens,                 # + ownership
                            owner: token_owner } ] }
batch_layout: { kind: runtime_sequence_state }                  # opaque runtime state handle
```

That is the whole vocabulary. It states **where the request axis is**, never
**which request occupies which position**.

`token_packed` carries an ordered `levels` chain rather than a single
`offsets`/`owner` pair. The one-level form above is exactly what the earlier flat
spelling `{ offsets, owner, axis }` meant; the flat spelling is **replaced**, not
retained alongside it, and that replacement is the one deliberate compatibility
break in the v1.1 surface
([`ENCODER_BATCHING.md` §6.1](ENCODER_BATCHING.md#61-schema-evolution-what-actually-happens-to-an-old-runtime)).

### 8.3 No row identity

Metadata **MUST NOT** contain `slot_ids`, `request_epochs`, `emit.row_ids`, or
any other serialized row identity. Row-to-request association is provided by the
runtime output API.

The substitute is derivability: from `batch_layout` alone, the runtime can
determine, for every value and every state cell, which axis must be permuted
when the batch changes. The validator enforces that this is always derivable
([§8.5](#85-compaction-derivability)).

### 8.4 Ragged emission

An emit is *row-wise* when its output is ragged — when a `valid_length` or a
`when` guard is present. Row-wise-ness is a property of the **output**, not of an
individual emit: if any emit of an output is ragged, every emit of that output is
row-wise. An append loop that mixes a ragged accept step with a single forced
token therefore emits both row-wise, and the consumer sees one consistent stream.

Rows are published positionally. Keys such as `tokens.row.3` are serialization
details and **MUST NOT** be parsed for behavior.

### 8.5 Compaction derivability

The validator rejects a workflow in which a row-wise output cannot be attributed
to a request axis, or in which a row-scoped component's ports disagree about
which axis carries rows. Concretely:

- a row-wise output **MUST** have a `request_aligned` batch layout;
- a component's `row_scope.axis` **MUST** be within every row-scoped port's rank
  and **MUST** equal every port's declared request axis;
- a component that holds per-request state — one that declares an effect domain
  or a `cache_affects_state` fact — and has request-aligned ports **MUST**
  declare a `row_scope`.

The last rule is what makes compaction possible without identities: it
guarantees that anything holding per-request state has said which axis its rows
lie on.

### 8.6 Mandatory row ABI

Every row-scoped native or stateful component **MUST** implement two runtime
operations:

```rust
fn compact(&mut self, selection: &[usize]) -> Result<()>;  // keep/reorder rows
fn release(&mut self, row: usize) -> Result<()>;           // drop one row
```

This is an **ABI invariant, not a negotiated capability**. There is no
`supports_compaction` flag, because a component that cannot compact cannot be
batched, and a package that cannot be batched is not a deployable package. The
trait is `onnx_genai_engine::RowScopedState`.

`selection` is positional and **MAY** repeat a source position — that is how a
beam clone or a speculative fork expands rows. An out-of-range position fails
loudly.

### 8.7 Runtime-minted row selection

Beam search and speculative expansion need to say "row 4 becomes rows 4, 5, and
6". That selection is minted by the runtime and enters the workflow as a typed
opaque value:

```yaml
role: { kind: runtime, version: "1.0", role: row_selection }
```

It is an `int64` gather of positions *within the current batch*. It carries no
scheduler identity, has no meaning outside the invocation that produced it, and
**MUST NOT** be persisted.

---

## 9. Adapters (LoRA)

### 9.1 What stays in metadata

- the **target manifest**: which parameters are adaptable, with their shapes and
  composition order — architecture-neutral and authoritative;
- **artifact bindings**: sources (PEFT + safetensors, ORT `.onnx_adapter`) and
  the ABI they satisfy;
- the **request-selection contract**: the typed, request-aligned inputs through
  which a caller selects adapters.

Runtime execution **MUST NOT** discover adapter targets from model-family
conventions. If a parameter is adaptable, the manifest says so.

### 9.2 What the runtime owns

Loading, caching, budgeting, eviction, residency, and the concrete application
implementation.

### 9.3 Selection without identity

Selection is expressed as request-aligned values, never as slot identities:

```yaml
selection:
  segments: request.adapter_segments   # int64[batch, 2] — offset/length per row
  counts:   request.adapter_counts     # int64[batch]
  scales:   request.adapter_scales     # float32[...]
  active:   request.active             # bool[batch]
```

`segments`/`counts` preserve **heterogeneous batching** — different rows may
carry different numbers of adapters — and **ordered composition** — a row may
stack several adapters in a declared order. The runtime associates rows with
requests through its output API; the package never learns a request identity.

Because the selection tensors are request-aligned, they compact with every other
row-scoped value. The adapter run context implements `RowScopedState`
([§8.6](#86-mandatory-row-abi)), so an adapter selection survives compaction by
construction.

---

## 10. Multimodal encoders

### 10.1 Independent encoder batching

Vision and audio encoders batch on a different axis from the decoder: a request
may carry zero, one, or many images, and images from many requests pack together.
Encoder values therefore use `token_packed`:

```yaml
image_features:
  contract:
    dtype: float32
    rank: 2
    batch_layout:
      kind: token_packed
      axis: 0
      levels:
        - offsets: image_offsets   # cu_seqlens-style prefix offsets
          owner: image_owner       # per-item owning row
```

An `offsets`/`owner` pair is the only fact needed to map packed items back to
rows, and one level is the whole chain when items sit directly in rows. A second
level expresses items that nest — the frames of a video clip
([§10.5](#105-generic-component-batching)). This replaced the earlier
flat `{ offsets, owner, axis }` spelling of the same fact; see
[`ENCODER_BATCHING.md` §6.1](ENCODER_BATCHING.md#61-schema-evolution-what-actually-happens-to-an-old-runtime)
for why the flat form was migrated rather than kept.

### 10.2 Externally suppliable results

An encoder result **MAY** be declared request-scoped and externally suppliable:

```yaml
inputs:
  image_features:
    externally_suppliable: true
```

This lets a runtime cache, precompute, or receive an encoder result from another
process. The value's **transport, cache identity, and residency are
runtime-owned**; metadata declares only that supplying it externally is legal and
what its type is.

### 10.3 Splicing into the decoder

The decoder consumes encoder results through request-aligned splice descriptors
— placeholder positions and lengths — never through a global image index. The
splice is therefore correct under compaction for the same reason everything else
is: it is request-aligned, so it permutes with its row.

### 10.4 What §10.1 did not have

**Status.** This subsection records the state that motivated
[§10.5](#105-generic-component-batching); the metadata half of it shipped in
[#2009](https://github.com/justinchuby/onnx-genai/pull/2009) (`0448f2bc6`) and
is preserved in the past tense rather than deleted, because the reasons a field
exists are not recoverable from the field. The runtime half — the interpreter,
the preprocessor, and the scheduler — is still open.

[§10.1](#101-independent-encoder-batching) states the layout of a packed encoder
result. It did not state whether a component may *see* more than one item in one
invocation, and nothing else did either:

- `WorkflowComponent` declared no batching capacity, bound, or co-batching
  precondition. It now carries `batch_capacity`
  (`crates/onnx-genai-metadata/src/schema/ir.rs:919`, within the struct at
  `ir.rs:883-990`);
- `token_packed` has no runtime consumer outside this crate —
  `BatchLayout::request_axis` deliberately returns `None` for it (`ir.rs:194-199`),
  so the interpreter's three request-axis call sites never see a packed value.
  **This one is still true**, and it is the phase after the metadata surface;
- the image preprocessing adapter passes exactly one encoded item per invocation
  (`crates/onnx-genai-engine/src/pipeline/workflow.rs:3135`), even though the
  preprocessor underneath accepts many;
- no declared preprocessing program could produce the `offsets`/`owner` pair that
  `token_packed` names, because the content-role vocabulary had no entry for
  either. `pack_offsets`, `pack_owner`, and `valid_lengths` are now in both the
  vision and audio vocabularies
  (`crates/onnx-genai-metadata/src/schema/mod.rs:349-363`, `mod.rs:449-456`),
  though nothing *produces* them yet;
- validation checked nothing about `offsets`/`owner` beyond the serving-emit rule
  in `validation.rs`, which accepted `request_aligned` or `token_packed` without
  distinguishing them. [§10.6](#106-packed-companions-must-validate) is now
  implemented;
- and nothing expresses *nesting*. A video clip is a sequence of frames inside an
  item, but the encoder-side vocabulary has no temporal validity value and no
  second ownership level, while `temporal_patch_size`
  (`crates/onnx-genai-metadata/src/schema/pipeline.rs:195-198`) replicates one
  frame rather than carrying a sequence. Video is fully expressible as a workflow
  *output* (`WorkflowOutputRole::Video`, `ir.rs:822-829`) and absent as an
  encoder input. `levels` supplies the nesting spelling (`ir.rs:109-118`); the
  frame-sequence *producer* remains absent.

The consequence of the last point was already visible: the schema doc comment
called `offsets` request-aligned, the canonical fixture declared it `shared`,
and [§10.1](#101-independent-encoder-batching) describes it as a
`cu_seqlens`-style `rows + 1` vector. Nothing rejected any of the three. All
three are reconciled by the surface below, which shipped in
[#2009](https://github.com/justinchuby/onnx-genai/pull/2009).

There was also no mechanism through which any of it could ship, and supplying one
was a precondition rather than a detail. The metadata structs were already closed
— `InferenceMetadata` is `#[serde(deny_unknown_fields)]`
(`crates/onnx-genai-metadata/src/schema/mod.rs:38-40`) and `schema/ir.rs` carries
45 more, including `TensorContract`, `BatchLayout`, `ComponentPorts`, and
`WorkflowComponent` — so an older runtime meeting a new field rejected the whole
document rather than ignoring it. At the time, no pre-deserialization gate
validated `schema_version`, so a future document failed field by field instead
of reporting that the runtime needed an upgrade. #2009 replaced that missing
mechanism with the normalized version contract now documented at
`schema/mod.rs:62-78` and enforced by
`crates/onnx-genai-metadata/src/version.rs:42,78`.

### 10.5 Generic component batching

**Status.** The metadata surface below — `batch_capacity`, `padding`, the
`levels` ownership chain, the companion roles, and the version gate — shipped in
[#2009](https://github.com/justinchuby/onnx-genai/pull/2009) (`0448f2bc6`) and
is validated at load. No runtime consumes it yet: nothing groups, nothing packs,
and no preprocessing program produces the companions. Read the rules as
normative and in force at load, and the runtime behaviour they describe as the
contract the remaining phases are written against
([`ENCODER_BATCHING.md`](ENCODER_BATCHING.md) §8).

The requirement keywords in this subsection are schema fields and validator rules
at load, and describe runtime behaviour that is not yet implemented. The design
of record, with the evidence, phasing, and acceptance matrix, is
[`ENCODER_BATCHING.md`](ENCODER_BATCHING.md).

Vision encoders — images and video — motivate the work; nothing about it is
modality-specific. The contracts carry shape symbols, bounds, lengths, and
ownership levels; a modality vocabulary only produces the semantic values those
contracts point at, so audio windows and text segments reach the same path
without a new concept. The surface is three additions, each absent by default:

```yaml
components:
  media_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      uniform_dimensions: [features]         # symbols that must agree across items
      budgets:                                # materialized-footprint bounds
        - { dimensions: [clips],  max_total: 4 }
        - { dimensions: [frames], max_total: 64 }
        - { dimensions: [frames, patches], max_total: 65536 }
    ports:
      inputs:
        pixel_values:
          dtype: float32
          rank: 3
          shape: [frames, patches, features]  # one flattened packed axis
          batch_layout:
            kind: token_packed
            axis: 0
            levels:                            # innermost first
              - { offsets: frame_offsets, owner: frame_owner }   # frames -> clips
              - { offsets: clip_offsets,  owner: clip_owner }    # clips  -> rows
          padding:
            - { dimension: patches, valid_lengths: patch_lengths }
```

Grouping faces three independent kinds of raggedness: how many items a request
owns, which an ownership level's `offsets`/`owner` answer; how far two items
differ in extent, which a `padding` entry answers on the dimension that differs;
and how many parts an item nests — frames in a clip, windows in an utterance —
which an *additional ownership level over the same packed axis* answers. Each is
declared on its own dimension and they never compete for one.

- **Declarations are keyed by shape symbol, never by axis index.** A component's
  ports differ in rank — a rank-3 payload, a rank-1 companion, a rank-2 pooled
  output — so a component-global axis integer cannot be interpreted coherently
  across them, while a `TensorDimension::Symbol`
  (`crates/onnx-genai-metadata/src/schema/decoder_abi.rs:266-273`) names the same
  quantity on every port that mentions it. The axis a value is packed on is a
  property of that value's own layout.
- **`batch_capacity` absent means one request row per invocation** — today's
  behavior exactly. A runtime **MUST NOT** group a component that has not
  declared a capacity, and every budget is an upper bound, never an obligation.
- **`budgets` bind the group's materialized footprint**, keyed by symbol: a
  packed dimension is charged the sum of the participating items' valid extents,
  a padded dimension is charged *enclosing count × padded extent* — the rectangle
  actually allocated, not the sum of the valid lengths — a composed entry
  multiplies its symbols' footprints, and a `request_expanded` axis is charged
  `rows × factor`. There is no separate item-count integer: the item bound is the
  budget on that level's own symbol. Items and packed positions bind
  independently, and a grouping layer **MUST NOT** reuse a decode-side row bound
  as an item bound — one request may contribute many items and many requests may
  contribute one between them. Device memory is not among these bounds; it is
  runtime measurement, never metadata ([§1.2](#12-non-goals)).
- **`uniform_dimensions`** are the symbols two items must agree on before they
  may share an invocation — spatial and temporal alike — which makes batch
  compatibility a derived predicate rather than a policy. They name **ordinary
  per-item dimensions only**: a symbol listed there **MUST NOT** be a port's
  flattened packed symbol and **MUST NOT** be an ownership level's unit-count
  symbol, since both count the *group* the scheduler assembled rather than a
  property of an item. A fixed frame count is therefore not a pinned level but
  the absence of one — the frames→clips level is dropped and `frames` becomes an
  ordinary per-clip dimension that items must agree on. Every free dimension
  **MUST** be reconciled either by a `padding` entry or by the packed axis; a
  component that declares batchability without declaring how raggedness is
  expressed is rejected at load. A dimension **MUST NOT** be both padded and
  pinned.
- **Pinned is not the same as known, and budgets are group-rooted.** A
  `budgets` entry is a nesting path read outermost-first whose first symbol
  **MUST** be group-rooted — the flattened packed symbol or a level's unit-count
  symbol — so a *singleton* entry naming a per-item symbol such as `patches` is
  rejected: it bounds one item's shape, not the invocation. A pinned symbol
  **MAY**, and where it contributes to a materialized footprint **MUST**, appear
  in a *composed* entry such as
  `{ dimensions: [frames, patches], max_total: 65536 }`. Pinning means equal
  *within* a group, not fixed *across* groups: one group may pin `patches` at 64
  and the next at 1024, and a footprint bound that omitted the pinned symbol
  would bound nothing. An earlier revision's "pinned or budgeted, never both" is
  withdrawn.
- **`padding`** is a list of entries, each linking a padded dimension to an
  `int64` `valid_lengths` companion giving how much of each item is real, so a
  runtime never fabricates padding it cannot describe. The companion is `shared`
  and its shape is exactly the axes **outer** to the padded axis, in order — one
  length per enclosing position. **Padding is appended, never prepended or
  interleaved:** real entries form a prefix. Lengths rather than a boolean mask,
  because right-padding makes a mask `O(items × extent)` state for an `O(items)`
  fact, and because the runtime must read those numbers to build and split a
  group — a payload-sized mask would have to come back from the device to do it.
  A component whose graph consumes a materialized mask declares that mask as an
  ordinary port and its program produces it; the lengths remain the single truth.
  One entry per dimension, and never on the dimension the layout packs: padding
  and packing are two answers to one question. **Nor may an emit trim a padded
  output:** `WorkflowNode::Emit.valid_length`
  (`crates/onnx-genai-metadata/src/schema/ir.rs:1169-1186`) is honored by slicing
  the payload (`crates/onnx-genai-engine/src/pipeline/workflow.rs:2215-2226` and
  `4852-4882`), so an emit that both trims and declares `padding` publishes a
  shorter tensor beside a length vector measuring the tensor it replaced —
  either lengths that overrun the payload, or a value the runtime rewrote behind
  the metadata that describes it. The rule is exclusion, not reconciliation: an
  `Emit` whose output declares `padding` **MUST NOT** carry `valid_length`, and
  the refusal names `padding.valid_lengths` as the authoritative account, since
  a trim is invisible in the contract while a padding entry is part of what the
  caller reads. The exclusion does not ask which axis the trim is about: an
  emit's growth axis is only a default (`ir.rs:1178-1185`), so narrowing it would
  mean guessing in the direction of admitting the ambiguous case. A `token_packed`
  output needs no separate rule, because an emit-level `valid_length` into a
  layout with no request axis is already refused at load as ragged emission
  lacking a row axis
  (in `validate_compaction_derivability`,
  `crates/onnx-genai-metadata/src/validation.rs:4811-4825`); a packed **and**
  padded output hears from both rules, which is correct — they answer different
  questions. The exclusion is keyed on `padding` being non-empty rather than on
  the layout, because a packed value may legally pad: the no-double-spelling rule
  in [§10.6](#106-packed-companions-must-validate) forbids a `padding` entry only
  on the dimension the layout *packs*. Scoping it to layouts with a request axis
  would exempt the value with the most extents a trim can misdescribe.
  `padding` does not by itself make a component padding-invariant — that remains
  the profile's `batch_invariance` declaration.
- **Ownership is an ordered chain of levels over one physically packed axis.**
  `TokenPacked` declares `axis` — which **MUST** be `0` wherever `token_packed`
  appears, not only on a component that declares a capacity, because the runtime
  splits any packed value per request and every such split wants a contiguous
  span — and `levels`, one or two `{ offsets, owner }` pairs, innermost first. Level 0
  maps packed positions to their parent unit, the last level maps units to
  request rows, and composing them gives each row a single contiguous span. There
  is no second physical packed axis: frames are flattened across clips and across
  requests, and the frame→clip and clip→row maps are bookkeeping over that one
  axis. Two content roles, `pack_offsets` and `pack_owner`, join the preprocessing
  vocabularies so a declared program can produce them at any level, and a
  preprocessing-program value named as a level companion **MUST** carry the
  matching one. That is stricter than the `padding` reference, which resolves by
  name and treats the length role as descriptive, and deliberately so: length
  vectors have established modality spellings that predate this design, while the
  companion roles are new and have nothing to accommodate. A third role,
  `valid_lengths`, is **new**: no such role existed — the audio vocabulary
  offered `valid_frames`, `valid_samples`, `sample_lengths`, `frame_lengths`, and
  `validity_mask` (`crates/onnx-genai-metadata/src/schema/mod.rs:428-456`) — and
  an earlier revision's claim that it generalized an existing role is withdrawn.
  It coexists with those names rather than replacing them: a `padding` entry
  references a value **by name**, so an audio program may point at a
  `frame_lengths` value it already emits, while a modality with no established
  spelling uses the generic role instead of inventing one. Owner values carry a
  **position**, never a request identity ([§8.3](#83-no-row-identity)). Depth
  stops at two levels, so a third is a deliberate schema change rather than
  something a package asserts into existence.
- **Raggedness leaves a workflow with the metadata that decodes it.** An emitted
  `token_packed` value publishes every level's `offsets` and `owner`, and an
  emitted padded value publishes each entry's `valid_lengths` — published meaning
  *emitted by some declared step*, since an output nothing writes delivers an
  empty vector beside a ragged payload. The serving rule that rejects an emitted
  rank > 0 `shared` value
  (in `validate_compaction_derivability`,
  `crates/onnx-genai-metadata/src/validation.rs:4858-4862`) is carved out for
  exactly those referenced companions — `int64`, of the rank that reference
  demands, named by another emitted value's layout or `padding` entry in the same
  workflow — and for nothing else. The rank is per reference rather than a flat
  1: `offsets` and `owner` are rank 1 by construction, but a `valid_lengths` has
  one entry per position of the axes *outer* to the dimension it bounds, so a
  value padded on axis 2 publishes a rank-2 length vector.
  Withholding a length vector is not a smaller version of the same package: since
  a materialized validity mask is rejected for the contract, that vector is the
  only account of the padding that exists.
- **A row selection is lifted, never applied to a packed axis.**
  `BatchLayout::request_axis()` returns `None` for `TokenPacked`
  (`crates/onnx-genai-metadata/src/schema/ir.rs:194-199`) while `is_row_scoped()`
  reports true, so compaction resolves each destination row's unit range through
  the outer level, then each unit's positions through the inner level, producing
  an item permutation; companions are then **recomputed**, never gathered.
  `row_scope.axis` is therefore the component's **request row** axis and
  **MUST NOT** be a packed item axis.

Ownership is unchanged in shape: metadata describes the bound, the modality
vocabulary defines the semantic values, the preprocessor produces the per-item
tensors and their packing metadata, the scheduler decides which pending items
co-batch, the interpreter builds and splits the grouped invocation, and both
backends execute it through the one component-execution seam with identical
results. A runtime **MUST NOT** make two items compatible by changing them:
trimming frames, resampling a clip to a common frame count, or downscaling to a
common resolution are semantic changes, so the correct response to an
incompatible pair is two groups.

Two runtime obligations travel with the metadata and are stated here because
they bound what "the interpreter executes it" may cost. First, grouping
**MUST NOT** introduce a host round-trip for a value that is already device
resident, and splitting a packed result back to rows is an aliasing operation —
which is the practical reason the packed axis is pinned to 0 and the ownership
rules demand contiguity, since a no-copy view is a contiguous element window and
"a slice along an inner axis is not a contiguous range"
(`crates/onnx-genai-ort/src/value.rs:1524-1543`). Second, backend support for
grouped execution is asked **before** a group is formed, never discovered by
attempting one, and it is recorded per (component implementation, operator class,
execution provider) rather than as one global flip: a triple that has not proven
parity reports that it cannot group, receives no group, and the workload runs
item by item as it does today. Declining is safe; attempting is not. Neither fact
is a metadata field — residency and execution capability are runtime-owned
([§10.2](#102-externally-suppliable-results)) — and both are specified in
[`ENCODER_BATCHING.md` §5.1](ENCODER_BATCHING.md#51-grouped-buffers-aliasing-residency-and-what-padding-costs).

**Shipping this surface required a version gate, not a compatibility claim, and
the gate shipped with it** (`crates/onnx-genai-metadata/src/version.rs:78`,
`SUPPORTED_SCHEMA_VERSION` `v1.1` at `version.rs:42`).
`InferenceMetadata` and every struct this design extends are
`#[serde(deny_unknown_fields)]` (`crates/onnx-genai-metadata/src/schema/mod.rs:38-40`
and 45 occurrences in `schema/ir.rs`), so an older runtime **rejects the whole
document** when it meets `batch_capacity`, `padding`, or `levels` — it does not
ignore them. The current `schema_version` contract states the accepted grammar
and normalization (`schema/mod.rs:62-78`), and the shipped gate reads the version
from a generic parse and rejects an unsupported version with one actionable
message **before** struct deserialization (`version.rs:42,78`):

- **Normalization.** `[v]major[.minor]`, minor defaulting to 0. All three
  spellings already in the tree — absent (14 of the 39 `inference_metadata.yaml`
  files), `v1` (19), and `1.0` (6) — normalize to **v1.0**, so no existing
  document changes meaning or bytes. Anything unparseable is rejected as
  malformed.
- **Direction.** Reject when the document's major differs from the runtime's,
  **and** when the document's minor exceeds the runtime's supported minor.
- **Deliberately stricter than the in-repo precedent.**
  `onnx-model-package` parses `<major>.<minor>` and then ignores the minor
  entirely, gating only on major (`crates/onnx-model-package/src/lib.rs:563-579`).
  That is right for a container whose unknown parts are inert; it is wrong here,
  because `deny_unknown_fields` makes unknown fields a hard parse failure and
  because a runtime that silently skipped a `padding` entry while grouping would
  produce wrong numbers.
- **Canonical emission.** This additive surface is **v1.1**. A writer **MUST**
  stamp `v1.1` when the document carries any of these fields and **MUST NOT**
  otherwise, so packages that do not declare `batch_capacity` keep their current
  bytes and version strings exactly and their minimum runtime does not move.

Grouping still introduces **no new capability identifier** — but not because an
old runtime would otherwise refuse a package it can execute, which
`deny_unknown_fields` makes false. A capability is a load-time promise that
correct execution *requires* a behavior
([§4.3a](#43a-capability-admission-and-complete-built-in-catalogue)); no
package's correctness requires that its encoder be batched, so a runtime that
parses the document and chooses not to group stays correct. Version gating says
"this document uses a newer vocabulary"; a capability says "you must do this or
be wrong". This is also the general direction: a fact the workflow structure
already determines is not additionally serialized as a flag.

### 10.6 Packed companions must validate

**Status.** Implemented in
[#2009](https://github.com/justinchuby/onnx-genai/pull/2009) (`0448f2bc6`). The
load-time half of this subsection is in force; the invocation-time checks it
also specifies belong to the interpreter phase and are not implemented.

Full rules and their negative fixtures are
in [`ENCODER_BATCHING.md` §4](ENCODER_BATCHING.md#4-strict-token_packed-validation).
In summary, a `token_packed` value's companions are checked at **load**: every
level's `offsets` and `owner` **MUST** resolve to declared values; each is
`shared`, `int64`, rank 1, with level `k`'s `owner` carrying that level's unit
count and its `offsets` carrying the parent count plus one; `axis` **MUST** be
`0` for every packed value, whether or not the component declares
`batch_capacity`; a companion that is a preprocessing-program output carries
`pack_offsets` or `pack_owner`; `levels` holds one or two entries, innermost
first; every port naming a given `{ offsets, owner }` pair
agrees on that pair's extent symbols — consistency is keyed on **pair identity,
not on level index**, because a pair legitimately sits at level 1 of an input and
level 0 of an output that pooled the inner level away; a packed value declares no
`padding` entry **on the dimension it packs**, and no port is both
`request_expanded` and packed on one axis.

**Packing depth is capped at two levels.** Parts in items and items in rows
covers every known workload — frame → clip → request, frame → window → request,
token → segment → request — and each further level multiplies the validation
surface, the split implementation, and the corruption cases that must be tested.

**Every output level declares who produces its raggedness.** The declaration is
per **level**, not per value: each entry of a packed output's `levels` carries
`extent: preserved | produced`. `preserved` means this level's units correspond
one-to-one and in order with an input level's, which the validator checks by
comparing the referenced pair's extent symbols against the input port's;
`produced` means the graph decides the count, so that level's `offsets` and
`owner` **MUST** be declared outputs of the same component and naming an input
companion there is rejected. Input levels carry no `extent`. A value-wide flag
was considered and rejected as a category error: a token-merging encoder
*produces* its token→clip level while *preserving* the clip→row level it never
touched, and one flag can state only one of those. An output level omitting
`extent` is rejected, because the runtime would otherwise guess whether the
input's offsets still describe the result and split at the wrong boundaries in
silence.

**Serving admits companions, and only companions.** A serving workflow rejects an
emitted value of rank > 0 that declares `shared`
(in `validate_compaction_derivability`,
`crates/onnx-genai-metadata/src/validation.rs:4858-4862`), which would reject
the very companions a ragged emit is required to publish — a packed value's
`offsets` and `owner`, and a padded value's `valid_lengths`. The carve-out is
minimal and decidable from the workflow's own declarations — its outputs and its
steps, with no runtime information: a `shared` emitted value
is admitted **iff** it is `int64`, carries the rank that reference demands, and
is named as an `offsets` or `owner` of another emitted value's layout, or as the
`valid_lengths` of another emitted value's `padding` entry, in the same workflow;
anything else keeps the existing rejection. The rank is read from the reference,
not fixed at 1: an `offsets` and an `owner` are rank 1 by construction, while a
`valid_lengths` has the rank [§10.5](#105-generic-component-batching)
fixes — the number of axes outer to the padded one, equivalently `rank == axis`
for the padded dimension's axis index — so a flat rank-1 admission
would refuse a companion this design elsewhere requires, and would refuse it by
advising the one layout a companion may not declare. A companion must also be
**emitted**, not merely declared, and so must the value that names it: an output
no step writes is an empty vector beside a ragged payload, which is this rule's
failure case wearing the appearance of compliance, and asking for an emit on one
side while accepting a bare declaration on the other lets a `shared` vector walk
past the serving rule beside a padded output nothing writes. That check is
whole-workflow rather than path-sensitive, because "written by some declared
step" is what is decidable without evaluating branch predicates. A companion is
never compacted and never split like a payload — each request receives its own
span plus **rebased**, zero-based offsets for that span,
and the slice of any `valid_lengths` that indexes its own items, which needs no
rebasing because a length is already relative to what it measures. A declared
`owner` output is **internal**: it must be declared so the workflow validates and
the runtime can check the level, and it is never delivered, because its values
are positions within a grouping the caller cannot see
([§8.3](#83-no-row-identity)). Per-request owners, where a consumer wants
them, are **derived** by the runtime from the rebased offsets.

`offsets` is `shared` rather than `request_aligned` for a structural reason: an
exclusive prefix sum is not permutation-followable. Permuting rows does not
permute a prefix-offset vector, it invalidates it, so a runtime that changes the
grouping recomputes `offsets` instead of gathering it. The doc comment that
called it request-aligned was corrected when the rule landed.

At **invocation** the runtime verifies, at every level, that extents resolve
consistently, `offsets[0] == 0`, monotonicity, that the last offset equals the
child count, that `owner[i]` is in range, and that each parent's children are
contiguous; that every `valid_lengths` companion has the exact shape above and
lies in `[0, padded_extent]`; and that the assembled group satisfies every budget
and every pinned dimension. It reports a violation by naming the value, the
level, the index, and the two facts that disagree — never by clamping or by a
best-effort split. A request whose span is empty receives an empty span with rank
preserved — never a fabricated placeholder — and a group with no items is not
invoked at all. **None of these checks may cause a host transfer:** companions
the runtime built are host-resident already, and companions a component produced
are checked for dtype, rank, and resolved extent without reading data, with
value-level arithmetic done on the companion-only transfer the split already
requires.

---

## 11. Cache correctness dependencies

### 11.1 Derived, not declared

A prefix cache entry is only reusable if everything that could have changed the
state is the same. Enumerating that by hand is how caches go wrong, so the
dependency set is **derived**:

`onnx_genai_metadata::cache::cache_dependencies` walks the workflow SSA graph
and the component dataflow backwards from every emitted value and every state
write, through producers and value aliases (phi merges and lowered transfers),
and collects:

| Dependency | Always included when applicable |
| --- | --- |
| component/model implementation identity | yes |
| adapters that reach the state | yes |
| preprocessing and encoder results that reach the state | yes |
| generation-affecting profile facts | yes |
| externally supplied state | yes |

### 11.2 Native and external components

A native component has no visible dataflow, so it declares the non-dataflow
facts that affect state:

```yaml
grammar:
  implementation: { kind: binding }
  cache_affects_state: [grammar.parser_table]
```

This is the *only* thing a native component declares about caching. It does not
propose a key, a hash, or a namespace.

### 11.3 Runtime ownership

The runtime constructs the cache key, hash, salt, and tenant namespace from the
derived dependency set. Cross-process cache identity and request salting are
runtime and security responsibilities.

A validator test proves that removing a LoRA adapter or a multimodal encoder
input from a workflow changes the derived dependency set — the dependencies
cannot be omitted by accident.

---

## 12. State

### 12.1 Semantic kinds, not storage

A state group declares **what the state means**, never where it lives:

```yaml
state_service:
  groups:
    decoder_cache:
      kind: full_attention      # full_attention | sliding_attention | mla
                                # | recurrent_ssm | cross_attention | encoder | ...
      sequence_axis: 2
      layout: bnsh
      logical_lengths: cache_lengths
      aliasing: permitted       # forbidden (default) | permitted | required
      total_length: ...
      reuse:
        prefix_reusable: true
        evictable_prefix: false
      capabilities:
        rollback_positions: 32
        snapshot: true
        fork: true
        cascade: [...]
      checkpoint:                 # optional; absence means private state
        adapter: onnx-genai.kv-checkpoint
        version: "1"
      ports:
        model:
          cache_0: { input: past_key_values.0.key, output: present.0.key }
```

Metadata **MUST NOT** select `paged`, `shared_buffer`, or `separate` storage, a
slot-allocation algorithm, a device, or an execution provider.

### 12.2 Retained graph ABI facts

Removing storage policy must not remove real graph constraints. These are ABI
facts, and they stay:

- **`sequence_axis` and `layout`** — where a sequence dimension exists and how
  the tensor is laid out. Fixed-size replacement state has no sequence axis and
  omits it;
- **`logical_lengths`** — the graph-visible per-row valid length, when the graph
  exposes one;
- **`aliasing`** — whether the graph permits, requires, or forbids the runtime
  aliasing a `present` output onto its `past` input;
- **`total_length`** — in-place and total-length semantics.

`aliasing` is the honest replacement for the old `shared_buffer` policy flag. The
old flag said *"use one buffer"* — a deployment decision. `aliasing` says
*"this graph is or is not correct when past and present are the same memory"* —
a property of the graph. It defaults to `forbidden`, so a graph that never stated
its aliasing legality is never aliased.

### 12.2a Fixed-size recurrent state

Linear-attention accumulators, state-space carries, and causal-convolution
history are all fixed-size recurrent state from the runtime's perspective. They
therefore use the same generic declaration:

```yaml
state:
  linear_accumulator:
    contract:
      dtype: float16
      rank: 4
      shape: [batch, heads, key_feature, value_feature]
      batch_layout: { kind: request_aligned, axis: 0 }
    scope: invocation
    initializer: initializer.linear_accumulator
    recurrence: { kind: invariant }
    management: runtime
    release_boundary: session
    service_group: linear_recurrence
  causal_conv_history:
    contract:
      dtype: float16
      rank: 3
      shape: [batch, channels, kernel_history]
      batch_layout: { kind: request_aligned, axis: 0 }
    scope: invocation
    initializer: initializer.causal_conv_history
    recurrence: { kind: invariant }
    management: runtime
    release_boundary: session
    service_group: conv_recurrence
serving:
  state_service:
    groups:
      linear_recurrence:
        kind: recurrent
        layout: bhkv
        update: { kind: replace }
        ports:
          decoder:
            accumulator:
              { input: recurrent_state, output: updated_recurrent_state }
      conv_recurrence:
        kind: recurrent
        layout: bct
        update: { kind: replace }
        ports:
          decoder:
            history: { input: conv_state, output: updated_conv_state }
```

The groups are separate because their shapes, graph ports, checkpoint
compatibility, and rollback cascade may differ—not because linear attention and
causal convolution need different serialized state kinds. `replace` means the
component emits the complete next fixed-size tensor. Such a group omits
`sequence_axis`; introducing an algorithm-specific kind would add no runtime
semantic information.

### 12.2b Fixed-capacity state and the write contract

A growing cache and a fixed-capacity ("static") cache differ in one respect that
the runtime cannot infer: where the next write lands. A growing cache appends,
so the destination *is* the current length. A static cache scatters into a
preallocated buffer at a position the graph receives as data, and that position
is chosen by whatever produced it — a prior step, a policy graph, or a scheduler
decision reified into a tensor. Nothing about the tensor distinguishes it from
any other integer control input.

`update` closes that gap:

```yaml
state_service:
  groups:
    decoder_cache:
      kind: full_attention
      sequence_axis: 1
      layout: bsh
      logical_lengths: cache_lengths     # graph-visible valid prefix per row
      aliasing: permitted
      update:
        kind: indexed_scatter            # append | replace | indexed_scatter
        write_indices: write_indices     # a rank-1 semantic state cell, one slot per row
        capacity: package.capacity       # a graph-visible integer scalar
        write_indices_ports:             # which input port carries destinations
          model: write_indices
        kv_length_ports:                 # which input port carries the valid KV length
          model: cache_lengths
      checkpoint: { adapter: onnx-genai.kv-checkpoint, version: "1" }
      ports:
        model:
          key_cache:   { input: key_cache,   output: updated_key_cache,   role: key,   layer: 0 }
          value_cache: { input: value_cache, output: updated_value_cache, role: value, layer: 0 }
```

Each part earns its place:

- **`write_indices` is a state cell, not a step output.** The cursor is part of
  the cache's identity: forking a session forks it, rewinding restores it, and a
  checkpoint that saved the buffer without it would restore a cache that
  overwrites its own history. Declaring it as state is what makes those
  operations coherent.
- **`capacity` separates the buffer's size from its contents.** `logical_lengths`
  is the valid prefix; capacity is the wall. Both are needed, because the whole
  point of a static cache is that they differ.
- **`write_indices_ports` and `kv_length_ports` name the ports.** The runtime
  checks destinations *before* the invoke, which requires knowing which bound
  input carries them. These are the facts that cannot be recovered from the
  graph: a write cursor and a valid length are both rank-1 integer vectors, so
  they are shape- and dtype-indistinguishable from each other and from any
  other integer control input, and the runtime cannot invert a cell name back to
  a body SSA value at invoke time. `logical_lengths` names a *state cell*;
  `kv_length_ports` names the *port* that cell reaches, and both are needed
  because a cell is not a port.
- **`role` and `layer` on a port pair.** A split cache exposes two
  shape-identical buffers per layer; only the producer knows which half is
  which, and the alias key is a producer-chosen label whose lexicographic order
  is not the graph's (`layer.10` sorts before `layer.2`). A runtime pairing
  buffers positionally would silently transpose two layers' caches, and no shape
  check could catch it. `role` distinguishes the halves and `layer` fixes the
  order.
- **Group-bound buffers must be shape-`invariant`.** A fixed capacity that
  changes shape is a contradiction; the validator refuses it rather than letting
  the two claims drift.

What this does **not** do is allocate anything. Capacity is read from a value the
package already exposes; the buffers are whatever the caller bound. Physical
allocation, placement, and device remain runtime-owned exactly as in §12.1 — the
declaration only says what the graph will do to memory it is given, so the
runtime can check it first.

The check is not decorative. `ScatterND` accepts negative indices as
from-the-end addressing, so a corrupted destination is *valid ONNX* that
silently writes into another row's history; and an out-of-range one is an
out-of-bounds write on providers that do not bounds-check. Validating
destinations against declared capacity before execution is the only place that
distinction can be caught while it is still an error rather than a wrong answer.
Because fusing a scatter component into an execution island would hide its
destinations behind the island boundary, such components are excluded from
fusion: an unchecked scatter is undefined behaviour, which is not a trade a
performance optimization may make.

#### The static-cache ABI lives in the workflow

The port ABI of a static cache is not a separate declaration beside the
workflow. It **is** the workflow: `write_indices_ports` and `kv_length_ports`
name the control inputs, the group's `ports` name the per-layer buffers, and
`role`/`layer` say which buffer is which. A runtime that drives the decode graph
directly reads the same declaration the workflow engine executes, resolved
through `decoder_io()` (§4.1a).

An earlier revision permitted a retired `static_cache` block beside the
workflow, on the reasoning that a direct-drive runtime needs the port ABI while
the group supplies the bindings. That was the wrong repair. The problem it
solved was real — a workflow package genuinely had nowhere to name control ports
that are shape-indistinguishable from one another — but the fix admitted a
second writable answer to a question the workflow already answers, which is the
defect §4.1a forbids. Adding the two port maps to `update` puts the missing fact
where the binding that consumes it lives, and declaring the pair beside a
workflow is now rejected.

### 12.2c FP8 and other narrow cache element types

The dtype vocabulary includes `float8_e4m3fn` and `float8_e5m2`, and a package
may declare state in them. This is a representability decision with a sharp
boundary, because three capabilities are involved and they are supported
independently:

| Capability | Who decides | Status |
| --- | --- | --- |
| Declare FP8 state and ports | metadata format | supported |
| Validate the document | validator | supported |
| Allocate and bind FP8 buffers | runtime | supported |
| Compute on FP8 in a kernel | execution provider | provider-dependent |

The rule that follows: **validation must not pre-empt the provider.** Metadata is
portable and cannot know which provider will load a package, so refusing an FP8
dtype at validation time reports a missing kernel as a malformed document — and
those two problems have opposite remedies. A capability gap tells the reader to
change provider or build; a schema error tells them to edit a file that was
correct. The runtime therefore carries FP8 as far as it can and fails, when it
must, with the provider's own error naming the operator and the element type.

Measured on the CPU provider (ORT 1.28): an FP8 tensor allocates, binds, and
round-trips through a session, so FP8 state is *loadable*. `ScatterND` has no FP8
kernel there, so an FP8 *static* cache is not executable on that provider — the
session fails to load with a type error naming `ScatterND` and
`tensor(float8e4m3fn)`. That is an execution-provider blocker, not a metadata
limitation, and it moves on its own when a provider registers the kernel.

Measured on the CUDA provider (onnxruntime-gpu 1.29.0, H200): the blocker is real
there too, but it surfaces in a **different shape**, and the difference matters to
whoever has to read the error. FP8 KV is reachable only through
`GroupQueryAttention`, whose `k_scale`/`v_scale` arrive as node inputs 12 and 13.
With FP8 KV the node matches no registered kernel, so graph partitioning leaves it
unassigned and session creation fails during initialization:

```
transformer_memcpy.cc:253 IsNodeCompatibleWithProvider
  Provider type for GroupQueryAttention node 'node_GroupQueryAttention_9' is not set
```

Note what that message does **not** say: it names neither FP8 nor the element
type. Read alone it looks like a malformed graph, which is precisely the
misreading §12.2c exists to prevent. A three-way split attributes it exactly:

| Node | KV element type | Result |
| --- | --- | --- |
| 14-input GQA, scales at 12/13, quant attributes set | `float` | loads and runs |
| same node, unchanged arity and attributes | `float8_e4m3fn` | node unassigned, init fails |
| no FP8 pass at all | `float16` | loads and runs |

Holding arity and attributes fixed and moving only the element type isolates the
cause to the **KV type constraint** — `tensor(float8e4m3fn)` is absent from the
CUDA GQA `past`/`present` type list in 1.29.0. It is not the scale-input arity,
not the quantization attributes, and it is not a CUDA illegal memory access: no
kernel is ever launched. So this is a missing kernel registration, which is the
same class of blocker as the CPU `ScatterND` gap and moves the same way.

One consequence for producers: a fixed-capacity export built from `TensorScatter`
plus the ai.onnx `Attention` operator has no GQA node at all, so it has nowhere to
carry `k_scale`/`v_scale` and cannot express FP8 KV regardless of the type list.
A producer that is asked for both should fail closed with that diagnosis rather
than silently emit 16-bit floats — the request was for something the operator set
cannot represent, and a quiet dtype substitution turns an unsatisfiable request
into a plausible wrong answer.

Two runtime storage paths remain narrower than the format, and say so by name
rather than by refusing the document: the **paged KV cache** stores fp32 and
16-bit float pages only, and **host-side KV growth** materializes buffers through
per-dtype host representations. Both are backend capabilities; a package that
declares FP8 state is well formed and simply cannot use those backends.

### 12.3 Capabilities and cascade

`capabilities` declares the bounds within which a group can be rewound, snapshot,
or forked, and `cascade` names the groups that must move with it. The runtime
decides *when* to use them; the package decides *whether it is legal*.

### 12.4 Lifetime and release boundary

- **Ordinary tensors** use SSA liveness. No annotation.
- **Runtime-managed and external state** has no last reader in the dataflow, so
  it **MUST** declare a `release_boundary` (`invocation` | `session`).

The validator enforces the corollaries: a workflow-owned cell **MUST NOT**
declare a release boundary (it is freed by liveness), a cell that binds a state
service group **MUST** be `management: runtime` (the group is the runtime's
storage), and a `session` release boundary requires session scope.

### 12.5 Sessions

Session state is normative and minimal. Metadata declares:

- the **scope** (`invocation` or `session`);
- the typed **mutation semantics**;
- the lease capability the reader must implement (`session_state_lease`).

The runtime owns session IDs, TTL, storage, locking, migration, and retention.
Interactive world-model, robotics, and streaming-observation workloads enter as
separate workflow invocations with session state in between; the portable IR
adds no network-aware `receive` or `await` operation.

#### 12.5a How the next invocation reaches leased state

Scope says *keep this*. It does not say how the next invocation reaches what was
kept, and a package that leaves that unanswered advertises a continuity it does
not have — every turn restarts and the failure reaches a caller as a model that
forgot what it was told. A lease is reached through exactly three mechanisms, and
a document uses one of them. Which one each session cell uses is answered once,
by `classify_session_state`, and read by both the validator and the runtime —
computing it twice is how they came to disagree.

- **A loop carries the cell.** Its lease is what seeds the carry when the pass
  enters the loop, in place of the initializer the document names. This is what a
  full-duplex or streaming package does.
- **A state service group holds it.** The group's alias names the `input` port
  the graph reads and the `output` port that advances it, so the lease replaces
  the value the cell's initializer names and the alias's output is what the next
  lease holds. An alias with no output port is refused: the lease could be read
  and never advanced, so every turn would replay the first.
- **The request binding rejoins it**, declared on the lease:

  ```yaml
  conversation:
    contract: { dtype: int64, rank: 2, shape: [batch, conversation_length],
                batch_layout: { kind: request_aligned, axis: 0 } }
    class: semantic
    scope: session
    initializer: request.input_ids
    recurrence: { kind: bounded, axis: 1, max: package.max_context }
    management: runtime
    release_boundary: session
    session:
      policy: exclusive
      continuation:
        kind: prompt_prefix
        prompt_input: request.input_ids   # must carry role prompt_tokens
        tokens_output: tokens             # must carry role tokens
  ```

  The value bound to `prompt_input` becomes the cell's value followed by the
  caller's tokens; when the invocation completes the cell becomes that
  concatenation followed by what was published to `tokens_output`. A session
  holding nothing contributes nothing, so a conversation's first turn and a
  request with no session are the same execution — declaring a conversation
  costs a package nothing when nobody asks for one.

  The `recurrence` bound is load-bearing: a continuation is not loop-carried and
  so never reaches the carry path's recurrence check, and this is the only place
  it is honoured. Separately, the conversation's *current* length is what a front
  end adds to a request's own before enforcing a context limit, because a
  prompt-prefix conversation really is prefilled again on every turn — and it is
  the only carrier for which that is true. A turn whose conversation would exceed it is refused before it
  runs, and a turn whose own generation would exceed it is refused rather than
  stored — a session left in a state its own declaration forbids has no way
  back. Neither refusal changes what the session already held; `reset_session`
  releases it.

  This is what a decoder whose prefill starts from empty state declares. Such a
  package's cache is rebuilt from the conversation on each turn; nothing in the
  document asks the prefill to accept a cache it was never authored to take, and
  no runtime has to invent that it should.

The validator enforces the corollaries. A continuation must be `scope: session`,
`class: semantic`, `management: runtime`, `release_boundary: session`, and must
grow along a bound that names a declared input with a value by the time a turn is
admitted; it must name a declared `prompt_tokens` input and a declared `tokens`
output whose contracts match the cell's; it must not also be loop-carried, which
would be two answers about the same value; and a workflow declares at most one,
because a package has one conversation. A session-scoped cell binding a
`service_group` must resolve to a declared group that aliases it.

A **semantic** session-scoped cell with none of the three is refused at load:
nothing in the document says how the next invocation reaches it. Advisory state
is exempt, because it is droppable by declaration.

A package that publishes a token stream and declares no session state at all is
refused a session at `create_session` rather than handed one whose turns silently
restart. That refusal is typed — `PackageCapabilityError` — so a front end
answers it as a request/package mismatch (HTTP 409) rather than as a server
fault. A package that publishes no token stream has no conversation to lose and
keeps its session handle.

A lease declared `policy: exclusive` is single-flight: a second turn that starts
while one is in flight is refused by name rather than allowed to read a
conversation the first is about to replace.

### 12.5b What a prompt-prefix conversation costs

A `prompt_prefix` continuation carries the conversation as **tokens**, not as a
cache: the package's own cache cells are invocation-scoped and released when the
invocation ends, so turn *N* re-prefills every earlier turn. Over a conversation
of *N* tokens that is O(N²) prefill work, against O(N) for a decode core whose
paged KV survives the turn.

That is the cost of continuing a conversation a package can *express*. A decoder
whose prefill accepts only an empty cache has no port for a prior session length,
so a runtime handing it the previous turn's cache would produce a mask and a
cache that disagree. A package that wants the linear cost declares its cache
session-scoped and is executed by a core that keeps it. A package that publishes no token stream has no
conversation to lose, and its session is an ordinary handle.

### 12.6 Private state and checkpoints

Internal state is private by default. An internal state cell is **not**
automatically a package output.

Portable state moves only through an explicit **versioned checkpoint adapter**,
declared on the state group:

```yaml
decoder_cache:
  checkpoint: { adapter: onnx-genai.kv-checkpoint, version: "1" }
```

That adapter is the *only* portable, cross-build state path. It is deliberately
not a wire format: metadata names the adapter and its version, and the adapter
owns the encoding.

A **runtime-owned or external** state cell (`management: runtime | external`)
**MUST NOT** be published as a package output unless its group declares a
checkpoint adapter. Publication is detected on the **emitted value**, never on
the output key: an `emit` names an SSA value and an output key that need not
match, so reading the output name would let `emit { value: cache, output:
cache_dump }` export runtime-owned state under an alias. Absence means private, so a group that never said anything is
never exported. Workflow-owned cells are exempt: they are ordinary typed tensors
with a graph-visible representation, so publishing one carries no cross-build
hazard.

Prefill/decode disaggregation and encoder/decoder state interchange are
**private distributed runtime protocols**. They are fast precisely because they
are not portable: they require a matching runtime protocol and build on both
ends. A checkpoint is portable and slow; a P/D transfer is private and fast.
Confusing the two is how a cluster silently corrupts state across a rolling
upgrade, so the contract keeps them separate and names neither as a fallback for
the other.

### 12.7 State representation and quantization

Graph-visible state representation is **inferred from the graph**: tensor dtype,
Q/DQ nodes, scale ports, `Attention` attributes, and the declared graph ABI. It
is not a separate metadata field, because two sources for one fact is one source
too many.

The runtime-private cache representation — including whether the runtime
quantizes its own cache — is **runtime-owned and runtime-validated**. Metadata
carries no KV quantization mode and no KV quantization tolerance. The runtime
policy type is `onnx_genai_kv::KvQuantPolicy`.

This does not touch **model-weight** quantization intent, which remains in
`quantization` and is a published property of the package.

---

## 13. Speculative execution

### 13.1 What metadata declares

```yaml
speculative:
  proposer: { ... }             # typed ports
  target: { ... }               # typed ports
  shared_state: [decoder_cache] # shared state groups
  shared_weights: []            # shared weight bindings
  vocabulary: { kind: identical }
  rollback_state: [cache]       # workflow state cells that must rewind
  max_proposal_width: 8
  distribution_preserving: true
```

### 13.2 What the runtime owns

Proposal width K, tree shape, scheduling, kernels, and whether speculation runs
at all.

### 13.3 Effects and speculation

Two independent axes govern whether a region can be speculated:

| `retry` | Meaning |
| --- | --- |
| `pure` | no external effect |
| `idempotent` | repeating is harmless |
| `transactional` | can be committed or rolled back atomically |
| `non_retryable` | must not be repeated |

| `speculation_safety` | Meaning |
| --- | --- |
| `none` | cannot be speculated |
| `clonable` | can be duplicated per branch |
| `rewindable(max_depth)` | can be undone up to `max_depth` positions |

These are **not the same question**. An idempotent effect can be retried and
still be impossible to rewind: appending to an audit log twice is harmless, but
un-appending is not a thing. The validator therefore reads
`speculation_safety` — never the retry class — when checking a speculative
region.

The **speculative region** is the innermost loop body that invokes both
`speculative.proposer` and `speculative.target`, not just those two components.
Every component invoked in that body — a grammar sidecar, a routing head, a
logit processor — runs on every speculated position, so its effects are held to
the same rewind bound. When the two roles are not invoked inside a common loop
there is no iterated region and only the named components are constrained.

### 13.4 Rollback validation

`speculative.rollback_state` names **workflow state cells**. A cell bound to a
state service group inherits that group's `rollback_positions` bound, and
`cascade` is walked transitively. The validator rejects a speculative region
whose effects or state cannot roll back to `max_proposal_width`.

### 13.5 Hybrid linear-attention targets

A Qwen3.5-style target combines sequence-growing attention KV with fixed-size
linear-attention and causal-convolution state. All three advance while the
target scores a proposal, so all three belong in `rollback_state`.

Attention KV can discard a rejected suffix along its sequence axis. A
replacement accumulator or convolution history has no sequence axis and cannot
be truncated. To commit only `accepted_len`, the runtime must either retain the
replacement state produced at every proposal prefix or restore the
pre-proposal snapshot and replay the accepted prefix. The metadata expresses
this capability with a sufficient `rollback_positions` bound, `snapshot: true`,
and a `cascade` connecting all affected groups; it does not prescribe which
runtime algorithm implements the rollback.

The complete config is
[`20-qwen3_5-hybrid-speculative-decoding.yaml`](../../examples/inference_metadata/catalogue/20-qwen3_5-hybrid-speculative-decoding.yaml).
Its proposer is independent and recomputed from committed tokens. A persistent
draft model needs its own state groups in `rollback_state`; mutable target state
is not `shared_state` unless both graph ABIs truly share that exact state.

### 13.6 Automatic enablement

The runtime auto-enables speculation only when **every component** in the
workflow declares `bitwise` or `distribution_preserving` equivalence
([§7.2](#72-equivalence-classes)). A component with no contract counts as
`semantic` and withholds consent on its own, as does an empty workflow.
Otherwise the caller must opt in — by naming a mode on the request or in the
engine configuration.

---

## 14. Generation

### 14.1 Authoritative defaults

```yaml
generation:
  defaults:
    do_sample: true
    temperature: 0.6
  overrides:
    temperature:
      input: request.temperature
      constraint: { minimum: 0.0, maximum: 2.0 }
    top_k:
      input: request.top_k
```

Package defaults are **authoritative**. A reasoning model that ships
`do_sample: true, temperature: 0.6` — precisely because greedy decoding makes it
loop — decodes stochastically by default. A runtime's built-in fallback applies
only when the caller is silent *and* the package declares nothing.

### 14.2 Structural, fail-loud overrides

A caller **MAY** override only fields listed in `generation.overrides`, and each
listed field **MUST** name a request-sourced typed workflow input. Support is
therefore **structural**: a field is overridable exactly when the workflow has an
input wired to carry it.

An override of an unlisted field, or a value outside a declared `constraint`,
**MUST** fail loudly. Silently dropping it would let a request decode at the
package default while the caller believed their value took effect.

The runtime entry point is
`GenerateOptions::resolve_generation_contract`.

---

## 15. Preprocessing and generated inputs

Preprocessing uses typed semantic contracts. Implementations **MAY** be ONNX or
native. A runtime **MUST NOT** infer preprocessing behavior from a model-family
name.

`preprocessing.image` and `preprocessing.audio` are the two declared programs.
Both have the same shape — an ordered list of generic `transforms` plus named
`outputs` that bind program-local values to workflow SSA names — and both draw
their operation names and content roles from one open vocabulary per modality,
so a new family adds a fixture, not a runtime branch. Every parameter is model
**data**: a mel-bin count, an FFT size, a sample rate, and a target window all
live in the package. A CTC acoustic model declares
`resample`/`downmix`/`zero_mean_unit_variance` over raw samples; an
encoder-decoder speech model declares `resample`/`pad`/`log_mel` over a fixed
window. The runtime reads the same fields either way and never dispatches on
which one it is.

In workflow metadata each program is materialized by exactly one
manifest-pinned adapter invocation (`onnx-genai.image-preprocess@1` or
`onnx-genai.audio-preprocess@1`) that takes a `uint8` rank-1 `encoded` input,
and every declared output **MUST** carry a `TensorContract` compatible with the
adapter port it binds to. A package **MAY** instead hand the server an
already-featurized media tensor by declaring a `media` runtime input with the
full contract and no program; there the input's own shape states the geometry.

A program's outputs cannot currently state how several items pack together: the
content-role vocabulary has no entry for per-item offsets or per-item ownership,
so the `offsets`/`owner` pair that `token_packed` names
([§8.2](#82-declared-layout-facts)) has no declared producer. Nor can a program
state how an item nests — the frames of a video clip, the windows of an
utterance — or how much of a padded item is real. Two level-agnostic roles,
`pack_offsets` and `pack_owner`, close the first two gaps at any nesting level,
and a third role, `valid_lengths`, closes the third. That role is **new**, not a
generalization of an existing one: the audio vocabulary offered `valid_frames`,
`valid_samples`, `sample_lengths`, and `frame_lengths` and no `valid_lengths`,
and all three roles are now declarable
(`crates/onnx-genai-metadata/src/schema/mod.rs:449-456`) though nothing produces
them. Since a padding contract references its length value by name, those
modality-specific spellings keep working alongside the generic one
— see [§10.5](#105-generic-component-batching) and
[`ENCODER_BATCHING.md`](ENCODER_BATCHING.md).

Application policy inputs — a grammar, a JSON Schema, a regex — are **request
data**, not metadata. What metadata carries is everything needed to interpret
that request data correctly:

```yaml
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32000
    byte_level: true
    artifacts:
      - location: tokenizer.json
    special_tokens:
      bos: { id: 1, content: "<s>" }
      eos: { id: 2, content: "</s>" }
  constraint_languages:
    - dialect: llguidance.lark
      version: "1"
      component: grammar
```

The tokenizer's declared vocabulary facts and artifact location are part of the
semantic contract, while byte-level integrity belongs to distribution. The
constraint-language dialect and version matter because a caller-supplied grammar
is only meaningful against a named dialect, and the component that interprets it
is named so the dependency is derivable ([§11](#11-cache-correctness-dependencies)).

---

## 16. Distributed execution

Metadata declares **legal** sharding facts:

```yaml
model:
  sharding:
    tensor:   { shard_axes: {...}, replicated: [...] }
    pipeline: { stages: [...], cross_stage_ports: [...] }
    expert:   { experts: N, expert_axis: 0 }
```

The caller and runtime choose the TP, PP, and EP **degree**, device mapping,
placement, and collective backend. Metadata never picks a topology.

Portable metadata does **not** standardize a cross-runtime KV/cache wire format.
Cross-process state movement is either a private runtime protocol or a versioned
checkpoint ([§12.6](#126-private-state-and-checkpoints)).

---

## 17. Legacy import

Import is **one-way and fail-closed**:

```
genai_config.json  ──►  InferenceMetadata
```

There is no export path and no reverse synthesizer. A reverse synthesizer would
have to approximate facts the new contract states precisely, and an approximation
that looks like a package is worse than no package.

`onnx_genai_genai_config::import` walks the *raw* JSON — not just the
deserialized struct, because serde silently discards keys the wire types do not
name — and classifies every key path:

- **consumed** (`CONSUMED_KEYS`): read by the converter;
- **known-dropped** (`KNOWN_DROPPED_KEYS`): deliberately not represented, with a
  recorded reason (deployment policy, kernel varlen ABI, scheduler identity);
- **unrecognized**: anything else.

Any dropped key is an error by default. `--allow-lossy`
(`ImportOptions::allow_lossy`) downgrades it to a recorded list in
`ImportReport::dropped_keys` — the list is the point: a lossy import must be able
to say exactly what it discarded.

CLI: `import_genai_config [--allow-lossy] <genai_config.json>`.

---

## 18. Migration

Packages written against the previous contract migrate as follows:

| Before | After |
| --- | --- |
| `serving.slot_ids: <value>` | delete; runtime owns row identity |
| `emit.row_ids: <value>` | delete; ragged rows come from the output API |
| `adapters.selection.slot_ids/request_epochs` | delete; use `segments`/`counts`/`scales`/`active` |
| `kv_service:` | `state_service:` with semantic `kind` per group |
| `storage: paged` / `shared_buffer: true` | delete; declare `aliasing:` instead |
| `slot_allocation: <mode>` | delete |
| KV quant mode/tolerance | delete; runtime-owned (`KvQuantPolicy`) |
| implicit effect domains | declare `pipeline.workflow.effects` |
| native stateful components | add `row_scope` and `cache_affects_state` |
| contracts | add `equivalence` (defaults to `semantic`) |
| state cells bound to a group | add `management: runtime` + `release_boundary` |
| top-level `model.io` (with or without a `pipeline.workflow`) | run `migrate_model_io <package-dir>`; it declares port roles in `components.<c>.ports.roles` and cache ports in the owning `state_service` group |
| `model.io.static_cache` | `state_service.groups.<g>.update` (`indexed_scatter`, `write_indices_ports`, `kv_length_ports`) plus `role`/`layer` on the group's port pairs |
| `pipeline.models.<c>.io` | deleted with the composite IR; use `components.<c>.ports` |

Every removed field is rejected by name, so migration failures are precise rather
than mysterious.

### 18.1 Removal of `model.io` — complete

The staged path this section used to describe has been carried out. What landed:

1. `model.io` was deserialize-only, reachable through one deprecated accessor,
   and rejected beside a workflow. *(done)*
2. The `genai_config.json` importer states its result as a `pipeline.workflow`
   rather than a `model.io` block. Import remains one-way and fail-closed
   (§17); a *foreign* producer's format is converted into this project's one
   representation, which is the whole point of an importer. *(done)*
3. The staged warning step was skipped: with the field deleted there is nothing
   left to warn *about*, and a warning that still loaded the package would have
   kept the second answer alive for another release. *(superseded)*
4. The field and its accessor are deleted, and the key is rejected by name.
   `DecoderAbi` (formerly `ModelIoSpec`) survives as the *resolved* result of
   `decoder_io()`: a derived value with no serialized form, now living beside
   the recognizer that produces it rather than in the serialized schema. *(done)*

**Converting a package.** `migrate_model_io <package-dir>` rewrites a retired
block as the canonical workflow. It is deliberately an offline tool rather than
a load-time step: a runtime that repaired packages in memory would mean the
package on disk said one thing and the runtime executed another, which is the
second authoritative answer this rule exists to prevent. A package whose ports
were previously guessed from its ONNX graph states them once with
`--abi <ports.yaml>`.

**Why a single decoder is not a special case.** It is a workflow with one ONNX
component and one runtime-bound token policy. Its generation loop, cache
aliases, and token emit use the same constructs a multi-component workflow uses,
so there is no decoder-shaped branch anywhere in the runtime — the decoder is
recognized structurally, as the sole component that consumes the autoregressive
sequence and produces logits.

`generation.speculative_decoding.io` is the same class of debt for a proposer
graph and is unchanged here: it describes a model with no workflow component of
its own, so it needs a canonical component before it can follow this path.

#### The `DecoderAbi` type is not a serialized block

The resolved decode ABI and the retired `model.io` key were routinely mistaken
for each other — enough that the type has been renamed `DecoderAbi` and moved
beside the recognizer that produces it. A downstream producer once read
`StaticCacheDecodeSession::new(.., io: Option<&DecoderAbi>)`
as proof that the scatter driver *requires* a serialized `model.io.static_cache`
block, and concluded that the driver and the coexistence rule were jointly
unsatisfiable. They are not. The parameter is the **resolved** decode ABI —
whatever `decoder_io()` returned — and for a workflow package that value is
synthesized from the `state_service` group's `indexed_scatter` update and the
decoder component's port roles. Nothing reads the serialized key.

A type signature cannot show this, so it is pinned against a real graph instead.
`crates/onnx-genai-ort/tests/workflow_derived_static_cache.rs` loads
`tests/fixtures/tiny-llm-scatter-workflow/`, a package with **no `model:` block
at all**, resolves its ABI through `decoder_io()`, and drives the ONNX scatter
fixture through `StaticCacheDecodeSession`: the graph classifies, prefill runs,
and a decode step advances the write cursor by exactly one row position. Its
sibling `tests/fixtures/tiny-llm-scatter/` is the same graph converted by
`migrate_model_io`, so the generated form and the hand-authored one are both
exercised against the same driver.

#### Declare roles; do not transcribe the graph

Making the workflow the sole ABI raises a fair objection from producers: if a
component's port list must be restated in YAML, the package carries a second
copy of something the `.onnx` file already states authoritatively, and the two
can drift. That objection is accepted. The canonical form asks for the part no
graph carries and nothing more.

A `TensorContract` under `ports.inputs`/`ports.outputs` is **optional**. An ONNX
artifact already names its inputs, their element types, and their ranks; a
producer whose artifact is the authority may omit the transcription entirely.
What an ONNX graph cannot state is which of several same-typed ports carries
which *meaning* — `input_ids` and `position_ids` are both rank-2 `int64`, and
nothing in the graph distinguishes the autoregressive sequence from a positional
index. That is why `ports.roles` is required, and it is one line per role:

```yaml
ports:
  roles:
    input_ids: token_ids
```

With that single declaration and no contracts at all, `decoder_io()` resolves
the token port *and* the full `static_cache` ABI — the latter derives from the
state-service group's port aliases, never from the component's port map. This is
covered by `declared_roles_alone_yield_the_scatter_abi`, which strips
`ports.inputs` and `ports.outputs` from the canonical fixture and asserts the
ABI survives intact.

The resolver therefore treats a role declaration as the naming authority and
consults the declared contracts only to break a duplicate-role tie. An earlier
form required a role's port to *also* appear in `ports.inputs`, which quietly
inverted the intent: a producer who declared `input_ids: token_ids` and stopped
there had that declaration discarded, and the runtime fell back to matching the
spelling `input_ids` — reintroducing, in the one place it is least visible, the
name-guessing this resolver exists to abolish. Absence of a transcription is not
a claim that a port does not exist. A role naming a port the graph does not
expose is still caught, against the live session, which is strictly stronger
than any echo of the graph in YAML.

The inverse mistake is the more tempting one, so it is now rejected rather than
tolerated. A producer migrating to the workflow-only form may reasonably assume
transcribing a full `TensorContract` for every port is the *more* complete
declaration and that roles are optional shorthand. It is the other way around:
contracts without roles resolve nothing, `decoder_io()` returns `None`, and the
runtime silently falls back to inferring ports from shapes — the behaviour this
form exists to remove. That failure used to validate cleanly. When a workflow's
*only* neural component owns attention state, the package depends on the
single-decoder lowering, so a missing `token_ids`/`inputs_embeds` role is now a
validation error naming the component and the role. Workflows with several
neural components — an encoder-decoder pair, a speculative draft and verifier, a
TTS talker and code predictor — are exempt: each is driven through its own
invoke bindings, `decoder_io()` deliberately declines to nominate one as "the"
decoder, and requiring a declaration nothing reads would be noise.
`port_contracts_do_not_substitute_for_a_declared_role` pins this.

---

### 18.2 Session state: what tightened, and what it rejects

`scope: session` used to validate on its own. It now has to say **how the next
invocation reaches what the lease keeps** (§12.5a), because a lease nothing
carries is written back on every pass and read by nothing — the package
advertises a conversation that silently restarts every turn.

A document that validated before and does not now is one of these. None is a
rename; each is a statement the document was missing.

| Rejected | Why | Fix |
| --- | --- | --- |
| a **semantic** session cell that no loop carries, no state service group holds, and whose lease names no `continuation` | nothing reaches the kept value | carry it in the loop, bind it to a group, or declare `session.continuation` |
| a session cell naming a `service_group` the document does not declare | the lease has nothing to hold | declare the group, or drop `service_group` |
| a session cell whose group declares no alias for it | same | add the alias, with `input` and `output` ports |
| a group-only-carried cell whose alias declares no `output` port | the lease could be read and never advanced, so every turn replays the first | declare the port that advances the state |
| a group-only-carried cell whose alias `input` port no step binds | the lease would have no reader | invoke the component binding that port |
| a group-only-carried cell whose alias `input` port is bound to a value a step produces | the step would overwrite the lease | bind a workflow input, or carry the cell in the loop |
| a `session.continuation` that is not `scope: session`, `class: semantic`, `management: runtime`, `release_boundary: session`, growing on the final axis, or is also loop-carried | the contract contradicts itself | see §12.5a |
| a `session.continuation` whose `recurrence.max` names anything but a declared input with a value | the bound could not be read before the turn that would exceed it | name a required input, or an optional one with a default |
| two `session.continuation` cells in one workflow | a package has one conversation | keep one |

**Advisory session state is exempt** from the reachability rule: it is droppable
by declaration, so a lease nothing reads costs correctness nothing.

Runtime behaviour changed alongside it, for packages rather than documents:

* a package that publishes a `tokens` output and declares no session state at all
  still **loads and generates**, but `create_session` refuses it — typed, so a
  server answers 409 rather than 500. Stateless generation is untouched;
* a package that publishes no token stream is unaffected and keeps its session
  handle;
* `session_token_count` means the same thing on every backend (§12.5a). Whether
  a request is *charged* for it is a different question, gated on one query —
  `Engine::prepends_session_conversation`, read from the shared classifier. It is
  true for `SessionStateCarrier::PromptContinuation` and nothing else, because
  that is the one carrier whose mechanism is the prompt binding. A decode core
  keeps its conversation in KV, and a loop-carried or group-held lease lives in a
  cache the package bounds itself; charging either would count each turn twice,
  inflating `usage`, halving the usable context and refusing requests at roughly
  half the model's limit.

## 19. Invariants

A conforming document satisfies all of these. Each is validator-enforced, and
enforced on the path a producer actually runs: `load_metadata_package` — the
entry point behind the `validate_metadata` binary and behind package loading —
checks the document-level invariants, not only the pipeline-scoped ones. A rule
reachable only from a direct `validate_metadata` call is not an invariant, it is
a suggestion; the ban on a second serialized ABI was briefly in that state and
`loading_a_package_rejects_a_second_serialized_abi` now holds it in place.

1. **No serialized row identity.** Row axis and layout are derivable from
   `batch_layout`; native compaction (`compact`/`release`) is a mandatory ABI.
2. **Batch compaction is total.** LoRA selection, native grammar state, and
   vision state all survive compaction because all are row-scoped and declared.
3. **Cache dependencies are complete.** LoRA, profile, and multimodal inputs
   cannot be omitted from the derived dependency set.
4. **Speculative rollback reads `speculation_safety`,** never the retry class.
5. **Equivalence class gates substitution.** A `semantic` contract is never
   automatically substituted.
6. **Graph-visible KV ABI and runtime-private storage are distinct.** The former
   is inferred from the graph; the latter is runtime-owned.
7. **Alias legality survives.** `aliasing` retains the real graph constraint that
   `shared_buffer` used to imply.
8. **Canonical identity is stable.** Normalized identity protects external plan
   and checkpoint compatibility, across key order, YAML-versus-JSON encoding,
   numeric spelling, null and empty fields, and ignorable-profile skipping.
   Normalization is syntactic and deliberately conservative: it never merges two
   documents it cannot prove equivalent from syntax alone, because a spurious
   identity change costs one recompile while a spurious match serves a stale
   plan against changed semantics.
9. **Generation override support is structural and fail-loud.**
10. **Constraint dialect and replaceable tokenizer artifacts are represented.**
11. **One-file profile evolution works.** Required profiles fail a reader that
    does not understand them; ignorable profiles are skipped.
12. **Session scope is normative.**
13. **Checkpoint and private P/D transfer are explicitly different paths.**
14. **One serialized graph ABI.** `pipeline.workflow` is the only place a
    package states its executable port ABI. `model.io` is import-only, is read
    only when no workflow is present, and is rejected beside one, so no document
    holds two conflicting answers.

---

## 20. Conformance

### 20.1 Validation

`onnx_genai_metadata::validate_metadata` runs the full semantic validation.
The CLI is:

```sh
cargo run -p onnx-genai-metadata --bin validate_metadata -- <path>
```

### 20.2 Tests

| Requirement | Test |
| --- | --- |
| Invariants 1–13 | `crates/onnx-genai-metadata/tests/redesign_invariants.rs` |
| Removed fields rejected | `metadata_fixtures.rs::removed_row_identity_fields_are_rejected_fail_closed` |
| Row-wise emit / ragged outputs | `metadata_fixtures.rs::row_wise_emit_requires_a_request_aligned_batch_layout` |
| State semantics vs. allocator policy | `metadata_fixtures.rs::state_service_declares_semantics_not_allocator_policy` |
| Adapter artifact compatibility | `crates/onnx-genai-metadata/tests/adapter_artifact_compat.rs` |
| Canonical identity | `crates/onnx-genai-metadata/src/identity.rs` unit tests |
| Cache dependency derivation | `crates/onnx-genai-metadata/src/cache.rs` + `redesign_invariants.rs` |
| Row ABI (`compact`/`release`) | `crates/onnx-genai-engine/src/pipeline/row_state.rs` |
| Generation override fail-loud | `crates/onnx-genai-engine/src/config.rs::generation_contract_tests` |
| Legacy import fail-closed | `crates/onnx-genai-genai-config/src/import.rs` unit tests |
| One canonical graph ABI, bare and composite | `crates/onnx-genai-metadata/tests/canonical_graph_abi.rs` |
| `model.io` beside a workflow rejected | `static_cache_and_fp8_state.rs::model_io_beside_a_workflow_is_refused` |
| Static-cache ABI recognized from the workflow | `static_cache_and_fp8_state.rs::the_decode_abi_is_recognized_from_the_workflow_alone` |
| Per-layer cache order follows declared `layer` | `static_cache_and_fp8_state.rs::per_layer_cache_order_follows_the_declared_layer_index` |
| End-to-end workflows | `crates/onnx-genai-engine/tests/onnx_genai_workflow_conformance.rs` |
| Canonical packages | `tests/fixtures/onnx_genai_workflows/` |
| Row axis derivable without identity | `redesign_invariants.rs::the_row_axis_is_derivable_without_any_serialized_row_identity` |
| Row-scoped carriers survive compaction | `redesign_invariants.rs::every_row_scoped_carrier_survives_batch_compaction` |
| Checkpoint vs. private transfer | `redesign_invariants.rs::portable_checkpoints_are_distinct_from_private_state_transfer` |
| Speculative region spans the loop body | `redesign_invariants.rs::the_speculative_region_covers_every_component_in_the_loop_body` |
| State cannot be exported under an alias | `redesign_invariants.rs::runtime_owned_state_cannot_be_exported_under_an_alias` |
| Identity ignores numeric spelling | `redesign_invariants.rs::semantic_identity_ignores_how_a_number_was_spelled` |
| Absent contract withholds consent | `crates/onnx-genai-engine/src/speculative/mod.rs::equivalence_gate_tests` |

The last four close bypasses found by review rather than by a failing build.
Each was written to fail against the implementation it guards, so a regression
that reopens the bypass fails the suite rather than passing silently.

### 20.3 Schema sync

`schema/inference_metadata.schema.json` is generated:

```sh
cargo run -p onnx-genai-metadata --bin gen_schema
```

`cargo test -p onnx-genai-metadata` fails when the committed schema differs from
the Rust source.

---

## Appendix A: examples

### A.1 Minimal request-aligned decode loop

```yaml
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit]
    effects:
      decode:
        retry: idempotent
        speculation_safety: rewindable
    inputs:
      request.tokens:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, sequence]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: runtime, version: "1.0", role: input_ids }
        source: { kind: request }
    state:
      cache:
        contract:
          dtype: float32
          rank: 4
          shape: [batch, heads, sequence, head_dim]
          batch_layout: { kind: runtime_sequence_state }
        scope: invocation
        initializer: cache.initial
        recurrence: { kind: invariant }
        service_group: decoder_cache
        management: runtime
        release_boundary: invocation
    serving:
      state_service:
        groups:
          decoder_cache:
            kind: full_attention
            sequence_axis: 2
            layout: bnsh
            aliasing: permitted
```

### A.2 Native grammar component

```yaml
grammar:
  implementation: { kind: binding }
  row_scope: { axis: 0, stateful: true }
  effects: [grammar]
  cache_affects_state: [grammar.parser_table]
  ports:
    inputs:
      token:
        dtype: int64
        rank: 2
        shape: [batch, generated]
        batch_layout: { kind: request_aligned, axis: 0 }
    outputs:
      guided:
        dtype: int64
        rank: 2
        shape: [batch, generated]
        batch_layout: { kind: request_aligned, axis: 0 }
```

It is native, it holds per-request parser state, it declares its row axis so the
runtime can compact it, and it declares the non-dataflow fact that its parser
table participates in cache identity. Nothing about it is a special case.

### A.3 Workflow pattern catalogue

This section is self-contained. It shows how the five workflow operations
(`sequence`, `invoke`, pre-test `loop`, `branch`, and `emit`) compose into the
major inference workflow families. The names in the examples are descriptive,
not dispatch keys: a runtime executes the typed graph and never switches on
`whisper`, `diffusion`, `gemma`, or any other model-family string.

The catalogue is a constructive representability argument, not a claim that
every future model is already implemented. A workflow family is representable
when its computation can be decomposed into:

1. typed component invocations;
2. SSA values and explicitly scoped state cells;
3. structural control flow using the five operations;
4. typed preprocessing and postprocessing components; and
5. request-aligned, packed, invariant, or runtime-sequence-state layouts.

If a model requires a semantic operation outside those primitives, the package
must fail validation until the contract is extended. A model-family-specific
runtime branch is never the fallback.

#### A.3.1 Autoregressive text, MoE, hybrid attention, and shared KV

```text
sequence
  invoke tokenizer_or_accept_tokens
  invoke prefill
  loop while any_row_active
    invoke decoder
    invoke sampler
    invoke stop_test
    emit token row-wise
```

The decoder may contain dense, MoE, recurrent, sliding-attention, or
full-attention layers; those are component internals. Graph-visible recurrent
state is split into semantic state groups:

```yaml
serving:
  state_service:
    groups:
      sliding:
        kind: sliding_attention
        sequence_axis: 2
        layout: bnsh
      global:
        kind: full_attention
        sequence_axis: 2
        layout: bnsh
      recurrent:
        kind: recurrent
        layout: bhkv
        update: { kind: replace }
```

Each cache-owning graph port binds to one cell in the appropriate group.
Different groups may have different head dimensions. A layer that borrows
another layer's K/V has no independent cells; the graph wiring identifies the
source. Continuous batching changes runtime row placement and physical storage,
not this workflow. Request-aligned tokens, sampler inputs, LoRA routes, and stop
flags compact together; runtime-sequence-state cells follow the same row
permutation through the state-service ABI.

#### A.3.2 Optional multimodal generation

```text
sequence
  branch image_is_present
    then
      invoke image_preprocess
      invoke vision_encoder
      invoke image_token_mixer
    else
      invoke text_only_embedding
  invoke decoder_prefill
  loop while any_row_active
    invoke decoder
    invoke sampler
    emit token row-wise
```

`image_is_present` comes from the generic input-presence capability. An audio
or video encoder is the same pattern with a different typed preprocessing
program and tensor contract. Packed media tokens carry offsets or owner
mappings; they do not invent serialized request IDs. Cross-attention or
projected media state is request-aligned and loop-invariant, so it can be reused
through decode and moved during compaction. A single package can therefore mix
media-present and media-absent rows without selecting a different model family.

#### A.3.3 Encoder-decoder speech recognition

```text
sequence
  invoke audio_preprocess(encoded_audio -> log_mel)
  invoke encoder(log_mel -> encoder_hidden)
  invoke decoder_prefill(encoder_hidden, prompt_tokens)
  loop while any_row_active
    invoke decoder(encoder_hidden, token, self_cache)
    invoke sampler
    emit token row-wise
```

The encoder output is a request-aligned, loop-invariant state cell. Decoder
self-attention state grows or slides; cross-attention state is invariant or is
materialized once when the graph exposes separate cross K/V ports. Unequal
audio lengths and early EOS are represented by per-row lengths and active
masks, so one row ending cannot truncate another.

#### A.3.4 CTC transcription and other non-generative tasks

```text
sequence
  invoke audio_preprocess(encoded_audio -> samples, attention_mask)
  invoke acoustic_encoder(samples, attention_mask -> logits, frame_lengths)
  invoke ctc_decode(logits, frame_lengths -> token_ids)
  emit transcript
```

There is no generation loop and no carried state. The task profile declares the
decoding facts needed to interpret logits:

```yaml
profiles:
  transcription:
    task: speech_to_text
    decoding:
      kind: ctc
      blank_id: 0
      collapse_repeats: true
      time_axis: 1
      class_axis: 2
      lengths: frame_lengths
```

Embedding, reranking, classification, reward scoring, detection, and ordinary
encoder inference are the same one-pass shape: preprocess if needed, invoke one
or more components, then emit typed outputs. They use task-specific profiles
instead of pretending to be token generation.

#### A.3.5 Image diffusion with CFG and multistep solvers

```text
sequence
  invoke text_encoder
  invoke schedule
  invoke latent_initializer(seed -> latent, rng_offset)
  loop while step < step_count
    invoke denoiser(latent, conditional_embedding -> conditional_noise)
    invoke denoiser(latent, unconditional_embedding -> unconditional_noise)
    invoke guidance_combine
    invoke solver_step(latent, guided_noise, history -> next_latent, next_history)
    emit latent_trajectory append
  invoke vae_decoder
  emit image
```

Two denoiser invocations express classifier-free guidance without falsely
claiming that two physical rows are two requests. First- and higher-order
solvers differ only in whether the loop carries history cells. The schedule,
RNG, guidance equation, and solver are replaceable typed components, so Euler,
DDIM, flow matching, and DPM-style methods do not require strategy enums in the
workflow IR.

#### A.3.6 Image editing and other multi-source diffusion

```text
sequence
  invoke source_image_preprocess
  invoke source_vae_encoder
  invoke source_latent_pack
  invoke text_encoder
  invoke vision_encoder
  invoke schedule
  invoke latent_initializer
  loop while step < step_count
    invoke conditioning_concat(source_latent, target_latent, embeddings)
    invoke denoiser
    invoke guidance_combine
    invoke solver_step
  invoke target_latent_unpack
  invoke vae_decoder
  emit image
```

Source and target tokens have explicit slice/packing components and contracts.
ControlNet, adapters, masks, depth maps, and reference images add typed
conditioning invocations and values to the same graph. They do not change the
control-flow vocabulary.

#### A.3.7 Video generation and causal chunked decode

```text
sequence
  invoke text_encoder
  invoke latent_initializer(rank_5_shape)
  loop while denoise_step < step_count
    invoke video_denoiser
    invoke solver_step
  loop while decode_chunk < chunk_count
    invoke causal_video_vae(latent_chunk, conv_cache -> frames, next_conv_cache)
    emit frames append axis=2
```

Video latent state is rank 5 and the emitted output grows on its declared time
axis, not an assumed final axis. Causal VAE convolution caches are ordinary
state cells with invocation lifetime. Spatial scale facts belong to port
contracts or component metadata. Text-to-video, image-to-video, and
video-to-video differ in their conditioning prefix, not in denoising or
temporal publication semantics.

#### A.3.8 Masked or discrete diffusion

```text
sequence
  invoke token_initializer
  loop while masked_positions_remain
    invoke masked_model
    invoke confidence_or_schedule
    invoke token_update
    emit token_trajectory append
  emit completed_tokens
```

This proves that a loop need not be autoregressive or operate on continuous
latents. The carried value is the partially filled token grid; termination
comes from the mask state rather than EOS.

#### A.3.9 TTS, neural codecs, and full-duplex speech-to-speech

Text-to-speech composes a text-token loop, an acoustic-token loop, and codec
decode:

```text
sequence
  loop while text_or_semantic_stream_active
    invoke language_model
    invoke sampler
  loop while acoustic_stream_active
    invoke acoustic_model
    invoke sampler
  invoke codec_decoder
  emit audio
```

A full-duplex model repeats that structure per audio frame and nests a fixed
acoustic-substep loop:

```text
loop while session_active
  invoke user_codec_encoder
  invoke temporal_model
  loop while codebook_index < codebook_count
    invoke acoustic_depformer
    invoke sampler
  invoke delay_ring_commit
  invoke agent_codec_decoder
  emit audio guarded by readiness
```

Interleaved text and audio streams are separate values with explicit delays.
Temporal KV, delay rings, and codec convolution state are heterogeneous cells
with session or invocation scope. A phase-boundary codec reset is represented
as a release boundary, not inferred from a model name.

#### A.3.10 Speculative decoding

```text
loop while any_row_active
  sequence
    invoke proposer
    invoke target_verifier
    branch proposal_accepted
      then invoke commit
      else invoke rollback_and_correct
    emit accepted_tokens row-wise
```

The speculative region names every component whose state must rewind.
`speculation_safety` states whether that rewind is legal; retry idempotency is a
separate property. Proposal width and scheduling remain runtime choices.

#### A.3.11 LoRA and request-selected adapters

LoRA does not create a second workflow. Adapter selection, ordered composition,
and scales are request-aligned inputs to the affected component invocation:

```text
invoke decoder(
  hidden,
  adapter_segments,
  adapter_counts,
  adapter_scales,
  adapter_active
)
```

The package's target manifest states which parameters are adaptable. The
runtime owns loading, caching, eviction, and kernel selection. Because adapter
selection participates in dataflow and cache dependencies, two requests with
different adapters cannot accidentally share an incompatible prefix cache.

### A.4 Coverage and proof levels

Representability, executability, and model correctness are different claims:

| Level | Meaning |
| --- | --- |
| **R** | A complete document validates and contains all required dataflow, state, and control flow. |
| **X** | The generic runtime executes a deterministic package through the complete workflow. |
| **W** | A real-weight run matches an upstream implementation numerically or token-for-token. |
| **B** | Multiple unequal request rows, permutation, compaction, or early completion are verified. |

The current evidence is:

| Workflow family | R | X | W | B | Evidence represented by the example |
| --- | --- | --- | --- | --- | --- |
| Autoregressive decoder | yes | yes | yes | yes | Prefill/decode, heterogeneous KV, shared KV, row-wise EOS |
| Optional VLM | yes | yes | yes | partial | Image-present/text-only branch; workflow execution is currently per request |
| Encoder-decoder ASR | yes | yes | yes | yes | Raw audio, encoder state, cached decode, unequal early EOS |
| CTC ASR | yes | yes | yes | yes | One-pass logits, frame lengths, collapse, transcript |
| Image diffusion | yes | yes | yes | yes | CFG, RNG, Euler/DPM-style history, VAE decode |
| Image editing | yes | yes | yes | partial | Source/target packing and full denoise; distinct source images depend on caller/upstream batching |
| Video diffusion | yes | yes | yes | yes | Rank-5 latent, causal chunk decode, non-terminal emit axis |
| Masked diffusion | yes | yes | synthetic | yes | Discrete iterative refinement |
| TTS and codec | yes | yes | producer graphs | yes | Nested token generation and audio publication |
| Full-duplex speech | yes | component path | yes | trace parity | Nested frame/acoustic loops; full engine-driven session loop remains pending |
| Speculative decoding | yes | yes | synthetic | yes | Accept, reject, rollback, correction |
| LoRA composition | yes | yes | artifact parity | yes | Ordered adapters survive row compaction |
| Embedding/classification/reranking | yes | trivial sequence | model-dependent | ordinary batching | Task profile plus invoke and emit |
| Fixed-capacity (static) cache | yes | yes | real weights | yes | Indexed scatter, per-row cursors, unequal lengths, inactive-row freeze, rewind |
| FP8 cache state | yes | provider-dependent | no | n/a | Declared, validated, allocated, and bound; compute awaits an FP8 kernel |

“Partial” and “pending” are deliberate. They identify runtime or upstream
limitations without weakening the representability claim. In particular:

- Foundry Local serializes workflow requests, so it is not evidence for
  continuous batching.
- A full-duplex speech graph is representable and its real components have
  trace parity, but the generic engine has not yet driven the complete
  long-lived frame loop.
- Model preprocessing must be fully declared. Supplying precomputed embeddings
  proves the downstream workflow but not a raw-media request path.
- Runtime-private storage formats such as paged KV do not require a new
  workflow pattern; they are implementations of the declared state-service
  semantics.
- FP8 state is representable, validated, allocatable, and bindable, and its
  “provider-dependent” execution level is a measurement, not a hedge: the CPU
  provider has no FP8 `ScatterND` kernel, so a static FP8 cache fails to load
  there with the provider's own type error, and the CUDA provider (1.29.0) has
  no `float8_e4m3fn` in its GQA `past`/`present` type list, so the node is left
  unassigned at partitioning and initialization fails without ever launching a
  kernel. §12.2c records both shapes, why the CUDA message is the more
  misleading of the two, and why neither must be turned into a validation error.
- The fixed-capacity cache row claims **real weights** on the strength of a CUDA
  run of a real static-cache export (onnxruntime-gpu 1.29.0, H200), not of an
  upstream comparison: a `B=2` prefill reproduces the per-row `B=1` result
  exactly (max |Δ| = 0), `TensorScatter` writes land inside the declared
  `[0, nonpad)` prefix and nowhere else, decode with divergent per-row cursors
  (one row advancing while another has finished) holds max |Δ| ≤ 3.1e-6 against
  the single-row reference across three steps, and a finished row reclaims its
  own last slot on each subsequent write without ever leaking past it. That is
  the `indexed_scatter` discipline of §12.2b executing as specified on a real
  export, including the inactive-row behaviour the declaration promises.

The canonical executable packages cover decoder, VLM, image diffusion, guided
diffusion, masked diffusion, speculative decoding, codec, TTS, video, and
adapters. Each declares its graph ABI only in its workflow, so the bare
single-ONNX decoder package and the three-graph VLM package are evidence for the
same representation rather than for two (§4.1a,
`crates/onnx-genai-metadata/tests/canonical_graph_abi.rs`). CTC and audio
preprocessing have profile/validator coverage, while real-weight Whisper, CTC,
image-edit, diffusion, video, VLM, and heterogeneous KV runs establish the
higher proof levels. These examples therefore cover the known workflow *shapes*
without claiming that an untested model automatically has a correct exporter.

---

## Appendix B: decision index

The original approved decisions map to sections as follows.

| Decisions | Section |
| --- | --- |
| 1–6 (scope, compatibility, one document, profiles) | [§1](#1-goals-and-non-goals), [§4](#4-schema-model) |
| 7–11 (component implementations, preprocessing) | [§7](#7-components-and-implementation-substitution), [§15](#15-preprocessing-and-generated-inputs) |
| 12–19 (runtime-owned execution) | [§3](#3-ownership-layers), [§5](#5-ownership-layers-in-the-schema) |
| 20–25 (batch and request identity) | [§8](#8-batching-varlen-and-paged-attention), [§10](#10-multimodal-encoders), [§11](#11-cache-correctness-dependencies) |
| 26–32 (state) | [§12](#12-state) |
| 33–36 (generation, speculative, LoRA, state quantization) | [§9](#9-adapters-lora), [§13](#13-speculative-execution), [§14](#14-generation), [§12.7](#127-state-representation-and-quantization) |
| 37–39 (distributed execution) | [§16](#16-distributed-execution) |
