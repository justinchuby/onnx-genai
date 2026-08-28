# Inference metadata specification

Status: **Normative**

This is the normative specification of the portable inference-metadata contract.
It defines the target contract independently of any current parser, runtime,
exporter, or optimization. Implementation and conformance work is tracked in
[#2303](https://github.com/justinchuby/onnx-genai/issues/2303).

Requirement keywords (**MUST**, **MUST NOT**, **SHOULD**, **MAY**) are used in
the RFC 2119 sense.

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

1. **Deployment policy.** Device-memory and allocator budgets, placement,
   execution providers, paging, tiering, quality of service, and deadlines are
   not metadata. Static artifact footprint bounds are metadata; measured
   resource availability is not. See [§5](#5-ownership-layers-in-the-schema).
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
7. **API or network streaming.** HTTP/SSE/gRPC/WebSocket framing, flushing,
   buffering, retry, reconnect, backpressure, and disconnect behavior are
   runtime/API contracts. Metadata describes transport-neutral publication to
   the workflow output boundary, not delivery beyond that boundary; see
   [§6.4](#64-workflow-output-publication-and-revisions).

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
- correctness dependencies for cache reuse;
- typed workflow output publication and revision semantics;
- the exact versioned tool render/parse protocol a package requires;
- cross-invocation state identity, lifecycle, initialization, dataflow, update,
  and commit/rollback semantics.

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
- diagnostics, determinism levels, and trust policy;
- session-turn transaction mechanics and recovery;
- tool-protocol adapter implementation;
- API/network delivery, buffering, and backpressure;
- physical state transport and retention.

### 3.3 Caller

The caller supplies **request data**: prompts, images, audio, grammars, JSON
schemas, adapter selections, and overrides of generation fields the package
structurally exposes ([§14](#14-generation)).

For tools, the caller also supplies the offered functions, descriptions, JSON
Schemas, `tool_choice`, prior calls, and tool results. Any externally supplied
state is ordinary typed request data connected to a declared workflow input.

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

### 4.2a Reserved fields, local names, and extension identifiers

Metadata has five different naming classes. They are not interchangeable:

| Naming class | Who defines it | Examples | Reader behavior |
| --- | --- | --- | --- |
| **Reserved schema field** | This specification and generated JSON Schema. | `pipeline`, `workflow`, `inputs`, `contract`, `dtype`, `state`, `recurrence`, `special_tokens`. | Unknown keys fail because typed core objects deny unknown fields. A producer cannot add `workflow.my_option` or `contract.vendor_hint`. |
| **Package-local identifier/reference** | The package author, within the map or scope that owns it. | Keys under `workflow.inputs`, `outputs`, `components`, `state`, `effects`, `serving.state_service.groups`, `profiles`, adapter artifacts, and branch cases; SSA value names; component and state-group references. | The spelling is author-defined, but every reference must resolve, names must be unique in their scope, and the runtime must not infer semantics from the spelling. |
| **Artifact-defined name** | The referenced artifact. | ONNX input/output and initializer names used by component ports, invoke bindings, optional inputs, state aliases, and adapter targets. | Must match the artifact exactly. It is not a portable semantic vocabulary and must never be guessed from a model family. |
| **Extensible semantic identifier** | A registered producer/runtime extension, normally owner-qualified and versioned. | Capability strings, adapter ABI keys, component contract IDs, adapter application/loader IDs, checkpoint adapter IDs, constraint dialects, profile kinds, and extensible operation/vocabulary strings. | Built-ins have normative semantics. Unknown extensions may parse where the schema declares an extension branch, but execution must fail closed unless the runtime implements that exact identifier/version. |
| **Ordinary data value** | Package contents or request contract. | Artifact locations, source URIs, revisions, symbolic dimension labels, provenance labels, and numeric bounds. | Treated as data under the surrounding reserved field; it does not create a new schema field or runtime capability. |

Two rules prevent ambiguity:

1. A `BTreeMap<String, ...>` does **not** automatically mean arbitrary schema
   extension. Its keys are customizable only for the purpose documented by that
   map. For example, `workflow.components.decoder` may use a different local
   component name, while `ComponentContract.bindings.logits` is a semantic role
   defined by that contract.
2. A free-form `String` does **not** automatically mean the runtime may ignore
   an unfamiliar value. Known vocabularies are reserved; extension branches are
   explicit. An owner-defined identifier should use an owner-qualified name
   such as `com.example.audio-preprocess` and a separate version, and a runtime
   that has not registered it must reject the package.

Comments are unrestricted review prose and have no parsed semantics. Adding a
comment never creates an extension point or changes the canonical YAML object.

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

### 4.3a Capability admission

A capability identifier names versioned semantic behavior required for correct
execution. It is not a model name, implementation preference, deployment policy,
or feature-enable request. Required identifiers and versions **MUST** be selected
exactly; an unsupported or ambiguous requirement **MUST** fail before execution.

Requirements that follow from workflow structure **MUST** be derived rather than
duplicated as manually authored flags. Extension identifiers are permitted only
on declared extension surfaces and remain fail-closed. Component contracts,
adapter ABIs, and state bounds keep their own identities and do not become
capabilities merely because a runtime implements them.

Output publication is governed by the declared output protocol and ordinary
`emit`; it has no separate streaming or transport-delivery capability.

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

### 4.5 Tensor shape is the sole rank authority

`shape` is required and its length is the tensor rank. Each dimension is fixed,
symbolic, or explicitly unconstrained as `Any`; separate `Any` occurrences do
not imply equal extents. `shape: []` is scalar and `[Any]` is rank one.

Serialized `rank` **MUST NOT** exist. A reader **MUST** reject an omitted shape,
a retired `rank` field, or an attempt to infer placeholder axes.

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
| `emit` | publishes a typed value to its declared workflow output boundary |

Control-flow location defines lifecycle. A loop's authored `setup` executes
exactly once before its first body evaluation, including an empty or zero-trip
loop. An optimized execution path **MUST** preserve that exactly-once behavior or
decline only the optimization and use generic workflow execution. A package
**MUST NOT** be rejected solely because a loop setup is non-empty.

`transfer` is internal lowered IR only. The planner introduces transfers after
placement; metadata **MUST NOT** serialize one.

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

### 6.4 Workflow output publication and revisions

`emit` publishes a typed SSA value to a declared workflow output boundary. The
publication is outward and observable through the host runtime contract but is
transport-neutral: metadata does not define HTTP/SSE/gRPC/WebSocket framing,
flush timing, buffering, reconnect, retry, or backpressure.

Each output declares exactly one protocol family. Every emit targeting it
selects only an operation allowed by that family and preserves its typed value,
guard, valid length, and effect ordering:

| Family | Allowed publication semantics |
| --- | --- |
| materialized value | replace, or append on a declared growth axis |
| discrete events | ordered typed occurrences |
| typed revisions | versioned `append`, `replace`, `retract`, or optional `finalize` |

An emit **MUST NOT** redefine the output family. Output identity, row behavior,
payload contract, and growth rules are output-level invariants; workflow
control/dataflow and effect ordering determine publication order.

A typed revision envelope identifies its output/stream, enclosing transaction,
deterministic sequence, revision, required lineage/base, operation, and typed
payload where applicable. Unknown versions or operations, illegal bases,
duplicate `finalize`, post-finalize updates, and family/operation mismatches
**MUST** fail closed.

`abort_to_baseline` is a typed turn/transaction outcome, not a
revision-envelope operation. It identifies the aborted transaction and its
recorded committed baseline, and invalidates every provisional publication
owned by that transaction.

`finalize` closes one revision stream early. It is optional and remains
provisional until the enclosing turn commits. Successful turn commit finalizes
ordinary outputs and every still-open revision stream by default. Turn abort
undoes every provisional publication, including an early close, to the output
heads recorded by the transaction described in [§12.5](#125-sessions).
Per-stream `retract` invalidates named revision lineage; it is not turn abort.

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

### 8.1 Semantic batching contract

Metadata declares request axes, row-scoped state, lengths/masks/write-index
companions, inactive-row preservation, and legal compaction or release. It never
declares request identity, physical slots, fairness, capacity allocation,
backfill timing, or scheduler policy.

A shared live-row forward is an optimization. It may advance one logical decode
step for each eligible live row, while prompt/context forwards may update state
without publishing a token. Eligibility is derived from the complete workflow,
state, and output contracts. If a selected backend cannot execute a correct
shared forward, the runtime **MUST** use isolated execution unless the package
independently requires an unsupported semantic capability.

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

`token_packed` carries an ordered `levels` chain. Each level declares the
offsets and positional ownership needed to recover the next enclosing level.

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

### 8.6 Compaction and release

Every row-scoped state or effect participant admitted to a shared batch **MUST**
support semantic selection/reordering and release. Selection is positional,
may repeat a source row, and applies consistently to every row-aligned carrier;
out-of-range positions fail closed. Inactive or released rows **MUST NOT** mutate
state or receive another row's publications.

These semantics define shared-batch eligibility, not package deployability and
not a particular language trait or method signature. A runtime lacking them for
one execution path declines shared batching and preserves behavior through
isolated execution.

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

Because the selection tensors are request-aligned, they follow the same
selection, compaction, and release semantics as every other row-scoped value.

---

## 10. Multimodal encoders

### 10.1 Independent encoder batching

Image, video, audio, and text encoders may batch on a different axis from the
decoder: a request may carry zero, one, or many items, and items from many
requests may pack together. Dense encoders use `request_aligned`; ragged item
counts use `token_packed`:

```yaml
image_features:
  contract:
    dtype: float32
    shape: [packed_items, features]
    batch_layout:
      kind: token_packed
      axis: 0
      levels:
        - offsets: image_offsets   # cu_seqlens-style prefix offsets
          owner: image_owner       # per-item owning row
```

An `offsets`/`owner` pair maps packed items to the next enclosing level. One
level is sufficient when items sit directly in rows; additional declared levels
represent nested ownership such as frames within clips.

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

### 10.4 Generic component batching

A component may declare that grouping multiple request-owned items into one
invocation is semantically equivalent to isolated execution. The declaration
states only portable facts: grouping bounds, dimensions that must agree,
padding with explicit valid lengths, and ordered packed-ownership levels.
Absence of that declaration means isolated invocation.

Packed and padded values **MUST** carry enough typed companions to reconstruct
request-local spans without serialized request identities. Companions and
payloads must agree on dtype, rank, extents, ownership order, and validity; a
reader **MUST** reject ambiguous, missing, inconsistent, non-monotone, or
out-of-range structure rather than guess or clamp.

The runtime decides whether and how to form groups. It **MUST NOT** change input
semantics to make items compatible, and unsupported grouped execution falls back
to isolated execution. Placement, host/device transfers, allocation, group size,
and scheduling remain runtime policy.

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

Every value that can affect state **MUST** participate in the derived dependency
set.

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

The same rule applies to tiering. Metadata **MUST NOT** name CPU/device/disk
spill targets, budgets, watermarks, migration thresholds, compression,
prefetch, or load-versus-recompute policy. It declares only semantic state,
portable checkpoint form, reuse/eviction legality, and bounds. A runtime may
move eligible storage between tiers only while preserving the exact semantic
value. Spillability and legal prefix eviction are separate questions: being
movable does not make state disposable, and being recomputable does not select a
placement tier.

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

### 12.2c Cache element types

A package may declare any state element type admitted by the schema and graph
ABI. Metadata validation **MUST NOT** reject a portable type merely because a
particular backend lacks an allocator or kernel. Backend capability is resolved
at admission, with an actionable error naming the incompatible state, operator,
type, and backend.

A producer **MUST NOT** silently substitute another type when the requested
graph ABI cannot represent the declared state. Runtime-private cache formats may
be narrower than the portable contract, but that limits only those backends.

### 12.3 Capabilities, cascade, fork, and reuse

State-group capabilities declare legal rollback bounds, snapshots, and forks;
`cascade` names all groups that must move with an operation. The runtime chooses
when and how to exercise them.

A session fork creates a child at a declared position with every semantic state
and effect participant reproduced there, after which parent and child mutate
independently. Copy-on-write is optional. A prefix-cache hit only reuses
compatible computation and creates no session identity. Fork and
`reuse.prefix_reusable` **MUST NOT** imply one another.

A runtime exposes fork only when every participant and transitive cascade can be
forked. Otherwise it declines fork before creating the child; it does not fork a
subset.

### 12.4 Lifetime and release boundary

- **Ordinary tensors** use SSA liveness. No annotation.
- **Runtime-managed and external state** has no last reader in the dataflow, so
  it **MUST** declare a `release_boundary` (`invocation` | `session`).

The validator enforces the corollaries: a workflow-owned cell **MUST NOT**
declare a release boundary (it is freed by liveness), a cell that binds a state
service group **MUST** be `management: runtime` (the group is the runtime's
storage), and a `session` release boundary requires session scope.

### 12.5 Sessions

Session metadata declares typed semantic state, lifecycle, initialization or
restore bindings, readers, writers, update discipline, final writer, release,
and transaction behavior. Runtime session IDs, locks, TTL, storage, placement,
retention, and physical row/cache identity are not package semantics.

For every state value, initialization and restore **MUST** be unambiguous.
Readers and writers are derived from workflow dataflow, component bindings,
loop carries, and state-service aliases. All control-flow paths that write a
value **MUST** join to one final writer before commit. Author-defined map keys,
component order, filenames, modality names, and model-family names have no
semantic authority. Storage management does not redefine dataflow.

Readers MUST resolve those declarations to one typed state plan before
execution. The plan carries the state identity, lifecycle and release boundary,
source binding, reader and writer edges, update relation, final writer,
snapshot/fork eligibility, and transaction participation. Validation and every
executor consume that same plan; no allocator, storage placement, or
storage-management field may select a source or final writer independently.
Consequently, a caller-provided seed and a value carried from an earlier
invocation differ only in which declared binding supplies the plan's source,
not in their validation or execution mechanism.

At turn admission, before mutation or publication, the runtime records one typed
committed baseline for the complete state/effect write set and each workflow
output's committed head, cursor, lineage, and closure state. The baseline and
turn have stable identities. Provisional publications carry transaction and
revision lineage; rollback targets **MUST NOT** be inferred from names, payloads,
emit order, or container order.

Commit atomically advances every participating state, effect, and output head.
Abort, cancellation, execution failure, or commit failure restores/retracts the
whole turn to its recorded baseline. `commit_only` exposes nothing before commit;
`provisional_revisions` may expose typed provisional publications and the typed
`abort_to_baseline` turn/transaction outcome defined in
[§6.4](#64-workflow-output-publication-and-revisions). A participant unable to
join the transaction causes admission to fail before mutation. An exclusive
lease is a concurrency primitive, not the transaction itself.

When a runtime advances several positional rows through one shared forward,
each admitted row remains an independent turn participant. Its baseline
**MUST** cover every mutable row-owned value the turn can touch, including
semantic state and cache residency, logical cursors and sequence bookkeeping,
random/constraint state, staged output and completion state, and
runtime-visible journals or events. Commit publishes that row's complete write
set; abort restores only that row and discards its staged publications. Rows
whose turns did not abort retain their committed progress and continue
independently. A retry is admitted from the restored baseline and **MUST**
produce the same result under the package's declared determinism contract.

Shared execution does not create a second preparation authority. Canonical
prepared inputs remain immutable, a positional row-selection plan is a
transient execution view over them, and replay applies the same plan to the
same prepared source values. Transaction snapshots and restoration follow the
selected rows without rewriting prepared inputs, inferring scheduler identity,
or converting optional shared execution into a package requirement. If any
selected participant cannot snapshot and restore completely, the runtime
declines the shared optimization before mutation and executes an equivalent
isolated path.

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

#### Prefill/decode lifecycle and final writer

Separate prefill and decode artifacts are ordinary workflow components. On any
invocation, prior restored state, explicit external initialization, or empty
initialization is selected by typed dataflow. Prefill consumes the selected
state when its graph ABI permits, and its result flows to decode. The turn
commits the dataflow-derived final writer.

If the artifacts use different private state representations, a compatible
versioned conversion **MUST** be declared or the pairing rejected before
execution. A portable checkpoint is not an implicit hot-path conversion.

### 12.7 State representation and quantization

Graph-visible state representation is **inferred from the graph**: tensor dtype,
Q/DQ nodes, scale ports, `Attention` attributes, and the declared graph ABI. It
is not a separate metadata field, because two sources for one fact is one source
too many.

The runtime-private cache representation — including whether the runtime
quantizes its own cache — is **runtime-owned and runtime-validated**. Metadata
carries no KV quantization mode and no KV quantization tolerance.

This does not touch **model-weight** quantization intent, which remains in
`quantization` and is a published property of the package.

### 12.8 Stateful token-context features

A package may derive deterministic features from explicit token identities and
bounded token history through ordinary typed components and state dataflow.
N-gram hashing, learned embeddings, projections, gating, convolution, and
residual injection are graph semantics, not architecture or model identifiers.

When embeddings and token-context features are both consumed, token IDs **MUST**
be bound explicitly; reverse embedding lookup is forbidden. Each history is a
separate semantic state participant with declared initialization, recurrence,
update, lifecycle, compaction, checkpoint/fork, and rollback behavior.
Full-sequence, chunked-prefill, and decode execution **MUST** agree at equivalent
boundaries. Geometry and table contents are package facts; placement, sharding,
offload, prefetch, and memory budgets are runtime policy.

---

## 13. Speculative execution

### 13.1 What metadata declares

The canonical speculative contract declares proposer and target components,
typed port bindings, vocabulary relation, immutable shared weights, shared or
private state, rollback participants and bound, and distribution equivalence.
Those facts **MUST NOT** be inferred from filenames, model families, or proposal
names. Overlapping legacy discovery/configuration surfaces are not alternative
portable authorities.

Proposal forms may use flat blocks or candidate trees. A candidate-tree proposer
must declare candidate tokens, parent topology or mask, verification outputs,
accepted path, proposal probabilities required for sampling, and rollback of all
affected state. Unknown proposal-contract identities or versions fail closed.

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

### 13.7 DFlash

DFlash is a distinct flat-block proposer conditioned on declared target hidden
features. Its contract declares conditioning, masked candidate positions,
draft-private state, candidate tokens, proposal probabilities, immutable weight
sharing, and accepted-prefix rollback for target and draft state.

The built-in structural identities are exact:

- `dflash_flat_block` version `"1"` is base DFlash: one verifier-produced
  anchor followed by masked positions predicted in parallel. Target hidden
  outputs are named with component/output provenance and concatenation axis;
  the target embedding and output projection are immutable initializer
  relationships, not copied proposer weights.
- version `"2"` is the selector/convolution form. It declares the selected
  path, top-k candidate ids, conditional probabilities, selector rank/top-k,
  grouped convolution kernel/group sizes, and the anchor predecessor rule.
  Merely finding optional tensors or familiar names never upgrades version 1.

Every mutable proposer/target state-service alias belongs to the one
`rollback_state` participant set. `accepted_prefix_state` gives exactly one
commit mechanism for each member: sequence-axis truncation, or an explicit
per-prefix snapshot output for fixed recurrent/convolution state.

The target verifies candidates exactly. Greedy execution accepts the longest
matching prefix; sampling is permitted only when the declared proposal
probabilities support distribution-preserving correction. EOS, context limits,
zero/partial/full acceptance, cancellation, and failure preserve the transaction
contract.

Shared batching and compaction are optional optimizations. A runtime may execute
rows in isolation without changing DFlash conformance. Candidate-tree proposals
are independent typed forms and neither block nor redefine DFlash; a DFlash
variant with additional selector or convolution semantics requires its own
versioned contract.

The executable Qwen3.8-27B reference fixture takes the published DFlash 2
checkpoint geometry as its source (block size 8, five target taps
5/19/33/47/61, five drafter layers), and exercises the common base flow:
target-feature injection, shared embedding and LM head, and parallel block
prediction. It deliberately declares version 1 and does not claim the version-2
selector/convolution extension. Vocabulary, hidden width, and learned tensors
are reduced deterministically; this is structural/equation conformance, not
official-weight numerical parity.

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

`preprocessing.image`, `preprocessing.video`, and `preprocessing.audio` are the
declared programs. Image and video share one vision program type; video adds
`sample_frames` and `pad_frames` to the spatial transform vocabulary rather than
duplicating it. All three programs are ordered generic `transforms` plus named
`outputs` that bind program-local values to workflow SSA names. Every parameter
is model **data**: a mel-bin count, an FFT size, a sample rate, a frame-sampling
rule, and a target window all live in the package. A CTC acoustic model declares
`resample`/`downmix`/`zero_mean_unit_variance` over raw samples; an
encoder-decoder speech model declares `resample`/`pad`/`log_mel` over a fixed
window. The runtime reads the same fields either way and never dispatches on
which one it is.

In workflow metadata each program is materialized by one manifest-pinned
adapter invocation (`onnx-genai.image-preprocess@1`,
`onnx-genai.video-preprocess@1`, or `onnx-genai.audio-preprocess@1`) and every
declared output **MUST** carry a `TensorContract` compatible with its adapter
port. A package **MAY** instead supply an already-featurized media tensor with
the full contract and no program; there the input's shape states the geometry.
Text encoders use the same typed-tensor path.

A program output can declare `pack_offsets`, `pack_owner`, and
`valid_lengths`. These level-agnostic roles let image, video, and audio programs
name the companions referenced by `token_packed` and `padding`. Existing audio
roles such as `frame_lengths` and `sample_lengths` remain valid because a
padding contract references its companion by value name. See
[§10.4](#104-generic-component-batching).

Application policy inputs — a grammar, a JSON Schema, a regex — are **request
data**, not metadata. What metadata carries is everything needed to interpret
that request data correctly:

```yaml
schema_version: v1.2
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32000
    byte_level: true
    special_tokens:
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: [2]
    artifacts:
      - location: tokenizer.json
  constraint_languages:
    - dialect: llguidance.lark
      version: "1"
      component: grammar
```

Numeric model/control ids have one package authority:
`package.tokenizer.special_tokens`. Co-locating the ids with the vocabulary
contract makes their namespace explicit. Their text spellings, added-token
mappings, and chat templates remain in the declared tokenizer assets and are
not repeated in metadata.

The numeric fields are `pad_token_id`, `bos_token_id`, the ordered
`eos_token_id` list, `sep_token_id`, `decoder_start_token_id`,
`image_token_id`, `video_token_id`, `audio_token_id`, and
`vision_start_token_id`. A producer
derives them from the package's authoritative source configuration. When more
than one source is present, package-authored `genai_config.json` wins over the
pinned `generation_config.json`, which wins over pinned model `config.json`;
`tokenizer_config.json` is only a fallback, and a string there is resolved
through the pinned tokenizer asset. The producer records provenance rather than
leaving the runtime to repeat this precedence.

`request.eos_ids`/`request.eos_lengths` are optional request overrides. Their
runtime roles receive the effective request set: the explicit request set when
present, otherwise `package.tokenizer.special_tokens.eos_token_id`. They carry
no authored literal default and do not become a second package authority.

EOS values and EOS execution are separate contracts. A portable
`onnx-genai.termination-predicate` graph computes done/active state from the
effective values; an `onnx-genai.token-policy` binding declares equivalent
runtime-native semantics. Neither owns the ids. A v1.2 autoregressive workflow
with a `generation_eos` loop must declare non-empty EOS facts and invoke one of
those contracts. A speculative package must do the same explicitly. Encoder,
embedding, diffusion, and other non-token-generation workflows need neither.

The tokenizer's declared vocabulary facts and artifact location are part of the
semantic contract, while byte-level integrity belongs to distribution. The
constraint-language dialect and version matter because a caller-supplied grammar
is only meaningful against a named dialect, and the component that interprets it
is named so the dependency is derivable ([§11](#11-cache-correctness-dependencies)).

### 15.1 Versioned tool-call protocol

The caller owns offered tools, descriptions, JSON Schemas, choices, prior calls,
and results. A tool-capable package declares the exact protocol identity and
version needed to render those values and parse model output. The protocol
covers template placement, envelopes, incremental boundaries, escaping, call
identities, multiple calls, and complete/incomplete/malformed outcomes.

A runtime **MUST** select only the declared protocol. Unknown, unsupported, or
ambiguous identities and versions fail closed; parser trial order, model-family
matching, and a boolean such as `supports_tools` are forbidden. Whether the
implementation is native or portable is not part of the package contract.

The declaration is an exact pair under `package`, not an implementation
registry entry:

```yaml
schema_version: v1.3
package:
  tool_protocol:
    identity: tagged-json
    version: v1
```

`identity` and `version` are opaque protocol-owned strings. They select the
request/template renderer and output-envelope parser together; a package that
does not support tools omits `tool_protocol` entirely.

Forced `tool_choice` output requirements are adapter-owned too:
`tagged-json@v1` supplies its tagged JSON grammar, while `atem-xml@v1`
explicitly supplies no engine JSON grammar because its envelope is XML. A
runtime must not apply one protocol's grammar to another protocol.

This server currently supplies these exact v1 adapters:

| Declaration | Request/template rendering | Output envelopes |
| --- | --- | --- |
| `tagged-json@v1` | Supplies the offered OpenAI `tools` JSON value to a chat template; without a template, prefixes it with `<\|tools\|>\n`. It emits the caller's `tool_choice` after `<\|tool_choice\|>\n`. | One or more adjacent `<tool_call>{...}</tool_call>` envelopes, separated only by whitespace. Every object has `name`, optionally `id`, and `arguments` or `parameters`. |
| `atem-xml@v1` | Supplies the same `tools` JSON template value; without a template, prefixes it with `<atem:tools>\n`. It uses the same explicit `tool_choice` placement. | One or more adjacent `<atem:invoke name="...">...</atem:invoke>` envelopes. Each argument is `<atem:parameter name="...">JSON-or-text</atem:parameter>`; `&quot;`, `&apos;`, `&lt;`, `&gt;`, and `&amp;` are unescaped exactly once. |

For both adapters, explicit IDs are preserved and absent IDs become stable
`call_<index>` values in envelope order. Duplicate IDs, empty/oversized names
or IDs, excess calls, non-whitespace envelope interstitials, invalid JSON, and
text outside XML parameters are malformed. Before an opening envelope parsing
is `NoCall`; after an unclosed opening envelope it is `Incomplete`; a complete,
valid sequence is `Complete`. Feeding the same UTF-8 output in arbitrary chunks
MUST produce the same typed result as feeding it at once. This server caps each
rendered or parsed protocol payload at 64 KiB and each collection at 32 calls.
At the buffered-generation and SSE-streaming boundaries, `Incomplete` and
`Malformed` are typed protocol failures that name the declared identity/version
and boundary; they are never returned as assistant content. `NoCall` remains
ordinary assistant content.

All caller-provided tool data, template values, and model-produced envelopes are
untrusted structured input and **MUST** be bounded and validated. Metadata grants
no authority to execute or select tools.

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

An importer **MUST** inspect the complete source document and classify each
source field as consumed, deliberately dropped with a reason, or unrecognized.
Dropped or unrecognized information is an error by default. An explicitly
lossy conversion **MUST** report every discarded field; it must never silently
approximate missing semantics.

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
| KV quant mode/tolerance | delete; runtime-owned |
| implicit effect domains | declare `pipeline.workflow.effects` |
| native stateful components | add `row_scope` and `cache_affects_state` |
| contracts | add `equivalence` (defaults to `semantic`) |
| state cells bound to a group | add `management: runtime` + `release_boundary` |
| top-level `model.io` | declare port roles in `components.<c>.ports.roles` and cache ports in the owning `state_service` group |
| `model.io.static_cache` | `state_service.groups.<g>.update` (`indexed_scatter`, `write_indices_ports`, `kv_length_ports`) plus `role`/`layer` on the group's port pairs |
| `pipeline.models.<c>.io` | deleted with the composite IR; use `components.<c>.ports` |
| inferred tool format / `supports_tools` | declare exact `package.tool_protocol.identity` and `.version`, or omit the section when tools are unsupported |

Every removed field is rejected by name, so migration failures are precise rather
than mysterious.



## 19. Invariants

A conforming document satisfies these rules:

1. Workflow dataflow is the sole executable graph ABI; retired parallel ABI
   blocks are rejected.
2. Row axes and ownership are derivable without serialized request identity.
3. Shared-batch selection, compaction, release, state, effects, and publications
   preserve isolated-execution semantics; unsupported sharing falls back.
4. Cache dependencies include every value that can change cached state.
5. Speculative rollback uses declared rollback semantics, never retry class.
6. Automatic substitution requires the declared equivalence class.
7. Graph-visible state ABI and runtime-private storage policy remain distinct.
8. Canonical semantic identity is stable under its declared normalization.
9. Generation overrides, tool protocols, and extension versions fail closed.
10. Session state has unambiguous initialization/restore, dataflow-derived
    readers and writers, one final writer, lifecycle, and atomic transaction.
11. Checkpoints and private runtime state transfers are distinct paths.
12. Workflow outputs publish through one declared protocol family; emit
    operations cannot create a second authority.

## 20. Conformance

A runtime or producer claims conformance only for contract versions and features
it implements completely. Unsupported identifiers, versions, state/effect
participants, output protocols, proposal forms, and optimization preconditions
fail before semantic mutation. Implementation slices, fixtures, diagnostics,
and acceptance evidence are tracked in
[#2303](https://github.com/justinchuby/onnx-genai/issues/2303), not in this
specification.

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
          shape: [batch, sequence]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: runtime, version: "1.0", role: input_ids }
        source: { kind: request }
    state:
      cache:
        contract:
          dtype: float32
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
        shape: [batch, generated]
        batch_layout: { kind: request_aligned, axis: 0 }
    outputs:
      guided:
        dtype: int64
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

If the workflow `logits` contract pads the dimension selected by `time_axis`,
its padding entry names the one authoritative frame-count output:

```yaml
pipeline:
  workflow:
    outputs:
      logits:
        contract:
          shape: [batch, frames, vocab]
          padding:
            - { dimension: frames, valid_lengths: frame_lengths }
      frame_lengths:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: { kind: shared }
profiles:
  transcription:
    outputs:
      logits: logits
      frame_lengths: frame_lengths
    decoding:
      kind: ctc
      time_axis: 1
      class_axis: 2
      lengths: frame_lengths
```

`decoding.lengths` is a profile role, not a second tensor name: resolving it
through `profiles.transcription.outputs` must yield the exact
`padding.valid_lengths` workflow output. This prevents padded frames from being
decoded and prevents two contradictory length sources. Unpadded CTC remains
valid without `decoding.lengths`. Every CTC profile also exposes the decoded
tensor through the canonical `outputs.logits` role; renaming that role to an
alias such as `emissions` is rejected rather than bypassing logits-contract
validation.

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
