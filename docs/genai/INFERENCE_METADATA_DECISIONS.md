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

Signing, provenance, attestation, and mirroring. Format-specific checksums
(for example `adapters[*].sha256`) remain where an existing loader ABI needs
them; there is no global metadata digest requirement.

---

## 4. Schema model

### 4.1 Document shape

```yaml
schema_version: v1          # required
required_capabilities: []   # capabilities the reader must implement
model: {...}                # bare single-model facts and graph I/O
quantization: {...}         # model-weight quantization intent
pipeline: {workflow: {...}} # the executable workflow (composite packages)
adapters: {...}             # LoRA target manifest and artifact bindings
preprocessing: {...}        # typed preprocessing contracts
package: {...}              # tokenizer and constraint-language facts
generation: {...}           # authoritative defaults + override surface
profiles: {...}             # task profiles
speculative: {...}          # proposer/target compatibility facts
hardware_requirements: {...}
```

A bare single-model decoder **MAY** use `model.io` alone. A composite package
**MUST** use exactly one `pipeline.workflow` and **MUST NOT** declare top-level
`model.io`.

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
batch_layout: { kind: token_packed, offsets: cu_seqlens,
                owner: token_owner, axis: 0 }                   # packed tokens + owner map
batch_layout: { kind: runtime_sequence_state }                  # opaque runtime state handle
```

That is the whole vocabulary. It states **where the request axis is**, never
**which request occupies which position**.

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
- **artifact bindings**: sources (PEFT + safetensors, ORT `.onnx_adapter`),
  their exact `sha256`, and the ABI they satisfy;
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
      offsets: image_offsets     # cu_seqlens-style prefix offsets
      owner: image_owner         # per-item owning row
      axis: 0
```

`offsets` and `owner` are the only facts needed to map packed items back to rows.

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

- **`sequence_axis` and `layout`** — where the sequence dimension is and how the
  tensor is laid out;
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

### 13.5 Automatic enablement

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
        sha256: "…"          # exact bytes, so the vocabulary is unambiguous
    special_tokens:
      bos: { id: 1, content: "<s>" }
      eos: { id: 2, content: "</s>" }
  constraint_languages:
    - dialect: llguidance.lark
      version: "1"
      component: grammar
```

The tokenizer artifact's exact bytes matter: a grammar compiled against a
different vocabulary produces a different token mask. The constraint-language
dialect and version matter for the same reason — a caller-supplied grammar is
only meaningful against a named dialect, and the component that interprets it is
named so the dependency is derivable ([§11](#11-cache-correctness-dependencies)).

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

Every removed field is rejected by name, so migration failures are precise rather
than mysterious.

---

## 19. Invariants

A conforming document satisfies all of these. Each is validator-enforced.

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
10. **Constraint dialect and exact tokenizer bytes are represented.**
11. **One-file profile evolution works.** Required profiles fail a reader that
    does not understand them; ignorable profiles are skipped.
12. **Session scope is normative.**
13. **Checkpoint and private P/D transfer are explicitly different paths.**

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
| Adapter artifact compatibility | `tests/adapter_artifact_compat.rs` |
| Canonical identity | `crates/onnx-genai-metadata/src/identity.rs` unit tests |
| Cache dependency derivation | `crates/onnx-genai-metadata/src/cache.rs` + `redesign_invariants.rs` |
| Row ABI (`compact`/`release`) | `crates/onnx-genai-engine/src/pipeline/row_state.rs` |
| Generation override fail-loud | `crates/onnx-genai-engine/src/config.rs::generation_contract_tests` |
| Legacy import fail-closed | `crates/onnx-genai-genai-config/src/import.rs` unit tests |
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
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
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
