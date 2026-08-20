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
        kind: indexed_scatter            # append (default) | indexed_scatter
        write_indices: write_indices     # a rank-1 semantic state cell, one slot per row
        capacity: package.capacity       # a graph-visible integer scalar
        write_indices_ports:             # which input port carries destinations
          model: write_indices
      checkpoint: { adapter: onnx-genai.kv-checkpoint, version: "1" }
      ports:
        model:
          key_cache:   { input: key_cache,   output: updated_key_cache }
          value_cache: { input: value_cache, output: updated_value_cache }
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
- **`write_indices_ports` names the port.** The runtime checks destinations
  *before* the invoke, which requires knowing which bound input carries them.
  This is the one fact that cannot be recovered from the graph: destinations are
  shape- and dtype-indistinguishable from any other integer control input, and
  the runtime cannot invert a cell name back to a body SSA value at invoke time.
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
        sequence_axis: null
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
| Fixed-capacity (static) cache | yes | yes | synthetic | yes | Indexed scatter, per-row cursors, unequal lengths, inactive-row freeze, rewind |
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
  there with the provider's own type error. §12.2c records why that must not be
  turned into a validation error.

The canonical executable packages cover decoder, VLM, image diffusion, guided
diffusion, masked diffusion, speculative decoding, codec, TTS, video, and
adapters. CTC and audio preprocessing have profile/validator coverage, while
real-weight Whisper, CTC, image-edit, diffusion, video, VLM, and heterogeneous
KV runs establish the higher proof levels. These examples therefore cover the
known workflow *shapes* without claiming that an untested model automatically
has a correct exporter.

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
