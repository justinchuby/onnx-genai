# Workflow and component producer contract

Inference metadata is a concise structured workflow. Tensor math belongs in ONNX artifacts;
the runtime interprets only `sequence`, `invoke`, `loop`, `branch`, and `emit`.

## Policy components from first principles

A generative system is not one neural network call. It is a repeated
computation that decides what to run next, updates semantic state, and
eventually publishes a result. A **policy component** is an executable,
versioned tensor program for one of those decisions or state transforms.
Usually it is a small ONNX graph invoked like any other component.

The word "policy" here means **portable workflow semantics**, not an
administrator's deployment policy:

| Meaning of "policy" | Examples | Owner |
|---|---|---|
| **Executable workflow policy component** | sampling equations, EOS/length termination, diffusion scheduler step, CFG combine, accepted-prefix verification, cache-length/position update, overlap stitching | package/workflow; executed by the runtime |
| **Deployment, scheduling, or QoS policy** | request priority, deadlines, batching window, device placement, memory budget, cache eviction, tenant quotas, speculative enablement/width | runtime/service deployment; never authored as model semantics |

The distinction is correctness-critical. A sampler equation changes generated
tokens and therefore travels with the package. A scheduler priority changes
when a request runs but not what the package means, so it stays runtime-owned.

### Three kinds of executable component

```text
caller tensors
     |
     v
+------------------- workflow ---------------------------------------+
|  native/runtime binding  ->  policy graph  ->  neural model graph  |
|  decode bytes, grammar       resize, sample,   decoder, denoiser,   |
|  engine, telemetry ABI       terminate, CFG    encoder, codec       |
|              \________________ typed SSA __________________/         |
+---------------------------------------------------------------------+
     |
     v
runtime services: placement, batching, memory, KV allocation, QoS
```

| Component kind | What it represents | Typical implementation | What it MUST NOT hide |
|---|---|---|---|
| **Neural model component** | Learned mapping whose parameters are model weights. | ONNX decoder, encoder, denoiser, VAE, codec. | Sampling, scheduler defaults, host loops, or model-family runtime branches. |
| **Native/runtime binding** | A capability whose semantics require a host library, device service, external state, or privileged effect. | Encoded-media decode, grammar engine, telemetry, parameter overlay. | Undeclared I/O, effects, filesystem/network access, or unversioned action semantics. |
| **Policy component** | Portable tensor equations and semantic state transitions surrounding learned models. | Small ONNX graph, pure unless an explicit adapter ABI is required. | Deployment placement, queue policy, physical cache allocation, or request identity. |

An ONNX file is not automatically a neural model: a six-node graph that
computes `uncond + scale * (cond - uncond)` is a policy component. Conversely,
a workflow `loop` is not itself a policy component; it is structural
orchestration that invokes components.

### Why policy is executable

Sampling, termination, scheduler equations, cache-length and position
updates, classifier-free guidance (CFG), state transforms, semantic scatter,
accepted-prefix compaction, overlap stitching, and resampling all affect
observable output. Leaving them in prose or model-family host code creates
several incompatible truths:

1. different runtimes fill in different defaults;
2. a model-family branch becomes the only specification of an equation;
3. producer and runtime upgrades silently change output;
4. the computation cannot be validated, substituted, optimized, or fused;
5. state and row behavior become implicit and fail under batching or rollback.

Encoding the math as a component gives it typed ports, an artifact, a
versioned semantic contract, explicit state, and an exact place in control
flow. The runtime remains generic: it executes `invoke`, not
`if model_family == ...`.

Examples of **semantic** transforms that belong in policy graphs:

- greedy or seeded stochastic sampling, logits filtering, and grammar-mask
  application;
- EOS/maximum-length predicates and per-row active/done updates;
- attention-mask, position, logical-length, token-history, and semantic
  scatter/gather updates;
- diffusion timestep lookup, CFG combination, Euler/DPM-style solver
  equations, noise/state scaling, and inpainting blends;
- speculative acceptance/correction selection, accepted-prefix truncation,
  rollback transforms, and proposal-budget adaptation;
- codec/codebook loop counters, delay-pattern transforms, overlap-add/crossfade
  stitching, and sample-rate conversion;
- image/audio normalization, patch/grid construction, token expansion, and
  multi-axis position construction when expressible as portable tensor math.

Examples that **do not** belong there:

- physical paged-KV append, block-table allocation, row-slot assignment, or
  runtime compaction machinery;
- queue ordering, batch formation, placement, graph capture, cache eviction,
  deadlines, and tenant isolation;
- immutable architecture facts such as head count, vocabulary size, cache
  aliasing legality, or a component's ONNX opset imports.

## Policy component contract

Every policy component is governed by the same rules as every other workflow
component, with additional attention to state and batching.

### Inputs, outputs, and semantic bindings

- Ports have dtype, rank, shape, optionality, and `batch_layout`.
- The workflow binds SSA values to concrete ports.
- A versioned `contract.id`/`version` gives stable semantic role names through
  `contract.bindings`; implementations may use different concrete port names.
- Parameters contain only non-tensor ABI facts. Request-varying controls such
  as temperature, guidance scale, seed, or maximum length are typed tensors,
  not baked artifact constants.
- State transitions return next state explicitly. Ordinary tensor state is not
  an external effect.

```yaml
components:
  sampler:
    implementation: { kind: onnx, artifact: policies/token_sampler.onnx }
    contract:
      id: onnx-genai.token-sampler
      version: "2"
      bindings:
        logits: logits
        active: active
        done: done
        temperature: temperature
        top_k: top_k
        top_p: top_p
        min_p: min_p
        seed: seed
        counter: counter
        token: token
        next_counter: next_counter
      parameters: { batching: per_row, inactive_rows: preserve }
```

The contract is semantic; it never asks the runtime to choose that
implementation. Application-overridable components are substituted only after
contract, port, effect, and equivalence validation.

### Determinism, effects, and state ownership

Policy graphs SHOULD be pure. Randomness is explicit counter-based state:
`seed` and `counter` are inputs, and `next_counter` is an output. The same
inputs then identify the same result subject to the selected implementation's
declared numerical equivalence.

External mutation uses a declared effect domain. Its retry class (`pure`,
`idempotent`, `transactional`, or `non_retryable`) and speculation safety
(`none`, `clonable`, or bounded `rewindable`) determine whether replay,
branching, or speculation is legal. A runtime MUST NOT infer safety because an
operation happens to have a harmless name.

State has one owner:

- the workflow owns semantic tensor values and declares invocation/session
  scope, recurrence, and release boundary;
- a state-service group lets the runtime own storage and lifecycle while the
  package owns state kind, graph aliases, update discipline, and rollback/fork
  bounds;
- the runtime privately owns request IDs, slots, epochs, page tables, storage
  layout, and compaction algorithms.

### Batching and row semantics

Policy components are batch programs, not scalar callbacks repeated by host
code. `request_aligned` ports carry one entry per request along a declared
axis; `token_packed` ports carry offsets and ownership; `shared` values are
invariant; `runtime_sequence_state` is runtime-owned.

A batching-safe per-row policy MUST:

- produce each row from only that row's semantic inputs and explicit shared
  values;
- preserve inactive/done rows and their RNG/state when its contract says so;
- avoid serialized request, slot, or epoch identifiers;
- tolerate one runtime permutation being applied to every request-aligned
  value during compaction;
- publish ragged rows through `emit.valid_length`/`emit.when`, not by inventing
  row IDs.

Tensor scatter/gather that changes **semantic** state belongs in a policy
component. Moving live request rows between physical slots is runtime
compaction and does not.

### Portability, versioning, admission, and security

Portable policy components use the same ONNX semantics and typed workflow
contract as model components. Version the semantic contract when role meaning,
required bindings, state transition, or equivalence changes. Merely changing
an optimized graph without changing meaning does not require a new contract.

Admission is fail-closed:

1. schema and semantic validation reject unknown core fields, missing values,
   invalid control flow, incompatible ports, undeclared effects, and malformed
   state;
2. the workflow manifest declares every required capability, including those
   derived from its structure;
3. known contracts enforce exact versions/actions/binding obligations;
4. component substitution must satisfy contract, port ABI, effects, state,
   and equivalence;
5. a missing capability or unsupported ABI rejects the package before work is
   issued.

An ONNX policy graph receives only bound tensors and has no implicit
filesystem, network, clock, process, or device-control authority. A native
binding is a larger trust boundary: its ABI MUST enumerate actions, bindings,
effects, and state, and the runtime SHOULD sandbox or restrict it according to
deployment trust policy. Metadata carries no artifact hashes; signatures,
provenance, and byte integrity belong to the distribution layer.

### Performance and fusion

Component boundaries are semantic, not kernel boundaries. After validation and
override selection, adjacent pure same-device policy and neural components may
be linked into an execution island. That permits logits processing, sampling,
state update, and termination to remain device-resident and optimizer-visible.

Host control, external effects, device changes, dynamic allocation, and
stateful native bindings form real boundaries. CUDA Graph capture additionally
requires stable addresses/shapes and explicit RNG state. Fusion MUST preserve
the component contracts and MUST NOT make a request-selected replacement,
effect, inactive-row rule, or rollback point disappear.

## End-to-end examples

### Autoregressive decoder

```text
prompt -> [decoder neural graph] -> logits
          -> [last-position/logits policy]
          -> [sampler policy] -> token
          -> [termination policy] -> done/active/continue
          -> [token, mask, position, logical-length policies]
          -> next loop iteration

runtime beside the loop: KV storage, batching, placement, page allocation
```

The decoder carries present state to the next past state. A policy graph is
needed only where tensor math changes semantic state, for example updating a
fixed-cache cursor or truncating an accepted prefix. Physical KV append is not
a policy graph.

### Diffusion scheduler and CFG

The text encoder, denoiser, and VAE are neural components. The schedule/timestep
lookup, counter-based initial noise, model-input scaling, CFG equation, solver
step, history update, and final scaling are policy components. The workflow
loop states exactly how often they run. Euler and DPM++ can implement the same
solver contract with different policy artifacts; runtime device placement and
batch scheduling remain unchanged.

### Speculative verification and rollback

The proposer and target are neural components. A verifier policy computes
accepted tokens, `accepted_len`, correction choice, and next RNG state. Emits
publish only the valid accepted prefix. The runtime rewinds every state group
named by the speculative contract before committing the accepted result.

Qwen3.5 hybrid speculative decoding is a useful example, not a schema case:
full-attention KV, linear-attention accumulator, and causal-convolution history
are three ordinary state-service groups with mutually declared rollback
cascades. All must snapshot/fork/rewind to the same accepted position. The
schema contains no `qwen` branch.

### Audio/music nested loops

A SenseNova/Music3-style package can use an outer talker/frame loop, an inner
predictor/codebook loop, neural talker/predictor/codec components, and policy
components for sampling, termination, delay-pattern updates, cache lengths,
frame assembly, overlap stitching, and resampling. Session state permits
streaming interaction; typed emits publish audio chunks.

This is still only nested `loop`/`invoke`/`emit` with typed state. "SenseNova"
or "Music3" is never a schema discriminator, and the same structure applies to
other speech, music, or multi-codebook generators.

### Multimodal preprocessing

Encoded media enters through a typed optional request input. A versioned native
binding may decode a compressed format; portable policy graphs then resize,
normalize, pad/crop, construct grids, expand placeholder tokens, and calculate
position coordinates. Neural vision/audio encoders consume the resulting
tensors, and an embedding/mixer component feeds the decoder.

`present_as` branches around absent media without fake tensors. Multiple packed
outputs use explicit offsets/ownership. Device placement and preprocessing
cache eviction are runtime policies, while transform equations and packing
semantics travel with the package.

## Decision guide and anti-patterns

| Question | Put it here |
|---|---|
| Is it portable tensor math that changes observable semantics or semantic state? | **Policy graph/component** |
| Does it only order, repeat, branch, or publish already-defined computations? | **Workflow step/control flow** |
| Does it require a host library, privileged runtime service, external mutation, or non-ONNX facility? | **Versioned native/runtime binding** with explicit effects |
| Is it immutable truth about artifacts or architecture? | **Authored package fact** |
| Does it choose resources, timing, isolation, optimization, or service quality? | **Runtime deployment/scheduling/QoS policy** |

Anti-patterns:

- `if family == "..."` in runtime code to select sampling, scheduler, cache
  update, preprocessing, or rollback equations;
- prose such as "use the usual sampler" or an unstated default seed/schedule;
- encoding deployment choices (`cuda`, page size, batch window, priority) in a
  policy component;
- using a native binding for pure tensor math merely to avoid writing an ONNX
  graph;
- hiding external mutation or RNG behind a component declared pure;
- serializing row IDs, slots, epochs, page tables, or transfer steps;
- treating component boundaries as mandatory sessions/host round trips;
- adding a model name, SenseNova/Music3 mode, or Qwen-specific state kind when
  existing structural state/control-flow contracts express the behavior.

## Structural execution semantics

- Root `steps` execute once per invocation, in order.
- `loop.setup` executes once each time that loop is entered.
- `loop.steps` execute once per iteration.
- A session-state initializer runs once when the session cell is created.
- Artifact loading and session restoration are runtime lifecycle operations, not workflow steps.

There are no phases, strategies, `run_once`, or execution-frequency flags. A producer expresses
frequency by placing an invocation at the corresponding structural location.

## Surface form and lowering

The serialized document uses logical value names and concise state carries:

```yaml
steps:
  - kind: loop
    setup:
      - kind: invoke
        component: initialize
        inputs: { value: request_state }
        outputs: { value: state.initial }
    steps:
      - kind: invoke
        component: update
        inputs: { current: state, update: delta }
        outputs: { next: state.next }
    continue_when: continue
    max_iterations: max_iterations
    carried:
      - cell: state
        initial: state.initial
        next: state.next
```

Inside the body, `state` means the current carried value. After the loop, `state` means the final
value. The loader lowers this form to strict lexical SSA names, branch phi values, loop read/write
versions, and linear effect tokens. Those compiler-generated names are not serialized.

Effects are inferred from structure. Pure ONNX components with tensor-threaded RNG/state declare no
effect. `effects` on a component are reserved for real external mutation: emit streams, session
mutation, telemetry, or stateful adapter ABIs. The compiler threads and joins those effects through
sequences, branches, and loops.

Loops are pre-test: `continue_when` is evaluated before iteration zero, so false initially produces
a zero-trip loop and leaves every carry at its initial value. It may be `bool[]` or `bool[B]`.
`termination: generation_eos` marks only an autoregressive generation loop whose predicate
represents EOS completion. When the request sets `stop_on_eos: false`, that loop runs exactly
`max_iterations` without inspecting a device-resident predicate. Other loops retain
`termination: predicate` (the default) and always evaluate `continue_when`.
Inactive rows retain their previous carried values while active rows advance. `max_iterations` is
an integer safety bound. Branch predicates remain invocation-level scalars; row-wise tensor choice
is ordinary ONNX policy math rather than host control flow.

Optional request/application tensors use `present_as` to define an initial scalar bool SSA value.
The runtime sets it from actual caller presence without inventing a fake tensor. Consumers must
branch before reading the optional input and merge a real alternative value:

```yaml
inputs:
  request.image:
    contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
    role: { kind: runtime, version: "1.0", role: media }
    source: { kind: request }
    required: false
    present_as: request.image_present
steps:
  - kind: branch
    predicate: request.image_present
    cases:
      "true": # preprocess and encode request.image
        ...
    default: # produce an empty [0, hidden_size] feature tensor
      ...
```

`present_as` requires the `input_presence` capability and cannot be combined with `required: true`
or a literal default. Request-sourced presence is available for media, constraint, and session ID
roles, whose absence is observable; application-sourced tensors may use it generally. For text-only
VLM requests, the absent branch should produce the embedding graph's supported zero-image feature
tensor; clients do not pass sentinel or fake image bytes.

Row emission is derived, not declared. An emit is row-wise when its output is ragged — when
`emit.valid_length` slices that row's final axis or `emit.when` guards it. `emit.when` optionally
suppresses a row event: a true guard with length zero emits an empty row; a false guard emits
nothing. Guards and lengths may be singleton-broadcast or per-row.

Row-wise-ness is a property of the OUTPUT, not of one emit. If any emit of an output is ragged,
every emit of that output publishes rows, so an append loop that mixes a ragged accept step with a
single forced token yields one consistent stream. A row-wise output must have a
`request_aligned` batch layout; the validator rejects one that does not.

Aggregate outputs retain one tensor under the declared output. Rows are published positionally:
compatibility keys use `output.row.<position>`, but consumers use the structured output API rather
than parsing those names. Metadata never serializes a row identity — no `row_ids`, no `slot_ids`,
no `request_epochs`. Under compaction the runtime applies one permutation to every request-aligned
value and every row-scoped component, and associates output rows with requests through its own API.
See [INFERENCE_METADATA_DECISIONS.md](genai/INFERENCE_METADATA_DECISIONS.md#8-batching-varlen-and-paged-attention).

ONNX port contracts may be omitted when the artifact is authoritative. Declare only semantic
bindings, bounds/overrides, cross-component constraints, or adapter ports that ONNX cannot describe.
Device transfers are planner-derived from resource placement rather than authored workflow nodes.

## Execution islands and graph capture

Workflow component boundaries are semantic composition boundaries, not mandatory ORT session,
kernel-launch, or CUDA Graph boundaries. After validation and SSA lowering, the planner links
adjacent pure, effect-free ONNX invokes on the same device into one execution island. Intermediate
SSA tensors remain internal graph values, so ORT can optimize across decoder, logits-processing,
sampling, and termination components without host round trips.

CUDA island outputs remain device-resident in the workflow value store. The interpreter
materializes them on the host only for host control, stateful host adapters, ragged emission, or
public package outputs; separate same-device islands exchange device tensors without a CPU hop.

Control flow, device changes, explicit effects, and stateful host adapters delimit islands. The
default implementation of an application-overridable pure ONNX component remains fusible. A
request selecting a replacement executes the preserved unfused invoke sequence, so the fused
default cannot silently ignore a per-request replacement.

CUDA capture is decided per island and concrete shape signature. Eligibility requires:

- a CUDA placement with stable device-resident I/O bindings;
- static or runtime-specialized bounded shapes, with no data-dependent allocation/control ops;
- no implicit ONNX RNG (counter RNG seed/offset must be tensor inputs/state);
- no host control or external mutation inside the island; and
- kernels accepted by the selected execution provider during capture.

The first run resolves artifact-inferred dynamic extents and establishes stable bindings. A
subsequent equal-shape run captures, then later runs replay. Unsupported graph features, allocator
failure, shape changes, or provider capture errors fall back to ordinary island execution rather
than changing workflow semantics. `PipelineEngine::execution_island_diagnostics()` reports each
island's components, device, session/capture/replay counts, transfers, synchronizations, stable
memory, observed CUDA memory high-watermark delta, and fallback reason. The benchmark methodology
and acceptance bar are defined in
[`WORKFLOW_PERFORMANCE_CONFORMANCE.md`](WORKFLOW_PERFORMANCE_CONFORMANCE.md).

## Versioned component contracts

`contract` describes semantics; it never selects execution behavior. Execution is always ordinary
`invoke` against the component implementation.

```yaml
contract:
  id: onnx-genai.token-sampler
  version: "1"
  bindings:
    logits: logits
    token: token_ids
  parameters:
    mode: greedy
```

`bindings` maps stable semantic roles to concrete component port names. `parameters` contains
non-tensor contract data. New policy implementations require metadata plus ONNX, not host code.

### Token sampler: `onnx-genai.token-sampler@1`

| Role | Contract |
|---|---|
| `logits` | floating `[B,V]` or `[B,S,V]`; sample the final position |
| `token` | `int64[B]` |
| `temperature`, `top_k`, `top_p`, `min_p` | request-provided scalar or `[B]` when used |
| `grammar_mask` | optional request/adapter-provided `bool[B,V]` |
| `rng_seed`, `rng_offset` | required for seeded stochastic mode, `int64[B]` |
| `rng_next_offset` | seeded mode output, `int64[B]` |

`parameters.mode` is `greedy` or `seeded_stochastic`. Sampling controls are typed workflow inputs;
they are never baked into the artifact, so ordinary request option changes do not regenerate
ONNX. Seed and offset are explicit counter-based RNG state (for example Philox or Threefry),
loop-carried as semantic state.

Min-p filtering is ordinary parameterized ONNX policy math. A fixed-shape implementation may keep
tokens satisfying `logit >= max_logit + log(min_p)` before normalized categorical sampling. The
request supplies `min_p`; it is never baked into the artifact, and the `[B,V]` shape remains
capture eligible.

A component may set `application_overridable: true`. The application can then select another
package-declared ONNX component for that invocation. The replacement must implement the same
contract ID/version, semantic binding set, tensor port ABI, and effects. Concrete ONNX port names
may differ because the runtime remaps them through `contract.bindings`. This is the extension
point for fundamentally custom sampling policy; changing temperature, top-k, top-p, seed, or a
grammar mask uses ordinary workflow inputs instead.

The planner links the default pure ONNX implementation normally. Selecting a replacement switches
that island invocation to its validated component sequence; host/stateful replacements therefore
form an explicit optimization boundary without slowing the default path.

### Batched policy island contracts: version 2

Policy components may join a fused execution island only through version 2 contracts with string
parameters `batching: per_row` and `inactive_rows: preserve`. Version 1 artifacts remain valid
unfused components, but their scalar-compatible ABI cannot establish continuous-batching safety.

The version 2 token sampler binds per-row `logits`, `active`, `done`, `temperature`, `top_k`,
`top_p`, `min_p`, `seed`, and `counter`, and returns `token` plus `next_counter`. These are the
exact semantic binding names; greedy workflows use the same ABI with `top_k=1`, while inactive or
done rows return the sentinel token and preserve their counter.

The termination predicate binds `tokens: int64[B]`, `eos_ids: int64[B,Emax]`,
`eos_lengths: int64[B]`, `iteration: int64[1]`, `max_iterations: int64[B]`, and
`active: bool[B]`. It returns `done: bool[B]`, `next_active: bool[B]`, and the reduce-any loop
control `continue: bool[1]`. Inactive rows remain done and cannot reactivate.

State update binds `current: int64[B,1]`, `update: int64[B,1]`, `active: bool[B]`, and
`done: bool[B]`, returning `next: int64[B,1]`. Suppressed rows preserve current state.

All per-row ports have a symbolic leading `batch` axis; `iteration` and `continue` are singleton.
Inactive rows preserve RNG counters and semantic state and emit no token.
This ABI permits stable max-batch buffers and CUDA graph replay while the active row set, row
parameters, logical lengths, and EOS sets change between iterations.

Version 2 admission is exact: the public symbol is literally `batch`, not a component-qualified
variant, and the 11/9/5 sampler, termination, and state-update binding sets accept no additional
grammar, RNG-alias, previous-done, length, or iteration-broadcast roles.

### Termination predicate: `onnx-genai.termination-predicate@1`

Inputs: `tokens: int64[B]`, `eos_ids: int64[E]`, zero-based `iteration: int64[]|[B]`, and
`max_iterations: int64[]|[B]`. Output: `done: bool[B]`. Loop continuation polarity is explicit in
the workflow; use an ONNX boolean component when inversion is required.

### Solver step: `onnx-genai.solver-step@1`

Inputs: `state: T[B,...]`, `estimate: T[B,...]`, induction `step: int64[]|[B]`, and
`schedule: T[N,...]`. Output: `next_state`, contract-compatible with `state`. Euler, multistep,
and other algorithms are different ONNX artifacts under the same invoke semantics.

### Masked update: `onnx-genai.masked-update@1`

Inputs: `state: int64[B,S]`, `proposal: int64[B,S]`, `mask: bool[B,S]`, and
`step: int64[]|[B]`. Outputs: `next_state: int64[B,S]` and `next_mask: bool[B,S]`. Optional
counter-based RNG roles use the sampler naming above.

### Speculative verifier: `onnx-genai.speculative-verifier@1`

Inputs: `target_scores: T[B,K,V]`, `proposed_tokens: int64[B,K]`, optional
`proposal_scores: T[B,K,V]`, and optional explicit RNG state. Outputs:
`accepted_tokens: int64[B,K+1]`, `accepted_len: int64[B]`, `done: bool[B]`, and optional
`rng_next_offset`. A branch selects accepted versus correction state; branch outputs are the phi
values used by subsequent steps. `emit.valid_length: accepted_len` streams only the valid prefix.

### Generic state update: `onnx-genai.state-update@1`

Inputs: `current` and `update`; output: `next`. All three use the component's declared or inferred
tensor contracts. This component is optional and exists only when tensor math is required, such
as dense gather, truncate, rollback, or format conversion. Normal decoder present-KV is directly
carried to next-iteration past-KV; it does not require a `kv_update.onnx`.

Physical shared-buffer or paged-KV allocation, append, slot assignment, compaction, and in-place
mutation are generic runtime KV services driven by declared serving values and resource
contracts. They are not workflow policy math and are not modeled as host state-update opcodes.

### Adaptive proposal budget: `onnx-genai.adaptive-proposal-budget@1`

Bindings cover `current_k`, `accepted`, `evaluated`, `committed_tokens`,
`filled_proposal_budget`, `draft_ms`, `target_ms`, `estimates`, `next_k`, and `next_estimates`.
Estimate state is advisory and may be invocation- or session-scoped. It may be reset or dropped and
is excluded from semantic checkpoints; changing it must not change output correctness or
distribution.

## Versioned adapters

Adapters are selected by `implementation.abi` plus `implementation.version`, both pinned in the
manifest. Their optional `contract` supplies semantic bindings and parameters.

- `onnx-genai.grammar-guidance@1`: parameter `action` is `clone`, `lookahead`, or `commit`;
  bindings are `state`, `tokens`, `valid_length`, `transition_table`, `next_state`,
  `consumed_length`, `logits_mask`, `forced_tokens`, and `forced_length`.
- `onnx-genai.telemetry@1`: parameter `action` is `start` or `elapsed`; bindings are `timestamp`
  and, for elapsed, `duration_ms`.

Stateful adapters declare semantic state cells and an external effect domain. Grammar clone,
lookahead, and commit are reusable by any workflow; ONNX components apply masks and sample.

## Parameter adapters (LoRA)

Top-level `adapters` is the migrated, versioned `onnx-genai.adapters@1` ABI. It replaces the
earlier `InferenceMetadata.adapters` capability list and the short-lived `workflow.adapters`
prototype; there is one catalog for bare and composite packages. Adapter discovery, verification,
loading, caching, activation, deactivation, and eviction are runtime lifecycle operations, never
workflow steps or `run_once` nodes. Composite packages reference request SSA inputs declared by
`pipeline.workflow`.

```yaml
adapters:
  target_manifest:
    targets:
      - id: decoder.layers.0.q_proj
        component: decoder
        initializer: layers.0.attention.q_proj.weight
        layer_index: 0
        node_name: /model/layers.0/attention/q_proj/MatMul
        output_name: layers.0.attention.q_proj.output
        activation_dtype: float16
        input_features: 4096
        output_features: 4096
        rank: 8
        alpha: 16.0
        output_slice:                # optional resolved fused-output range
          role: query
          offset: 0
          width: 4096
          rank: 8
          alpha: 16.0
        graph_inputs:                 # Phase-1 portable graph-native seam
          a: lora.layers.0.q_proj.a
          b: lora.layers.0.q_proj.b
          scale: lora.layers.0.q_proj.scale # optional dynamic scale
  discovery_fallback: disabled        # or tooling_only; execution never guesses
  selection:
    segments: request.lora_segments    # int64[batch,max_adapters]
    adapter_counts: request.lora_counts # int64[batch]
    scales: request.lora_scales        # float32[batch,max_adapters]
    active: request.active              # optional bool[batch]
    max_adapters: 4
  application_capability: onnx-genai.adapters@1
  portable_fallback: true
  cache: { max_entries: 16, eviction: lru }
  planning:
    bucket_by_adapter_set: true
    stable_buffers: true
    invalidate_capture_on_eviction: true
  artifacts:
    summarizer:
      index: 0
      identity: example.summarizer
      version: "1"
      rank: 8
      alpha: 16.0
      dtype: float16
      provenance:
        producer: mobius
        source: hf://example/summarizer
        revision: <immutable revision>
      weights:
        - format: hf_peft
          loader_capability: onnx-genai.adapters.hf-peft@1
          location: adapters/summarizer/adapter_model.safetensors
          config_location: adapters/summarizer/adapter_config.json
          scale_encoding: alpha_over_rank
        - format: ort_genai
          loader_capability: onnxruntime.lora-adapter@1
          location: adapters/summarizer/adapter.onnx_adapter
          scale_encoding: baked
      bindings:
        - target: decoder.layers.0.q_proj
          weight_key: layers.0.attention.q_proj
```

### Durable contract integrated from PRs #318 and #374

This matrix audits Phase 1 at `813a9b53` and Phase 2 at `326fddcf`.

| Prior surface | Exact prior location/type | v1 disposition | Current location/type |
|---|---|---|---|
| Top-level adapter metadata | #318/#374 `schema/adapters.rs::LoraCapabilities`, `InferenceMetadata.adapters` | **Adapt** | `schema/mod.rs::InferenceMetadata.adapters` now carries `AdapterServiceContract`; no parallel `workflow.adapters` |
| Declared target map | #374 `schema/adapters.rs::LoraTargetManifest`, `LoraTargetDescriptor`, `LoraTargetSlice` | **Adapt** | `schema/ir.rs::LoraTargetManifest`, `LoraTargetDescriptor`, `LoraTargetSlice`; exact node/output identity and resolved labeled slices are retained, while semantic module/layer and Qwen-specific discovery stay in producer tooling |
| PEFT loader contract | #318 `engine/lora/format.rs::load_peft_adapter` | **Reuse ABI, separate implementation** | `AdapterWeightFormat::HfPeft` declares config+safetensors and loader capability; loader implementation belongs in a main-targeted runtime PR |
| ORT FlatBuffer loader | #374 `engine/lora/format.rs::load_onnx_adapter` and `adapter_schema.fbs` | **Reuse ABI, separate implementation** | `AdapterWeightFormat::OrtGenai` maps to upstream `TORT` version 1; Mobius may emit PEFT instead |
| Canonical loaded artifact | #318/#374 `engine/lora/format.rs::LoadedAdapter`, `LoadedAdapterModule` | **Adapt** | all declared source formats normalize to `AdapterArtifact` + manifest-keyed `AdapterTargetBinding`; format-specific names never reach execution planning |
| Phase-1 delta branch | #318 `runtime-session/lora_inject.rs::inject`, `LoraInjection`, `OverrideFeed` | **Reuse** | manifest `graph_inputs.a/b` identifies overridable optional inputs; base-only binds neither input and remains bit-identical |
| Graph discovery | #318 `build_manifest`; #374 declared-primary `build_manifest(..., declared_manifest)` | **Adapt** | `target_manifest` is authoritative; `discovery_fallback: tooling_only` permits importer/load tooling to produce it, never runtime execution guessing |
| Per-row routing | #374 `GroupedLoraInjection.segments_input`, `NativeBatchedDecodeSession::set_lora_routes` | **Adapt** | `selection.segments[B,K]` is the Phase-2 segment ID generalized to ordered K-way composition and carried with semantic slot ID+epoch |
| Adapter/module alignment | #374 `inject_grouped_multi` / `AdapterModuleSetMismatch` | **Reuse invariant** | artifacts bind stable target IDs from one manifest; missing targets contribute zero, duplicate target bindings fail, and every referenced target must resolve |
| Paged pool and budget | #374 `LoraWeightPool`, `BudgetedLoraPool`, shared `ByteBudget`, `LoraPoolRegistration` | **Reuse, separate implementation** | metadata keeps cache/budget-independent lifecycle policy; pool, registry teardown, and device residency remain generic runtime PR surfaces |
| Grouped custom op | #374 `pkg.nxrt::GroupedLoraDelta` CPU/CUDA kernels | **Retire as metadata ABI** | no custom-op requirement or admission field; use graph-native standard ONNX, existing EP kernel dispatch, or a separately justified LoRA EP capability |
| Capture seam | #318 eager capture rejection; #374 persistent pool seam | **Adapt** | stable buffers plus plan variants keyed by capability, shape, ordered adapter sets, and scales; no address rebinding after capture |
| Legacy CLI/manager | #318/#374 `LoraManager`, `AdapterId`, CLI `--adapter` | **Retire from metadata** | application APIs resolve stable artifact indices into immutable request SSA; lifecycle manager implementation remains outside metadata |

### Artifact, manifest, and compatibility identity

Artifact map keys are package-local aliases and `index` is the stable segment ID. Indices are
unique and contiguous from zero. Every source file is beneath `adapters/<alias>/`. Byte-level
integrity belongs to the distribution layer rather than inference metadata. `hf_peft` pairs
`adapter_config.json` with safetensors with `scale_encoding: alpha_over_rank`. `ort_genai` is
upstream `.onnx_adapter` (`TORT`, format version 1) with its static scale already baked into factors
and therefore requires `scale_encoding: baked`; a loader must not apply alpha/rank again. `json` is the
float32 RFC 8785 reference bundle `{"targets":{"<weight_key>":{"a":[...],"b":[...]}}}`. A
manifest-keyed safetensors source uses `<weight_key>.a` and `<weight_key>.b`. Source formats may
coexist for one artifact only when they encode the same canonical factors.

The authoritative manifest contains exact generic graph identities. Producer/import tooling owns
architecture-specific work such as fused-QKV discovery and lowers it to exact `node_name`,
`output_name`, and a labeled `output_slice`; execution never branches on a model family.
Targets and fused slices may retain optional Phase-2 rank/alpha policy, which every artifact
binding must satisfy after applying its per-binding or artifact defaults.
`graph_inputs` preserves Phase-1 optional overridable A/B inputs.
If absent, a capable runtime may apply an immutable parameter overlay or invoke a portable standard
ONNX delta component. Base initializers are never modified.

Compatibility is established by the authoritative target manifest and live graph
admission: component identity, initializer and output bindings, tensor geometry,
ports/state ABI, capabilities, and graph-input seams must resolve. Compatible
artifact bytes may be replaced without changing inference metadata.

### Request routing, composition, and batching

`segments[B,K]` generalizes Phase-2's one `segments[B]` route. For row `r`, slots
`[0,adapter_counts[r])` contain contiguous artifact indices in deterministic composition order; all
remaining slots are exactly segment `-1` and scale `0`. Unknown IDs and duplicates fail loud.
Scales are finite and within `[-16,16]`. A K=1 lowering feeds the Phase-2 grouped route directly.
For K>1, a planner either repeats the graph-native delta branch in axis order or builds an
equivalent grouped plan; both must preserve the same accumulation order:

`base(x) + Σ scale[k] * (alpha[k] / rank[k]) * B[k] * (A[k] * x)`

Per-binding rank/alpha override artifact defaults. Adapters missing a manifest target contribute
zero. Routing carries no serialized identity: selection is request-aligned, so compaction applies
one permutation to segments, counts, scales, active flags, and model state, and the runtime
associates rows with requests through its own request table. Inactive rows are base-only and do not
load or mutate adapter state.

### Lifecycle, capability, and capture

Existing ORT GenAI load/activate/unload APIs are reused when negotiated capability covers the full
batch. They do not alone guarantee heterogeneous rows, composition, dynamic scales, or I/O binding,
so the planner may bucket rows by complete ordered adapter set, materialize an equivalent overlay,
or invoke the portable graph branch. Unknown IDs, fingerprint/checksum mismatch, missing targets,
shape/dtype/rank mismatch, conflicting bindings, nonfinite scales, and unsupported loader or
application capabilities are structured errors.

Pool entries are reference-counted and cannot be evicted while active. Registry teardown releases
all registrations. Captured variants own stable adapter buffers and addresses; no rebinding is
allowed after capture. Shape, capability, ordered adapter set, or scale-bucket changes select a new
plan or recapture. Eviction invalidates captures that reference evicted buffers. Generic pool,
kernel, and allocator improvements are intentionally separate from this metadata PR.

## Compact structural examples

### Decoder

```yaml
steps:
  - kind: invoke                 # once: tokenize/bind prompt
    component: tokenizer
    inputs: { text: prompt }
    outputs: { tokens: prompt_tokens }
  - kind: loop
    termination: generation_eos
    setup: []                    # once on loop entry
    steps:                       # each generated token
      - kind: invoke
        component: decoder
        inputs: { tokens: tokens, cache: cache }
        outputs: { logits: logits, cache: cache.next }
      - kind: invoke
        component: sampler
        inputs:
          logits: logits
          temperature: temperature
          top_k: top_k
          top_p: top_p
          grammar_mask: grammar_mask
        outputs: { token_ids: token }
      - kind: emit
        value: token
        output: tokens
        mode: append
    continue_when: continue
    max_iterations: max_output_tokens
    carried:
      - { cell: cache, next: cache.next }
```

Here `cache.next` is the decoder's present-KV output and becomes `cache` on the next iteration.
The carry describes logical dataflow and bounded growth; the generic KV service may realize it
with shared or paged storage without adding a workflow component.

Logical cache cells bind explicitly to named KV service groups:

```yaml
state:
  cache:
    contract: { dtype: float16, rank: 4, shape: [batch, heads, sequence, head_dim] }
    class: semantic
    scope: invocation
    initializer: empty_cache
    recurrence: { kind: growing, axis: 2, increment: accepted_len, max: max_context }
    service_group: decoder_cache
  cache_lengths:
    contract: { dtype: int64, rank: 1, shape: [batch] }
    class: semantic
    scope: invocation
    initializer: initial_cache_lengths
    recurrence: { kind: invariant }
serving:
  active: active
  done: done
  accepted_len: accepted_len
  state_service:
    groups:
      decoder_cache:
        kind: full_attention
        sequence_axis: 2
        layout: bnsh
        logical_lengths: cache_lengths
        aliasing: permitted
        reuse: { prefix_reusable: true, evictable_prefix: false }
        capabilities: { rollback_positions: 32, snapshot: true, fork: true }
        # Optional. Absent means the group's state is private: it may still move
        # over a private runtime protocol that needs a matching build on both
        # ends, but it is not a package output.
        checkpoint: { adapter: onnx-genai.kv-checkpoint, version: "1" }
        ports:
          decoder:
            cache: { input: past_key_values, output: present_key_values }
```

The service group is the SEMANTIC contract, not an allocation contract. It declares what the state
means (`kind`), the real graph ABI facts (`sequence_axis`, `layout`, graph-visible
`logical_lengths: int64[B]`, past/present `aliasing`, total-length semantics), reuse semantics, and
the rollback/snapshot/fork bounds within which the runtime may operate. `active` and `accepted_len`
drive per-row growth and rollback. Both cache and length cells are loop-carried; inactive rows
retain their prior lengths. The workflow carry remains logical decoder dataflow. A separate
state-update ONNX component is used only for real tensor transforms such as gather or truncation.

Metadata does NOT select a storage mode. `paging`, `allocation`, `storage`, `shared_buffer`, and
slot-allocation policy are runtime deployment decisions and are rejected if serialized. `aliasing`
replaces `shared_buffer`: it states whether the graph is correct when past and present share
memory — a property of the graph, not a request to use one buffer — and defaults to `forbidden`.
Legacy top-level `kv_cache` and `model.runtime_configurable.kv_cache` hints
are rejected. See
[INFERENCE_METADATA_DECISIONS.md](genai/INFERENCE_METADATA_DECISIONS.md#12-state).
Bare decoder documents use ordinary functional past/present dataflow; metadata attention family,
dtype strings, or artifact custom-metadata tags never select shared-buffer execution. Explicit
low-level ORT shared-buffer and static-cache APIs remain available as generic mechanisms, but are
not selected by inference-metadata loading.

### Vision-language

```yaml
steps:
  - kind: invoke                 # once
    component: image_preprocess
    inputs: { encoded: image }
    outputs: { pixel_values: pixels, grid: grid }
  - kind: invoke                 # once
    component: vision_encoder
    inputs: { pixel_values: pixels, grid: grid }
    outputs: { features: image_features }
  - kind: invoke                 # once
    component: embedding
    inputs: { image_features: image_features, tokens: prompt_tokens }
    outputs: { embeddings: embeddings }
  - kind: loop                   # decoder body per token
    termination: generation_eos
    setup: []
    steps:
      - kind: invoke
        component: decoder
        inputs: { embeddings: embeddings, cache: cache }
        outputs: { logits: logits, cache: cache.next }
    continue_when: continue
    max_iterations: max_output_tokens
    carried: [{ cell: cache, next: cache.next }]
```

### Diffusion

```yaml
steps:
  - kind: invoke                 # once: initialize latent
    component: initialize
    inputs: { noise: noise }
    outputs: { sample: latent.initial }
  - kind: loop
    setup: []                    # once on loop entry
    steps:                       # each solver step
      - kind: invoke
        component: denoiser
        inputs: { sample: latent, step: diffusion_step }
        outputs: { estimate: estimate }
      - kind: invoke
        component: solver
        inputs: { state: latent, estimate: estimate, step: diffusion_step }
        outputs: { next_state: latent.next }
    continue_when: continue
    max_iterations: num_steps
    iteration:
      value: diffusion_step
      contract: { dtype: int64, rank: 0, shape: [] }
    carried: [{ cell: latent, initial: latent.initial, next: latent.next }]
  - kind: invoke                 # once after loop
    component: decoder
    inputs: { latent: latent }
    outputs: { image: image }
  - kind: emit
    value: image
    output: image
    mode: replace
```

Package `inputs` and `outputs` are the only public boundary. `emit` is the only streaming/final
publication primitive, with `replace`, `append`, or `event` mode and optional integer
`valid_length`.

## Producer validation

The Rust library entry point `onnx_genai_metadata::load_pipeline_spec` parses and runs semantic
validation. CI and external producers can use the fail-closed CLI:

```bash
cargo run -p onnx-genai-metadata --bin validate_metadata -- \
  path/to/package-or-inference_metadata.yaml
```

A package directory resolves `inference_metadata.yaml`, `.yml`, or `.json`. The command accepts
multiple paths, reports each invalid document with an actionable error, and exits nonzero if any
fails. JSON Schema is useful for authoring, but this validator is authoritative for cross-component
SSA, recurrence, KV service-group, contract-binding, and capability invariants.

## v1 producer migration checklist

1. Replace loop `condition` with pre-test `continue_when`. Initialize it before loop entry; carry
   the next activity value when the body changes it. A false initial value is a valid zero-trip.
2. Emit per-row prefixes with `valid_length: int64[B]`; do not reduce across rows. Row emission
   follows from raggedness — there is no `row_ids`. Consume structured results through the output
   API; flattened `output.row.<position>` keys (plus event suffixes) are compatibility-only.
3. Declare serving `active: bool[B]`, `done: bool[B]`, and `accepted_len: int64[B]`. Inactive rows
   retain prior logical carry values. Do not declare `slot_ids`: row identity is runtime-private.
4. Bind each cache tensor state through `service_group`, and declare the cell
   `management: runtime` with a `release_boundary`. Each state group declares its semantic `kind`,
   `sequence_axis`, `layout`, graph-visible `logical_lengths: int64[B]`, past/present `aliasing`,
   reuse semantics, rollback/snapshot/fork capabilities, and component/cell port aliases. Declare
   `aliasing: permitted` (or `required`) when the graph is correct with a shared buffer; it
   defaults to `forbidden`.
   Every row-scoped native or stateful component declares `row_scope` and any
   `cache_affects_state` facts, and must implement the mandatory `compact`/`release` row ABI.
5. Keep ONNX ports omitted when artifact inference is sufficient. Request inputs use
   `source: { kind: request }`; the versioned runtime `role` is the sole request-field identity.
6. Run `validate_metadata` on every generated package directory. JSON Schema alone is not
   sufficient; the CLI also checks semantic invariants, package-relative artifact existence, and
   package-root containment.
