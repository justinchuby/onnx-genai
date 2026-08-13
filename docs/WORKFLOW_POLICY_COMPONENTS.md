# Workflow and component producer contract

Inference metadata is a concise structured workflow. Tensor math belongs in ONNX artifacts;
the runtime interprets only `sequence`, `invoke`, `loop`, `branch`, and `emit`.

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

`emit.when` optionally suppresses an event. `emit.valid_length` accepts `int[B]` and emits a ragged
event per active row, slicing each row to its runtime prefix. This defines EOS behavior explicitly:
a workflow may emit the EOS token, suppress it with `when`, or emit a zero-length row.
Ragged package values use `output.row.<index>`; event mode adds
`output.row.<index>.<event-index>`. Once an output becomes ragged, later appends without a length
are split by row and append to the same row streams.

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

## Compact structural examples

### Decoder

```yaml
steps:
  - kind: invoke                 # once: tokenize/bind prompt
    component: tokenizer
    inputs: { text: prompt }
    outputs: { tokens: prompt_tokens }
  - kind: loop
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
  slot_ids: slot_ids
  kv_service:
    paging: paged
    allocation: runtime
    compaction: true
    groups:
      decoder_cache:
        sequence_axis: 2
        layout: bnsh
        logical_lengths: cache_lengths
        storage: shared_buffer
        ports:
          decoder:
            cache: { input: past_key_values, output: present_key_values }
```

The service group is the allocation contract: `active`, `accepted_len`, and the semantic
`logical_lengths: int64[B]` cell drive paging, slots, per-row growth/rollback, compaction, and
permitted past/present aliasing. Both cache and length cells are loop-carried; inactive rows retain
their prior lengths. The workflow carry remains logical decoder dataflow. A separate state-update
ONNX component is used only for real tensor transforms such as gather or truncation.

The service binding is the only serialized storage-selection contract. Legacy top-level
`kv_cache`, `model.runtime_configurable.kv_cache`, and `model.io.kv_update` hints are rejected.
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
2. Emit per-row prefixes with `valid_length: int64[B]`; do not reduce across rows. Consume ragged
   results from `output.row.<index>` (and the event suffix for event mode).
3. Declare serving `active: bool[B]`, `done: bool[B]`, `accepted_len: int64[B]`, and
   `slot_ids: int64[B]`. Inactive rows retain prior logical carry values.
4. Bind each cache tensor state through `service_group`. Each KV group declares
   `sequence_axis`, `layout`, semantic `logical_lengths: int64[B]`, storage mode, and
   component/cell past-present port aliases. Use `shared_buffer` to permit physical aliasing.
5. Keep ONNX ports omitted when artifact inference is sufficient. Request inputs use
   `source: { kind: request }`; the versioned runtime `role` is the sole request-field identity.
6. Run `validate_metadata` on every generated package directory. JSON Schema alone is not
   sufficient; the CLI also checks semantic invariants, package-relative artifact existence, and
   package-root containment.
