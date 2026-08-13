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
    condition: continue
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

ONNX port contracts may be omitted when the artifact is authoritative. Declare only semantic
bindings, bounds/overrides, cross-component constraints, or adapter ports that ONNX cannot describe.
Device transfers are planner-derived from resource placement rather than authored workflow nodes.

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
| `temperature`, `top_k`, `top_p` | request-provided scalar or `[B]` when used |
| `grammar_mask` | optional request/adapter-provided `bool[B,V]` |
| `rng_seed`, `rng_offset` | required for seeded stochastic mode, `int64[B]` |
| `rng_next_offset` | seeded mode output, `int64[B]` |

`parameters.mode` is `greedy` or `seeded_stochastic`. Sampling controls are typed workflow inputs;
they are never baked into the artifact, so ordinary request option changes do not regenerate
ONNX. Seed and offset are explicit counter-based RNG state (for example Philox or Threefry),
loop-carried as semantic state.

A component may set `application_overridable: true`. The application can then select another
package-declared ONNX component for that invocation. The replacement must implement the same
contract ID/version, semantic binding set, tensor port ABI, and effects. Concrete ONNX port names
may differ because the runtime remaps them through `contract.bindings`. This is the extension
point for fundamentally custom sampling policy; changing temperature, top-k, top-p, seed, or a
grammar mask uses ordinary workflow inputs instead.

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
Estimate state is advisory and request-scoped; changing or resetting it must not change output
correctness or distribution.

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
    condition: continue
    max_iterations: max_output_tokens
    carried:
      - { cell: cache, next: cache.next }
```

Here `cache.next` is the decoder's present-KV output and becomes `cache` on the next iteration.
The carry describes logical dataflow and bounded growth; the generic KV service may realize it
with shared or paged storage without adding a workflow component.

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
    condition: continue
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
    condition: continue
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
