# Workflow policy component contract

This is the producer interface for policy math used by workflow metadata. Policy
components are ordinary ONNX components invoked by SSA workflow nodes. The
`policy` block maps stable semantic roles to physical ONNX port names; the
component `ports` block supplies each port's `TensorContract`.

There is no host fallback for sampling, solving, masking, speculative acceptance,
or state-update math. A package requiring that behavior must ship the ONNX
artifact. Admission/KV paging/compaction remain runtime services declared by the
workflow serving contract.

## Common rules

- `B`, `T`, `K`, `V`, and state dimensions are symbols unified with package and
  component contracts.
- Floating ports may use `float16`, `bfloat16`, `float32`, or `float64`.
- Token, index, counter, and length ports are integer tensors; producers should
  use `int64` unless an artifact contract requires another integer dtype.
- Predicates and masks are `bool`.
- Every policy declares an effect domain also listed in the component's
  `effects`. Its `invoke` consumes the current linear token and produces the next.
- Seeded operations use counter-based Philox or Threefry implemented by the ONNX
  artifact. `seed`, `offset`, and `next_offset` are `int64[B]`. Seed and offset
  are inputs; next offset is an output and is loop-carried. Greedy sampling must
  not declare RNG ports.
- Physical ONNX names are unrestricted. Workflow invocation maps SSA values to
  those names explicitly.

An invocation has this common form:

```yaml
kind: invoke
component: policy
inputs: { physical_input: ssa.input }
outputs: { physical_output: ssa.output }
effects:
  policy_effect: { consumes: effect.0, produces: effect.1 }
```

## Versioned adapter invocation

Pre/post-processing adapters are ordinary workflow components, not out-of-band
bindings. Their `implementation.kind` is `adapter`; `abi` and `version` must be
pinned identically in `manifest.adapter_abis`. Inputs and outputs use the same
typed port maps, SSA scoping, state cells, and effect transitions as ONNX
components. Opaque application inputs are passed only through explicitly named
workflow inputs.

The runtime registry currently provides `onnx-genai.image-preprocess@1`:

| Port | Direction | Contract |
|---|---|---|
| `encoded` | input | `uint8[encoded_bytes]`, one encoded image |
| producer-declared ports | output | exact `TensorContract` from `preprocessing.image.outputs[].contract` |

Each `preprocessing.image.outputs[]` entry names its processor-local `source`
and the workflow SSA value in `name`. The adapter invoke maps a physical output
port to that same SSA name. Legacy `component.input` endpoint binding is invalid
in workflow documents.

```yaml
preprocessing:
  image:
    transforms:
      - { op: decode, outputs: [decoded] }
      - { op: convert_rgb, inputs: [decoded], outputs: [rgb] }
      - { op: resize, inputs: [rgb], outputs: [pixels], size: 224,
          mode: stretch, interpolation: bilinear }
    outputs:
      - { source: pixels, name: image.pixel_values, content: pixels,
          dtype: float32,
          contract: { dtype: float32, rank: 4, shape: [1, 3, 224, 224] } }
# manifest.adapter_abis: { onnx-genai.image-preprocess: "1" }
# graph:
- kind: invoke
  component: image_preprocess
  inputs: { encoded: request.image }
  outputs: { pixel_values: image.pixel_values }
  effects: {}
```

Stateful streaming tokenizer/media adapters declare state ports/cells and
linear effect transitions. Other media codecs and postprocessors must be ONNX
components or separately versioned registered ABIs; the runtime does not infer
semantics or provide model-specific host fallbacks. Public outputs should be
emitted after the post-adapter invoke and declare `stage: post_adapter`.

## Grammar guidance adapter

Grammar is the versioned stateful adapter ABI
`onnx-genai.grammar-guidance@1`, reusable by every workflow. Producers compile
the grammar to a dense DFA transition tensor. The runtime adapter advances or
clones semantic grammar state; it never samples or edits logits.

Each grammar component declares `adapter.role: grammar_guidance` and one
`action`: `clone`, `lookahead`, or `commit`.

| Semantic port | Direction | Contract |
|---|---|---|
| `state` | input | `int64[B]`, current DFA state |
| `tokens` | input | `int64[B,T]` |
| `valid_length` | input | `int64[B]`, requested token prefix |
| `transition_table` | input | `int64[S,V]`; `-1` is invalid, otherwise next state |
| `next_state` | output | `int64[B]` |
| `consumed_length` | output | `int64[B]`, valid prefix consumed |
| `logits_mask` | output | `bool[B,V]` for the resulting state |
| `forced_tokens` | output | `int64[B,1]` |
| `forced_length` | output | `int64[B]`, zero or one |

`clone` ignores tokens, copies the state, and emits guidance for that state;
bind `valid_length` to an explicit zero tensor. `lookahead` advances a clone
until the first invalid token and reports the consumed prefix without changing
the committed cell. `commit` advances committed state and fails if any token in
the requested prefix is invalid. Every action consumes and produces the
declared grammar effect token.

```yaml
- kind: invoke
  component: grammar_clone
  inputs: { state: grammar.body, tokens: proposal.tokens,
            valid_length: zero, transition_table: grammar.transitions }
  outputs: { next_state: grammar.clone, consumed_length: clone.consumed,
             logits_mask: clone.mask, forced_tokens: clone.forced,
             forced_length: clone.forced_length }
  effects: { grammar: { consumes: grammar.0, produces: grammar.clone_effect } }
- kind: invoke
  component: grammar_lookahead
  inputs: { state: grammar.clone, tokens: proposal.tokens,
            valid_length: proposal.length, transition_table: grammar.transitions }
  outputs: { next_state: grammar.lookahead, consumed_length: grammar.valid_length,
             logits_mask: grammar.lookahead_mask,
             forced_tokens: grammar.lookahead_forced,
             forced_length: grammar.lookahead_forced_length }
  effects:
    grammar: { consumes: grammar.clone_effect, produces: grammar.lookahead_effect }
```

An ONNX component combines `grammar.valid_length` with verifier acceptance,
applies `logits_mask`, and honors `forced_tokens`. After verification, invoke
`commit` from the original committed state with the final accepted prefix. There
is no speculative-specific grammar runtime path.

## Token sampler

`role: token_sampler`

| Semantic port | Direction | Contract |
|---|---|---|
| `logits` | input | float `[B,V]` |
| `temperature` | optional input | float `[B]` |
| `top_k` | optional input | integer `[B]` |
| `top_p` | optional input | float `[B]` |
| `seed`, `offset` | stochastic input | `int64[B]` |
| `token` | output | integer `[B]` |
| `next_offset` | stochastic output | `int64[B]` |

Minimal workflow:

```yaml
kind: invoke
component: sampler
inputs: { logits: decoder.logits, seed: request.seed, offset: rng.offset }
outputs: { token: sampled.token, next_offset: rng.next_offset }
effects: { rng: { consumes: rng.0, produces: rng.1 } }
```

Use `mode: greedy` without RNG mappings, or `mode: seeded_stochastic` with an
`rng` role mapping.

## Termination predicate

`role: termination_predicate`

| Semantic port | Direction | Contract |
|---|---|---|
| `tokens` | input | integer `[B]` |
| `eos_ids` | input | integer `[E]` |
| `iteration`, `max_iterations` | input | integer `[B]` |
| `done` | output | `bool[B]` |

Minimal workflow:

```yaml
kind: invoke
component: termination
inputs: { tokens: sampled.token, eos_ids: package.eos, iteration: loop.i,
          max_iterations: request.max_iterations }
outputs: { done: loop.done }
effects: { termination: { consumes: termination.0, produces: termination.1 } }
```

The ONNX graph defines EOS and limit semantics. The loop condition reads the
declared scalar or per-row done value according to the serving contract.

## Loop induction SSA

A loop may declare its zero-based current iteration as a typed SSA value:

```yaml
kind: loop
setup: { ... }
body: { ... }
condition: loop.continue
max_iterations: request.max_iterations
iteration:
  value: loop.i
  contract: { dtype: int64, rank: 1, shape: [batch] }
carried: [...]
```

The contract is `int64` rank 0, or rank 1 with an explicit broadcast shape.
The executor materializes `0, 1, ...` before each body execution. The value is
available to the body and its condition, including solver steps, schedules,
RNG counters, and emit indices. It is not available in setup and does not
escape the loop. Nested loops must use distinct names; lexical shadowing is
rejected. Reverse indices, remaining counts, and other derived values are
computed by an invoked ONNX component from this primitive induction value.

## Solver or scheduler step

`role: solver_step`

| Semantic port | Direction | Contract |
|---|---|---|
| `state` | input | float `[B,...]` |
| `estimate` | input | float, shape compatible with `state` |
| `step` | input | integer `[B]` |
| `schedule` | input | float `[S]` |
| `next_state` | output | same contract as `state` |

Minimal workflow:

```yaml
kind: invoke
component: solver
inputs: { state: latent.current, estimate: denoiser.estimate,
          step: loop.i, schedule: package.schedule }
outputs: { next_state: latent.next }
effects: { solver: { consumes: solver.0, produces: solver.1 } }
```

## Masked update

`role: masked_update`

| Semantic port | Direction | Contract |
|---|---|---|
| `state`, `proposal` | input | integer `[B,T]` |
| `mask` | input | `bool[B,T]` |
| `step` | input | integer `[B]` |
| `seed`, `offset` | optional RNG input | `int64[B]` |
| `next_state` | output | same contract as `state` |
| `next_mask` | output | `bool[B,T]` |
| `next_offset` | optional RNG output | `int64[B]` |

Minimal workflow:

```yaml
kind: invoke
component: masked_update
inputs: { state: tokens.current, proposal: denoiser.proposal, mask: mask.current,
          step: loop.i, seed: request.seed, offset: rng.offset }
outputs: { next_state: tokens.next, next_mask: mask.next,
           next_offset: rng.next_offset }
effects: { update: { consumes: update.0, produces: update.1 } }
```

## Speculative verifier

`role: speculative_verifier`

| Semantic port | Direction | Contract |
|---|---|---|
| `target_scores` | input | float `[B,K,V]` |
| `proposed_tokens` | input | integer `[B,K]` |
| `proposal_scores` | optional input | float `[B,K,V]` |
| `seed`, `offset` | optional RNG input | `int64[B]` |
| `accepted_tokens` | output | integer `[B,K]` |
| `accepted_len` | output | integer `[B]` |
| `done` | output | `bool[B]` |
| `next_offset` | optional RNG output | `int64[B]` |

Minimal workflow:

```yaml
kind: invoke
component: verifier
inputs: { target_scores: target.scores, proposed_tokens: proposal.tokens,
          proposal_scores: proposal.scores, seed: request.seed, offset: rng.offset }
outputs: { accepted_tokens: accepted.tokens, accepted_len: accepted.len,
           done: loop.done, next_offset: rng.next_offset }
effects: { verify: { consumes: verify.0, produces: verify.1 } }
```

Acceptance and correction-token math belongs entirely to the artifact.
`accepted_len` may also drive the generic serving service's row/KV bookkeeping.

## Adaptive proposal budget

`role: adaptive_proposal_budget` is ONNX policy math. It observes the same typed
signals as adaptive speculative scheduling without introducing a host
`AdaptiveKController`:

| Semantic port | Direction | Contract |
|---|---|---|
| `current_k` | input | integer `[B]` |
| `accepted` | input | integer `[B]` |
| `evaluated` | input | integer `[B]` |
| `committed_tokens` | input | integer `[B]` |
| `filled_proposal_budget` | input | `bool[B]` |
| `draft_ms`, `target_ms` | input | float `[B]` |
| `estimates` | input | float `[B,K_slots]` advisory estimator state |
| `next_k` | output | integer `[B]` |
| `next_estimates` | output | same contract as `estimates` |

The artifact may estimate per-K throughput and probe adjacent K values. Both
`current_k` and `estimates` are loop-carried `class: advisory` state. They may
control proposal work/budget only; they must not feed token selection, grammar,
RNG, verifier acceptance, or any path that changes output correctness or
distribution.

```yaml
kind: invoke
component: adaptive_budget
inputs: { current_k: proposal_k.body, accepted: verifier.accepted,
          evaluated: verifier.evaluated, committed_tokens: committed.length,
          filled_proposal_budget: proposal.filled, draft_ms: telemetry.draft_ms,
          target_ms: telemetry.target_ms, estimates: adaptive.body }
outputs: { next_k: proposal_k.next, next_estimates: adaptive.next }
effects: { adaptive: { consumes: adaptive.0, produces: adaptive.1 } }
```

Timing may be supplied by the application or by the optional generic
`onnx-genai.telemetry@1` adapter. `action: start` produces an `int64` scalar
monotonic timestamp; `action: elapsed` consumes it and produces a `float32`
scalar duration in milliseconds. Any batching/broadcast is explicit ONNX math.

### Runtime-length emission

An `emit` may name `valid_length`, an integer scalar or rank-one SSA value. At
execution it must contain exactly one non-negative element. The runtime emits
`value[..., 0:valid_length]`: slicing is always on the final axis and happens
before package-output contract validation. The length must not exceed that axis.
This applies uniformly to `replace`, `append`, and `event`.

The source and package-output contracts must have the same dtype and rank and
compatible non-final dimensions; the output's final dimension may use a distinct
symbol. Packages using this operand declare the `emit_valid_length` capability.
Per-row lengths for a batch are not a dense tensor prefix and must instead use a
declared ragged/offset contract or separate row events.

```yaml
kind: emit
value: accepted.tokens       # int64[B,K]
valid_length: accepted.len   # int64 scalar or [B], containing one element at runtime
output: tokens               # int64[B,A], A = accepted.len
mode: append
effect_name: stream
effect: { consumes: stream.0, produces: stream.1 }
```

## Generic state update

`role: state_update`

| Semantic port | Direction | Contract |
|---|---|---|
| `current` | input | declared state-cell contract |
| `update` | input | component-defined typed update payload |
| `next` | output | same contract as `current` |

Minimal workflow:

```yaml
kind: invoke
component: state_update
inputs: { current: cache.current, update: decoder.cache_delta }
outputs: { next: cache.next }
effects: { state: { consumes: state.0, produces: state.1 } }
```

Append, scatter, accumulation, paging policy, and other tensor mutation math are
not workflow operators. The ONNX component computes `next`; loop carry or session
state wiring publishes it as the next cell version.

### Rollback and selected-prefix state

Rollback is ordinary policy math, not a KV-specific host operation. An ONNX
component consumes tentative state plus a typed selection/length and returns the
selected state. A branch phi chooses that result or a correction result, and the
joined SSA value is the loop carry's `body_output`:

```yaml
state:
  history:
    contract: { dtype: float32, rank: 3, shape: [batch, sequence, width] }
    scope: invocation
    initializer: history.current
    recurrence: { kind: bounded, axis: 1, max: max_context }

# In the loop body:
- kind: invoke
  component: accepted_prefix
  inputs: { state: history.body, valid_length: accepted.len }
  outputs: { selected: history.accepted }
  effects: {}
- kind: branch
  predicate: accept
  cases:
    "true":
      kind: invoke
      component: binding
      inputs: { value: history.accepted }
      outputs: { value: branch.accepted }
      effects: {}
    "false":
      kind: invoke
      component: correction
      inputs: { state: history.body }
      outputs: { selected: branch.corrected }
      effects: {}
  outputs:
    history.selected:
      cases: { "true": branch.accepted, "false": branch.corrected }
  effects: {}
```

`bounded` recurrence is generic shape policy: the declared axis may grow or
shrink between iterations, both the current and next extent must be at most the
non-negative integer `max` SSA value, and dtype, rank, and every other axis remain
invariant. Packages using it declare `bounded_state_recurrence`. The invoked
component performs all truncation, gathering, or compaction.

## Semantic and advisory state

Every state cell declares `class`:

- `semantic` (the default): KV, RNG, grammar, and any value that can affect
  outputs. It participates in checkpoint/replay and may use invocation or
  session scope. Session mutation remains lease-ordered.
- `advisory`: telemetry-derived estimates and proposal budgets that affect only
  work scheduling. Advisory cells must use invocation scope, reset on every
  request, and are never serialized as session state.

Both classes remain explicit loop-carried SSA with state read/write effects.
Class does not authorize hidden host mutation.

### Semantic checkpoint and replay

The runtime exposes `checkpoint_session(session_id)` and
`restore_session_checkpoint(session_id, checkpoint)`. A checkpoint contains
only `class: semantic`, `scope: session` cells. Restoring first removes the
session's current semantic cells and then installs cloned checkpoint values;
advisory state is never captured or restored. Replaying the same workflow
inputs from that checkpoint must reproduce semantic outputs.

A world-model workflow uses no special runtime dispatch:

```yaml
- invoke observation encoder: observation + latent.body -> latent.observed
- invoke action policy: latent.observed -> action.selected
- branch action.selected:
    true:  invoke environment step A -> environment.low
    false: invoke environment step B -> environment.high
  phi: latent.next
- loop-carry latent.next as semantic session state
- emit action events and final latent state
```

The executable conformance package covers observation ingestion, latent session
state, action selection, environment branching, looping, event/final emission,
checkpoint, state advancement, restore, and deterministic replay.

## Conditional joins

`branch` keeps case-local SSA and effect tokens isolated. Values escape only through
typed `outputs` phi mappings, with one source for every case and the default when
present. Each output's source contracts must unify.

Linear effects use explicit `effects` merges. Every case starts from the same
`incoming` token and may produce a distinct local successor; the branch publishes one
new `produces` token for subsequent nodes:

```yaml
kind: branch
predicate: proposal.accepted
cases:
  "true":
    kind: invoke
    component: accept_state
    inputs: { value: proposal.tokens }
    outputs: { value: accepted.tokens }
    effects: { state: { consumes: state.0, produces: state.accepted } }
  "false":
    kind: invoke
    component: correction_state
    inputs: { value: verifier.tokens }
    outputs: { value: corrected.tokens }
    effects: { state: { consumes: state.0, produces: state.corrected } }
outputs:
  next.tokens:
    cases: { "true": accepted.tokens, "false": corrected.tokens }
effects:
  state:
    incoming: state.0
    cases: { "true": state.accepted, "false": state.corrected }
    produces: state.joined
```

Case-local names such as `accepted.tokens` and `corrected.tokens` are unavailable
after the branch; only `next.tokens` and `state.joined` escape.
