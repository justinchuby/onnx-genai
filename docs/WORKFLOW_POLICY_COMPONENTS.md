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
