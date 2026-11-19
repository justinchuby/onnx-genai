# Cyclic reactive dataflow IR — v6 temporal semantics

v6 extends the pressure-tested v5 design with derived prelude/reaction/postlude execution. It does not add authored phases.

## Lifecycle endpoints

A state node exposes two read endpoints:

```yaml
nodes:
  latent:
    kind: state
    initial: noise.sample
    next: solver.next_latent
    transition: {kind: replace}

  denoise:
    kind: component
    component: denoiser
    inputs:
      latent: latent.current     # committed snapshot for this reaction

  decode_image:
    kind: component
    component: vae_decoder
    inputs:
      latent: latent.final       # present once after the final commit
```

The clock exposes lifecycle values:

```text
reaction.index   int64 scalar, current zero-based reaction ordinal
clock.completed  unit/event, present once after repeat_while becomes false
```

`clock.completed` allows a data-free or effect-only postlude to be represented without a `phase` annotation.

## Temporal availability classes

Every value is assigned one inferred availability class:

| Class | Meaning | Examples |
|---|---|---|
| `stable` | Available unchanged for the invocation; may be hoisted | request/package inputs, pure nodes depending only on stable values |
| `reaction` | Produced once per reaction | `reaction.index`, `state.current`, nodes depending on reaction values |
| `final` | Produced once after final reaction commit | `state.final`, `clock.completed`, nodes depending on final values |

Rules:

1. Authors do not write an availability/phase label.
2. A pure node's class is the join of its input classes: `stable < reaction < final`.
3. A node consuming a maybe-present scalar-gated value inherits that presence condition unless an explicit merge removes it.
4. `state.initial` accepts only stable values. Any transitive dependency on `state.current`, `state.final`, `reaction.index`, or `clock.completed` is invalid.
5. `state.next` accepts reaction values only.
6. `repeat_while` accepts scalar-bool reaction values only.
7. A reaction value cannot feed a final-only consumer without crossing `state.final` or an explicit final sampling primitive.
8. A final value cannot feed state.next or repeat_while.

## Derived regions

The compiler derives, rather than reads, three execution regions:

```text
prelude:
  pure stable closure required by state.initial and reaction inputs
  + initialization effects explicitly terminated before clock start

reaction:
  nodes depending on reaction.index or state.current
  + their data/effect dependencies
  + state.next candidates, output publications, and repeat_while

postlude:
  nodes depending on state.final or clock.completed
  + final output publications/effects
```

A stable pure node used by both reaction and postlude executes once and its value is reused. It is not duplicated into two authored regions.

## Diffusion proof

```yaml
graph:
  clock:
    kind: synchronous
    index: reaction.index
    repeat_while: continue.value
    sample_repeat: after_commit

  nodes:
    schedule:
      kind: component
      component: diffusion_schedule
      inputs: {}                         # stable, hoisted

    noise:
      kind: component
      component: latent_noise
      inputs: {seed: request.seed}       # stable, prelude

    denoise:
      kind: component
      component: denoiser
      inputs:
        sample: latent.current           # reaction
        timestep: timestep.value
        conditioning: conditioning.value # stable, reused

    solver:
      kind: component
      component: solver_step
      inputs:
        sample: latent.current
        estimate: denoise.noise_pred
        step: reaction.index
        schedule: schedule.values

    latent:
      kind: state
      initial: noise.sample
      next: solver.next_state
      transition: {kind: replace}

    final_scale:
      kind: component
      component: tensor_scale
      inputs:
        tensor: latent.final              # final
        scale: package.decoder_scale

    vae:
      kind: component
      component: vae_decoder
      inputs:
        latent: final_scale.scaled        # final; executes once

outputs:
  latent_trajectory:
    publication:
      value: solver.next_state            # reaction; appends every commit
      mode: append
      axis: 3

  latent:
    publication:
      value: latent.final                 # final; publishes once
      mode: replace

  image:
    publication:
      value: vae.image                    # final; publishes once
      mode: replace
```

This preserves the current structured workflow's `loop` followed by one VAE decode without serializing a loop body or finalize phase.

## Final-state semantics

For reaction `i`:

```text
current_i = committed state before reaction i
candidate_i = state.next computed during reaction i
committed_i = transition(current_i, candidate_i, commit selection)
```

If the post-commit repeat predicate is false:

```text
state.final = committed_i
clock.completed becomes present
```

If the predicate is true, `committed_i` becomes `state.current` for reaction `i+1` and `state.final` remains absent.

For bundled state, endpoints are member-qualified:

```text
decoder_cache.current.key.0
decoder_cache.final.key.0
```

## Reaction-wide atomicity

All state under the single v1 clock commits atomically per reaction, regardless of SCC membership. `state.final` becomes visible only after that complete transaction succeeds. Output publications fed by reaction values occur after each successful reaction commit; publications fed by final values occur after the final commit.

## Effect lifecycle

Each effect domain has a separate linear chain for each derived execution class:

```text
effects.<domain>.prelude_start  -> ... -> prelude sink

effects.<domain>.reaction_start -> ... -> reaction sink
  (a fresh reaction token is issued only after the previous reaction commits)

effects.<domain>.final_start    -> ... -> final sink
```

The chain used by a node is inferred from its data availability. A data-free effect node anchors itself by consuming one lifecycle token; no `phase` field is needed.

```yaml
nodes:
  final_notification:
    kind: component
    component: notify
    inputs: {completed: clock.completed}
    effects:
      notifications: effects.notifications.final_start
```

A node may not connect effect tokens from different lifecycle chains. Non-retryable reaction effects remain subject to the transaction safety rules from v5.

## Validation additions

Reject with provenance when:

- a state initializer is not stable;
- a state next candidate is not reaction-valued;
- a reaction value escapes directly into postlude execution;
- a final value feeds the reaction;
- a postlude node can fire before `clock.completed`/state.final is present;
- an effect token crosses lifecycle classes;
- a final publication depends on an uncommitted candidate rather than state.final;
- a stable-looking ONNX node is effectful and therefore cannot be hoisted.
