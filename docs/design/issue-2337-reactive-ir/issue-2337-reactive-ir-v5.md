# Cyclic reactive dataflow IR — v5 (pressure-tested)

## Corrections from non-LLM/speculative workloads

1. **Atomicity is per reaction, not per SCC.** Under the v1 single clock, every state update in one reaction commits or aborts together. This covers diffusion latent/scheduler/RNG even when their graph edges place them in separate SCCs. Atomicity is still per workflow invocation/row group, never a device-wide or cross-request barrier.
2. **Graph-visible ONNX I/O must be statically typed.** ONNX permits incomplete ValueInfo in general; this package standard does not. Every ONNX port used by a workflow edge/state must declare sufficient dtype/rank/shape constraints to prove compatibility, or package loading fails and names the deficient graph port.
3. **Append separates candidate production from commit selection.** Speculative and beam execution may evaluate K candidates and commit M selected positions.
4. **No first-class state-version fork in v1.** Candidate trees are tensor/component dataflow. A state has one candidate `next`; transition commit selects the accepted prefix/path. Unselected candidates never become committed state.
5. **`reaction.index` is precisely current-reaction data.** Reaction 0 observes index 0. The predicate is computed during that reaction and sampled after its commit. A component deciding whether another reaction is needed compares the completed-count semantics explicitly; no implicit pre/post increment exists.

## Implicit node outputs

A node instance declares inputs but does not redeclare component outputs:

```yaml
components:
  denoiser:
    implementation: {kind: onnx, artifact: denoiser.onnx}
    # Output names/types come from ONNX ValueInfo.

  runtime_policy:
    implementation: {kind: binding}
    ports:
      # Binding signatures are explicit because no ONNX artifact defines them.
      inputs: {...}
      outputs:
        token: {dtype: int64, rank: 1, shape: [batch]}

nodes:
  conditional:
    kind: component
    component: denoiser
    inputs: {...}

  sample:
    kind: component
    component: runtime_policy
    inputs: {...}

  consumer:
    kind: component
    component: solver
    inputs:
      estimate: conditional.noise_pred  # ONNX-declared output
      token: sample.token               # binding-declared output
```

This is sufficient because:

- component signatures are authoritative;
- node ID namespaces each invocation;
- unused outputs need no declaration;
- the same component can be instantiated multiple times under different node IDs;
- output aliasing/renaming is unnecessary—the endpoint itself is the SSA value;
- workflow publication is the only explicit external output binding.

## Final transition vocabulary

### Replace

```yaml
transition:
  kind: replace
```

The entire candidate `next` becomes committed state.

### Append with prefix commit

```yaml
transition:
  kind: append
  axis: 2
  candidate_increment: proposal.evaluated
  commit:
    kind: prefix
    length: acceptance.length
  bound: package.max_context
```

`next` may contain all K evaluated positions. Only the first M positions after current state become visible, where M is `acceptance.length`.

Ordinary Qwen decode is the same operation:

```yaml
transition:
  kind: append
  axis: 2
  candidate_increment: package.one_token
  commit: {kind: prefix, length: accepted_len.next}
  bound: package.max_context
```

### Append with gathered-path commit

```yaml
transition:
  kind: append
  axis: 2
  candidate_increment: candidate_tree.evaluated
  commit:
    kind: gather
    indices: acceptance.path_indices
    length: acceptance.length
  bound: package.max_context
```

The gather indices select candidate positions in commit order. They are request-aligned integer tensor data, not physical page IDs or runtime slots.

### Indexed scatter

```yaml
transition:
  kind: indexed_scatter
  axis: 2
  indices: write_cursor.current
  candidate_count: proposal.evaluated
  commit:
    kind: prefix
    length: acceptance.length
  logical_length: cache_lengths.current
  capacity: package.max_context
```

The core defines logical destinations and committed positions. Physical pages, block tables, allocators, and device placement remain runtime-owned.

## Default reaction transaction

```text
reaction i:
  1. snapshot every state participating in this clock
  2. evaluate the init-free instantaneous graph using index=i
  3. stage every state candidate and transition selection
  4. if any required node/effect fails: abort all staged state
  5. otherwise: atomically commit all state for this reaction
  6. publish commit-coupled outputs
  7. sample repeat_while; if true, begin reaction i+1
```

An implementation may commit independent descriptors/pages in parallel, but visibility is atomic. No unrelated request waits on this transaction.

## Workload proof 1: diffusion / flow

```yaml
graph:
  clock:
    kind: synchronous
    index: reaction.index
    repeat_while: continue.next
    sample_repeat: after_commit

  nodes:
    noise:
      kind: component
      component: latent_noise
      inputs: {seed: request.seed}

    denoise:
      kind: component
      component: denoiser
      inputs:
        sample: latent.current
        timestep: schedule_lookup.timestep
        conditioning: conditioning.hidden

    solve:
      kind: component
      component: solver_step
      inputs:
        sample: latent.current
        estimate: denoise.noise_pred
        history: history.current
        step: reaction.index
        schedule: schedule.values

    latent:
      kind: state
      initial: noise.sample
      next: solve.next_state
      transition: {kind: replace}

    history:
      kind: state
      initial: history_init.zeros
      next: solve.next_history
      transition: {kind: replace}
```

Classifier-free guidance is two denoiser nodes plus a pure combine node. Schedule, latent, history, and RNG commit in one reaction even if they are separate SCCs. VAE decode is downstream of final committed state and need not be inside the cycle.

## Workload proof 2: RNN / SSM / linear attention

```yaml
nodes:
  cell:
    kind: component
    component: recurrent_step
    inputs:
      hidden_states: request.hidden_states
      accumulator: recurrent.current.accumulator
      conv_history: recurrent.current.conv_history

  recurrent:
    kind: state
    ownership: runtime
    scope: session
    release: session_end
    transition: {kind: replace}
    members:
      accumulator:
        initial: request.linear_accumulator
        next: cell.next_accumulator
      conv_history:
        initial: request.conv_history
        next: cell.next_conv_history
```

No recurrent/SSM/conv state kind is needed. The component graph and replace transition completely determine runtime behavior; optional interfaces may prove a specialized substitution.

## Workload proof 3: speculative / candidate tree

```yaml
nodes:
  propose:
    kind: component
    component: proposer
    inputs:
      tokens: tokens.current
      width: proposal_width.current

  verify:
    kind: component
    component: verifier
    inputs:
      proposed_tokens: propose.tokens
      past_key_values.0.key: verifier_cache.current.key.0
      past_key_values.0.value: verifier_cache.current.value.0
      # ...

  accept:
    kind: component
    component: speculative_acceptance
    inputs:
      target_scores: verify.scores
      proposed_tokens: propose.tokens
      parent_indices: propose.parent_indices
      seed: request.seed
      counter: rng.current

  verifier_cache:
    kind: state
    ownership: runtime
    transition:
      kind: append
      axis: 2
      candidate_increment: verify.evaluated
      commit:
        kind: gather
        indices: accept.path_indices
        length: accept.accepted_len
      bound: package.max_context
    members:
      key.0:
        initial: request.verifier_key_0
        next: verify.present.0.key
      value.0:
        initial: request.verifier_value_0
        next: verify.present.0.value
      # ...

  tokens:
    kind: state
    initial: request.tokens
    next: accept.committed_tokens
    transition: {kind: replace}

  grammar:
    kind: state
    initial: request.grammar
    next: grammar_commit.next
    transition: {kind: replace}

  rng:
    kind: state
    initial: request.rng_counter
    next: accept.next_counter
    transition: {kind: replace}
```

Verifier cache, tokens, grammar, RNG, adaptive proposal state, active/done, and lengths commit in the same reaction. Rejected branches remain uncommitted candidate data. Runtime page reservation/release is a lowering of the gather commit, not authored graph state.

## Linear effect tokens

```yaml
effects:
  external_store:
    retry: transactional
    speculation_safety: {...}

nodes:
  read:
    kind: component
    component: external_read
    effects: {external_store: effects.external_store.start}

  write:
    kind: component
    component: external_write
    inputs: {value: read.value}
    effects: {external_store: read.effects.external_store}

effect_sinks:
  external_store: write.effects.external_store
```

Effect tokens are linear typed endpoints. Fan-out, missing sinks, unordered same-domain writers, and unsafe retry/speculation combinations fail validation. A gated effect must pass through an explicit `effect_merge`.

## Final v1 primitive set

- component node with implicit signature-defined outputs;
- generic scalar or bundled state node;
- `replace`, `append`, and `indexed_scatter` transitions;
- `prefix` and `gather` commit selection;
- scalar firing gate with explicit value/effect merge;
- single synchronous reaction clock with current zero-based index;
- reaction-wide atomic commit and post-commit repeat predicate;
- typed linear effect tokens;
- output publication bindings;
- optional versioned interfaces that prove legal optimized substitution.

No sequence, loop, branch, invoke, emit, carried-value list, state semantic kind, physical cache plan, node output redeclaration, or second decoder ABI remains.
