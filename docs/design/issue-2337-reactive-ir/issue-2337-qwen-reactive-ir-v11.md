# Qwen cyclic reactive IR — v11

v11 tightens candidate representation, padded-prefill logits, and continuation admission.

## Candidate source is orthogonal to transition kind

`transition.kind` answers how committed state relates to old state:

```text
replace | append | indexed_scatter
```

`candidate_source.kind` answers what the producer emitted:

```text
full  = complete materialized candidate state, including the old prefix
delta = only this reaction's new candidate payload
```

### Qwen full candidate

```yaml
nodes:
  decoder_cache:
    kind: state
    transition:
      kind: append
      axis: 2
      candidate_source:
        kind: full
        suffix_length: cache_plan.candidate_count
      commit:
        kind: gather
        indices: cache_plan.indices
        length: cache_plan.commit_length
      bound: package.max_context
```

For logical current state `[A, B]`, reaction-0 `decode.present` may be:

```text
[A, B, prompt_0, PAD, prompt_1, sampled_token]
```

The declared candidate suffix is:

```text
[prompt_0, PAD, prompt_1, sampled_token]
```

A gather commit with indices `[0, 2, 3]` produces logical committed state:

```text
[A, B, prompt_0, prompt_1, sampled_token]
```

Validation requires the full candidate prefix to represent the same logical state as `decoder_cache.current`. This is a declared graph relationship; optimized interfaces may prove or implement it without comparing tensor contents.

### Delta-producing component

```yaml
nodes:
  sequence_state:
    kind: state
    transition:
      kind: append
      axis: 2
      candidate_source:
        kind: delta
        length: candidate.count
      commit:
        kind: prefix
        length: accepted.count
```

A delta source emits only new candidate positions. It avoids forcing a delta-native model to concatenate/materialize the old prefix.

`replace` consumes a full replacement and therefore needs no candidate-source choice. Indexed-scatter transitions use the same explicit full/delta distinction when a producer may emit either a complete updated buffer or only updates.

## Padded prefill logits

The old `last_token_logits` policy selects the last physical sequence column. That is correct only when the final valid prompt token is physically trailing.

v11 makes the logical selection explicit:

```yaml
nodes:
  prefill_context:
    kind: component
    component: prefill_context_plan
    when: {value: reaction.first, equals: true}
    inputs:
      input_ids: request.input_ids
      prompt_lengths: request.prompt_lengths
      prior_lengths: cache_lengths.current
      max_context: package.max_context

  prefill_logits:
    kind: component
    component: gather_sequence_logits
    when: {value: reaction.first, equals: true}
    inputs:
      logits: prefill.logits
      indices: prefill_context.last_valid_prompt_index
```

`last_valid_prompt_index` is per-row logical selection data derived from the declared prompt mask/layout. It supports left padding, right padding, and other declared layouts without model-name branches.

Steady decode has sequence width one, so `decode_logits` may use a direct squeeze/gather-at-zero component.

## Commit plan

The provisional binding/component plan remains:

```yaml
nodes:
  cache_plan:
    kind: component
    component: state_commit_plan
    inputs:
      first: reaction.first
      prompt_candidate_count: prefill_context.prompt_candidate_count
      prompt_indices: prefill_context.prompt_commit_indices
      prompt_length: prefill_context.prompt_commit_length
      accepted_length: accepted_length.total
```

Its prompt inputs are conditionally present iff `reaction.first` is true. That presence relation is part of the binding signature and statically checked.

Reaction 0:

```text
candidate_count = physical prompt width + 1
indices         = valid prompt candidate indices ++ sampled-token index when accepted
commit_length   = valid prompt length + accepted length
```

Reaction 1+:

```text
candidate_count = 1
indices         = [0] when accepted, otherwise absent/empty
commit_length   = accepted length
```

Indices are relative to the candidate suffix, never physical pages or cache slots.

## Session continuation

Qwen durable session state contains:

- decoder KV bundle;
- logical cache lengths;
- any other non-reconstructible session state required by the graph.

Attention mask is reconstructed on reaction 0 from durable lengths plus the new prompt. Packages with non-reconstructible sparse/custom masks persist those masks explicitly instead.

Qwen does **not** persist the full next-token logits tensor. A new session invocation therefore requires a non-empty prompt. This is an admission constraint, not a hidden runtime guess:

```yaml
inputs:
  request.input_ids:
    contract: {...}
    constraints:
      valid_sequence_length: {min: 1}
```

Scope of that rule:

- a new invocation on an existing durable session must append at least one prompt token;
- suspension/resumption of the same invocation retains working logits and is unaffected;
- another package may support empty-prompt continuation by explicitly persisting logits or declaring a read-only recomputation component.

`max_reactions = 0` remains a successful no-op. The non-empty prompt is validated, but no prefill, state mutation, or publication occurs when the clock does not start.

## Batch contract

The logits gather and commit-plan components are row-independent:

```yaml
components:
  gather_sequence_logits:
    implementation: {kind: binding}
    batching:
      kind: row_independent
      default_axis: 0
    ports:
      inputs:
        logits: {dtype: float32, rank: 3, shape: [batch, sequence, vocabulary]}
        indices: {dtype: int64, rank: 1, shape: [batch]}
      outputs:
        selected: {dtype: float32, rank: 2, shape: [batch, vocabulary]}
```

Binding signatures remain explicit. If the gather ships as ONNX instead, its dtype/rank/shape come from ONNX ValueInfo while the non-ONNX row transform remains explicit.

## Structural bundle interaction

A full candidate bundle and a delta bundle preserve each member's individual signature. The candidate-source mode applies uniformly to every member in one transition-homogeneous bundle. If one member returns full state and another returns delta, they must be separate state bundles or pass through an explicit adapter that makes their representations agree.

No tensor is concatenated merely to construct, gate, merge, or project a structural bundle; these nodes erase to port/handle mappings during compilation.
