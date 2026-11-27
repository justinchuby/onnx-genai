# Qwen cyclic reactive IR — v9

v9 resolves the concrete reaction-0 cache/session gaps identified after the issue checkpoint.

## Accepted corrections

- `max_reactions = 0` is a successful no-op: no prompt ingestion, session mutation, or publication.
- Durable Qwen session state contains KV plus logical cache lengths.
- Attention mask is reconstructed explicitly from durable lengths plus the new prompt; models with non-reconstructible masks persist them as state instead.
- Reaction 0 remains `prefill -> sample -> decode`, so durable KV covers all committed input/output tokens.
- A graph component produces a dynamic logical commit plan. This choice remains provisional but is the working design.
- State exposes readable `current` and `final` endpoints only. `next:` is a consumer-side input binding and creates no `state.next` alias.

## State endpoint rule

```yaml
accepted_len:
  kind: state
  initial: package.zero_batch
  next: accepted_length.total
  transition: {kind: replace}

cache_plan:
  kind: component
  component: state_commit_plan
  inputs:
    accepted_length: accepted_length.total  # producer, not accepted_len.next
```

Readable state endpoints are:

```text
accepted_len.current
accepted_len.final
```

## Durable session state

```yaml
nodes:
  decoder_cache:
    kind: state
    ownership: runtime
    scope: session
    release: session_end
    members:
      key.0:
        initial: empty_state.past_key_values.0.key
        next: decode.present.0.key
      value.0:
        initial: empty_state.past_key_values.0.value
        next: decode.present.0.value
      # ... through layer 23 ...

  cache_lengths:
    kind: state
    ownership: runtime
    scope: session
    release: session_end
    initial: package.zero_batch
    next: next_cache_length.total
    transition: {kind: replace}

  attention_mask:
    kind: state
    scope: invocation
    initial: absent
    next: next_mask.next_attention_mask
    transition: {kind: replace}
```

At admission, an existing session supplies `decoder_cache.current` and `cache_lengths.current`; otherwise their initializers supply empty values. Attention mask is invocation-local and initially absent.

## First-reaction context reconstruction

`prefill_context` is an ordinary typed component. It may be shipped as ONNX or implemented by a registered binding with an explicit signature.

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
```

It produces:

```text
prefill_attention_mask    mask for prior durable prefix + padded new prompt
body_attention_mask       mask used to decode the sampled token
prompt_candidate_count    number of physical prompt candidate positions
prompt_commit_indices     valid prompt positions within that candidate suffix
prompt_commit_length      number of logical prompt positions per row
prefill_logical_lengths   prior_lengths + prompt_commit_length
```

These are graph values with ONNX/binding-declared types. No cache page IDs, scheduler slots, request IDs, or physical destinations appear.

## Reaction-0 prefill and steady merge

```yaml
nodes:
  prefill:
    kind: component
    component: model
    when: {value: reaction.first, equals: true}
    inputs:
      input_ids: request.input_ids
      attention_mask: prefill_context.prefill_attention_mask
      past_key_values.0.key: decoder_cache.current.key.0
      past_key_values.0.value: decoder_cache.current.value.0
      # ... through layer 23 ...

  prefill_logits:
    kind: component
    component: last_token_logits
    when: {value: reaction.first, equals: true}
    inputs: {logits: prefill.logits}

  carried_logits:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: logits.current

  effective_logits:
    kind: merge
    inputs: [prefill_logits.last_logits, carried_logits.value]
    require: exactly_one_present

  carried_cache:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: decoder_cache.current

  prefill_cache:
    kind: bundle
    when: {value: reaction.first, equals: true}
    members:
      key.0: prefill.present.0.key
      value.0: prefill.present.0.value
      # ...

  effective_cache:
    kind: merge
    inputs: [prefill_cache.value, carried_cache.value]
    require: exactly_one_present
```

## Effective logical context

```yaml
nodes:
  carried_lengths:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: cache_lengths.current

  effective_lengths:
    kind: merge
    inputs: [prefill_context.prefill_logical_lengths, carried_lengths.value]
    require: exactly_one_present

  carried_mask:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: attention_mask.current

  effective_mask:
    kind: merge
    inputs: [prefill_context.body_attention_mask, carried_mask.value]
    require: exactly_one_present

  next_mask:
    kind: component
    component: decoder_step_update
    inputs:
      attention_mask: effective_mask.value
      logical_length: effective_lengths.value
```

Presence analysis proves that initially absent `attention_mask.current` is not read on reaction 0.

## Per-reaction accepted and cache lengths

The previously abbreviated policy nodes are explicit:

```yaml
nodes:
  next_generated_length:
    kind: component
    component: generated_length_update
    inputs:
      left: generated_lengths.current
      right: package.one_token
      active: active.current
      done: done.current

  emitted_length:
    kind: component
    component: generated_length_update
    inputs:
      left: package.zero_batch
      right: package.one_token
      active: active.current
      done: done.current

  accepted_length:
    kind: component
    component: cache_length_update
    inputs:
      left: package.zero_batch
      right: package.one_token
      active: active.current
      done: done.current

  next_cache_length:
    kind: component
    component: cache_length_update
    inputs:
      left: effective_lengths.value
      right: accepted_length.total
      active: package.active
      done: package.not_done
```

`accepted_length.total` and `emitted_length.total` are per-reaction 0/1 row counts. `next_cache_length.total` is cumulative logical length after prompt and the accepted sampled token.

The final `active/done` constants on `next_cache_length` indicate that the already-filtered accepted count is authoritative; alternatively this addition may be a smaller pure integer-add component. It must not reapply activity filtering and accidentally drop prompt length.

## Dynamic cache commit plan

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

On reaction 0 it produces a logical selection equivalent to:

```text
candidate_count = padded_prompt_candidate_count + 1
indices         = valid_prompt_candidate_indices ++ [sampled_token_candidate_index]
commit_length   = prompt_valid_length + accepted_length
```

On reaction 1+:

```text
candidate_count = 1
indices         = [0] when accepted, otherwise empty
commit_length   = accepted_length
```

The state transition is one mechanism for both paths:

```yaml
  decoder_cache:
    kind: state
    # members omitted here
    transition:
      kind: append
      axis: 2
      candidate_increment: cache_plan.candidate_count
      commit:
        kind: gather
        indices: cache_plan.indices
        length: cache_plan.commit_length
      bound: package.max_context
```

Commit indices are relative to the candidate suffix of `decode.present`, ordered exactly as positions should appear in logical state. They are not absolute cache offsets or physical storage addresses.

Runtime validation requires, per row:

```text
0 <= commit_length <= candidate_count
indices length >= commit_length
selected indices are in [0, candidate_count)
resulting logical length <= bound
```

The selected index order defines logical append order. Physical paged lowering may reserve all candidates and retain references only for selected positions.

## Sample, termination, and decode

```yaml
nodes:
  sample:
    kind: component
    component: token_sampler
    inputs:
      logits: effective_logits.value
      temperature: request.temperature
      top_k: request.top_k
      top_p: request.top_p
      min_p: request.min_p
      seed: request.seed
      counter: rng_counter.current
      active: active.current
      done: done.current

  termination:
    kind: component
    component: termination
    inputs:
      tokens: sample.token
      eos_ids: termination_config.row_eos_ids
      eos_lengths: termination_config.eos_lengths
      iteration: reaction.index
      max_iterations: termination_config.max_iterations
      active: active.current

  decode:
    kind: component
    component: model
    inputs:
      input_ids: token_update.next
      attention_mask: next_mask.next_attention_mask
      past_key_values.0.key: effective_cache.value.key.0
      past_key_values.0.value: effective_cache.value.value.0
      # ... through layer 23 ...
```

Sampling/output use `active.current`; termination produces the next active/done state. Therefore a sampled EOS is published and decoded into KV before that row becomes inactive.

## Output

```yaml
outputs:
  tokens:
    publication:
      operations:
        - kind: append
          value: token_update.next
          when: active.current
          valid_length: emitted_length.total
```

One ordered typed publication batch is staged per row/reaction. Reaction working commits make state available internally; invocation durable commit persists final session state and output heads.

## No-op zero budget

```text
max_reactions = 0
=> no reaction 0
=> no prefill
=> no session mutation
=> no publication
=> successful no-op
```

An API may reject zero output budget separately, but the workflow semantics are unambiguous.

## Remaining concern

`state_commit_plan` is currently a binding/component-produced logical plan. Before finalizing the schema, compare it against a transition-local first/steady form for authoring complexity and static optimization. The canonical form should remain one or the other, never both.
