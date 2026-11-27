# Qwen cyclic reactive IR — v8

This version applies the v7 execution model to Qwen while preserving the invariant:

> At durable invocation commit, model state covers the complete committed input and output token history.

## Two transaction levels

```text
admission baseline
  -> reaction 0 working commit
  -> reaction 1 working commit
  -> ...
  -> postlude
  -> invocation durable commit
```

- A reaction working commit atomically advances all state for one logical row and makes it visible to the next reaction.
- It does not overwrite durable session state or committed output heads.
- Invocation durable commit occurs only after clock and postlude success.
- Failure/cancellation restores the admission baseline.
- Early externally visible output must use the provisional-revision protocol and can be invalidated by abort-to-baseline.

## Clock

```yaml
graph:
  clock:
    kind: synchronous
    max_reactions: request.max_iterations
    index: reaction.index
    first: reaction.first
    repeat_while: termination.continue
    sample_repeat: after_working_commit
```

- `reaction.first` is a derived scalar bool equivalent to `reaction.index == 0`; it is not independently authored state.
- Qwen requires at least one reaction when prompt ingestion must become durable. A request with zero output budget performs no reaction and therefore does not alter session state.

## Components

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    ports:
      # dtype/rank/shape come from ONNX ValueInfo.
      roles:
        input_ids: token_ids
        attention_mask: attention_mask
        logits: logits
      batch:
        request_aligned:
          axis: 0
          # Port groups may be used; metadata does not infer request identity
          # from the symbol or artifact port spelling.
          ports: [input_ids, attention_mask, logits, past_key_values, present]

  # Remaining ONNX policy components declare artifacts and non-ONNX facts only.
  token_sampler: {implementation: {kind: onnx, artifact: policies/token_sampler.onnx}}
  termination: {implementation: {kind: onnx, artifact: policies/termination.onnx}}
  token_state_update: {implementation: {kind: onnx, artifact: policies/token_state_update.onnx}}
  last_token_logits: {implementation: {kind: onnx, artifact: policies/last_token_logits.onnx}}
  decoder_state_initializer: {implementation: {kind: onnx, artifact: policies/decoder_state_initializer.onnx}}
  decoder_step_update: {implementation: {kind: onnx, artifact: policies/decoder_step_update.onnx}}
  cache_length_update: {implementation: {kind: onnx, artifact: policies/cache_length_update.onnx}}
  termination_batch_initializer: {implementation: {kind: onnx, artifact: policies/termination_batch_initializer.onnx}}
  token_to_slot: {implementation: {kind: onnx, artifact: policies/token_to_slot.onnx}}
  generated_length_update: {implementation: {kind: onnx, artifact: policies/generated_length_update.onnx}}
```

Binding implementations—not shown in this Qwen package—retain explicit input/output tensor signatures for static analysis.

## Initialization state

```yaml
graph:
  nodes:
    initialize:
      kind: component
      component: decoder_state_initializer
      inputs:
        prompt_tokens: request.input_ids
        prompt_lengths: request.prompt_lengths
        max_iterations: request.max_iterations

    decoder_cache:
      kind: state
      ownership: runtime
      scope: session
      release: session_end
      transition:
        kind: append
        axis: 2
        candidate_increment: package.one_token
        commit: {kind: prefix, length: accepted_len.next}
        bound: package.max_context
      aliasing: permitted
      reuse: {prefix: allowed, evict_prefix: forbidden}
      capabilities: {rollback_positions: 1, snapshot: true, fork: true}
      members:
        key.0:
          initial: initialize.past_key_values.0.key
          next: decode.present.0.key
        value.0:
          initial: initialize.past_key_values.0.value
          next: decode.present.0.value
        # ... explicit members through layer 23 ...
      interfaces:
        onnx-genai.attention-state:
          version: '1'
          bindings:
            layers:
              - {index: 0, key: key.0, value: value.0}
              # ... through layer 23 ...

    logits:
      kind: state
      initial: absent
      next: decode_logits.last_logits
      transition: {kind: replace}

    token:
      kind: state
      initial: initialize.token_slot
      next: token_update.next
      transition: {kind: replace}
```

For session scope, an existing durable value replaces the declared initializer at admission. The initializer is evaluated only for missing state. Initially-absent state infers its tensor type from `next` and guarded consumers.

## Reaction 0 prefill block

```yaml
nodes:
  prefill:
    kind: component
    component: model
    when: {value: reaction.first, equals: true}
    inputs:
      input_ids: request.input_ids
      attention_mask: initialize.attention_mask
      past_key_values.0.key: decoder_cache.current.key.0
      past_key_values.0.value: decoder_cache.current.value.0
      # ... through layer 23 ...

  prefill_logits:
    kind: component
    component: last_token_logits
    when: {value: reaction.first, equals: true}
    inputs: {logits: prefill.logits}

  prefill_cache:
    kind: bundle
    when: {value: reaction.first, equals: true}
    members:
      key.0: prefill.present.0.key
      value.0: prefill.present.0.value
      # ... through layer 23 ...
```

The prefill block reads the admission/working cache. For a new session this is empty initialized state; for a continuing session it is restored durable state.

## Steady-state carried inputs

```yaml
nodes:
  carried_logits:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: logits.current

  carried_cache:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: decoder_cache.current

  effective_logits:
    kind: merge
    inputs: [prefill_logits.last_logits, carried_logits.value]
    require: exactly_one_present

  effective_cache:
    kind: merge
    inputs: [prefill_cache.value, carried_cache.value]
    require: exactly_one_present
```

Presence analysis proves:

- reaction 0: prefill values present, carried values gated absent;
- reaction 1+: prefill values absent, carried values present;
- exactly one source reaches each merge;
- `logits.current` is never read before its first successful write.

The compiler produces a one-time prefill block and a branch-free steady block. It need not test `reaction.first` after entering steady state.

## Common sample and decode block

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

  token_slot:
    kind: component
    component: token_to_slot
    inputs: {token: sample.token}

  token_update:
    kind: component
    component: token_state_update
    inputs:
      current: token.current
      update: token_slot.slot
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

  next_mask:
    kind: component
    component: decoder_step_update
    inputs:
      attention_mask: attention_mask.current
      logical_length: cache_lengths.current

  decode:
    kind: component
    component: model
    inputs:
      input_ids: token_update.next
      attention_mask: next_mask.next_attention_mask
      past_key_values.0.key: effective_cache.value.key.0
      past_key_values.0.value: effective_cache.value.value.0
      # ... through layer 23 ...

  decode_logits:
    kind: component
    component: last_token_logits
    inputs: {logits: decode.logits}
```

Reaction 0 therefore executes:

```text
prefill(prompt, baseline cache)
  -> sample first token
  -> decode first token
  -> working commit token/logits/KV/control state
```

Reaction 1+ executes:

```text
sample(carried logits)
  -> decode sampled token against carried KV
  -> working commit
```

The final durable KV includes every committed output token, including EOS when EOS is published.

## Remaining state

```yaml
nodes:
  active:
    kind: state
    initial: package.active
    next: termination.next_active
    transition: {kind: replace}

  done:
    kind: state
    initial: package.not_done
    next: termination.done
    transition: {kind: replace}

  rng_counter:
    kind: state
    initial: request.rng_counter
    next: sample.next_counter
    transition: {kind: replace}

  generated_lengths:
    kind: state
    initial: initialize.generated_lengths
    next: next_generated_length.total
    transition: {kind: replace}

  accepted_len:
    kind: state
    initial: package.zero_batch
    next: accepted_length.total
    transition: {kind: replace}

  cache_lengths:
    kind: state
    initial: initialize.cache_lengths
    next: next_cache_length.total
    transition: {kind: replace}

  attention_mask:
    kind: state
    initial: initialize.attention_mask
    next: next_mask.next_attention_mask
    transition: {kind: replace}
```

All request-aligned state for one row participates in that row's reaction working commit and invocation durable commit.

## Output

```yaml
outputs:
  tokens:
    contract:
      dtype: int64
      rank: 2
      shape: [batch_size, generated_sequence]
      batch_layout: {kind: request_aligned, axis: 0}
    role: tokens
    stage: pre_adapter
    publication:
      operations:
        - kind: append
          value: token_update.next
          when: active.current
          valid_length: emitted_length.total
```

One ordered publication batch is staged per row/reaction. Under commit-only mode it becomes durable with the invocation; under provisional-revision mode it may be observed early and is reconciled by final commit or abort-to-baseline.

## Effect contract correction

```yaml
effects:
  external_store:
    commit_behavior: transactional  # pure | transactional | after_invocation_commit
    retry: idempotent
    speculation_safety: {...}
```

- `pure`: no externally observable mutation; may feed state.next.
- `transactional`: implements prepare/working-savepoint/final-commit/abort and may participate in the invocation transaction.
- `after_invocation_commit`: consumes only durable/final values; its outputs cannot affect state.next or repeat_while, and its failure cannot roll back committed state.
- Retry and speculation safety remain independent axes.

## Acyclic packages

`graph.clock` is optional. Encoder, embedding, codec, and other pure DAG workflows execute once without a fake one-iteration clock. A workflow with recurrent feedback must declare a clock.
