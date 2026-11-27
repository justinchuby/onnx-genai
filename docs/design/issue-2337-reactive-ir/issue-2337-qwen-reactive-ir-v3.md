# Cyclic reactive dataflow IR — v3

## Locked semantic core

```text
single synchronous clock
+ typed component nodes
+ generic scalar/bundled state nodes
+ scalar firing gates and explicit merges
+ stateful-SCC atomic commit
+ explicit post-commit repeat predicate
```

- Every cycle must cross a state node; deleting state nodes leaves an acyclic instantaneous graph.
- Reaction N reads one committed snapshot. Every `state.next` is tentative until the whole stateful SCC succeeds.
- Each stateful SCC is one atomic transaction by default. An explicit transaction declaration exists only to merge otherwise independent SCCs.
- `repeat_while` is sampled after commit. False stops the region without undoing its final reaction.
- The initialization subgraph is the reverse dependency closure of all `state.initial` bindings. It runs once and may not depend on any `state.current`.
- Scalar `node.when` controls firing and yields absent outputs when false. `bool[batch]` is ordinary tensor data and never implicitly changes firing/compaction.
- Untaken gated values must meet at an explicit `merge`/`select`; YAML order never chooses a writer.

## One edge mechanism

All ordinary edges are consumer-side input bindings. State is not allowed to invent a second `current_to` edge syntax.

```yaml
nodes:
  decode_model:
    kind: component
    component: model
    inputs:
      input_ids: token_update.next
      attention_mask: next_mask.next_attention_mask
      past_key_values.0.key: decoder_cache.current.key.0
      past_key_values.0.value: decoder_cache.current.value.0

  token_update:
    kind: component
    component: token_state_update
    inputs:
      current: token.current
      update: token_slot.slot
      active: active.current
      done: done.current

  token:
    kind: state
    initial: initialize.token_slot
    next: token_update.next
    transition: {kind: replace}

  decoder_cache:
    kind: state
    ownership: runtime
    scope: invocation
    release: invocation_end
    transition:
      kind: append
      axis: 2
      increment: accepted_len.next
      bound: package.max_context
    aliasing: permitted
    reuse: {prefix: allowed, evict_prefix: forbidden}
    capabilities: {rollback_positions: 1, snapshot: true, fork: true}
    members:
      key.0:
        initial: prefill_model.present.0.key
        next: decode_model.present.0.key
      value.0:
        initial: prefill_model.present.0.value
        next: decode_model.present.0.value
      # ... explicit members through layer 23 ...
    interfaces:
      onnx-genai.attention-state:
        version: '1'
        bindings:
          layers:
            - {index: 0, key: key.0, value: value.0}
            # ... through layer 23 ...
```

Reference grammar: split `<node-id>.<port-path>` at the first `.`. Node IDs cannot contain `.`; component/state port paths are opaque after that split.

## Type authority

- Workflow request/output boundaries retain explicit contracts.
- ONNX node signatures come from ONNX `ValueInfo`; metadata keeps only port roles and non-ONNX behavioral facts.
- Binding node signatures are explicit in metadata for static analysis.
- State serializes no dtype/shape. Its `initial`, `current` consumers, and `next` endpoints must unify.
- Symbolic dimensions are artifact-local until an edge unifies them.

## Qwen graph skeleton

```yaml
schema_version: vNext
pipeline:
  workflow:
    inputs:
      request.input_ids: {contract: {...}, role: {...}, source: {kind: request}}
      request.max_iterations: {contract: {...}, role: {...}, source: {kind: request}}
      # ... remaining request/package inputs ...

    outputs:
      tokens:
        contract: {dtype: int64, rank: 2, shape: [batch, generated_sequence]}
        role: tokens
        stage: pre_adapter
        publication:
          value: token_update.next
          mode: append
          when: active.current
          valid_length: emitted_length.total
          # Publication from a stateful reaction occurs only after its SCC commits.

    components:
      model:
        implementation: {kind: onnx, artifact: model.onnx}
        ports:
          roles: {input_ids: token_ids, attention_mask: attention_mask, logits: logits}
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

    graph:
      clock:
        kind: synchronous
        index: reaction.index
        repeat_while: termination.continue
        sample_repeat: after_commit

      nodes:
        # Derived one-shot initialization closure.
        initialize:
          kind: component
          component: decoder_state_initializer
          inputs:
            prompt_tokens: request.input_ids
            prompt_lengths: request.prompt_lengths
            max_iterations: request.max_iterations

        prefill_model:
          kind: component
          component: model
          inputs:
            input_ids: request.input_ids
            attention_mask: initialize.attention_mask
            past_key_values.0.key: initialize.past_key_values.0.key
            past_key_values.0.value: initialize.past_key_values.0.value
            # ... through layer 23 ...

        prefill_logits:
          kind: component
          component: last_token_logits
          inputs: {logits: prefill_model.logits}

        termination_config:
          kind: component
          component: termination_batch_initializer
          inputs: {...}

        sample:
          kind: component
          component: token_sampler
          inputs:
            logits: logits.current
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

        decode_model:
          kind: component
          component: model
          inputs:
            input_ids: token_update.next
            attention_mask: next_mask.next_attention_mask
            past_key_values.0.key: decoder_cache.current.key.0
            past_key_values.0.value: decoder_cache.current.value.0
            # ... through layer 23 ...

        decode_logits:
          kind: component
          component: last_token_logits
          inputs: {logits: decode_model.logits}

        token:
          kind: state
          initial: initialize.token_slot
          next: token_update.next
          transition: {kind: replace}

        logits:
          kind: state
          initial: prefill_logits.last_logits
          next: decode_logits.last_logits
          transition: {kind: replace}

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

        # generated_lengths, accepted_len, cache_lengths, rng_counter, and
        # attention_mask use the same scalar replace transition.

        decoder_cache:
          # Generic bundled state exactly as defined above.
          kind: state
          # ...

    serving:
      active: active.current
      done: done.current
      accepted_len: accepted_len.current
```

## Branch replacement

```yaml
nodes:
  optional_encoder:
    kind: component
    component: encoder
    when: request.has_media      # scalar bool only
    inputs: {media: request.media}

  fallback_embedding:
    kind: component
    component: empty_embedding
    when: request.no_media

  encoder_result:
    kind: merge
    inputs:
      - optional_encoder.hidden
      - fallback_embedding.hidden
    require: exactly_one_present
```

A gated state update must merge with its prior value explicitly, or use a state-level scalar `when` whose false behavior is normatively “retain current.” Multiple writers to `state.next` are always invalid.

## Default atomicity

For Qwen, token/logits/active/done/length/RNG/mask/KV participate in one cyclic SCC, so they commit or abort together without repeating a transaction name on every state.

```yaml
# Only needed when structurally independent SCCs must commit together.
transactions:
  coupled_rollout:
    merge_sccs: [draft_cycle, target_cycle]
    isolation: snapshot
    abort: retain_previous
```

The listed IDs refer to compiler-stable named cyclic regions, not duplicated state-member lists.

## Optional optimization interfaces

An interface contract may label canonical members to prove that an implementation substitution is legal. It cannot add ports, edges, state, or execution order. Generic execution must remain correct without using it; a runtime that selects an optimization requiring the interface must understand its exact ID/version or fail closed.

## Core transition vocabulary

```yaml
transition: {kind: replace}
transition: {kind: append, axis: 2, increment: accepted_len.next, bound: package.max_context}
transition: {kind: indexed_scatter, axis: 2, indices: cursor.current,
             logical_length: length.current, capacity: package.max_context}
```

These constrain the graph-visible `current -> next` relationship. They do not prescribe allocation, paging, device placement, or kernel selection.
