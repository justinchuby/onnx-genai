# Cyclic reactive dataflow IR — v2

## Decisions so far

1. Core targets single-clock cyclic tensor workflows; multi-clock async streaming is deferred and rejected by v1.
2. Cycles are legal only through generic bundled/scalar `state` nodes.
3. No model-semantic `state kind` exists.
4. Core enumerates only behavior that changes execution.
5. Termination is an explicit `repeat_while`, sampled after successful reaction commit.
6. State types/shapes are inferred by unifying connected typed endpoints; state never repeats a contract.
7. ONNX component signatures come from ONNX `ValueInfo`; YAML does not repeat them.
8. Binding component signatures remain explicit in YAML for static analysis.

## Type authority

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    # Inputs/outputs and dtype/rank/shape come from model.onnx.
    ports:
      roles:
        input_ids: token_ids
        attention_mask: attention_mask
        logits: logits

  runtime_policy:
    implementation: {kind: binding}
    contract:
      id: onnx-genai.token-policy
      version: '1'
    # A binding has no artifact signature, so it is explicit.
    ports:
      inputs:
        logits: {dtype: float32, rank: 2, shape: [batch, vocabulary]}
        active: {dtype: bool, rank: 1, shape: [batch]}
      outputs:
        token: {dtype: int64, rank: 1, shape: [batch]}
        done: {dtype: bool, rank: 1, shape: [batch]}
```

Type-checking rules:

- ONNX symbolic dimensions are scoped to that artifact, not globally matched by spelling.
- An edge unifies producer and consumer dtype, rank, dimensions, and batch layout.
- `state.initial`, every `state.current_to`, and `state.next` must all unify.
- A conflict reports the state/member, both endpoints, both resolved contracts, and artifact provenance.
- YAML-only validation checks syntax/references. Full graph type-check reads ONNX headers; external tensor data is not needed.

## Compact Qwen shape

```yaml
schema_version: vNext
pipeline:
  workflow:
    inputs:
      # Request boundary contracts remain explicit.
      request.input_ids: {contract: {...}, role: {...}, source: {kind: request}}
      request.max_iterations: {contract: {...}, role: {...}, source: {kind: request}}
      request.temperature: {contract: {...}, role: {...}, source: {kind: request}}
      # ...

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
          commit_with: decode_reaction

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
        # Initialization closure is derived from state.initial dependencies.
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
          inputs:
            input_eos_ids: request.eos_ids
            input_eos_lengths: request.eos_lengths
            input_max_iterations: request.row_max_iterations
            fallback_max_iterations: request.max_iterations
            active: package.active

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
            # KV ports are connected once by decoder_cache.members.

        decode_logits:
          kind: component
          component: last_token_logits
          inputs: {logits: decode_model.logits}

        token:
          kind: state
          initial: initialize.token_slot
          current_to: token_update.current
          next: token_update.next
          transition: {kind: replace}
          atomic_with: decode_reaction

        logits:
          kind: state
          initial: prefill_logits.last_logits
          current_to: sample.logits
          next: decode_logits.last_logits
          transition: {kind: replace}
          atomic_with: decode_reaction

        # active, done, generated_lengths, accepted_len, cache_lengths,
        # rng_counter, and attention_mask follow the same scalar state shape.

        decoder_cache:
          kind: state
          ownership: runtime
          scope: invocation
          release: invocation_end
          atomic_with: decode_reaction
          transition:
            # Relationship between committed current and candidate next. The state
            # node does not append tensors a second time.
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
              current_to: decode_model.past_key_values.0.key
              next: decode_model.present.0.key
            value.0:
              initial: prefill_model.present.0.value
              current_to: decode_model.past_key_values.0.value
              next: decode_model.present.0.value
            # ... explicit members through layer 23 ...

          # Optional proof used only by implementations that substitute a generic
          # state transition with an attention-state service.
          interfaces:
            onnx-genai.attention-state:
              version: '1'
              bindings:
                layers:
                  - {index: 0, key: key.0, value: value.0}
                  # ... through layer 23 ...

    serving:
      active: active.current
      done: done.current
      accepted_len: accepted_len.current
```

## State transition vocabulary

These names constrain the observable relationship between `current` and `next`:

```yaml
transition: {kind: replace}

transition:
  kind: append
  axis: 2
  increment: accepted_len.next
  bound: package.max_context

transition:
  kind: indexed_scatter
  axis: 2
  indices: write_indices.current
  logical_length: cache_lengths.current
  capacity: package.max_context
```

They do not prescribe allocation, storage layout, kernel choice, device, page size, or whether the runtime materializes `next`. A runtime may hold descriptors and commit only changed pages/lengths if the graph-visible result is equivalent.

## Removed from the published Qwen YAML

- Every ONNX component's duplicated `ports.inputs/outputs` dtype/rank/shape block.
- Every state's duplicated tensor contract.
- Authored `steps`, `setup`, `carried`, `invoke`, and `emit` syntax.
- `state_service.groups.decoder_cache.ports`, because bundle members are the canonical wiring.
- Serialized execution schedule, physical cache plan, and deployment policy.

Port roles, request/output boundary contracts, runtime binding signatures, state transitions, legal substitutions, and all artifact port connections remain explicit.
