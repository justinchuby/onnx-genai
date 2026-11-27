# Qwen cyclic reactive IR - v22 graph-complete candidate

This document applies the v19-v21 design to the published 24-layer Qwen 2.5
package. It is a graph-complete candidate: all KV ports and bundle members are
spelled out, and no historical clock, gate, boolean `when`, or authored
continuous-batching block remains.

It is still design material, not accepted metadata.

## Behavioral summary

```text
reaction 0:
  reconstruct prompt context from durable logical lengths
  -> prefill prompt against durable KV
  -> gather logits at each row's last valid prompt token
  -> sample
  -> decide accepted token / next activity
  -> decode sampled token
  -> commit valid prompt KV plus accepted sampled-token KV

reaction 1+:
  carried logits + compact committed KV
  -> sample
  -> decide accepted token / next activity
  -> decode sampled token
  -> commit accepted sampled-token KV

completion:
  expose final state -> durable invocation commit
```

The sampled EOS token is published and decoded before its row becomes
inactive, so durable KV covers every committed input and output token.

## Candidate YAML

```yaml
pipeline:
  workflow:
    manifest:
      id: qwen2_5-text-generation
      version: '1'

    inputs:
      request.input_ids:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, prompt_sequence]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true
        constraints:
          valid_sequence_length: {min: 1}

      request.prompt_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.max_iterations:
        contract:
          dtype: int64
          rank: 0
          shape: []
          batch_layout: {kind: shared}
        source: {kind: request}
        required: true

      request.max_generated_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.temperature:
        contract:
          dtype: float32
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.top_k:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.top_p:
        contract:
          dtype: float32
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.min_p:
        contract:
          dtype: float32
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.seed:
        contract:
          dtype: uint64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.rng_counter:
        contract:
          dtype: uint64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      request.eos_ids:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, max_eos]
          batch_layout: {kind: request_aligned, axis: 0}
          padding:
            - {dimension: max_eos, valid_lengths: request.eos_lengths}
        source: {kind: request}
        required: true

      request.eos_lengths:
        contract:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: {kind: request_aligned, axis: 0}
        source: {kind: request}
        required: true

      package.max_context:
        contract:
          dtype: int64
          rank: 0
          shape: []
          batch_layout: {kind: shared}
        source: {kind: artifact, value: max_context}
        required: true

    outputs:
      tokens:
        contract:
          dtype: int64
          rank: 2
          shape: [batch, generated_sequence]
          batch_layout: {kind: request_aligned, axis: 0}
        role: tokens
        family: {kind: materialized}
        publication:
          operations:
            - kind: append
              value: token_slot.slot
              valid_length: termination.emitted_length

    components:
      model:
        implementation: {kind: onnx, artifact: model.onnx}
        contract:
          id: onnx-genai.autoregressive-decode
          version: '1'
          bindings:
            input_ids: input_ids
            attention_mask: attention_mask
            position_ids: position_ids
            logits: logits
            layers:
              - {index: 0, past_key: past_key_values.0.key, past_value: past_key_values.0.value, present_key: present.0.key, present_value: present.0.value}
              - {index: 1, past_key: past_key_values.1.key, past_value: past_key_values.1.value, present_key: present.1.key, present_value: present.1.value}
              - {index: 2, past_key: past_key_values.2.key, past_value: past_key_values.2.value, present_key: present.2.key, present_value: present.2.value}
              - {index: 3, past_key: past_key_values.3.key, past_value: past_key_values.3.value, present_key: present.3.key, present_value: present.3.value}
              - {index: 4, past_key: past_key_values.4.key, past_value: past_key_values.4.value, present_key: present.4.key, present_value: present.4.value}
              - {index: 5, past_key: past_key_values.5.key, past_value: past_key_values.5.value, present_key: present.5.key, present_value: present.5.value}
              - {index: 6, past_key: past_key_values.6.key, past_value: past_key_values.6.value, present_key: present.6.key, present_value: present.6.value}
              - {index: 7, past_key: past_key_values.7.key, past_value: past_key_values.7.value, present_key: present.7.key, present_value: present.7.value}
              - {index: 8, past_key: past_key_values.8.key, past_value: past_key_values.8.value, present_key: present.8.key, present_value: present.8.value}
              - {index: 9, past_key: past_key_values.9.key, past_value: past_key_values.9.value, present_key: present.9.key, present_value: present.9.value}
              - {index: 10, past_key: past_key_values.10.key, past_value: past_key_values.10.value, present_key: present.10.key, present_value: present.10.value}
              - {index: 11, past_key: past_key_values.11.key, past_value: past_key_values.11.value, present_key: present.11.key, present_value: present.11.value}
              - {index: 12, past_key: past_key_values.12.key, past_value: past_key_values.12.value, present_key: present.12.key, present_value: present.12.value}
              - {index: 13, past_key: past_key_values.13.key, past_value: past_key_values.13.value, present_key: present.13.key, present_value: present.13.value}
              - {index: 14, past_key: past_key_values.14.key, past_value: past_key_values.14.value, present_key: present.14.key, present_value: present.14.value}
              - {index: 15, past_key: past_key_values.15.key, past_value: past_key_values.15.value, present_key: present.15.key, present_value: present.15.value}
              - {index: 16, past_key: past_key_values.16.key, past_value: past_key_values.16.value, present_key: present.16.key, present_value: present.16.value}
              - {index: 17, past_key: past_key_values.17.key, past_value: past_key_values.17.value, present_key: present.17.key, present_value: present.17.value}
              - {index: 18, past_key: past_key_values.18.key, past_value: past_key_values.18.value, present_key: present.18.key, present_value: present.18.value}
              - {index: 19, past_key: past_key_values.19.key, past_value: past_key_values.19.value, present_key: present.19.key, present_value: present.19.value}
              - {index: 20, past_key: past_key_values.20.key, past_value: past_key_values.20.value, present_key: present.20.key, present_value: present.20.value}
              - {index: 21, past_key: past_key_values.21.key, past_value: past_key_values.21.value, present_key: present.21.key, present_value: present.21.value}
              - {index: 22, past_key: past_key_values.22.key, past_value: past_key_values.22.value, present_key: present.22.key, present_value: present.22.value}
              - {index: 23, past_key: past_key_values.23.key, past_value: past_key_values.23.value, present_key: present.23.key, present_value: present.23.value}
        port_layouts:
          input_ids: &request_rows
            batch_layout: {kind: request_aligned, axis: 0}
          attention_mask: *request_rows
          position_ids: *request_rows
          logits: *request_rows
          past_key_values.0.key: *request_rows
          past_key_values.0.value: *request_rows
          past_key_values.1.key: *request_rows
          past_key_values.1.value: *request_rows
          past_key_values.2.key: *request_rows
          past_key_values.2.value: *request_rows
          past_key_values.3.key: *request_rows
          past_key_values.3.value: *request_rows
          past_key_values.4.key: *request_rows
          past_key_values.4.value: *request_rows
          past_key_values.5.key: *request_rows
          past_key_values.5.value: *request_rows
          past_key_values.6.key: *request_rows
          past_key_values.6.value: *request_rows
          past_key_values.7.key: *request_rows
          past_key_values.7.value: *request_rows
          past_key_values.8.key: *request_rows
          past_key_values.8.value: *request_rows
          past_key_values.9.key: *request_rows
          past_key_values.9.value: *request_rows
          past_key_values.10.key: *request_rows
          past_key_values.10.value: *request_rows
          past_key_values.11.key: *request_rows
          past_key_values.11.value: *request_rows
          past_key_values.12.key: *request_rows
          past_key_values.12.value: *request_rows
          past_key_values.13.key: *request_rows
          past_key_values.13.value: *request_rows
          past_key_values.14.key: *request_rows
          past_key_values.14.value: *request_rows
          past_key_values.15.key: *request_rows
          past_key_values.15.value: *request_rows
          past_key_values.16.key: *request_rows
          past_key_values.16.value: *request_rows
          past_key_values.17.key: *request_rows
          past_key_values.17.value: *request_rows
          past_key_values.18.key: *request_rows
          past_key_values.18.value: *request_rows
          past_key_values.19.key: *request_rows
          past_key_values.19.value: *request_rows
          past_key_values.20.key: *request_rows
          past_key_values.20.value: *request_rows
          past_key_values.21.key: *request_rows
          past_key_values.21.value: *request_rows
          past_key_values.22.key: *request_rows
          past_key_values.22.value: *request_rows
          past_key_values.23.key: *request_rows
          past_key_values.23.value: *request_rows
          present.0.key: *request_rows
          present.0.value: *request_rows
          present.1.key: *request_rows
          present.1.value: *request_rows
          present.2.key: *request_rows
          present.2.value: *request_rows
          present.3.key: *request_rows
          present.3.value: *request_rows
          present.4.key: *request_rows
          present.4.value: *request_rows
          present.5.key: *request_rows
          present.5.value: *request_rows
          present.6.key: *request_rows
          present.6.value: *request_rows
          present.7.key: *request_rows
          present.7.value: *request_rows
          present.8.key: *request_rows
          present.8.value: *request_rows
          present.9.key: *request_rows
          present.9.value: *request_rows
          present.10.key: *request_rows
          present.10.value: *request_rows
          present.11.key: *request_rows
          present.11.value: *request_rows
          present.12.key: *request_rows
          present.12.value: *request_rows
          present.13.key: *request_rows
          present.13.value: *request_rows
          present.14.key: *request_rows
          present.14.value: *request_rows
          present.15.key: *request_rows
          present.15.value: *request_rows
          present.16.key: *request_rows
          present.16.value: *request_rows
          present.17.key: *request_rows
          present.17.value: *request_rows
          present.18.key: *request_rows
          present.18.value: *request_rows
          present.19.key: *request_rows
          present.19.value: *request_rows
          present.20.key: *request_rows
          present.20.value: *request_rows
          present.21.key: *request_rows
          present.21.value: *request_rows
          present.22.key: *request_rows
          present.22.value: *request_rows
          present.23.key: *request_rows
          present.23.value: *request_rows
        batch_capacity: {}

      decoder_state_initializer:
        implementation:
          kind: onnx
          artifact: policies/decoder_state_initializer.onnx

      admission_policy:
        implementation:
          kind: onnx
          artifact: policies/admission_policy.onnx

      prefill_context_plan:
        implementation:
          kind: onnx
          artifact: policies/prefill_context_plan.onnx

      gather_sequence_logits:
        implementation:
          kind: onnx
          artifact: policies/gather_sequence_logits.onnx

      steady_decode_context:
        implementation:
          kind: onnx
          artifact: policies/steady_decode_context.onnx

      token_sampler:
        implementation:
          kind: onnx
          artifact: policies/token_sampler.onnx

      termination_policy:
        implementation:
          kind: onnx
          artifact: policies/termination_policy.onnx

      token_to_slot:
        implementation:
          kind: onnx
          artifact: policies/token_to_slot.onnx

      length_add:
        implementation:
          kind: onnx
          artifact: policies/length_add.onnx

      squeeze_decode_logits:
        implementation:
          kind: onnx
          artifact: policies/squeeze_decode_logits.onnx

      cache_commit_policy:
        implementation: {kind: binding}
        contract:
          id: onnx-genai.state-commit-policy
          version: '1'
        signature:
          inputs:
            prompt_candidate_count:
              required: false
              dtype: int64
              rank: 0
              shape: []
              batch_layout: {kind: shared}
            prompt_indices:
              required: false
              dtype: int64
              rank: 2
              shape: [batch, prompt_sequence]
              batch_layout: {kind: request_aligned, axis: 0}
            prompt_length:
              required: false
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
            accepted_length:
              required: true
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}
          outputs:
            candidate_count:
              dtype: int64
              rank: 0
              shape: []
              batch_layout: {kind: shared}
            indices:
              dtype: int64
              rank: 2
              shape: [batch, candidate_sequence]
              batch_layout: {kind: request_aligned, axis: 0}
            commit_length:
              dtype: int64
              rank: 1
              shape: [batch]
              batch_layout: {kind: request_aligned, axis: 0}

    effects: {}

    graph:
      nodes:
        initialize:
          kind: component
          component: decoder_state_initializer
          inputs:
            input_ids: request.input_ids
            prompt_lengths: request.prompt_lengths

        initial_cache:
          kind: bundle
          members:
            key.0: initialize.past_key_values.0.key
            value.0: initialize.past_key_values.0.value
            key.1: initialize.past_key_values.1.key
            value.1: initialize.past_key_values.1.value
            key.2: initialize.past_key_values.2.key
            value.2: initialize.past_key_values.2.value
            key.3: initialize.past_key_values.3.key
            value.3: initialize.past_key_values.3.value
            key.4: initialize.past_key_values.4.key
            value.4: initialize.past_key_values.4.value
            key.5: initialize.past_key_values.5.key
            value.5: initialize.past_key_values.5.value
            key.6: initialize.past_key_values.6.key
            value.6: initialize.past_key_values.6.value
            key.7: initialize.past_key_values.7.key
            value.7: initialize.past_key_values.7.value
            key.8: initialize.past_key_values.8.key
            value.8: initialize.past_key_values.8.value
            key.9: initialize.past_key_values.9.key
            value.9: initialize.past_key_values.9.value
            key.10: initialize.past_key_values.10.key
            value.10: initialize.past_key_values.10.value
            key.11: initialize.past_key_values.11.key
            value.11: initialize.past_key_values.11.value
            key.12: initialize.past_key_values.12.key
            value.12: initialize.past_key_values.12.value
            key.13: initialize.past_key_values.13.key
            value.13: initialize.past_key_values.13.value
            key.14: initialize.past_key_values.14.key
            value.14: initialize.past_key_values.14.value
            key.15: initialize.past_key_values.15.key
            value.15: initialize.past_key_values.15.value
            key.16: initialize.past_key_values.16.key
            value.16: initialize.past_key_values.16.value
            key.17: initialize.past_key_values.17.key
            value.17: initialize.past_key_values.17.value
            key.18: initialize.past_key_values.18.key
            value.18: initialize.past_key_values.18.value
            key.19: initialize.past_key_values.19.key
            value.19: initialize.past_key_values.19.value
            key.20: initialize.past_key_values.20.key
            value.20: initialize.past_key_values.20.value
            key.21: initialize.past_key_values.21.key
            value.21: initialize.past_key_values.21.value
            key.22: initialize.past_key_values.22.key
            value.22: initialize.past_key_values.22.value
            key.23: initialize.past_key_values.23.key
            value.23: initialize.past_key_values.23.value

        decoder_cache:
          kind: state
          initial: {kind: value, value: initial_cache.value}
          next: decode_cache.value
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
          storage_ownership: runtime
          scope: session
          release: session_end
          aliasing: permitted
          reuse: {prefix: allowed, evict_prefix: forbidden}
          capabilities:
            rollback_positions: 1
            snapshot: true
            fork: true
          interfaces:
            onnx-genai.attention-state:
              version: '1'
              bindings:
                layers:
                  - {index: 0, key: key.0, value: value.0}
                  - {index: 1, key: key.1, value: value.1}
                  - {index: 2, key: key.2, value: value.2}
                  - {index: 3, key: key.3, value: value.3}
                  - {index: 4, key: key.4, value: value.4}
                  - {index: 5, key: key.5, value: value.5}
                  - {index: 6, key: key.6, value: value.6}
                  - {index: 7, key: key.7, value: value.7}
                  - {index: 8, key: key.8, value: value.8}
                  - {index: 9, key: key.9, value: value.9}
                  - {index: 10, key: key.10, value: value.10}
                  - {index: 11, key: key.11, value: value.11}
                  - {index: 12, key: key.12, value: value.12}
                  - {index: 13, key: key.13, value: value.13}
                  - {index: 14, key: key.14, value: value.14}
                  - {index: 15, key: key.15, value: value.15}
                  - {index: 16, key: key.16, value: value.16}
                  - {index: 17, key: key.17, value: value.17}
                  - {index: 18, key: key.18, value: value.18}
                  - {index: 19, key: key.19, value: value.19}
                  - {index: 20, key: key.20, value: value.20}
                  - {index: 21, key: key.21, value: value.21}
                  - {index: 22, key: key.22, value: value.22}
                  - {index: 23, key: key.23, value: value.23}

        cache_lengths:
          kind: state
          initial: {kind: value, value: initialize.cache_lengths}
          next: next_cache_lengths.total
          transition: {kind: replace}
          storage_ownership: runtime
          scope: session
          release: session_end

        policy_state:
          kind: state
          initial: {kind: absent}
          next: next_policy_state.value
          transition: {kind: replace}
          storage_ownership: workflow
          scope: invocation

        logits:
          kind: state
          initial: {kind: absent}
          next: decode_logits.selected
          transition: {kind: replace}
          storage_ownership: workflow
          scope: invocation

        generation:
          kind: reactor
          limit: request.max_iterations
          continue: termination.continue

        first_policy:
          kind: component
          component: admission_policy
          when: generation.first
          inputs:
            current_lengths: cache_lengths.current
            prompt_lengths: request.prompt_lengths
            max_generated_lengths: request.max_generated_lengths
            max_context: package.max_context

        first_policy_state:
          kind: bundle
          members:
            active: first_policy.next_active
            done: first_policy.done
            generated_lengths: first_policy.next_generated_lengths
            rng_counter: request.rng_counter

        effective_policy_state:
          kind: merge
          arms:
            - {when: generation.first, value: first_policy_state.value}
            - {when: generation.steady, value: policy_state.current}

        prefill_context:
          kind: component
          component: prefill_context_plan
          when: generation.first
          inputs:
            input_ids: request.input_ids
            prompt_lengths: request.prompt_lengths
            prior_lengths: cache_lengths.current
            max_context: package.max_context

        prefill:
          kind: component
          component: model
          when: generation.first
          inputs:
            input_ids: request.input_ids
            attention_mask: prefill_context.prefill_attention_mask
            position_ids: prefill_context.prefill_position_ids
            past_key_values.0.key: decoder_cache.current.key.0
            past_key_values.0.value: decoder_cache.current.value.0
            past_key_values.1.key: decoder_cache.current.key.1
            past_key_values.1.value: decoder_cache.current.value.1
            past_key_values.2.key: decoder_cache.current.key.2
            past_key_values.2.value: decoder_cache.current.value.2
            past_key_values.3.key: decoder_cache.current.key.3
            past_key_values.3.value: decoder_cache.current.value.3
            past_key_values.4.key: decoder_cache.current.key.4
            past_key_values.4.value: decoder_cache.current.value.4
            past_key_values.5.key: decoder_cache.current.key.5
            past_key_values.5.value: decoder_cache.current.value.5
            past_key_values.6.key: decoder_cache.current.key.6
            past_key_values.6.value: decoder_cache.current.value.6
            past_key_values.7.key: decoder_cache.current.key.7
            past_key_values.7.value: decoder_cache.current.value.7
            past_key_values.8.key: decoder_cache.current.key.8
            past_key_values.8.value: decoder_cache.current.value.8
            past_key_values.9.key: decoder_cache.current.key.9
            past_key_values.9.value: decoder_cache.current.value.9
            past_key_values.10.key: decoder_cache.current.key.10
            past_key_values.10.value: decoder_cache.current.value.10
            past_key_values.11.key: decoder_cache.current.key.11
            past_key_values.11.value: decoder_cache.current.value.11
            past_key_values.12.key: decoder_cache.current.key.12
            past_key_values.12.value: decoder_cache.current.value.12
            past_key_values.13.key: decoder_cache.current.key.13
            past_key_values.13.value: decoder_cache.current.value.13
            past_key_values.14.key: decoder_cache.current.key.14
            past_key_values.14.value: decoder_cache.current.value.14
            past_key_values.15.key: decoder_cache.current.key.15
            past_key_values.15.value: decoder_cache.current.value.15
            past_key_values.16.key: decoder_cache.current.key.16
            past_key_values.16.value: decoder_cache.current.value.16
            past_key_values.17.key: decoder_cache.current.key.17
            past_key_values.17.value: decoder_cache.current.value.17
            past_key_values.18.key: decoder_cache.current.key.18
            past_key_values.18.value: decoder_cache.current.value.18
            past_key_values.19.key: decoder_cache.current.key.19
            past_key_values.19.value: decoder_cache.current.value.19
            past_key_values.20.key: decoder_cache.current.key.20
            past_key_values.20.value: decoder_cache.current.value.20
            past_key_values.21.key: decoder_cache.current.key.21
            past_key_values.21.value: decoder_cache.current.value.21
            past_key_values.22.key: decoder_cache.current.key.22
            past_key_values.22.value: decoder_cache.current.value.22
            past_key_values.23.key: decoder_cache.current.key.23
            past_key_values.23.value: decoder_cache.current.value.23

        prefill_logits:
          kind: component
          component: gather_sequence_logits
          inputs:
            logits: prefill.logits
            indices: prefill_context.last_valid_prompt_index

        prefill_cache:
          kind: bundle
          members:
            key.0: prefill.present.0.key
            value.0: prefill.present.0.value
            key.1: prefill.present.1.key
            value.1: prefill.present.1.value
            key.2: prefill.present.2.key
            value.2: prefill.present.2.value
            key.3: prefill.present.3.key
            value.3: prefill.present.3.value
            key.4: prefill.present.4.key
            value.4: prefill.present.4.value
            key.5: prefill.present.5.key
            value.5: prefill.present.5.value
            key.6: prefill.present.6.key
            value.6: prefill.present.6.value
            key.7: prefill.present.7.key
            value.7: prefill.present.7.value
            key.8: prefill.present.8.key
            value.8: prefill.present.8.value
            key.9: prefill.present.9.key
            value.9: prefill.present.9.value
            key.10: prefill.present.10.key
            value.10: prefill.present.10.value
            key.11: prefill.present.11.key
            value.11: prefill.present.11.value
            key.12: prefill.present.12.key
            value.12: prefill.present.12.value
            key.13: prefill.present.13.key
            value.13: prefill.present.13.value
            key.14: prefill.present.14.key
            value.14: prefill.present.14.value
            key.15: prefill.present.15.key
            value.15: prefill.present.15.value
            key.16: prefill.present.16.key
            value.16: prefill.present.16.value
            key.17: prefill.present.17.key
            value.17: prefill.present.17.value
            key.18: prefill.present.18.key
            value.18: prefill.present.18.value
            key.19: prefill.present.19.key
            value.19: prefill.present.19.value
            key.20: prefill.present.20.key
            value.20: prefill.present.20.value
            key.21: prefill.present.21.key
            value.21: prefill.present.21.value
            key.22: prefill.present.22.key
            value.22: prefill.present.22.value
            key.23: prefill.present.23.key
            value.23: prefill.present.23.value

        effective_logits:
          kind: merge
          arms:
            - {when: generation.first, value: prefill_logits.selected}
            - {when: generation.steady, value: logits.current}

        effective_cache:
          kind: merge
          arms:
            - {when: generation.first, value: prefill_cache.value}
            - {when: generation.steady, value: decoder_cache.current}

        effective_lengths:
          kind: merge
          arms:
            - {when: generation.first, value: prefill_context.prefill_logical_lengths}
            - {when: generation.steady, value: cache_lengths.current}

        steady_context:
          kind: component
          component: steady_decode_context
          when: generation.steady
          inputs:
            logical_lengths: cache_lengths.current

        decode_attention_mask:
          kind: merge
          arms:
            - {when: generation.first, value: prefill_context.decode_attention_mask}
            - {when: generation.steady, value: steady_context.decode_attention_mask}

        decode_position_ids:
          kind: merge
          arms:
            - {when: generation.first, value: prefill_context.decode_position_ids}
            - {when: generation.steady, value: steady_context.decode_position_ids}

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
            counter: effective_policy_state.value.rng_counter
            active: effective_policy_state.value.active
            done: effective_policy_state.value.done

        termination:
          kind: component
          component: termination_policy
          inputs:
            tokens: sample.token
            eos_ids: request.eos_ids
            eos_lengths: request.eos_lengths
            current_lengths: effective_lengths.value
            generated_lengths: effective_policy_state.value.generated_lengths
            max_generated_lengths: request.max_generated_lengths
            max_context: package.max_context
            iteration: generation.index
            active: effective_policy_state.value.active
            done: effective_policy_state.value.done

        token_slot:
          kind: component
          component: token_to_slot
          inputs:
            token: sample.token

        next_policy_state:
          kind: bundle
          members:
            active: termination.next_active
            done: termination.done
            generated_lengths: termination.next_generated_lengths
            rng_counter: sample.next_counter

        next_cache_lengths:
          kind: component
          component: length_add
          inputs:
            left: effective_lengths.value
            right: termination.accepted_length

        cache_plan:
          kind: component
          component: cache_commit_policy
          inputs:
            prompt_candidate_count: prefill_context.prompt_candidate_count
            prompt_indices: prefill_context.prompt_commit_indices
            prompt_length: prefill_context.prompt_commit_length
            accepted_length: termination.accepted_length

        decode:
          kind: component
          component: model
          inputs:
            input_ids: token_slot.slot
            attention_mask: decode_attention_mask.value
            position_ids: decode_position_ids.value
            past_key_values.0.key: effective_cache.value.key.0
            past_key_values.0.value: effective_cache.value.value.0
            past_key_values.1.key: effective_cache.value.key.1
            past_key_values.1.value: effective_cache.value.value.1
            past_key_values.2.key: effective_cache.value.key.2
            past_key_values.2.value: effective_cache.value.value.2
            past_key_values.3.key: effective_cache.value.key.3
            past_key_values.3.value: effective_cache.value.value.3
            past_key_values.4.key: effective_cache.value.key.4
            past_key_values.4.value: effective_cache.value.value.4
            past_key_values.5.key: effective_cache.value.key.5
            past_key_values.5.value: effective_cache.value.value.5
            past_key_values.6.key: effective_cache.value.key.6
            past_key_values.6.value: effective_cache.value.value.6
            past_key_values.7.key: effective_cache.value.key.7
            past_key_values.7.value: effective_cache.value.value.7
            past_key_values.8.key: effective_cache.value.key.8
            past_key_values.8.value: effective_cache.value.value.8
            past_key_values.9.key: effective_cache.value.key.9
            past_key_values.9.value: effective_cache.value.value.9
            past_key_values.10.key: effective_cache.value.key.10
            past_key_values.10.value: effective_cache.value.value.10
            past_key_values.11.key: effective_cache.value.key.11
            past_key_values.11.value: effective_cache.value.value.11
            past_key_values.12.key: effective_cache.value.key.12
            past_key_values.12.value: effective_cache.value.value.12
            past_key_values.13.key: effective_cache.value.key.13
            past_key_values.13.value: effective_cache.value.value.13
            past_key_values.14.key: effective_cache.value.key.14
            past_key_values.14.value: effective_cache.value.value.14
            past_key_values.15.key: effective_cache.value.key.15
            past_key_values.15.value: effective_cache.value.value.15
            past_key_values.16.key: effective_cache.value.key.16
            past_key_values.16.value: effective_cache.value.value.16
            past_key_values.17.key: effective_cache.value.key.17
            past_key_values.17.value: effective_cache.value.value.17
            past_key_values.18.key: effective_cache.value.key.18
            past_key_values.18.value: effective_cache.value.value.18
            past_key_values.19.key: effective_cache.value.key.19
            past_key_values.19.value: effective_cache.value.value.19
            past_key_values.20.key: effective_cache.value.key.20
            past_key_values.20.value: effective_cache.value.value.20
            past_key_values.21.key: effective_cache.value.key.21
            past_key_values.21.value: effective_cache.value.value.21
            past_key_values.22.key: effective_cache.value.key.22
            past_key_values.22.value: effective_cache.value.value.22
            past_key_values.23.key: effective_cache.value.key.23
            past_key_values.23.value: effective_cache.value.value.23

        decode_logits:
          kind: component
          component: squeeze_decode_logits
          inputs:
            logits: decode.logits

        decode_cache:
          kind: bundle
          members:
            key.0: decode.present.0.key
            value.0: decode.present.0.value
            key.1: decode.present.1.key
            value.1: decode.present.1.value
            key.2: decode.present.2.key
            value.2: decode.present.2.value
            key.3: decode.present.3.key
            value.3: decode.present.3.value
            key.4: decode.present.4.key
            value.4: decode.present.4.value
            key.5: decode.present.5.key
            value.5: decode.present.5.value
            key.6: decode.present.6.key
            value.6: decode.present.6.value
            key.7: decode.present.7.key
            value.7: decode.present.7.value
            key.8: decode.present.8.key
            value.8: decode.present.8.value
            key.9: decode.present.9.key
            value.9: decode.present.9.value
            key.10: decode.present.10.key
            value.10: decode.present.10.value
            key.11: decode.present.11.key
            value.11: decode.present.11.value
            key.12: decode.present.12.key
            value.12: decode.present.12.value
            key.13: decode.present.13.key
            value.13: decode.present.13.value
            key.14: decode.present.14.key
            value.14: decode.present.14.value
            key.15: decode.present.15.key
            value.15: decode.present.15.value
            key.16: decode.present.16.key
            value.16: decode.present.16.value
            key.17: decode.present.17.key
            value.17: decode.present.17.value
            key.18: decode.present.18.key
            value.18: decode.present.18.value
            key.19: decode.present.19.key
            value.19: decode.present.19.value
            key.20: decode.present.20.key
            value.20: decode.present.20.value
            key.21: decode.present.21.key
            value.21: decode.present.21.value
            key.22: decode.present.22.key
            value.22: decode.present.22.value
            key.23: decode.present.23.key
            value.23: decode.present.23.value

      effect_roots: {}
```

## Session admission

For `scope: session`, a present durable value at admission replaces the
declared initializer for that state. The initializer supplies only missing
session state:

- `decoder_cache`: an empty 24-layer K/V bundle;
- `cache_lengths`: zero logical lengths.

Continuation therefore restores KV and logical lengths together. Qwen does not
persist the full next-token logits tensor, so every new invocation, including
continuation, requires a non-empty prompt. Suspension/resumption of the same
invocation retains working `logits` and is unaffected.

`policy_state` and `logits` are invocation-local. Both begin absent.
`first_policy_state` supplies reaction-zero policy values; the first successful
working commit initializes `policy_state` before `generation.steady` can read
it. `logits.current` is likewise read only after reaction zero writes
`decode_logits.selected`.

## Padding-correct reaction zero

`prefill_context_plan` derives all prompt facts from declared prompt lengths:

```text
prefill_attention_mask
prefill_position_ids
last_valid_prompt_index
decode_attention_mask
decode_position_ids
prompt_candidate_count
prompt_commit_indices
prompt_commit_length
prefill_logical_lengths
```

The prefill logits gather uses `last_valid_prompt_index`; it never assumes the
last physical sequence column is valid.

The reaction-zero decode mask covers:

```text
durable prefix + physical padded prompt + sampled token
```

while masking invalid prompt slots. Its sampled-token position ID uses the
logical valid prompt length. The later cache gather removes invalid prompt
positions, so steady reactions reconstruct their mask from compact durable
logical lengths alone.

## Commit plan

On `generation.first`:

```text
candidate_count = physical prompt width + 1
indices = valid prompt candidate ordinals
          ++ sampled-token ordinal when accepted
commit_length = valid prompt length + accepted length
```

On `generation.steady`:

```text
candidate_count = 1
indices = [0] when accepted, otherwise empty
commit_length = accepted length
```

Prompt inputs are optional on the `cache_commit_policy` binding and present
only on `generation.first`. Required `termination.accepted_length` anchors the
component to `generation.pulse`. There is one static gather transition in both
compiled blocks.

The model contract proves that every `decode.present.*` output extends the
corresponding bound past input. On reaction zero that past input is the prefill
bundle, whose contract proves it extends `decoder_cache.current` by the physical
prompt width. These composed contract relations establish the v20 full-source
prefix proof without comparing tensor contents.

## Zero-pulse behavior

For this graph, `request.max_iterations = 0` emits no pulse:

- no prefill or decode runs;
- no state samples `next`;
- no token publication is present;
- completion exposes the admission baseline;
- the no-op durable commit succeeds.

The general v19 lifecycle still runs completion and commit. This particular
Qwen graph has no completion-scoped publication, transactional effect, or
post-commit effect, so its observable result remains the historical successful
no-op.

## Derived continuous batching

The YAML contains no continuous-batching addon. The runtime may derive prefill
and steady model-call grouping from `batch_capacity`, `port_layouts`, state
capabilities, and backend evidence.

Each invocation still owns its reactor, state transition, publication batch,
and working/durable commit. If no grouping certificate is available, the exact
same graph executes in isolated mode.
