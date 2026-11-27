# Qwen 2.5 reactive dataflow IR — v0

> **Design sketch, not accepted schema.** This shows the proposed replacement for the authored `steps` / `loop` / `invoke` / `emit` syntax in the published Qwen metadata. Existing input and component port contracts are abbreviated where they are unchanged.

## Proposed rules

1. `components` declare reusable typed implementations; `graph.nodes` instantiate them.
2. Node outputs are addressed as `<node-id>.<component-output-port>`. Node IDs cannot contain `.`; the remainder is the artifact-defined opaque port name.
3. A `state` node is the only legal delay. Its `current` value is the committed snapshot for reaction N; `next` is staged for N+1.
4. Every cycle must cross a `state` node. Removing state nodes must leave an acyclic graph.
5. State initialization is a dataflow closure: ancestors referenced only by `state.initial` run once. No authored `setup` region is needed.
6. A reaction snapshots state, evaluates nodes, stages writes, then atomically commits each transaction. Outputs coupled to a transaction publish only after commit.
7. `repeat_while` is sampled after commit. False ends the cyclic region; it does not roll back the final reaction.
8. A state member has exactly one `next` producer. Selection/merge happens explicitly before the state node.
9. `state_service.ports` disappears. State-member bindings plus `role`/`layer` are the canonical decoder ABI.
10. The runtime derives SCCs, initialization closure, invariant hoisting, transaction members, effect order, decoder ABI, and physical KV lowering. None are serialized as a second plan.

## Proposed YAML shape

```yaml
schema_version: vNext
pipeline:
  workflow:
    manifest:
      semantics:
        reaction: synchronous
        state_visibility: next_reaction
        failure: abort_transaction

    inputs:
      # Existing request/package input contracts remain unchanged.
      request.input_ids: { ... }
      request.max_iterations: { ... }
      request.prompt_lengths: { ... }
      request.eos_ids: { ... }
      request.eos_lengths: { ... }
      request.temperature: { ... }
      request.top_k: { ... }
      request.top_p: { ... }
      request.min_p: { ... }
      request.seed: { ... }

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
          value: token_update.next
          mode: append
          when: active.current
          valid_length: emitted_length.total
          transaction: decode_step

    components:
      # Existing typed implementation declarations stay here unchanged.
      model: { ... }
      token_sampler: { ... }
      termination: { ... }
      token_state_update: { ... }
      last_token_logits: { ... }
      decoder_state_initializer: { ... }
      decoder_step_update: { ... }
      cache_length_update: { ... }
      termination_batch_initializer: { ... }
      token_to_slot: { ... }
      generated_length_update: { ... }

    graph:
      reaction:
        # Typed intrinsic supplied by the reaction scheduler. This replaces the
        # loop induction variable, but is still explicit data in the graph.
        index:
          value: reaction.index
          contract: {dtype: int64, rank: 1, shape: [1]}
        repeat_while: termination.continue

      transactions:
        decode_step:
          commit: reaction_success
          abort: retain_previous
          isolation: snapshot

      nodes:
        # --- Initialization closure ---
        # These nodes are not marked as a special phase. They are run once because
        # their outputs are reachable only from state.initial bindings.
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
            # ... explicit artifact ports through layer 23 ...
            past_key_values.23.key: initialize.past_key_values.23.key
            past_key_values.23.value: initialize.past_key_values.23.value

        prefill_logits:
          kind: component
          component: last_token_logits
          inputs:
            logits: prefill_model.logits

        # --- Reaction graph ---
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

        sample_slot:
          kind: component
          component: token_to_slot
          inputs: {token: sample.token}

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

        token_update:
          kind: component
          component: token_state_update
          inputs:
            current: token.current
            update: sample_slot.slot
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

        next_cache_length:
          kind: component
          component: cache_length_update
          inputs:
            left: cache_lengths.current
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
            # KV input ports are wired by decoder_cache.members below. This avoids
            # repeating the same 48 bindings in both the graph and state service.

        decode_logits:
          kind: component
          component: last_token_logits
          inputs: {logits: decode_model.logits}

        # --- Scalar/tensor delayed state ---
        token:
          kind: state
          contract: {infer: [initialize.token_slot, token_update.next]}
          initial: initialize.token_slot
          next: token_update.next
          scope: invocation
          transaction: decode_step

        logits:
          kind: state
          contract: {infer: [prefill_logits.last_logits, decode_logits.last_logits]}
          initial: prefill_logits.last_logits
          next: decode_logits.last_logits
          scope: invocation
          transaction: decode_step

        generated_lengths:
          kind: state
          contract: {infer: [initialize.generated_lengths, next_generated_length.total]}
          initial: initialize.generated_lengths
          next: next_generated_length.total
          scope: invocation
          class: semantic
          transaction: decode_step

        active:
          kind: state
          contract: {infer: [package.active, termination.next_active]}
          initial: package.active
          next: termination.next_active
          scope: invocation
          class: semantic
          transaction: decode_step

        done:
          kind: state
          contract: {infer: [package.not_done, termination.done]}
          initial: package.not_done
          next: termination.done
          scope: invocation
          class: semantic
          transaction: decode_step

        accepted_len:
          kind: state
          contract: {infer: [package.zero_batch, accepted_length.total]}
          initial: package.zero_batch
          next: accepted_length.total
          scope: invocation
          class: semantic
          transaction: decode_step

        cache_lengths:
          kind: state
          contract: {infer: [initialize.cache_lengths, next_cache_length.total]}
          initial: initialize.cache_lengths
          next: next_cache_length.total
          scope: invocation
          class: semantic
          transaction: decode_step

        rng_counter:
          kind: state
          contract: {infer: [request.rng_counter, sample.next_counter]}
          initial: request.rng_counter
          next: sample.next_counter
          scope: invocation
          class: semantic
          transaction: decode_step

        attention_mask:
          kind: state
          contract: {infer: [initialize.attention_mask, next_mask.next_attention_mask]}
          initial: initialize.attention_mask
          next: next_mask.next_attention_mask
          scope: invocation
          transaction: decode_step

        # One logical state node owns all 48 Qwen KV tensors. Each member declares
        # its semantic identity and the one producer/consumer pair. This replaces:
        #   * state.cache_0 ... state.cache_47
        #   * loop.carried entries for cache_0 ... cache_47
        #   * serving.state_service.groups.decoder_cache.ports
        decoder_cache:
          kind: state
          scope: invocation
          class: semantic
          management: runtime
          release_boundary: invocation
          transaction: decode_step
          commit_length: accepted_len.next
          recurrence:
            kind: bounded
            axis: 2
            max: package.max_context
          service:
            kind: full_attention
            sequence_axis: 2
            layout: bnsh
            aliasing: permitted
            reuse: {prefix_reusable: true, evictable_prefix: false}
          members:
            key.0:
              role: key
              layer: 0
              contract: {infer: decode_model.past_key_values.0.key}
              initial: prefill_model.present.0.key
              current_to: decode_model.past_key_values.0.key
              next: decode_model.present.0.key
            value.0:
              role: value
              layer: 0
              contract: {infer: decode_model.past_key_values.0.value}
              initial: prefill_model.present.0.value
              current_to: decode_model.past_key_values.0.value
              next: decode_model.present.0.value
            # Entries remain explicit because role/layer cannot be inferred from
            # artifact port spelling. Producers should generate these 48 records.
            # ... key.1/value.1 through key.22/value.22 ...
            key.23:
              role: key
              layer: 23
              contract: {infer: decode_model.past_key_values.23.key}
              initial: prefill_model.present.23.key
              current_to: decode_model.past_key_values.23.key
              next: decode_model.present.23.key
            value.23:
              role: value
              layer: 23
              contract: {infer: decode_model.past_key_values.23.value}
              initial: prefill_model.present.23.value
              current_to: decode_model.past_key_values.23.value
              next: decode_model.present.23.value

    serving:
      active: active.current
      done: done.current
      accepted_len: accepted_len.current
      # No state_service block: decoder_cache.service is its canonical source.
```

## Why this is materially better than a mechanical `steps -> nodes` rewrite

| Current authored fact | v0 canonical source |
|---|---|
| `invoke.inputs` + `invoke.outputs` | node inputs + implicit typed component outputs |
| `loop.setup` | ancestors of `state.initial` |
| `loop.steps` order | data/effect dependencies |
| `loop.carried` | `state.initial` / `state.next` |
| `loop.iteration` | typed `reaction.index` |
| `loop.continue_when` | `graph.reaction.repeat_while` |
| `emit` | output `publication` binding |
| `state.cache_0..47` | `decoder_cache.members` |
| `state_service.groups.decoder_cache.ports` | the same `decoder_cache.members` |
| transaction membership | each state node's `transaction` |
| physical contiguous/paged KV plan | runtime-derived, never serialized |

## Deliberate non-features

- No glob/regex/template syntax for `past_key_values.*`: role and layer are semantic facts and cannot be inferred from artifact names.
- No authored schedule, SCC list, init phase, execution island, CUDA graph, page size, block table, slot mapping, or device placement.
- No implicit multi-writer merge, implicit absent value, or YAML-order priority.
- No second flattened decoder ABI beside the graph.

## Open design risks

1. **KV as one bundled state node vs 48 ordinary state nodes.** The bundle removes three-way duplication and gives KV atomicity naturally, but introduces `members` as a structured state type.
2. **`repeat_while` vs pure quiescence.** An explicit post-commit predicate is easy to diagnose and exactly preserves the existing final reaction. Pure “repeat while any state write is enabled” is smaller but makes termination implicit and harder to validate.
3. **Contract inference.** Inferring a state member's contract from connected typed ports removes duplicate shapes. The validator must require all inferred contracts to unify and produce provenance-rich errors.
4. **Initialization closure.** This avoids a second control-flow language, but the validator must reject a state initializer that depends (transitively) on any `state.current` value.
5. **Effectful nodes.** This Qwen graph is mostly tensor dataflow plus publication. General workflows still need linear effect-token ports or a resource-effect declaration; YAML order must never become the fallback.
