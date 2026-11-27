# Cyclic reactive dataflow IR — v1 decisions

## Scope locked in this iteration

- Core v1 supports **single-clock cyclic tensor workflows**.
- This covers autoregressive decoders, diffusion/flow iterations, RNN/SSM, causal convolution, codec loops, and weather/video rollout.
- Multi-clock/asynchronous streaming is a future extension. v1 must reject cross-clock constructs rather than guess.
- Bundled state is generic; Qwen KV is one use, not a core state category.
- There is **no `state kind` enum**. Metadata configures observable runtime behavior instead of naming what a state “means.”

## Generic state node

```yaml
graph:
  transactions:
    decode_step:
      isolation: snapshot
      commit: reaction_success
      abort: retain_previous

  nodes:
    decoder_cache:
      kind: state

      # Temporal behavior
      initial_visibility: reaction_0
      update_visibility: next_reaction
      transaction: decode_step

      # Ownership/lifetime behavior
      ownership: runtime
      scope: invocation
      release: invocation_end

      # Update behavior. These are closed, executable semantics—not model kinds.
      update:
        kind: append                 # replace | append | indexed_scatter
        axis: 2
        amount: accepted_len.next    # per-row committed positions
        bound: package.max_context

      # Legal runtime transformations
      aliasing: permitted            # forbidden | permitted | required
      reuse:
        prefix: allowed
        evict_prefix: forbidden
      capabilities:
        rollback_positions: 1
        snapshot: true
        fork: true

      members:
        key.0:
          contract: {infer: decode_model.past_key_values.0.key}
          initial: prefill_model.present.0.key
          current_to: decode_model.past_key_values.0.key
          next: decode_model.present.0.key
        value.0:
          contract: {infer: decode_model.past_key_values.0.value}
          initial: prefill_model.present.0.value
          current_to: decode_model.past_key_values.0.value
          next: decode_model.present.0.value
        # ... explicit members through key.23/value.23 ...
```

The same core state node expresses other model families without adding kinds:

```yaml
# Diffusion / flow latent
latent:
  kind: state
  initial: noise.sample
  next: denoiser.next_latent
  transaction: denoise_step
  ownership: workflow
  scope: invocation
  update: {kind: replace}

# RNN / SSM recurrent bundle
recurrent:
  kind: state
  initial: initializer.hidden
  next: cell.next_hidden
  transaction: recurrent_step
  ownership: runtime
  scope: session
  update: {kind: replace}
  members:
    hidden: {current_to: cell.hidden, next: cell.next_hidden}
    conv: {current_to: cell.conv_state, next: cell.next_conv_state}

# Fixed-capacity scatter cache
cache:
  kind: state
  transaction: decode_step
  ownership: runtime
  update:
    kind: indexed_scatter
    axis: 2
    indices: write_cursor.current
    logical_length: cache_length.current
    capacity: package.max_context
```

## Optional interface contracts

The core does not call `decoder_cache` “KV,” “attention,” or “Qwen.” A runtime can execute it generically from its update behavior.

A physical substitution such as paged attention needs additional proof about member/port relationships. That proof is optional and versioned; it is not a second graph and cannot override any edge:

```yaml
    decoder_cache:
      kind: state
      # ... canonical members and behavior above ...
      interfaces:
        onnx-genai.attention-state:
          version: '1'
          bindings:
            layers:
              - index: 0
                key: key.0
                value: value.0
              # ... through layer 23 ...
```

Rules:

1. `interfaces` may only reference canonical state members; it cannot declare ports or edges.
2. Unknown interface ID/version fails closed if the selected implementation requires it.
3. Ignoring an optional interface must still permit correct generic execution.
4. The interface proves substitution compatibility; physical page size, page tables, slots, allocators, device, and eviction policy remain runtime-owned.
5. If paged execution is mandatory rather than optional, the component/runtime capability contract declares that requirement explicitly.

## Qwen structural rewrite

The v0 graph shape remains, with these corrections:

```yaml
pipeline:
  workflow:
    outputs:
      tokens:
        # existing contract/role/stage
        publication:
          value: token_update.next
          mode: append
          when: active.current
          valid_length: emitted_length.total
          transaction: decode_step

    graph:
      clock:
        kind: synchronous
        index: reaction.index
        repeat_while: termination.continue

      nodes:
        # Components referenced only by state.initial form the derived init closure.
        initialize: {kind: component, component: decoder_state_initializer, inputs: {...}}
        prefill_model: {kind: component, component: model, inputs: {...}}
        prefill_logits: {kind: component, component: last_token_logits, inputs: {...}}

        sample: {kind: component, component: token_sampler, inputs: {...}}
        termination: {kind: component, component: termination, inputs: {...}}
        token_update: {kind: component, component: token_state_update, inputs: {...}}
        next_mask: {kind: component, component: decoder_step_update, inputs: {...}}
        decode_model: {kind: component, component: model, inputs: {...}}
        decode_logits: {kind: component, component: last_token_logits, inputs: {...}}

        token:
          kind: state
          initial: initialize.token_slot
          next: token_update.next
          transaction: decode_step
          update: {kind: replace}

        logits:
          kind: state
          initial: prefill_logits.last_logits
          next: decode_logits.last_logits
          transaction: decode_step
          update: {kind: replace}

        decoder_cache:
          kind: state
          # generic behavior + 48 explicit members + optional interface contract
          # as shown above

    serving:
      active: active.current
      done: done.current
      accepted_len: accepted_len.current
```

## What is enumerated in core

Only values that directly change execution:

- `clock.kind`: currently only `synchronous` in v1.
- state `update.kind`: `replace | append | indexed_scatter`.
- transaction isolation/commit/abort policy.
- ownership, scope, release boundary.
- aliasing and reuse legality.
- rollback/snapshot/fork capabilities.
- output publication mode and commit coupling.

Model/domain labels are not core state kinds. Versioned interface contracts exist only where a runtime substitution needs additional, checkable structure.

## Remaining decision

Should cycle termination remain the explicit post-commit `clock.repeat_while`, or be inferred from quiescence (no enabled state update)? The explicit predicate preserves the current Qwen final-step behavior and gives better diagnostics; quiescence is smaller but makes termination implicit.
