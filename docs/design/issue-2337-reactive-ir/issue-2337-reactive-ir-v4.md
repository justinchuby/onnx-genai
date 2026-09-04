# Cyclic reactive dataflow IR — v4 candidate

This is the first semantically closed candidate from the Qwen design exercise.

## 1. Core graph

```yaml
graph:
  clock:
    kind: synchronous
    index: reaction.index
    repeat_while: termination.continue
    sample_repeat: after_commit

  nodes:
    <node-id>:
      kind: component | state | merge | effect_merge
      # kind-specific fields
```

- One clock in v1; multi-clock/asynchronous flow is rejected and deferred.
- Component outputs are addressed as `<node-id>.<artifact-port>`. Split at the first dot; node IDs cannot contain dots.
- All ordinary edges are consumer-side `inputs` bindings.
- Every instantaneous cycle is invalid. Every accepted cycle crosses at least one `state` node.
- `repeat_while` is scalar bool and sampled only after successful commit of the reaction.

## 2. Component nodes

```yaml
components:
  decoder:
    implementation: {kind: onnx, artifact: model.onnx}
    ports:
      roles: {input_ids: token_ids, logits: logits}
    # Signature is read from ONNX ValueInfo.

  policy:
    implementation: {kind: binding}
    contract: {id: onnx-genai.token-policy, version: '1'}
    ports:
      # Binding signatures are explicit for static analysis.
      inputs: {...}
      outputs: {...}

nodes:
  decode:
    kind: component
    component: decoder
    when: request.use_decoder       # optional scalar firing gate
    inputs:
      input_ids: token.current
```

- ONNX dtype/rank/shape declarations are not duplicated in metadata.
- Workflow request/output boundary contracts remain explicit.
- Binding component signatures remain explicit.
- `bool[batch]` is ordinary tensor data and never changes graph firing implicitly.
- A false scalar `when` makes outputs absent; every maybe-absent use must pass through an explicit merge or an equally gated consumer.

## 3. State nodes

```yaml
nodes:
  latent:
    kind: state
    initial: noise.sample
    next: denoiser.next_latent
    transition: {kind: replace}
    ownership: workflow
    scope: invocation
```

State is generic and carries no model-semantic kind. Core state behavior is limited to facts that affect execution:

```yaml
transition: {kind: replace}
transition: {kind: append, axis: 2, increment: accepted_len.next,
             bound: package.max_context}
transition: {kind: indexed_scatter, axis: 2, indices: cursor.current,
             logical_length: length.current, capacity: package.max_context}
```

Additional execution facts:

```yaml
ownership: workflow | runtime | external
scope: invocation | session
release: invocation_end | session_end | row_release
aliasing: forbidden | permitted | required
reuse: {prefix: allowed | forbidden, evict_prefix: allowed | forbidden}
capabilities: {rollback_positions: 1, snapshot: true, fork: true}
```

- A state serializes no type contract. `initial`, every consumer of `current`, and `next` must unify from workflow boundaries, ONNX ValueInfo, or explicit binding signatures.
- `next` has exactly one producer. No last-writer-wins behavior exists.
- `transition` constrains the observable `current -> next` relationship; it does not prescribe allocation, copying, paging, device, or kernel.
- Optional scalar `when` means false retains current. Per-row retention is expressed by tensor data such as zero increment or by a component that selects current/candidate.

## 4. Bundled state

```yaml
nodes:
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
        initial: prefill.present.0.key
        next: decode.present.0.key
      value.0:
        initial: prefill.present.0.value
        next: decode.present.0.value
      # ... explicit members through layer 23 ...
    interfaces:
      onnx-genai.attention-state:
        version: '1'
        bindings:
          layers:
            - {index: 0, key: key.0, value: value.0}
            # ... through layer 23 ...

  decode:
    kind: component
    component: model
    inputs:
      past_key_values.0.key: decoder_cache.current.key.0
      past_key_values.0.value: decoder_cache.current.value.0
```

- `members` supplies the canonical feedback edges.
- Optional interfaces only prove implementation substitution legality. They may reference canonical members but cannot add/override ports, edges, state, or order.
- Generic execution remains correct when no interface is used.

## 5. Initialization

No `setup` region or phase annotation exists.

```text
init closure = reverse dependency closure of all state.initial bindings
```

The compiler executes that acyclic closure once before reaction 0. It rejects:

- any initializer transitively depending on `state.current`;
- a cycle inside the init closure;
- an initializer whose resolved type conflicts with `next` or a current consumer;
- effectful initialization that violates its effect contract.

This naturally places Qwen empty-cache construction, prefill, and first-logits extraction in initialization without introducing a second control-flow language.

## 6. Transactions

Each stateful SCC is one atomic transaction by default:

1. snapshot all state in the SCC;
2. evaluate against that snapshot;
3. stage all `next` candidates;
4. commit all state together, or retain all previous state on failure;
5. publish commit-coupled outputs;
6. sample `repeat_while`.

Qwen token/logits/active/done/length/RNG/mask/KV are one SCC and therefore need no repeated transaction name.

Only otherwise independent SCCs that require joint atomicity declare an override:

```yaml
transactions:
  speculative_pair:
    merge:
      - state: draft_cache
      - state: target_cache
    isolation: snapshot
    abort: retain_previous
```

The override names anchor state nodes, not a duplicated list of every member.

## 7. Output publication

```yaml
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
```

- Publication from a stateful reaction occurs after successful transaction commit.
- Failed reactions publish nothing unless an output explicitly declares a versioned provisional/retraction protocol.
- Publication order for one output stream is a derived linear effect; package authors do not serialize a second sequence.

## 8. Branching and presence

```yaml
nodes:
  encoder:
    kind: component
    component: encoder
    when: request.has_media
    inputs: {media: request.media}

  fallback:
    kind: component
    component: empty_embedding
    when: request.no_media

  embedding:
    kind: merge
    inputs: [encoder.hidden, fallback.hidden]
    require: exactly_one_present
```

- Scalar gates control node presence.
- Per-row predicates stay tensor data.
- `merge` is typed and explicit; it does not use YAML order as priority.
- State on an untaken path must explicitly retain current or merge a selected candidate before `next`.

## 9. Linear effects

Top-level domains declare retry/speculation behavior:

```yaml
effects:
  external_store:
    retry: transactional       # pure | idempotent | transactional | non_retryable
    speculation_safety: {...}
```

Effectful binding/component signatures expose typed linear effect ports. Node instances connect them exactly like data ports:

```yaml
nodes:
  load:
    kind: component
    component: state_reader
    effects:
      external_store: effects.external_store.start

  write:
    kind: component
    component: state_writer
    inputs: {value: load.value}
    effects:
      external_store: load.effects.external_store

effect_sinks:
  external_store: write.effects.external_store
```

Rules:

- Every effect token has exactly one consumer; fan-out is invalid.
- A domain has one start token and one terminal sink per init/reaction execution.
- Two effectful nodes in one domain with no token path between them are rejected rather than ordered by YAML.
- A gated effect produces a maybe-present token and must join through `effect_merge` before the sink.
- Init-closure effects run once. Reaction effects run once per committed attempt, subject to retry/speculation safety.
- Non-retryable effects cannot occur before a fallible/abortable operation in the same transaction unless the graph establishes a safe commit boundary.

## 10. Qwen mapping

| Current YAML | v4 source |
|---|---|
| `steps[].invoke` | component node + consumer-side inputs |
| `loop.setup` | derived initialization closure |
| `loop.steps` | data/effect dependencies |
| `loop.carried` | state `initial` and `next` |
| `loop.iteration` | typed `reaction.index` |
| `loop.continue_when` | post-commit `clock.repeat_while` |
| `emit` | output publication binding |
| 48 state contracts | inferred from ONNX endpoints |
| ONNX component port contracts | ONNX ValueInfo |
| `state_service.ports` | bundle members + optional interface proof |
| repeated transaction names | stateful SCC atomicity |
| physical KV plan | runtime-only lowering |

The 24-layer key/value member list remains explicit: member identity and optimization pairing are semantic bindings that cannot be safely inferred from artifact port spelling. Producers should generate those records; the core schema should not become a regex/template language.

## 11. Required validation diagnostics

Every rejection names the relevant node/state/member/port/effect and provenance:

- instantaneous algebraic-loop path;
- unseeded or state-dependent initializer path;
- conflicting endpoint contracts and source artifacts;
- ambiguous/multiple state-next producers;
- maybe-absent value used without a compatible gate/merge;
- non-linear or unterminated effect-token path;
- unsupported transition behavior or interface version;
- transaction/effect retry incompatibility;
- repeat predicate not scalar bool or unavailable at post-commit boundary.
