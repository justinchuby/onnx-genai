# Inference metadata review findings

Status: **Non-normative review synthesis**

[`INFERENCE_METADATA_DECISIONS.md`](INFERENCE_METADATA_DECISIONS.md) remains the
normative specification and wins if this document conflicts with it. This
companion does not add accepted schema. It consolidates where a fact belongs,
what the current schema can express, what the runtime currently executes, and
what remains a gap.

“Package” below means portable correctness metadata. “Runtime” includes serving,
scheduling, placement, and transport policy. Examples marked **Conceptual YAML**
are design sketches and are not asserted to parse.

## 1. Ownership and expressibility matrix

| Concern | Package owns | Caller owns | Runtime owns | Current expression / implementation | Gap or proposed direction |
| --- | --- | --- | --- | --- | --- |
| Publication and streaming | Typed output, publication point, `replace` / `append` / `event`, growth axis and valid prefix | Whether and how to consume incremental results | Buffering, SSE/gRPC/WebSocket delivery, backpressure, coalescing | `WorkflowStep::Emit` and `WorkflowEmitMode` express workflow publication | No standard delta/retract/revision envelope or proven end-to-end delivery contract |
| Interrupted session turn | Semantic state set, recurrence, lease requirement, rollback/snapshot/fork bounds, effect semantics | Cancellation and whether to retry a turn | Atomicity mechanism, failure recovery, visibility of publications | Session leases, workflow checkpoints, state-group capabilities, KV checkpoint/restore, and specialized recurrent snapshots exist | No atomic turn transaction spanning KV, recurrent state, RNG, grammar, effects, conversation state, and output visibility |
| Fork versus prefix reuse | Whether each state group is forkable or prefix-reusable and its correctness dependencies | Fork request and branch point | New session identity, CoW/storage strategy, prefix index and admission | `capabilities.fork`, `reuse.prefix_reusable`, KV primitives, and a typed engine fork capability exist | Engine backends currently do not expose end-to-end session fork |
| Prefill placement | Prefill computation and handoff as ordinary structural steps | Prompt and prior-session input | Fusion, placement, prefill/decode scheduling | Root steps and `loop.setup` have distinct execution frequency | Per-token cursor rejects non-empty setup and root work |
| Separate prefill/decode artifacts | Component ports, invokes, state aliases/carries, and ordering | Prompt and selected session | Artifact sessions, private handoff transport, memory residency | Workflow SSA can order separate components and bind their values | Multiturn prior-state handoff and final-writer/commit ownership are not complete across current runtime paths |
| Tensor rank and shape | Exact ordered dimensions, including explicitly unconstrained ones | Concrete runtime extents | Shape specialization and validation | Today `TensorContract` serializes required `rank` plus optional `shape`; validator checks consistency | **Proposed, not implemented:** make `shape` the sole rank authority, add explicit unconstrained `Any`, remove serialized `rank` |
| Continuous batching | Row relationship, masks/length companions, state semantics, and legal shared-forward structure | Independent requests | Admission, backfill, row table, compaction, batch size, fairness | Current supported decode paths dynamically fill fixed physical rows and advance live rows together | Runtime support is narrower than the portable workflow model; terminology should describe live-row iteration, not “row-major” |
| Speculative decoding | Proposer/target structure, vocabulary relation, shared state/weights, rollback bound and distribution claim | Opt-in where distribution is not preserved | Enablement, width up to bound, scheduling, candidate tree shape | Workflow-native `SpeculativeContract` is structural; legacy `SpeculatorConfig` still overlaps for sidecar discovery | No tree-candidate form; legacy overlap should converge on workflow-native structure |
| Tool calls | Tokenizer/template protocol facts and a versioned typed parse/render contract | Offered tools, JSON schemas, `tool_choice`, messages, and tool results | Protocol implementation and API translation | Server accepts caller tool fields but parses ATEM/Qwen/Llama/Mistral formats in hardcoded order | Replace family parsers with a declared, versioned typed protocol; do not add `supports_tools` |
| State spill and tiering | State semantics, snapshot/checkpoint form, reuse and eviction legality | Optional QoS intent through runtime API | CPU/device/disk placement, budgets, thresholds, migration and prefetch | State groups separate semantics from storage; runtime and KV crates implement tier policy | Package metadata must not name “CPU spill” or a placement policy |
| Overlapping-window ASR | Required bounded semantic state and typed window/stitch/revision protocol | Audio chunks and any caller-retained overlap | Window scheduling, transport and resource policy | Audio preprocessing, workflow state, and emits are individually expressible | No standardized window/overlap/stitch/revision contract |
| Learned PLE n-gram embedding | Graph-internal computation, or typed component/state contracts if externally implemented | Ordinary prompt/request data | Generic component execution and optimization | No repository schema, fixture, or ONNX port ABI verifies Flash-Next PLE | Do not confuse learned PLE with prompt-lookup speculative n-gram matching or dispatch by model family |

## 2. Publication is not transport

`emit` describes a workflow-visible publication:

- `replace` makes the emitted value the output's current value;
- `append` grows the output on the declared `axis` (the final axis only where
  that default is valid);
- `event` publishes an ordered occurrence rather than building one accumulated
  result;
- `when` suppresses publication, and `valid_length` limits the valid prefix.

These are semantic operations, not a promise that an HTTP response flushes after
every invocation. Transport framing, buffering, retry, backpressure, and client
disconnect behavior remain runtime/API concerns.

This distinction matters for ASR. A partial transcript may replace a hypothesis,
append only stable text, or publish an event, but the current contract cannot say
“delete characters 14–18,” “supersede hypothesis 7,” or “this segment is now
final.” There is no standardized typed delta/retract/revision envelope, nor
conformance that follows an `event` from workflow execution through a serving
transport to a caller. ASR revision should therefore be a versioned typed output
protocol, not an undocumented interpretation of `append`.

## 3. Session turns need a transaction boundary

The repository already has useful pieces:

- `scope: session` cells require `session_state_lease`;
- session continuation declares the conversation input/output relationship;
- `StateGroupCapabilities` declares `rollback_positions`, `snapshot`, `fork`,
  and `cascade`;
- `StateCheckpointContract` identifies a portable checkpoint adapter;
- the workflow runtime snapshots semantic session cells;
- KV implementations expose checkpoint/restore/fork primitives;
- native speculative decode snapshots recurrent state and rebuilds an accepted
  prefix after partial rejection.

The public `SessionCheckpoint` is currently a session id plus logical token
position; restore still fails closed for runner-backed decoder state that cannot
be rewound. It must not be mistaken for a complete transaction snapshot.

They do not collectively define a turn transaction. Cancellation or failure can
occur after some combination of KV update, recurrent-state mutation, RNG
advance, grammar transition, external effect, conversation update, and emitted
output. An atomic contract must define:

1. the complete semantic write set;
2. the before-turn snapshot or undo mechanism;
3. a single commit point;
4. abort behavior for every participating effect domain; and
5. publication visibility—especially whether provisional output may escape
   before commit and how it is retracted if the turn aborts.

An exclusive lease prevents two writers; it does not make one writer atomic.
Likewise, speculative rollback is bounded proposal recovery, not automatically
a whole-turn transaction.

## 4. Forking is not prefix caching

A **session fork** creates a new logical conversation at a declared position.
The child must be independently mutable and must reproduce every semantic state
participant: KV, recurrent/conv state, RNG, grammar, and other session cells.
Copy-on-write is an implementation optimization, not the meaning of fork.

A **prefix-cache hit** reuses computation for a compatible input prefix. It may
seed a new request without creating a parent/child relationship, and correctness
depends on declared reuse dependencies and state-group
`reuse.prefix_reusable`. Cache keys, hashing, tenancy, lookup, and eviction are
runtime policy.

The metadata distinction is present (`capabilities.fork` versus
`reuse.prefix_reusable`), and `onnx-genai-kv` has CoW fork and prefix-cache
primitives. The public engine protects fork behind `SessionForkCapability`, but
current backends return no capability because complete runner state cannot yet
be cloned safely. Metadata declarations therefore exceed the current
end-to-end runtime path.

## 5. Prefill, decode, and prior state

Control-flow placement gives frequency:

- root-prefix steps run once before an entered loop;
- `loop.setup` runs once each time that loop is entered;
- loop body steps run per iteration;
- root-suffix steps run after the loop.

Root-prefix placement is the clearest default for preprocessing and prefill that
belongs to the complete invocation. `loop.setup` is appropriate when the setup
belongs specifically to each entry of a nested or reusable loop. Neither needs
a serialized “phase.”

The run-to-completion interpreter supports this structure. The scheduler's
`WorkflowGenerationCursor`, however, is a per-token drive: it requires one
top-level generation loop, rejects non-empty `loop.setup`, and rejects
session-scoped workflow state because such state is published when a pass
completes. That is a runtime limitation, not a reason to put prefill in a
per-token body.

Separate prefill and decode ONNX artifacts are structurally ordinary
components. A root invoke can produce state consumed by a loop invoke, and
state-service aliases can describe the ports. Multiturn adds a harder case: a
later turn may begin with prior session state, prefill only its new prompt, then
hand the resulting state to decode. The current workflow/runtime combination
does not yet provide one complete contract for:

- selecting restored prior state versus empty initialization;
- ordering restore, incremental prefill, and decode across runtime routes;
- private state handoff between distinct artifact sessions; and
- identifying the final committed writer when prefill and decode both update a
  state group.

Portable `checkpoint` adapters are for cross-build export. A fast same-runtime
prefill/decode handoff may remain private, but its semantic ownership and commit
ordering still must be explicit.

## 6. TensorContract direction: shape is rank

The chosen design direction is **proposed and not implemented**:

1. `shape` becomes required and its list length is the sole rank authority.
2. A dimension can be fixed, symbolic, or explicitly unconstrained as `Any`.
3. The serialized `rank` field is removed.

**Conceptual YAML — not accepted by the current schema:**

```yaml
contract:
  dtype: float16
  shape: [batch, Any, hidden]
  batch_layout: {kind: request_aligned, axis: 0}
```

`Any` means an unconstrained extent at that position. It is not a reusable
symbol and does not assert equality with another `Any`. This removes two
serialized answers to rank while preserving the important distinction between
“this axis exists but its extent is unconstrained” and “the contract omitted
shape information.”

Today `TensorContract` still requires `rank: usize`, permits `shape` to be
absent, and validation compares `shape.len()` with `rank`. Existing fixtures
therefore remain evidence of the current format, not the proposed one.

## 7. Continuous batching terminology and boundary

One continuous-batch iteration performs one shared forward that advances each
**live physical row** by one logical decode token when that row has pending
logits. Rows may have different logical history lengths; dense tensors carry
that raggedness through per-row lengths, masks, write indices, or other declared
companions. Finished rows are removed, queued requests are admitted into freed
slots, and later requests backfill capacity without changing existing request
semantics.

“Row-major” is ambiguous here: it normally describes memory layout, while the
important property is one-token-per-live-row iteration with request-aligned
state. Use “continuous live-row batch” or “shared live-row forward.”

Metadata owns only facts required for correctness: `batch_layout`, padding and
valid-length companions, state update semantics, and a structurally batchable
body. Dynamic admission, row identity, physical slots, backfill timing,
compaction, fairness, and maximum batch size are runtime policy.

The current `ContinuousBatchManager` implements FIFO admission/backfill over a
fixed physical capacity for ORT static-cache, eligible shared-buffer
past/present, and configured native CUDA persistent-batch decode paths. It also
accounts for prompt-context forwards that emit no token. That implementation is
real but narrower than the general schema; generic workflows and arbitrary
state combinations cannot use the route, and its supported subset must not
become a package declaration.

## 8. Speculative decoding structure

The workflow-native contract is `SpeculativeContract`. Its current fields name
`proposer`, `target`, `proposal_execution`, `port_bindings`, `shared_state`,
`shared_weights`, `vocabulary`, `max_proposal_width`,
`distribution_preserving`, and `rollback_state`. `proposal_execution` is
`block` or `chained`; chained execution explicitly names
`token_embedding_input`, `logits_output`, recurrence bindings, and optional
folded-carry sources.

**Conceptual YAML — structurally representative, not a copyable fixture:**

```yaml
speculative:
  proposer: draft
  target: verifier
  proposal_execution: {kind: block}
  port_bindings: {}
  shared_state: [decoder_kv]
  shared_weights: []
  vocabulary: {kind: identical}
  max_proposal_width: 8
  distribution_preserving: true
  rollback_state: [decoder_kv]
```

This states compatibility and rollback limits. The runtime still chooses
whether to speculate and a width no larger than `max_proposal_width`.

`SpeculatorConfig` is an overlapping legacy/sidecar discovery surface read from
Hugging Face `config.json`, not the canonical workflow-native execution
contract. For MTP, the fields consumed by current resolution include
`proposal_type`, `model`, `num_speculative_tokens`, `target_hidden_output`,
`target_hidden_layout`, `target_hidden_size`, `hc_mult`, `mtp_hidden_output`,
`mtp_state_output`, `kv_mode`, `embedding`, and `lm_head`.

A migration to one structural declaration would map concepts, not copy keys:

| Sidecar fact | Workflow-native home |
| --- | --- |
| `model` | Proposer component `implementation.artifact` |
| `target_hidden_output` | Explicit target output reference/binding |
| `target_hidden_layout`, `target_hidden_size`, `hc_mult` | Typed proposer/target port shapes |
| `mtp_hidden_output`, `mtp_state_output` | Proposer output ports and, where stateful, a recurrence binding |
| `kv_mode` | State ownership plus accepted-prefix rollback semantics |
| `embedding`, `lm_head` | Explicit target initializer sharing/source |
| `num_speculative_tokens` | A package rollback upper bound only if proven; actual proposal width remains runtime policy |

The parser can resolve a fully specified MTP sidecar, but this does not remove
the duplication with `SpeculativeContract`, define that migration
normatively, or supply a canonical workflow fixture for every MTP shape.

`ProposalType::DFlash` is only recognized as an enum spelling and resolves to
`NotYetSupported` in this repository. This document intentionally makes no
claim about the DFlash algorithm. Likewise, runtime policy may choose a
candidate tree shape, but the current workflow-native schema has no typed form
for a proposer that produces a candidate tree and its parent topology. Such a
package needs a new structural contract rather than an overloaded flat block.

## 9. Tool protocols are typed interfaces

The caller owns the request-specific tool set: function names, descriptions,
JSON schemas, `tool_choice`, prior assistant `tool_calls`, and tool results. The
package's tokenizer/chat template owns protocol facts needed to render those
values and recognize model output. The runtime implements that declared
protocol and translates typed calls to the serving API.

The current server request types correctly accept caller tools and choice, but
`parse_tool_calls` tries hardcoded ATEM, Qwen, Llama, then Mistral parsers. That
is a model-family protocol registry embedded in serving code, not portable
package metadata.

The direction should be a versioned typed protocol contract with explicit
render and parse semantics, streaming boundary behavior, escaping, call-id
rules, and failure handling. It may be implemented by a registered native
adapter or a portable component. A `supports_tools: true` boolean is
insufficient: it neither identifies the wire grammar nor lets a reader fail
closed on an unsupported version.

## 10. Semantic state is not a placement tier

Package metadata may state:

- whether state is `semantic` or `advisory`;
- invocation or session scope and recurrence;
- state kind, graph ports, update discipline and logical lengths;
- snapshot/fork/rollback bounds and cascades;
- portable checkpoint adapter and version;
- `prefix_reusable` and `evictable_prefix` legality.

It must not state “spill to CPU,” a device/disk target, watermarks, budgets,
prefetch distance, migration thresholds, or an eviction algorithm. Those are
runtime configuration and may differ across machines without changing package
meaning.

“State” and “cache” are not synonyms. KV can be semantic session state even
when its physical representation is managed as a cache. An encoder result or
prefix entry may be a disposable acceleration cache. Conversely, recurrent
state may be small and non-cache-like but still mandatory for correctness.
Placement can tier any eligible storage; eviction is legal only where package
semantics say recomputation/reset or prefix removal preserves correctness.

## 11. Whisper-like overlapping windows

Two valid ownership models must remain distinct:

1. **Caller-retained overlap.** The caller resends the needed audio overlap.
   Metadata declares the input/window requirements; the runtime need not retain
   raw audio between calls.
2. **Package-required bounded session state.** The workflow retains declared,
   bounded semantic state such as a raw-sample tail, feature tail, time cursor,
   encoder carry, or provisional transcript. The state is leased,
   checkpointed/forked with the session where required, and advanced
   structurally.

Raw tail, computed features, encoder carry, time cursor, and provisional text
are not interchangeable. A package should declare only what its algorithm
requires; the runtime must not guess a Whisper-family policy. Current audio
preprocessing, session state, and publication primitives can model individual
pieces, but there is no standard contract connecting window size/stride,
overlap ownership, timestamp basis, stitching, hypothesis revision, and final
segment commitment. Until such a typed protocol exists, end-to-end
overlapping-window streaming is not a portable metadata claim.

## 12. Learned PLE is not prompt lookup

The reviewed Qwen3.8 Flash-Next “PLE n-gram embedding” topic refers to a learned
model computation. Prompt-lookup speculative decoding instead searches tokens
already present in the prompt/history and proposes a matching suffix. Sharing
the phrase “n-gram” does not give them the same state, inputs, training
semantics, or acceptance path.

A learned PLE belongs inside an ONNX graph when exportable. If it must remain a
separate implementation, it should be a typed component with explicit tensor
ports and any recurrent/session state declared like every other component.
Neither route permits `if model_family == ...` dispatch.

This repository currently has no verified ONNX fixture, workflow contract, or
port ABI for that PLE computation. External algorithm descriptions are
therefore **not yet package-verified** here. A future design must begin from an
audited artifact's real ports and state rather than inventing dimensions,
lookup rules, or update equations in metadata.

## 13. Prioritized gap table

This table is formatted so an issue can reference a stable group and row.

### Spec / schema

| Priority | Gap | Acceptance evidence |
| --- | --- | --- |
| P0 | Atomic session-turn transaction and publication visibility | Typed write set and commit/abort semantics cover KV, recurrent, RNG, grammar, conversation cells, effects, and provisional output |
| P0 | Tensor shape as sole rank authority | `Any` dimension is specified; serialized `rank` is removed from schema, fixtures, validator, and docs |
| P0 | Versioned tool protocol | Caller/package/runtime ownership is typed; at least two protocols validate without family dispatch |
| P1 | Revision-capable output envelope | Delta/retract/supersede/final operations are typed and ordered independently of transport |
| P1 | Separate prefill/decode state handoff | Restore/init, prefill writer, decode writer, and final commit ownership are unambiguous |
| P1 | Overlapping-window audio contract | Window/stride, overlap owner, time basis, retained state, stitching, revision, and finality are typed |
| P1 | Candidate-tree speculative form | Candidate tokens and parent topology have validated tensor contracts and rollback semantics |
| P2 | Workflow-native replacement for legacy sidecar overlap | MTP packages execute from one canonical structural declaration |

### Runtime

| Priority | Gap | Acceptance evidence |
| --- | --- | --- |
| P0 | End-to-end turn rollback | Injected cancellation at every mutation/publication boundary leaves either the entire old turn or entire new turn visible |
| P0 | End-to-end session fork | Capability is returned only when all semantic state forks; parent/child diverge independently |
| P1 | Per-token drive with setup/root/session state | Cursor runs setup exactly once, preserves root suffixes, and publishes session state transactionally |
| P1 | General live-row batching | Dynamic admission/backfill works beyond the current supported decode-path subset with declared ragged companions |
| P1 | Declared tool-protocol execution | Server selects a registered versioned protocol from package facts; hardcoded family chain is absent |
| P2 | Policy-only tiering controls | Runtime config chooses device/CPU/disk budgets and thresholds while metadata remains placement-free |

### Fixtures / conformance

| Priority | Gap | Acceptance evidence |
| --- | --- | --- |
| P0 | Interrupted-turn fault matrix | Fixture covers KV, recurrent, RNG, grammar/effect, and output visibility failures |
| P1 | Separate-artifact multiturn fixture | Real prefill and decode artifacts hand prior state across at least two turns |
| P1 | Revising streaming-ASR fixture | Overlapping windows revise then finalize a hypothesis through an API transport |
| P1 | Workflow-native MTP fixture | Real proposer/target ports and state validate and execute without legacy-only discovery |
| P1 | Continuous-batch ragged/backfill fixture | Unequal prompt/history lengths, early finish, compaction, and late admission preserve per-request results |
| P2 | Learned-PLE artifact audit | Checked-in or reproducibly obtained ONNX artifact establishes real ports/state before schema work |
| P2 | DFlash evidence gate | No runtime claim until a typed contract, implementation, and conformance fixture exist |

## 14. Repository evidence

Authoritative and schema evidence:

- `docs/genai/INFERENCE_METADATA_DECISIONS.md`
- `docs/genai/MOBIUS_WORKFLOW_PRODUCER.md`
- `docs/architecture/WORKFLOW_RUNTIME_UNIFICATION.md`
- `crates/onnx-genai-metadata/src/schema/ir.rs`
- `crates/onnx-genai-metadata/src/schema/package.rs`
- `crates/onnx-genai-metadata/src/schema/generation.rs`
- `crates/onnx-genai-metadata/src/parser.rs`
- `crates/onnx-genai-metadata/src/session_state.rs`
- `crates/onnx-genai-metadata/src/decoder_workflow.rs`
- `crates/onnx-genai-metadata/src/validation.rs`
- `schema/inference_metadata.schema.json`

Runtime evidence:

- `crates/onnx-genai-engine/src/pipeline/workflow.rs`
- `crates/onnx-genai-engine/src/pipeline/mod.rs`
- `crates/onnx-genai-engine/src/pipeline/runtime_state.rs`
- `crates/onnx-genai-engine/src/pipeline/speculative.rs`
- `crates/onnx-genai-engine/src/batched.rs`
- `crates/onnx-genai-engine/src/config.rs`
- `crates/onnx-genai-engine/src/engine/mod.rs`
- `crates/onnx-genai-engine/src/native_speculative.rs`
- `crates/onnx-genai-engine/src/decode/state.rs`
- `crates/onnx-genai-kv/src/lib.rs`
- `crates/onnx-genai-kv/src/prefix_cache.rs`
- `crates/onnx-genai-kv/src/local_tiered.rs`
- `crates/onnx-genai-server/src/types.rs`
- `crates/onnx-genai-server/src/routes/completions.rs`
- `crates/onnx-genai-server/src/sse.rs`

Fixture evidence:

- `tests/fixtures/onnx_genai_workflows/speculative/inference_metadata.yaml`
- `tests/fixtures/tiny-mtp-full/inference_metadata.yaml`
- `tests/fixtures/tiny-mtp-full/mtp/model.onnx.textproto`
- `tests/fixtures/onnx_genai_workflows/speech_wav/inference_metadata.yaml`
- `tests/fixtures/onnx_genai_workflows/static_cache/inference_metadata.yaml`
