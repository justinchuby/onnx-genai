# Runtime fork and rewind API

## API landed

`onnx-genai-engine` now exposes persistent-session rewind primitives:

- `Engine::checkpoint_session(session_id) -> SessionCheckpoint`
- `Engine::restore_session(checkpoint)`
- `Engine::rewind_session_by(session_id, RewindTokenCount) -> SessionPosition`
- `Engine::rewind_session_to(session_id, SessionPosition)`
- `Engine::prepare_session_fork(source, SessionPosition) -> SessionForkPlan`
- `Engine::fork_session(SessionForkPlan) -> SessionId`

`SessionCheckpoint` is a small public token-boundary handle:

```rust
pub struct SessionCheckpoint {
    pub session_id: SessionId,
    pub position: SessionPosition,
}
```

`SessionPosition` and `RewindTokenCount` are newtypes, not bare `usize`, so
callers must opt into absolute-vs-relative token-boundary conversion.

`SessionForkPlan` is a consuming capability and an inspectable participant list.
Admission deep-clones every fallible value and validates every backend before a
child exists. Publication rejects a stale plan if the source advanced after
admission.

## Rewind behavior and cost model

Rewind mutates only the named persistent session. It first validates the target
logical position, backend rewind support, and retained KV range. Only after that
gate passes does it truncate the logical token vector and reuse the existing
speculative-decoding machinery:

- target state: `rewind_target_state_to_len`
- draft state, when loaded: `rewind_draft_state_to_len`
- decode internals: `rewind_decode_state_to_len`

Cost is `O(pages removed)` for paged KV plus backend-specific mutation:

- runner-backed static-cache / shared-buffer paths are rejected before session
  mutation because the runner rewind is not yet transactional;
- sliding-window past/present rewinds only positions still physically retained;
- paged materialized past rewinds pages and reloads materialized past tensors;
- ORT-owned KV without paged materialization is rejected by
  `rewind_decode_state_to_len` because the tensors cannot be safely truncated.

`rewind_session_by` rejects attempts to rewind before token zero.
`rewind_session_to` rejects positions past the current logical token length and
rejects unsupported backends or sliding-window evicted positions without changing
the session tokens, KV token count, decode state, or paged KV cache.

## Prefix-cache invariants

Prefix cache entries own independent page-table references. A session rewind
releases only that session's references. Therefore:

1. cached prefix pages remain valid after the originating session rewinds;
2. divergent writes after rewind hit page-table copy-on-write when a retained
   prefix still references the page;
3. pages are reclaimed only when neither any session nor prefix cache entry holds
   a reference;
4. no cached prefix may point at a reclaimed page.

The engine test `cached_prefix_pages_survive_rewind_and_divergent_write` locks
this invariant without requiring a model load.

## Fork design

The low-level paged KV cache already supports CoW fork:

- fork is `O(number_of_prefix_pages)`;
- no tensor payload is copied at fork time;
- each retained page's refcount is incremented;
- the first divergent write to a shared page allocates and copies that page only.

The engine-level plan also covers target/draft decode state, fixed recurrent and
convolution bindings, token continuation, RNG/constraint boundary state,
workflow state/effects, and output head/cursor/lineage/closure. ORT
ZeroCopyRebind runners deep-clone exported KV into a fresh runner. Materialized
paged KV uses CoW pages and can fork a retained earlier committed position when
no non-position-addressable participant is present. Static/shared-buffer and
native participants without clone/import support fail before child creation.

## Backend support matrix

| Decode path | Rewind | Fork |
|---|---|---|
| Native single-session backend | Not supported by persistent-session APIs | Declined before child: no per-session native snapshot/import participant |
| ORT static-cache/shared-buffer runner | Cursor rewind only where supported | Declined before child: fixed mutable buffers expose no clone/import |
| ORT ZeroCopyRebind PastPresent runner | Backend-specific | Current committed position; exported KV and fixed state are cloned |
| ORT sliding-window/materialized paged KV | Supported only to retained positions | Current or retained historical committed position; CoW pages |
| Draft/speculative session state | Rewound alongside target | Forked as one transitive target/draft cascade when every participant admits |
| Interpreted workflow state | Checkpoint adapter/runtime dependent | Current committed boundary; semantic values, effects, and output baselines clone together |

## REPL integration notes

The REPL should call `prepare_session_fork`, display any actionable participant
refusal, then consume the returned plan with `fork_session`.

## Verification approach

The model-free engine tests now cover:

- direct CoW sharing and divergent-write copy-on-write
  (`paged_kv_fork_shares_prefix_then_diverges_copy_on_write`);
- prefix-cache page safety after rewind and divergence
  (`cached_prefix_pages_survive_rewind_and_divergent_write`);
- transactional failure for unsupported rewind paths
  (`failed_rewind_of_windowed_evicted_position_leaves_session_unchanged` and
  `failed_rewind_of_ort_owned_kv_leaves_session_unchanged`) plus
  runner-backed rejection before tokens or paged KV are touched
  (`failed_rewind_of_runner_backed_state_leaves_session_unchanged`);
- randomized fork / rewind / append / remove operation sequences with
  `proptest`, checking that every live session's length is independent and every
  page refcount exactly matches live sequence references
  (`paged_kv_refcounts_match_live_sequences_for_random_fork_rewind_interleavings`).

Neither `proptest` nor `quickcheck` was already present in the workspace, so this
slice adds `proptest` as an engine dev-dependency.

Miri was not applied: the touched KV fork/rewind/page-table paths are safe Rust
and the current changes did not add or modify `unsafe` or raw-pointer code in
`onnx-genai-kv`.

TLA+ was deliberately skipped for this slice. The repository already has TLA+
tooling under `specs/tla`, but the local environment has no Java runtime and no
`TLA2TOOLS_JAR`. More importantly, the implemented fork behavior exercised here
is source-worker-local page-table reference counting plus a typed
prepare/publish protocol with exact runtime assertions and proptest coverage.
A TLA+ model becomes worthwhile if fork publication later spans workers or a
distributed mutable-state protocol.

Follow-up: enable static/shared-buffer and native fork only after those backends
expose independent clone/import participants whose validation prebuilds all
fallible artifacts.
