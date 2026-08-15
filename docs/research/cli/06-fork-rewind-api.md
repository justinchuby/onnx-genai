# Runtime fork and rewind API

## API landed

`onnx-genai-engine` now exposes persistent-session rewind primitives:

- `Engine::checkpoint_session(session_id) -> SessionCheckpoint`
- `Engine::restore_session(checkpoint)`
- `Engine::rewind_session_by(session_id, RewindTokenCount) -> SessionPosition`
- `Engine::rewind_session_to(session_id, SessionPosition)`
- `Engine::session_fork_capability() -> Option<SessionForkCapability>`
- `Engine::fork_session(&SessionForkCapability, source, SessionPosition) -> SessionId`

`SessionCheckpoint` is a small public token-boundary handle:

```rust
pub struct SessionCheckpoint {
    pub session_id: SessionId,
    pub position: SessionPosition,
}
```

`SessionPosition` and `RewindTokenCount` are newtypes, not bare `usize`, so
callers must opt into absolute-vs-relative token-boundary conversion.

`fork_session` is capability-gated. Current backends return `None` from
`session_fork_capability`, so unsupported engines cannot be asked to fork through
the typed API. The internal implementation still fail-closes if reached, rather
than silently deep-copying KV or aliasing mutable decoder state.

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

## Fork design and current gate

The low-level paged KV cache already supports CoW fork:

- fork is `O(number_of_prefix_pages)`;
- no tensor payload is copied at fork time;
- each retained page's refcount is incremented;
- the first divergent write to a shared page allocates and copies that page only.

However, engine sessions also contain decoder runner state (`DecodeState`), and
that state is not generally cloneable or importable today. Enabling `fork_session`
without a full runner-state story would create one of two bad outcomes:

- a deep-copy fork that violates the promised CoW cost model; or
- an aliasing fork where parent and child mutate the same decoder KV buffers.

For that reason `Engine::session_fork_capability` currently returns `None`. The
API is present so REPL/server work can target the intended surface, but no
backend is advertised as fork-capable until it can satisfy the invariants.

## Backend support matrix

| Decode path | Rewind | Fork |
|---|---|---|
| Native single-session backend | Not supported by persistent-session APIs; `require_ort_backend` rejects it | Not supported |
| ORT static-cache runner | Not supported; rejected before session mutation until runner rewind has a transactional prepare/commit path | Not enabled; runner state is not cloneable/importable |
| ORT shared-buffer GQA / PastPresent runner | Not supported; rejected before session mutation until ORT PastPresent rewind can be prepared without mutating shared buffers | Not enabled; mutable shared buffers cannot be aliased safely |
| ORT sliding-window past/present | Supported only to retained positions; evicted gaps reject cleanly before session mutation | Not enabled; fork positions before retained start must reject, and state cloning is unresolved |
| ORT materialized paged KV without runner | Supported when paged KV metadata exists; ORT-owned KV without paged materialization rejects cleanly before session mutation | Low-level KV can CoW, but engine-level decode state still needs safe reconstruction/import |
| Draft/speculative session state | Rewound alongside target to the aligned prefix | Not enabled |

## REPL integration notes

The REPL should call `checkpoint_session` at turn boundaries and use
`rewind_session_to` / `restore_session` for `/undo-turn` or named checkpoint
restore. `/fork` should first check `session_fork_capability`; with current
backends it must report unsupported until a later engine slice enables a backend
in the matrix above.

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
is single-threaded page-table reference counting with exact runtime assertions
and proptest coverage; the engine-level fork is still disabled. A TLA+ model
becomes worthwhile when enabling a real fork backend with runner import/clone
ordering or cross-session prefix sharing beyond the current single-threaded
`Rc`-style page accounting.

Follow-up: enable runner-backed rewind only after ORT static-cache and
shared-buffer/PastPresent runners expose a prepared rewind whose validation
prebuilds all fallible artifacts and whose commit cannot fail after logical
tokens or paged KV are truncated.
