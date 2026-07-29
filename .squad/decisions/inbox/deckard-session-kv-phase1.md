# Session-Persistent KV Cache — Phase 1 Implementation

**Author:** Deckard  
**Date:** 2026-07-29  
**Status:** Implemented  
**PR:** squad/session-kv-phase1  

## Decision

Implemented Roy's Phase 1 design: remove the unconditional `reset()` from the native
decode session's multi-turn path and add incremental prefill so a continued conversation
only prefills new tokens.

## Key Design Choices

### Cache Invalidation

The API computes `common_prefix_len(session_tokens, new_prompt_tokens)` on every call.
If the new prompt diverges from the cached history, the KV is rewound to the divergence
point via the existing `rewind()` machinery. The *default* behavior is safe: the
stateless `generate()` path still resets unconditionally.

### resume_from Capping

`resume_from = min(prefix_len, native.current_len())` — because the session token history
includes the last generated token which was sampled but never fed through the model.
Without this cap, `resume_from > current_len` triggers an unnecessary full-reset fallback.

### Weight-Transpose Cache Interaction

Phase 1 does not change model/executor lifetime. The global weight-transpose caches
(#353) are keyed by data pointer and cleared on `Executor::drop`. Since one
`InferenceSession` per `Engine` lifetime is preserved, the interaction is nil.

### Single-Session Limitation (Phase 1)

Only one native session is supported. Attempting to create a second fails explicitly.
The stateless `generate()` path remains unchanged.

## Verification

1. `native_session_incremental_matches_stateless` — token-identical output
2. `native_session_rewind_produces_correct_output` — divergent prefix correctness
3. `native_session_creation_guards` — API safety rails
4. `NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS` counter + dispatch_manifest.toml row
