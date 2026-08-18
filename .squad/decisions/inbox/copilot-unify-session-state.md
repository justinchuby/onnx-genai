### 2026-08-18: Session-state policy is unified behind one `SessionStore` seam (PR #1255)

**By:** Coding agent (spawned by Squad coordinator), on the owner's standing
"no 区别对待" directive.

**What:** `crates/onnx-genai-engine/src/engine/session_state.rs` now owns the
backend-independent session policy (lookup + "session {id} not found" text, the
rewind bound-checks and their exact strings, checkpoint arithmetic). Native and
ORT both route the six public session methods through it via `SessionStore`
adapters in `runtime.rs`, exactly mirroring the `KvPrefixStore` precedent from
#1170. Error strings are authored once; rendered text is byte-identical.

**Guard (do not remove):** two-part DRY guard, like `ReusedPrefix` +
`KV_REWIND_CALLERS`. (1) compile-time `CheckedPosition` newtype — the only way to
obtain a rewind target is through the one shared bound check; (2) test-time
tripwire `the_rewind_bound_check_lives_only_in_the_shared_policy` fails if
`"cannot rewind session"` is open-coded outside `session_state.rs`. A future
third backend must implement `SessionStore`, not copy the policy. Do NOT widen
the tripwire allowlist to get a green test.

**Asymmetries deliberately kept:** `create_session` (object construction, not
policy); ORT `validate_rewind` validates draft/target/paged-KV while native's is
`Ok(())` (persistent in-process decoder, always admits); ORT token `truncate`
stays inside `rewind_target_state_to_len` because it is shared with the
speculative-decode hot path — the shared policy owns the bound check, not the
truncation.

**Why it is trustworthy:** falsified — inverting the single shared bound check
turns BOTH backends red (3 ORT `failed_rewind_of_*` + native
`native_session_rewind_by_truncates_logical_length`, a new test added to close a
real coverage gap); reverting restores green. Verified: 414 lib tests pass,
clippy clean (default + native), native-backend suite green except a pre-existing
host-RAM KV-budget test, and CUDA (RTX 4060) native_engine ran on-GPU 16 passed /
1 pre-existing unrelated failure.

**Status:** PR #1255 open, not merged, awaiting owner review.
