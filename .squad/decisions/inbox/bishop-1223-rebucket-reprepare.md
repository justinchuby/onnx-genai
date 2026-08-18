### 2026-08-18: #1223 — workspace re-prepare on rebucket

**By:** Bishop

**What:** Gap is REAL and now fixed. Rebucketing did NOT re-prepare governed
workspace: `prepare_with_device_bindings` runs once per generation and latches
`workspace_preparation_required`, after which `execute_kernel` refused to
(re)allocate a prepared slot. Within one shape bucket #1221 made that safe; a KV
growth to a new bucket (or a prompt in a different bucket than its decode steps)
left the reserved `SessionPersistent`/`StepScoped` slot absent or undersized,
reproducing #1221's two failure modes cross-bucket ("workspace invariant
mismatch" / "reached execution without prepared workspace"). Fix: allow a
prepared session to re-prepare (grow) its governed workspace slot **on eager
(non-capture) dispatch** — exactly the dispatch a rebucket forces (the growing-KV
decode path declines capture; a capture-eligible model re-warms eagerly after the
KV-growth graph invalidation before it re-captures). Growth stays forbidden while
recording a captured segment, so a replayed graph's baked workspace pointer is
never invalidated under it. The change lives on the shared executor workspace
path (`dispatch.rs::execute_kernel`), so it is general to every
governed-workspace operator, not special-cased to `Attention`.

**Why:** `Attention`'s route-dependent lifetime classification (decode →
`SessionPersistent`, prefill → `StepScoped`) is what makes it hit this first, not
what makes it unique; the correct fix generalizes #1221 rather than adding another
Attention-specific prepare pass. Gating growth on the eager disposition keeps the
prepared-workspace invariant intact for capture/replay while letting the one safe
point (the rebucket re-warm) re-prepare.

**Evidence:** New executor unit test
`prepared_session_reprepares_workspace_when_execution_rebuckets` reserves a 2-row
`SessionPersistent` slot via prepare, then executes a 4-row bucket. Reverting the
one-line guard fails it with `workspace invariant mismatch: execute requires 4096
bytes aligned to 512, prepared 2048 bytes aligned to 256`; with the fix it grows
in place and passes. `cargo test -p onnx-runtime-session --lib` = 186 passed;
clippy + rustfmt clean.
