# Decision: Fix-forward CUDA test-honesty check for pure-CPU dummy_fill_and_crossover probe

**Author:** Cohaagen (Rust engineer)
**Date:** 2026-08-11
**PR:** #789 (branch `squad/fix-cuda-honesty-dummy-fill`)
**Scope:** `.github/scripts/verify_cuda_test_honesty.py` (CI only)

## Context

While rebasing #784 (DRY decoder-io refactor) onto current main, the
`CUDA compile (Linux x86_64)` check was failing. Investigation showed the
failure is **not** from the refactor and **not** a compiler error — it is a
pre-existing main breakage in the `Verify CUDA test inventory and skip honesty`
step, present on main since **#776** (`237d3654`) and every commit after
(verified on #786 `9e16e83e` and #784 `aaadd971`).

```
CUDA test honesty check failed:
  - dummy_fill_and_crossover: 4 tests executed without gpu-tests; CUDA tests must be ignored, not pass
  - dummy_fill_and_crossover: 4 tests passed with gpu-tests on a no-CUDA host; CUDA tests must fail loud or remain ignored
```

## Root cause

`.github/scripts/verify_cuda_test_honesty.py` requires every test target under
the CUDA crates' `tests/` dirs to be **ignored (or fail loud) on a no-CUDA
host** unless whitelisted in `ALWAYS_RUN`. #776 added
`crates/onnx-runtime-cuda-memory/tests/dummy_fill_and_crossover.rs` — four
**pure-CPU** deterministic probes (safe dummy-fill value via additive-masking
algebra; fixed-stride+dummy vs bucket-growth memory crossover from real model KV
geometry). They issue no CUDA calls and legitimately pass on the CPU lane (the
filename deliberately omits the `_gpu` suffix its device-bound siblings carry),
but were never whitelisted.

## Decision

Add `dummy_fill_and_crossover` to `ALWAYS_RUN`, mirroring the existing carve-out
for `capture_sync_contract` (a pure-CPU static audit that also legitimately
passes on the CPU lane). One-line whitelist plus a justification comment; no
test behavior changes. This is the "genuine CPU-only probe, not a GPU test"
case the script's own docstring already sanctions.

## Impact

Unblocks the `CUDA compile (Linux)` lane for every PR based on current main. The
check was informational (non-required), so it did not block #784's merge, but it
was red on all PRs and should be green.

## Note for CUDA owners

If the intent was instead for these probes to live in a non-CUDA crate (they use
no CUDA), that relocation is a follow-up; whitelisting is the minimal,
behavior-preserving fix that restores a green CPU lane now.
