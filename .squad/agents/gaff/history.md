# Gaff — History

## Project context
- Review specialist for onnx-genai correctness, runtime/loader boundaries, transactional semantics, and validation quality.
- Joined 2026-07-12 after phases 1-4, tool use/grammar/chat-template, Qwen2.5-0.5B, Hermes E2E, and static-cache KV work were established.

## Condensed prior record through 2026-07-28
- Reviewed and approved multiple ORT2 loader, shape-inference, fused-domain, EPContext, C-ABI, external-data, and conformance changes after checking byte fidelity, dispatch invariants, path confinement, FFI behavior, and model-backed tests.
- Used reviewer lockout discipline on real blockers: unauthenticated debug exposure, MatMul+Add fusion shape guard gaps, duplicate EPContext primary identity over-rejection, unsupported-op user-facing opset leakage, and thread/benchmark provenance issues.
- Helped consolidate performance and CUDA/native guidance: benchmark comparisons must be matched and reproducible; CUDA EP kernel work must remain correct across supported SM architectures rather than only sm_90.
- Recorded CLI maintainer-tool backlog context: the CLI is a development/maintainer harness, not a consumer product; `docs/research/cli/00-backlog.md` is the backlog source of truth.
- For PR #289, rejected REPL Phase 1 when non-TTY parsing drifted from the plain path, then approved Zhora's revision after parser parity and newline lifecycle fixes.
- For PR #291, owned the runner-backed rewind boundary after Leon's deeper rejection; runner-backed static-cache/shared-buffer rewind now rejects during validation until a transactional prepared rewind exists.
- Authored PR #321 for issue #63, landing Phase-3b live GPU weight-offload allocation, H2D copy, binding, and byte-identical one-weight device execution.
- Recent consolidated decisions remain authoritative in `.squad/decisions.md`; this history was summarized by Scribe because it exceeded the 15KB threshold.

## 2026-07-28T17:40:00+0000
Reviewed #364: blocked the partial-cache stat divergence, then approved the guarded fix.

## 2026-07-29T12:30:00Z — tiny-reasoning-fixture REJECT (PR #410)

Mutation proof on Batty's PR #410 sampling fix: commented out `resolve_sampling_defaults`
per-turn, recreating the #385/#392 forced-greedy bug. Suite stayed green. Issued REJECT;
Batty locked out of `test/tiny-reasoning-fixture`; Leon took ownership for round 2+.

Durable rule recorded: "Assert on what the code did, not on a summary of what it should
have done." (`.squad/decisions.md`, reconstructed rules section, 2026-07-29)

Separately: confirmed rubber-duck's empty-answer diagnosis — `quick --greedy
--max-new-tokens 3` stopping on `</think>` committed an empty assistant turn while
`manifest.json` asserted the non-emptiness invariant.

Inbox drop `gaff-review-reasoning-fixture.md` was lost when the worktree was deleted
before Scribe ran; content reconstructed into `.squad/decisions.md`.

## 2026-08-11 — env var verifier: filename-reference false-positive fix

CI job 93646716235 failed because `verify_documented_env_vars.py` matched
`NXRT_ABI` from the filename `docs/NXRT_ABI.md` referenced in
`EP_PLUGIN_EXPORT_PR.md`. Added a negative lookahead `(?!\.(?:md|rst|toml)\b)`
to `ENV_PATTERN` so documentation cross-references are not mistaken for
environment variables. Updated the module docstring to explain the exclusion.
Re-proved the gate still catches genuinely undocumented variables (`NXRT_FAKE_KNOB`
experiment passed).

## 2026-08-11 — PR #31973 Review: AVX2 LayerNorm/RMSNorm kernel

Reviewed microsoft/onnxruntime PR #31973 (branch `nxrt/mlas-avx2-layernorm`).
Independently verified: 40/40 tests pass, AVX2+FMA dispatch guard correct,
Welford pairwise merge formula matches textbook, no alignment/UB issues,
precision claims confirmed (AVX2 Welford genuinely more accurate than scalar).
No blocking findings. Two substantive items: (S1) unnecessary sum accumulation
in RMSNorm path when MeanOut is null, (S2) test reference variance formula
comment clarity. Decision written to `decisions/inbox/gaff-review-pr31973.md`.

## 2026-08-11 — Independent Re-Review of PR #762 (B1-B4 corrective wave)

- **Task:** Adversarial re-review of PR #762 after B1–B4 blocker fixes
- **Verdict:** All four blockers resolved. No new blocking findings. 2 nits.
- **B1 (output dtypes):** ✅ Resolved. `CompiledKernelEntry.output_dtypes` sourced from graph. LayerNorm shapes correct.
- **B2 (ReleaseEpFactory void):** ✅ Resolved. Returns `*mut OrtStatus` in macro, both shims, and ABI test.
- **B3 (NxrtStatus cross-allocator):** ✅ Resolved. Inline `[u8; 256]` buffer, no heap, no `c_char`.
- **B4 (CUDA fail-open):** ✅ Resolved. Zero factories unconditionally, `CanCopy` returns false.
- **Tests:** 245 passed, 0 failed from clean state. Cast/Where/Shape tests assert real dtypes and values.
- **Output:** `.squad/decisions/inbox/gaff-rereview-pr762.md`

## 2026-08-11 — Re-review PR #762 (post-Sapper fix, commits 2ca515eb7..7aba5cb93)

- **Request:** Verify all four B4 defects are genuinely fixed.
- **Verdict:** 3 BLOCKING, 4 SUBSTANTIVE, 2 NITS. **Not ready to leave draft.**
- **B1 (CRITICAL):** Shared EP raw pointer is dangling — `MutexGuard` drops at end of `if let` block, pointer becomes use-after-free. Affects allocator, stream, and data transfer. (factory.rs:586-592, 686-692, 747-753)
- **S4 (SHOWSTOPPER):** Constructor panic bomb is called unconditionally by `create_ep_factories` (factory.rs:152) to read EP name. On a real CUDA host, `CreateEpFactories` would always panic→fail. The claimed "1 factory on GPU" path is unreachable.
- **B3:** `CopyTensors` wraps both src and dst as device buffers regardless of direction. H→D/D→H would pass host pointers to `cudaMemcpyDeviceToDevice`. (transfer.rs:608-660)
- **B2:** `CanCopy` same-device uses pointer equality on opaque `OrtMemoryDevice*`. D2D on same GPU would fail closed.
- **Defect 1 (shared runtime):** NOT fixed (dangling pointer + unreachable path).
- **Defect 2 (CreateDataTransfer):** PARTIAL (struct correct, CopyTensors wrong).
- **Defect 3 (GetHandle):** PARTIAL (handle stored, but EP pointer dangling).
- **Defect 4 (Free size):** YES (design correct).
- **Key answer:** Code fails closed by accident (panic bomb), not by design. Would crash if panic bomb were removed.
- **Tests verified:** 18/18 pass in targeted crates. Clippy pre-existing failure in `onnx-genai-engine`.
- **Output:** `.squad/decisions/inbox/gaff-rereview-762-cuda.md`

## 2026-08-11 — Verify #762 CUDA EP Fix (d64a49d59)

- **Task:** Verify Nabil's revision against B1/B3/S4 blockers from previous rejection.
- **B1 (UAF):** GENUINELY FIXED. `EpRef::Shared(Arc<Mutex<..>>)` — no raw pointers from guards.
- **B3 (copy direction):** GENUINELY FIXED. `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType` classifies; dispatches correctly.
- **S4 (panic bomb):** GENUINELY FIXED. `create_ep_factories_for_shared_ep` takes name directly.
- **B2 (pointer equality):** Deferral justification is WRONG — `MemoryDevice_GetDeviceId` exists in ORT 1.27 bindings. Fail-closed but functionally broken for D2D same-device.
- **New issue:** Mutex held across `cudaStreamSynchronize` serializes all operations (perf, not correctness).
- **Tests:** 16/16 pass. B1 test is non-vacuous; B3 tests classification only; S4 test is marginal.
- **Verdict:** Conditional pass — fix B2 (5 lines) before leaving draft, or accept D2D-same-device is broken.
- **Output:** `.squad/decisions/inbox/gaff-verify-762-cuda-fix.md`
