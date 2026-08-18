# Rachael — History (compacted 2026-08-12)

**Role:** CLI/server/API implementer for onnx-genai. Owns OpenAI-compatible server behavior, REPL/maintainer-tool UX, endpoint routing, streaming/session invariants, and user-visible runtime controls while preserving non-TTY byte stability.

## Durable lessons
- Server surface includes `/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming, `X-Session-Id`, session lifecycle routes, tools/tool_choice/tool-role handling, JSON response constraints, FIM, image parts, and audio routing.
- Server DoS/session hardening is canonical: `max_output_tokens=4096`, `max_sessions=256` LRU, 128-bit CSPRNG session ids, context-token caps, loopback/no-auth deployment notes.
- Static-cache HTTP concurrency uses a single engine driver thread and channels; do not reintroduce shared Engine locking.
- Batched-driver admission is bounded by `max_pending` with HTTP 429; output delivery must be non-blocking.
- `/v1/debug/*` must be default-off and redact session identifiers.
- Zero-copy mmap initializer borrowing landed; producer-aliasing soundness restrictions added by Zhora.
- Qwen Sigmoid fusion must recognize `Mul(x, Sigmoid(x))`, allocation-free, with multi-consumer negative coverage.
- REPL Phase 1: non-TTY stdin/stdout byte-stable; TTY uses `reedline`; slash parser from declarative registry; `/fork`/`rewind` out of Phase 1.
- TTY output owns exactly one separator newline when generated text lacks a trailing newline.
- EP test hardening: `disable_cpu_ep_fallback=1` + EP assignment assertions; non-vacuity proven by forced wrong-EP assertions.
- `NXRT_REQUIRE_ORT_TESTS` gate must route all skip paths (fixture-missing, dlopen failure) through the gate; `.ok()?` silent skips are forbidden.
- `find_ort_lib_dir` must honour `NXRT_ORT_LIB_DIR` → `CARGO_TARGET_DIR`/debug/build → workspace default with platform-aware library names.

## Historical context (pre-2026-08-11)
Wave coverage through 2026-07-28: CLI UX research, REPL redesign, Phase 1 TTY/plain split with `reedline`, PR #300/#346 lockout revisions, REPL stats two-line block. Full detail in `history-archive.md`.

## Recent entries

## 2026-08-11 — PR #762 final test hardening

**Commit:** `c1d2556b5`

- `layernorm_dynamic_axis.rs`: `disable_cpu_ep_fallback=1` + EP assignment assertion for `LayerNormalization`.
- New conformance tests: `conformance_add_float16`, `conformance_add_bfloat16`, `conformance_layer_norm_multi_output`, `conformance_layer_norm_neg_axis`, `conformance_rms_norm` — all with assignment assertions.
- 14 total EP assignment assertions across two test files.
- BL1 shape assertions (mean/inv_std shape `[2, 3, 1]`) fully preserved. Non-vacuity proved.

## 2026-08-12 — PR #762: Test-integrity gate hardening

- Fixed `find_ort_lib_dir` to honour `CARGO_TARGET_DIR` (matching `cdylib_resolve.rs` logic).
- Routed fixture-missing and dlopen-failure skip paths through the `NXRT_REQUIRE_ORT_TESTS` gate.
- Added `NXRT_REQUIRE_ORT_TESTS=1` to CI `CLI ORT (Linux x86_64)` lane + cpu-plugin test step.
- Test counts: 40 passed, 0 failed. Commit: `88f9de8df`.

## 2026-08-12 — PR #762 ready for review (gate hardening complete)

All five red CI jobs trace to pre-existing `onnx-genai-server` failure on `main`; branch does not touch that crate. 283 passed / 0 failed. Gaff confirmed gate coverage genuinely closed. Freysa completed `find_ort_lib_dir` consolidation. PR #762 marked ready for review.

*Full pre-2026-08-11 history in `history-archive.md`.*

## 2026-08-12 — PR #32001 (microsoft/onnxruntime): Cross-target Accelerate blocker

- Added `parser.error()` rejections for `--android`, `--build_wasm`, `--rv64` alongside `--use_apple_accelerate`.
- Refactored all Apple Accelerate rejection tests to assert diagnostic message content (not bare SystemExit).
- 4 new tests (android, build_wasm, build_wasm_static_lib, rv64). Test count: 13→17.
- Documented Catalyst/macabi CMake limitation: `PLATFORM_NAME` comes from external toolchain, no reliable direct-CMake detection.
- Commit: `184f76a00e`. PR left in draft for Opus review.

## 2026-08-17T16:25Z — PR #1134 GEMV streaming review approved

- Approved `squad/gemv-streaming @ b6a5648c` after verifying the prefetch pipeline preserves lane/depth mapping, fp16 accumulation order, zp/no-zp behavior, dispatch guard safety, and leaves argmax tie-break untouched.
- Reran targeted CUDA validation: bit-identity test 1/1 and GEMV suite 11/11.
- Outcome: correctness/numerics gate green; PR #1134 merged.

## 2026-08-17T17:10Z — Review: gateup-vec SwiGLU bias-fold (PR #1137), 🟢 APPROVE default-ON

- Reviewed `e54cae31` (`matmul_nbits.rs` only). Verdict 🟢 APPROVE; default-ON SAFE.
- Verified the magic-bias fold is exact, not approximate: independently checked all nibble codes 0..15 in fp16 — folded `x-1032` / `fma x*(1/16) -72` produce identical raw fp16 bits to scalar `code-8`. Constants exact fp16 (`0x6408`, `0xd480`, `1/16=0x2c00`).
- Confirmed accumulation order, lane→nibble mapping, RMS prologue, SwiGLU epilogue, and argmax/tie-break unchanged; `_vec` dispatch is symmetric-only and asymmetric `_zp` routes to existing kernels. Reran targeted CUDA tests on idle GPU0: bit-identity test 1/1, `gate_up` suite 10 passed.
- Non-blocking: unit-test comment overstates M>1 as `_vec` coverage (M>1 → prefill/Marlin). Wording noise, not a blocker. Merged as `70cc06ad`.

## 2026-08-17T18:05Z — Review: gateup-occ occupancy-raise (PR #1139), 🟢 APPROVE default-ON

- Reviewed `squad/gateup-rms-stage @ 11a01fae` (`matmul_nbits.rs` only, author Luv). Verdict 🟢 APPROVE; default-ON SAFE.
- Key: `_vec_occ` wrapper bodies are character-for-character identical to their `_vec` parents after the signature — same `matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false,{decomposed},true>(...)` instantiation, same args; only source diff is `__launch_bounds__(256, 8)` + symbol name. No math/reduction/accumulation-order change; body uses explicit fp16 FMA / fixed source-order reductions (not contraction-sensitive), so launch_bounds is storage placement only.
- Dispatch symmetric-only (`occ = !has_zp && gate_up_occ_enabled()`); `_zp` inputs select the existing asymmetric kernel first. Argmax/tie-break untouched. Reran on idle GPU0: bit-identity test 1/1; `gate_up` suite 10 passed, 3 ignored.
- Non-blocking: `_vec_occ` only on RMS-fused decode path → broad test comment slightly overstates coverage (genuinely exercised for eligible M=1 RMS; no-op for non-RMS / M>1). Luv tightened the wording in the default-ON flip. Merged as `0636a759`.

## 2026-08-17T18:40Z — Review: gateup-preperm byte-identical, shelved on perf

- Reviewed `squad/gateup-preperm @ 6629f0aa`; 🟢 APPROVE on byte-identity because hoisted staging supplies exactly the old post-permute bits, preserving RMS reduction, fp16 FMA accumulation order, symmetric-RMS-only dispatch, and argmax behavior. Wallace safety was GREEN, but perf was NO-GO, so coordinator shelved/not merged.
## 2026-08-18T00:35Z — V2-Lite planner/oracle gates closed

- Issued the initial 🔴 on the silent DeepSeek-V2-Lite golden move, then 🟢 accepted the oracle-policy reframe with four conditions, then 🟢 approved the final combined planner + QMoE oracle artifact.
- Established precedent: int4 GEMV/QMoE CPU bit-identity is not the oracle when CPU and CUDA use different f32 accumulation orders; use f64-bounded evidence plus deterministic backend output and explicit golden rationale.
- Final merge: PR #1150 landed on `main` at squash `e075a715`; reviewer lockout was honored throughout.


## 2026-08-18T03:15Z — QMoE/classifier/planner review gates closed

- Approved Luv's default-OFF QMoE occupancy lever (#1167) with a non-blocking scope caveat.
- Rejected Deckard's first V2-Lite mask classifier for two safety blockers and enforced reviewer lockout; re-approved Wallace's tightened revision after present-KV and root graph-output negative tests.
- Approved Leon's `_d1` additive-mask query-axis workspace-planner fix (#1181) as exact and fail-closed; Wallace's hardware A/B then closed the DeepSeek MoE performance mandate at 1.79× capture/eager with 0.000% divergence.
