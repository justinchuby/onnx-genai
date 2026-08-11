# Luba — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Native CUDA EP beats/parity ORT on several Foundry models; correctness suite green (int8/block32 f64-adjudicated in #190). Team reorganized into pods; CPU & Edge pod formed to broaden hardware coverage beyond CUDA/Metal.
- **Role:** ARM CPU / QNN EP Engineer — ARM64 CPU (NEON/SVE) perf + Qualcomm QNN NPU execution provider, edge/Windows-on-ARM.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.
## 2026-07-27T04:35:00-07:00 — Scribe update: Mac CPU EP PR #227

- Native Mac CPU EP now has Apple-Silicon-general NEON paths for multi-thread GEMV, SDPA, SiLU, and direct-from-mmap FP16 GEMV; runtime feature detection/dispatch is expected for SIMD paths instead of machine-specific tuning.
- FP16 works because Apple Silicon NEON can widen f16 loads directly while ORT CPU widens before GEMM; keep this architectural distinction in mind for CPU EP work on other platforms.
- The campaign learned that untested SIMD paths are as risky as placeholders; new AVX/NEON/SVE/QNN paths need guard-break tests and paired scalar/reference checks.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Fixed PR #294 aarch64 build by cfg-gating the x86-only perf probe after Drake lockout.

## 2026-08-11 — B3: NxrtStatus cross-module allocator fix (PR #762 rejection)

**Problem:** `NxrtStatus.message` was heap-allocated in the plugin (`CString::into_raw`) and freed in the host (`CString::from_raw`/`Drop`). Across a `cdylib` boundary with different CRTs this is UB (Windows heap corruption).

**Fix:** Replaced `*mut c_char` with inline `[u8; 256]` buffer + `message_len: u32`. `NxrtStatus` is now a pure value type — no heap, no pointers, no `Drop`, no cross-module free. `message_str()` is no longer `unsafe`.

**Also fixed:** Two `as *const i8` casts in `loader.rs` and `provider_adapter.rs` that fail on aarch64 (where `c_char = u8`). Changed to `as *const c_char`.

**Tests:** 32 nxrt-abi unit tests pass, 4 nxrt-host unit + 10 roundtrip tests pass. Clippy + fmt clean.

**Note for Chew:** Two `as *const i8` casts remain in `tests/nxrt_abi_roundtrip.rs:173,187` — need the same `c_char` fix.
