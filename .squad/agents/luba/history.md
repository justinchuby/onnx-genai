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
