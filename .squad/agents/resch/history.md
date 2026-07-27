# Resch — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Native CUDA EP beats/parity ORT on several Foundry models; correctness suite green (int8/block32 f64-adjudicated in #190). Team reorganized into pods; CPU & Edge pod formed to broaden hardware coverage beyond CUDA/Metal.
- **Role:** Intel CPU Optimization Engineer — x86-64 CPU-EP perf (AVX2 baseline, AVX-512/VNNI), MLAS interplay, int8 DP4A GEMV.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:05Z — Fixed pre-existing main CI red by rustfmt-formatting `decode_spmd.rs`; direct main commit `1bf119af` unblocked dependent PRs.
