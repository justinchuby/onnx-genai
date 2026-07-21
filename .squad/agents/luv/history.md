# Luv — History

## 2026-07-12: Joined
Hired as an additional Code Reviewer (alongside Gaff) as the codebase grew to 9 crates with many concurrent workstreams. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Focus: correctness/safety gates on decode, sampling, KV, concurrency, and API contracts. Strict reviewer-lockout semantics apply on rejection. Validate green with real exit codes; never approve on style alone.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Batty's issue #14 vision token-expansion wiring and rejected multi-image over-count plus missing `tokens_per_tile` guards; Batty was locked out and Leon owned the fix.

## 2026-07-16T00:00:02Z — MatMulNBits GEMV tiling result
- Evaluated four- and eight-column direct-int4 GEMV tiling; both regressed at 24 and 96 threads because of register pressure, spills, and non-contiguous packed-weight streams.
- Reverted the experiments and documented the negative result in `79c52a6`; the one-column GEMV remains the production path.

## 2026-07-16T00:00:00Z — CUDA M2 op-coverage delivery
- Landed `16c1e92`: f32 `com.microsoft::Silu` and standard-domain `ai.onnx::SimplifiedLayerNormalization` CUDA registrations, matching CPU EP coverage.
- Holden cleared independent parity checks; the CUDA suite passed 114/114.
- 2026-07-19T07:55:00Z: Approved PR #32 after capability, half-argmax, options-forwarding, retained-integration, and CI verification.

- 2026-07-19T12:40Z: Re-verified Bryant's conformance refresh counts (875 pass / 890 fail / 1,765 CUDA skip) and approved the measurement-only update.
## 2026-07-19T14:10Z — Bitwise/Hardmax review cycle
- 🔴 Rejected Pris's `43df6c0`, locking Pris out, then 🟢 approved Deckard's `7fe8961` revision for fp16/bf16 Hardmax and genuine bitwise broadcast/rejection coverage.


- **2026-07-19T16:15:00Z — CPU-EP review:** Rejected activation f32-only and f64-narrowing implementations, then approved Sapper’s true-f64 correction; activations landed as `39edb76`.


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Approved AffineGrid/Col2Im/CenterCropPad (`8e49948`) with a non-blocking Col2Im dilation-test nit.

- 2026-07-19: Drove Unique through three rejection cycles: O(n²)/NaN/dtype shortcomings, unreachable String execution, then runtime-layer String UB. Approved after unsafe String handling was removed; final kernel supports safe numeric/bool/bf16 and reports String unsupported.

## 2026-07-19T21:30:00Z — oneDNN removal review
- 🟢 Approved Bryant's `453d280` oneDNN CPU GEMM removal after verifying clean references/submodule removal, 620 CPU-EP library tests, 28 tracer tests, and registry-count integrity.
- 25 clippy lints observed remain pre-existing.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Cross-agent update: vendored MLAS is now the opt-in CPU-GEMM parity path; follow-ups include buffer reuse, prepacked B, dtype coverage, int4, default flip, and Windows MASM.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Approved Sapper’s decode-pool residency and Roy’s guarded GQA parallelism after concurrency, numerical-order, opt-out, feature-gate, and E2E parity checks.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T13:15:00Z — GQA metadata fold landed
- Folded batch-1 GQA metadata into fused prep, removing 24 launches/token while preserving bounds/latch semantics and exact tokens. Holden approved; merged as `bd30e6c`, moving the stack from ~691 to ~710 tok/s at 256.
