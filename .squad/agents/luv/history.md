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
