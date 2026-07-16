# Luv — History

## 2026-07-12: Joined
Hired as an additional Code Reviewer (alongside Gaff) as the codebase grew to 9 crates with many concurrent workstreams. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Focus: correctness/safety gates on decode, sampling, KV, concurrency, and API contracts. Strict reviewer-lockout semantics apply on rejection. Validate green with real exit codes; never approve on style alone.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Batty's issue #14 vision token-expansion wiring and rejected multi-image over-count plus missing `tokens_per_tile` guards; Batty was locked out and Leon owned the fix.

## 2026-07-16T00:00:02Z — MatMulNBits GEMV tiling result
- Evaluated four- and eight-column direct-int4 GEMV tiling; both regressed at 24 and 96 threads because of register pressure, spills, and non-contiguous packed-weight streams.
- Reverted the experiments and documented the negative result in `79c52a6`; the one-column GEMV remains the production path.
