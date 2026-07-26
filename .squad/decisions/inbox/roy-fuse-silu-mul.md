### 2026-07-26: Fuse CPU SiLU + Mul SwiGLU gate
**By:** Roy
**What:** Added the optimizer rewrite from `com.microsoft::Silu` plus same-shape `Mul` to `com.microsoft::SiluMul`, with a single-consumer guard, plus the registered CPU kernel.
**Why:** Removes the separate graph dispatch and preserves exact standalone SiLU behavior, including MLAS's runtime-gated implementation and f16/bf16 intermediate rounding.
