### 2026-08-26T11-00-38: CUDA MHA indices promote before the first multiplication
**By:** Leon
**What:** CUDA MHA indices promote before the first multiplication
**References:** PR #2193, crates/onnx-runtime-ep-cuda/src/kernels/multi_head_attention.rs, commit 17f5b5545
**Why:** Every MHA NVRTC pointer/linear index rooted in int geometry now casts its first operand to long long before multiplication. Casting a completed int product is unsafe because individually valid B and S can have B*S > INT_MAX while usize byte geometry remains valid. Host validation continues to accept such geometry when checked usize products/bytes fit; it does not add an aggregate i32 rejection.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
