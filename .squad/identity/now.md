# Team Focus — now

**Current focus:** Roadmap CUDA/CPU parity and performance.

**OVERNIGHT LANDED (14 PRs total):** #239 (#45/#46 CLOSED samplers), #246 (#58 CPU f16/bf16 GEMM), #249 (#68 CLOSED FP8 CSA residency), #248 (#74 ScatterND+QLinearMatMul), #256 (#56 CLOSED 2-bit CPU GEMV/GEMM), #264 (fmt/clippy cleanup), #263 (#67 CUDA ops 89→102), #259 (#71 CLOSED CUDA discovery+Python provider), #265 (#58 CPU GEMM SIMD), #266 (#67 CUDA ops 102→113), #267 (#86 varlen attention), #268 (#51 CLOSED fp16 VAE safety), #269 (#67 CUDA ops 114→117), #270 (#69 conformance profile).

**ISSUES CLOSED:** #45, #46, #68, #56, #71, #51. **ADVANCED (open):** #58, #74, #67, #86, #69.

**REMAINING UNBLOCKED roadmap candidates:** #67 more ops (diminishing — most simple ops now covered), #13 tracing/§31 (server-side, check other-squad server-route conflicts), #58 macOS Accelerate linkage (can't validate on Linux box), #69 remaining conformance depth.

**BLOCKED:** Justin merging Mobius #404/#423/#430 (+#79/#80/#82/#83): GLM/DeepSeek real-weight E2E + CUDA-vs-ORT Foundry-Local benchmark — the CORE remaining deliverable.

**DEFER (other-squad refactor turf, DO NOT TOUCH):** #55/#75/#78/#85/#206/#207/#228/#230 (schema/executor/decode/session).

**Updated:** 2026-07-27T15:41:00+0000
