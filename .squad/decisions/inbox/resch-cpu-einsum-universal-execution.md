### 2026-09-03: CPU Einsum executes the semantic plan exhaustively
**By:** Resch
**What:** CPU Einsum consumes `EinsumSemanticPlan` directly. Optimized mode keeps compatible view/reduction/MatMul routes, executes bounded exact-DP or deterministic-greedy contraction trees, and falls back to GenericNative; `generic-native` forces the universal index program. Typed execution-local scratch follows the planner precision policy and is retained only through the session-owned governed TLS pool.
**Why:** ONNX legality is independent of fast-path shape or operand count. A canonical scalar/index fallback makes every schema-valid expression executable while preserving fixed reduction order, alias-safe publication, opset-gated BF16, wrapping fixed-width integers, and bounded process/thread retention.
