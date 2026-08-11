### 2026-08-07: PR #728 C1 capture-classifier review
**By:** Harry
**Verdict:** REJECT

## Blocking findings

1. **The growing-symbol gate is not closed under shape inference, so a growing KV axis can be marked capture-safe.** `node_capture_seq_independent` checks only exact `SymbolId` membership in output shapes (`crates/onnx-runtime-session/src/executor/kernel_cache.rs:147-157`). Elementwise broadcast inference can replace two distinct symbolic inputs with the lower-ID representative (`crates/onnx-runtime-shape-inference/src/context.rs:468-482`). Thus an op broadcasting `[seq_kv, D]` with `[batch, D]` (runtime batch `1`) may emit the `batch` symbol even though its output extent follows `seq_kv`; the new gate returns true and admits a stale launch geometry. The negative test only covers direct propagation of the same KV symbol (`executor/tests.rs:845-855`), so it misses this alias case. At minimum, this pointwise-family classifier must reject when any input or output references a growing symbol, with a regression test for symbol unification/aliasing.

2. **The growing-source collector is incomplete for a supported stateful CUDA op.** It recognizes only GQA, default `Attention`, and `IndexShare` (`kernel_cache.rs:41-59`). `CompressedSparseAttention` derives growing cache-record dimensions from `total_sequence_length`, minting fresh symbols for outputs 1/3/5 (`crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs:115-129,166-190`), but none enter the set. A pointwise op on such a cache-shaped value is therefore classified independent, especially when no recognized attention node exists and the set is empty (`kernel_cache.rs:94-127,143-156`). This contradicts the PR’s all-model safety claim.

3. **The oracle re-anchor no longer guards the changed captured autoregressive path.** The “primary lock” teacher-forces one fixed multi-token context on a fresh engine (`crates/onnx-genai-engine/tests/qwen36_35b_a3b_qmoe_divergence.rs:247-282`); that is prefill and does not exercise the C1 decode replay being shipped. The only autoregressive run accepts and logs *any* token at index 119 (`:284-312`). Therefore an actual captured-decode corruption to 5342, 279, or an unrelated token still passes as long as fixed-context prefill remains correct. The dense and fp32 checks are also optional on artifact presence (`:320-371`). Preserve the #722 allowance, but assert a bounded, independently adjudicated set/invariant for the live captured stream rather than making it wholly non-fatal.

## Confirmed sound parts

- The EP trait addition is default no-op (`onnx-runtime-ep-api/src/kernel.rs:449-475`) and is set immediately after kernel creation, before insertion/warm execution (`kernel_cache.rs:260-283`), so CPU and unrelated kernels remain unaffected.
- The native engine routes only `token_ids.len() == 1` through the capture state machine and sends multi-token input to eager execution (`onnx-genai-engine/src/native_decode/cuda.rs:430-448,481-553`), so the requested seq_len>1 guard is present.
- `git diff --check` passed. Targeted session classifier tests passed (2/2).

**Revision owner:** Deckard should revise; Cohaagen is locked out for this rejected revision.

**REJECT: the classifier can lose or omit growing-symbol dependencies, and the re-anchored test does not fail on captured autoregressive corruption.**
