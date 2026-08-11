### 2026-08-07: PR #728 round-3 re-review
**By:** Harry
**Verdict:** REJECT

## Blocking finding

1. **Finding 1 is fixed only for the enumerated elementwise family; the claimed unification closure is not complete.** `op_broadcasts_elementwise` admits only the default-domain elementwise/comparison/variadic/`Where` list (`crates/onnx-runtime-session/src/executor/kernel_cache.rs:320-348`), but other shape handlers invoke the same representative-selecting broadcast logic. In particular, `MatMul` broadcasts its batch dimensions and writes that result into the output (`crates/onnx-runtime-shape-inference/src/handlers/linalg.rs:157-180`), using `broadcast_dim`'s lower-ID symbolic representative (`crates/onnx-runtime-shape-inference/src/context.rs:468-482`). Therefore `[seq_kv, M, K] @ [batch, K, N] -> [batch, M, N]` can erase the raw growing symbol from the output, and a downstream pointwise consumer carrying only `batch` is still classified capture-safe. I reproduced this on `817eee53`: a focused MatMul-alias consumer probe failed because the closed set remained `{SymbolId(1)}` and omitted `batch`. `Einsum` also calls `broadcast` for ellipses (`handlers/einsum.rs:121-125`); `Expand` and `Concat` directly call `broadcast_dim` (`handlers/movement/transform.rs:354-407`, `handlers/movement/concat_slice.rs:62-80`). The closure must cover every shape-inference path that can substitute a growing symbol into another representative, not only elementwise ops.

## Confirmed closed / sound

- Within the enumerated elementwise family, the union-find is transitive across arbitrary alias chains (`kernel_cache.rs:360-398`). Symbolic-vs-`1` correctly preserves the symbolic ID, while symbolic-vs-static-non-`1` produces the static extent rather than another symbolic representative (`context.rs:437-442,464-467`), so those cases need no union.
- CSA output 5 is `[query[0], index_heads, query_seq, selections]`; `selections` is its last axis (`handlers/custom_ops.rs:181-195`). `last_axis_outputs = &[5]` and collection at `kernel_cache.rs:99-111,210-218` close Finding 2.
- Both new tests pass on HEAD. The downstream-alias test asserts the consumer, not merely the aliasing op (`executor/tests.rs:958-990`), and I verified it fails on pre-closure commit `f8876baa`. The CSA test consumes output 5 and asserts eager (`executor/tests.rs:1077-1165`).
- Moving the alias case into its dedicated regression does not remove the prior pinned/direct-growing coverage.

**Revision owner:** Batty should revise; Cohaagen, Deckard, and Leon remain locked out.

**REJECT: symbol-unification closure still omits non-elementwise broadcast handlers, with a reproduced MatMul downstream-consumer escape.**
