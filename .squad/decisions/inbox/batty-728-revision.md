### 2026-08-07: Path-B authoritative symbol-unification record closes the growing-set over ALL broadcasting handlers (PR #728, round-4)

**By:** Batty

**What**

Harry's round-3 REJECT (Finding 1, partial) showed Leon's executor-side growing-set
closure (`close_growing_under_symbol_unification`) only covered **elementwise**
broadcast ops. But shape inference substitutes one symbol for another (keeping the
lower-id representative, discarding the other's identity) in every handler that
broadcasts — MatMul batch dims (`linalg.rs:175`), Einsum ellipsis (`einsum.rs:123`),
Concat non-concat axes (`concat_slice.rs:79`), Expand (`transform.rs:406`) — all of
which funnel through the single `InferenceContext::broadcast` → `broadcast_dim`
chokepoint. Harry reproduced an escape at the MatMul batch broadcast
(`[seq_kv,M,K] @ [batch,K,N] -> [batch,M,N]` erases growing `seq_kv` into pinned
`batch`; a downstream consumer carrying only `batch` wrongly stayed capturable →
silent decode corruption on CUDA-graph replay). The root fragility is that the
executor **re-implemented a partial copy** of inference's unification and had to be
kept in sync with every handler — the enumeration drift that caused 3 reject rounds.

**Chosen: Path B (authoritative source, complete-by-construction).** Shape inference
now RECORDS every symbol unification it performs, at the single chokepoint, and
persists it onto the `Graph`. The executor reads that authoritative map and closes
its growing set over it — covering MatMul, Einsum, Concat, Expand, elementwise, and
any FUTURE handler, with **zero enumeration in the executor**. This deletes the whole
"did we mirror every handler" bug class.

Changes (file:line):

- `crates/onnx-runtime-ir/src/graph.rs:31` — added `pub symbol_unifications:
  Vec<(SymbolId, SymbolId)>` to `Graph` (additive; `Graph` is `Default`/`Clone`; no
  external struct-literal constructors → safe).
- `crates/onnx-runtime-shape-inference/src/context.rs` — `SymbolInterner` gained a
  private `unifications: Vec<(SymbolId,SymbolId)>` (init in `new()`),
  `record_unification(&mut self, a, b)`, and `pub fn unifications(&self)`. In
  `broadcast_dim`, the `(None,None) => (Some(sa),Some(sb))` arm (context.rs:507) now
  calls `record_unification(sa, sb)` **before** returning the unchanged representative
  (`if sa.0 <= sb.0 { a } else { b }`). ONLY this two-distinct-symbolic branch records;
  symbolic-vs-1 and symbolic-vs-static-non-1 (context.rs:437-467) do NOT union
  (requirement 4). Additive → inference output byte-identical.
- `crates/onnx-runtime-shape-inference/src/infer.rs:180` — `infer_graph_scoped`
  overwrites `graph.symbol_unifications = interner.unifications().to_vec()` after the
  `fresh_symbols` registration loop (overwrite, not append: each run is a complete pass).
- `crates/onnx-runtime-session/src/executor/kernel_cache.rs` — rewrote
  `close_growing_under_symbol_unification` to build the `SymbolUnionFind` from
  `graph.symbol_unifications` (early-return if empty), mark every class containing a
  growing member growing. **REMOVED** `op_broadcasts_elementwise` and
  `broadcast_dim_from_right` (the partial re-implementation). Updated source-3 doc in
  `compute_capture_growing_symbols` and embedded the authoritative drift-detection grep:
  `grep -rn "\.broadcast(\|\.broadcast_dim(" crates/onnx-runtime-shape-inference/src/handlers/`.

Tests:

- `crates/onnx-runtime-session/src/executor/tests.rs` — NEW
  `matmul_batch_alias_keeps_downstream_consumer_eager` (Harry's exact escape): `past_key`
  mints growing `seq_kv`; MatMul `[seq_kv,8,16]@[batch,16,32]->[batch,8,32]` (batch
  created first → lower id → surviving rep); Sigmoid consumer sees only `batch`. Runs
  REAL inference, asserts growing ⊇ {`seq_kv`,`batch`} and consumer stays eager. Updated
  Leon's `growing_symbol_alias_keeps_downstream_consumer_eager` to make its inputs graph
  inputs and run real inference so the record→close path is exercised end-to-end. Kept
  Leon's CSA output-5 (`last_axis_outputs=&[5]`) test unchanged.
- `crates/onnx-runtime-shape-inference/tests/graph_inference.rs` — NEW `unifies` helper +
  `broadcast_records_symbol_unification_for_matmul_batch_dims` and
  `..._for_concat_non_concat_axes` proving non-elementwise handlers record unifications
  AND that inferred output shapes are byte-identical (lower-id rep pinned).

**Pre-fix / post-fix confirmation:** stashed the 4 source files back to HEAD `817eee53`
(keeping the new test), ran `matmul_batch_alias_keeps_downstream_consumer_eager` — it
FAILS: `the representative batch ... must be in the CLOSED growing set ... got
{SymbolId(1)}` (only `seq_kv`; the elementwise-only closure ignores the MatMul alias).
`git stash pop` restored the fix → test PASSES.

**Why**

Path B directly fixes Harry's stated *root* fragility instead of patching symptoms.
`broadcast_dim` is the ONE place inference collapses two distinct symbols to a single
representative, so recording there is complete by construction — no per-op list to
drift. Justin strongly prefers general/DRY/drift-proof solutions over per-op
enumeration. Recording is purely additive (a `Vec` push; representative unchanged), so
inference stays byte-identical — verified by the full shape-inference suite (275 op_rules
+ 41 graph_inference + others) all passing. Path A (extend the executor union-find to 4
more handlers) was rejected: it would re-introduce exactly the enumeration the last 3
rounds proved fragile.

**No over-poisoning:** cross-symbol unions are only recorded on genuine aliases (two
DISTINCT symbols meeting at a broadcast axis). Real transformer MatMul/Concat batch dims
broadcast LIKE symbols (batch↔batch), which are `sa==sb` and skipped. So the growing set
only grows where a growing symbol was genuinely erased into another representative —
which SHOULD poison. Expect growing-set to stay ~11 on 35B-A3B and the 34-segment
collapse to hold. NOT measured locally (35B oracle is ~13 min/config and GPU-bound) —
coordinator commands below.

**Validation run locally (all pass):**
- `cargo fmt --all` + `cargo fmt --all --check` — OK
- `cargo clippy -p onnx-runtime-ir -p onnx-runtime-shape-inference -p onnx-runtime-session --all-targets` — clean
- `cargo clippy -p onnx-runtime-session --features cuda --all-targets` — clean
- `cargo test -p onnx-runtime-ir` — 67 pass
- `cargo test -p onnx-runtime-shape-inference` — all pass (byte-identical inference)
- `cargo test -p onnx-runtime-session` — all pass (0 failed)
- `cargo test -p onnx-runtime-session --features cuda --lib` — 144 pass (incl. the new MatMul test + device_binding cuda-graph tests)

**Coordinator: EXACT commands for final 35B oracle + segment/growing-set validation
(shared GPU — I did NOT run these):**

```bash
# 1) Growing-set size (expect 11) + segment count (expect 34) on 35B-A3B QMoE:
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_LOG_GROWING_SYMBOLS=1 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo run --release -p onnx-runtime-session --features cuda --bin profile_native -- \
  --pipeline --backend native --ep cuda --steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8 -t0

# 2) Byte-exact vs fp32 oracle — capture ON and OFF, both oracle configs:
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle -- --nocapture
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_hybrid_continuation_matches_fp32_oracle -- --nocapture
# repeat both with ONNX_GENAI_CUDA_GRAPH=0 to confirm capture-OFF parity.
```

Do NOT merge.
