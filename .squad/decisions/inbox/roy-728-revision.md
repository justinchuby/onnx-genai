### 2026-08-07: PR #728 round-5 — make the growing-symbol classifier an authoritative HARD VETO for CUDA-graph capture

**By:** Roy

**What**

Harry's round-5 REJECT: the build-time growing-symbol classifier's verdict was
NOT enforced as a hard veto. `capture_shape_eligible` OR-ed the classifier bit
with two runtime-extent fallbacks, so a node the classifier CORRECTLY
disqualified could still be admitted to capture and silently corrupt decode.

**Exact change** — `crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs`:

Before (HEAD `0fd87df3`, elementwise.rs:1058-1059):
```rust
pub(crate) fn capture_shape_eligible(seq_independent: bool, shape: &[usize]) -> bool {
    seq_independent || shape.iter().product::<usize>() == 1 || is_fixed_decode_shape(shape)
}
```
After (elementwise.rs:1061-1063):
```rust
pub(crate) fn capture_shape_eligible(seq_independent: bool, _shape: &[usize]) -> bool {
    seq_independent
}
```
- Dropped the two OR terms (`product == 1`, `is_fixed_decode_shape`).
- Deleted the now-unused `is_fixed_decode_shape` helper (was only referenced by
  the OR term; removing it keeps clippy dead-code clean).
- Kept the fn signature (`shape` → `_shape`) so all callers are unchanged:
  `bitwise.rs:369`, `pointwise.rs:796`, `prelu.rs:203`,
  `elementwise.rs:540,812,947` (all pass `self.capture_seq_independent`), so the
  central fix covers every pointwise/bitwise/prelu/binary/silu-mul caller.
- No runtime fallback retained (see safety argument below).

**Why — `false` is definitively unsafe in BOTH classifiers, so the OR terms are
pure hazard.** `seq_independent` is now `node_capture_seq_independent(graph, node,
set)` = every output symbol provably pinned (FAIL-SAFE default) / no output symbol
proven growing (DENYLIST). In BOTH, `seq_independent == false` is a definitive
"not provably safe to capture". The `|| product==1 || is_fixed_decode_shape` terms
inspect only the *runtime* extents of the current step and can therefore ONLY ever
WRONGLY override a real disqualification — they can never make a genuinely-unsafe
node safe. Concretely, `is_fixed_decode_shape([N]) == true` for ANY rank-1 shape
(empty leading product == 1), so a `Reshape([seq_kv*8]) → Sigmoid` the classifier
disqualified (growing symbol, `seq_independent=false`) was re-admitted, baking a
stale growing extent `[320]` that must be `[328]` next step → silent decode
corruption on replay. The historical `is_fixed_decode_shape` heuristic was a
pre-classifier crutch for single-token decode (token axis symbolic-but-pinned-to-1);
the classifier now covers that case natively — query `seq_len` is treated PINNED
and static feature dims yield `seq_independent=true` — so the fallback adds nothing
but hazard. Preferred strict `seq_independent`-only veto shipped: over-eager is
correctness-safe (at worst a perf regression the coordinator GPU-measures), while
over-capture is corruption.

**No runtime fallback retained; regression reasoning.** The only nodes that were
capturable ONLY via `is_fixed_decode_shape`/`product==1` with `seq_independent=false`
are, by definition, nodes the classifier did NOT prove pinned — i.e. exactly the
nodes that must stay eager for correctness. "Provably non-growing" is precisely
what the classifier computes, so the correct composition is `seq_independent`
alone; any runtime-extent add-back would have to consult the same symbol metadata
(not raw extents) and would then be redundant with `seq_independent`. The residual
concern is graphs where inference never ran and every kernel defaults
`seq_independent=false` (all eager): that is a perf, not correctness, outcome, and
is the safe direction. Coordinator GPU A/B confirms coverage on the real 35B graph.

**New capture-LEVEL regression tests** (`elementwise.rs` `mod tests`, pure-logic,
no GPU required):
- `classifier_disqualified_node_is_never_capture_eligible_regardless_of_shape` —
  asserts `capture_shape_eligible(false, &[320]) == false` (and `[328]`, `[1]`,
  `[1,1,1]`, `[]`, `[320,8]`).
- `disqualified_growing_reshape_consumer_yields_no_capture_signature` — mirrors
  `UnaryKernel::run`'s gating (`capture_shape_eligible(seq_independent, shape)
  .then_some(sig)`) for the `Reshape([seq_kv,8],[-1]) → seq_kv*8 → Sigmoid`
  consumer with `capture_seq_independent=false` and growing extent `[320]`;
  asserts the capture signature is `None`, so `capture_support()` reports
  Unsupported (the whole point: a disqualified node yields no capture-safe
  signature).
- `classifier_pinned_node_stays_capture_eligible` — positive: a pinned
  single-token decode shape `[1,1,32,128]` with `seq_independent=true` stays
  eligible.
- `pinned_single_token_consumer_yields_capture_signature` — positive
  capture-level companion: pinned consumer still produces `Some` signature
  (capture coverage for genuinely fixed decode preserved).

**Confirmed FAIL on HEAD 0fd87df3 / PASS now.** Reproduced the old
`capture_shape_eligible` body standalone: `old(false,[320]) = true`,
`old(false,[328]) = true`, `old(false,[1]) = true`, and the old growing-signature
`is_none() = false` — i.e. every new assertion fails on the old body. All four
tests PASS after the fix.

**Unit-test results (local, all green):**
- `cargo fmt --all --check` — clean.
- `cargo clippy -p onnx-runtime-ep-cuda --lib` (default) and
  `--features cuda --all-targets` — no new warnings (pre-existing GQA/CSA test
  doc-lint warnings only, untouched).
- `cargo test -p onnx-runtime-ep-cuda --lib` — 284 passed, 0 failed (2 ignored),
  incl. the 4 new capture-level tests.
- `cargo test -p onnx-runtime-session --lib` — 145 passed (classifier predicate
  suite + Batty/Leon/Sebastian tests green).
- `cargo test -p onnx-runtime-session --features cuda --lib` — 148 passed
  (segmented-graph / decode-inline device-binding tests green).

**Coordinator: EXACT commands to GPU-re-measure (I did NOT run these).** Expected:
still 34 captured segments, growing/disqualifying set 11, byte-exact oracle.
Default classifier = FAIL-SAFE (env unset); flip with
`ONNX_GENAI_CAPTURE_CLASSIFIER=denylist`.

```bash
# Segment count + growing/disqualifying set size on 35B-A3B QMoE (FAIL-SAFE default):
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_LOG_GROWING_SYMBOLS=1 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo run --release -p onnx-runtime-session --features cuda --bin profile_native -- \
  --pipeline --backend native --ep cuda --steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8 -t0

# (optional A/B) DENYLIST-over-provenance — same run with the switch flipped:
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  ONNX_GENAI_CAPTURE_CLASSIFIER=denylist \
  ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_LOG_GROWING_SYMBOLS=1 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo run --release -p onnx-runtime-session --features cuda --bin profile_native -- \
  --pipeline --backend native --ep cuda --steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8 -t0

# Byte-exact vs fp32 oracle — capture ON (~13 min):
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle -- --nocapture
CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_GRAPH=1 \
  cargo test --release -p onnx-runtime-session --features cuda \
  qwen36_35b_a3b_qmoe_native_cuda_hybrid_continuation_matches_fp32_oracle -- --nocapture
# repeat both with ONNX_GENAI_CUDA_GRAPH=0 to confirm capture-OFF parity.
```

Do NOT merge. Code-fix HEAD sha: 73053709 (this doc commit sits on top).
