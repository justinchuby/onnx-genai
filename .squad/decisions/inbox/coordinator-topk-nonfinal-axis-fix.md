# CPU-EP TopK: k-major output layout for non-final axes

Date: 2026-08-11
Owner: Squad (coordinator; sub-agent spawn hit a canary false-positive and returned 0 useful turns, so fixed inline per fallback rule)

## Bug
`TopKKernel::execute` (crates/onnx-runtime-ep-cpu/src/kernels/selection.rs) built the
values/indices outputs by sequential `push` in loop order `outer -> inner(i) -> rank`,
producing memory layout `[outer][inner][k]`. ONNX TopK output keeps the input shape with
`shape[axis]=k`, i.e. the correct layout is k-major `[outer][k][inner]`. The rank-`r`
winner for slot (outer, i) must land at flat index `(outer*k + r)*inner + i`.

The two layouts are EQUAL only when `inner == numel(shape[axis+1..]) == 1`, i.e. the
reduced axis is the final axis. So the bug was latent for final-axis TopK (the common
router/argmax-style case) but produced wrong values+indices ordering for any
**non-final-axis TopK with k>1** — dtype-independent, pre-existing. CUDA EP / ORT emit
k-major.

## Fix
Pre-size the output vectors to `numel(outputs[0].shape)` and write each selected element
to its strided destination `(outer*k + rank)*inner + i` instead of pushing sequentially.
Removed the now-unused `sorted` struct field (the kernel always emits sorted winners,
which satisfies both `sorted=1` and `sorted=0`); documented the intent in the factory.

## Validation
- New tests: `topk_non_final_axis_uses_k_major_layout` (shape [2,3,2], axis=1, k=2) and
  `topk_first_axis_is_k_major` (shape [3,2], axis=0, k=2) — both assert the k-major
  values AND indices. Old push-order code would emit e.g. `[5,3,4,2]` vs correct
  `[5,4,3,2]`, so the tests are discriminating.
- Final-axis path unchanged: existing `topk_bfloat16_values_match_widened_reference` and
  `topk_and_nonzero` still pass verbatim (inner==1).
- `cargo test -p onnx-runtime-ep-cpu --lib selection`: 22 passed / 0 failed.
- `cargo fmt --all`; `cargo clippy -p onnx-runtime-ep-cpu --lib` clean.
