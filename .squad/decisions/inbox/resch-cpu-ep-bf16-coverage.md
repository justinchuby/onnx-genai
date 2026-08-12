# Decision: Native bfloat16 coverage for the CPU EP (all ops)

**Author:** Resch (Intel CPU Optimization Engineer)
**Branch:** `squad/cpu-ep-bf16-coverage`
**Scope:** `crates/onnx-runtime-ep-cpu`
**Requested by:** Justin (@justinchuby) — "全面检查cpu ep对bfloat16的原生支持 一口气支持所有op"
(comprehensively audit the CPU EP's native bf16 support and support bf16 for ALL ops in one sweep)

## TL;DR

The CPU EP already had **broad, first-class bf16 support** via the shared
compute-in-f32 machinery in `src/dtype.rs` (`dispatch_arith` / `dispatch_float`
macros + `to_dense_f32_widen` / `write_dense_f32_narrow`). Auditing all **194
registered op keys**, every kernel that references `Float16` also referenced
`BFloat16` — i.e. no op special-cased f16 while forgetting bf16. The gaps were a
small set of **f32-locked kernels that already compute in f32** but rejected
non-f32 dtypes at their claim/execute gate. I widened those to accept bf16 (and
f16) through the same helpers, and added a data-driven conformance sweep as the
regression lock.

## Audit result (counts)

- **Already supported bf16 (verified):** the overwhelming majority — all standard
  elementwise/unary/binary math, activations, reductions, softmax family,
  normalizations, movement ops, MatMul/Gemm, attention (MHA, GQA, packed varlen,
  SDPA, linear/rotary), conv/pooling, Clip/Where, Cast, NonZero, quant float
  sides, etc. These route dtype dispatch through the shared macros/helpers.
- **Newly added bf16 (4 ops):** `DFT`, `VarlenAttention` (`pkg.nxrt`), `MoE`
  (`com.microsoft`), `IndexShare` (`pkg.nxrt`).
- **Deliberately excluded (bf16 not a valid/native type):**
  - Integer/bitwise ops — `BitShift`, `BitwiseAnd/Or/Xor/Not` (ONNX integer type
    constraints; bf16 is not in the constraint set).
  - Quantized-integer ops — `QLinearMatMul` (int8/uint8 operands),
    `DynamicQuantizeLinear` (uint8 output); the nbit/block-quantized MatMul/MoE
    (`MatMulNBits`, `BlockQuantizedMatMul`, `BlockQuantizedMoE`, `QMoE`,
    `GatherBlockQuantized`) keep int-quantized weights but already accept bf16
    activations/scales through their existing dispatch.
  - `CompressedSparseAttention` — a specialized FP8-E4M3 / FP4-E2M1
    compressed-KV attention oracle whose query/`current_kv` dtype (Float32) is
    part of its compressed-cache contract; it is not a standard ONNX op with a
    published bf16 type constraint. Left f32-anchored on purpose.
  - Int/bool-output, value-agnostic ops — `Shape`, `Size` (ignore element
    values), `ArgMax`/`ArgMin`/`TopK` (int indices), comparison/logical ops
    (bool output). These already accept bf16 *inputs* where relevant (e.g.
    `NonZero` widens through `to_dense_f32_widen`).

## Why these 4 needed fixing (and how)

All four already computed in f32 internally but gated their dtype at claim/exec:

- **`DFT`** (`kernels/dft.rs`): read via `to_dense_f32`, hard-rejected
  `input.dtype != Float32`. ONNX DFT's `T1` constraint includes `bfloat16`.
  Fix: read via `to_dense_f32_widen("DFT", …)`, write via
  `write_dense_f32_narrow("DFT", …)`, drop the reject. Compute stays in f32.
- **`VarlenAttention`** (`kernels/varlen_attention.rs`): already widened Q/K/V
  with `to_dense_f32_widen` and narrowed the output, yet the `unsupported_reason`
  claim gate and the execute gate rejected non-f32 — inconsistent with its
  sibling `PackedVarlenAttention`, which already accepts f32/f16/bf16. Fix:
  relax both gates to `Float32|Float16|BFloat16` and require Q/K/V share one
  float dtype.
- **`MoE`** (`kernels/moe.rs`, `com.microsoft`): "Phase-1 Float32 reference"
  gate rejected non-f32 on every float input/output, but all reads went through
  `to_dense_f32` and the write through `write_dense_f32`. Fix: widen every float
  input (`to_dense_f32_widen(…).into_owned()`), narrow the output, relax the
  dtype gate to the float set with an all-same-dtype requirement.
- **`IndexShare`** (`kernels/index_share.rs`, `pkg.nxrt`): same pattern across
  Q/K/V/past/bias and three outputs, plus a claim-metadata dtype check. Fix:
  widen/narrow all float I/O, add `is_supported_float` + `require_float_dtype`
  helpers (float set + match anchor), relax the claim-metadata check.

The uniform principle (per the project's DRY dtype design): **compute in f32,
widen on read, narrow to the output dtype on write** — never a bespoke bf16
arithmetic path.

## Conformance test (the "一口气支持所有op" guarantee + regression lock)

- New integration test `tests/bf16_conformance.rs`: a data-driven table of ~68
  minimal valid nodes spanning every shared dispatch path (unary math,
  activations, binary elementwise, all reductions, softmax/logsoftmax/hardmax,
  layernorm/lpnorm, Clip/Where/Transpose/Concat/MatMul/Gemm/CumSum/Expand/Tile,
  DFT). Each case runs in **f32** and **bf16** and asserts (1) bf16 executes
  without error and (2) its result matches the f32 reference within a
  magnitude-scaled bf16 tolerance. Two tests: `every_registered_float_op_supports_bf16`
  and `no_op_rejects_bf16_at_runtime`, plus a dedicated `VarlenAttention` case.
- Added a shared bf16 harness helper `Tensor::bool(...)` in
  `benches/common/mod.rs` (needed for `Where`).
- Added focused in-crate bf16 tests for the multi-input oracles that don't fit
  the flat table: `moe::tests::moe_top1_relu_runs_natively_in_bf16_matching_f32`
  and `index_share::tests::index_share_runs_natively_in_bf16_matching_f32`.

## Validation

- `cargo test -p onnx-runtime-ep-cpu`: **all green** (1048 lib + 3 conformance +
  existing parity/regression suites; 0 failures).
- `cargo fmt --all`: applied.
- `cargo clippy -p onnx-runtime-ep-cpu --lib --tests`: clean (the single
  `dot_kernel` unused-variable warning in `matmul_nbits.rs` is pre-existing on
  `origin/main` and untouched by this change).
