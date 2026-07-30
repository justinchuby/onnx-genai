# #67 Coverage-completion + robustness (cohaagen)

Branch: `squad/67-coverage-polish` (off `origin/main`). Draft PR refs #67.

Three independent, low-risk items closing the small gaps surfaced during #480/#484.
All three landed clean; each proven against the CPU EP oracle and on real qwen3.5-0.8b nodes.

## Item 1 — `RotaryEmbedding` for `com.microsoft`

- **Gap:** CUDA registered only `ai.onnx::RotaryEmbedding` (opset 23). The
  `com.microsoft` contrib variant (used by qwen3.5 text decoders) fell to CPU.
  The two ops share identical rotation math; only the input ORDER differs:
  - contrib: `(X, position_ids, cos_cache, sin_cache)`
  - default: `(X, cos_cache, sin_cache, position_ids?)`
- **Fix (DRY):** added a `contrib: bool` to `RotaryEmbeddingKernel` and a shared
  `rotary_kernel_from_node` factory helper; a single `resolve_input_order(contrib, n)`
  maps `(cos_i, sin_i, pos_i)`. All hardcoded `inputs[1]/[2]/[3]` refs (execute,
  dtype check, cache validation, CUDA-graph capture signature, launch args) now
  route through the resolved indices. Registered `RotaryEmbeddingContribFactory`
  as `OpKey("RotaryEmbedding","com.microsoft",1)`.
- **Bug fixed en route:** the claim-time and execute-time dtype checks used
  `inputs[..3]` / `.take(3)`, which for the contrib ordering wrongly compared the
  Int64 `position_ids` against the float dtype. Now checks `X`/`cos`/`sin` by index.
- **Claim gate:** `unsupported_reason(contrib, dtypes)`; provider wires the
  `com.microsoft` branch alongside `ai.onnx`.
- **Parity:** new `tests/rope_contrib_gpu.rs` — fp32/fp16/bf16 × interleaved∈{0,1}
  on `[1,2,2,8]` with `position_ids`, tol-exact vs CPU EP (1e-4/3e-3/3e-2). Green.
- **Placement proof:** qwen3.5-0.8b `text.onnx` → **12/12** com.microsoft
  RotaryEmbedding nodes now claim CUDA (were 0 before).
- Coverage-of-coverage: "RotaryEmbedding" already a covered op name (keys on name,
  not domain), so no new CUDA_COVERED_OPS entry required.

## Item 2 — `Bool`-input `NonZero`

- **Gap:** CUDA NonZero handled f32/f16/bf16 only. The CPU EP oracle ALSO rejected
  Bool (`to_dense_f32_widen` has no Bool path), so real models with a Bool NonZero
  (qwen3.5 `embedding.onnx`) failed on BOTH EPs.
- **Fix (DRY):**
  - CUDA: added `DEFINE_NONZERO(unsigned char, bool_)` to the existing NVRTC macro
    (the generic `nz<T>` template already does `v != (T)0`); `Bool` dtype arm in the
    execute dispatch; `DataType::Bool` added to the `standard_claims::nonzero` gate.
  - CPU: `NonZeroKernel` now reads a Bool mask via a new `to_dense_bool` helper
    (strided `read_strided::<u8>` with `want=Bool`) and unifies the nonzero test into
    a `Vec<bool>` predicate; all other dtypes still widen to f32. This makes the CPU
    EP a valid parity oracle AND fixes CPU fallback for these models.
- **Parity:** conformance sweep case `NonZero[bool,rank2]` (ExactBytes vs CPU);
  CPU unit test `nonzero_accepts_bool`.
- **Placement proof:** qwen3.5-0.8b `embedding.onnx` → **1/1** Bool NonZero on CUDA.

## Item 3 — GatherBlockQuantized odd-blocks-per-row gate + honest doc

- **Gap (from Melina's #480 review):** the CUDA bits=4 zero-point unpack uses GLOBAL
  nibble addressing (by `block_id`); CPU/ORT pack zero points PER ROW. They agree
  ONLY when blocks-per-row is even (a multiple of `components = 8/bits`); bits=8
  (`components==1`) always agrees. Odd-bpr-with-zp previously failed only
  INCIDENTALLY via a confusing zp size-mismatch, and GBQ had no claim gate at all
  (claim-then-fail — violates the decisions.md rule).
- **Fix:**
  - Explicit LOUD execute bail: `zero_points && components>1 &&
    !blocks_per_row.is_multiple_of(components)` → clear message naming the
    blocks-per-row and layout gap, before the incidental size check.
  - New claim gate `unsupported_reason(node, shapes)`: computes blocks-per-row from
    STATIC data shape; declines odd-bpr-with-zp at claim time; conservative (returns
    None → still claims, execute bail is the backstop) when shapes are symbolic or zp
    absent. Wired for `com.microsoft::GatherBlockQuantized` in provider.rs.
  - Doc: softened "mirrors ORT exactly" → "matches ORT for the `uint8` path" plus a
    dedicated "Zero-point layout precondition (even blocks-per-row)" section stating
    the precondition and the explicit refusal. Did NOT change the per-row addressing
    itself (out of scope; even-bpr is the real int4 embedding layout).
- **Test:** `claim_gates_gpu::gather_block_quantized_odd_blocks_per_row_with_zero_points_declines`
  — odd bpr declines with a "blocks per row" reason; even bpr still claims. Existing
  `gather_block_quantized_gpu.rs` fp32/fp16 bits4/bits8 parity still green.
- **Placement proof:** qwen3.5-0.8b `embedding.onnx` GBQ (even-bpr real layout) →
  **1/1** on CUDA (unaffected — gate only bites odd-bpr).

## Verification

- `cargo test -p onnx-runtime-ep-cuda --features cuda`: lib **274 passed**;
  conformance 4 passed (sweep incl. new Bool case, coverage-of-coverage green);
  rope_contrib 1, claim_gates 4, gather_block_quantized 2, rope_capture 1 — all green.
- `cargo test -p onnx-runtime-ep-cpu --lib` NonZero incl. `nonzero_accepts_bool` green.
- `cargo fmt --all --check` clean; clippy `-p onnx-runtime-ep-cuda` (cuda / no-cuda),
  `-p onnx-runtime-ep-cpu`, and `-p onnx-genai-engine --features cuda,native-backend`
  — zero new warnings from touched files.

## Follow-ups / remaining

- GBQ per-row nibble addressing for TRUE odd-bpr int4 layouts is intentionally NOT
  implemented (no such layout in the real models; would need a parity-proven odd case).
  The loud gate + honest doc is the correct scope.
