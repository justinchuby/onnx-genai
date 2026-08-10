# Validation Contract: Tensor/Shape Data Crossing the ORT Plugin-EP ABI

**Author:** Leon (Engine Dev)
**Date:** 2026-08-10
**Scope:** `crates/onnx-runtime-ep-plugin/src/compute.rs`, `kernel_ctx.rs`

## Principle

All shapes, dims, dtypes, and data pointers received from ORT are **untrusted input** (a malicious ONNX model reaches this code). Every value crossing the `extern "C"` ABI boundary must be validated before use. The contract is **fail closed**: return an `OrtStatus` error with an actionable message, never silently default, clamp, or coerce.

## Validation Rules

### 1. Dimensions (N2 fix)

- Every `i64` dimension from `GetDimensions` is validated via `validate_dims()`.
- **Negative dims are rejected.** At Compute time, all dimensions must be concrete. ORT's `-1` symbolic sentinel arriving here is an upstream bug that must surface immediately.
- Zero dims are **accepted** (legal in ONNX for empty tensors).

### 2. Element-count overflow

- `shape.iter().product()` is replaced with `checked_mul` fold everywhere.
- If the product of dimensions overflows `usize`, the operation fails with an error naming the shape.

### 3. Byte-length overflow

- `element_count * dtype.byte_size()` uses `checked_mul`.
- Overflow is reported with both counts named.

### 4. Data pointer nullity

- Non-zero-element tensors **must** have a non-null data pointer. Null = device-only memory or corruption → rejected.
- Zero-element tensors **may** have null data (legal: no bytes to read).

### 5. Dtype safety

- `DataType::from_onnx(elem_type)` returns `None` for unsupported types → rejected with the raw enum value named.
- No tensor is reinterpreted as a different element type; the dtype flows from ORT through to `TensorView`.

### 6. Panic guard (N1 — already present)

- `compute_execute` wraps the entire body in `std::panic::catch_unwind`.
- A caught panic produces `fail_status("Compute: internal panic")` — never UB from unwinding across `extern "C"`.
- All other callbacks (`CreateState`, `ReleaseState`, factory callbacks) are identically guarded.

### 7. Shape-inference overflow in intermediate buffers

- The multi-node routed path uses `checked_mul` for both element count and byte length when allocating `IntermediateBuf`.
- `read_i64_tensor` (used by Reshape/Slice inference) uses checked element-count arithmetic.

## What Is NOT Validated Here (out of scope)

- Output shapes from `infer_shapes` are trusted (computed from validated inputs by our code).
- `allocate_output` trusts the shape we pass to ORT's `KernelContext_GetOutput`.
- Device-memory tensors are rejected (v1 limitation), not supported.
