# Leon — f16/bf16 Marshaling and Fail-Closed Policy for CPU EP

**Date:** 2026-08-10T23:12:00Z
**Author:** Leon (Engine Dev — KV & Buffers)
**Branch:** squad/ep-plugin-parity-cuda
**Files modified:** `crates/onnx-runtime-ep-plugin/src/compute.rs`, `crates/onnx-runtime-ep-plugin/src/kernel_ctx.rs`

---

## TASK 1 — NEW-1 (compute_release_state catch_unwind)

### Finding (from Holden's audit)

`compute_release_state` (`compute.rs:~1557`) was an `extern "C"` callback with
no `catch_unwind` guard. A panic in its body (including any future custom `Drop`
on `ComputeState`) would unwind across the `extern "C"` boundary — undefined
behaviour.

### Fix

Wrapped the body in `std::panic::catch_unwind(AssertUnwindSafe(…))`, discarding
the result with `let _ = …`. Because the function returns `void` and there is
no `OrtStatus` channel, a caught panic is swallowed. This is the correct pattern
for void-returning `extern "C"` callbacks — the same approach used throughout
the codebase for void paths.

### Audit of all extern "C" entry points

Three `extern "C"` callbacks exist in `compute.rs`:

| Function | Returns | Guard before fix | Guard after fix |
|---|---|---|---|
| `compute_create_state` (line 626) | `*mut OrtStatus` | ✅ `catch_unwind` + `fail_status` | ✅ unchanged |
| `compute_execute` (line 654) | `*mut OrtStatus` | ✅ `catch_unwind` + `fail_status` | ✅ unchanged |
| `compute_release_state` (line 1557) | `void` | ❌ missing | ✅ added `catch_unwind`, panic swallowed |

No other `extern "C"` entry points exist in the two owned files.

---

## TASK 2 — f16/bf16 Marshaling Layer

### ORT ↔ DataType mapping (verified against `bindings.rs`)

```
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  = 10  →  DataType::Float16  = 10
ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16  →  DataType::BFloat16 = 16
```

Both map via `DataType::from_onnx(raw_i32)` which is a direct enum discriminant
cast — no table lookup required. `DataType::Float16.byte_size()` and
`DataType::BFloat16.byte_size()` both return `2`, so the existing `checked_mul`
overflow guards in `validate_dims` handle them correctly with no special-casing.

### Fail-closed policy for unsupported dtypes

`DataType::from_onnx` returns `None` for any unrecognised ORT element type
(including `UNDEFINED = 0` and future reserved values). The `read_inputs` path
converts this to:

```
Err(format!("unsupported element type {elem_type} for input {i}"))
```

The raw numeric ORT value is included in the message so the caller can diagnose
which type was rejected. This error propagates as an `OrtStatus` error — never
silently coerced or defaulted.

### TensorView / DataType alignment

`TensorView::validate()` checks `byte_offset % byte_size == 0`. For f16/bf16
`byte_size = 2`, so the view correctly requires 2-byte alignment of the element
origin. No special casing is required — the existing alignment check in
`validate_view` covers 2-byte types correctly.

### Public interface exposed for Deckard

**Signature:**

```rust
// crates/onnx-runtime-ep-plugin/src/kernel_ctx.rs
pub const CPU_EP_SUPPORTED_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Uint8,
    DataType::Int8,
    DataType::Uint16,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Bool,
    DataType::Float16,   // 2 bytes/element; ORT value 10
    DataType::Float64,
    DataType::Uint32,
    DataType::Uint64,
    DataType::BFloat16,  // 2 bytes/element; ORT value 16
];
```

**Access path:** `onnx_runtime_ep_plugin::kernel_ctx::CPU_EP_SUPPORTED_DTYPES`

Deckard must import this slice in `ep.rs` to populate `GetKernelRegistry` type
constraints. **Do not copy-paste the list** — importing keeps the type-constraint
advertisement and this marshaling layer in sync.

To convert each dtype to its ORT enum value: `dt.to_onnx() as u32` (or `i32`)
gives the `ONNXTensorElementDataType` value.

---

## Tests added (both in `--lib` suite)

| Test | File | What it covers |
|---|---|---|
| `release_state_swallows_panic_safely` | `compute.rs` | NEW-1 guard: panic in drop path is caught |
| `f16_dtype_round_trip` | `kernel_ctx.rs` | `from_onnx(10)` == `Float16`, `to_onnx()` == 10, `byte_size()` == 2 |
| `bf16_dtype_round_trip` | `kernel_ctx.rs` | `from_onnx(16)` == `BFloat16`, `to_onnx()` == 16, `byte_size()` == 2 |
| `f16_byte_length_computation` | `kernel_ctx.rs` | [4,8] f16 → 32 elements × 2 bytes = 64 bytes |
| `bf16_byte_length_computation` | `kernel_ctx.rs` | [3,5] bf16 → 15 elements × 2 bytes = 30 bytes |
| `f16_byte_length_overflow_guard` | `kernel_ctx.rs` | element_count = 2^63 → byte_len = 2^64 overflows usize |
| `unsupported_dtype_fails_closed` | `kernel_ctx.rs` | `from_onnx(0)` returns `None`; error names the value |
| `cpu_ep_supported_dtypes_contains_f16_and_bf16` | `kernel_ctx.rs` | `CPU_EP_SUPPORTED_DTYPES` includes both half-precision types |

## Test results

```
cargo test -p onnx-runtime-ep-plugin --lib
  test result: ok. 90 passed; 0 failed

cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings
  Finished (clean)

cargo test -p onnx-runtime-ep-cpu-plugin
  test result: ok. 15 passed; 0 failed
```

## Blocked / deferred

- End-to-end f16/bf16 through a live ORT session requires Deckard's
  `GetKernelRegistry` type constraints to be advertised first (PR gating on
  `ep.rs`). Marshaling layer is ready; integration test can be added by Pris
  once Deckard's EP type constraints land.
- Sub-byte types (Int4/Uint4) are deliberately excluded from
  `CPU_EP_SUPPORTED_DTYPES` — they use packed storage and their stride model
  differs. Inclusion requires a separate marshaling path.
