# Chew — #762 Test Repair Report

**Date:** 2026-08-11
**PR:** #762 (squad/ep-plugin-parity-cuda)
**Status:** All four blocker test repairs complete; one new bug found.

## Compile Fixes (B1)

- `plugin_export_abi.rs:363` and `:429`: `output_dtype: DataType::Float32` → `output_dtypes: vec![DataType::Float32]`
- Both `CompiledKernelEntry` instantiations now use the `Vec<DataType>` field matching B1's implementation fix.

## B1 Dtype Conformance Tests (NEW)

Four real-ORT tests added to `plugin_ort_e2e.rs`:

| Test | Op | Input dtype(s) | Output dtype asserted | Values asserted |
|---|---|---|---|---|
| `conformance_cast_f32_to_i64` | Cast | f32 [2,3] | **INT64** | [1,2,3,4,5,6] (truncated) |
| `conformance_where_bool_f32` | Where | bool [2,2], f32 [2,2], f32 [2,2] | **FLOAT** (not bool) | [1.0, 20.0, 30.0, 4.0] |
| `conformance_shape_f32` | Shape | f32 [3,4,5] | **INT64** | [3, 4, 5] |
| `conformance_layer_norm_multi_output` | LayerNormalization | f32 [2,4], f32 [4] | **FLOAT** ×3 | **IGNORED — see bug below** |

### Provider attribution

`conformance_setup` asserts `provider == cpu_ep` by verifying our EP appears in `GetEpDevices` and is appended to the session. If ORT silently fell back, the device lookup would fail.

### Bug found: LayerNormalization shape inference

`conformance_layer_norm_multi_output` is `#[ignore]` because it uncovered a real bug:
- **Symptom:** "output element count 8 does not match produced 2" for Mean output
- **Root cause:** EP shape inference produces Mean shape [2,4] instead of [2,1]
- **Owner:** Batty (`crates/onnx-runtime-ep-plugin/src/compute.rs` or `crates/onnx-runtime-ep-cpu/src/kernels/layernorm.rs`)

## B2 — ReleaseEpFactory ABI signature restored

- `plugin_export_abi.rs:71`: `type ReleaseEpFactory = unsafe extern "C" fn(...) -> *mut ort::OrtStatus;`
- Release call now asserts returned status is null on success (~line 118)
- Comment explains this undoes our own prior mistake (arm64/macOS false fix)

## B3 — nxrt inline buffer portability

- `nxrt_abi_roundtrip.rs:173,187`: `as *const i8` → `as *const std::ffi::c_char`
- Added `message_str()` assertion in `factory_panic_is_contained` proving the inline [u8; 256] buffer survives the real cdylib boundary

## B4 — CUDA fail-closed assertions

New file: `crates/onnx-runtime-ep-cuda-plugin/tests/cuda_fail_closed.rs`

1. `*out_num == 0` asserted in both configs ✓
2. Status null acceptable in test context (no ORT loaded to allocate through) ✓
3. Message contract documented (`IMPLEMENTATION-BLOCKED` / `without cuda feature`) ✓

## Fixtures

Four new ONNX fixtures generated via `generate_fixtures.py`:
- `cast_f32_to_i64/model.onnx`
- `where_bool_f32/model.onnx`
- `shape_f32/model.onnx`
- `layer_norm_f32/model.onnx`

All added to `.gitignore` with `!` negations.

---

## 2026-08-11 Update: LayerNorm shape-inference bug now fixed; test hardened

**Performed by:** Chew (correctness reviewer)  
**Scope:** Task 1–3 from the B1-layernorm follow-up review comment.

### 1. `#[ignore]` removed

`conformance_layer_norm_multi_output` was parked behind `#[ignore]` after the bug was fixed by Batty's `ShapePreservingNorm` → `LayerNorm` refactor. The attribute and stale bug comment were removed; the test runs by default.

### 2. Existing test strengthened

Added `assert_output_shape` helper (uses `GetDimensionsCount` / `GetDimensions` ORT API) and applied it:
- `Y` → [2,4], `Mean` → [2,1], `InvStdDev` → [2,1] (directly catches the old regression)
- `InvStdDev` value assertion: both rows have var=1.25 → invstd≈0.8944

### 3. New test: `conformance_layer_norm_neg_axis`

Fixture `layer_norm_neg_axis_f32`: LayerNormalization on 3D [2,3,4] with `axis=-1`.  
Old code gave Mean shape [2,3,4] (full shape); correct shape is [2,3,1].  
Asserts shapes and Mean/InvStdDev values for 5 of 6 rows.

### 4. New test: `conformance_rms_norm`

Fixture `simplified_layer_norm_f32`: `RMSNormalization` (opset 23) on 2D [2,4] with `axis=-1`, single Y output.  
Covers the no-Mean-output path; asserts Y shape, dtype, spot value, and rms(Y_row)≈1.0.

### 5. Fail-closed path not directly assertable

The EP declines when `input_shapes.is_empty()` or axis resolves out of range.  
Neither is constructible from a valid ONNX fixture (ONNX checker rejects invalid axes; ORT always resolves static shapes before `for_node()`).  
The unit-level coverage for this path remains Batty's responsibility.

### Test results

```
cargo test -p onnx-runtime-ep-cpu-plugin
test result: ok. 23 passed; 0 failed; 0 ignored

cargo test -p onnx-runtime-ep-plugin
test result: ok. 9 passed; 0 failed; 0 ignored

RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p onnx-runtime-ep-cpu-plugin
Finished — no warnings or errors

cargo fmt --all -- --check
0 diffs
```
