# Leon — History (compacted 2026-08-11)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-08-10:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization. Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI). CUDA graph/capture correctness. PR #291 rewind policy split. Unified native CUDA/ORT KV capacity policy. EP plugin compute hardening (BL2/BL3 slot fidelity wave 1). Clippy dead_code cleanup. NEW-1 fix + f16/bf16 marshaling. Stream EP memory leak fix. Device data-transfer contract.

Older detailed work archived in `history-archive.md`.

## 2026-08-11 — Device data-transfer contract (`transfer.rs`)

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762)

**Created:** `crates/onnx-runtime-ep-plugin/src/transfer.rs` — ORT `OrtDataTransferImpl` adapter.

**What:**
- `DeviceDataTransfer` (basic) and `DeviceDataTransferFull` (with OrtApi) adapters
- Copy-direction matrix: H→D, D→H, D→D(same) supported; cross-device + H→H rejected
- Stream-ordered copy via `copy_async` + `Fence` + `wait_fence`
- Ownership: Box::into_raw/from_raw lifecycle, EP borrowed not owned
- Mock device EP with non-host-dereferenceable address space for testing
- 21 new tests covering direction matrix, fail-closed CanCopy, ownership/leak detection, device-pointer guards

**Validation:**
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-plugin` → 154 lib + 9 parity passed
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 23 passed
- `cargo check --workspace` → success

**Not proven:** Nothing here proves CUDA works. Hardware-gated.

## 2026-08-11 — BL2/BL3: Optional slot positional integrity (PR #762)

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762, draft)
**Triggered by:** Third independent Opus review rejection — silent corruption class.

### BL2 — Omitted optional outputs (graph_reader.rs)

**Root cause:** `filter_map` in `from_ort_graph()` dropped empty-named output
slots, compacting the Vec. SkipLayerNormalization with signature
`(output, "", "", sum)` became `[output, sum]` (len 2), causing the kernel to
write mean into position 1 (which was really the sum slot).

**Fix (preferred — slot-map, not fail-closed):** Empty-named outputs get
placeholder ValueIds with `DataType::Undefined`. In compute.rs fast path,
Undefined-dtype slots receive local scratch buffers; ORT output indices
increment only for present slots. Kernel sees full arity (4) and writes
to correct positions.

### BL3 — Omitted optional inputs (compute.rs)

Added `NodeInputSource::Absent` variant. Compute loop provides
`TensorView::absent(DataType::Undefined)` for Absent slots. Kernels detect
absence via `is_absent()`.

**Caveat:** `ep.rs:597` still emits `Ort(0)` for None inputs — Sebastian's
BL1 pass must change it to `Absent`. The single-node fast path works because
it passes inputs from ORT directly (no routing table).

### Nonblocker — unwrap_or(DataType::Float32)

All three instances replaced with fail-closed error. A short `output_dtypes`
vector is now a hard compute failure, not a silent Float32 guess.

### Tests (all real ORT, all numerical)

| Test | Asserts | Pre-fix behavior |
|------|---------|------------------|
| `skip_layer_norm_output_sum_position` | sum[0..8] == X+skip | Got mean (2.625) |
| `clip_omitted_min_with_max` | Y == clip(X, -∞, 5) | Would alias min=X |
| `skip_layer_norm_omitted_beta_bias` | LN(X+skip, γ=1, β=0) | Would alias β=X |
| `simplified_layer_norm_two_outputs_position` | inv_std correct | Position coverage |

**Validation:**
- `cargo test --no-fail-fast -p onnx-runtime-ep-plugin -p onnx-runtime-ep-cpu-plugin -p onnx-runtime-ep-cuda-plugin` → 215 passed / 0 failed
- `cargo clippy --all-targets -- -D warnings` on all 3 crates → clean
- `cargo fmt --check` → clean

**Cannot fix within file ownership:** `ep.rs:597` (`NodeInputSource::Ort(0)` for
None inputs in `build_subgraph_routing`). Documented for Sebastian.

## 2026-08-11 — PR #762 third corrective wave: BL2/BL3 optional slot fidelity

**Task:** Fix BL2 (output slot compaction by filter_map) and BL3 (absent inputs aliased to input 0).

**Commits:** `6ce94f033`, `49f39633b`, `e5dbed0dd`

- `graph_reader.rs` now preserves positional output slots with `ValueId` placeholders for empty-named slots.
- `NodeInputSource::Absent` variant added; compute loop passes `TensorView::absent()` for absent inputs.
- 3 `unwrap_or(DataType::Float32)` fallbacks removed; fails closed with explicit error.

**Outcome:** Fix correct at graph/compute level. However, Luv's review found the optional-slot conformance tests were vacuous — EP was declining the nodes at `ep.rs:275` (Undefined-dtype output check). BL2 fix was dead code in the ORT plugin path. Mariette corrected. Pris found BL1 regression test lacked fallback guard; Rachael hardened.

**Lesson reinforced:** A passing test is not evidence the code under test ran. `disable_cpu_ep_fallback=1` + `Session_GetEpGraphAssignmentInfo` assertions are both required.

## 2026-08-11 — PR #31988 TensorRT build fix

- **Task**: Clear last real build blocker (Build Linux TensorRT x64 Release).
- **Root cause**: `matmul_nbits_cols_per_block_test.cc` (host .cc) included `matmul_4bits_common.cuh` which pulls `<cuda_bf16.h>` → CUB device headers. Host compiler can't resolve `blockIdx`/`__threadfence`.
- **Verdict**: OURS — PR #31678 (unrelated) has TensorRT green; ours red.
- **Fix**: Extracted `SelectColsPerBlock` + constants to `matmul_4bits_cols_per_block.h` (host-only). Test uses host header; `.cuh` re-exports via include.
- **nvcc local**: Installed (12.0). Full compile not feasible (gsl/onnxruntime deps missing) but host header verified standalone with g++.
- **New head**: `34fe91e8dd`
- **Invariants**: All four preserved — no reduction order/routing/wide-n/split-K changes; only header organization.

## 2026-08-12 — PR #31988 TensorRT build fix

- **Task**: Clear `Build Linux TensorRT x64 Release` blocker on PR #31988.
- **Root cause**: `matmul_nbits_cols_per_block_test.cc` (host `.cc`) included `matmul_4bits_common.cuh`, which pulls `<cuda_bf16.h>` → CUB device headers. ~40 `'blockIdx' was not declared` errors in host compilation context.
- **Verdict: OURS** (not inherited) — cross-PR comparison: #31678 (unrelated) TensorRT green; #31988 red. Disproved Deckard's initial "CUDA-13 base-codebase" assumption.
- **Fix**: Extracted `SelectColsPerBlock`, `kColsPerThreadBlock`, `kTargetCtasPerSm` into `matmul_4bits_cols_per_block.h` (host-only, no device includes). Test uses only this header; `.cuh` re-exports via `#include`. All four invariants preserved (routing/output/wide-n/split-K unchanged).
- **Head**: `34fe91e8dd`.
