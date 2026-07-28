# Holden decision: Miri in CI for tractable unsafe crates

Date: 2026-07-27
Branch: `ci/miri-unsafe-crates`

## Tractable-crate analysis

Checked with `cargo tree -p <crate> -e normal,dev` for `ort-sys`, `onnx-runtime-ep-cuda`, `cudarc`, and `mlas-sys`:

- `onnx-runtime-memory`: no hits; included with full `cargo +nightly miri test`.
- `onnx-runtime-dlpack`: no hits; included with full `cargo +nightly miri test`.
- `onnx-runtime-ep-api`: pulls `onnx-genai-ort-sys`; included targeted pure Rust ABI/DeviceBuffer/registry/tensor/weight/mock-EP tests. Excluded Linux-only legacy plugin-loader tests because they compile and dlopen C fixtures via native process/FFI calls.
- `onnx-runtime-ep-cpu`: pulls `onnx-genai-ort-sys` through `onnx-runtime-ep-api`; full tests also include OS affinity and Rayon-heavy execution that Miri cannot reliably interpret. Included targeted pure-unsafe subsets: `strided::tests`, `provider::tests`, and `dtype::tests`.
- `onnx-runtime-session`: pulls `onnx-genai-ort-sys` through EP crates and has a dev-only `cudarc` dependency for CUDA WAR tests. Included targeted pure session ownership/bounds subsets: tensor, sequence, executor view-bounds/checked-size, prefetch, and device-binding tests.
- `onnx-runtime-capi`: pulls `onnx-genai-ort-sys` through session. Included C status/null/handle/pointer/session-option tests; excluded full end-to-end roundtrip from the Miri lane because it enters Rayon/crossbeam worker internals rather than the C ABI pointer-safety surface.
- `onnx-runtime-ep-cuda`: excluded; direct CUDA/cudarc/native driver path.
- `onnx-genai-ort`: excluded; direct native ONNX Runtime FFI.
- `onnx-genai-engine`: excluded; pulls native ORT through `onnx-genai-ort`; CUDA/native backend coverage remains compile/test coverage outside Miri.

## Flags and borrow model

The CI job keeps Miri's default Stacked Borrows model. I deliberately did not switch to Tree Borrows: the project wants strict raw-pointer ownership checking for `DeviceBuffer`, DLPack, strided views, and C handles. The only Miri flag used is `-Zmiri-disable-isolation`, and only for ep-api/session/C API tests that create temporary registry, model, or sidecar files.

## Cost and scheduling

Miri now lives in `.github/workflows/miri.yml` instead of the general CI workflow. That scopes the weekly `schedule` trigger to Miri only, keeps nightly-toolchain failures from being read as general CI failures, and reflects that Miri has a different cadence/owner/toolchain. The workflow runs per-PR and on `main`/`ci/**` pushes when Cargo, `.github/workflows/miri.yml`, or one of the covered crate paths changes; it also runs weekly to catch nightly Miri/toolchain drift even when code is quiet. Its concurrency group matches CI: PRs group by pull-request number, pushes group by SHA, and schedule/workflow_dispatch are separated by the final boolean so a scheduled run cannot share a group with PR or push runs. The measured Linux lane is about seven minutes, so per-PR path-limited execution is cheap enough and is the primary signal; weekly is only a drift backstop. The job prints `MIRI_TIMING <lane>: <seconds>` per lane for durable per-crate timing from GitHub logs.

Local Windows smoke timings before CI were: `onnx-runtime-memory` 166s, `onnx-runtime-dlpack` 33s, `onnx-runtime-ep-api` 155s, `onnx-runtime-ep-cpu strided` about 6s interpreted time, `provider` about 140s interpreted time, and `dtype` about 6s interpreted time. Linux CI timings are authoritative and should be read from `MIRI_TIMING` lines.

## Findings

Miri found one small, unambiguous issue while enabling the lane: `onnx-runtime-ep-cpu::provider::tests::deallocate_rejects_cross_device_buffer` intentionally panicked before freeing a fabricated allocation. That was test-only, but it would make Miri red and could hide real leaks, so the test now catches the expected cross-device panic payload and reconstructs/drops the boxed slice, so unrelated panics no longer satisfy the invariant test. No production soundness defect was found in the covered local smoke run.

## Coverage

Do not upload Codecov coverage from this Miri job. Miri is an interpreter-based soundness checker; coverage instrumentation and Miri are not a useful or reliable composition, and treating Miri as a coverage contributor would misrepresent the coverage signal. The regular test/coverage lanes remain responsible for Codecov upload.
