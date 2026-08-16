# Coco — History Archive

## ARCHIVED 2026-08-12 (compaction wave)

### 2026-07-12: Joined
Metal/MPS kernel engineer for Apple Metal EP for ONNX Runtime. Owns data/quantization/elementwise kernels: GatherBlockQuantized, quantize/dequantize, KV ops, elementwise/activations, reshape/transpose/cast.

### 2026-07-14T19:05:00Z — Tracer AutoDiagnosis
Tracer AutoDiagnosis and roofline module merged in `8607687`; Hodge review GREEN.

### 2026-07-15 — oneDNN wheels
Bundled oneDNN in Linux and macOS Python wheels (merged `ef89a95`).

### 2026-07-16 — onnx-rs text-format port review
Cleared merged commit `23e4995` 🟢: 10 added tests, 89 passing onnx-rs tests.

### 2026-07-22 — WP-B landed
WP-B landed: raw-protobuf authority fixes completed the epic.

### 2026-07-28 — Shape-inference catalog batch 2
PR #339 (`b1f9d3bb`): Det, LpPool, GlobalLpPool, MaxUnpool, Col2Im, CenterCropPad; registry 181→187 operators.

### 2026-08-11 — B1/B2 blocker fixes (PR #762)

- **B1:** `HashSet<ValueId>` replacing string sentinel on `OutboundGraphReader`.
- **B2:** `filter_map → map` at both call sites (ep.rs:238, ep.rs:493); `ShapeInference::for_node` takes `&[Vec<Option<usize>>]`; `build_conv` fails closed on unknown dims.
- 269 passed, 0 failed (+3 new tests).

### 2026-08-11 — PR #762: B1/B2 correction detail

Commit `38625fb38`. ValueIds are arena indices unforgeable from model content. `DIM_UNKNOWN` constant. Compile-time `conformance_mixed_partition` node counter via `dlsym`.

### 2026-08-12 — PR #32001 scope correction (macOS arm64 gate)

`onnxruntime_target_platform STREQUAL "arm64"` gate; warn-and-disable not FATAL_ERROR; rescoped option descriptions. Head `52db6351b5`.

### 2026-08-12 — PR #31993 NaN fix (hardware sNaN quieting)

NaN assertion: `isnan` + sign + payload modulo quiet bit. RNE tie corrected to 1 + 2⁻¹¹. Removed `-march=armv8.2-a+fp16`. macOS gate via `TARGET_OS_OSX`. Head `02a9f34`.
