### 2026-08-14: Committed ONNX test fixtures are textproto unless external-data or ORT-loaded

**By:** Leon

**What:** Converted 29 committed binary `model.onnx` fixtures to git-friendly ONNX
protobuf TextFormat (`model.onnx.textproto`), and established the convention that
committed inline-weight ONNX fixtures are stored as textproto going forward.

Convention: a committed ONNX fixture is stored as `model.onnx.textproto` when it
has **inline weights** and is loaded through **our own loader**
(`onnx_runtime_loader`, which auto-detects TextFormat via `is_textproto_path`).
It stays binary `model.onnx` only when one of the keep-binary reasons below applies.

Converted (29):
- 28 EP-conformance fixtures under `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/*`
  (add_1x4, add_bfloat16, add_broadcast, add_dynamic_dim, add_float16, add_int32,
  add_skip_layer_norm_mul, cast_f32_to_i64, chain_add_mul, clip_no_min,
  layer_norm_bf16_absent_output, layer_norm_dynamic_axis, layer_norm_f16_absent_output,
  layer_norm_f32, layer_norm_neg_axis_f32, matmul_2d, matmul_batched_nd,
  matmul_initializer_weights, mixed_partition, nonzero_1x4, shape_f32,
  simplified_layer_norm_f32, simplified_layer_norm_two_outputs,
  skip_layer_norm_bf16_absent_output, skip_layer_norm_f16_absent_output,
  skip_layer_norm_no_beta_bias, skip_layer_norm_output_sum, where_bool_f32).
- `tests/fixtures/tiny-deepseek-v2-qmoe-attention/model.onnx`.

To convert the cpu-plugin fixtures — which are executed by **real ORT** via
`CreateSession(path)` (ORT's on-disk parser cannot read TextFormat) — a shared
test-harness seam was added (`tests/common/ort_session.rs::create_session`): for a
`*.textproto` path it reads the file, converts to binary protobuf in-memory
(`onnx_std::textproto::to_binary`), and calls `CreateSessionFromArray`; binary
`*.onnx` still loads via `CreateSession`. This mirrors the production seam in
`onnx-genai-ort`'s `Session::new`. `onnx-std` was added as a cpu-plugin dev-dep.

Kept binary (with reason), grouped by category:
- **External weight data (`model.onnx.data` sidecar):** `tests/fixtures/tiny-llm-sharedbuffer`
  (its purpose is the external-data/shared-buffer load path), `tests/fixtures/tiny-glm52-qmoe-indexshare`,
  and `crates/onnx-runtime-ep-cpu/tests/fixtures/qmoe_weight_offload` (initializers
  reference external `weights.bin`). Textproto has no external-data directory context.
- **Executed by real ONNX Runtime (ort C API cannot parse TextFormat):**
  `crates/onnx-genai-ort/tests/fixtures/speculator-eagle3/model.onnx` (8-byte placeholder,
  ORT-loaded), and the 9 `crates/onnx-genai-genai-config/tests/fixtures/vlm-*/*.onnx`
  ORT-GenAI package model files (referenced by filename in `genai_config.json`).
- **Byte placeholders (not real ONNX):** the 3
  `crates/onnx-model-package/tests/fixtures/valid-package/*/model.onnx` (0-byte).
- **Intentional dual format:** `tests/fixtures/tiny-llm-scatter` deliberately carries both
  `model.onnx` and `model.onnx.textproto` for serialized-ONNX stress; binary left in place
  (`prefer_binary_onnx_twins` selects it).
- **Converted-then-reverted:** `tests/fixtures/tiny-native-scalar-gqa`. Its sole test
  (`engine_native_scalar_gqa_runs_without_metadata_permission`) fails in this environment
  with a pre-existing Resource-Governor "KV page geometry unknown" error — verified to fail
  **identically with the original binary**, so the failure is unrelated to format. Because a
  converted fixture must be validated by a green test and this one cannot be, it was reverted
  to binary to honor the no-unvalidated-conversion rule (it is otherwise structurally convertible).

**Why:** Binary `.onnx` blobs are opaque in review and diffs; textproto is
line-diffable, greppable, and reproducible, while our loader parses it transparently.
Restricting conversion to inline-weight, loader-consumed fixtures keeps the ORT and
external-data load paths (which genuinely need binary) intact. Each conversion was
round-trip verified (binary→Model→textproto→re-parse→identical `ModelProto` bytes, plus
matching loader graph shape) before the binary was removed, and every touched crate's
test suite was re-run green.
