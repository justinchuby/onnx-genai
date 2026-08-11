# Roy — small fixes (#701, #467)

## #701 — ORT paged-KV reuse guard for hybrid recurrent state
- Added `ort_session_has_recurrent_state(session, io)` in
  `crates/onnx-genai-engine/src/kv_bridge.rs`, mirroring the native
  `has_recurrent_state()` gate (#700). It is purely structural (RULES.md §2):
  a state counts as recurrent only when the model I/O spec declares it as a
  loop-carried `state_pair` input AND that input's shape has a static
  penultimate (feature) axis.
- Threaded the check into the ORT paged-reuse **decision** in
  `Engine::prepare_session_prefix` (`engine/runtime.rs`) via a new
  `Engine::ort_session_has_recurrent_state()` accessor — NOT deep inside
  `load_materialized_past`.
- Guaranteed no-op for every attention-only model loadable today (they declare
  no `state_pairs`); guard only trips for hybrid recurrent models, forcing a
  correct full recompute.

## #467 — dedup canonical model-dir-missing error literal
- Hoisted the triplicated `"model directory does not exist: {}"` in
  `crates/onnx-genai-ort/src/loader.rs` to a single
  `fn model_dir_missing_err(root: &Path) -> OrtError`, referenced at all three
  `pub fn load*` sites. Produced error text is byte-identical.

## Incidental (unblock verification)
- `engine/load.rs` `ModelIoSpec` construction (native-backend) was missing the
  `kv_layout` field added by #782 — a pre-existing build break under
  `--features native-backend`. Added `kv_layout: None` so the required
  `cargo build -p onnx-genai-engine --features native-backend` verification
  passes.

## Verification
- `cargo build -p onnx-genai-ort` ✓
- `cargo build -p onnx-genai-engine --features native-backend` ✓
- `cargo test -p onnx-genai-ort loader` ✓ (9 passed; missing-dir text unchanged)
- `cargo test -p onnx-genai-engine --lib kv_bridge` ✓ (23 passed, incl. 2 new)
- `cargo fmt --all` ✓
