### 2026-07-27: Split ORT decode by cache and session family
**By:** Dillon
**What:** Replaced `crates/onnx-genai-ort/src/decode.rs` with a facade and six focused submodules:

- `decode/mod.rs` — 201 lines; public option/signature types, batched trait, and re-exports.
- `decode/dynamic.rs` — 1,550 lines; dynamic past/present decode and captured-step tests.
- `decode/kv_growth.rs` — 465 lines; shared KV bucket growth, host/CUDA prefix copying, and tests.
- `decode/static_cache.rs` — 1,210 lines; scalar and batched static-cache sessions.
- `decode/shared_batch.rs` — 476 lines; continuous-batch shared-buffer session.
- `decode/io.rs` — 196 lines; KV-name pairing and static-cache signature detection.
- `decode/tensor.rs` — 149 lines; logits, cloning, empty tensor, and allocation helpers.

All existing public types remain available from `onnx_genai_ort::decode` through facade re-exports. The `decode_contract`-based `KvNamingConvention`, `kv_suffix`, and `name_contains_present_key_value` call sites were moved unchanged into `decode/io.rs`; no local classifier copies were introduced.

`cargo fmt -p onnx-genai-ort` was run. Gates passed:

- `cargo build -p onnx-genai-ort`
- `cargo test -p onnx-genai-ort` (all unit, integration, and doc tests)
- `cargo clippy -p onnx-genai-ort --all-targets -- -D warnings`
- `cargo build -p onnx-genai-engine`

**Why:** The original 4,239-line file mixed materially different cache ownership and batching models. The split is pure code motion and clarifies ownership without changing algorithms, allocation, CUDA annotations, or the public facade.
