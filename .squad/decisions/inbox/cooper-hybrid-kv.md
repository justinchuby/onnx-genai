### 2026-07-23: Graph-derived per-layer KV/state metadata unblocks hybrid SSM/attention decoders (qwen3.5-2b-text)
**By:** Cooper

**What:**
Made the SingleDecoder (text) metadata pipeline derive per-layer KV/state topology
*from the ONNX graph* instead of expanding a uniform `past_key_values.%d.key/value`
pattern over `0..num_hidden_layers`. Hybrid decoders (mixed dense-attention +
conv/recurrent linear-attention layers) now load and decode on the native CPU EP
without regressing uniform dense-KV models.

Per-layer state topology of `qwen3.5-2b-text-generic-cpu-1` (24 layers, `layer_types`
= 3×linear_attention then 1×full_attention, repeating):
- **Dense full-attention layers: 3, 7, 11, 15, 19, 23** — expose
  `past_key_values.N.key` / `.value` shaped `[batch, 2, past_seq, 256]`
  (+ `present.N.key` / `.value`). These map to `io.kv_inputs` / `io.kv_outputs`.
- **Linear-attention layers (the other 18): 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22** —
  expose `past_key_values.N.conv_state` `[batch, 6144, 3]` +
  `past_key_values.N.recurrent_state` `[batch, 16, 128, 128]` (+ matching
  `present.N.*`). These map to `io.state_pairs` (loop-carried recurrent state).

**Root cause (file:line):**
- `crates/onnx-genai-genai-config/src/lib.rs` — `to_inference_metadata` /
  `decoder_io_json` blindly expanded the KV name patterns over every layer index,
  so it declared `past_key_values.0.key` etc. for layers that only expose
  `conv_state`/`recurrent_state`.
- That bad contract then failed at consumption:
  - native: `crates/onnx-genai-engine/src/native_decode.rs:2513`
    → `missing native KV metadata for 'past_key_values.0.key'`.
  - ORT loop: `crates/onnx-genai-engine/src/decode.rs:609`
    → `io.kv_inputs declares input 'past_key_values.0.key' but the graph does not expose it`.
- A correct graph-driven deriver (`strict_decoder_state`) already existed but was
  wired ONLY into the multimodal/VLM builder, never the text SingleDecoder path.

**Fix (general, not a qwen3.5 hack):**
1. `onnx-genai-genai-config/src/lib.rs`:
   - Refactored `strict_decoder_state` to operate on a `&ModelGraphInfo` (port
     names + dtypes + shapes) so any decoder graph can be inspected.
   - `decoder_io_json` / `to_inference_metadata` now accept an optional decoder
     graph. When present and it exposes separate key/value ports, emits *sparse*
     `kv_inputs`/`kv_outputs` for exactly the dense layers found, plus
     `state_pairs` for every `conv_state`/`recurrent_state` port. When absent or
     the graph can't be inspected, falls back to the original pattern expansion
     (so no currently-loading model changes behavior). Added
     `to_inference_metadata_with_graph`, `inference_metadata_from_dir_with_graph`,
     `graph_decoder_state` (returns `Ok(None)` to trigger safe fallback).
2. `onnx-genai-engine/src/engine.rs`:
   - ORT/standard path (`genai_config_compat_metadata`, used at the session-present
     load) builds a `ModelGraphInfo` from `session.inputs()/outputs()` and calls
     the graph-aware metadata builder.
   - Native path (`from_native_model_directory`, which `--backend native` uses and
     which builds metadata *before* a session exists) now reads the graph directly
     from the model file via `onnx_runtime_loader::load_model` +
     `onnx_runtime_ir` (both already normal deps; external weight data is never
     read) through new `genai_config_compat_metadata_from_model_path` /
     `decoder_graph_info_from_model_path` / `ir_dtype_name`. These are
     `#[cfg(feature = "native-backend")]`-gated.
3. `onnx-genai-engine/src/native_decode.rs`:
   - The native target constructor now folds `io.state_pairs` into
     `kv_inputs`/`present_outputs`/`present_to_past` so each recurrent
     conv_state/recurrent_state port is seeded, bound and fed back. Native already
     distinguishes recurrent (fixed-extent) vs growable KV structurally via
     `is_recurrent_state_shape`, so the existing conv/linear-attention kernels
     (causal_conv.rs / linear_attention.rs) drive compute once the ports are bound.

**Verification:**
- `qwen3.5-2b-text` native load + 16-token greedy decode succeeds:
  `", I am a 20 year old male. I have been experiencing a"` (14.3 tok/s).
- No regression on uniform dense-KV `qwen3-0.6b` native decode:
  `"ED\nI need to find the value of the function $ f(x) ="`.
- Tests:
  - `onnx-genai-genai-config` (lib): 20 passed — includes new regression tests
    `hybrid_decoder_derives_sparse_kv_and_state_pairs` and
    `uniform_decoder_graph_matches_pattern_expansion` (assert sparse KV + state
    pairs for a synthetic mixed dense/conv/recurrent config, and that a uniform
    graph still matches pattern expansion).
  - `onnx-genai-engine` (lib, default features): 159 passed — matches baseline.
  - `onnx-runtime-ep-cpu --features mlas`: 827 passed (817 + 10) — unchanged.
  - NOTE: running the engine lib with `--features native-backend,mlas` shows 17
    fixture failures, but these are **pre-existing on the untouched baseline**
    (the synthetic fixture files aren't valid ONNX protobuf, so the pre-existing
    `model_requires_native_backend` proto-parse fails during backend selection —
    unrelated to this change; verified by re-running the baseline with the same
    flags).

**Remaining scope / risk:**
- ORT-loop (non-native) end-to-end decode of qwen3.5 was not smoke-tested here;
  the Part-1 metadata fix should enable it since `decode.rs` already consumes
  `state_pairs`, but it wasn't run end-to-end.
- `native_decode.rs` `kv_layer_count()` now over-counts for hybrid models
  (includes recurrent ports); it is only used for cosmetic profile_native display
  output — left as-is, flagged.
- Proposer/speculative and CUDA hybrid paths were not touched.
- Graph-aware derivation only activates when the genai_config declares *separate*
  key/value patterns; models expressing KV differently still use pattern
  expansion (safe fallback).
