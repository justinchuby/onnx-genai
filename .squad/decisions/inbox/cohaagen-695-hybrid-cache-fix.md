# Decision — Disable prefix/KV-mirror reuse for hybrid Mamba models (#695)

- Date: 2026-08-06
- Author: Cohaagen (EP/runtime + numerics)
- Branch: `squad/fix-695-hybrid-cache`
- Issue: #695 (hybrid Mamba prefix-cache reuse silently produces wrong continuation logits)

## Mechanism (confirmed in code)

The KV-mirror / prefix-reuse support gate only checked attention KV geometry and
never excluded hybrid recurrent decoders:

- `NativeDecodeSession::supports_device_kv_mirror` → `DecodeCudaState::kv_bindings_paged_rank4`
  gated on rank-4 f32/f16 **KV** bindings only
  (`crates/onnx-genai-engine/src/native_decode/mod.rs`, `.../cuda.rs:1608`).
- `NativeDecodeSession::supports_host_kv_mirror` gated on rank-4 f32 **KV** only.
- Device prefix reuse `DecodeCudaState::seed_prefix` (`cuda.rs:1679`) writes only
  `kv_binding_range`. The conv/recurrent `fixed_state_binding_range` is re-zeroed
  **only** on `rewind(0)` (`cuda.rs:1743-1760`) and never reconstructed for a
  reused prefix.

So a hybrid Mamba model (rank-4 KV **plus** a non-empty recurrent
`fixed_state_binding_range`) passed the gate: on continuation it restored
attention KV but ran a fresh-zero recurrent state, silently emitting wrong
next-token logits (Qwen3.6-35B-A3B: reused-engine argmax 279 vs fresh-engine
oracle 33803).

## Fix (correctness-first, DRY, name-free)

Both mirror-support gates now return `false` when the decoder carries recurrent
state, detected via the existing generic `has_recurrent_state()`
(graph-metadata `is_recurrent_state_shape`, no model name). `supports_paged_kv`
then declines and the engine does a full recompute on continuation.

- Single-shot generation is **unaffected and byte-identical**: fresh-engine
  decode never consults these gates and already starts recurrent state at zero.
- Longer-term perf-preserving option (persist/restore terminal recurrent state
  per cached prefix) is deferred; correctness ships now.

## Verification

- New always-on unit test `native_kv_mirror_gate_excludes_hybrid_recurrent_decoders`
  (host, no GPU): hybrid decoder → both gates false + `has_recurrent_state()`;
  pure-dense control → host mirror stays enabled.
- New GPU regression test `qwen36_35b_a3b_hybrid_continuation_matches_fresh_engine`
  (env-gated on the 35B-A3B artifact + CUDA, reuses the divergence-test helpers):
  reused post-decode engine teacher-force argmax **== fresh-engine oracle argmax
  == 33803**, tie inside the fp32 margin band. **PASSED** on GPU 0 (46s).
- Single-shot unaffected: the regression test's autoregressive `greedy_stream`
  reproduces token 33803 at index 119, byte-identical to the recorded lock.
- `cargo fmt --all --check`, `cargo clippy -p onnx-genai-engine --features
  native-backend` clean; 65 `native_decode` lib tests pass.

## Note

The pre-existing `qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle` lock needs
cuDNN on the library path (`LD_LIBRARY_PATH=.../cudnn9.19_cuda13/lib`) and runs a
very slow 35B CPU fp32-oracle leg; its cuDNN-discovery flake is environmental and
unrelated to this gate change.
