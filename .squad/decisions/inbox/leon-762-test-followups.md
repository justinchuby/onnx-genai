### 2026-07-30: PR #762 EP-plugin test-quality followups (3 fixes)

**By:** Leon

**What:** Closed 3 test-quality gaps found in the PR #762 EP-plugin suite where a
test claimed more than it proved.

1. **Tautological CUDA diagnostic test** (`onnx-runtime-ep-cuda-plugin`).
   `cuda_fail_closed.rs::cuda_plugin_diagnostic_message_is_actionable` used to
   assert against string literals defined inside the test body, so it always
   passed. Refactored `src/lib.rs` to expose the plugin's *real* fail-closed
   diagnostic via a new public `fail_closed_diagnostic() -> Option<String>`
   (and a shared `NO_CUDA_FEATURE_DIAGNOSTIC` const), used by both
   `CreateEpFactories` and the test. The test now asserts on the actual
   emitted diagnostic content (asserted-on-real-output; not deleted). The
   `OrtStatus` *string* still needs a live ORT host to materialize, but the
   message *content* is produced entirely by this crate, which is what we
   assert on.

2. **6 non-reproducible fixtures** (`onnx-runtime-ep-cpu-plugin`). 28 `.onnx`
   fixtures were committed but `generate_fixtures.py` only produced 22. Added
   generator functions for the 6 opaque blobs (`layer_norm_dynamic_axis`,
   `skip_layer_norm_output_sum`, `simplified_layer_norm_two_outputs`,
   `skip_layer_norm_no_beta_bias`, `clip_no_min`, `matmul_initializer_weights`)
   and wired them into `__main__`. All 28 fixtures now regenerate
   **byte-for-byte identical** to the committed blobs (`git diff` empty). Notes:
   `matmul` weights use `numpy_helper.from_array` (raw_data encoding);
   `simplified_layer_norm_two_outputs` builds attributes manually (epsilon
   before axis) with an explicit empty node domain and skips the onnx checker
   (its op isn't in the standard schema registry at opset 21).

3. **Value-blind "no_overflow" tests** (`onnx-runtime-ep-cpu-plugin`).
   `optional_slots.rs::layer_norm_{f16,bf16}_absent_output_no_overflow` only
   asserted `!output.is_null()`. Added value assertions comparing `Y` against a
   hand-computed LayerNorm oracle (added `f16_to_f32`/`bf16_to_f32` helpers),
   matching the SkipLayerNorm sibling tests. Tolerance 0.02 (f16) / 0.05 (bf16).

**Why:** A test that asserts on a local copy of a string, or only that a run
didn't crash, gives false confidence: a corrupt-but-non-crashing result or a
wrong real diagnostic would pass. Fixtures with no generator are opaque and
un-auditable. These fixes make the suite prove what its names claim. Scope was
strictly these 3 followups; no other tests weakened. ORT was available locally,
so all affected tests ran and passed (not skipped).
