# GAP-3 Inc-C — test-rigor fix (Mary, authorized reviser)

## Context
I reviewed and **rejected** GAP-3 Inc-C: the parity test
`native_paged_prefix_reuse_matches_fresh_and_ort` was **vacuous for KV
geometry**. It asserted `warm == cold == ORT` on argmax tokens, but on
`tiny-gemma4-vlm` the argmax is invariant to the reused-prefix KV, so a
key/value **swap** — and even a fully **zeroed** mirrored KV — still produced
identical tokens and PASSED. Only the no-op mirror (a) and forced-reused-0 (c)
mutations failed; the geometry-corruption mutation (b) passed. The token
asserts had no discriminating power over the mirror geometry.

Cohaagen (author) is locked out; I made the fix as the authorized reviser.

## Production behavior: unchanged (correct-by-construction)
This is a **test-only** rigor fix. The native `mirror_last_present_kv` and the
ORT `mirror_present_kv_to_pages` call the identical
`extract_present_token` / `layer_tensor_config` / `append_token_kv` primitives,
so they mirror byte-identical pages by construction. No decode-loop / native
core behavior changed.

## What I changed
1. **`crates/onnx-genai-engine/src/pipeline/mod.rs`** — added one small,
   read-only **test-support accessor** on `PipelineEngine`:
   `materialize_published_prefix_kv(&mut self, request)`. It reconstructs the
   exact prefix key the paged decode path publishes under
   (`digest_request_identity` + `prefix_key`), does a **non-mutating**
   `PrefixCache::lookup`, attaches the matched pages to a throwaway sequence
   purely to read them, materializes the per-layer K/V, then drops the
   throwaway sequence **without freeing the pages** (attach did not retain
   them, so the prefix cache is left exactly as found). Gated
   `#[cfg(feature = "native-backend")]` — i.e. only compiled in the native
   builds the paged mirror exists in, and consistent with the existing
   ungated `page_stats` / `page_usage` diagnostic methods. This is the minimal
   test-support seam; the three named files stay byte-identical.

2. **`crates/onnx-genai-engine/tests/native_pipeline_backend_selection_parity.rs`**
   — added a **direct byte/element-equality** assertion: after the warm native
   run, read the native-mirrored paged KV for the shared prefix and assert it
   equals the ORT-mirrored KV for the same prefix (`MaterializedKv: PartialEq`).
   This catches key/value swap, zeroing, head-stride, seq-offset and page-index
   errors regardless of argmax sensitivity. Kept the existing reuse>0 and token
   asserts.

## NO production code edits to the three named files
`git diff HEAD` shows **zero** changes to:
- `native_decode/mod.rs`
- `pipeline/decoder_component.rs`
- `pipeline/flat_autoregressive.rs`
Only `pipeline/mod.rs` (test-support accessor) and the test file changed.

## Non-vacuity proof — all three mutations now FAIL
Each mutation was applied to production, rebuilt, run on GPU 0, then reverted.

### (a) no-op / bailing mirror — `return Ok(())` at top of native `mirror_last_present_kv`
FAILS:
```
Error: What: cannot collect 1 KV page(s) covering 2 token(s) for sequence 0;
it holds only 0. Why: the sequence was not mirrored to that length ...
test native_paged_prefix_reuse_matches_fresh_and_ort ... FAILED
```

### (b) key/value swap in native mirror — `LayerKv { key: value, value: key }`
FAILS on the NEW byte assert (tokens still matched — argmax invariant):
```
assertion `left == right` failed: native-mirrored paged KV for the shared
prefix diverged byte-for-byte from the ORT-mirrored KV ...
  left:  MaterializedLayerKv { key: [0.5, 0.5, 0.5, 1.5, ...], value: [0.0, 0.0, 0.0, 1.0, ...] }
  right: MaterializedLayerKv { key: [0.0, 0.0, 0.0, 1.0, ...], value: [0.5, 0.5, 0.5, 1.5, ...] }
test native_paged_prefix_reuse_matches_fresh_and_ort ... FAILED
```
This is the case that PREVIOUSLY PASSED. It is now caught.

### (c) forced reused=0 — `let reusable = reusable * 0;` in `claim_paged_prefix`
FAILS:
```
paged native decode must reuse the shared prefix (reused 0 tokens);
zero reuse means the present-KV mirror never populated the pages
test native_paged_prefix_reuse_matches_fresh_and_ort ... FAILED
```

## Clean green run (mutations reverted)
```
running 2 tests
gap3 inc-c paged native reuse: reused=4 warm=[7, 0, 5] native_cold=[7, 0, 5]
  ort_cold=[7, 0, 5] prefix_kv_len=4 layers=1
test native_paged_prefix_reuse_matches_fresh_and_ort ... ok
gap3 inc-a cpu backend-selection parity: ort/hybrid/pure_native = [0,5,6,7]
gap3 inc-a cuda gqa backend-selection parity: hybrid/pure_native = [0,5,6,7]
test pure_native_pipeline_selection_matches_ort_and_hybrid ... ok
test result: ok. 2 passed; 0 failed
```
`cargo fmt --all --check` clean; `cargo clippy` clean.

## Note for Harry (re-reviewer)
Run this test binary with `--test-threads=1`. The two `#[test]` fns in this file
share a **process-global** native-decoder-device env var (documented in the file
header). Running them in cargo's default parallel mode races that var and the
CUDA-GQA test flakes with an ORT-CPU `GroupQueryAttention head_size` error —
this race is **pre-existing on HEAD** (verified: clean HEAD fails identically in
parallel and passes single-threaded), not introduced by this fix.
