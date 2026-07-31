# GAP-3 Inc-C — native present-KV mirroring → paged native pipeline decode

**Author:** Cohaagen (EP/runtime)
**Branch:** `feat/gap3-inc-c-present-kv` (off `origin/main` `8612907f`, includes merged Inc-A #565)
**Status:** implemented, tests green — ready for Mary's independent review. NO PR opened.

## STEP 0 — independence check (result: NOT blocked)

Closing the S2 bail (`pipeline/decoder_component.rs`
`NativePipelineDecoder::mirror_last_present_kv`) does **not** require the parked
per-EP `CudaGraphLifecycle` handle-keyed device-graph registry (the deferred Scan
blocker). S2 is **host-KV plumbing**, not device-graph capture:

- The mirror **reads** the decoder's accumulated present KV from the native
  session's growable host tensors (`self.past`, keyed by past-input name) and
  **writes** freshly-decoded tokens into the paged cache through the *same*
  `kv_bridge` primitives the ORT decoder uses (`extract_present_token` +
  `PagedKvCache::append_token_kv`).
- The seed (prefix reuse) **reads** a materialized paged prefix and writes it
  back into `self.past`.

Neither path touches `provider.rs plan_capture_region`, the capture executor, or
any CUDA-graph lifecycle handle. Confirmed independent → proceeded.

## Scope shipped

**Host-resident growable f32 KV path only.** The full paged round-trip — present-KV
**mirror-write** + shared-prefix **seed-read** — for native multi-component decode
on the non-paged→paged transition. This is what makes the non-vacuity bar real:
write-only mirroring never affects tokens within a single generation, so Inc-C
implements the whole round-trip and proves it via cross-request reuse.

Gated in by `NativeDecodeSession::supports_host_kv_mirror()` — true only when the
session keeps KV as host-growable rank-4 f32 tensors (`cuda.is_none() &&
cpu_kv.is_none()` and every KV past input is `Float32` rank-4). Device-resident
(CUDA) and in-place-GQA (CPU) present-KV read-out is **Inc-D** (see below); those
decoders keep the Inc-A non-paged flat-AR path — no regression.

## What was wired

### `native_decode/mod.rs` (+3 methods on `NativeDecodeSession`)
- `supports_host_kv_mirror()` — the gate above.
- `host_present_kv(past_name) -> Option<(Vec<f32>, Vec<usize>)>` — reads the
  growable host present tensor (`[1, num_kv_heads, total_len, head_dim]`).
- `seed_growable_kv(entries, current_len)` — writes materialized-prefix KV back
  into `self.past` and sets `current_len` (so the next `decode_with_step_inputs`
  `past_len == current_len` assertion holds). Bails on device/in-place sessions.

### `pipeline/decoder_component.rs` — S2 closed
- `PipelineDecoderComponent` trait: added default methods `supports_paged_kv()`
  (default `true`; native overrides with the host-KV gate) and
  `load_paged_prefix(kv_model, materialized)` (default bails — only native/ORT
  paging-capable decoders implement it).
- `NativePipelineDecoder::mirror_last_present_kv` — **the S2 bail is gone.** Reads
  `host_present_kv` per layer, then for each freshly-decoded token position
  `retained_past_len + offset` slices with `extract_present_token(...)` and
  appends via `cache.append_token_kv(...)` — byte-identical geometry to the ORT
  `mirror_present_kv_to_pages`.
- `NativePipelineDecoder::load_paged_prefix` — builds `[1, num_kv_heads, seq,
  head_dim]` per layer from `kv_model.layer_configs` + `materialized.layers`
  (same layout ORT `materialized_past_values`/`past_shape` inject) and calls
  `seed_growable_kv`. Bails on `start_position != 0 || sink_len != 0`
  (discontinuous attention-sink prefixes = Inc-D, matching ORT's own restriction).

### `pipeline/flat_autoregressive.rs` — construction wiring + DRY refactor
- Native decoder built **up front** (before the paged gate) so its KV can be
  seeded before the loop; `native_supports_paging` computed from
  `supports_paged_kv()`.
- `paged_enabled` now admits a paged-capable native decoder:
  `self.paged.is_some() && digest.is_some() && (!use_native_decoder ||
  native_supports_paging)`.
- **DRY:** factored the shared claim/lookup/materialize logic into
  `claim_paged_prefix(...)`; `admit_paged_sequence` (ORT → `load_materialized_past`)
  and the new `admit_native_paged_sequence` (native → `decoder.load_paged_prefix`)
  are thin wrappers over it. Only the KV *sink* differs; the paging machinery is
  shared. No parallel paging path invented.

**No changes** to the decode loop (`paged_decode.rs` / `PipelineDecodeLoopBackend`),
`native_decode/{backend,cuda,cpu}` core, capture core, `provider.rs`, or S2's
neighbors. Loop is backend-agnostic; it just no longer hits a bail.

## Present-KV geometry

- Paged layer KV layout = `[num_kv_heads, seq, head_dim]` row-major =
  `[1, H, seq, Dh]` — identical for ORT inject (`materialized_past_values`),
  native seed (`load_paged_prefix`), and native present read (`host_present_kv`).
- Per-token extract/append operate on `[num_kv_heads, head_dim]` slices via the
  shared `extract_present_token` + `append_token_kv`, so native and ORT mirror
  **byte-identical pages**. Absolute token index = `retained_past_len + offset`
  (`retained_kv_len` returns `past_len`, no sliding window this increment).

## Correctness evidence (two-tier, token-exact)

Test: `tests/native_pipeline_backend_selection_parity.rs::native_paged_prefix_reuse_matches_fresh_and_ort`
(fixture `tiny-gemma4-vlm`, naive/Concat-KV decoder → host-growable f32 → paged).

Two prefix-sharing requests on **one** pure-native engine (`page_size = 2` to force
multi-page sharing):
1. First turn primes the prefix cache via the native present-KV **mirror**.
2. Second turn reuses it via `load_paged_prefix` **seed**.

Asserted together:
- **Reuse engaged:** `prefix_reused_tokens == 4 > 0` — the mirror populated the
  pages and the seed consumed them (a silent full-prefill fallback reports 0).
- **Differential:** warm native == cold pure-native run (`[7, 0, 5]`).
- **ORT oracle:** warm native == cold ORT paged decode (`[7, 0, 5]`).

**Non-vacuity:** (a) if construction reverts to the S2 bail, the paged native run
`?`-errors; (b) if the mirror/seed geometry is wrong (mismatched head/page/seq
offset), the warm tokens diverge from the cold/ORT oracles and the asserts fire;
(c) if reuse silently no-ops, `reused > 0` fails. Only a geometrically-correct
round-trip passes all three.

## Regressions re-run (cuda,native-backend, `CUDA_VISIBLE_DEVICES=2`) — all green

- Inc-A #565 `native_pipeline_backend_selection_parity` — 2 passed (incl. new case).
- #384 `native_pipeline_decoder_parity` — 14 passed.
- #541 `native_cuda_captured_step_inputs_parity`, `native_step_component_parity`,
  `native_cuda_pipeline_decoder_parity` — passed.
- #543 `qwen35_0_8b_hybrid_text_decode_e2e` — passed (hybrid env-flag path unchanged).
- #554 `multimodal_reuse_e2e` (session/prefix reuse) — 1 passed.
- #544 `weight_offload_native_cuda_e2e` — ignored (needs real int4 export; env-gated).
- lib unit tests — 350 passed.
- `cargo fmt --all --check` clean; clippy (cuda,native-backend,--tests) clean;
  no-feature build clean (cfg gating verified).

## Does 35B-A3B decode natively end-to-end yet? — **NOT on GPU. Needs Inc-D.**

Inc-C unblocks **paged native pipeline decode for host-growable f32 KV** decoders.
Qwen3.6-35B-A3B on GPU keeps its present KV **device-resident** (and GQA-in-place),
so `supports_paged_kv` is `false` and it stays on the Inc-A non-paged path. The
present-KV *threading contract* is now proven correct and DRY on the host path;
extending it to the device path is mechanical read-out, not new geometry.

### Inc-D gap (precise)
1. **Device-resident present-KV read-out** — mirror CUDA `DecodeCudaState` present
   KV via `DeviceIoBinding::read_bytes` (copy_to_host) into the same
   `extract_present_token` path; seed back via device upload. (This is the GPU
   blocker for 35B-A3B.)
2. **In-place-GQA CPU KV** — read/seed `DecodeCpuKvState` (or run with
   `ONNX_GENAI_CPU_INPLACE_KV=0` to fall onto the growable path already covered).
3. **f16 / non-rank-4 caches** — lossless round-trip through the paged store.
4. **MoE routed-expert specifics** (mobius#82 territory) — only if 35B-A3B needs
   more than present-KV threading; **not** pulled into Inc-C.
5. **Discontinuous attention-sink prefixes** (`start_position != 0 || sink_len != 0`)
   — currently bails in both native and ORT seed.

## Handoff
- **Inc-D:** device/in-place/f16 present-KV read-out (items 1–3, 5) → true 35B-A3B
  GPU paged native decode.
- Capture-core / device-graph-registry stays parked; Inc-C did not touch it.
