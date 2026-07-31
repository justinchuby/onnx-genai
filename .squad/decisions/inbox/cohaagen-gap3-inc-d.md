# GAP-3 Inc-D — device-resident present-KV read-out → paged native pipeline decode

**Author:** Cohaagen (EP/runtime) · **Branch:** `feat/gap3-inc-d-device-present-kv` (off origin/main `096dfbca`, includes merged Inc-C #566)
**Status:** implemented, tested green on CUDA (GPU 2, H200). NOT PR'd — awaiting independent opus review (Mary/Harry).

## What Inc-D lifts

Inc-C paged only the **host-growable f32** KV path; a native **CUDA** decoder (present-KV in device
bindings) reported `supports_paged_kv=false` and fell back to the Inc-A non-paged flat-AR path. Inc-D
lifts that gate for **device-resident f32 rank-4 CUDA present-KV**, so a native CUDA pipeline decoder
runs **PAGED** (cross-request KV reuse) — the last runtime gap for Qwen3.6-35B-A3B GPU decode of the
present-KV-threading kind. f16 / non-rank-4 / in-place-CPU KV stay honestly gated → non-paged fallback
(no silent-wrong paged run).

## What wired (construction/plumbing only — no kernel-core, no capture-core edits)

1. **`DecodeCudaState::read_present_kv(&mut self, past_name)`** (`native_decode/cuda.rs`): pure
   **post-step** reader. Finds the KV binding whose `input_name == past_name`, `read_bytes()` →
   `Tensor::from_raw(f32, physical_shape)` → `to_vec_f32()`. Returns the **full capacity-padded**
   buffer paired with the **physical/capacity** shape `[1,H,max_len,head_dim]`. Runs after the decode
   step's own device→host sync (KV bindings hold committed present-KV once the step returns), exactly
   like the per-step logits read at `read_logits` — **no new synchronize**, and the GQA in-place alias
   is intra-kernel so a post-step reader is race-free (per the scoping note).
2. **`DecodeCudaState::seed_prefix(&mut self, session, entries, seq_len)`**: device counterpart of the
   host `seed_growable_kv`. Grows the bucket if needed (`ensure_capacity`), writes each head's compact
   prefix into its **capacity-offset** slot (`write_bytes(head * max_len*Dh * 4, …)` — per-head ranged
   writes because the device buffer is capacity-strided while the prefix is compact), advances KV
   logical length (`set_logical_len`), and marks the mask prefix `[0, seq_len)` attendable
   (`extend_mask`, since the per-step decode only extends `[seq_len, total)`).
3. **`DecodeCudaState::kv_bindings_f32_rank4`** — the device gate predicate (every KV binding f32 &
   physical rank 4).
4. **`NativeDecodeSession` unification** (`native_decode/mod.rs`): `present_kv(&mut self)` and
   `seed_kv(&mut self)` dispatch host-growable (Inc-C) vs device-CUDA (Inc-D) by which store the
   session keeps; `supports_device_kv_mirror` mirrors `supports_host_kv_mirror` for the device path.
   Both device and host feed the **same** `extract_present_token` + `append_token_kv` geometry into the
   **same host f32 paged store** — DRY, byte-comparable with ORT.
5. **`decoder_component.rs`**: `NativePipelineDecoder::mirror_last_present_kv` now calls the unified
   `present_kv`; `supports_paged_kv = host OR device`; `load_paged_prefix` calls the unified `seed_kv`.
   `mirror_last_present_kv` signature `&self → &mut self` (device read mutates transfer bookkeeping) —
   rippled to the **trait decl** + the **ORT impl** (mechanical); the paged loop already holds the
   decoder mutably at `paged_decode.rs:253`, so no call-site change.

No changes to `native_decode` decode kernels, `standard_attention`/GQA, the parked capture core, or
`provider.rs plan_capture_region`. This is pure post-step read + pre-step seed plumbing.

## Physical-shape stride handling (the "max_len wrinkle")

The device buffer is allocated at `max_len` in the sequence axis and never re-packed, so its head-axis
row-major stride is `max_len*head_dim`. `read_present_kv` therefore pairs the capacity buffer with the
**physical** shape (not the logical valid shape) so `extract_present_token`'s strides address the padded
buffer. Feeding the logical valid shape would compute a head stride of `valid_len*head_dim` and, when
`max_len > valid_len` **and `H > 1`**, silently read the wrong head rows. This single decision is
isolated in the pure fn **`device_present_kv_view(buffer, physical_shape, logical_shape)`**.

> **H=1 caveat (important for reviewers):** every existing CUDA fixture has `num_kv_heads == 1`, at
> which the head stride is unused and the physical-vs-logical distinction is invisible. So the stride
> bug is proven by a **deterministic H=2 unit test**, not the integration fixture (see mutation (b)).

## Evidence (token-exact + byte-equality, two-tier)

Fixture `tiny-gemma4-vlm-cuda` (device-resident KV on the CUDA EP), native decoder pinned `cuda:0`.

- **Differential (tokens):** `native_paged_prefix_reuse_matches_ort_on_cuda_device` — paged-native-CUDA
  warm == cold pure-native-CUDA == ORT oracle == closed-form ids `[0,5,6,7]`.
- **Byte-equality:** the CUDA-mirrored paged-KV for the shared prefix is **byte-identical** to the
  ORT-mirrored KV (`materialize_published_prefix_kv`, reuse-independent). This fixture's argmax is
  invariant to the reused-prefix KV (KV is a Concat pass-through, not read for logits), so the byte
  comparison — not the tokens — is what catches a device read/seed **value** error.
- **Reuse engaged:** warm request reuses `> 0` prefix tokens (device mirror-write populated pages,
  device seed-read consumed them).
- **Inc-C host path** (`native_paged_prefix_reuse_matches_fresh_and_ort`) still green through the unified
  `present_kv`/`seed_kv`.

### Non-vacuity — 3 mutation proofs (apply → rebuild → run `--test-threads=1` → revert)

| # | mutation | test that fires | observed failure |
|---|----------|-----------------|------------------|
| (a) | gate revert: `supports_device_kv_mirror` → `false` | `native_paged_prefix_reuse_matches_ort_on_cuda_device` | `paged native CUDA decode must reuse the shared prefix (reused 0 tokens)` |
| (b) | wrong stride: `device_present_kv_view` returns `logical_shape` | `device_kv_view_uses_physical_stride` (lib, H=2) | `device view must carry the physical shape — left [1,2,2,2] right [1,2,4,2]` |
| (c) | forced no-reuse: native `mirror_last_present_kv` publishes nothing | `native_paged_prefix_reuse_matches_ort_on_cuda_device` | `cannot collect 1 KV page(s) … it holds only 0` (no pages published → reuse impossible) |

All three revert cleanly (verified `cargo test` green after each revert).

## Regressions (all green / correctly gated)

`native_cuda_pipeline_decoder_parity` (Inc3a device KV) ✅ · `native_pipeline_backend_selection_parity`
(Inc-A/C/D, 3/3) ✅ · `native_cuda_captured_step_inputs_parity` (#541) ✅ · `qwen35_0_8b_hybrid_text_decode_e2e`
(#543) ✅ · `multimodal_reuse_e2e` (#554, 14/14) ✅ · lib unit tests 351/351 ✅ ·
`qwen35_0_8b_hybrid_native_cuda_e2e` (#541) & `weight_offload_native_cuda_e2e` (#544) → *ignored*
(real-model + CUDA gated; no export locally — Inc-D touches no offload/hybrid construction). `cargo fmt
--all --check` clean.

## Does 35B-A3B now decode natively PAGED end-to-end on GPU?

**Present-KV threading: YES for f32 rank-4 device GQA KV.** The mirror + seed geometry that MoE
present-KV threading needs now runs on the device path, paged, token- and byte-exact vs ORT.

**Precise remaining gaps (honestly gated OFF → non-paged fallback, not silent-wrong):**
- **f16 device KV.** Gate is f32-only. If the 35B-A3B export keeps device KV in **f16**, it stays
  non-paged until an **Inc-D.1** slice (f16 read-out + lossless f16↔f32 paged round-trip, or an f16
  paged store). This is the most likely blocker for the real export — confirm the 35B-A3B decoder KV
  dtype.
- **CPU in-place-GQA f32** — evaluated, **not free**: needs its own ORT-CPU-acceptable H≥2 GQA fixture.
  Left gated (Inc-D.1). No device model needs it.
- **Sink-discontinuous prefixes** (`start_position != 0`) — still Inc-D+ (mirrors ORT's own restriction).
- **MoE routed-expert specifics** (mobius#82 territory) — out of scope; not required for present-KV
  mirroring, which is what Inc-D closes.

## Handoff

- **Inc-D.1 (recommended next):** f16 device present-KV read-out (the one gate flip most likely blocking
  the real 35B-A3B export) + CPU in-place-GQA f32. Both need H≥2 GQA fixtures with an ORT oracle; the
  device path itself is proven, so Inc-D.1 is dtype/fixture work, not new plumbing.
- **Blast radius stayed normal:** no standard_attention/kernel-core, no capture-core, no staging. The
  scoping GO held — the read-out was mechanical + race-free as predicted.
