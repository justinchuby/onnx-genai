# GAP-3 Inc-D.1 — f16 device-resident present-KV → paged native CUDA decode

**Author:** Cohaagen (EP/runtime) · **Branch:** `feat/gap3-inc-d1-f16-present-kv` (off origin/main `4d777636`, incl. merged Inc-D #567)
**Status:** DONE — committed, NOT PR'd. Independent opus review required (author-lockout: Cohaagen is author).
**Reviewer:** Mary or Harry (whoever is cleaner).

## What this unlocks
Real-model native **paged** multi-component decode. Inc-D only lifted the paged gate for **f32** device-resident KV; every real target model exports **f16** KV (confirmed: `gemma4-e2b-onnx/decoder` present-KV is FLOAT16; qwen3-30b-a3b likely f16). Inc-D's `dtype==f32` gate sent them ALL to the non-paged fallback. Inc-D.1 relaxes the gate to `cuda && (f32||f16) && rank==4` so f16 device-resident rank-4 CUDA GQA present-KV runs paged.

## Gate change (honest)
`DecodeCudaState::kv_bindings_f32_rank4` → **renamed** `kv_bindings_paged_rank4` (`native_decode/cuda.rs:1507`):

    matches!(binding.dtype, DataType::Float32 | DataType::Float16) && physical_shape().len() == 4

Caller = `supports_device_kv_mirror` (`native_decode/mod.rs`, doc updated). **Still gated → non-paged fallback (no silent-wrong paged run):** bf16, non-rank-4, in-place / CPU-resident caches, sink-discontinuous prefixes. Only f16 (plus the existing f32) rank-4 CUDA GQA flips on.

## The `half` convert (matches ORT exactly — requirement #1)
Host paged store stays `Vec<f32>` for f16 models (identical to ORT: `kv_bridge.rs:453-457` "Paged storage holds KV widened to f32; narrow back … the exact inverse for an fp16 model"). Two shared helpers in `native_decode/tensor.rs`, both using the **same `half` routines ORT uses** — NOT the hand-rolled `f16_to_f32` bit-twiddle (that stays logits-argmax only):

- **Read (widen f16→f32):** `kv_dtype_to_f32(tensor)` (`tensor.rs:68`) — f16 via `half::slice::{HalfBitsSliceExt::reinterpret_cast, HalfFloatSliceExt::to_f32_vec}`, matching ORT `to_vec_f32_lossy` (`onnx-genai-ort/src/value.rs:398`). f32 pass-through; bf16/other bail defensively.
- **Seed (narrow f32→f16):** `f32_slice_to_dtype_bytes(dtype, values)` (`tensor.rs:37`) — f16 via `half::f16::from_f32`, matching ORT `from_f32_slice_as` (`value.rs:173`). Refactored `tensor_from_f32_as` (embedding-input path) to share this encoder — DRY, one narrower for both paths.

### Wiring (dtype-branch, no kernel-core edits)
- `read_present_kv` (`cuda.rs:1541`): reads `binding.dtype`, `Tensor::from_raw(dtype, physical_shape, bytes)`, widens via `kv_dtype_to_f32`. (Inc-D hardcoded `DataType::Float32`.)
- `seed_prefix` (`cuda.rs:1588`): `elem_size = dtype.checked_storage_bytes(1)`, encodes bytes via `f32_slice_to_dtype_bytes(dtype, compact)`, write offset scaled by `elem_size` (was `size_of::<f32>()`). Physical/capacity strides use `max_len` (Inc-D's stride wrinkle preserved).
- Pure post-step reader — reads present-KV AFTER the decode step's existing sync (like per-step logits `read_logits`). No standard_attention/GQA kernel, capture-core, `plan_capture_region`, or ORT-side edits.

## Correctness evidence

### Fixture — `tests/fixtures/tiny-gemma4-vlm-cuda-f16` (Concat-KV, FLOAT16 KV)
Built by `scripts/build_tiny_gemma4_vlm_cuda_f16.py` (twin of the Inc-D `tiny-gemma4-vlm-cuda`). Concat-KV decoder (NOT GQA) because **GQA fixtures have no ORT-CPU oracle** — the ORT CPU GQA kernel rejects the tiny head_size (documented in the parity test). Decoder KV declared elem_type 10 (FLOAT16) for past/present key+value; logits stays FLOAT.

**KEY debugging finding — KV must be arithmetic-EXACT, not just f16.** First fixture used `value = key + 0.5`; the f16 `Add` lands on a round-to-even midpoint that CUDA and ORT-CPU kernels round differently → byte-equality broke (exactly the scoping-note Q2 kernel-divergence risk). **Fix: `value = key * 2`** (multiply-by-2 is bit-exact in f16 — pure exponent increment) and `key = Cast(embeds)`. Both bit-exact on every kernel → CUDA-vs-ORT byte-equality holds.

### Byte-equality oracle (unchanged from Inc-C/D — the crux, resolved)
Because ORT ALSO widens f16→f32 in its paged store (same `half` widen), both sides land in an f32 store → raw byte-equality still holds. The parity test asserts the device-mirrored paged-KV `MaterializedKv` == the ORT-mirrored KV **byte-for-byte** for the shared prefix.

### Tests (all pass, in isolation and batched 4/4)
- `native_paged_prefix_reuse_matches_ort_on_cuda_device_f16` (new) — paged-native-f16 == non-paged-native-cold == ORT-cold oracle (tokens) **and** device-mirrored pages **byte-equal** CUDA-vs-ORT. Shares `run_device_paged_reuse_parity(fixture, label)` with the Inc-D f32 test (DRY).
- 3 conversion unit tests (`native_decode::tensor::kv_convert_tests`): `kv_f16_widen_matches_half_reference` (native widen == `half::f16::to_f32` reference), `kv_f16_roundtrip_is_bit_exact` (f16→f32→f16 bit-exact by construction), `kv_f32_widen_is_identity`.
- Inc-D H=2 stride test `device_kv_view_uses_physical_stride` still green (geometry unchanged).

### 3 non-vacuous mutation proofs (apply → rebuild → run f16 test `--test-threads=1` → FAIL → revert)
- **(a) revert gate to f32-only** (`cuda.rs:1507` → `Float32` only) → f16 fixture falls to non-paged → `reused 0` → `reused > 0` assert FIRES. ✓
- **(b) wrong convert** (`kv_dtype_to_f32` f16 branch → `bits.iter().map(|b| f32::from(*b))`, i.e. raw u16 bits instead of widen) → mirrored KEY = 15360.0 (raw bits of f16 1.0) vs correct 1.0 → **byte-equality assert FIRES**. ✓
- **(c) forced reused=0** (disable mirroring: `paged.as_mut().filter(|_| false)` in `paged_decode.rs:246`) → `reused 0` → `reused > 0` assert FIRES. ✓
All three reverted; clean tree = only the intended diffs.

### Regression (no regressions)
- Parity suite (Inc-A/C/D + new f16): **4/4 pass** together and in isolation.
- `#554` multimodal_reuse_e2e: 14/14 ok. `multi_session`: 2/2 ok. `#541` native_cuda_captured_step_inputs_parity: ok. `#543` qwen35_0_8b_hybrid_text_decode_e2e: ok.
- Real-model e2e correctly `ignored` without local ONNX export: hybrid_native_cuda (#543), weight_offload (#544).
- `cargo fmt --all --check`: clean.

## Does gemma4-e2b now decode natively paged?
**Fixture-level: yes** — f16 device-resident rank-4 CUDA present-KV now mirrors into the paged cache byte-exact vs ORT. gemma4-e2b's decoder is exactly this shape (FLOAT16 rank-4 KV), so its present-KV threading is now on the paged path. Not smoke-tested on the real model end-to-end (real gemma4-e2b also blocked by the stale vision export, orthogonal to this KV gap); fixture-level validation is the oracle, per plan.

## Remaining gaps / handoff
- **Inc-D.2 (bf16):** `qwen3-30b-a3b` is `torch_dtype=bfloat16`; if its ONNX export keeps bf16 KV it stays gated → non-paged. Trivial follow-up: `f32_slice_to_dtype_bytes`/`kv_dtype_to_f32` already have the bf16 arms structured; just flip the gate to include `BFloat16` and add a bf16 fixture + widen unit test. **Byte-equality caveat:** must first confirm ORT widens bf16→f32 in its paged store the same way (same crux as f16 — verify before flipping).
- **qwen3-30b-a3b MoE:** MoE FFN routing produces NO KV (orthogonal); num_kv_heads=4/head_dim=128 = standard rank-4 GQA. So present-KV dtype is the only decode-path gate. Local model is HF safetensors (no ONNX export) → no local A3B end-to-end oracle; unblocked pending an f16/bf16 export.
- Non-rank-4, sink-discontinuous, CPU-in-place-GQA remain gated (later increments).
