### 2026-07-31 — Generalize ORT cached-value cloning to all POD dtypes

**By:** Harry (shape-inference/ORT sub-agent), requested by Justin.

**What:**
Replaced the two per-dtype bail sites in ORT value cloning with a single
general, dtype-agnostic raw-bytes fallback, keeping the existing typed fast
paths (f32/f16/bf16/i64) untouched.

- `crates/onnx-genai-engine/src/decode/values.rs` `clone_value()` — the terminal
  `dtype => anyhow::bail!("unsupported cached ORT value dtype: {dtype:?}")` arm is
  now `dtype => Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), value.shape(), dtype)`.
- `crates/onnx-genai-ort/src/value.rs` `clone_owned()` — the terminal
  `other => Err(InvalidArgument("cannot clone tensor with dtype ..."))` arm is now
  `other => Value::from_raw_bytes(self.as_raw_bytes()?.to_vec(), &self.shape, other)`.

Coverage before → after:
- **Before:** only Float32 / Float16 / BFloat16 / Int64 could be cloned; every
  other dtype errored. Bool in particular blocked gemma-3n (audio mask is Bool),
  and Int32 / Uint8 cached inputs were equally rejected.
- **After:** all remaining POD dtypes round-trip through the shared
  `to_raw_bytes`/`from_raw_bytes` seam — **Bool, Int8, Int16, Int32, Uint8,
  Uint16, Uint32, Uint64, Float8E4M3, Float8E5M2** — with zero per-dtype code.
  `DataType::{size_of,to_onnx}` and `create_tensor_with_data` already handle every
  variant, so the fallback is exhaustive by construction.

**Why:** Justin's standing directive — fix the whole class, DRY and general, not
just Bool. A single raw-bytes fallback solves every POD dtype at once and cannot
silently regress (guarded by tests).

**Device-resident finding (investigated, as required):**
- `to_raw_bytes` does NOT check host-residency; it reads the ORT data pointer
  directly. The *existing* typed fast paths (`to_vec_f32`/`to_vec_i64` →
  `tensor_data_to_vec`, and the `zero_rank3_row`/`to_raw_bytes` readers) share
  that same host-only assumption. So the clone paths already required
  host-resident inputs for *every* dtype — this change does not introduce a new
  device hazard.
- To be strictly safer than the pre-existing typed paths, the new fallback uses
  `as_raw_bytes()` (NOT `to_raw_bytes()`), which calls `is_host_resident()` and
  returns a **precise `InvalidArgument`** for a device-resident tensor instead of
  misreading a device pointer as host memory. A stray device value therefore
  fails loudly, never corrupts silently.
- Reachability check (do device values actually reach here?):
  - `clone_owned` — its only callers are the captured-decode logits snapshot in
    `crates/onnx-genai-ort/src/decode/dynamic.rs:~889`, and the CUDA
    device-logits branch (`if cap.logits_on_device { … return; }`, ~line 862) is
    taken *before* `clone_owned` is ever called. `clone_owned` runs only on host
    logits. No device value reaches it.
  - `clone_value` — the decode/pipeline cached inputs it copies are host-resident;
    `crates/onnx-genai-engine/src/pipeline_cache.rs` reads the very same values
    with `as_raw_bytes()` (which errors on device tensors) as a matter of course.
  - Conclusion: no correct device path is needed here today; the precise-error
    fallback is the right guard if that invariant ever changes.

**Tests added (regression prevention, not happy-path only):**
- `crates/onnx-genai-ort/src/value.rs` `mod clone_owned_tests` — Bool / Int32 /
  Uint8 round-trip (dtype + shape + raw bytes identical, and clone owns a distinct
  OrtValue), plus an **empty** `[0]` Bool tensor, a **multi-dim** `[2,3]` Bool
  mask, and a typed-fast-path (i64) non-regression case. (6 tests)
- `crates/onnx-genai-engine/src/decode/values.rs` `mod clone_value_tests` — the
  gemma-3n Bool cached-input case, Int32, empty Bool, multi-dim Bool, and an i64
  fast-path non-regression case. (5 tests)

**Validation:** `cargo fmt --all --check` clean; `cargo test -p onnx-genai-ort value`
(18 lib tests green, incl. the 6 new); `cargo test -p onnx-genai-engine --lib clone`
(5 new green); `cargo build -p onnx-genai-ort -p onnx-genai-engine` clean;
`cargo clippy` on both crates clean. Env: `ONNX_GENAI_ORT_LIB` → ORT 1.27.

**gemma-3n Bool-mask blocker:** resolved *at this cloning layer* — a Bool cached
input (the audio mask) now clones instead of erroring `unsupported cached ORT
value dtype: Bool`. Any remaining gemma-3n bring-up work is outside this seam.
