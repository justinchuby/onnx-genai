# Chew — History

## 2026-07-12: Joined
Hired as a Code Reviewer specializing in numerics/precision as the runtime took on fp16/Q4 quantization, GQA KV, and Mobius model conversion. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: a prior Q4 GGUF→ONNX conversion "loaded but produced garbage" (missing Qwen2 biases + wrong reverse-permute) and a sampling RNG bug returned token 0 — exactly the silent precision defects to catch. Verify against references; require coherent output, not just successful load.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Leon's DESIGN §40 SWA/attention-sink work and approved it with three optional LOW nits; no rejection lockout needed.

## 2026-07-13T23:15:17Z — §38 K1/K2 review

Reviewed §38 K1 (`crates/onnx-genai-kv/src/connector.rs`) and K2 (`crates/onnx-genai-kv/src/local_tiered.rs`).

- **Top risk verified clean:** `LocalTieredConnector` owns SEPARATE `PageTable`/`PrefixCache` instances from the engine's in-process cache — no refcount aliasing, no double-free risk.
- **Real defect found:** `cpu_load_ms_per_page` declared and defaulted but never read in `locate()`. Load estimate was always `on_cpu * 1.0` (implicit 1 ms/page) regardless of configured rate.
- Verdict: 🟡 **ship-with-recommendations**. Defect remediated by Zhora (commit 30ee870) before K3 landed.

### Shared context for future KV connector reviews
- The engine's `prefix_cache` (refcounted, `lookup_shared`/`release_shared`) and the connector's `PrefixCache`/`PageTable` must remain STRICTLY SEPARATE — any aliasing creates double-free risk.
- `KvTensorRef` is currently a size-only placeholder — no real KV bytes are stored/fetched yet. K4-materialize will require giving it a real device-tensor handle.
- Prefix-dependent hash invariant is now in place (Zhora, commit ac12480): `KvCacheKey` equality ⟹ identical prefix through that chunk.

## 2026-07-13T23:50:16Z — §38 K4 review

Reviewed Leon's K4 real KV byte materialization (commit `786e268`, read-only, Leon locked out). Verdict: 🟡 **SHIP-with-advisories**.

- **Byte-layout symmetry confirmed correct:** extract (`chunk_payload_from_exported`) and inject (`past_kv_from_payloads`) are symmetric; all four layout sites (extract, inject, `materialize_sequence`, past-tensor shape) agree. No transpose/stride mismatch.
- **No false hits:** prefix-dependent cumulative FNV-1a hash ensures `KvCacheKey` equality ⟹ identical prefix through that chunk.
- **Fetch-vs-recompute gate correct, no off-by-one.**
- **All deferred paths safely no-op** (non-runner, non-f32, continuing session).
- **Gold test rigorous** — simulates fresh node, asserts non-trivial fetch + token-identical output vs full recompute.
- **Advisory A1 → Pris:** `tiny-llm` fixture is single-layer; add multi-layer gold fixture to close cross-layer ordering dimension.
- **Advisory A2 → Batty:** `try_connector_kv_injection` should gracefully fall back (`Ok(None)`) on `import_runner_kv` failure instead of hard-failing `generate`.

