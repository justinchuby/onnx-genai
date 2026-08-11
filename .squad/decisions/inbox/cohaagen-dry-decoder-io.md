# Decision: DRY decoder-io derivation glue into a shared helper

**Author:** Cohaagen (Rust engineer)
**Date:** 2026-08-11
**Branch:** `squad/dry-decoder-io`
**Scope:** `onnx-genai-engine`, `onnx-genai-genai-config`

## Context

Two functions built an identical `ModelIoSpec` from a graph-derived
`DerivedDecoderIo` plus the graph port names, with ~40 lines duplicated
verbatim:

1. `crates/onnx-genai-engine/src/native_decode/load.rs` →
   `NativeDecodeLoad::derive_fallback_io` (graph source = live
   `InferenceSession` ports).
2. `crates/onnx-genai-engine/src/engine/load.rs` →
   `maybe_fill_hybrid_io_from_graph` (graph source = disk graph via
   `decoder_graph_info_from_model_path`, gated on `#[cfg(feature =
   "native-backend")]`).

Both shared the same steps after obtaining a `ModelGraphInfo`: canonical
`GenAiConfig::derive_decoder_io_from_graph` classification, the
non-empty-`state_pairs` recurrent-hybrid safety gate, name-presence binding of
the conventional non-KV ports, and the `ModelIoSpec` assembly.

## Decision

Extracted **one** shared helper:

```rust
// crates/onnx-genai-genai-config/src/compatibility.rs (impl GenAiConfig)
pub fn derive_model_io_spec_from_graph(
    graph: &ModelGraphInfo,
) -> Option<onnx_genai_metadata::ModelIoSpec>
```

`onnx-genai-genai-config` already depends on `onnx-genai-metadata` (no cycle),
and it owns `DerivedDecoderIo`/`ModelGraphInfo`, so it is the natural home. The
helper encapsulates: canonical derivation → empty-`state_pairs` gate →
name-presence binding → `ModelIoSpec` construction.

- `derive_fallback_io` now only maps live session ports into a `ModelGraphInfo`
  (the sole session-specific part) and delegates to the helper.
- `maybe_fill_hybrid_io_from_graph` keeps its authoritative `io.is_some()`
  early-return and disk graph load, then delegates to the helper and, on `Some`,
  sets `metadata.model.get_or_insert_with(default).io = Some(io)`.

## Constraints preserved

- **Behavior-preserving refactor.** The `io.is_some()` authoritative-wins gate
  and the `state_pairs.is_empty()` recurrent-hybrid safety gate are unchanged.
  (Incidentally, the engine-side spec now also carries `kv_layout: None`
  explicitly, matching the native-decode spec — the two are unified.)
- `#[cfg(feature = "native-backend")]` gating on
  `maybe_fill_hybrid_io_from_graph` retained.
- Removed now-unused `LoopStatePair` / `BTreeMap` imports in
  `native_decode/mod.rs`.

## Validation

- `cargo fmt --all`
- `cargo build -p onnx-genai-engine --features native-backend` — clean, no warnings.
- `cargo test -p onnx-genai-genai-config derive` — 5 passed (incl. 2 new helper
  tests: dense → `None`, hybrid → `Some` with correct bindings).
- `cargo test -p onnx-genai-engine --features native-backend --lib native_decode`
  — 68 passed.
