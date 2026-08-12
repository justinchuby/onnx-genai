# Decision: vestigial sliding_window classified as global → shared-buffer KV path

**By:** Deckard (Systems Dev), req. by Justin (@justinchuby) via Muse-Glimmer-30B decode-perf campaign
**Date:** 2026-08-12
**Branch:** squad/decode-path-swa-classify

## What

`detect_model_decode_path` (`crates/onnx-genai-engine/src/decode/metadata.rs`) no
longer routes a model to the capture-unstable growing/paged KV path
(`PastPresent { shared_buffer: false }`) merely because
`inference_metadata.yaml` declares a `sliding_window`. A declared window is now
treated as **active only when the exported decoder graph actually enforces it**
— i.e. an ORT `GroupQueryAttention` (or related attention op) carries a positive
`local_window_size` attribute (graph-truth). A metadata-only/vestigial window
(GQA with no `local_window_size` → global attention) falls through to the
shared-buffer / fixed-capacity path.

## Why

Muse-Glimmer-30B was MISCLASSIFIED: its 52 GQA ops carry **no**
`local_window_size` (verified: `local_window_size` occurrences = 0 in
`decoder/model.onnx`; `do_rotary` = 52), and its `genai_config.json` declares
**no** sliding_window with `past_present_share_buffer: true`. The
`sliding_window: 2048` existed only in our generated `inference_metadata.yaml`.
That vestigial window forced the growing/paged path, which is capture-unstable
(non-fixed KV addresses/shapes) and blocks CUDA-graph capture — the only lever
to lift native decode from ~11.4 tok/s toward ORT's ~40 (decode is
dispatch-bound; ~1600 launches/token, GPU ~99% idle — Sebastian's diagnosis).

## How (graph-truth SWA detection)

- New `graph_enforces_sliding_window(&onnx_runtime_ir::Graph)`: true iff any
  attention op (`GroupQueryAttention`/`MultiHeadAttention`/`Attention`/
  `SparseAttention`) has `local_window_size` Int > 0 (ORT default -1 = global).
  Recurses into control-flow subgraph bodies.
- New `effective_sliding_window(declared, graph)`: returns the declared window
  only when the graph enforces it; drops it (→ `None`) when the graph computes
  global attention. **Conservative when no graph is available** (window kept) so
  real SWA models whose graph we cannot read are never regressed.
- `detect_model_decode_path` gained a `sliding_window_graph: Option<&Graph>`
  param and applies `effective_sliding_window` before path selection (also
  prevents a vestigial window from falsely blocking the static-cache path).
- `resolve_metadata_and_decode_path` (engine/load.rs) loads the decoder graph
  via `onnx_runtime_loader::load_model` (graph interface only — external 15GB
  weights stay as descriptors) **only when a window is declared**, and passes it
  in.

## Regression guarantee (Gemma/Mistral SWA preserved)

- Unit tests (`decode/tests.rs`): graph with `local_window_size>0` enforces;
  graph without (or -1/0) does not; subgraph recursion; `effective_sliding_window`
  drops vestigial, preserves real, stays conservative without a graph.
- Session-backed test (`kv_bridge.rs::vestigial_metadata_window_routes_to_shared_buffer`):
  same real session + window=2, a **vestigial** graph → `shared_buffer: true`
  (capture-stable); a **real windowed** graph → `shared_buffer: false,
  sliding_window: Some(2)` (unchanged windowed paged path).

## Coordination

- Touched `decode/metadata.rs` (added helpers + one param to
  `detect_model_decode_path`), `engine/load.rs` (2 call sites +
  `resolve_metadata_and_decode_path` signature), `decode/tests.rs`,
  `kv_bridge.rs` (tests). Flag for Batty if his LOAD fix (Blocker 1) also edits
  `decode/metadata.rs`.
- No KV/buffer-lifecycle changes — pure classification (Leon unaffected).
- End-to-end Muse-Glimmer decode not measured here (blocked on Batty's load
  path); Sebastian measures captures/launches/tok-s once load+classify land.
