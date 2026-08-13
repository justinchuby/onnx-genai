# Sebastian — History (compacted 2026-08-12T06:00:00Z)

**Role:** Owns DESIGN §26 batched serving, runtime/server performance, and cross-runtime benchmark analysis for `onnx-genai`. Preserve `submit`/`step`/`poll` batching semantics, force single-thread ORT for exact-equality real-model tests, and use canonical benchmark/observability harnesses for runtime comparisons.

## Durable lessons
- §26 Stage A/B: `Engine::generate_batched_static` and `ContinuousBatchManager`; byte-denominated VRAM/RAM limits and transactional lowering.
- CPU decode profiling showed ORT `session.run` dominates (~98.9%); fp32 `lm_head` quantization and op fusion are major levers.
- `filter_map` is wrong wherever position or rank is load-bearing; use `map → Vec<Option<usize>>`.
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently.
- `cargo test --workspace` silently truncates on failure — always use `--no-fail-fast`.
- Never commit `.squad/` files to external repos.

## Recent work (current wave, 2026-08-12)

### 2026-08-12 — PR #31973 v2: architecture-specific dispatch threshold fix
Renamed `kAvx2DispatchThreshold` → `kKernelDispatchThreshold`. Fixed `CatastrophicCancellationPasses` to exercise accuracy branch. Renamed `AdversarialPrecisionReport` → `DISABLED_`. Removed N=7 benchmark. Head `72e02cd92c`.

### 2026-08-12 — PR #762 S1/S2/S3 resolution (commit a5448fa36)
S1: `production_scratch_alloc(numel, dtype)` helper + 2 new canary tests (`scratch_buffer_wider_write_absorbed_by_padding`, `scratch_buffer_detects_oversized_write`).
S2: `TensorMut::validate_write_dtype()` — exact match for present, byte-size gate for absent. `mark_absent()` invariant documented.
S3: `NodeOutputSink::Absent` variant — `build_subgraph_routing` no longer allocates phantom slots.
Nits: removed 4 no-op identity transmutes.
280 passed / 0 failed. Clippy clean. fmt clean. Miri: 4/4 canary tests clean.

Full pre-compaction history in `history-archive.md`.

## 2026-08-12 — CUDA-graph capture arc: escalation diagnosis + PRs #855, #854 (MERGED)

Produced the measured escalation diagnosis that framed the whole 5-blocker chain
(classify → load → pin → bf16-kernel → skip-norm), then authored links 4 and 5:
- **#855** — bf16 capture-safe `gqa_decode` kernel; 54 → 2 segments; 22.52 tok/s; f64-oracle
  max_abs 1.953e-3 (Chew 🟢 APPROVE on H200, fp32 accumulation airtight).
- **#854** — bf16 skip-norm capture-flag fix; 2 → 1 segment, **0 seams**; 23.13 tok/s
  (+33% capture ON); rebased onto main after #855 squash-merged.
Shared arc result: native CUDA decode **11.4 → 23.13 tok/s**, capture fully engaged. Next lever
(dispatched separately): Cast round-trip elimination (Cast 40.1% / 626 casts/token).
