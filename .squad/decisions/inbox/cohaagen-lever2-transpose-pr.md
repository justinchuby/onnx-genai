# Lever 2 PR-1 — CUDA Transpose capture-enablement (persistent metadata)

- Date: 2026-08-01
- By: Cohaagen
- Branch: `squad/lever2-transpose-capture`
- Lineage: #443/#543 (capture correctness), Lever 2 of `cohaagen-27b-decode-perf.md`
- Review: Harry (author-lockout, no auto-merge)

## What & why

The CUDA `Transpose` kernel used `launch_metadata`, which on **every call** did
`alloc_raw` + `htod` (perm/stride upload) + `synchronize()` + `free_raw`. The
`synchronize()` exists solely to guard the immediate `free_raw`, and it serializes
the stream on every one of the ~624 Transpose calls per 27b decode step — in BOTH
the eager (Scan-child) and captured paths. It also makes Transpose report
`CaptureSupport::Unsupported`, so it can never fold into the parent CUDA graph.

The fix migrates `Transpose` to the in-repo `PersistentMetadata` +
`launch_persistent_metadata` pattern already blessed for **Expand** (movement.rs)
and **Slice**: the perm/stride metadata buffer is cached and reused, so a
fixed-decode-signature Transpose uploads/syncs **once** at warm-up and then runs
with **no per-call alloc/htod/sync**. Once warmed on its exact
shape/dtype/perm signature, the kernel reports `CaptureSupport::Supported` and
errors if the signature changes mid-capture (identical guard to Expand).

**Byte-exactness:** identical kernel (`transpose_bytes`), identical metadata bytes;
only the metadata-buffer lifecycle changes. No numerics change.

## Blast radius

- ONLY `crates/onnx-runtime-ep-cuda/src/kernels/movement.rs` (TransposeKernel +
  TransposeFactory) and one new EP test.
- Shared capture machinery (`capture.rs::subgraph_graph_capturable`,
  `capture_quarantine_ops`, segmenter, run.rs) **untouched** — Transpose simply
  begins reporting `Supported` once warmed.
- **Tile deferred** (separate PR): it shares `launch_metadata` but ALSO reads
  `repeats` device→host via `host_ints`, so it needs repeats-warming — not the
  same trivial mechanism.

## Correctness evidence (all pass)

- **New EP unit test** `transpose_warmed_metadata_captures_and_matches_eager`
  (tests/construction_gpu.rs): unwarmed → declines capture; first eager run warms
  → `Supported`; cached metadata re-run byte-identical; capture+replay
  byte-identical to eager. Non-vacuous both directions.
- **Full `construction_gpu` EP suite:** 19/19 pass (Transpose/Expand/Slice/Concat/
  Split/Tile capture-eligibility + parity) — capture surface unregressed.
- **27b oracle gate** `native_autoderive_io_cuda_e2e::stock_export_auto_derives_io_and_matches_cpu_oracle`:
  PASS — stock qwen3.6-27b-int4 native CUDA == CPU fp32 oracle, byte-identical.
- **Small-model gate** `qwen3_0_6b_native_cuda_e2e`: PASS — decode matches ORT / pinned golden.
- **Small-model decode** (profile_native) token IDs `[12095, 11, 323, 279, 6722,
  315, 15344, 374, 21718, 13, 576, 6722, 315, 9625, 374, 1083, 279, ...]` ==
  pinned golden (`qwen3_0_6b_native_cuda_e2e.rs:24`).
- ORT-CUDA crashes on the 27b → CPU fp32 is the oracle (per #384 lineage).

## Measured yield (stock qwen3.6-27b-int4, CUDA, --steady --decode-skip 8 --warmups 2 --runs 3 --tokens 64)

| config              | decode ms/tok | tok/s | first 16 token IDs |
|---------------------|--------------:|------:|--------------------|
| baseline (origin/main) | 165.27     | 6.05  | [11751, 13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 11751, 369, 279, 6511, 321] |
| this PR (Transpose desync) | 156.74 | 6.38  | [11751, 13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 11751, 369, 279, 6511, 321] |

- Delta: **-8.5 ms/tok (~5.4% faster, ~1.05×)**, token IDs **byte-identical**.
- Honest caveat: this is one op of the Lever-2 non-Scan swarm. It does **NOT**
  beat ORT (17.38 tok/s / 57 ms/tok) alone — that is the two-lane result
  (Lever 2 full swarm + Inc-1b Scan-body capture). The body-internal Transposes
  get the desync benefit now; their capture-fold benefit needs Inc-1b (the Scan
  child never captures today).

## Verdict

Bounded, byte-exact, real (non-no-op) down-payment on Lever 2. Ship for Harry review.
