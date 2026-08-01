# Lever 2 op-2 scope — CUDA `Tile` capture-enablement (persistent metadata + warmed repeats)

- Date: 2026-08-01
- By: Cohaagen
- Branch (planned): `squad/lever2-op2-tile-capture`
- Lineage: Lever 2 of `cohaagen-27b-decode-perf.md`; follows #574 (Transpose). #443/#543 capture invariants.
- Review: Harry (author-lockout, no auto-merge)

## Which op & why (data-backed)

Fresh post-#574 per-op profile of the stock qwen3.6-27b-int4 native CUDA decode
(`ONNX_GENAI_PROFILE_OPS=1`) — the remaining non-Scan swarm ops and their status:

| op | ~ms (profiled) | capture today | verdict |
|----|---:|----|----|
| Reshape/Unsqueeze/Squeeze | large count | Supported (warmed) + `dtod_async`, **no eager sync** | already desync'd — skip |
| Concat | 45 | Supported (warmed) | already eligible — skip |
| Slice | 35 | Supported (warmed) | already eligible — skip |
| Split | 24 | Supported (warmed) | already eligible — skip |
| Cast / Constant / Add / Mul / Exp / … | 70 / 73 / … | `Supported`, but **unconditional eager `synchronize()`** | NOT the persistent-metadata pattern; shared across ~50 kernels → out of scope (see below) |
| **Tile** | **22** | **Unsupported** (declines) | **the only remaining `launch_metadata` op — this PR** |

`Tile` is the single remaining kernel in `movement.rs` still on the per-call
`launch_metadata` path (movement.rs:1115) AND it is the last op that *declines*
capture. Its 576 calls/step (~2 per LinearAttention Scan body × 288 invocations)
run in the **eager** Scan bodies, where its per-call syncs fire every time.

## Why Tile declines / syncs today (cite)

`TileKernel::execute` (movement.rs:1082) does **two** per-call host syncs:
1. `host_ints(&self.runtime, &inputs[1], "Tile")` (movement.rs:1091) reads the
   `repeats` tensor device→host via a blocking `dtoh` (movement.rs:190) — a
   per-call sync that is *illegal mid-capture*.
2. `launch_metadata` (movement.rs:205) does `alloc_raw` + `htod` + `synchronize()`
   + `free_raw` every call; the `synchronize()` (movement.rs:252) only guards the
   immediate `free_raw`.

`capture_support` therefore returns `Unsupported` with the exact reason
"Tile reads repeats on the host, allocates per-call metadata, and synchronizes
the stream" (movement.rs:1126).

## The fix (same PersistentMetadata pattern as Transpose/Expand + warmed repeats)

Key observation: the `tile_bytes` kernel metadata is `[output.shape, input.shape,
input_strides]` — it **never uses `repeats`**. `repeats` is read *only to validate*
that `output[i] == input[i] * repeats[i]`. Because the executor allocates the
output buffer from shape inference on the same `repeats`, a fixed
`(dtype, input_shape, output_shape)` signature mathematically fixes `repeats`
(`repeats[i] = output[i] / input[i]`). So:

- Add `PersistentMetadata` + `warmed_signature: Option<{dtype, input_shape,
  output_shape}>` to `TileKernel` (mirrors Transpose/Expand exactly).
- Read+validate `repeats` via `host_ints` **only on the unwarmed eager path**
  (first sight of a signature) — preserving the original validation precisely.
  Once warmed, skip the device→host read entirely (steady decode + capture).
- Replace `launch_metadata` with `launch_persistent_metadata` (cached buffer, no
  per-call alloc/htod/sync/free). Metadata bytes are **identical**.
- Guard: error on signature change mid-capture; `capture_support` → `Supported`
  once warmed. `metadata` built from `output.shape` (== the previously-validated
  `expected`), so byte-identical launch.

**Byte-exact:** same `tile_bytes` kernel, identical metadata bytes; only the
metadata-buffer lifecycle and the *timing/gating* of the repeats validation change.

## Why NOT Cast/Constant (honest)

Cast/Constant report `Supported` already; their cost is the **unconditional eager
`synchronize()`** (`if is_capturing { Ok } else { synchronize }`) shared by ~50
kernels. Removing it is a shared eager-execution-correctness change (WAR/eviction
model, prefetch.rs:133), NOT a bounded single-op persistent-metadata swap, and
needs capture-team review — out of scope for this PR. (Their real fix is Inc-1b:
capture the Scan body so it stops running eager.)

## Blast radius

- ONLY `crates/onnx-runtime-ep-cuda/src/kernels/movement.rs` (`TileKernel` +
  `TileFactory`) + two new EP tests + a one-line test-helper tweak
  (`build_movement_kernel` declares Tile's input_1 as Int64, like Expand/Reshape).
- Shared capture machinery (segmenter, `subgraph_graph_capturable`,
  `capture_quarantine_ops`, run.rs) untouched — Tile just starts reporting
  `Supported` once warmed.
- No other op batched in.

## Test plan (byte-exact; positive + NEGATIVE from the start)

- **New positive EP test** `tile_warmed_metadata_captures_and_matches_eager`:
  fixed-shape Tile (e.g. input `[2,2]`, repeats `[2,1]` → `[4,2]`) — unwarmed
  declines; first eager run warms → `Supported`; cached-metadata re-run
  byte-identical; capture+replay byte-identical to eager.
- **New negative EP test** `tile_rejects_signature_change_during_capture`: warm A,
  begin capture, feed a different shape → assert `Err` containing "changed during
  CUDA graph capture".
- **Full `construction_gpu` suite** stays green (incl. existing
  `tile_multi_axis_repeats`).
- **Byte-exact gates:** 27b oracle e2e `native_autoderive_io_cuda_e2e` (native ==
  CPU fp32 oracle) + `qwen3_0_6b_native_cuda_e2e` (golden) must stay byte-identical.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` exit 0;
  `cargo check --locked …`; `cargo fmt --all --check`.

## Honest yield estimate

Desyncs Tile's ~576 calls/step (two syncs each) in the eager Scan bodies.
Est. reclaim of the ~22 ms Tile bucket's host-sync fraction → ~5–15 ms/step,
projecting ~157 → ~145–152 ms/tok on the 27b. Does **NOT** beat ORT (57 ms/tok)
alone — this is another down-payment on the Lever-2 swarm; the 3× needs the full
swarm + Inc-1b Scan-body capture.

## Verdict

Bounded (one kernel, ~40 lines), byte-exact, in-pattern, real (22 ms bucket). BUILD.
