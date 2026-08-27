### 2026-08-20: Zero-copy metadata view for CUDA-EP Reshape — GO (greedy +3.66%, MTP +0.75%, bit-identical)

**By:** Deckard
**Branch/PR:** `squad/reshape-zerocopy-view` (off origin/main `461309ca3`)

**What:** The CUDA EP's `Reshape` kernel unconditionally did a device-to-device
`cudaMemcpy` on every call (416 calls/forward). Op-profiling (graphs-off) had it as
the single dominant op in the M=1 greedy forward (~6.5 ms) and super-linear in the
M=2 MTP verify (~35.75 ms, 5.34× the M=1 cost — the +29 of the +31 ms base→verify
delta). Fix: give `ReshapeKernel` a zero-copy metadata **view** — when the input is
contiguous and output element-count matches, `view_outputs` returns a
contiguous-strided `ViewOutput` aliasing input 0 (the same EP-agnostic view path the
CPU EP already uses for Reshape/Slice/Transpose), skipping the alloc + copy entirely.
`view_outputs` now also receives the executor-resolved `output_shapes` so a device EP
never reads a device-resident shape operand while composing a view during capture.

**Capture-safety (the crux):** installing a view frees the reshape output's stale
build-time buffer via `ep.deallocate`, which on the CUDA EP synchronizes the copy
stream during pooled unmap. That sync is **illegal during CUDA-graph capture**; before
the fix it aborted graph recording and quarantined *every* Reshape to an eager seam
(305 captured segments / 304 seams, greedy **62 → 54 tok/s** regression). Fix:
`install_view_outputs` parks the orphaned buffer in a new
`Executor::capture_deferred_frees` while capturing (`OpCaptureTrace::Captured`), and
`run_plan_segmented` flushes it (normal `deallocate`) right after
`end_device_graph_capture` closes the segment, where syncs are legal again. Outside
capture the free is issued inline as before.

**Aliasing predicate (conservative — alias only when provably safe):**
1. input 0 is contiguous (row-major, offset-0 strides) — else fall back to copy;
2. output element count == input element count (`out_numel == data.numel()`);
3. reuse the existing view machinery's guarantees: `view_bounds` gates the composed
   view against the source allocation, and the source value is `pinned` for the rest
   of the run (a pinned buffer is never reused/freed while a live view aliases it —
   conservative liveness, no use-after-free). External graph outputs are rejected from
   the view path. Reshape is pure data movement and is not written in place downstream,
   so the alias is read-only-shared and satisfies the MatMulNBits/Marlin contiguous-
   input contract.

**Numbers (median-of-5, Qwen3.8-27B int4, single H200, graphs-on, tokens=96,
warmups=2, `--runs 1` loop; base=origin/main `461309ca3` binary, branch=this commit;
both slots replay, fallbacks=0):**

| path   | baseline | branch | delta |
|--------|----------|--------|-------|
| greedy | 62.52 tok/s | **64.81 tok/s** | **+3.66%** |
| MTP    | 35.86 tok/s | **36.13 tok/s** | +0.75% |

- Greedy capture: **1 captured segment / 0 seams** on both baseline and branch
  (regression fixed). MTP capture: 19/18 + 146/145 segments/seams on **both**
  baseline and branch — those seams are inherent to the MTP graph, not introduced
  here.
- Per-phase (spec_phases): per_base 21.18→20.65 ms, per_verify 49.12→48.53 ms.
- **Bit/token-identity:** greedy `generated_token_ids` and MTP `generated_token_ids`
  are **byte-identical** branch vs origin/main; MTP `speculative_stats` identical
  (verify_steps=36, proposed=71, accepted=61, acceptance=85.9%).

**Why the graphs-off 35 ms did NOT become a 35 ms graphs-on win:** under CUDA-graph
replay the DtoD copy is baked into the captured graph and replays cheaply, so removing
it nets a real but modest gain (copy cost was largely hidden by replay). The honest
result is +3.66% greedy / +0.75% MTP, well short of the projected 70+/48–50 tok/s that
the graphs-off per-op number suggested. **Greedy clears the ≥2% honest gate; MTP is a
small positive (below 2%).** Helps every model (the copy elision is model-agnostic).

**Tests:** ep-cuda `--features cuda,cuda-13000` 473 passed / 1 failed (pre-existing
`a_module_restored_from_cached_ptx…` = `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, stale PTX
cache, unrelated); session 187+ passed; engine `--features native-backend` 582+ passed;
CPU EP view impls updated for the new `output_shapes` param.

**Executor seam touched:** `view_outputs` trait signature (`+output_shapes`),
`Executor::capture_deferred_frees` field, deferred-free in `install_view_outputs` +
post-capture flush in `run_plan_segmented`. `alias_input_as_output` (the in-place
path) has the identical `ep.deallocate` pattern but was left unchanged — it did not
regress any capture test; if a future in-place model aborts capture, apply the same
deferral there.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
