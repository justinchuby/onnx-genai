# DeepSeek MLA eager-fallback: root cause, perf improvement, and re-scope

**Author:** Rachael (worker) · **Date:** 2026-07-23 · **State branch:** `bench/ort-vs-native-cuda`
**Fix/perf branch:** `perf/deepseek-mla-capture` @ `24531c4` (off origin/main `1fe314f`), pushed, NOT merged.

## TL;DR

The DeepSeek-V2-Lite native decode runs **eager** (`cuda_graph: enabled=false`,
captures=0) **not because of the `dtod` sync** (Holden's hypothesis), but because of a
**structural property of the MLA `Attention` path**: whole-step CUDA-graph capture is
declined **up front at binding time**, before any op executes. Making DeepSeek capture
requires reworking how the `Attention` op consumes its KV cache — a substantial,
correctness-critical engine change that should be **re-scoped to the attention/engine
owner**. `dtod` is exonerated as the capture blocker.

I did ship one correct, low-risk, in-scope improvement on the branch: a **capture-safe
stream-ordered async copy** for Reshape/Squeeze (removes ~200+ per-token stream drains,
modest eager speedup, takes movement off the capture-illegal sync path). It does **not**
by itself enable DeepSeek capture.

I also **discovered a separate pre-existing bug**: DeepSeek greedy **decode is
nondeterministic** run-to-run (should be impossible for greedy). It exists on clean
`main` and is NOT caused by my change. Needs its own investigation.

---

## (a) Confirmed cause — with evidence (NOT the dtod)

CUDA-graph capture is auto-enabled for the native decode step only when the topology is
structurally graph-safe. For DeepSeek it is declined at `DecodeCudaState::new` by this
gate (`native_decode.rs:1919`):

```
graph_enabled = graph_enabled && !bindings.iter().any(has_dynamic_logical_input_shape())
```

Instrumented run (blk32) — the tripping bindings:

```
graph_enabled(before dyn check)=true declined_aux=[]
dyn-binding attention_mask       logical=[1,0]         physical=[1,4096]
dyn-binding past_key_values.N.key   logical=[1,16,0,192]  physical=[1,16,4096,192]
dyn-binding past_key_values.N.value logical=[1,16,0,128]  physical=[1,16,4096,128]   (all 27 layers)
```

The same run on **Qwen2.5-0.5b** prints **zero** dynamic bindings → `enabled=true captures=2`.

Why the difference: `has_dynamic_logical_input_shape` is true when a binding
`expose_logical_input_shape`s and its logical≠physical shape. `expose_logical_input_shape`
is set by `binding_consumers_use_physical_capacity` → `kernel_input_uses_physical_capacity`
(`executor.rs:1122`):

```rust
node.domain == "com.microsoft" && node.op_type == "GroupQueryAttention" && matches!(input_index, 3|4)
```

Only **GroupQueryAttention** KV inputs are recognized as fixed-capacity consumers. DeepSeek's
attention is the **default-domain `Attention`** op (confirmed: `('', 'Attention') x27` in the
model), whose kernel "derives past length from the cache tensor extent itself" — so its KV
and mask bindings expose their **logical, growing** extent → dynamic → capture declined.

Corroborating evidence:
- `ONNX_GENAI_CUDA_GRAPH=1` (force ON) → **still `enabled=false`**: the dynamic-binding gate
  overrides the env, proving it is structural, not a toggle. Same for **block-128**.
- Second, independent structural blocker in the kernel itself
  (`standard_attention.rs:1142`): `StandardAttentionKernel::capture_support()` returns
  `Unsupported("setup synchronously uploads per-batch control arrays and reads
  nonpad_kv_seqlen D2H")`. Even if the KV bindings were fixed-capacity, the kernel's per-step
  D2H read (`dense_i64` → `dtoh`, line ~616) and synchronous `htod` control-array uploads
  (line ~1016) are capture-illegal.

**Conclusion:** the eager fallback is caused by the MLA `Attention` path, on two counts
(dynamic KV/mask bindings + kernel capture_support), **before** the `dtod`/`copy_reshape`
sync is ever reached. Holden's dtod hypothesis is disproven by `enabled=false` (capture is
never even attempted, so the in-capture illegality of `cuMemcpyDtoD` is moot for the fallback).

## (b) What I shipped (branch `perf/deepseek-mla-capture` @ `24531c4`)

A capture-safe, stream-ordered async copy for Reshape/Squeeze:
- `copy_reshape` (`movement.rs`) now uses `dtod_async` on the EP compute stream instead of
  the synchronous default-stream `dtod`. Same-stream ordering vs the producer/consumers is
  the correctness argument (no host `synchronize()` needed). This removes the per-call stream
  drain (~200+/token on the MLA path) and takes movement off the capture-illegal `cuMemcpyDtoD`
  path — a prerequisite for any future MLA capture.
- The synchronous `dtod` (with its stream drain, the merged race fix) is **unchanged** for the
  other callers (provider copy, CSA checkpoint), so the merged fix and its regression test are
  preserved.
- New unit test `dtod_async_is_ordered_after_same_stream_producer` guards the same-stream
  ordering invariant (companion to `dtod_waits_for_pending_stream_writes`).

**This does NOT enable DeepSeek capture** (still `enabled=false` — the two structural blockers
above remain). It is a correctness-neutral eager perf + capture-readiness improvement.

### Before / after (native CUDA, GPU6, prompt "The capital of France is", 32 tok, warmup 1)

| model     | before (main, sync dtod) | after (async copy) |
|-----------|--------------------------|--------------------|
| blk32     | ~22.85 tok/s             | ~24–25 tok/s       |
| blk128    | ~22.85 tok/s             | ~25.5 tok/s        |
| capture   | enabled=false, cap=0     | enabled=false, cap=0 (unchanged) |

Output stays coherent (pos4 argmax `8913 ' Paris'`; text mentions Paris / Eiffel Tower).

Qwen2.5-0.5b (dense, regression check): still `enabled=true captures=2 replays=26 fallbacks=0`,
**deterministic**, ~222 tok/s (no regression).

## (c) NEW pre-existing bug discovered — DeepSeek greedy decode is nondeterministic

On clean `main` (sync dtod, my change reverted), DeepSeek greedy decode gives **different
token ids run-to-run** starting around generated token index ~4:

```
[8913,13,185,549, 427, 96575,25943,317,254,1094, 9679,18972]
[8913,13,185,549, 19305,280,7239,317,254,28071,13,185]
[8913,13,185,549, 6077, 280,7239,317,8913,13,185,549]
```

- Greedy decode MUST be deterministic → this is a real correctness bug.
- **`CUDA_LAUNCH_BLOCKING=1` does NOT fix it** → it is **not** a launch-ordering async race
  (unlike the prefill bug I fixed in `1fe314f`). Likely **stale/uninitialized KV-cache memory**
  or a data-dependent decode path at longer sequence.
- **Qwen decode is deterministic** (same `copy_reshape`), so it is **MLA-specific**.
- NOT caused by my async change (baseline is equally nondeterministic).

This is out of scope for the capture/perf task but should be triaged — it may share a root
with the MLA KV-extent handling below (attention reading beyond the valid past length into
uninitialized padding).

## (d) Recommended re-scope (owner: attention/engine)

To actually make DeepSeek MLA capturable (the perf lever the task wanted), the `Attention`
op path needs:
1. **Fixed-capacity KV consumption** — teach the executor that the default-domain `Attention`
   past_key/past_value inputs use physical capacity (extend `kernel_input_uses_physical_capacity`),
   AND make `StandardAttentionKernel` read the valid past length from a device scalar
   (total_sequence_length / a seqlens-style signal) rather than the tensor extent. Today the two
   are coupled: the kernel uses the extent, so the binding must expose the logical shape. Note
   `nonpad_kv_seqlen` exists but is currently mutually exclusive with an in-op past cache.
2. **Capture-safe kernel setup** — remove the per-step D2H read of `nonpad_kv_seqlen` and the
   synchronous `htod` control-array uploads (precompute on-device / persistent bindings) so
   `capture_support()` can become Supported.
3. Movement is already capture-safe after this branch (item b).
4. Investigate the (c) decode nondeterminism, likely coupled to (1) (uninitialized KV padding).

This is a substantial, correctness-critical change to MLA attention — recommend routing to the
attention/engine owner, not folding into the movement/runtime perf lever.

## Gate (source .cudaenv.sh; CUDA_VISIBLE_DEVICES=6; clean worktree off origin/main 1fe314f)

- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **203 passed / 0 failed**
  (202 baseline + new `dtod_async_is_ordered_after_same_stream_producer`; the merged
  `dtod_waits_for_pending_stream_writes` still passes).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` → clean.
- Qwen2.5-0.5b native decode → coherent, deterministic, captures, no perf regression.

**Branch:** `perf/deepseek-mla-capture` @ `24531c4`. Do NOT merge — fresh reviewer to verify.
The captures>=1 verification target is **NOT met** and requires the (d) re-scope.
