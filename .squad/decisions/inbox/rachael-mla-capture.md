# DeepSeek MLA capture-enablement — implementation attempt, empirical findings, and honest status

**Author:** Rachael (worker) · 2026-07-24
**Branch:** `perf/deepseek-mla-capture-enable` @ `de5188c` (stacked on correctness fix `621936f`, off main `24531c4`) — **pushed, NOT merged**
**Bottom line:** Full CUDA-graph capture for the default-domain `Attention` decode step is a **large, correctness-critical, multi-crate rework** and is **NOT safely landable in one stacked pass**. I landed the one **safely-separable, verified** brick (stream-ordered capture-safe copy-back) and I am **reporting the rest with empirical evidence** rather than forcing an unsafe change — per your fallback instruction. **No capture numbers are fabricated: capture is still off.**

## What I landed (safe, verified, on the branch)
Convert the KV-growth **copy-back** (introduced by the correctness fix) from
synchronous `dtod` (full `synchronize()` + `cuMemcpyDtoD`, capture-illegal) to
`dtod_async` (`cuMemcpyDtoDAsync`) on the EP compute stream. `build_kv`
(producer) and the next step's `build_kv` (consumer) run on that same stream, so
the copy is stream-ordered without a full sync; the execute-exit `synchronize()`
still drains it (eager blocking semantics preserved). This is scope item 3's
"copy-back is async/stream-ordered" and a prerequisite for capture.
- Determinism preserved (blk32 & blk128, 3× identical: `[8913,13,185,549,19305,280,7239,317,254,28071,13,185]`, " Paris.\n…").
- tok/s ~unchanged (blk32 23.9, blk128 25.3) — the per-op syncs dominate, so this alone is not the perf lever.
- Gate: `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **204/0**; clippy `--lib -D warnings` clean.
- Qwen (GQA) unaffected: captures intact, coherent.

## Why full capture is not a one-pass stacked change (EMPIRICAL evidence)
I probed scope item 1 directly: taught `kernel_input_uses_physical_capacity`
(executor.rs:1122) to recognize the default-domain `('' , Attention)` KV inputs
(4,5) as physical-capacity (mirroring GQA), then ran the real model. Result —
hard error (probe reverted):
```
external output 'present.0.key' has Float16 [1, 16, 4096, 192] (25165824 bytes),
kernel requires Float16 [1, 16, 4101, 192] (25196544 bytes)
```
This is decisive: exposing the KV at capacity makes `StandardAttentionKernel`
derive `past_seq` from the **capacity extent (4096)** instead of the **valid
length (5)** — it then wants `total_seq=4101` and mismatches the 4096 buffer.
**Capacity recognition alone breaks correctness; the kernel MUST receive the
valid length on-device.** GQA avoids this because it has a `seqlens_k` input
(produced on-device by an attn-mask subgraph); the default-domain Attention
decode path has **no such device valid-length signal** — it derives length
purely from the (growing) KV tensor extent.

## The real, coupled blockers (all must land together for capture)
Note: for M=1 decode the **launch grid geometry is already fixed**
(`total_rows = batch*q_heads*q_seq = q_heads`, standard_attention.rs:1112) — so
geometry is NOT the blocker. The blockers are per-step *varying values/shapes*:
1. **No device valid-length ABI.** The kernel needs `past_seq`/`total_seq`/causal
   offset/pad frontier from **device memory** (updated out-of-band between
   replays), not from the host-known tensor extent nor host-computed
   `offsets`/`pad_limits` uploaded via `htod` each step (standard_attention.rs:
   971–1052 — the `htod` is the root capture-illegal op). Requires a new
   device valid-length input plumbed native_decode → executor binding → kernel
   (analogous to GQA `seqlens_k`), OR deriving it from the mask's last dim.
2. **`total_seq` passed as a by-value kernel arg** (line ~1147) is baked at
   capture time → stale on replay. Must become a device read.
3. **Fixed-capacity KV + fixed-slot append.** With capacity layout the kernel
   must append the new token at slot `[valid_len]` (device-read) at fixed head
   stride `max_len`, not restride the dense cache each step. This also removes
   the copy-back entirely — but requires (1).
4. **Growing internal mask tensor.** The Attention `attn_mask` input
   (`v_model.Unsqueeze_18`) is a graph-internal tensor of length `total_seq`
   produced each step from `attention_mask`; its size **grows per step**. Under
   capture, intermediate buffers are fixed-size, so the mask subgraph must be
   made **capacity-length** (padded), which is an **executor
   `kernel_input_uses_padded_capacity` extension across the mask-reformat ops
   and/or an export-side change** — the hardest and least-localized blocker,
   likely needing Mobius/export or a broader executor effort.
5. Per-step scratch alloc (scores) + entry/exit `synchronize()`
   (standard_attention.rs:717, 1173) must be pre-allocated/removed for the
   captured region.

## Recommendation (sequencing)
Full capture ≈ reimplementing the standard-Attention decode as a **fixed-capacity,
device-valid-length, masked** kernel (what GQA/flash already do) PLUS making the
DeepSeek attn-mask subgraph capacity-length. This is a **dedicated multi-crate
feature** (kernel + executor binding + native_decode device scalar + mask-subgraph
capacity), not a stacked one-pass change, and it is **correctness-sensitive**
(getting masking of the uninitialized capacity tail wrong reintroduces exactly
the class of garbage the correctness fix eliminated). Recommend scoping it as its
own tracked effort with staged, separately-verified commits:
(a) device valid-length ABI + kernel reads it (keep eager, verify correctness),
(b) fixed-capacity KV + fixed-slot append (verify correctness, removes restride —
    a real eager perf win independent of capture),
(c) capacity-length mask subgraph (executor/export),
(d) flip `capture_support()` → Supported + remove per-op sync/htod/alloc, verify capture.
Correctness (`621936f`) is already delivered and independent. This branch adds
the safe async copy-back brick; it does not claim capture.

## Verify (this branch, GPU 3, clean rebuild)
- blk32: 3× identical tokens, coherent, 23.94 tok/s, `captures=0`.
- blk128: 2× identical tokens, coherent, 25.31 tok/s, `captures=0`.
- Qwen2.5-0.5b (GQA): coherent, `captures=3 replays=39 fallbacks=0` — no regression.
- Gate: lib 204/0, clippy `--lib -D warnings` clean.

## Owner
The staged capture feature spans kernel (standard_attention.rs) + executor
(binding/capacity recognition) + native_decode (device valid-length) + mask
subgraph (executor/export/Mobius). Needs coordinator sequencing given the
cross-crate/export surface. Rachael can own (a)+(b) (kernel/engine); (c) likely
needs export/executor collaboration.
