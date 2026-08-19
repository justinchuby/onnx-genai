# Decision: LinearAttention decode kernel — warp-cooperative rewrite

**Author:** Gaff (CUDA-kernel specialist)
**Date:** 2026-08-19
**Branch:** `squad/linattn-warp-coop`
**Scope:** `crates/onnx-runtime-ep-cuda/src/kernels/linear_attention.rs`
(+ parity coverage in `tests/linear_attention_gpu.rs`)

## Context

Deckard's attribution: native's core linear-attn decode kernel
`linear_attention_f16` was 20.9µs/call vs ORT's fused
`LinearAttentionDecodeColKernel` 9.0µs/call (~2.4×, ~+185µs/step over ~18
recurrent layers). Root cause was structural, not algorithmic: each thread held
its full state column in `float sc[MAX_D_K=256]`, which spills to LOCAL memory,
and walked d_k=128 SERIALLY four times per step (decay / retrieval / delta /
readout). ncu: dominated by Long-Scoreboard on local-memory traffic, launch only
`batch·H_kv·d_v` threads (grid-starved, most SMs idle) — latency/issue-bound, not
compute/BW-bound.

## What landed

Warp-cooperative kernel `linear_attention_*_coop` (f32/f16/bf16), mirroring
ORT's structure:

- **One WARP owns each state column** (b, h_kv, j). Lane `l` holds the d_k rows
  `i = l, l+32, ...` in registers (`sc[LA_MAX_SLOTS]`, ≤8 f32) — no local-memory
  spill.
- **The two d_k dot products** (retrieval `r = Sᵀk`, readout `o = qᵀS`) become
  `__shfl_xor` warp reductions instead of a 128-iteration serial loop.
- **Launch grows 32×** in blocks (one warp, not one thread, per column) → fills
  the SMs.
- Decay / delta / state read-write stay per-element (bit-identical to the serial
  kernel); only the two reductions change summation order.

## Results

**ncu (fp16, qwen3.5-0.8b, H200, isolated, `--graph-profiling node`):**

| metric | serial (before) | coop (after) |
|---|---|---|
| local **load** sectors | 163,840 | **0** |
| local **store** sectors | 98,304 | **0** |
| kernel duration (ncu-serialized) | 21,664 ns | ~12,510 ns (**−42%**) |
| SM throughput | 1.31% | 13.3% |
| achieved occupancy (warps active) | 12.0% | 21.2% |
| registers/thread | 56 | 64 |
| grid | 8 blocks (2048 threads) | 256 blocks (2048 warps) |

The local-memory spill — the dominant bottleneck — is fully eliminated. The
per-issue Long-Scoreboard ratio rises 4.6 → 10.5, but only because the cheap
high-count local/loop instructions that diluted it are gone; absolute kernel
time drops 42% and E2E tok/s rises.

**End-to-end decode (fp16io, interleaved paired A/B, env-toggle same binary):**
coop beats serial by **+7.4 to +10.6 tok/s every round** (~240 → ~249 tok/s,
~+3.2%). Paired delta cancels shared-box noise.

## Correctness — ULP-divergent but greedy-stable (validated)

The golden locks compare **greedy TOKEN-ID sequences** (`Vec<u32>` argmax
stream), NOT bit-exact logit/hidden bytes (`assert_native_matches_golden` in
`tests/common/decode_lock.rs`). The warp reductions sum the retrieval/readout dot
products in tree order rather than the serial kernel's left-to-right order, so
fp32 results shift at the ULP level (accumulation stays in f32). Since the retrieval
`r` feeds the state update, this ULP shift can propagate across decode steps —
but argmax is unaffected:

- **qwen3.5-0.8b text decode golden lock: PASS** (byte-identical token IDs).
- **qwen3.5-2b text decode golden lock: PASS** (second model, different head
  geometry — confirms generality, not 0.8b-special-cased).
- **GPU CPU-oracle parity suite: PASS** for BOTH the coop and serial kernels
  across GQA / inverse GQA / key-sharing / per-key-dim decay / shared beta / all
  four `update_rule`s, plus new d_k=128 / 96 / 130 (multi-slot and
  non-multiple-of-warp tail-guard) configs.

So: the change is ULP-divergent-but-greedy-stable. The greedy locks (the shipping
gate) pass byte-identical, so it is safe to default on. If a future model ever
needs bit-exact reproduction of the serial reduction order, the opt-out flag
restores it.

## Properties

- **General:** keys on head_count / d_k / d_v / dtype (f16/bf16/f32); handles
  `d_k` not a multiple of the warp via `i < d_k` guards; GQA / inverse GQA / dual
  head sizes (no regression to the dual-head-size support). `MAX_D_K = 256` cap
  unchanged.
- **Capture-safe:** static shapes, warp-uniform grid-stride bound (whole warp
  enters/exits together → no partial-warp `__shfl_xor_sync` divergence), no host
  sync.
- **Opt-out:** `ONNX_GENAI_CUDA_DISABLE_LINATTN_WARP_COOP=1` (or `true`/`on`)
  falls back to the original serial kernel. Read once at kernel construction
  (before capture), so it is capture-stable.

## Verdict

A clean structural win that closes most of the native↔ORT linear-attn gap:
−42% kernel time, spill eliminated, +~8 tok/s E2E, byte-identical greedy output
on two models. Default-on, de-riskable via the opt-out flag.
