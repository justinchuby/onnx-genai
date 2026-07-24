# Rachael — DeepSeek MLA capture stack: (a′) landed + verified; (c)+(d) blocker characterized

**Branch:** `perf/deepseek-mla-capture` @ `e14d7df` (stacked cleanly on merged main `53afab0`; pushed, NOT merged — fresh reviewer verifies).
**State branch for this note:** `bench/ort-vs-native-cuda`.
**GPU:** verified on idle GPU 6 (0 MiB before benching; CPU-perf team shares the box).

## Summary

The full (a′)→(c)→(d) capture stack decomposed into a **verified, safe, reviewable
(a′)** now on the branch, and a **coupled (c)+(d) remainder** that I did NOT force —
per Justin's no-unverified-regression mandate and the coordinator's explicit
"STOP and report with evidence if a blocker is intractable mid-build" escape hatch.
I landed (a′), proved it does not regress real DeepSeek or dense capture, and
characterized the exact (c)+(d) blocker with source + empirical evidence, plus a
de-risked plan.

## (a′) — Device valid-length for default-domain Attention decode  [LANDED, VERIFIED]

**Change (1 commit, `e14d7df`, `standard_attention.rs` only):**
- New `derive_len` NVRTC kernel scans the additive attention-mask bias for its
  valid-length frontier (0 bias = valid, large-negative = padding) → writes the
  real length to a device `i32`.
- `build_kv` and `attention_row` take an **optional** `dev_len` device pointer.
  When non-null, the kernel reads the valid attended length (and the fixed-slot
  append position) from **device memory** instead of host shape metadata (whose
  extent is frozen at CUDA-graph capture). Null pointer = host-derived path,
  **bit-for-bit unchanged** → eager/prefill/dense and GroupQueryAttention are
  untouched.
- Eligible only for the fixed-capacity, mask-masked (`is_causal=false`),
  single-query (`q_seq==1`) decode step — i.e. exactly the DeepSeek MLA decode.

**Why this is the pivot:** it removes the decode step's dependence on the growing
logical KV/mask extent for its length — the prerequisite that lets the step
become capture-eligible. Verified in isolation (eager), as sequenced.

**Verification (all green):**
- blk32 greedy ×3 back-to-back → **identical** `[8913,13,185,549,19305,280,7239,317,254,28071,13,185]` (" Paris.\nThe currency of France is the Euro.").
- blk128 greedy ×2 → identical, same token ids.
- Qwen2.5-0.5b (GQA dense) → `captures=2 replays=26 fallbacks=0`, coherent " Paris. It is the largest city…", 202.88 tok/s (unchanged — GQA path untouched).
- Eager tok/s: blk32 23.28 → **24.06** (no regression; the extra tiny `derive_len` launch/op is noise-level).
- Gate: `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` = **205/0**.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` = clean.

Capture stays OFF (bindings still grow) — expected; that is (c)+(d).

## (c)+(d) — Capture enablement  [NOT LANDED — blocker characterized with evidence]

Getting capture to ENGAGE requires the KV+mask bindings to be fixed-capacity so
the capture gate opens (`native_decode.rs:1919` declines on
`has_dynamic_logical_input_shape()`), plus removing the kernel's per-step sync /
per-op alloc / htod and flipping `capture_support()→Supported`.

**The genuine blocker (source + empirical):** the mask-frontier length heuristic
in (a′) is **decode-only**. For DeepSeek Attention (verified on both exports:
domain `''`, **no `is_causal` attr** → causality is entirely via the additive
`Unsqueeze_18` mask, inputs `[…Reshape_38/39/30, Unsqueeze_18, past_key, past_value]`):
- **Decode** (`q_seq==1`): mask row is `[…,max_len]` with a single padding
  frontier → `derive_len` gives the real length. ✔ (this is what (a′) uses).
- **Prefill** (`q_seq>1`): mask is causal `[1,1,prompt_len,max_len]` — row 0 allows
  only key 0, so a first-row frontier scan yields 1, not `prompt_len`. ✘ The
  heuristic cannot recover the key extent from a causal mask.

So a robust, general capture path needs the kernel to be **device-length-driven
for prefill AND decode** — i.e. an explicit **GQA-style device seqlens/valid-length
scalar** (mirroring `seqlens_k`), not a mask heuristic. That scalar has no
existing graph input on the DeepSeek export (GQA gets `seqlens_k` as a real ONNX
input; default-domain Attention does not), so it needs either IR graph surgery to
add the input, or an executor side-channel binding delivered to the kernel.

**Additional coupled work confirmed for (c)+(d):**
1. **Fixed-cap binding switch** — cleanest path: keep **prefill eager + growing**
   (it already invalidates the graph, `native_decode.rs:1374`), then **switch the
   KV+mask bindings to fixed-capacity (logical=physical=max_len) at the
   prefill→decode boundary**, and **defer the capture gate check** from `new()`
   (1919) to capture time. This sidesteps the prefill-with-fixed-cap breakage
   entirely.
2. **Host-prologue rework** in `standard_attention.rs::execute()` — at fixed cap,
   `past_key.seq` (logical) becomes the *capacity* (max_len), so
   `total_seq = key_past_seq + k_cur.seq` and `capacity_key = key_cap > total_seq`
   break (they compute `max_len+1`). The prologue must treat `past_key.seq` as
   capacity and take the real length from the device scalar.
3. **(d) sync/scratch removal** — remove entry/exit `synchronize()`
   (standard_attention.rs:738,1250), replace per-op `alloc_raw`/`free_raw`
   (scores + control) with fixed module-global scratch (mirror `gqa_decode.rs`),
   drop the `offsets`/`pad_limits` htod (no-ops on the DeepSeek path:
   `is_causal=false`, `pad_limit=-1`), then flip `capture_support()→Supported`.
4. **Capture regression test** locking device-valid-length + capture-eligibility.

**Perf prize unchanged:** Pris's profiling says whole-step capture takes DeepSeek
~25 → ~57–61 tok/s (2.4×). (a′) is the foundation; (c)+(d) delivers the win.

## Recommendation / sequencing

- Merge (a′) `e14d7df` after fresh review (safe, verified, no perf change — it's
  the enabling refactor).
- Sequence (c)+(d) as a focused follow-up on top of `e14d7df` using the
  **prefill-eager + post-prefill fixed-cap switch + deferred gate + explicit
  device seqlens ABI (prefill+decode) + kernel sync/scratch removal** plan above.
  The device-seqlens ABI (delivery to the kernel) is the real design decision:
  IR graph-input surgery vs executor side-channel. Owner: engine (native_decode +
  executor + standard_attention). No Mobius/export dependency (the mask island is
  engine-synthesizable, confirmed earlier).

## Gate/build reference
`source /home/justinchu/onnx-genai/.cudaenv.sh`; build
`cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native`
(touch the changed file first to avoid stale-binary reuse); models
`ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32` and `…-real-int4` (blk128),
`qwen2.5-0.5b-int4-onnx-native`.
