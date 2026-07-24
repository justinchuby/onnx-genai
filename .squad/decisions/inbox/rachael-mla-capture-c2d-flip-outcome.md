# DeepSeek MLA capture — (c2)+(d) outcome: capture ENGAGES + non-regressing, but 2.4x is OUT OF (a′)(c)(d) SCOPE

**Agent:** Rachael (worker)
**Branch:** `perf/deepseek-mla-capture` @ `cf40abb` (stacked on merged a′+c1 base `9ab1c6e`) — pushed, NOT merged.
**Verdict:** Capture engages, correct, deterministic, dense-safe, gate 207/0. **BUT the whole-step ~57–61 tok/s (2.4x) is NOT reachable within the Attention-path (a′)(c)(d) scope** — decisive evidence below. Recommend re-sequencing the remaining win as a separate general/MoE-op capture-safety effort.

## What landed (staged commits)
- **(c2) `af907e3`** — freeze default-domain `('','Attention')` KV/mask bindings to fixed capacity at single-token decode:
  - `executor.rs::kernel_input_uses_physical_capacity` (~1122): default-domain Attention KV inputs 4,5 treated as physical-capacity (mask-driven, non-causal), mirroring GQA.
  - `executor.rs` present-shape widening (~3269): made capacity-aware (`kv_capacity_bound`) so a physically-bound past does not inflate present beyond the capacity buffer (fixes a prefill `accepts_output` overflow: past_phys 4096 + cur 5 = 4101 > cap 4096).
  - `native_decode.rs::extend_mask(start,end,expose_len)`: freezes mask binding logical to `[1,max_len]` at decode (prefill keeps growing + runs eager); construction mask logical `[1,max_len]` so `graph_enabled` isn't killed.
- **(d) `cf40abb`** — make the default-domain Attention decode capture-safe + flip `capture_support`:
  - Persistent `StdAttnWorkspace` (module-scoped scratch, no per-op alloc on capture path; `reserve()` refuses to grow during capture, Drop frees).
  - Entry/exit `synchronize()` guarded on `!is_capturing()`; control `htod` uploads guarded out of capture.
  - A warmed fixed-capacity device-valid-length single-token decode step records a capture signature; `capture_support()` → `Supported` only when the signature is present (eager/dense/growing decline).
  - **Regression test** `capture_support_gated_on_warmed_device_valid_length_signature` (general): fresh kernel declines; Supported only after a device-valid-length decode signature is warmed — fails if `capture_support` reverts to unconditional Supported or the device-length requirement is dropped.

Design note: I did NOT IR-prune the `Unsqueeze_18` island. I froze the *binding* logical shape at decode instead (de-risked, no export/IR surgery). This makes the KV/mask BINDINGS capture-eligible, but does NOT make the island's intermediate ops capturable — see seam evidence.

## VERIFICATION (all on idle GPU 6, `source .cudaenv.sh`)
### Capture ENGAGES — both real exports
```
blk128: cuda_graph: enabled=true captures=1 replays=9  fallbacks=0
blk32 : cuda_graph: enabled=true captures=4 replays=244 fallbacks=0
```
### Determinism + coherence (3× blk32, 2× blk128) — IDENTICAL every run
```
generated_token_ids: [8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]
```
pos0 = 8913 ' Paris' → " Paris.\nThe currency of France is the Euro." ✅

### Dense non-regression (Qwen2.5-0.5b GQA) — UNTOUCHED
```
cuda_graph: enabled=true captures=4 replays=84 fallbacks=0
throughput: 268.80 tok/s ; coherent (" Paris. It is the largest city ...")
```
GQA path bit-for-bit unchanged (my rule is default-domain-Attention-only).

### GATE
- `CUDA_VISIBLE_DEVICES=6 cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **207 passed / 0 failed** (206 baseline + 1 new capture test).
- clippy `--lib -D warnings` clean: ep-cuda, session, engine.
- clean worktree; committed + pushed.

## HEADLINE (the part that falls short) — eager vs captured, blk32, tokens=64/runs=3/warmups=1
```
EAGER  (ONNX_GENAI_CUDA_GRAPH=0): 25.87 tok/s, 38.66 ms/step  (captures=0)
CAPTURED                        : 27.71 tok/s, 36.09 ms/step  (captures=4 replays=244 fallbacks=0)
```
**+7% only** — NOT the ~57–61 tok/s (2.4x) target.

## WHY 2.4x is out of scope — DECISIVE seam evidence (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`, blk32, 1 decode step)
The decode step fragments into **~727 eager seams**, of which only ~36 are in (c2)/(d) scope. Per-op-type seam counts:
```
268 Reshape   ("copy path, not a capture-validated zero-copy view" / data-dependent shape)
 81 Split     ("reads runtime split sizes on host + trailing stream synchronize")
 54 Concat    ("trailing host stream synchronize")
 53 Cast      (data-dependent output shape unresolved)
 52 Mul
 27 Expand    ("allocates/uploads/frees per-call broadcast metadata + synchronize")
 27 Attention (data-dependent input shape unresolved — consume mask island output)
 26 QMoE / TopK / Softmax / ScatterElements / MatMul / GatherElements  (data-dependent)
  ~9 mask-island ops (CumSum/Slice/Unsqueeze/GreaterOrEqual/And/Where/Cast)
```
**~690 of the ~727 seams are inherently non-capture-safe general + MoE ops (Reshape/Split/Concat/Cast/Mul/Expand + QMoE/TopK/Softmax/Scatter/Gather/MatMul) that live entirely OUTSIDE the Attention path.** Each seam is an eager fallback (with per-op alloc/sync), so per-step orchestration overhead (Pris: ~60% of the 40.5 ms/token) persists. Even a *perfect* mask-island prune would only unblock the 27 Attention nodes (they seam because they consume the data-dependent mask island), leaving ~690 seams — the 2.4x cannot materialize until those general/MoE ops are made capture-safe.

## RECOMMENDATION / OWNER ROUTING
- **LAND** this branch as a verified, non-regressing **capture foundation**: the default-domain Attention decode path is now genuinely capture-eligible (device valid-length ABI + fixed-cap bindings + capture-safe scratch/sync), engages with fallbacks=0, is deterministic + coherent, and does not regress dense/GQA. It is the correct prerequisite for the whole-step win. (Fresh reviewer verifies; do NOT merge yet.)
- **RE-SEQUENCE the 2.4x** as a NEW, larger effort: capture-safety for the general-op kernels (Reshape zero-copy view, Split/Concat without host-side sync, Expand without per-call alloc, Cast) + the MoE path (QMoE/TopK/Softmax/ScatterElements/GatherElements/MatMul must resolve output shapes before capture — likely fixed-capacity/token-count bounds). This is cross-cutting kernel work (engine + QMoE owner + matmul), NOT Attention-path (a′)(c)(d).
- My honest read: I did NOT force a fake whole-step number, and I did NOT push a regressing half-state — this is a strictly-improving, verified, separable piece + the evidence you need to scope the real remaining work.

Pinned file:line for the follow-up:
- Seam sources: general-op kernels in `crates/onnx-runtime-ep-cuda/src/kernels/` (reshape/split/concat/cast/expand) + `qmoe.rs` + `matmul*.rs`; each needs a capture-safe (no host-sync, no per-op alloc, resolved-shape) path.
- Attention path is done: `standard_attention.rs` capture_support (~1594), workspace (~559+), device valid-length (a′/c1) already merged.
