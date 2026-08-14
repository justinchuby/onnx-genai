# Native batch-N decode step — stage 2b-impl scoping (#750)

**Status: COMPLETE — all four increments landed (#913, #920, #924, #930).**
This document is kept as the record of the split and the batch-1 baseline every
increment held byte-identical. Where an increment corrected the plan below, the
correction is noted inline; **the corrections are the more useful part of this
document now**, because three of the four returned one.

Outcome, reproduced independently by the coordinator rather than taken from the
PRs: **CUDA graph capture survives batch-N while weights stream** — at N=2 and
N=4, `enabled=true captures=1 invalidations=0 fallbacks=0` with
`htod_bytes_per_token` 1,234,271,232 and 547,528,704, `byte_hit_rate` 68.64% →
72.17% (solo, managed streaming forced, `qwen14b-zp`, context 64). That was the
open risk #891 flagged, and it is now closed positively. The `1/N` weight-stream
amortization holds on the real decode path, with the #866 offset measurable
within the sweep: `byte_hit_rate` flat ~62% at N=1,2,4, then −5.12 points at N=8
as evictions go 1273 → 1678, which is exactly why N=8 sits +13.5% above the
ideal line.

**One near-miss worth carrying forward.** #930's first version reported capture
and weight streaming as *mutually exclusive on this build*, from a real decline
message naming a real predicate. Its sweep harness called
`load_with_resolved_io`, which passes `cuda_offload_policy: None`, so
`weight_offload_stable_va` fell to its deliberately conservative
`unwrap_or(false)` and capture declined **by construction** — the harness
asserted pointer-instability and the runtime was never asked. Rewiring the sweep
onto the governed `Engine::from_dir` path produced the result above. See
`.github/skills/measurement-discipline/SKILL.md` failure mode 1; #930 now prints
the predicate's *inputs* and their source, so `!stable_va` can no longer be
mistaken for "this hardware cannot do it".

**Original verdict, which held:** the smallest *correct* (token-producing)
batch-N persistent-binding decode step does not fit in one reviewable PR. This
document proposes the split, with file/line evidence for every coupled site, and
records the measured batch-1 baseline that every increment must hold
byte-identical.

This is deliberately the same shape of deliverable the owner accepted for stage 1
(#844, a staging plan) and stage 2b (#891, which landed a guard and *proposed*
this rewrite as a separate change): "a landed increment with a guard beats a
large one that stalls." Here the landed artifact is the baseline + the
decomposition that makes each following increment safe to attempt.

## 0. What each increment corrected in this plan

| Increment | Correction returned |
|---|---|
| **#913** (2b-impl-1) | None — the inventory in §3/§4 held for the shape/IO-binding layer. Added the control this document did not ask for: the geometry is unit-tested at `batch = 3`, where a transposed axis actually shows up, rather than only at 1 where it cannot. |
| **#920** (2b-impl-2) | Two. §3/§4 place `live_prefix_ranges` in another crate; it is in this one (`native_decode/kv_commit.rs:122`). And "KV growth stride math already batch-general" holds **only for head-major**: a *fixed full-context stride* commit is batch-general in both layouts because the batch axis is outermost and sequences sit a constant `capacity × bytes_per_token` apart, but a *growing bucket* is not, because seq-major's per-sequence stride **is** the mutable capacity, so growth would relocate every sequence `b > 0`. That case is now a named `bail!` rather than a silently capacity-dependent stride — which would have passed every batch-1 test and corrupted the first batch-N run. |
| **#924** (2b-impl-3) | None to the split. Confirmed §5's `write_decode_inputs` placement: writing N selected tokens is inseparable from the batch-N caller, so it correctly stayed in 2b-impl-4. |
| **#930** (2b-impl-4) | The harness artifact above. Also confirmed that pinning the batch symbol at session construction is required for capture, as §2.1 assumed. |

## 1. Batch-1 baseline — measured solo on hardware (the "before")

`profile_native` on `qwen14b-zp`, RTX 4060 Laptop (8 GB), **solo**
(`nvidia-smi --query-compute-apps` verified empty before the run; `0 MiB` used),
`ONNX_GENAI_CUDA_GRAPH=1`, `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1`, 16 emitted
tokens, current `main` (commit `10da8b47`):

| metric | value | gate |
|---|---:|---|
| `generated_token_ids` | `[96347, 3375, 724, 11, 358, 2776, 14589, 311, 6723, 429, 498, 3003, 2581, 6617, 315, 752]` | **exact** match to the #750 reference — batch-1 byte-identity ✓ |
| `htod_bytes_per_token` | 2,349,010,944 | headline (batch-invariant, contention-invariant per #884) |
| `page_ins_per_token` | 188.0 | headline |
| `htod_bytes` (total) | 37,584,175,104 | — |
| `hit_rate` / `byte_hit_rate` | 78.32% / 70.16% | resident set is batch-invariant |
| `evictions` | 2,672 | full generation incl. prefill on an over-budget model |
| `cuda_graph` (full) | `captures=2 replays=28 fallbacks=0 invalidations=1` | `captures>0, fallbacks==0` ✓ (#796) |
| `cuda_graph_measured` | `captures=1 replays=14 fallbacks=0 invalidations=1` | **measured recapture ≈ 1** for uniform fixed-N, confirming #891's prediction |
| `peak_committed_physical_bytes` | 7,725,907,968 | `< managed_limit 7,726,753,178` ✓ (#798) |
| `oversubscribed_bytes` | 0 | ✓ (#798) |
| `ref_underflows` / `byte_underflows` / `unaccounted_committed_bytes` | 0 / 0 / 0 | ✓ |
| `total_weight_bytes` / `kv_bytes_per_token` | 8,330,595,870 / 196,608 | over budget (`fits=false`) |

`effective_htod_gbps=1.718` and `decode=23,779 ms/token` are **not** headline —
this box's wall-clock is WDDM-paging-dominated and ranged widely across runs
(#863); only the byte/page counters are trustworthy (#884). The per-token bytes
here differ from #844's 4.11 GB and #891's synthetic sweep because current `main`
runs the elastic-budget (#866) managed-streaming path at the auto-resolved 7.73 GB
budget rather than the earlier fixed 4 GiB synthetic budget; this is the correct
current-main anchor for a 2b-impl PR.

**This baseline is the regression contract for every increment below:** the
16-token stream must stay byte-identical and the four gate rows must stay green.

## 2. Why "one PR" is not achievable: the decode step is atomic at the
"produces correct tokens" level

The stage 2a/2b guards in `crates/onnx-genai-bench/tests/fused_batch_prefill.rs`
exercise the **stateless** `session.run` fused-forward
(`NativeDecodeSession::run_fused_batch_forward`, `mod.rs:265`), which already
accepts `[N, …]` shapes because it hands owned inputs straight to ONNX. **That is
not the path 2b-impl must change.** The persistent, CUDA-graph-captured decode
step lives in `DecodeCudaState` (`native_decode/cuda.rs`) and is **GPU-only with
no CPU test seam** — the synthetic-decoder tests run the CPU/eager path, never
`DecodeCudaState`. So every change to the persistent path can only be validated
by: (a) build/clippy, (b) the batch-1 byte-identity run above, (c) the mandatory
`mobius_seqmajor_growth_parity_native_cuda` GPU test, and — only once the whole
capability exists — (d) a new batch-N row-identity GPU guard.

Crucially, **there is no partial guard.** A decode step that produces a token
needs *all* of: batched input binding, batched KV allocation, batched KV
growth/commit, batched mask, batched logits read, batched greedy argmax, and
batched capture-symbol pinning. Drop any one and the step either mis-shapes a
binding (hard fail) or silently corrupts a row (the #892 failure class, which
collapsed 16→3 correct tokens). You cannot land "half a batched decode" that
still decodes, so the increments below are validated by *batch-1 non-regression*
until the final one flips batch-N on with a row-identity guard.

### 2.1 The batch-aware *caller* is 2c, and is out of scope here

The public contract `NativeDecodeSession::decode(token_ids: &[TokenId], past_len)`
(`mod.rs:873`) and the greedy fast path `decode_cuda_greedy(token_id, past_len)
-> TokenId` (`cuda.rs:1304`) are **single-sequence**: they take one sequence's
tokens and return one row. A genuine batch-N decode step — "N sequences stepping
together" — needs a caller that presents N sequences and consumes N rows. In this
codebase that caller is the generation driver / `continuous_batch_manager`, which
#891 and this task place in **2c**. Therefore 2b-impl can only deliver the
*capability* plus a direct GPU test (as the fused-forward guards call
`run_fused_batch_forward` directly); it cannot wire a batch-N caller without
crossing into 2c.

## 3. Coupled-site inventory (with evidence)

Grouped by how much work each is. The pleasant surprise: the **KV growth *stride*
math is already batch-general**; the blockers are shape allocation, the mask, the
IO bindings, the device argmax, and capture-symbol pinning.

> **Corrected by #920.** That surprise holds only for **head-major**. A *fixed
> full-context stride* commit is batch-general in both layouts — the batch axis
> is outermost, so sequences sit a constant `capacity × bytes_per_token` apart
> and nothing relocates. A *growing bucket* is different: seq-major's
> per-sequence stride **is** the mutable bucket capacity, so growing it would
> relocate every sequence `b > 0`'s already-written KV. `kv_growth_byte_layout`
> now `bail!`s on that case rather than computing a capacity-dependent stride,
> which would have passed every batch-1 test and corrupted the first batch-N
> run. Read the two paths as separate; this section conflated them.

### Already batch-general (block-count based, no change or trivial)

- `kv_growth_byte_layout` (`cuda.rs:3543`) and
  `copy_kv_prefix_device_to_device[_in_place]` (`cuda.rs:3561+`): stride/segment
  math is driven by `blocks = product(dims before seq_axis)`, i.e. `batch*kv_heads`
  (head-major) or `batch` (seq-major). These already re-stride N batch blocks
  correctly. **This is the part people assume is the blocker; it is not.**

### Hard-pinned to batch 1 (must change, in rough dependency order)

1. **Allocation shape.** `persistent_state_shapes` (`cuda.rs:1459`) pins KV axis
   0 → `1` (both `fixed` and non-`fixed` arms). `DecodeCudaState::new`
   (`cuda.rs:2178`) allocates mask `[1, max_len]` (`cuda.rs:2224`), input_ids
   `[1,1]` / embeds `[1,1,hidden]` (`cuda.rs:2504-2520`), position_ids `[1,1]` or
   `[rank,1,1]` (`cuda.rs:2529`), logits via `persistent_output_shape`
   (`cuda.rs:2131`, collapses symbolic → `1`). All must take a `batch` axis.
2. **Mask writer.** `extend_mask` (`cuda.rs:2752`) writes ones for one row;
   `grow_vmm_mask_in_place` hard-codes `vec![1, new_capacity]` (`cuda.rs:2088`).
   Uniform-length batch is N identical rows — still N rows to write and expose.
3. **Input/position writer.** `write_decode_inputs` (`cuda.rs:3001`) writes one
   i64 at offset 0; `write_position_binding` (`cuda.rs:3011`) writes `rank`
   copies for one row. Need N tokens × N rows (× rank).
4. **Logits read.** `read_logits` (`cuda.rs:3213`) reads `[1,vocab]`. Need
   `[N,vocab]`.
5. **Greedy device argmax — crosses crates.** `read_greedy_result`
   (`cuda.rs:3227`) calls `binding.device_argmax(vocab, …)` returning one token;
   the kernel + `device_argmax_scratch_words` live in **`onnx-runtime-ep-cuda`**
   (its own CI clippy gate, `-p onnx-runtime-ep-cuda --features cuda`). Batched
   argmax = N results → separate crate change.
6. **VMM growth/commit request geometry.** `vmm_growth_requests` (`cuda.rs:1971`)
   commits a flat `0..bytes` per binding — fine for a dense head-major bucket at
   any batch (contiguous from 0), but `grow_vmm_mask_in_place` and the seq-major
   commit path `seq_major_kv_commit_requests` (`cuda.rs:1624`, via
   `kv_commit::live_prefix_ranges` in `onnx-genai-kv`) produce **one dense prefix
   per batch row**; at N>1 seq-major yields N scattered fragments per binding.
   `live_prefix_ranges` must be proven batch-correct (another crate).

   > **Corrected by #920.** `live_prefix_ranges` is in *this* crate —
   > `crates/onnx-genai-engine/src/native_decode/kv_commit.rs:122` — not
   > `onnx-genai-kv`, so no cross-crate coordination was needed. It now takes
   > `batch` and emits one dense run per sequence at a fixed full-context
   > stride, which is relocation-free; see the correction on §3 for why the
   > growing-bucket path is the one that cannot be made batch-general in
   > seq-major.
7. **Capture-symbol semantics.** `collect_unit_symbols` (`cuda.rs:2102`) treats
   batch (axis 0) as a *unit* symbol collapsed to `1`; under batch-N it must be
   *pinned to N*, not collapsed, or every auxiliary output and the attention grid
   mis-size. `pin_fixed_capacity_kv_capture_symbols` (session-side) pins the seq
   symbol; the batch symbol needs the analogous constant-pin so capture admits a
   batch-N grid and `captures>0, fallbacks==0` still holds.
8. **Batch-N entry points + row-identity guard.** New `decode_cuda_batch` /
   `decode_cuda_greedy_batch` returning N rows, a `mod.rs` method reachable from a
   test, and a GPU guard asserting row `i` of a batch-N decode is byte-identical
   (`f32::to_bits`) to a batch-1 decode of the same token stream — the persistent-
   path analogue of `fused_batch_prefill`'s guard.

Plus the mrope, decode-inline sibling (`run_one_token_inline`, `cuda.rs:3147`),
and routed-capture paths each need the batch dim threaded, and the ~120 KB
`native_decode/tests.rs` callers of every changed signature must be updated.

## 4. Proposed split

Each increment holds §1 byte-identical (driver still calls at batch=1) and passes
the mandatory `mobius_seqmajor_growth_parity_native_cuda` GPU test solo; only 2b-4
turns batch-N on.

- **2b-impl-1 — shapes & IO bindings.** Thread a `batch` (constructor-fixed to 1)
  through `persistent_state_shapes`, `persistent_output_shape`, the mask/input/
  position/logits allocations, `extend_mask`, `write_decode_inputs`, `read_logits`.
  Guard: build/clippy + §1 byte-identity + mobius. No behavior change at N=1.
- **2b-impl-2 — batched KV growth/commit.** Generalize `vmm_growth_requests`,
  `grow_vmm_mask_in_place`, and the seq-major `live_prefix_ranges` path to N,
  with pure unit tests on the range math (CPU-testable arithmetic) plus mobius.
- **2b-impl-3 — batched device argmax.** `onnx-runtime-ep-cuda` batched argmax +
  `read_greedy_result` returning N; guarded by its own clippy gate and a device
  test.
- **2b-impl-4 — batch symbol pinning, entry points, row-identity guard.** Flip
  batch-N on end-to-end; add the GPU row-identity guard and the recapture-count
  measurement per batch shape.

If any increment proves independently large (2b-impl-2's seq-major scatter is the
likeliest), split again — the same rule that produced #844 and #891.

## 5. The KV-multiplication trade (restated against #891)

Batching amortizes the weight stream (`htod_bytes_per_token` ∝ 1/N) but adds a
per-token floor `L × kv_bytes_per_token` that **never** amortizes, and a hard
ceiling `N_max ≈ budget / (L × kv_bytes_per_token) ≈ 19 @ 2048 ctx`. With #866's
elastic budget, batched KV growth *reclaims* weight residency, so as N grows we
expect `htod_bytes_per_token` to fall from the 1/N amortization **but partly
offset upward** as the shrinking weight budget lowers `hit_rate`. 2b-impl-4 must
report both movements, not a bare speedup, and must show the crossover if batched
decode ever costs more than it saves at a reachable N on this hardware.

## 6. Measurement protocol for the implementation PRs

Lead with `htod_bytes_per_token` and `page_ins_per_token`; never headline
wall-clock (#863). Verify `nvidia-smi --query-compute-apps` empty before **every
individual** run (#851). Report the **measured** recapture count per batch shape
(§1 shows ≈1 for batch-1). Assert token identity whenever residency behaviour is
touched — it is the #892 corruption detector.
