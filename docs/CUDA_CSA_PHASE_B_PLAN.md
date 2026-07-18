# CUDA `CompressedSparseAttention` Phase B — Device-Resident Fused Kernel Plan

Status: **DRAFT for Justin's review** · Author: Keaton (CUDA architect) · Based on
`origin/main` 73629cd (Phase A landed).

This document decomposes the device-resident fused CUDA CSA kernel (doc
[`DEEPSEEK_CSA_MTP_RUNTIME.md`](DEEPSEEK_CSA_MTP_RUNTIME.md) §4.8, §6 Phase 6)
into ordered sub-phases **B0…B7**, each of which:

- lands to `main` on its own with the existing CPU-parity GPU tests still green;
- keeps the Phase A **host-staged path as a correctness fallback** until the
  device path fully replaces it (switchover in B7);
- has an explicit **done / pass bar** and a **rollback story**.

The CPU kernel
`crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs` is the
temporary Phase B implementation oracle, exactly as it was for Phase A
(`crates/onnx-runtime-ep-cuda/tests/compressed_sparse_attention_gpu.rs`). The
official BF16 reference remains the production numerical contract; B7
switchover requires official-golden parity or prior reconciliation of the CPU
oracle.

---

## Decisions for Justin (resolve before / during B0)

These block or reshape specific sub-phases. Recommended defaults are given; each
needs an explicit yes/no.

- **D1 — Parity target: CPU-f32 oracle vs. official BF16 goldens.**
  The CPU oracle models the frozen BF16/FP8/FP4 casts at the *finalize* stages
  (RMSNorm→RoPE→FP8/FP4, `half::bf16::from_f32`), **but its sparse-attention
  score/value reduction accumulates in pure f32 in ascending index order**
  (`dot` / `accumulate_value`, kernel lines ~2382–2430). The official
  `kernel.py` does a **BF16 `p_j` cast before the value GEMM** and BF16 QK. So
  the CPU oracle is *not* bit-identical to official attention numerics.
  **Question:** does the CUDA kernel target (a) the CPU oracle (f32 attention
  accumulation, our Phase A contract and current test gate), or (b) official
  BF16 kernel.py numerics? These differ measurably in the softmax denominator.
  **Recommendation:** use the **CPU oracle** as a temporary implementation gate
  so existing bit-parity tests keep catching device regressions, but do not
  freeze its f32 reduction as the production numerical contract. Reconcile the
  CPU oracle with official BF16 goldens, or add a separate official-golden gate,
  before B7 switchover. B1 may initially accumulate in f32 in fixed index order.

- **D2 — FP8/FP4 device compute strategy.**
  CSA needs **both** device *dequant* (reading `past_*` records) and device
  *quantize* (finalizing new records: FP8 E4M3 block-64 with power-of-two E8M0
  scale `2^ceil(log2(amax/448))`; FP4 E2M1 block-32 `2^ceil(log2(amax/6))`).
  `block_quantized_matmul.rs` has device **dequant** helpers only
  (`decode_weight`, `e2m1_doubled`, `e8m0_half_scale`) and is itself
  `cuda_graph_compatible()==false`. **Question:** reuse/extract those decode
  helpers into a shared device header and write new *quantize* device code, or
  build a self-contained CSA quant/dequant device module?
  **Recommendation:** extract the decode primitives into a shared
  `kernels/cuda/block_quant.cuh`-style NVRTC snippet, add matching quantize
  primitives there, and unit-test the round-trip against the CPU
  `block_dequant` module. Do **not** take a dependency on the graph-incompatible
  matmul kernel path.

- **D3 — Device-resident cache: fixed-capacity vs. growing; memory budget.**
  Graph capture requires **stable buffer addresses** (see
  `onnx-genai-ort/src/session.rs:51`). That mandates **fixed-capacity**
  device buffers sized at `max_seq_len`. Per-batch budget at ratio-4:
  `(max_seq_len/4)·583 B` compressed KV + `(max_seq_len/4)·68 B` index key +
  carries (`8·2·1024·4 B` + `8·2·256·4 B`) + dense ring `W·583 B`. **Question:**
  what `max_seq_len` and sliding-window `W` do we budget for, and is a
  fixed-capacity cap acceptable (fail-closed when exceeded)?
  **Recommendation:** fixed-capacity, sized from package metadata
  `max_seq_len`. Claim time validates static metadata and supported bounds;
  session/runner initialization reserves the buffers and fails before execution
  if the reservation cannot be satisfied. Do not make EP capability depend on
  transient free device memory.

- **D4 — Per-row cursor lengths (ragged batch).** §10-Q10 pins v1 to
  **equal-length compression/index cursors within a batch**; ragged per-row
  lengths are a fast-follow. **Question:** confirm Phase B stays equal-length
  (simple regular layout, one cursor set per forward) and defers ragged to a
  later phase. **Recommendation:** yes — equal-length only in B0…B7, enforced
  by validation. Fail fast or use a non-captured fallback for ragged rows, and
  retain per-row cursors as the immediate fast-follow.

- **D5 — Device top-k determinism & graph-capturability.** Ratio-4 selection is
  `score.topk(min(512, floor(total/R)), largest=True, sorted=True)` with a
  frozen tie order. The existing `topk.rs` CUDA kernel is
  `cuda_graph_compatible()==false`. **Question:** is a graph-capturable,
  deterministic device top-k (bit-identical tie order to the CPU oracle) in
  scope for B4/B6, or do we accept a bounded host readback of the (≤512) index
  set until B6? **Recommendation:** allow a host readback of *indices only* in
  B4 (kernel still not graph-capturable then), and make top-k fully
  device-resident + capturable in B6.

- **D6 — Checkpoint/restore ownership.** Stream-ordered cursor
  checkpoint/restore for speculative decode (§4.6): does `DecodeState` own the
  cursor journal and call kernel-exposed `checkpoint()`/`restore(cursor)`, or
  does the kernel own device-resident cursors that the engine rewinds?
  **Recommendation:** the backend/kernel owns authoritative device-resident
  logical-length cursors and token-to-auxiliary-state mapping; the engine owns
  composite checkpoint orchestration. Restore uses an opaque checkpoint plus an
  accepted-token offset, validates sequence/generation identity, and restores
  logical lengths plus active carry/index state without recompression. Physical
  inactive tails may remain stale when every reader is length-masked. If
  speculative writes can overwrite committed carry, checkpoint the bounded
  overwritten region.

- **D7 — Retiring the host-staged fallback.** CSA-7 was flipped to
  `native_csa_required` (§9 owner verdict). **Question:** in B7, remove the
  host-staged path entirely, or keep it behind a `--csa-oracle` debug flag as an
  in-process differential oracle? **Recommendation:** keep it behind an explicit
  test/diagnostic flag (invaluable for regression triage), but never use it as
  an automatic fallback or include it in production performance/CUDA Graph
  eligibility claims.

---

## Shared device data structures (frozen v1, from the claim gate + contract)

All widths below are already enforced by the Phase A claim gate
(`compressed_sparse_attention.rs` `validate_ratio4_claim` /
`validate_ratio128_claim`) and the CPU `CacheFormat::stored_width`.

| Stream | Persistent records (device-resident) | Incremental carry (device-resident) |
|---|---|---|
| attn `R=4` | `u8[B, max_seq_len/4, 583]` — hybrid FP8: 7×(64 E4M3 + 1 E8M0)=455 B + 64×BF16 RoPE=128 B | `f32[B, 8, 2, 1024]` (`score_state` init `-inf`; slots 0:4 prev block, 4:8 current) |
| attn `R=128` | `u8[B, max_seq_len/128, 583]` (fp8) **or** `f32[B, …, 512]` | `f32[B, 128, 2, 512]` (slot = `pos mod 128`) |
| index `R=4` | `u8[B, max_seq_len/4, 68]` — FP4: 4×(16 packed + 1 E8M0)=68 B | `f32[B, 8, 2, 256]` (2·ID=256) |
| dense ring | `u8[B, W, 583]` (or f32) physical ring, oldest→newest | — |
| logical cursors | `token_len, compressed_len, compression_carry_len, index_len, index_carry_len` (per forward, device-resident from B6) | — |

Symbols (v1): `D=512`, `RD=64`, `N=num_heads`, `I=index_num_heads`, `ID=128`,
`K=index_topk≤512`, ratios ∈ {4,128}.

## The fused pipeline stages (device targets, mirroring the CPU oracle)

1. **Compression update** — per-output-dimension FP32 softmax pool over the
   carry (`pool_ratio4_record` / `pool_ratio128_record`), `cast_BF16`, RMSNorm,
   compressed-RoPE at *uncompressed block start*, then FP8 finalize
   (`finalize_attention_record`). Emits new packed records + updates carry;
   **updates only new records**.
2. **Index-key update** *(R=4 only)* — independent R=4 overlap compressor,
   Hadamard(`1/√D`), FP4 E2M1 finalize (`finalize_index_record`).
3. **Index-query finalize** *(R=4 only)* — RoPE, Hadamard, FP4
   (`finalize_index_query`).
4. **Index scoring** *(R=4 only)* — `dot(qi,ki)` → `relu` → weighted sum over
   index heads with `w` (`select_ratio4_topk` scoring half).
5. **Selection** *(R=4 only)* — causal + valid-length mask, then
   `topk(min(512,…), largest, sorted)` with frozen tie order.
6. **Candidate assembly** — `[dense window ⧺ (R4 top-k | R128 all completed)]`
   compressed indices; `-1` invalid; **no global selected-KV materialization**.
7. **Sparse sink-softmax attention** — online block-of-64 softmax over the
   candidate list; learned `head_sink` as logit-only denominator mass added
   after the running max; value reduction → `Y`
   (`ratio4_attention` / `ratio128_attention`).
8. **Writeback** — present compressed KV / carry / index key / index carry /
   optional `selected_indices`.

---

## Sub-phase breakdown

### B0 — Device-execution scaffolding & stage-boundary parity harness

- **Scope.** No numeric change. Introduce (a) a fixed-capacity device buffer
  manager for the five state streams sized from metadata (D3), (b) a
  per-stage `CsaStageMode { Host, Device }` dispatch so each pipeline stage can
  independently run host-staged (oracle) or on device, defaulting **all-Host**
  (= Phase A behavior), and (c) a shared device quant/dequant NVRTC snippet
  scaffold (D2) with round-trip unit tests only. Add golden-capture plumbing.
- **Device structures.** Allocate but do not yet own the buffers listed above;
  in all-Host mode they are still threaded as `past_*→present_*` graph I/O.
- **Fused stages implemented.** None (dispatch skeleton only).
- **CPU parity.** Byte-identical to Phase A because all stages still delegate to
  the CPU oracle via the staged path.
- **Tests.** Existing `compressed_sparse_attention_gpu.rs` unchanged and green;
  new unit tests for the device FP8/FP4 quant↔dequant round-trip vs. CPU
  `block_dequant`.
- **CUDA-graph.** `cuda_graph_compatible()` stays `false`.
- **Risks.** Low. Mostly plumbing; risk is over-engineering the dispatch enum.
- **Complexity.** **S.**
- **Rollback.** Revert the commit; Phase A path untouched.

### B1 — Device sparse sink-softmax attention core (ratio-128 first)

- **Scope.** Implement stage 7 (and candidate read of stage 6) on device for
  **ratio-128**: online block-of-64 softmax with the sink added after the
  running max, matching the CPU oracle's **f32** score/value accumulation in
  ascending index order (D1). Compression (stages 1) stays host-staged this
  slice; the assembled candidate KV is uploaded and the attention runs on
  device, writing `Y` directly. Reuse `standard_attention.rs` online-softmax
  block pattern and `sparse_kv_gather.rs` index addressing (fused, no `selected`
  materialization).
- **Device structures.** Reads dequantized candidate records on the fly;
  `head_sink` f32[N]; candidate index list int32 (`-1` invalid).
- **Fused stages.** 6 (read) + 7.
- **CPU parity.** Ratio-128 `Y` vs. CPU oracle; determinism via fixed
  per-row reduction order.
- **Tests.** Extend the ratio-128 prefill→decode→decode GPU test to assert `Y`
  parity with the device attention stage enabled; keep state stages host.
- **CUDA-graph.** Still `false` (compression host round-trip remains).
- **Risks.** **Medium-high** — the online-softmax + sink-after-max + f32-order
  determinism is the numerical heart; mismatched accumulation order breaks
  parity. **This is one of the hardest slices.**
- **Complexity.** **L.**
- **Rollback.** Flip ratio-128 stage-7 dispatch back to `Host`.

### B2 — Device ratio-128 compression + device-resident FP8/f32 cache & carry

- **Scope.** Move stage 1 to device for ratio-128 (per-dim FP32 softmax pool,
  BF16 cast, RMSNorm, compressed-RoPE, FP8 E4M3 finalize), and keep
  `compressed_kv`/`carry` **device-resident across steps** with stable
  addresses (D3), updating **only new records** and resetting only the touched
  carry row (`pos mod 128`). Uses the B0 quant primitives (D2). After this
  slice **ratio-128 is fully device-resident** with no host round trip in steady
  decode.
- **Device structures.** `u8[B,·,583]`/`f32[B,·,512]` compressed KV, `f32[B,128,2,512]`
  carry, device cursors `compressed_len`, `compression_carry_len`.
- **Fused stages.** 1 + (B1's) 6–7 for ratio-128.
- **CPU parity.** Full ratio-128 output set (`Y`, `present_compressed_kv`,
  `present_compression_carry`) across the decode-boundary test.
- **Tests.** The existing ratio-128 stateful GPU test, now with device
  compression enabled; add a boundary test that crosses two 128-blocks.
- **CUDA-graph.** Ratio-128 path becomes *capture-ready in principle* but stays
  `false` until B6 (top-k readback absent for R128, so R128 could flip early —
  see B6 note).
- **Risks.** **Medium** — FP8 quantize numeric parity (power-of-two E8M0 scale,
  amax clamp `1e-4`) and carry-reset timing.
- **Complexity.** **L.**
- **Rollback.** Flip ratio-128 stage-1 dispatch to `Host`; buffers fall back to
  graph-threaded I/O.

### B3 — Device ratio-4 index-key compression (FP4 device path)

- **Scope.** Implement stage 2 on device: the independent R=4 overlap index
  compressor, Hadamard(`1/√ID`), FP4 E2M1 block-32 finalize
  (`finalize_index_record`); keep `index_key`/`index_carry` device-resident.
  Ratio-4 *attention* and *scoring* remain host-staged this slice.
- **Device structures.** `u8[B,·,68]` index key, `f32[B,8,2,256]` index carry,
  cursors `index_len`, `index_carry_len`.
- **Fused stages.** 2 (ratio-4).
- **CPU parity.** `present_index_key` / `present_index_carry` vs. oracle,
  including the `c=0` zero-value/`-inf` boundary and the overlap shift.
- **Tests.** New ratio-4 GPU test asserting index-stream parity across a decode
  boundary (mirrors the CPU
  `ratio4_index_compression_topk_and_streaming_match_independent_oracle`).
- **CUDA-graph.** `false`.
- **Risks.** **Medium** — FP4 E2M1 quantize parity and the overlap/`c=0`
  masking; reuses B2's quant lessons.
- **Complexity.** **M.**
- **Rollback.** Flip stage-2 dispatch to `Host`.

### B4 — Device ratio-4 index scoring + deterministic top-k selection

- **Scope.** Implement stages 3–5 on device: index-query finalize
  (RoPE/Hadamard/FP4), `dot→relu→weighted-head-sum` scoring, causal +
  valid-length masking, then deterministic `topk(min(512,…), largest, sorted)`
  reproducing the CPU oracle's exact tie order. Per D5, an index-only host
  readback is permitted here (kernel stays non-capturable).
- **Device structures.** Score buffer `f32[B,S,C]`, selected index buffer
  `int32[B,I,S,K]`.
- **Fused stages.** 3 + 4 + 5 (ratio-4).
- **CPU parity.** `selected_indices` **bit-identical** to the oracle, including
  tie order, `-1` padding, and topk clamp to available causal records.
- **Tests.** Extend the ratio-4 GPU test to assert `selected_indices` parity;
  adversarial tie-value fixtures.
- **CUDA-graph.** `false` (host index readback).
- **Risks.** **High** — deterministic tie-breaking bit-identical to PyTorch
  `sorted=True` semantics is the single most error-prone piece. **Hardest
  slice.**
- **Complexity.** **L.**
- **Rollback.** Flip stages 3–5 dispatch to `Host`.

### B5 — Device ratio-4 fused selection→attention (full device residency)

- **Scope.** Wire B1's fused attention core to consume the ratio-4 candidate
  list (dense window ⧺ device top-k) with **no `selected` materialization**,
  applying optional `attention_bias`. After this slice **ratio-4 is fully
  device-resident** end-to-end (still with the B4 index readback until B6).
- **Fused stages.** 6 + 7 (ratio-4), consuming B2–B4 device state.
- **CPU parity.** Full ratio-4 output set (`Y` + all present state +
  `selected_indices`).
- **Tests.** Full ratio-4 prefill→decode→decode GPU parity, incl. optional
  `attention_bias` case (mirrors the Phase A bias claim-gate coverage).
- **CUDA-graph.** `false` (B4 readback).
- **Risks.** **Medium** — candidate assembly ordering (window vs. compressed
  offset `+W`/`+S`) and sink handling under sparse padding.
- **Complexity.** **M.**
- **Rollback.** Flip ratio-4 attention dispatch to `Host` (falls back to B4
  host-staged attention on device index).

### B6 — CUDA-graph capture compatibility

- **Scope.** Eliminate every remaining host round trip and per-call
  alloc/free/sync: make the top-k fully device-resident and capturable (remove
  the B4 readback, D5), pool all workspaces, pin buffer addresses, and move the
  logical-length cursors device-resident so kernels advance them without host
  involvement. Flip `cuda_graph_compatible()` → `true` when the config is
  capturable, and let `supports_op` advertise capture eligibility.
- **Fused stages.** All, capture-clean.
- **CPU parity.** Unchanged outputs; add a **capture+replay** equivalence test
  (byte parity between eager and replayed decode).
- **CUDA-graph.** `true` for supported ratio/layout/dtype/shape; ratio-128 may
  flip as early as end of B2 if isolated.
- **Risks.** **High** — stable-address discipline across the whole state set;
  graph-safe device top-k; matches the known blockers that keep
  `matmul.rs`/`topk.rs` non-capturable today.
- **Complexity.** **L.**
- **Rollback.** Keep `cuda_graph_compatible()` returning `false`; eager device
  path (B1–B5) still fully functional.

### B7 — Stream-ordered checkpoint/restore + switchover

- **Scope.** Implement device `checkpoint()` /
  `restore_prefix(checkpoint, accepted)` of the five logical cursors and any
  bounded overwritten active carry state, with no recompress, stream-ordered
  for speculative decode (D6, §4.6). Switch the **default** path from
  host-staged to device; retain host-staged behind a diagnostic oracle flag
  (D7). Wire the
  observability metrics from §8 (attention mode per layer, bytes avoided,
  cursor lengths, stage timings, sink mass, rollback counts, host/device bytes).
- **Fused stages.** All + rollback semantics.
- **CPU parity.** Speculative accept/reject rollback parity vs. CPU oracle;
  greedy-token bit-identity (§10-Q14) across a draft/verify/correct sequence.
- **Tests.** Speculative rollback GPU test (accept-k then reject); MTP
  integration smoke; metrics presence assertions.
- **CUDA-graph.** `true`; steady decode shows **no host index/cache round
  trips** and measured speed/memory win over dense fallback (§6 Phase 6 pass
  bar).
- **Risks.** **Medium-high** — stream-ordered restore correctness under
  capture; composite checkpoint atomicity with target/MTP state (MTP-6).
- **Complexity.** **L.**
- **Rollback.** Flip default back to host-staged (fallback retained); disable
  capture.

---

## Pass-bar summary

| Sub-phase | Done / pass bar | Graph-capturable after |
|---|---|---|
| B0 | Byte-identical to Phase A; quant round-trip unit tests green | no |
| B1 | Ratio-128 `Y` parity (device attention, host state) | no |
| B2 | Ratio-128 full parity, device-resident state, no host round trip | (R128 candidate) |
| B3 | Ratio-4 index-stream parity | no |
| B4 | Ratio-4 `selected_indices` bit-identical incl. ties | no |
| B5 | Ratio-4 full parity, device-resident | no |
| B6 | Capture+replay byte parity; `cuda_graph_compatible()==true` | yes |
| B7 | Speculative rollback + greedy-token identity; metrics; default switchover | yes |

## Top risks (whole programme)

1. **Deterministic device top-k with frozen tie order (B4)** bit-identical to
   the CPU oracle / PyTorch `sorted=True` — highest-probability parity breaker.
2. **Attention numerics & parity target (B1 + D1)** — the CPU oracle uses f32
   attention accumulation while official kernel.py uses BF16 `p_j`; the device
   kernel must match whichever we choose, and online-softmax reduction order
   must be pinned for bit-parity.
3. **Stable-address / capture discipline across all five state streams + FP8/FP4
   device quantize parity (B2/B3/B6)** — power-of-two E8M0 scaling and
   fixed-capacity residency must hold across capture/replay and speculative
   rollback without silent divergence.

---

*See [`DEEPSEEK_CSA_MTP_RUNTIME.md`](DEEPSEEK_CSA_MTP_RUNTIME.md) §4.3–4.8, §8,
§10, and the Frozen Numerical Contract CSA subsections for the authoritative
equations and layout each sub-phase must reproduce.*
