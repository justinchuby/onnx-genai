# Memory Architecture: Unified Design

> Consolidates memory management design from
> [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md),
> [DESIGN.md](./DESIGN.md) §26.11 & §43.2,
> [MOE_SUPPORT.md](./MOE_SUPPORT.md) §7,
> [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) §8, and
> [DISTRIBUTED_RUNTIME.md](./DISTRIBUTED_RUNTIME.md) §3, §5, §12.

**Status:** Design — Consolidated
**Author:** Claw (with Justin)
**Date:** 2026-07-19

---

## Implementation status

This document describes the target architecture. Most of it is not built yet,
and the parts that are built are mostly not connected to anything. The table
below is what a reader would otherwise have to reconstruct by searching, and it
is easy to get wrong: a first pass at this used a glob that skipped every
top-level `src/*.rs` file and concluded the pressure protocol did not exist,
when in fact it is 1955 lines.

| layer | designed in | status | where |
|---|---|---|---|
| L1 EP memory | §2 | **implemented** | `ExecutionProvider::{allocate, deallocate, copy}` in `onnx-runtime-ep-api` |
| L2 Weight residency | §3 | **design only** | no `WeightResidencyManager` type exists |
| L3a DeviceGovernor | §4 | **implemented under its old name** | `ResourceGovernor`, `crates/onnx-genai-scheduler/src/governor.rs` |
| L3b HostGovernor | §5 | **implemented; adapter in `HostLeaseGovernor`** | `crates/onnx-genai-scheduler/src/pressure.rs`, 1955 lines, modelled in `specs/tla/PressureProtocol.tla` |
| L4 ClusterCoordinator | §6 | **design only** | no type exists |
| Lease contract | §1.1 | **implemented** | `crates/onnx-runtime-memory-governor` |
| Allocator contract | §1.2, §1.3, §1.5 | **implemented on all three backends** | `onnx-runtime-memory-governor/src/allocator.rs`; CPU EP, ONNX Runtime and CUDA EP each implement it |
| Virtual contiguity | §1.6 | **implemented; managed no-spill VMM is the default for native CUDA (#755)** | `crates/onnx-runtime-virtual-memory`; `CudaVmmAllocator` in `onnx-runtime-cuda-memory` reserves one range and maps 2 MiB granules on demand, leasing each before it maps it. On native CUDA the authority-governed VMM/pool path is now selected **by default, without a flag** (#755): memory-strategy inference runs unconditionally (#752) and drives policy, so a plain `serve` gets managed **no-spill** mode and does not rely on WDDM shared-memory fallback. An explicit `serve --vram-limit <bytes>` now **overrides the inferred device budget** rather than being the trigger. When the resolved budget cannot hold the package weights the runtime **automatically enables weight streaming/offload** instead of failing, so being larger than the budget is a supported configuration; a model that fits stays `FullResident` and does **not** page. **Exception (#864): on Windows/WDDM the OS shared-memory fallback pages over-budget weights from host RAM over PCIe ~30× faster than managed streaming for the single-touch decode access pattern, so managed weight streaming is *not* auto-enabled there — the effective strategy becomes `Compatibility` (`weight_offload_enabled=false`, `managed_no_spill=false`).** This affects only the *inferred* default: an explicit `ONNX_GENAI_WEIGHT_OFFLOAD=1`, `--vram-limit`, or device-budget override still selects managed streaming, and `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1` forces it. On Linux there is no shared-memory fallback, so the managed path is unchanged (#783). Failure to construct the required VMM arena/pool is fatal before model allocation and names both the requested limit and provider failure; it never silently falls back to ungoverned `cuMemAlloc`. The legacy allocator remains reachable for one release via `ONNX_GENAI_LEGACY_ALLOCATOR=1` (back-compat: `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0`), which restores the compatibility fallback. The resolved budget, chosen strategy and offload state are printed at startup and exposed in `/v1/resources` (`memory_strategy`). |
| Composability of the memory paths | — | **authority-governed native path implemented** | The historical independent VMM and weight-offload toggles did not compose (#704). On native CUDA — by default under #755, or under an explicit byte limit — native CUDA constructs one authority before the allocator, charges the physical-handle pool once, registers reloadable weight residency explicitly, and admits KV/workspace growth transactionally. The compatibility path remains available through the legacy-allocator opt-out; performance parity with WDDM remains separate work. |
| Weight offload vs the OS | — | **fixed 2026-08-12: WDDM shared memory is now the default on Windows for inferred-over-budget models (#874)** | Measured directly on `qwen14b-zp` (8.33 GB weights vs a 7.73 GB budget), same binary, same prompt, byte-identical output, solo with `nvidia-smi` verified empty before every run: **WDDM 8.09/7.78 tok/s with `htod_bytes_per_token = 0`** (true zero-copy — the kernel reads weights in place from host RAM over PCIe) against **managed streaming 0.11 tok/s**. On `main` immediately prior the managed default measured 0.05–0.08 tok/s, so the end-to-end effect on this box is ~100×. **The cause is structural, not a tuning gap:** each weight is read *exactly once per decode step* (922 initializers, ~867 lookups/step, `SequentialDense`), so both paths move the same bytes over the same link, but ours adds a CPU memcpy into pinned staging, a VRAM allocation, a `cuMemMap`, an eviction and a synchronize — to buy VRAM residency that is discarded before it is ever re-read. **Copying to VRAM only pays off when the data is re-read from VRAM before eviction, and there is no intra-step reuse.** #874 therefore stops auto-enabling managed streaming on Windows when the OS fallback is available; explicit requests (`ONNX_GENAI_WEIGHT_OFFLOAD=1`, `--vram-limit`, a device-budget override) are still honoured, and `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1` forces the managed path back (parsed so an *unrecognized* value keeps the fast default — the inverse of the `ASYNC_PAGEIN` trap). Linux is unchanged and must stay so: with no shared-memory fallback an over-budget model simply fails there, so managed streaming competes with "does not run", not with something faster (#783's lesson about not inheriting a platform-specific conclusion). Trade stated plainly: the `Compatibility` arm has **no device budget** (`managed_limit_bytes=None`), so on a host short of RAM an over-budget model can thrash, and the governor cannot see system-wide pressure (#863). Keeping the no-spill arena while merely disabling offload is **not** an option — the physical-handle pool is a hard cap and `cuMemCreate` cannot spill, so with nothing paging the remainder the load fails outright. The durable fix remains the hybrid in #864: a resident hot set **plus zero-copy cold reads**, which is what would beat the OS rather than merely stop losing to it. Historical figures this row previously carried (6.01 tok/s WDDM, 27× behind, `h2d_copy` 18.8% / `staging_fill` 9.0% / `vram_free` 9.0% / `vram_alloc` 2.3%, capture disabled under offload) are superseded: capture now runs under offload (#796) and the eviction-policy defect was fixed (#723). #705, #864, #874 |
| `--vram-limit` | — | **override, not trigger; managed no-spill is the default (#755)** | Under #755 memory-strategy inference runs on every native CUDA load and selects managed no-spill by default; `--vram-limit` now **overrides** the inferred device budget rather than being what turns the managed path on. If package weights exceed the resolved budget, weight offload is enabled automatically and its allowance is derived as `budget − governed device state`; the same authority owns physical VMM handles, mapped weights, KV, and workspace. A 6 GiB qwen2.5-14b int4 live run loaded, generated the expected `Paris`, stayed at 5,534,384,128 owned bytes, rejected an 8K request pre-header with 429, and completed four queued short requests without overlap-only rejection. Crossing the first physical KV-growth boundary transferred 201,326,592 bytes through one grant and reported 201,326,592 mapped KV bytes; later governed capacity exhaustion returned 429 rather than 500/OOM. Managed initialization failure is an early arithmetic/provider error, while `ONNX_GENAI_LEGACY_ALLOCATOR=1` (back-compat `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0`) restores the prior non-VMM/WDDM-capable compatibility fallback. Limitation: the native server's FIFO per-request engine queue verified ordering and absence of overlap-only rejection, but did not demonstrate simultaneous live KV accumulation. Caveat carried, not closed: `activations_bytes=unknown` and `runtime_overhead_bytes=unknown` are not subtracted (#514). PRs #717, #736. |
| Managed no-spill VMM default (#755) | — | **implemented; measured on native CUDA** | Memory-strategy inference selects the managed no-spill VMM path by default (no flag) on native CUDA, and auto-enables weight streaming when the resolved budget cannot hold the weights. Locked by the ignored live test `qwen2_5_0_5b_managed_vmm_default_e2e` (Qwen2.5-0.5B int4 mobius, same process): **(A)** default no flag → `strategy=FullResident`, `weight_offload_enabled=false`, `managed_no_spill=true`, resolved budget 7,730,940,928 bytes, committed 381,681,664 device bytes, **0 page-ins** (a fitting model does not start paging); **(B)** synthetic over-budget via an explicit 268,435,456-byte `--vram-limit` → `strategy=DynamicWeightResidency`, `weight_offload_enabled=true`, `managed_no_spill=true`, streaming engaged with **434 page-ins / 432 evictions**, committed bounded to 90,177,536 device bytes (« the cap — no WDDM spill), and at that extreme synthetic cap the residency arithmetic **refused cleanly** ("no page is evictable") rather than spilling; **(C)** `ONNX_GENAI_LEGACY_ALLOCATOR=1` → `managed_no_spill=false` (legacy allocator observable); **(D)** an explicit `--vram-limit` overrides the resolved device budget. Deterministic counters only; wall-clock omitted (this box ranged 3.9–28 tok/s across identical runs, and other agents build concurrently). A native large-model run that genuinely exceeds VRAM was not measured: `qwen14b-zp` lacks `inference_metadata.yaml` (#384), so the over-budget condition was synthesized with a small explicit budget, as noted. CUDA-graph segment count is not a public counter; capture-ON-under-offload on the stable-VA path is pinned by the #796 unit tests, not re-measured here. |
| WDDM prefers OS shared-memory over managed streaming by default (#864) | — | **implemented; measured on native CUDA (`qwen14b-zp`, genuinely over budget)** | On Windows/WDDM the #755 auto-enable of managed weight streaming for an *inferred* over-budget model is now skipped in favor of the OS shared-memory fallback (weights sit in host RAM and are read in place over PCIe), which #864 measured ~30× faster for the single-touch decode access pattern — copying a weight into VRAM only to evict it before any intra-step re-read is pure overhead. The effective strategy becomes `Compatibility` with `weight_offload_enabled=false`, `managed_no_spill=false`, no governed device budget; residency is handed to the OS (trade: on a host with little free RAM an over-budget model may thrash — the governor cannot see system-wide pressure, per #863). Decided in `memory_strategy.rs` (policy change only; the residency cache, arena, and offload implementation are untouched). Gated on `cfg!(windows)` as a pragmatic stand-in for WDDM — detecting WDDM vs TCC at this layer would cross into architecture; under TCC `cuMemCreate` fails at the physical limit rather than spilling, so TCC-mode Windows users needing the managed path use the force knob. **Scope guards:** honored, not overridden — an explicit `ONNX_GENAI_WEIGHT_OFFLOAD=1`, `--vram-limit`, or `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` still selects managed streaming; `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1` forces it (opt-in, unrecognized values keep the fast fallback — deliberately *not* the `ASYNC_PAGEIN` trap shape); Linux is unchanged (no fallback exists, #783). Measured direct A/B, same binary, same prompt, GPU verified clear before every run, **token IDs byte-identical across all 7 runs**: WDDM default n=4 **median 7.18 tok/s** (7.10–7.67), prefill ~191 ms, `htod_bytes_per_token=0`, `page_ins_per_token=0`; forced managed (`ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1`) n=3 **median 0.86 tok/s** (0.84–0.88), prefill ~1,097 ms, `htod_bytes_per_token=3,943,690,240`, `page_ins_per_token=372` — **8.3× on medians on a quiet box** (the issue's 30.7× was under this box's characteristic noise; the deterministic counters match exactly and confirm which path ran). Unit-locked in `memory_strategy.rs` (`wddm_over_budget_prefers_shared_memory_and_disables_streaming`, the WDDM/MoE/fitting/override/force variants, and `managed_default_no_flag_over_budget_model_auto_streams` proving Linux still streams). |
| Weight residency cache (CUDA) | — | **had a 0% hit rate for its entire life; fixed 2026-08-06** | `evict_to_fit` was a plain LRU evicting from the least-recently-used end, while the decode weight walk is a **cyclic sequential scan** over the layers — the pessimal pairing, which returns the hit rate to zero at *every* capacity. Measured: **6,936 page-ins, 0 hits, at both 3 GB and 6 GB budgets**, with staged bytes identical (7,870,916,608 per step) because miss traffic under this pathology is invariant to capacity. Rivals eliminated on hardware: the budget knob works (peak resident 3.0 vs 6.0 GB) and staged bytes reconcile with `page_ins × page size`. A stable-resident-subset policy measures **74.18%** against a `B/W ≈ 76%` ceiling, page-ins 6,936 → 1,791, evictions 6,286 → **0**. #720, PR #723 |
| KV page storage | — | **contract opened 2026-08-06; still host-only** | `onnx-genai-kv` implements paging, ref-counting/CoW, prefix sharing and quantization layout, and **never touched device memory**: storage was host `Vec`, `Device::Gpu(0)` a label on a struct field, and a tier migration one enum assignment moving zero bytes. Documented as a placeholder in `tiered.rs` from the start — the seam was designed in and the GPU backend behind it was never built. `PageTable` now holds `Box<dyn KvPageStore>` so a third party can supply a store without patching the crate; host stores expose slices, device stores expose an opaque span, and a host view of a device page requires explicit materialization. Stages 2–5 in #721. PR #726. Device KV paging is **not** owned by `onnx-genai-kv`: under the VMM design it is owned by the CUDA VMM layer (`CudaVmmAllocator`, the #740 physical-handle pool), with committed-granule admission (#745) and growth grants (#748). `native_decode/cuda.rs` still has no `PageTable`/`PagedKvCache` consumer, and `paged_gqa.rs` is a batch-1 CPU primitive. |
| Captured graphs across a VMM remap | — | **verified on hardware 2026-08-06** | A CUDA graph instantiated before `cuMemUnmap`/`cuMemCreate`/`cuMemMap` at the same virtual address replays correctly afterwards and writes into the **new** physical pages — sentinel-proven, closing the page-recycling confound. The growth-shaped case passes, and one physical handle mapped at two virtual addresses is readable by captured work through either. Untested and treated as unsafe: unmapping while a replay is in flight. `cuMemMap` during capture returns `CUDA_SUCCESS` but is **not** proven replayable, so growth is issued outside the captured segment. This is the premise under #721 stage 4 and under re-scoping #716, and its stated falsifier did not fire. PR #727 |
| Activation planning | — | **wired for measurement; not yet allocating** | `crates/onnx-runtime-memory` now has a consumer: the session executor builds a `ViewMap`, runs the planner, and reports peak vs naive. Measured 2.4x-2.7x on qwen2.5-0.5b, though see #671 for a review finding that may inflate that. Sharing slots for real is #670 |
| Native KV page size | — | **wrong unit** | `governor_kv_config` puts a token count in `page_size_bytes` when the geometry is unknown, so the native `bytes_per_token` is 1 (#628) |
| Governor integration, server path | — | **fixed 2026-08-06** | `/metrics` reported `vram_used_bytes = 0` while a 14B was loaded and generating, because the server read the scheduler's `ByteBudget` rather than the engine's lease ledger. Admission therefore saw a permanently empty card and never applied back-pressure: 16 concurrent requests cost **12.8x** the wall clock of 8, with zero rejections. Fixed in #711 (16-concurrent 472.8 s → 126.4 s median); the tail is still bad (#706) |
| Prefix reuse | — | **implemented and measured** | The one part of this document with an unambiguous end-to-end win. On qwen2.5-0.5b with a 5,122-token shared prefix, a warm request prefilled **9 tokens instead of 5,131** and TTFT fell from 137,850 ms to **2,758 ms** — a 44x reduction. Served by native-session KV rewind; the recurrent-state snapshots of #650/#672 are a separate path that native CUDA does not currently hit (`lookups=0`). Confirmed on the HTTP server too: hit rate **0.9886** across a concurrency sweep |
| Paged attention (vLLM-style) | — | **not built, and not the plan** | Verified against the tree: `block_table` does not appear in any CUDA attention kernel. `GroupQueryAttention` takes `past_key`/`past_value` as ordinary contiguous tensors and `CompressedSparseAttention` reads a flat pointer with a computed stride, so there is no input through which a page index could be passed. Making every kernel walk a block table cannot be made uniform across backends — we do not own ORT's kernels — so vAttention-style commit-on-demand under a contiguous virtual range is the chosen route instead (#656) |

### CUDA raw allocation audit (#736, 2026-08-10)

Updated through #736 (default-domain `Attention` staged K/V) on 2026-08-12.

Scope: production code in `crates/onnx-runtime-ep-cuda`,
`crates/onnx-runtime-session`, and CUDA kernel launch support. Test-only
`alloc_raw` calls under `#[cfg(test)]` are excluded. The only direct `cuMemAlloc`
entry point in this scope remains `CudaRuntime::alloc_raw`
(`crates/onnx-runtime-ep-cuda/src/runtime.rs:898`); the replaceable
`CudaDeviceAllocator`/VMM allocator path is the EP allocation authority seam, not
a kernel scratch bypass. `crates/onnx-runtime-session` has no CUDA raw allocation
call; it owns the governed workspace preparation path
(`executor/bindings.rs:16`, `executor/dispatch.rs:1538`).

| file/line | owner | byte formula | size source | lifetime class | status |
|---|---|---|---|---|---|
| `kernels/attention.rs:830-831` | legacy `Attention` Phase-2a scores + cuBLASLt workspace | `align256(batch * num_heads * sq * sk * elem_size) + WORKSPACE_BYTES` | prompt/cache dependent score matrix + static cuBLASLt ceiling | step-scoped | **governed by #753** inside one composite workspace; direct-execute compatibility/opt-out fallback remains raw |
| `kernels/csa_checkpoint.rs:126` | CSA checkpoint main carry snapshot | `main_carry_bytes.max(1)` | config/state shape | session-persistent | raw bypass |
| `kernels/csa_checkpoint.rs:127` | CSA checkpoint index carry snapshot | `index_carry_bytes.max(1)` | config/state shape | session-persistent | raw bypass |
| `kernels/csa_device_state.rs:160` | CSA device-state allocation/growth | `size` | config/state shape | session-persistent | raw bypass |
| `kernels/csa_device_state.rs:169` | CSA device-state replacement | `size.max(1)` | config/state shape | session-persistent | raw bypass |
| `kernels/elementwise.rs:699` | elementwise scalar/shape metadata upload | `metadata_bytes.len()` | op arity/rank | step-scoped | raw bypass |
| `kernels/fused_gemm.rs` | fused GEMM cuBLASLt workspace | selected heuristic `workspaceSize` (ceiling `WORKSPACE_BYTES`) | shape/dtype/algorithm dependent | session-persistent shared peak | **governed by #799** via the shared exact plan/execute helper + prepared workspace |
| `kernels/gemm.rs` | GEMM cuBLASLt workspace | selected heuristic `workspaceSize` (ceiling `WORKSPACE_BYTES`) | shape/dtype/algorithm dependent | session-persistent shared peak | **governed by #799** via the shared exact plan/execute helper + prepared workspace |
| `kernels/group_query_attention.rs` `workspace_requirement`, scores region carved at `:2774` (`WS_SCORES` region) | `GroupQueryAttention` f32 reference score buffer | `batch * num_heads * sq * present_capacity * sizeof(f32)` | prompt/cache dependent (`sq * present_capacity`) | session-persistent | **governed by #795** via `workspace_requirement` + prepared workspace, reserved only on the f32 reference path (now the score region of the #736 composite) |
| `kernels/group_query_attention.rs` `workspace_requirement` (packed Q/K/V staging region) | `GroupQueryAttention` packed QKV-projection staging | `align256(B·sq·num_heads·head_dim·elem) + 2·align256(B·sq·kv_num_heads·head_dim·elem)`, only when a packed QKV tensor is split | prompt-dependent (`sq`), elem = `dtype.byte_size()` (f32=4, f16/bf16=2) | session-persistent | **governed by #736** via `workspace_requirement` + prepared composite workspace, sized through the shared `gqa_workspace_layout`/`gqa_packed_staging_bytes` helpers; reserved only on the packed-input split route, `NONE` when Q/K/V arrive separately |
| `kernels/group_query_attention.rs` `gqa_transpose_scratch` + `gqa_workspace_layout` (`WS_Q_BNSH` / `WS_OUT_BNSH`) | `GroupQueryAttention` Q/output BNSH transpose scratch | Q: `align256(B·sq·num_heads·head_dim·elem)` when `sq>1`, packed input, or RoPE; output: the same aligned bytes only when `sq>1` | prompt/dtype/route dependent | session-persistent | **governed by #736** in the existing GQA composite; unpacked non-RoPE `sq==1` uses Q/output directly and charges `NONE` absent another region; packed `sq==1` overlays Q extraction with packed-Q staging |
| `kernels/group_query_attention.rs` `GqaWorkspace::reserve` | `GroupQueryAttention` remaining pooled slots (present K/V / metadata) | slot-specific `bytes.max(1)` | cache/batch dependent, then reused | session-persistent growable state | raw bypass; scores, packed Q/K/V staging, and BNSH transpose slots are excluded — carved from the prepared composite on governed execution |
| `kernels/index_share.rs:708` | `pkg.nxrt::IndexShare` selected-token attention workspace | aligned sum of reachable present staging, `B * q_heads * q_seq * selected_width * 4`, and optional `2 * B * sizeof(i64)` | prompt/cache dependent | session-persistent | **governed by #751** via `workspace_requirement` + prepared workspace |
| `kernels/index_transform.rs:150` | index-transform metadata | `bytes.len()` | op/rank/config | step-scoped | raw bypass |
| `kernels/indexing.rs:410` | indexing metadata | `bytes.len().max(1)` | op/rank/config | step-scoped | raw bypass |
| `kernels/matmul.rs:222` | f32 `M=1` MatMul cached-plan workspace | selected heuristic `workspaceSize` | shape/algorithm dependent | session-persistent cached plan scratch | raw bypass; separate from #799's shared non-GEMV path |
| `kernels/matmul.rs` | MatMul cuBLASLt workspace (non-GEMV routes only) | selected heuristic `workspaceSize` (ceiling `WORKSPACE_BYTES`) | shape/dtype/algorithm dependent | session-persistent shared peak | **governed by #799**; f32/fp16 `M=1` GEMV routes report `NONE` |
| `kernels/matmul_nbits.rs:3599` | `MatMulNBits` accuracy-4 decode activation quantization | `padded_k + (padded_k / block_size) * sizeof(f32)` | model config (`K`, `block_size`) | model-lifetime fixed scratch | raw bypass |
| `kernels/matmul_nbits.rs:3980` | `MatMulNBits` f32 dequantized weight fallback | `K * N * 4` | model config (`K`, `N`) | step-scoped/prefill fallback | raw bypass |
| `kernels/matmul_nbits.rs` | `MatMulNBits` f32 cuBLASLt workspace | selected heuristic `workspaceSize` (ceiling `WORKSPACE_BYTES`) | shape/dtype/algorithm dependent | session-persistent shared peak | **governed by #799**; direct GEMV/tiled routes report `NONE` |
| `kernels/matmul_nbits.rs:4425` | `MatMulNBits` RMSNorm prefill activation scratch | `M * K * sizeof(f16)` | prompt-dependent (`M`) and model config (`K`) | request/prefill-dependent | raw bypass |
| `kernels/matmul_nbits.rs:4942` | `MatMulNBits` decomposed SiLU scratch | `output.byte_size()` | output shape/prompt dependent | step-scoped | raw bypass |
| `kernels/matmul_nbits.rs:5200` | `MatMulNBits` gate/up RMSNorm prefill scratch | `M * K * sizeof(f16)` | prompt-dependent (`M`) and model config (`K`) | request/prefill-dependent | raw bypass |
| `kernels/mod_op.rs:222` | Mod metadata | `metadata_bytes.len().max(1)` | op/rank/config | step-scoped | raw bypass |
| `kernels/movement.rs:239` | movement op metadata | `bytes.len()` | op/rank/config | step-scoped | raw bypass |
| `kernels/nary.rs:264` | N-ary fp32 scratch | `n * sizeof(f32)` | output element count | step-scoped | raw bypass |
| `kernels/nary.rs:271` | N-ary metadata | `metadata_bytes.len().max(1)` | op arity/rank | step-scoped | raw bypass |
| `kernels/nonzero.rs:99` | NonZero strides metadata | `bytes.len()` | input rank | step-scoped | raw bypass |
| `kernels/normalization.rs:1563` | normalization metadata | `metadata_bytes.len()` | op/rank/config | step-scoped | raw bypass |
| `kernels/packed_varlen_attention.rs:581` | packed-varlen metadata | `bytes.max(1)` | batch/sequence metadata | step-scoped | raw bypass |
| `kernels/pad.rs:292` | Pad metadata | `metadata.len()` | rank/pads config | step-scoped | raw bypass |
| `kernels/pooling.rs:538` | pooling metadata | `bytes.len()` | rank/kernel config | step-scoped | raw bypass |
| `kernels/qlinear_matmul.rs:299` | QLinearMatMul metadata | `bytes.len()` | rank/quantization params | step-scoped | raw bypass |
| `kernels/qmoe.rs:1956` | legacy QMoE pooled scratch | slot-specific `bytes` | prompt/expert dependent | session-persistent growable scratch | raw bypass; separate from governed `BlockQuantizedMoE` path |
| `kernels/reduce.rs:711` | reduce base shape metadata | `base_bytes.len().max(1)` | rank/config | step-scoped | raw bypass |
| `kernels/reduce.rs:712` | reduce delta shape metadata | `delta_bytes.len().max(1)` | rank/config | step-scoped | raw bypass |
| `kernels/reduce.rs:720` | reduce axes metadata | `axes_bytes.len().max(1)` | rank/axes config | step-scoped | raw bypass |
| `kernels/resize.rs:259` | Resize metadata | `bytes.len()` | rank/scales/sizes config | step-scoped | raw bypass |
| `kernels/standard_attention.rs:633,777` (`std_attention_workspace_layout` / `workspace_requirement`), selected at `:1690-1766` | default-domain `Attention` f32 scores + staged dense K/V composite | scores `batch * q_heads * q_seq * total_seq * sizeof(f32)` on every route; staged key/value, when reachable, each `batch * kv_heads * present_seq * head_dim * element_bytes`, with 256-byte region alignment | prompt/cache/route dependent | step-scoped on per-call prefill/batched dense growth; session-persistent on capture-eligible single-token dense growth | **governed by #736** through one shared plan/execute layout; fixed-capacity append, no-past, and missing present-output routes add **zero** staged-K/V bytes (the always-live scores remain); direct-`execute` compatibility keeps self-owned scratch |
| `kernels/standard_attention.rs:1780,1799-1803` | default-domain `Attention` capture/eager control metadata | device valid length `sizeof(i32)`; offsets + pad limits `2 * batch.max(1) * sizeof(i64)` | batch dependent, negligible | step-scoped on eager calls; retained for capture-eligible decode | raw bypass; staged K/V and scores are excluded and governed by #736 |
| `kernels/varlen_attention.rs:531` | varlen attention metadata buffer | `bytes.max(1)` | batch/sequence metadata | step-scoped | raw bypass |
| `kernels/where_op.rs:123` | Where metadata | `metadata_bytes.len()` | broadcast rank/config | step-scoped | raw bypass |
| `runtime.rs:391` | capture-error latch | `sizeof(u32)` | static | session-persistent | raw bypass |
| `weight_paging.rs:377` | lazy weight page upload | `bytes.len()` | weight tensor storage | already-governed tensor/output allocation? no; weight paging has separate residency accounting | session-persistent until page eviction |
| `weight_paging.rs:424` | async lazy weight page upload | `bytes.len()` | weight tensor storage | weight residency/page lifetime | governed by weight residency budget, not workspace contract |
| `weight_paging.rs:479` | staged async lazy weight page upload | `len` | weight tensor storage | weight residency/page lifetime | governed by weight residency budget, not workspace contract |
| `weight_paging.rs:643` | multi-region lazy QMoE weight binding | `weight.region_bytes_len()` | weight tensor storage | weight binding/page lifetime | governed by weight residency budget, not workspace contract |

#751 governs the `IndexShare` slice — routing it through
`Kernel::workspace_requirement`, prepare-only planning, `reserve_workspace`, and
`MappedGrowthGrant` rather than a kernel-owned raw scratch pool. `IndexShare` is
a real but **low-byte** governed slice, not the largest live bypass in this
table. Before #751, its two `present_key`/`present_value` staging segments (the
largest part of the reservation) were reserved but unused on the common
three-output decode path, where present K/V is written directly to output
tensors. #751 now threads the static output arity through the shared layout
helper: the three-output path reserves only scores (and optional frontier
metadata), while the one-output path retains staging because execution uses it
(`kernels/index_share.rs:603-631`, `:867-910`). In decode `q_seq == 1` with a
sparse `selected_width`, so the live bytes are small. `IndexShare` is also a
`pkg.nxrt` custom op used only by GLM/DSA sparse-attention families, so it is not
on the hot path for dense models (#751).

The **`GroupQueryAttention` `WS_SCORES` slot** is governed by #795: the f32
reference-attention score buffer — `batch * num_heads * sq *
present_capacity * sizeof(f32)`, quadratic in `sq * present_capacity` and the
largest remaining live GQA scratch — now routes through
`Kernel::workspace_requirement` (reporting the exact bytes), prepare-only
planning, and a prepared workspace consumed by `execute_with_workspace`. Planning
and execution size the reservation through the **same** `gqa_reference_scores_bytes`
helper, so they cannot drift; a shortfall errors deterministically rather than
under-allocating. Because `sq * present_capacity` is prompt-dependent, the
executor grows the session-persistent slot transactionally through a
`MappedGrowthGrant` against the device authority, and capacity refusal surfaces
as the pre-header **429** typed path (`TierExhausted` / `CapacityUnavailable`,
#743) rather than a late 500/OOM. The compatibility/opt-out path (no
executor-prepared workspace) keeps its self-owned pooled scratch.

**Lifetime, and why it differs from Attention Phase-2a (#753).** GQA's slot
allocator grows each slot to the largest geometry seen and **retains** it across
decode/prefill steps — a standing claim held for the whole session — so
`WS_SCORES` is charged as `WorkspaceLifetime::SessionPersistent`, not the
`StepScoped` lifetime Phase-2a's per-call scratch used. A graph mixing QMoE,
IndexShare, Attention Phase-2a (step-scoped) and GQA (session-persistent) now
exercises both per-lifetime-class peaks; the two classes are reserved into
separate executor slots so neither peak stands in for the other.

**Over-reservation finding (the #751 defect class).** `WS_SCORES` is materialized
on **exactly one** GQA route: the f32 *reference* path, reached only when
`!use_fused && query dtype == f32 && !gqa_decode::supported(q_seq, head_dim)`
(i.e. f32 prefill `Sq > 1`, or f32 `head_dim > 128`). The fused-flash path, the
capture-safe f32/fp16 split-K decode kernels, and the phase-2a path **never
allocate a device score matrix** — they stream softmax through registers/shared
memory, and phase-2a owns its own scratch. Reserving `WS_SCORES`
unconditionally would therefore charge the device authority for bytes the common
fp16 decode path never touches — the same defect #751 found in IndexShare's
present K/V staging. The governed reservation threads that static signal (query
dtype + `gqa_decode::supported`) through the shared helper and reports
`WorkspaceRequirement::NONE` on every fp16/bf16 path and every f32 *decode* path,
so those charge **zero**. The only residual conservative reservation is f32
prefill that dynamically fuses on a short (`valid_seq ≤ 128`) sequence: the fused
win is a runtime measurement, so planning reserves to never under-allocate, and
that path (a fallback of a fallback) idles the buffer rather than corrupting.
Derived bytes: `B=1, H=32, Sq=1024, present_capacity=1024, f32` ≈ **128 MiB** on
the reference path; **0 bytes** on fp16 decode/prefill and f32 decode.

#799 governs the cuBLASLt GEMM follow-up. The initial hypothesis — one fixed
32 MiB session-persistent allocation — was conservative but still
over-reserved: `WORKSPACE_BYTES` is only
`CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`, while the selected algorithm exposes
its actual `workspaceSize`. On the RTX 4060 Laptop GPU regression shapes,
standard `MatMul`, `Gemm`, `MatMulNBits`, `FusedMatMulBias`, and most
`FusedGemm` cases selected **0 bytes**; the largest observed selection was
**96 bytes** (`FusedGemm`, transposed operands). Planning and execution now use
the same cuBLASLt plan helper and reserve the exact selected size, with a
deterministic error if execution ever selects more than the prepared slot.

The four newly governed kernel families share one **session-persistent peak**,
not four node reservations: on the measured shapes that is 96 bytes instead of
`4 * 32 MiB` (saving **134,217,632 bytes**, ≈128 MiB). Including the already
governed Attention Phase-2a workspace, the five-site naive accounting would be
160 MiB; measured planned cuBLASLt scratch is `32 MiB + 96 bytes`, the same
**134,217,632-byte logical-reservation saving**. Under the managed VMM's 2 MiB
physical granule, the 96-byte peak commits one granule, so the five-site physical
total is 34 MiB rather than 160 MiB — **132,120,576 bytes (126 MiB) saved**.
Attention is intentionally not folded into the persistent slot: its 32 MiB
region is live concurrently with its score matrix inside one step-scoped
composite buffer, while direct `execute` must retain the compatibility/opt-out
raw fallback. The executor therefore shares exactly where the lifetimes and
interface permit, rather than falsifying attribution to reach a nominal
one-buffer result.

#### Governed by #736: default-domain `Attention` f32 scores

The largest evidenced non-QMoE score bypass after #795/#799 is now governed:
`standard_attention.rs` (`workspace_requirement` at `:614`, consumed at `:1480`).
Every default-domain `Attention` dispatch materializes
`batch * q_heads * q_seq * total_seq * sizeof(f32)` scores in device memory
(`attention_row` stages the row scores in global memory for softcap/mask/softmax/
`probs·V`, in fp32 regardless of the f32/f16/bf16 operand dtype). At
`B=1, H=32, q_seq=total_seq=2048` that is **512 MiB** — larger than the
already-governed Phase-2a range (#753). It now routes through
`Kernel::workspace_requirement`, prepare-only planning, and a prepared workspace
consumed by `execute_with_workspace`; planning and execution size the reservation
through the **same** `std_attention_scores_bytes` helper, so they cannot drift,
and a shortfall errors deterministically rather than under-allocating. Capacity
refusal surfaces as the pre-header **429** typed path (`TierExhausted` /
`CapacityUnavailable`, #743) via `MappedGrowthGrant` (#748) against the device
authority. The compatibility/opt-out path (direct `execute`, no
executor-prepared workspace) keeps its self-owned scratch — pooled on the
capture-eligible route, per-call otherwise.

**Negative result — no route reports `NONE` (genuine use, unlike #751/#795).**
`attention_row` has **no** shared-memory/flash route: it always stages the score
matrix in global memory, so — unlike GQA's f32-reference-only `WS_SCORES` (#795)
or the fused Attention prefill path (#753) — *every* valid dispatch materializes
and reserves the fp32 scores; only unresolvable or non-float input metadata
reports `NONE` (execution then raises the precise error). This is recorded so the
next reader does not re-investigate expecting a route that charges zero.

**Both lifetime classes, modelled truthfully.** The score matrix is
route-dependent: a single-token decode (`batch == 1 && q_seq == 1`) is the only
shape that can take the capture-eligible pooled route whose buffer is retained
across steps, charged `WorkspaceLifetime::SessionPersistent` (small: linear in
`total_seq`, ~256 KiB at the example shape); every multi-token prefill (the
512-MiB-class quadratic worst case) and batched dispatch takes the per-call route
whose scratch is freed on each exit, charged `WorkspaceLifetime::StepScoped`.
Prepare-only planning reserves each class into its own executor slot (#753), so a
graph mixing them governs both. The classifier keys the split on static input
metadata (`batch`, `q_seq`), a conservative proxy for capture eligibility; the
frozen fixed-capacity decode path over-estimates `total_seq` by the padding rows,
which is safe (reserve ≥ consume) since execution re-derives the exact size
through the same helper and refuses on any shortfall.

**Over-reservation found: none.** The only sibling bypass — the staged-K/V and
capture-metadata slots (`:702,1396`) — is a genuinely different lifetime/geometry
from the scores and did not fall out of this change, so it is left raw and
tracked as the next candidate below.

#### Governed by #736: `GroupQueryAttention` QKV-projection staging

With the three quadratic score matrices governed (Phase-2a #753, GQA `WS_SCORES`
#795, default-domain `Attention` #736), the largest remaining evidenced non-QMoE
workspace bypass — the **`GroupQueryAttention` packed QKV-projection staging** —
is now governed. When Q/K/V arrive **packed** (both key/value inputs absent), the
unfused prep path splits the interleaved `[B, S, (H + 2·Hkv)·D]` query tensor
into three device staging buffers before the BSH→BNSH transpose and cache append.
Those buffers cost
`align256(B·sq·num_heads·head_dim·elem) + 2·align256(B·sq·kv_num_heads·head_dim·elem)`,
with `elem = dtype.byte_size()` (f32=4, f16/bf16=2). They now share the GQA
session-persistent composite workspace with the #795 `WS_SCORES` region: the
executor reserves one slot per lifetime class and hands the kernel one view, and
the f32 *reference prefill* route needs both the staging and the score matrix
live at once, so they occupy disjoint regions of the same reservation. The
staging regions come first (offsets depend only on shape metadata, so planning
and execution agree exactly); the capacity-dependent score matrix is last and
unpadded, so its exact geometry extends into the reserved peak without perturbing
the staging offsets, and the composite total equals exactly the score bytes when
no staging is present (preserving the #795 accounting). Planning
(`workspace_requirement`) and execution (`execute_with_workspace` → the split
branch of `run`) size every region through the **same**
`gqa_workspace_layout`/`gqa_packed_staging_bytes`/`gqa_reference_scores_bytes`
helpers, and `gqa_carve` refuses a short view deterministically rather than
under-allocating. Growth, 429 refusal (`TierExhausted`/`CapacityUnavailable`,
#743 via `MappedGrowthGrant` #748) and the self-owned compatibility/opt-out
pooled fallback behave exactly as for `WS_SCORES`.

**Lifetime — session-persistent, verified against code, not the table.** The
staging buffers are reserved from the same `GqaWorkspace` slot allocator that
grows each slot to the largest geometry seen and **retains** it across
decode/prefill steps (`group_query_attention.rs:1191`), so — like `WS_SCORES` —
the composite is charged `WorkspaceLifetime::SessionPersistent`, a single class
(the audit table's "session-persistent growable scratch" is correct here; unlike
the default-domain `Attention` scores — whose lifetime is route-dependent and
needed both classes modelled — this lifetime is **not** route-dependent, so one
class models the truth).

**Over-reservation found (the #751 defect class).** Staging is materialized on
**exactly one** family of routes — a *packed* QKV tensor split on the unfused
prep path (`packed_qkv && !fuse_prep`). Two evidenced routes charge **zero** for
it:

- **Unpacked inputs** (Q/K/V arrive as three separate tensors) never split, so
  `want_staging` is false and the staging regions are zero-length. This is the
  primary "size from use" finding: the pre-#736 pooled allocator would grow
  `WS_PACKED_Q/K/V` on any path that reached `reserve`, but the separate-input
  route writes Q/K/V straight through and touches none of them.
- **fp16/bf16 fused decode** (`q_seq == 1` with an aliased device KV cache) fuses
  the prep into the decode kernel and never splits. Because that fusion is a
  runtime pointer comparison (`fuse_prep` aliases the present cache), it is
  **not** visible at planning time, so planning conservatively reserves staging
  for every packed dispatch; on fused decode the reserved region is tiny
  (`q_seq == 1`) and left untouched rather than corrupting. Every packed
  *prefill* genuinely splits, so this is a fallback-of-a-fallback idle, not the
  common case.

The requirement therefore reports `WorkspaceRequirement::NONE` when neither
staging nor the f32 reference scores are needed (e.g. an unpacked-input fp16/bf16
dispatch), charging the device authority nothing.

**Derived bytes.** At `B=1, num_heads=32, head_dim=128` (so `q_hidden=4096`),
`kv_num_heads=8` (`k_hidden=1024`), `sq=1024`: f32 packed prefill stages
16 MiB (Q) + 2·4 MiB (K,V) = **24 MiB**, plus the ~128 MiB f32 reference score
region when that route is taken; fp16/bf16 packed prefill halves the staging to
**12 MiB** and charges **0** scores; unpacked-input and fused-decode routes charge
**0** for staging.

#### Governed by #736: `GroupQueryAttention` BNSH transpose scratch

The `WS_Q_BNSH` and `WS_OUT_BNSH` roles now join the existing GQA prepared
composite. Planning and execution both call `gqa_transpose_scratch` and
`gqa_workspace_layout`; `gqa_carve` rejects a short prepared view instead of
falling back to an invisible allocation. Direct `Kernel::execute` remains the
compatibility/opt-out path and retains self-owned pooled slots.

**Route result — partial over-reservation, not a blanket decode `NONE`.**
`Sq==1` makes the BSH↔BNSH index transform an identity, and output already
writes directly to BSH, so `WS_OUT_BNSH` is genuinely needed only for
multi-token routes. Query has two additional roles beyond transpose:

- unpacked, non-RoPE `Sq==1` reads the input Q directly and now charges zero for
  both transpose regions (and reports `WorkspaceRequirement::NONE` when staging
  and scores are also absent);
- unpacked RoPE decode needs one writable Q copy, so its query bytes are
  genuine even though the transpose itself is an identity;
- packed decode must extract Q. Fused prep writes that extraction into the Q
  region; unfused prep consumes/rotates packed-Q staging in place. The layout
  overlays those mutually-exclusive `Sq==1` roles instead of summing them;
- every `Sq>1` route needs both Q BSH→BNSH and output BNSH→BSH buffers.

Seq-major BSNH changes only KV-cache physical strides; all attention consumers
still take Q/output in BNSH internally, so it removes neither transpose.

**Lifetime — session-persistent, verified from `GqaWorkspace`.** The old
`WS_Q_BNSH`/`WS_OUT_BNSH` slots grew to the largest geometry and were retained
by the kernel across calls. The governed replacement therefore stays in the
same `WorkspaceLifetime::SessionPersistent` class as the other GQA composite
regions; capture, success, error, and cancellation release remain owned by the
executor's prepared-workspace grant.

**Derived bytes (not measured).** For `B=1, H=32, head_dim=128, sq=1024`,
Q and output each cost 16 MiB f32 or 8 MiB fp16/bf16, so transpose scratch is
**32 MiB f32** or **16 MiB fp16/bf16**. At `sq=1`, unpacked non-RoPE costs
**0**; unpacked RoPE costs 16 KiB f32 or 8 KiB fp16/bf16 for Q only. These are
derived from `dtype.byte_size()` and the code formula, not allocator telemetry.

#### Default-domain `Attention` staged K/V result

`standard_attention.rs` `WS_STAGE_KEY` / `WS_STAGE_VALUE` are now regions of the
existing governed default-domain `Attention` composite. Dense aliased growth is
genuine use: `build_kv` changes the per-head stride from `past_seq` to
`total_seq`, so it must rebuild into disjoint key/value buffers before copying
back. Planning and execution size those regions through
`std_attention_workspace_layout`; a prepared-slot shortfall errors
deterministically rather than falling back to `alloc_raw`.

**Route result — genuine on dense alias, zero on fixed append.** The prior
allocator did not allocate these slots unnecessarily; `stage_key` /
`stage_value` were already runtime-conditional. The governance risk was
introducing a new unconditional reservation. The static classifier preserves
the split: mask-driven non-causal routes whose mask width equals the physical
past-cache capacity, no-past, and absent present-output routes add **zero**
staging bytes; dense in-op-cache routes reserve only present outputs that exist.
The composite itself is not `NONE`
because this kernel always materializes fp32 scores — “zero” here is the staged
K/V component, not the independently genuine score region.

**Lifetime — both classes, verified from use.** Dense `B=1, Sq=1` growth is
capture-eligible and previously retained `WS_STAGE_KEY` / `WS_STAGE_VALUE` in
`StdAttnWorkspace`, so its governed composite is
`SessionPersistent`. Multi-token or batched dense growth allocated owned
per-call buffers and freed them on success or error, so it is `StepScoped`.
Executor-owned prepared slots now preserve those same release boundaries,
including cancellation. Direct `execute` remains the compatibility/opt-out path
with self-owned scratch.

**Derived bytes (not measured).** Each staged region is
`B · Hkv · present_seq · head_dim · element_bytes`; both K and V are live
simultaneously. At `B=1, Hkv=8, present_seq=2048, head_dim=128`, staging is
**16 MiB f32** or **8 MiB fp16/bf16** total for K+V, plus at most 255 bytes of
alignment before each region. Fixed-capacity append is **0 staged bytes**.

#### Next slice (chosen from this table): `MatMulNBits` f32 dequantized weights

The next evidenced non-QMoE bypass is the general/grouped
`MatMulNBits` fallback at `matmul_nbits.rs:3986`: it allocates
`K · N · sizeof(f32)` bytes, dequantizes the packed weight, uses it for one
cuBLAS GEMM, synchronizes, and frees it. This is a model-dimension-sized
step-scoped allocation already adjacent to the #799 governed cuBLAS workspace
contract, and it is route-dependent: direct GEMV and tuned accuracy-4 tiled
routes return before the allocation. Derived examples are **64 MiB** at
`K=N=4096` and **172 MiB** at `K=4096, N=11008`.

### KV layout and residency

This is the single authoritative home for KV layout, VMM geometry, residency,
and the measurements behind the design. Investigation notes link here rather
than copying conclusions, following the anti-staleness rule established in
#781.

**Status, as of 2026-08-12:**

- **Implemented:** head-major BNSH remains the default. PR #782 landed the
  seq-major BSNH fused fp16 single-token decode append/read pair, and #792
  replaced its layout enum with a symbolic stride descriptor and static
  per-layout NVRTC specialization. Flash prefill now uses the same descriptor
  for cache preparation and reads, enabling full seq-major prefill + fp16
  decode generations when the fused-flash shape gate applies. Unsupported
  readers and writers still fail rather than silently mis-index. The engine KV
  *growth byte geometry* is now layout-faithful (bsnh-fixed-stride): under
  in-place VMM growth a seq-major buffer grows on its capacity-independent
  `kv_heads × head_dim` stride and moves **0** KV bytes, measured end to end on a
  real model (see "KV layout and residency" below). The BNSH binding *metadata*
  shape is retained deliberately for both layouts — the GQA node reads
  `present_capacity` from BNSH axis 2 regardless of the `kv_layout` attribute and
  `DeviceIoBinding` requires matching axis order (#801) — so seq-major lives in
  the byte geometry, not a permuted shape. Growth invalidation is now
  **conditional** (#811): the engine no longer invalidates the captured graph on
  every KV bucket growth. Seq-major fixed-stride growth **keeps** the captured
  graph — device pointers and physical shapes are unchanged, batch-0 addressing
  is capacity-independent, and the mask is fully committed — so its
  growth-attributable invalidations fall **3 → 0**. Head-major still invalidates,
  correctly, because it re-strides every head stripe and moves 688,576 bytes on
  growth. Both are proven by the process-isolated parity test
  (`mobius_seqmajor_growth_parity_native_cuda.rs`), including a negative oracle
  that fails if a graph is kept across a growth that genuinely moved bytes.
- **Measured, not implemented:** token-major across all layers. Its residency
  floor and 192 KiB-stride read cost were measured in #787, but no production
  kernel or binding layout uses it.
- **Proposed, in progress:** make layout selection a per-EP, per-platform
  capability (#783). Binding views, layout negotiation, prefix multi-map,
  staggering, and commit-ahead policy described below remain design intent,
  not present-tense capabilities.

#### Governing rule: layout controls residency

`cuMemMap` maps a whole granule-aligned **window** of virtual address space onto
a whole physical granule. Partial mapping does not exist:

```
VA: |<------ granule ------>|<------ granule ------>|<------ granule ------>|
       window 0                 window 1                 window 2
       mapped or not            mapped or not            mapped or not

committed bytes = granule × (number of windows containing at least one live byte)
```

The allocator cannot compact live bytes that the tensor layout scattered across
VA windows. This is the same unsurprising behavior as a sparse file: writing one
byte at offsets 0, 1 MiB, and 2 MiB can consume three filesystem blocks for
three live bytes. Nothing is malfunctioning; the placement selected three
allocation units.

In the diagrams below, `X` is live KV, `.` is reserved but not live, and each
bracketed span is one physical-mapping window.

**Head-major BNSH scatters one live prefix per head stripe:**

```
one layer-side buffer, four heads shown:

 head 0              head 1              head 2              head 3
|X.................|X.................|X.................|X.................|
[==== window 0 ====][==== window 1 ====][==== window 2 ====][==== window 3 ====]
      COMMIT               COMMIT               COMMIT               COMMIT

Four tiny live fragments open four complete physical windows.
```

Each head owns its full `max_seq × head_dim` stripe. Across the model, the
near-empty floor is therefore `layers × 2 × kv_heads` windows — **but only when
the per-head stride is a fixed full context.** The engine grows a packed bucket
instead, so this figure does not describe its head-major path; see the floor
table below and #841.

**Seq-major BSNH is dense within each buffer, but K and V remain separate for
every layer:**

```
one layer-side buffer:

|XXXXXXXX........................................................|
[==== window 0 ====][==== window 1 ====][==== window 2 ====]
      COMMIT              unmapped              unmapped

all layer-side buffers:

layer 0 K: [X...][ ][ ]  COMMIT
layer 0 V: [X...][ ][ ]  COMMIT
layer 1 K: [X...][ ][ ]  COMMIT
...
layer 47 V:[X...][ ][ ]  COMMIT        (96 independent buffers on qwen14b)
```

The live prefix is dense, but every independent reservation has its own window
zero. The floor is `layers × 2` windows.

**Token-major across all layers makes the complete sequence one dense run:**

```
one reservation, with every layer's K and V interleaved by token:

| token 0: all layers | token 1: all layers | token 2 ...               |
|XXXXXXXXXXXXXXXXXXXXXX..................................................|
[========== window 0 ==========][========== window 1 ==========]
             COMMIT                           unmapped
```

The per-layer bindings become views into that reservation. The floor is one
window per sequence. This was measured, but that binding model is not built
(#787).

**A paged-attention block is fully allocated regardless of its internal
layout:**

```
pre-committed block pool:
 [ blk 0 ][ blk 1 ][ blk 2 ][ blk 3 ] ...
     ^        ^                 ^
   seq A    seq B             seq A       (selected through a block table)

inside any allocated block, the same bytes are occupied:
 BNHS [head][dim][token]   BNSH [head][token][dim]   BSNH [token][head][dim]
```

Paging therefore decouples physical residency from internal layout: layout is a
compute-side choice because the allocation unit is already fully used. Flat VA
plus VMM couples them: a window is only as full as the layout makes it (#783).

#### Measured floor progression

For qwen14b (`48` layers, K+V, `8` KV heads, head dimension `128`, fp16) on the
measured 2 MiB CUDA granule:

| Layout | Floor unit | Near-empty floor | Evidence/status |
|---|---:|---:|---|
| BNSH head-major, **fixed full-context stride** | `layers × 2 × kv_heads = 768` | ~1.5 GiB | Geometry measured in #772/#776/#787 — but **the engine never instantiates this stride**; see below |
| BNSH head-major, **as the engine actually runs it** | `layers × 2 = 96` | ~192 MiB | **Measured on qwen14b** (#841): head-major grows a *packed* bucket, so it already sits on the dense floor |
| BSNH seq-major | `layers × 2 = 96` | ~192 MiB | Driver-level residency (`vmm_kv_layout_residency_gpu`); **byte-identical to head-major end to end** (#841) |
| token-major across all layers | `1` per sequence | ~2 MiB | **768× measured reduction**, #787; not implemented |

> **The 768-granule head-major figure is the wrong baseline for the engine, and
> the `kv_heads×` ("8×") saving seq-major was pursued for does not exist on the
> path the engine actually runs.** Measured end to end on qwen14b in #841, with
> on-demand commit *active* (`growth_events ≥ 1`, i.e. not the pinned-knob
> artifact of #834): head-major and seq-major commit **byte-identical** physical
> KV (402,653,184 B at both bucket 512 and 1024), with byte-identical tokens.
>
> The mechanism is in `native_decode/cuda.rs`: head-major sets
> `new_shape[2] = new_capacity` on growth, so its per-head stride is *the current
> bucket*, not a fixed full context. It therefore re-packs **dense** and lands on
> the same `layers × 2` granules as seq-major. The 768-granule floor is a property
> of a *fixed full-context-stride* head-major layout — each head stripe
> `max_len × head_dim × elem` apart, so a short prefix scatters one granule per
> head — and the engine deliberately never builds that. Head-major runs at
> **0.25× of the 768 figure**, not 4× above the seq-major floor.
>
> **Seq-major's real, measured advantage is growth cost, not committed bytes:**
> on the 512→1024 step head-major moved `kv_bytes_moved = 100,667,392 B` and
> invalidated its captured graph, while seq-major moved **0 bytes** and kept the
> capture (`growth_keeps`). That is what the layout chain (#782/#792/#794/#797/
> #801/#812/#827) actually bought, and it is worth having — it is simply not a
> residency saving.
>
> **Lesson.** The `layers × 2 × kv_heads` floor was derived from layout geometry
> and then compared against the engine without first checking which stride the
> engine builds. `cuda.rs` had said "head-major grows a *packed* bucket whose
> per-head stride is `new_capacity` (the bucket, not a fixed full context)" since
> #827 — the arithmetic was right, the baseline was not. Before quoting a floor,
> confirm the code instantiates the geometry that floor assumes.

> **These floors are properties of the layout geometry, not of what the runtime
> currently commits on the end-to-end engine path.** Measured in #794 on both
> qwen2.5-0.5b and qwen14b, head-major and seq-major commit **identical**
> physical bytes (100,663,296 B and 402,653,184 B respectively), because the
> native CUDA bindings allocated bucket-sized packed shapes and committed flat
> bucket ranges without consuming the KV layout metadata — so seq-major changed
> **kernel indexing only**, not reservation or commit geometry. #787 reached the
> same conclusion from the other direction: the read path is free, and the cost
> sits on the binding layer.
>
> **Update (kv-binding-views).** The layout-aware commit *geometry* is now built
> and measured directly at the driver level. A `KvCommitLayout`-driven commit
> path (`crates/onnx-genai-engine/src/native_decode/kv_commit.rs`, unit-tested)
> computes, per binding, the live-prefix commit ranges the layout implies on a
> **fixed full-context stride**: seq-major is one dense run
> `0..ceil(live_bytes/granule)`; head-major stays one live fragment per head
> stripe. The GPU test `vmm_kv_layout_residency_gpu` (in `onnx-runtime-cuda-memory`)
> maps those ranges on the real 2 MiB granule and confirms, deterministically:
>
> | Model | Per-binding head-major | Per-binding seq-major | Ratio | Model-wide head-major | Model-wide seq-major |
> |---|---:|---:|---:|---:|---:|
> | qwen2.5-0.5b | 4 MiB (2 granules) | 2 MiB (1 granule) | **2× = kv_heads** | 192 MiB (96 granules) | 96 MiB (48 granules) |
> | qwen14b | 16 MiB (8 granules) | 2 MiB (1 granule) | **8× = kv_heads** | 1536 MiB (768 granules) | 192 MiB (96 granules) |
>
> **This ratio is real but does not transfer to the engine.** It is measured on a
> *fixed full-context stride*, as the paragraph above states. The engine's
> head-major path grows a **packed** bucket instead (`new_shape[2] =
> new_capacity`), so end to end the two layouts commit byte-identical physical KV
> (#841). See the floor table earlier in this section.
>
> The same test also grows a seq-major fixed-stride binding **under a captured
> replay** (granule 0 → 3 at a stable VA) and observes **0 re-captures** — the
> stable-stride win #782 could not demonstrate end to end, shown here on hardware.
>
> **Update (bsnh-fixed-stride).** The engine growth *byte geometry* is now
> layout-faithful, and a real seq-major generation runs end to end. Two findings
> refine the earlier "structural gap" note precisely:
>
> 1. **The KV binding metadata shape must stay BNSH for both layouts.** The CUDA
>    GQA node validates `past_key`/`past_value` as `[batch, kv_heads, seq,
>    head_dim]` and reads `present_capacity` from axis 2 *regardless of the
>    `kv_layout` attribute* (only the kernel's stride arithmetic re-specializes),
>    and `DeviceIoBinding` requires the logical and physical shapes to share axis
>    order. So `persistent_state_shapes` is *correctly* BNSH for seq-major too —
>    a permuted binding shape cannot express seq-major. Seq-major is realized in
>    the **growth/commit byte geometry** (`kv_growth_byte_layout`,
>    `apply_vmm_growth`, `build_grown_buffers`), which now maps the BNSH shape to
>    its BSNH byte layout `[batch, capacity, kv_heads, head_dim]` (grow axis 1)
>    and drives the copy/zero primitives over that. Head-major is byte-identical
>    to before (grow axis 2, one stripe per head).
>
> 2. **The "0 bytes moved on growth" fixed-stride property holds only on the
>    in-place VMM growth path**, not on the default bucket-*reallocation* path. A
>    reallocation fills a fresh buffer, so it copies the live prefix either way;
>    seq-major's win there is one contiguous copy instead of `kv_heads` stripes,
>    not a smaller byte count. Under `ONNX_GENAI_CUDA_VMM=1` (`commits_on_demand`,
>    growth maps granules onto the same base VA) the capacity-independent
>    `kv_heads × head_dim` per-token stride keeps the prefix in place.
>
> **Measured end to end** on `qwen2.5-0.5b-q4_0-mobius` (stock BNSH export vs a
> copy whose 24 GQA nodes carry `kv_layout=1`), forcing growth with
> `ONNX_GENAI_KV_MIN_BUCKET=8` and generating 48 greedy tokens (3 growth events,
> 8→16→32→64), test
> `crates/onnx-genai-engine/tests/mobius_seqmajor_growth_parity_native_cuda.rs`:
>
> | Growth path | Capture | Tokens | Physical committed bytes (head = seq) | d2d KV copy bytes head | d2d KV copy bytes seq |
> |---|---|---|---:|---:|---:|
> | in-place VMM (default since #798) | OFF | byte-identical | 100,663,296 | 688,576 | **0** |
> | in-place VMM (default since #798) | ON | byte-identical | 100,663,296 | 688,576 | **0** |
> | legacy realloc (`ONNX_GENAI_LEGACY_ALLOCATOR=1`) | OFF | byte-identical | 786,432 | 688,576 | 688,576 |
>
> The capture-ON default-VMM row was measured with **every layout/configuration
> in its own process**, interleaving head-major then seq-major in the same
> session. Since #811 made growth invalidation **conditional**, the two layouts
> no longer share a row:
>
> | Layout | Growth events | Captures | Replays | Invalidations | Growth keeps |
> |---|---:|---:|---:|---:|---:|
> | head-major | 3 | 4 | 39 | 4 | 0 |
> | seq-major | 3 | 1 | 45 | 1 | 3 |
>
> A same-generation, no-growth control (`min_bucket=64`) measured `1 capture,
> 45 replays, 1 invalidation` for each layout. Head-major's three bucket growths
> therefore add exactly three invalidations and three captures (it moves 688,576
> bytes per growth and must re-capture). Seq-major's forced-growth accounting now
> collapses **onto that no-growth control** — `1 capture, 45 replays, 1
> invalidation` — because #811 keeps the captured graph across every growth and
> records one `growth_keep` per growth (3 total). This is attributable in code as
> well as by counter delta: after `apply_vmm_growth`,
> `DecodeCudaState::ensure_capacity` compares a `binding_growth_signature`
> (device pointer + physical shape per binding) before and after the commit; on
> the seq-major fixed-stride path the signature is unchanged, so it calls
> `record_growth_decision(true, …)` with the named reason ("seq-major fixed
> full-context stride: device pointers and physical shapes unchanged, batch-0
> addressing capacity-independent, mask fully committed") and **keeps** the graph;
> head-major (and any signature change) invalidates. #797's driver-level `0
> re-captures` result is thus now **demonstrated end to end** through the engine's
> bucket-growth path. A negative oracle in the same test fails if a graph is ever
> kept across a growth that genuinely moved bytes.
>
> The 48-token stream is byte-identical head-major vs seq-major in every row —
> the first real seq-major generation that exercises *growth*. (#794's 32-token
> run never crossed the 256-token bucket, so it validated kernel indexing only.)
> Since #798 made managed no-spill VMM the default, the **default** growth path is
> in-place VMM, so seq-major now grows moving **0 KV bytes by default**; the
> legacy reallocation allocator (which copies the live prefix to a fresh buffer
> either way) is reachable only via the #755 opt-out. One residency caveat
> remains orthogonal to this shape/stride change:
>
> * **The dense-prefix commit is now wired into the live seq-major path
>   (dense-prefix-commit).** `DecodeCudaState::seq_major_kv_commit_requests`
>   consumes `kv_commit.rs::live_prefix_ranges` directly instead of duplicating
>   the byte arithmetic, so the engine's committed seq-major geometry and the
>   driver-level residency measurement (`vmm_kv_layout_residency_gpu`) are
>   single-sourced and cannot drift. Head-major stays byte-identical on its flat
>   bucket commit (`vmm_growth_requests`): on a *growing packed bucket* its
>   per-head stride is the bucket, so the `kv_heads` live-prefix fragments tile
>   the same contiguous `[0, bucket_bytes)` run — head-major's dense ranges
>   **equal** its bucket ranges, and it is left alone.
> * **Committed physical bytes remain equal head vs seq on this harness**
>   (measured `dense-prefix-commit`, process-isolated, all four vmm-inplace
>   configurations: 100,663,296 both; 786,432 both under legacy realloc), and
>   this is now understood to be the **granule floor, not an un-wired commit**.
>   On qwen2.5-0.5b (`kv_heads = 2`, `head_dim = 64`, fp16, hard-max context
>   512) a *single binding's entire reservation* is `512 × 2 × 64 × 2 =
>   131,072 B = 128 KiB`, far below the 2 MiB granule, so **both** layouts pin
>   at exactly one granule per binding — `48 bindings × 2 MiB = 100,663,296 B`,
>   which **is** the `layers × 2 = 48`-granule floor. Seq-major already sits on
>   that floor; head-major coincides because its 64-token bucket is also
>   sub-granule. The `kv_heads×` separation only materializes once a single
>   head's live stripe crosses a granule — `capacity × head_dim × elem ≥ 2 MiB`,
>   i.e. ≈8,192 tokens at head_dim 128/fp16 (the measured head-major crossover,
>   #776) — which a 512-max, 48-token run never approaches. So the dense-prefix
>   commit is correct and floor-faithful, but the driver-measured `kv_heads×`
>   residency win (`vmm_kv_layout_residency_gpu`, qwen14b at 32,768-token stride)
>   is **not** reproducible end-to-end on the qwen2.5-0.5b harness: its geometry
>   never leaves a single granule. Demonstrating it end-to-end needs a model/
>   context whose per-head stripe exceeds a granule (large `head_dim`/context),
>   not a change to this commit path.
> * **The qwen14b end-to-end measurement is now in (`qwen14b-floor`), and it
>   is a precise negative result** — the model #827 identified as the *only* one
>   whose per-head stripe reaches the granule (`8192 × 128 × 2 = 2 MiB` exactly,
>   at full context). Test
>   `crates/onnx-genai-engine/tests/qwen14b_kv_floor_sweep_native_cuda.rs` sweeps
>   the reserved KV capacity across the 2 MiB-per-stripe threshold
>   (`ONNX_GENAI_KV_MIN_BUCKET ∈ {1024, 2048, 4096, 8192}`), in-place VMM,
>   head-major vs a `kv_layout=1` copy, process-isolated, same session. Measured
>   committed **physical** KV bytes:
>
>   | Reserved capacity | Head-major committed | Seq-major committed | Ratio | Tokens |
>   |---:|---:|---:|---:|:--|
>   | 1024 | 402,653,184 B (192 gr) | 402,653,184 B (192 gr) | 1.00× | byte-identical |
>   | 2048 | 603,979,776 B (288 gr) | 603,979,776 B (288 gr) | 1.00× | byte-identical |
>   | 4096 | 1,006,632,960 B (480 gr) | 1,006,632,960 B (480 gr) | 1.00× | byte-identical |
>   | 8192 | 1,811,939,328 B (864 gr) | 1,811,939,328 B (864 gr) | 1.00× | byte-identical |
>
>   The `kv_heads×` floor **does not separate at equal committed length**: both
>   layouts commit byte-identically, ramping together.
>
>   **The mechanism is a property of this harness, not of the engine — and the
>   original write-up of this table got that wrong.** The sweep sets
>   `ONNX_GENAI_KV_MIN_BUCKET = capacity` at every point, and
>   `onnx_genai_kv::kv_capacity_bucket(len, hard_max)` is
>   `len.next_power_of_two().max(min_bucket).min(hard_max)`. Pinning
>   `min_bucket == capacity` *forces* `initial_bucket_len == capacity`, hence
>   `committed_len == capacity`. Concluding from that same number that "the engine
>   commits the full KV bucket eagerly" is circular: the knob produced the
>   observation. Re-running the identical child at the engine's **default**
>   `ONNX_GENAI_KV_MIN_BUCKET=256` yields, for seq-major, `committed_len = 256`
>   with `max_len = 8192` — a live-prefix commit far short of the bucket. **The
>   engine does commit on demand.**
>
>   Seq-major's fixed full-context stride is confirmed active (`max_len == 8192` at
>   every capacity vs head-major's `max_len == capacity`) and the token streams
>   match, so the seq-major kernel and layout resolution are correct; a fixed
>   stride makes *growth* free (#797's 0 bytes moved) and keeps the captured graph
>   (#811/#812). The head-major committed count is close to the predicted
>   `layers × 2 × kv_heads = 768` granules at full context (measured 864, the extra
>   ~96 being non-granule-aligned reservation bases spanning one extra granule per
>   binding); seq-major sits on the *same* number only because this harness pinned
>   it to the same committed length.
>
>   **Separation requires two conditions at once, and every measurement so far has
>   violated one of them:**
>   1. head-major capacity large enough that its per-head stripe reaches a granule
>      (`capacity × head_dim × 2 ≥ 2 MiB`, i.e. capacity ≈ 8192), **and**
>   2. the seq-major committed dense prefix left free to stay small — i.e.
>      `ONNX_GENAI_KV_MIN_BUCKET` *not* pinned to the capacity.
>
>   The sweep above violates (2) by construction. At the default bucket 256 both
>   layouts commit 192 granules because (1) is violated instead — a 256-token head
>   stripe is 64 KiB, sub-granule. The regime where the 8× can appear is small
>   bucket *and* long live prefix, and is measured separately. The test asserts
>   the equality as a #812-style guard, with its `committed_len` assertions
>   documented as *harness preconditions* rather than engine findings: if seq-major
>   ever commits less, the guard fires and this table must be updated — that would
>   be the win.
>
>   **Lesson (the third time this document has had to be corrected):** before
>   attributing a measured number to the system, check that no knob you set is
>   itself producing that number. See the "instrument itself" failure mode below.
> * **A fourth place the layout lives — the engine-side non-kernel KV
>   consumers — is now gated (#812, `seqmajor-physical-shape`).** Beyond kernel
>   indexing, commit geometry, and growth byte geometry, the device
>   present-KV host mirror (`DecodeCudaState::read_present_kv`) and the paged
>   prefix-reuse seed (`DecodeCudaState::seed_prefix`) stride the padded buffer
>   with hard-coded head-major (BNSH) arithmetic
>   (`capacity_head_stride = physical_shape[2] × head_dim × elem`, per-head
>   fragments). Under a seq-major (BSNH) physical buffer those offsets address
>   the wrong bytes — heads are interleaved per token, not laid out as capacity
>   stripes — so a host mirror or a shared-prefix seed would silently return or
>   write mis-indexed KV. Both now **error under seq-major** rather than
>   mis-mapping, matching the kernel-level "unsupported readers/writers fail
>   rather than silently mis-index" gate. This makes the working seq-major
>   surface a precise **condition**: seq-major is supported on the pure decode
>   path where our own GQA kernel is the sole consumer of the device KV; the
>   moment a `present_*`-reading consumer (host mirror / paged prefix reuse) is
>   engaged, the runtime refuses. Prefix sharing (#777/#809) therefore does not
>   > yet "fall out" for free under seq-major: its device seed is a head-major
>   consumer, so realizing it under seq-major requires teaching that seed the
>   BSNH byte geometry. The dense-prefix commit above does **not** address this
>   (it changes only the commit range geometry, not the seed/host-mirror stride
>   arithmetic), so the seq-major refusal on `present_*`-reading consumers still
>   stands and the BSNH seed remains a separate follow-on.
>
> Wall-clock is intentionally not reported: deterministic counters answer the
> invalidation question, while this shared box has shown large timing variance.

The small-model measurement makes the waste concrete: qwen2.5-0.5b committed
**96 head stripes × 2 MiB = 192 MiB to hold about 12 KiB** of live KV (#772).
The token-major probe repeated the comparison on qwen14b: the identical
196,608-byte one-token payload committed **1,536 MiB head-major versus 2 MiB
token-major**, the measured 768× reduction (#787).

This also preserves an important historical correction. The earlier
"tokens per granule" model was **wrong for head-major** and was publicly
retracted in #776: independently strided heads cannot pool their sub-granule
live bytes. The same model is **exactly right for token-major**, where all KV for
a token is contiguous in one reservation (#787). The granule, model, and VMM
are unchanged; only layout changes. That is the clearest demonstration that
layout is the governing variable.

#### Crossover against bucket growth

The crossover is the context length below which a fixed full-context stride
commits more physical memory than growing a packed bucket. For head-major:

```
crossover_head_major = granule / (head_dim × sizeof(dtype))
```

`layers` and `kv_heads` cancel, so this crossover is **model-size independent**.
Hardware probes spanning a 480× model-size range measured about **8,192 tokens**
at head dimension 128/fp16 and **16,384 tokens** at head dimension 64/fp16
(#776).

Seq-major changes the unit:

```
crossover_seq_major = granule / (kv_heads × head_dim × sizeof(dtype))
```

It is no longer model-independent: it scales with `kv_heads`. With the measured
2 MiB granule it is **1,024 tokens on qwen14b** and **8,192 on
qwen2.5-0.5b** (#778). Token-major has **no crossover**: packed growth and the
fixed flat view occupy the same dense byte stream, rounded once per sequence
(#787).

A live-length-bounded read leaves the reserved tail physically uncommitted,
including under captured graph replay; reading one byte into an uncommitted
tail faults. Thus reservation alone does not force commitment (#772). Layout
decides which windows the bounded live prefix touches, while an over-broad read
can additionally force the tail.

#### Granularity is the coupling, and it is platform-specific

On the development GPU,
`CU_MEM_ALLOC_GRANULARITY_MINIMUM == CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ==
2 MiB`; no finer device granule is available (#776). If the granule were one
byte, layout would be irrelevant to residency. On this CUDA device, the floor
must therefore be fixed by layout rather than by tuning the allocator.

Reported minimum mapping units span roughly 500× across relevant APIs (#783):

| Platform/API | Typical minimum unit | Consequence |
|---|---:|---|
| CUDA VMM | ~2 MiB, measured here | Layout severity described above |
| HIP VMM | commonly ~2 MiB; must be queried | Re-derive for #731; do not inherit CUDA's answer |
| Level Zero / Intel device-local | ~64 KiB | Much smaller layout penalty |
| Vulkan sparse buffers | ~64 KiB | Much smaller layout penalty |
| CPU `mmap` | 4 KiB | Usually negligible |
| Metal / Apple Silicon | no direct equivalent; unified-memory regime | Re-derive residency semantics (#608) |

At 64 KiB, the head-major head-dimension-128/fp16 crossover is about **256
tokens**, so realistic contexts generally exceed it (#783). Intel's reported
BNHS preference is therefore coherent with Level Zero's much finer pages: a
layout that is severe on this CUDA device can be reasonable on Intel.

**Design intent:** granularity must be queried as a device property and feed
memory-strategy inference (#735), not be hardcoded. Layout follows from
`(platform granularity, EP capability, model geometry)`. HIP support (#731)
must measure and re-derive rather than copy the CUDA choice.

#### Read and remap costs

**Measured read cost is not the obstacle.** Seq-major/head-major decode-read
ratios were **0.80-1.02 (within 2%)** for sequence lengths 512 through 32K
(#778). The token-major probe then held bytes and occupancy constant while
raising the token stride to 192 KiB: at a 6 GiB working set the bandwidth ratio
was **1.000**, about **207 GB/s (~80% of peak)** (#787). On this device the reads
are DRAM-bandwidth-bound independently of stride; 2 MiB-backed device memory
keeps the 192 KiB stride within TLB reach.

**Measured remap cost is a burst, not a per-token tax.** A granule supplies
about 8,192 tokens of headroom per head-major stripe or 1,024 per seq-major
buffer on qwen14b. Amortizing remaps across those tokens yields about 0.1% of a
step, but hides the user-visible event: all aligned buffers cross together on
one token. The measured qwen14b boundary burst was **49.5 ms head-major versus
4.5 ms seq-major** (#778).

Lockstep is an alignment artifact, not an inherent VMM requirement.
**Proposed, not built:** stagger buffer phases or commit ahead of the write
frontier. Commit-ahead is graph-safe because VMM preserves the virtual address;
bucket growth changes the address/stride and forces graph re-capture (#778).

#### Paged attention is the alternative architecture

Paged attention avoids layout-dependent residency by making every kernel follow
a block table. Its per-sequence allocation quantum is not merely one small
per-layer block: a vLLM-style 16-token growth step needs a block in every layer.
For qwen14b:

```
196,608 bytes/token × 16 tokens = 3 MiB per sequence
```

The equivalent VMM calculation is:

| Model | `granule / KV bytes per token` | 16-token paged quantum | Token-major VMM quantum |
|---|---:|---:|---:|
| qwen14b | 10.67 tokens | ~3 MiB | 2 MiB |
| qwen2.5-0.5b | 170.67 tokens | ~0.19 MiB | 2 MiB |

These arithmetic comparisons were verified in #787. Token-major plus VMM
matches or slightly beats the paged quantum for qwen14b without adding a block
table to every kernel, but is honestly coarser on the small model. It also keeps
physical memory returnable. The compared vLLM policy pre-commits a pool (about
90% of VRAM by default) for the process lifetime rather than returning unused
capacity to the system (#783).

#### Quantization is a split result

Because VMM's quantum is byte-denominated, token-major density automatically
scales with quantization on qwen14b: **10.7, 21.3, and 42.7 tokens per granule**
for fp16, fp8, and int4, with no tuning (#787). However, the per-sequence
**byte** floor stays 2 MiB at every dtype, while the waste in a token-denominated
16-token paged design shrinks with bytes per token. Under aggressive
quantization, paging can therefore have the smaller partial-allocation waste.

The durable design conclusion is narrower:

- Head-major makes KV quantization and residency **antagonistic**: halving bytes
  per token doubles the crossover in tokens (#776/#787).
- Token-major makes them **independent**: there is no crossover; the one-window
  floor remains one window at any dtype (#787).

#### Prefix sharing is the concurrency-scaled prize

The floor is a constant that amortizes with context length. Shared-prefix
savings scale with concurrent requests, the axis multi-request serving in #750
cares about.

**Shareability is arithmetic, not a layout (#777).** Sharing maps whole
granules, so a prefix is shareable exactly when each contiguous fragment is at
least one granule:

```
fragment_bytes = prefix_len × (contiguous bytes per fragment in that layout)
shareable      = fragment_bytes ≥ granule
multi_map_ops  = fragments × floor(fragment_bytes / granule)
```

Layout sets `fragment_bytes` (and the cost) but **not** the possibility. The two
genuine requirements are a **VMM-backed** KV buffer and `fragment_bytes ≥
granule` on the platform, with the granule **queried** from the driver rather
than assumed (#822). This is a tested predicate,
`onnx_runtime_memory_governor::shareability::evaluate_prefix_shareability`, that
replaces any "is this seq-major" check; a KV path refuses with a reason when the
arithmetic says a configuration is not shareable rather than making N private
copies. Layout only sets fragment size and cost:

- Head-major: a prefix is `layers × 2 × kv_heads` scattered fragments, each
  `head_dim × dtype` per token. On qwen14b that is *not* shareable at a 2 MiB
  granule below an **8,192-token** prefix (`granule / (head_dim × dtype)`), but
  *is* shareable at or above it, and shareable at any realistic prefix on a
  finer-granule EP (~64 KiB Level Zero/Vulkan, 4 KiB CPU `mmap`).
- Seq-major: `layers × 2` contiguous layer-side ranges (`kv_heads × head_dim`
  per token) — shareable at 2 MiB from a ~2,048-token prefix on qwen14b, at
  `floor` = 1 granule per fragment, i.e. 96 multi-map ops.
- Token-major: **one contiguous range covering every layer** — shareable at the
  smallest prefix, one physical-handle multi-map per sequence (cheapest cost).

The full shareability grid across layout × granule × prefix-length for both
qwen14b and qwen2.5-0.5b, and the derivation, live in
[`PREFIX_SHARE_INVESTIGATION.md`](./PREFIX_SHARE_INVESTIGATION.md). At this
device's measured 2 MiB granule (#776), token-major remains the cheapest and
seq-major the practical middle, but "head-major cannot share" is only true
*below the 8,192-token threshold at 2 MiB* — corrected from the earlier absolute
phrasing.

For a 2,000-token shared prompt and eight concurrent qwen14b requests, avoiding
the seven duplicate copies saves about **2.56 GiB** (#777/#787). This uses the
multi-map primitive proven in #727, but detection, lifetime, read-only/COW
enforcement, and 1:N handle bookkeeping are **not yet implemented**.

The #777 isolating GPU probe has now cleared all five primitive questions with
no kill finding — N-way multi-map under captured-graph replay, charge-once in the
real #740 ledger, non-sticky/non-corrupting write protection, coexistence with
the #759 dummy page, and a one-time ~5.5 ms pooled copy-on-write at the boundary.
Measured answers, the concurrency/saving and capacity tables, and the design for
an explicit pinned-prefix API (the smallest next increment) are in
[`PREFIX_SHARE_INVESTIGATION.md`](./PREFIX_SHARE_INVESTIGATION.md).

**First production consumer landed (#777).** The prefix-sharing primitive
(`create_shared_prefix`/`commit_shared_prefix`, #803) previously had no live
caller — only its definition and GPU tests. It is now reachable from production
code through an allocator-agnostic seam on the `DeviceAllocator` trait
(`create_shared_prefix` / `incremental_owned_bytes_for_shared_prefix` /
`commit_shared_prefix`, returning an opaque `dyn SharedDevicePrefix`), so a
caller holding only `dyn DeviceAllocator` can pin a token prefix once and have
subsequent sequences map it. Non-VMM allocators keep the default impls, which
refuse (`InvalidRequest`) rather than mis-map. The **seq-major** fused fp16 GQA
decode kernel is the first consumer: a GPU parity test
(`crates/onnx-runtime-ep-cuda/tests/gqa_shared_prefix_parity_gpu.rs`) drives the
real kernel over shared-prefix VMM KV and proves two sequences sharing one pinned
seq-major prefix (`layers × 2` contiguous ranges) produce **byte-identical**
output to two independent sequences. Measured at KV_HEADS=8, HEAD_DIM=128, f16,
1024-token prefix + 1024-token private tail per sequence: independent = 8
granules (16,777,216 B), shared = 6 granules (12,582,912 B); the prefix is
charged **once** (`incremental_owned_bytes_for_shared_prefix` = 0) and the second
sharer's admission is **only its private bytes** (4,194,304 B = its two private
tails), so sharing removes `(C−1)×(K_prefix+V_prefix)` = 2 granules.

**What remains structural.** The *engine generation loop* cannot yet call this
seam automatically: `persistent_state_shapes` in
`native_decode/cuda.rs` builds a hard-coded BNSH physical KV shape and no model
declares a seq-major end-to-end fixed-stride physical shape (#794 showed
seq-major changed only kernel indexing, not commit geometry). A BNSH/seq-major
fixed-stride physical-shape build is therefore the prerequisite for auto engine
use. Hash-based automatic detection, token-major (one multi-map per sequence),
and copy-on-write at divergence remain the later increments named below. The
delivered consumer is explicit (a caller declares the shared prefix) and is
restricted to prefixes that are read-only for the sharers' lifetime.

#### Layout belongs to the KV owner

Layout is a per-EP, per-platform capability, not a global constant (#783).
ONNX Runtime GQA is BNSH-only on every dispatch path:
`group_query_attention_helper.h` hardcodes
`past_kv_format = Q_K_V_BNSH`. The native backend may use seq-major while ORT
remains BNSH because each backend owns its KV buffers and neither reads the
other's bytes (#522, #726).

A per-step transpose bridge is disqualified: decode reads the complete live KV
every step, so a bridge adds a second full KV pass on the hot path (#783). For
heterogeneous multi-EP placement (#603), the design rule is therefore:
**the owner of the KV buffer chooses the layout; another EP must accept that
layout or not participate in that KV.** The proposed stride descriptor and
capability negotiation are tracked in #783. The fixed-stride/dummy-tail work in
#759 remains complementary fault-safety machinery; it cannot repair residency
that a layout has already scattered.

**Weight offload and CUDA graph capture no longer force each other off on the
managed no-spill path (#716).** The legacy pager's alloc/copy/free operations are
capture-illegal because each page-in returns a different device pointer and a
captured graph bakes pointers into its recorded nodes, so on that path enabling
offload still disables capture (see the module docs in
`crates/onnx-genai-engine/src/native_decode/cuda.rs`). **#716** removes the
mutual exclusion for the managed no-spill authority path: each retained weight
`key` is served from a **reserved-once device virtual address** whose physical
granules are committed on page-in and decommitted on eviction under the same VA
(mechanism proven survivable by #727; isolated end-to-end for a captured read
that tracks repaged granules in
`crates/onnx-runtime-cuda-memory/tests/vmm_stable_va_weight_slot_gpu.rs`, and at
the residency level in `weight_paging.rs`'s
`vmm_retained_weight_key_keeps_a_stable_virtual_address_across_repage`). Physical
granules still come from the #740 authority-scoped pool through `carve()`
suballocation and are charged once on the global `0→1 / 1→0` granule-ref
attribution — no second allocator, no per-weight reservation (the mistake that
made #733 net-negative). The addressing change is strictly underneath #723's
`StableResident` policy, which is unchanged. Gating: capture is kept ON under
offload only when `weight_offload_enabled && managed_no_spill` (the explicit
byte `--vram-limit` authority that installs the VMM arena + physical pool); the
pointer-unstable `alloc_raw`/`free_raw` compatibility path keeps the old
capture-off exclusion. This unblocks the capture-fragmentation wins that #708 and
#728 landed — those took 35B-A3B decode from **154 to 34** graph segments — for
large models that need offload, and was the hard prerequisite for #755 (managed
no-spill VMM as the default with automatic weight streaming), which has since
landed: managed no-spill is now the native CUDA default and over-budget loads
auto-enable streaming. Ordering rule the
safety relies on, enforced in code rather than by comment: page-ins (and thus the
eviction-driven `decommit` of any stable slot) run entirely under the residency
mutex, whole-step capture is capture-once/replay-many with no page-ins during
replay, and the engine additionally declines capture when a step reports a
bypass/eviction — so no `decommit` of a baked VA can occur while a replay is in
flight (the case #727 explicitly did not prove safe).

**Sequential weight prefetch does not work and was deleted (#715, #718).** It was
implemented, measured to produce no usable compute/transfer overlap on the dev
box, and removed in PR #715. AirLLM's own analysis (#718) reaches the same
conclusion independently — the bottleneck is disk/transfer bandwidth, not a lack
of overlap — and rates prefetch at ~10% by their own account. On a model that
does not fit there is not enough independent compute to hide the transfer behind,
so any doc presenting prefetch/compute overlap as the plan, the lever, or an
existing capability is stale. The withdrawn `h2d_enqueue_copy_ms` = 1.7% figure
that once retired this line of work is corrected to **18.8%** by CUDA-event
measurement (see the instrument-failure narrative below).

### The platform is part of the memory system

On Windows WDDM, `cudaMalloc` does not fail when it exceeds dedicated VRAM — the
driver spills into "Shared GPU Memory" backed by system RAM (47.8 GB on the
development machine, against 8 GiB of real VRAM). `cuMemGetInfo` reports only the
8 GiB, so **nothing in our code can see that it happened**.

Three consequences worth internalising before measuring anything here:

- **VRAM-floor comparisons are invalid on WDDM.** The non-arena arm is not bounded
  by VRAM, so "the arena needs a higher limit" says nothing about the arena.
  Measure **committed bytes**, which the driver bounds on both paths.
- **A load that "works" may be running out of system RAM.** That is a latency
  cliff, not a capability, and the governor cannot account for it.
- **The arena failing where the baseline succeeds is the arena being correct.**
  `cuMemCreate` allocates physical device pages and cannot pretend otherwise.
- **Allocating device pages is not the same as keeping them — WDDM pages *our*
  granules out too (#863, measured).** `cuMemCreate` cannot fake the allocation,
  but it confers no pin. Proven single-process: a `cuMemCreate` + `cuMemMap` +
  `cuMemSetAccess` allocator mirroring the engine arena committed **and touched
  every byte** of 9,984 MiB on an 8,188 MiB card; device-resident stayed capped at
  ~7,942 MiB while the process host working set reached 9.49 GB — roughly 2 GB of
  our own committed granules spilled to host RAM.

  Therefore **`peak_committed_physical_bytes < managed_limit_bytes` and
  `oversubscribed_bytes == 0` are guarantees about our ledger, not about physical
  residency.** Every "no WDDM spill" claim in this document and in PR validations
  (including #836's) should be read that way: the accounting is correct; the
  residency implication is not. The spill is a **latency** effect, transparent to
  kernel correctness — it does not corrupt results, and it is not the #851 fault.

  Two practical consequences. Our no-spill accounting keeps total committed under
  the resolved budget, so we do not normally provoke this ourselves; the exposure
  is **system-wide** pressure, which our ledger cannot see. And our residency is
  **advisory on WDDM exactly as the driver's is** — we can choose *what* to keep
  and in what order, and we know the layer walk where the driver is blind, but we
  cannot force anything to stay. It is also the best explanation for this box's
  wall-clock spreads (identical work: 24–223 s, 700–1,383 s; 3.9–28 tok/s).

  **Scoped to WDDM.** Under TCC, `cuMemCreate` should fail at the physical limit
  rather than spill; do not let this propagate to other platforms (the #783
  lesson). Repro: `bench-bins/vram_hog_vmm.py`.

Users can opt out via NVIDIA Control Panel → *CUDA — Sysmem Fallback Policy* →
*Prefer No Sysmem Fallback*, which makes floors measurable again. The better
long-term answer is #712: enforce our own limit before the driver ever reaches
its spill threshold, which works identically on every platform.

### How this area fails

Not with wrong code. With code that is right, tested, and never reached — and
with tests that cannot fail. Recording the pattern because it has cost more
here than any bug:

- The CUDA VMM arena was installed, logged that it was installed, and committed
  **zero bytes** for a full generation. The hook fired at governor adoption,
  which on the native path is after the session has built every tensor it will
  use. Found only by printing a byte count on drop (#659).
- Forty-four GPU test files skipped silently on a machine with a working GPU,
  because the Rust path had no NVIDIA wheel discovery. A skip and a pass look
  identical in `cargo test` output (#636).
- An allocation counter stopped counting when EP allocations were rerouted, and
  about twenty-five assertions quietly became `0 == 0` (#635).
- `onnx-genai-ort` was never in any CI test step, so seven merged pull requests
  went unmeasured (#631).
- A test asserting the activation planner's load-time and run-time figures agree
  held vacuously: its fixture contained no view-producing op, and that equality
  is exactly what a view breaks (#671).
- A test asserting a recurrent-state cache hit produces byte-identical output
  held vacuously: the fixture's vocab dimension was 1, so argmax always returned
  token 0, and its recurrent state was `Identity`, so it never evolved. The
  restore function could have been replaced with `Ok(())` (#672).

Two habits follow, and they are cheap:

**Measure a quantity, not an event.** "The arena is installed" and "the arena is
being used" are different claims, and only the second is worth making. A byte
count, a hit rate, a commit count — something that can read zero when the
feature is inert.

**Before trusting a correctness test, break the code and watch it go red.** Both
vacuous tests above were counted as coverage. Deliberately stubbing the function
under test takes a minute and is the only thing that would have caught either.

### The same failure, one level up: results

The pattern is not confined to code. A *measurement* can be right, reproducible,
and still support a conclusion it does not license. Two instances, both caught
only because someone asked a second question:

- **A prefetch A/B compared demand fallback against itself.** At a 96 MiB
  budget the guard silently declined every prefetch, so both arms ran the same
  path. The counters that would have shown it did not exist yet, and "the
  feature is enabled" was inferred from the absence of an error (#673).
- **The VMM arena appeared to reach a lower VRAM floor than `cuMemAlloc`**
  — 2.56 GiB against 2.60 GiB — and that 40 MiB was about to justify deleting
  the `cuMemAlloc` path. Re-running the sweep showed the arena floor is
  **non-monotonic**: it passes at 2.34–2.36 GiB, *fails* at 2.37–2.47, and
  passes again at 2.48. Every run that passed below the failure band did so
  while printing `the memory ledger refused the arena's committed bytes ... the
  ledger understates device use`. The arena did not fit in less memory; it
  proceeded while the ledger was knowingly wrong. **The measurement and the bug
  (#694) were the same event.**

So a third habit, for numbers rather than code:

**Ask what else could produce this number.** Name an alternative mechanism and
rule it out, or say that you could not. In both cases above the alternative was
"the feature was not actually doing what the arm's name says", which is the
first thing to check and the easiest to skip.

Two corollaries worth stating because both cost time here:

- **A run that warns about its own accounting is a failed run.** The floor test
  above could pass while printing that the ledger had rejected it — as a floor
  test, it could fail to fail.
- **Check the binary is the one you think it is.** A finding in #693 that the
  arena had committed 18.8 GiB on an 8 GiB card was measured with a stale
  `target/release` build and had to be retracted. On Windows, `Copy-Item`
  preserves the source mtime, so restoring a file from a `.bak` can leave cargo
  reusing a stale artifact.

### A third failure mode: the instrument itself

The two above are about code that never runs and numbers that mean something
else. There is a third, and it cost the most: **an instrument that runs, reports
a plausible number, and measures the wrong thing.**

`h2d_enqueue_copy_ms` reported host-to-device transfer at **1.7%** of decode step
time. That figure retired an entire line of work — prefetch depth, pinned versus
pageable staging, transfer overlap — and was published on #705 and #718.

It was wrong by an order of magnitude. The arithmetic gives it away: the same
step stages **7.33 GiB**, and 7.33 GiB in 30 ms would be ~250 GB/s, roughly ten
times this machine's PCIe link. The counter bracketed an *asynchronous enqueue
returning*, not a transfer completing.

The first repair renamed it `copy_wait_ms` and pointed it at
`compute_wait_fence`, whose own doc comment reads *"a stream-ordered, **non
host-blocking** cross-stream wait."* The same non-measurement survived under a
name that invited the wrong reading more strongly than before. Measured properly
with CUDA events — start event on the copy stream before `cuMemcpyHtoDAsync`, end
event after, host-block to read `cuEventElapsedTime` — transfer is **18.8%**.

Two habits, both cheap:

**Divide the bytes by the time and ask whether the answer is a real bandwidth.**
A number near link speed, near memcpy speed, or ten times either is telling you
what the counter actually bracketed. This is what caught it, and it takes
seconds.

**State, in a comment on every timing counter, exactly what lies between start
and stop and whether anything there blocks the host.** A host-side `Instant` span
around a stream-ordered call must never carry a name suggesting it waited for
device work. Enforced in `weight_paging.rs` since #715.

The corollary is about naming, not timing: a counter whose name overstates what
it brackets is worse than no counter, because it is believed. The same is true of
a counter that is emitted with nothing writing it — a row reading `0.00` cannot
be distinguished from a row that was never measured, which is why #715 removed
the dead ones and added a test that every emitted counter has a live writer.

**Two more instances of this mode landed on 2026-08-12, both in accounting
rather than timing.**

*The instrument measured a real thing that was not the thing named.*
`total_weight_bytes` reported the **file size** of the external-data blob, and
`qwen14b-zp`'s blob is **50.0% dead space** — 920 initializers reference
8,329,906,176 bytes of a 16,652,453,888-byte file, contiguously, with the first
blob at offset 8,322,547,712 (an orphaned prefix from a re-export that never
truncated). So the runtime believed the model had **2.00× more weights than it
has**, and strategy inference, budget resolution and `fits_resolved_device_budget`
all consumed that. The model is over the resolved device budget by **0.599 GB**,
not the ~8.9 GB that number implied — a 14× error in the margin, and the
difference between "nearly fits" and "stream the whole model forever". Fixed in
#856 (#853): the loader now sums what initializers reference and warns when file
size exceeds it by >10%.

**What caught it was arithmetic that could not close.** Measured traffic was
4.11 GB/step against a theoretical floor of 10.53 GB/step for a full per-step
scan — *below* the floor, which is impossible. Same habit as dividing bytes by
time: compute the bound the number must obey, and when it is violated, suspect
the instrument before the system.

*The number was right and the conclusion drawn from it was not.* Raising the
weight budget moved the residency `hit_rate` from **57.09% to 81.31%** — while
the gap to the streaming floor **widened** from 1.78× to 2.30×. The count-based
rate weights a 10 KiB norm like an 11 MiB projection, and misses skew large
(~11.9 MB average page-in). `htod_bytes` is what decode pays for, so a
byte-weighted rate was added alongside it; optimizing the count-based number can
improve the report while moving no bytes at all. **A rate over a population whose
members have wildly different costs is not a cost metric.**

### The pattern underneath all three

Telemetry that never reaches the operator, three times in one subsystem:
documented environment variables that nothing reads (#688); counters computed,
used to gate a profile section, and then never printed (#719); counters printed
with nothing writing them (#715).

The #719 case is the sharpest. A residency cache paging at a **0% hit rate**
printed a weight-offload section in which **every visible row read `0.00`** — and
the first reading of that output was that the cache was inert. The opposite was
true, and provably so: the section printing at all required one of the three
*hidden* counters to be non-zero. The instrument reported the exact inverse of
what it measured.

After the third instance the response should be a guard, not more vigilance.

The `Challenger` role (`.squad/agents/challenger/charter.md`) exists to apply
this systematically to any result that would change technical direction.

### A fourth failure mode: order-dependent test state

The three above are about telemetry that never reaches the operator. There is a
fourth that reaches the operator loud and clear, and is worse for it: **a test
result that depends on the order the tests ran in, reported as a real finding.**
Because the memory and capture work is measurement-driven, a wrong number does
not merely fail — it redirects the design. Two of these landed in one week.

**#804 (carried into #794 / #801): process-frozen configuration.**
`RuntimeConfig` is a process-wide snapshot frozen on first read (an `OnceLock` in
`onnx-genai-runtime-config`). A capture-OFF phase set `ONNX_GENAI_CUDA_GRAPH=0`
*after* that snapshot was already frozen, so the mutation had no effect and every
later phase in the same process inherited the frozen policy. The forced-growth
harness then reported `captures=0` and it was attributed — in a merged PR body —
to the model *structurally declining* CUDA graph capture. It does not: with each
policy phase isolated in its own process, the model captures (`captures=4`). Two
PRs (#794, #801) failed to reproduce an end-to-end measurement on this exact
mistake before #804 found it.

**#797: a context warmed by an earlier sibling subtest.** A residency GPU test
failed ~50% of the time on a *cold* CUDA context and passed otherwise. It looked
exactly like a correctness bug in growth-under-capture. The cause was a
test-harness ordering defect: the baseline fill and the readback ran on the
legacy **default stream** while the captured memset ran on a created
**non-blocking stream**, and those two are *mutually exempt from implicit
synchronization* — so the readback raced the memset and returned partial fills
**with no CUDA error**. `cuCtxSynchronize` did **not** fix it (synchronizing the
context imposes no order between two mutually-exempt streams), which is what
refuted the obvious race theory. It had only ever "passed" because an
alphabetically-earlier sibling subtest warmed the context first.

Both are the same family: **state that is frozen or warmed once per process,
making a result depend on test order.** The two remedies are as specific as the
two mechanisms:

- **Process isolation for anything that reads process-frozen config.** A test
  phase that needs a different `RuntimeConfig` value must run in its own process
  — set the environment, then spawn a child that reads it fresh — never mutate
  the environment after the snapshot may have frozen. `mobius_seqmajor_growth_
  parity_native_cuda.rs` does this (child-process-per-mode); it is the pattern,
  not the exception.
- **Single-stream discipline for device tests.** Route *every* device operation
  a test performs — baseline fills, captured-graph launches, memsets, readbacks,
  tail writes — through one stream, so a single `cuStreamSynchronize` is a total
  order. `onnx-runtime-cuda-memory::test_support::TestStream` makes this the easy
  path and documents the default-vs-non-blocking exemption at the call site.

After the second instance in a week the response is, again, a guard rather than
more vigilance. `runtime_config()` records the environment it froze from and, in
debug/test builds only (`debug_assertions`, so the release path is untouched and
cost-free), **panics loudly** if any variable that fed the frozen snapshot is
later observed to differ — a test can no longer silently believe it set a knob
that was already frozen (`onnx-genai-runtime-config`, this PR). The single-stream
helper is the guard for the second mechanism: a test written against `TestStream`
cannot reintroduce the default-stream/non-blocking split.

The discipline that catches both before they mislead is one line: **run any test
you touch in isolation as well as in the full suite, and confirm both pass.** An
order-dependent result is exactly the one that passes in a suite and fails alone
(or the reverse), so this is the property that names the defect.

### Keeping the documentation from going stale

The same discipline applies to the prose, not just the code and the counters.
Most of the staleness this subsystem accumulates is a *performance claim written
in the present tense that a later measurement overturned* — prefetch overlap
(#715), the 1.7% h2d figure (#718), the 0% residency cache (#723), the
"tokens per granule" KV framing (#776). Three rules keep it from recurring:

- **A performance claim in docs must cite the measurement that produced it.** A
  number with no PR/issue behind it is a guess, and a guess ages into a false
  fact. If you cannot name where it was measured, mark it as an estimate.
- **When measurement overturns a claim, correct it in place and say what
  superseded it** — do not silently delete the old claim. A reader who remembers
  the old number must be able to see it was tested and retired, with the PR/issue
  that did it. That traceability is what makes the docs trustworthy.
- **Distinguish measured fact from design intent from not-yet-built.** Aspirational
  design written in the present tense is the most common failure mode: a design
  document may describe an intended design, but must not claim it is implemented
  or measured when it is not.

The last five rows are newer than the layers above them and move fastest, so
each says where it actually is. "Decided" and "in `main`" are different states
and a reader should not have to guess which one a row means.

Two rows deliberately record something working as *not finished*. A contract with
no caller and a planner with no consumer are both easy to read as done — the
type is there, the tests pass — and both were found that way rather than
reported. `VirtualBuffer` is exercised only by a GPU test; nothing on the KV path
maps through it yet.

Two things follow that are worth stating plainly, because both are the kind of
gap that reads as "already handled" from the prose:

- **`HostGovernor`'s first caller is `HostLeaseGovernor`** (#598), which adapts
  it to the lease contract in §1.1. Before that it had no callers at all outside
  trace-event definitions: the ticketed protocol, its TLA+ model, and its
  priority/aging arbitration were all real and all unreached. The engine still
  charges a private ledger; swapping that for the adapter changes runtime
  behaviour and is deliberately a separate change.
- **Activations can now be sized, and still are not.** §4.6's
  `VramBreakdown.activations_bytes` is hardcoded to `0`. `onnx-runtime-memory`
  is a liveness-based activation planner, and it now has
  `peak_activation_bytes_at_bounds` so it can answer for a *dynamic* graph —
  planning from static shapes alone defers on any symbolic dimension, which is
  every LLM, so a reservation built on it was always zero. The planner still has
  no callers in the engine. Every budget derived from the device ceiling
  continues to assume activations are free.

There is also a tension between §3's weight tiering and what the engine does
today. `model_weight_bytes` measures the whole package and the governor
subtracts it wholesale, falling back to `reservation_applied: false` when that
leaves no room — with a comment that the reservation "must never be the reason a
model refuses to start". So for a model larger than VRAM, the reservation is
silently dropped rather than triggering streaming. The mechanism that should
turn "weights do not fit" into "stream them" instead turns it into "pretend they
are free". §3 is the design that fixes this, and it is unimplemented.

Related: #596 (decisions taken while implementing the first slice of L3), #598,
#608, #514 (wire the activation planner), #628 (native KV page size unit),
#620 (how this plugs into ORT and the native runtime).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Layer 1: EP Memory (Device-Local)](#2-layer-1-ep-memory-device-local)
3. [Layer 2: Weight Residency (Per-Session)](#3-layer-2-weight-residency-per-session)
4. [Layer 3a: DeviceGovernor (Per Compute Unit — Exclusive Memory)](#4-layer-3a-devicegovernor-per-compute-unit--exclusive-memory)
5. [Layer 3b: HostGovernor (Per Machine — Shared Memory)](#5-layer-3b-hostgovernor-per-machine--shared-memory)
6. [Layer 4: ClusterCoordinator (Cross-Node, genai-server)](#6-layer-4-clustercoordinator-cross-node-genai-server)
7. [Communication Layer](#7-communication-layer)
8. [Heterogeneous Device Support](#8-heterogeneous-device-support)
9. [Hardware Topology Variants](#9-hardware-topology-variants)
10. [Decision Log](#10-decision-log)
11. [Phased Implementation](#11-phased-implementation)
12. [Resolved Questions](#12-open-questions)
13. [References](#13-references)

---

## 1. Overview

### 1.0 One memory management path

The target, stated by the repository owner and recorded here because it decides
arguments this document would otherwise have to relitigate:

> 我们只需要一份内存管理就行 — we only need one memory management.
> Once the VMM work is complete it becomes the default, and the non-VMM path is
> deleted if VMM subsumes it.

Two consequences worth reading before adding anything to this design:

**A second authority for a question the ledger already answers should be
removed rather than wired.** `placement.rs` carried its own coordinated
weight/KV/scratch VRAM arbitration, landed deliberately as Phase 3a with wiring
left to Phase 3b. It was deleted rather than wired, because connecting it would
have *created* the second authority the ledger exists to end. Its placement
planner stayed: deciding *which* layers live on the device is a different
question, and it now takes its budget as an argument so it can ask the governor
for one.

**A reservation stops being needed the moment something else accounts for the
bytes truthfully.** The fixed device reservation covered weights, activations
and runtime overhead because nothing else knew about them. When the allocator
commits physically on demand and leases each granule, it knows about the
weights — so that half of the reservation is released at adoption, and only the
half nothing else accounts for is kept. Holding both charged the same memory
twice and made the ledger refuse the arena.

**Workspace lifetime is not physical mapping ownership.** Step-scoped and
session-persistent workspace remain distinct content/lease categories, but
allocations packed into one VMM arena use one `WorkspaceZone` mapped allowance.
The zone charges global granule transitions `0→1` and refunds `1→0`, so the
last surviving allocation may perform the refund regardless of which lifetime
first mapped the granule. An arena rejects a grant from another mapped zone
before physical allocation. The current native provider also suballocates KV
from that arena, so KV and workspace content metrics remain distinct while
their physical mapped attribution uses the same arena-zone allowance. Weight
paging has separate storage and mapped attribution.

Transactional mapped growth assigns authority capacity that belongs to no
mapped zone before asking a registered holder to shrink or reclaim. A failed or
overestimated grant returns that newly assigned allowance, while committed
growth keeps it with the mapped zone until that zone is dropped (or, for a zone
registered as reclaimable, a later request transfers it).

Mapped-zone refund is an allocator/provider responsibility, not a workspace
caller responsibility. Every VMM deallocation observes the arena's actual
global granule transition `1→0` and refunds the canonical arena allowance
itself. This includes ordinary executor buffers that happen to be the final
reference to a granule first mapped by governed workspace; specialized
workspace cleanup must not issue a second refund. Retained physical-pool
handles remain authority-owned—the refund concerns virtual mapped-zone
attribution only.

### 1.1 The lease contract, and who may replace it

Memory management in onnx-genai is organized as a five-layer hierarchy. Each layer
has a distinct scope, a distinct owner, and a distinct reason to exist:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 4: ClusterCoordinator  (cross-node, genai-server)                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Layer 3b: HostGovernor  (per MACHINE — shared host RAM + disk)    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer 3a: DeviceGovernor  (per DEVICE — exclusive VRAM)     │  │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │  Layer 2: WeightResidencyManager (per-session)         │  │  │  │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │  │  │
│  │  │  │  │  Layer 1: EP Memory (device-local allocate/free) │  │  │  │  │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │  │  │
│  │  │  └────────────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why five layers?**

| Layer | Scope | Question it answers |
|---|---|---|
| 1. EP Memory | One device, one allocation | "Where do these bytes live on this device?" |
| 2. Weight Residency | One session, one model | "Which weight regions are cold/warm/hot?" |
| 3a. DeviceGovernor | One device, all sessions | "How much VRAM/device memory can each session use?" |
| 3b. HostGovernor | One machine, all devices | "How much host RAM and disk can all devices share?" |
| 4. ClusterCoordinator | All machines, all nodes | "How should memory be coordinated across machines?" |

**Why the governor split?** A device's VRAM is exclusive — only that GPU uses it. But
host RAM and disk are shared across ALL devices on the same machine. With 8 GPUs, 8
independent per-device governors each managing `host_ram_limit` would fight over the
same physical RAM. The `HostGovernor` provides a single machine-wide authority for
shared resources, while each `DeviceGovernor` manages only its exclusive device memory.

Layers compose bottom-up: the cluster coordinator calls into host governors, host
governors coordinate device governors, device governors constrain residency managers,
and residency managers allocate through EPs. No layer bypasses the one below it.

> **Mapping to DESIGN.md §26.11:** The `ResourceGovernor` described in §26.11 maps to
> what is now called `DeviceGovernor` (per-device exclusive memory). The `host_ram_limit`
> and `disk_spill_limit` fields of `ResourceLimits` are delegated to the `HostGovernor`
> (per-machine shared memory). The §26.11 interfaces and semantics remain canonical;
> this document refines the ownership boundaries.

### 1.1 The lease contract, and who may replace it
The layers above describe *authorities*. This section describes the one
mechanism they all share, because a design where every layer invents its own
accounting is a design where two authorities disagree about the same bytes.

A component does not "allocate". It **leases**: it asks a `MemoryGovernor` for
bytes on a tier for a stated role, and holds a `MemoryLease` for as long as it
occupies them. Dropping the lease returns them.

**The ledger accounts for reservations, not allocations.** This is a deliberate
boundary rather than an unfinished edge, so it belongs here before anyone goes
looking for a bug that is not there: `used(Device)` will read lower than
`nvidia-smi`, and that is correct.

What is charged is a *standing claim* — something taken once and held: the
fixed weight and overhead reservation, weight residency, KV page pools, the
native decode path's KV tensors, recurrent state. What is not charged is
per-inference transients: workspaces, scratch buffers, an execution provider's
internals. Those appear and disappear inside one kernel, they are bounded by the
activation budget, and counting them exactly would put a governor round-trip on
every `Alloc` an execution provider makes — including ONNX Runtime's, which
allocates far more often than "once per tensor" suggests.

The rule that follows: **charge a claim at the point it becomes standing, never
per allocation.** Implementing `DeviceAllocator` therefore does *not* make an
allocator governed; that trait is about how you obtain device memory, not about
who is charged for it.

```rust
trait MemoryGovernor {
    fn reserve(&self, tier: Tier, bytes: u64, role: MemoryRole, holder: HolderId)
        -> Result<MemoryLease, MemoryError>;
    fn available(&self, tier: Tier) -> u64;
}
```

Four invariants, stated as the properties tests are written against:

- **G1** For every tier, live leases never exceed that tier's limit. There is no
  advisory mode; a limit that can be exceeded is not a limit.
- **G2** A lease is released exactly once, by `Drop`. There is no `release`
  method, because an explicit one can be skipped by an early return.
- **G3** The governor never *takes* memory. Under pressure it asks; the holder
  decides how much to give, and zero is a legitimate answer.
- **G4** A refused reservation leaves every existing lease undisturbed.

**Substitutability is the point, and it is enforced by where the tests live.**
A third party must be able to supply their own manager — that is what makes the
two backends interchangeable rather than merely parallel. `MemoryLease` is
therefore built over a `LeaseAccounting` trait rather than over this crate's own
ledger, and `MemoryLease::new` is public: an implementor charges their own books
and wraps the result, so G2 holds for their leases too.

This was not true when the contract was first written. `MemoryLease` held the
built-in ledger and had no public constructor, so `MemoryGovernor` was a trait
nobody outside the crate could implement — the accounting *had* to be ours. Every
test passed, because every test lived inside the crate where private fields are
reachable. The proof now lives in `tests/`, where it sees exactly what a third
party sees and stops compiling if the contract closes again.

### 1.2 Two directions, both backends

"Bring your own memory manager" needs traffic in both directions, and a backend
that supports only one is not substitutable for a backend that supports both.

| direction | ORT backend | native backend |
|---|---|---|
| the runtime allocates; we govern it | `GovernedAllocator` registered on the environment | the EP allocator, already ours |
| we allocate; the runtime borrows it | `Value::from_external_memory` | `Session::device_binding_from_external_memory` |

Two things about the ORT side are not obvious and cost real time to discover:

- **Registering an allocator governs nothing on its own.** A session must also
  set `session.use_env_allocators`, or ORT silently builds its own. The symptom
  is a governed allocator that installs cleanly and reports zero bytes forever —
  indistinguishable from a model that does not allocate. `SessionOptions`
  therefore sets it by default, and exposes it as a typed method rather than a
  string for callers to guess.
- **ORT will not wrap a custom allocator in its own arena.**
  `CreateAndRegisterAllocator` builds ORT's *own* allocator; it does not take
  ours. ORT's rejection message says as much — *"register the allocator as
  OrtDeviceAllocator even if the provided allocator has arena logic built-in"*.
  Whatever is registered **is** the per-request path, and has to be fast enough
  to be there.
- **The ORT backend's KV does not route through an allocator we can back with
  VMM.** `CreateAllocator` selects by *device*, not by implementation
  (`crates/onnx-genai-ort/src/allocator.rs:232`, `for_session_device` wraps the
  session's own EP allocator), the dynamic decode path lets ORT allocate
  `present.*` itself, and `RegisterAllocator` is fronted by ORT's BFC arena
  (§1.3). So the VMM residency and prefix-sharing wins land on the **native**
  backend; on the ORT backend the one viable route is a VMM-arena KV buffer bound
  via `CreateTensorWithDataAsOrtValue` + `IoBinding`. That route is **scoped, not
  implemented**. Running our own Rust EPs *inside* upstream ORT through the
  plugin-EP C ABI (`RegisterExecutionProviderLibrary`) landed for the CPU EP in
  **#762** (merged); the CUDA plugin shim compiles but is unvalidated on
  hardware, so plugin-EP-hosted VMM KV remains in-progress design intent, not a
  shipped capability.

### 1.3 Why the host allocator has no arena, and the device one will need one

An arena was built for the host path and then deleted, because it lost. The
numbers, on one machine, release build, alloc+free of decode-shaped sizes:

| design | cheap governor (atomics) | governor that takes a lock |
|---|---|---|
| per-request + side table | 18.0 ns | 30.9 ns |
| per-request + header | **15.2 ns** | 30.3 ns |
| bulk-leasing arena | 31.5 ns | 31.3 ns |

Two reasons, both of which generalise:

1. **`malloc` is already an arena, with per-thread caches**, so its fast path
   takes no lock. An arena layered on top adds one.
2. **The arena moved the lock rather than removing it.** Trading the governor's
   lock for the arena's own nets zero. Winning requires *no* lock on the hot
   path — thread-local free lists, i.e. reimplementing mimalloc to beat mimalloc.

The actual cost was the **side table** that recovered an allocation's size at
free time, since ORT's `Free` passes only a pointer. Storing the lease in a
64-byte header before each block removes the table and keeps G2: `Free` reads
the lease out, which moves ownership, so `Drop` still returns the bytes exactly
once.

**This does not generalise to device memory.** `cudaMalloc` is a synchronising
driver call in the microseconds with no thread cache — three orders of magnitude
worse than host `malloc`. That is why ORT ships a BFC arena for CUDA and not for
CPU, and a device-backed allocator here will need one too. The measurement above
argues that *host* memory already has an arena, not that arenas are wrong.

The benchmark is a warm single-threaded loop. It does not measure fragmentation
over a long run, cold page faults, RSS, or contention across ORT's intra-op
threads, and the default should not be considered settled on it alone.

### 1.5 Using the allocator ABI fully

`OrtAllocator` is more than `Alloc`/`Free`/`Info`, and the optional slots are
not decoration — two of them carry information we were otherwise inventing.

| slot | what ORT uses it for | ours |
|---|---|---|
| `Alloc` | allocations during `Run` | charged to the **run** role |
| `Reserve` | allocations while **building** a session (since 1.18) | charged to the **initialization** role |
| `Info` | which device this allocator serves | the memory info it was built with |
| `GetStats` | `Limit`, `InUse`, `TotalAllocated`, `MaxInUse`, `NumAllocs`, `NumReserves` | reported from the governor and our counters |
| `Shrink` | release memory held but not in use (since 1.25) | honest no-op: this allocator pools nothing |
| `AllocOnStream` | stream-aware device allocation (since 1.23) | null — this allocator owns host memory, which has no stream |

**`Reserve` is a free `MemoryRole` signal.** ORT documents it as existing so a
custom allocator can separate session setup from `Run`. Session setup is weights
and plan state; `Run` is activations. That is exactly the distinction eviction
ordering needs — weights go before KV because they are immutable and re-readable
from disk — so charging both to one role makes the cheapest thing to evict look
as expensive as the most.

Verified rather than assumed: building a session over the `tiny-llm` fixture
makes **15 allocations, all 15 through `Reserve`**. `AllocationRoles::split()`
is therefore the default; `AllocationRoles::uniform` restores single-role
charging for an allocator where the split does not apply.

**`GetStats` makes governed memory visible through ORT's own interface**, so
tooling that already reads allocator statistics sees it without knowing this
crate exists. `Limit` is what the governor will still grant plus what we hold —
the ceiling as this allocator experiences it, not the machine's.
`NumArenaExtensions` and `NumArenaShrinkages` are *omitted* rather than reported
as zero, because a zero would read as "an arena that never extended" rather than
"no arena".

**`Shrink` is implemented as a no-op that succeeds**, which is what ORT's own
documentation specifies for a non-arena allocator. Implemented rather than left
null because null and "nothing to give" are different answers, and because when
a device-backed allocator lands it *will* have an arena — this is the hook its
pressure response belongs in, and it is the same shape as G3.

### 1.6 One allocator per lifetime, not one allocator

The mistake worth naming: treating "the allocator" as one thing. There are four
kinds of memory here with genuinely different lifetimes, and what unifies them
is the lease contract above — not a shared allocator.

| memory | lifetime | pattern | structure | paged or streamed? |
|---|---|---|---|---|
| activations, workspace | one step | same shapes every step, thousands per step | header + `malloc` (§1.3) | no — freed each step |
| KV cache | per sequence | fixed-size pages, shared, forked, migrated | paged pool | **yes** — device↔host↔disk |
| weights | model lifetime | immutable, huge, re-readable from disk | mmap + residency manager (§3) | **yes** — weight streaming |
| large logically-contiguous buffers | per step or persistent | one address range over scattered pages | `VirtualRange` | **yes** — pages remapped under a stable address |

An arena serves none of the last three: it hands out physically contiguous
blocks with no identity, which is the opposite of what paging needs, and it
would pin weights in RAM, which is the opposite of what streaming needs.

---

## 2. Layer 1: EP Memory (Device-Local)

The `ExecutionProvider` trait exposes raw device memory operations. This is the
lowest layer — purely local, no cross-session awareness, no policy.

**Reference:** `crates/onnx-runtime-ep-api/src/provider.rs`

```rust
pub trait ExecutionProvider: Send + Sync {
    /// Allocate `size` bytes of device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;

    /// Free a previously allocated device buffer.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Copy bytes between host and device, or device-to-device.
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;

    // ... other EP methods (execute, supports_op, etc.)
}
```

Every higher layer ultimately calls through these primitives. The EP knows nothing
about weight tiers, budgets, or other sessions.

### 2.1 Kernel scratch/workspace governance audit (#736)

There are two ways a CUDA kernel can obtain device scratch:

- **Governed** — the kernel reports a size through
  `Kernel::workspace_requirement` and the session executor reserves it via
  `ExecutionProvider::{reserve_workspace, prepare_mapped_growth, allocate}`
  *before execution*. This counts against the device authority, so a
  capacity shortfall surfaces as a pre-header **HTTP 429** (`MemoryOverload`)
  through `DriverFailure::from_engine_error`, never as a late 500/OOM.
  Prepare-only planning (#747) sizes the workspace from resolved shapes at
  admission; `MappedGrowthGrant` (#748) grows it transactionally against the
  same authority. Planning and execution size through the *same* helper.
- **Ungoverned** — the kernel calls `CudaRuntime::alloc_raw` (a thin
  `cuMemAlloc`/pool wrapper) directly. These bytes are invisible to the
  authority: under the managed no-spill path (default on native CUDA under #755,
  or an explicit `--vram-limit`) they are the residual
  `activations_bytes=unknown` (#514) and can drive a late device OOM instead
  of a clean 429.

**Measured finding: size from use, not allocation.** Five of six slices found
over-reservation rather than only an ungoverned allocation: IndexShare reserved
unreachable present-K/V staging (#751), GQA reserved `WS_SCORES` outside its f32
reference route (#795), cuBLASLt treated a 32 MiB ceiling as demand although
measured algorithms used 0–96 bytes (#799), GQA packed QKV staging is populated
only when a packed tensor is split (#736), and GQA's query BNSH slot was
unnecessary for unpacked non-RoPE `Sq==1` while its output slot is never needed
for `Sq==1` (#736). The default-domain `Attention` scores were the genuine-use
exception: every route stages fp32 scores, so none reports `NONE` (#736). The
detailed byte accounting remains in the
[site-by-site audit](#cuda-raw-allocation-audit-736-2026-08-10) above.

**Per-slice issue map and a correction to this document (#800/#802).** The
#736 governance epic ran as seven route-conditional slices, each threading a
static signal through the shared sizing helper rather than deleting bytes: the
f32-only GQA `WS_SCORES` (#795), the packed-input-only GQA QKV staging (#806),
the prefill-only staging (#810), and the dense-alias-only default-domain
`Attention` staged K/V (#813) were the over-reservation slices; the
default-domain `Attention` fp32 scores were the one slice where the bytes are
**genuinely needed** on every route (#802). Recorded so the next audit does not
re-derive it: #800 found **this document** had been wrong — it described the
Attention Phase-2a scores as fp16-sized when the kernel sizes them from the
tensor dtype (`elem_size = dtype.element_size()`,
`crates/onnx-runtime-ep-cuda/src/kernels/attention.rs:368`); the audit table and
formulas above now carry `elem_size`, not a fixed fp16 width.

**Audit guidance (design intent, derived from #751, #795, and #799):**

1. Enumerate dispatch routes and buffer reads/writes before classifying the
   allocation; `alloc_raw` identifies the source, not whether that route uses it.
2. Treat constant-sized reservations as suspect: they commonly encode an API
   ceiling, a worst-case shape, or a layout artifact rather than measured use.
3. Find the static route signal — output arity, dtype/shape dispatch, or
   single-call scope in these slices — and thread it through the shared sizing
   helper so planning and execution cannot drift.
4. Do not govern a bypass before making its reservation path-sensitive. The
   naive governed form can convert invisible waste into charged waste; #799
   would have charged five 32 MiB ceilings for algorithms using under 100 bytes.
5. Record a genuine-use result too, so the next audit does not repeat it (#736).

This is admission control, not tidiness: every byte charged against the device
authority tightens committed-granule admission (#745), reducing admissible
concurrency on the multi-request-serving axis measured by #750 and #777.

**Implemented today.** The executor has one central
`is_planned_workspace_node` predicate
(`crates/onnx-runtime-session/src/executor/bindings.rs`) covering QMoE,
IndexShare, Attention Phase-2a, GQA (`WS_SCORES`, packed QKV-projection staging,
and BNSH-transpose scratch folded into one composite
reservation), the
default-domain `Attention` score + route-required staged-K/V composite, and the
standard/fused GEMM family
(#747, #751, #753, #795, #799, #736), rather than parallel per-feature checks.
Prepare-only planning tracks and reserves one peak per lifetime class
(`SessionPersistent` and `StepScoped`), so a graph mixing the two governs both
without summing sequential users of the same slot (#753).

**Governed today**

| kernel / node | file · site | byte formula | size class | lifetime | mechanism |
|---|---|---|---|---|---|
| `BlockQuantizedMoE` (QMoE) | `block_quantized_moe.rs` `workspace_requirement` | routing + dequant + GEMM staging over `tokens × experts × hidden` | config-derived (static per model) | session-persistent | #747 prepare-only + #748 grant |
| `pkg.nxrt::IndexShare` | `index_share.rs` `workspace_requirement` | selected-token scores plus only the staging actually reachable for the path | prompt/cache dependent | session-persistent | #751 + prepared workspace |
| `com.microsoft::Attention` Phase-2a | `attention.rs` `run_attention_phase2a` (governed branch) | `align256(batch·num_heads·sq·sk·elem) + 32 MiB` cuBLASLt workspace | prompt-dependent; score matrix is ~256–512 MiB at B=1/H=32/S=2048 | step-scoped | #753 + prepared workspace |
| `com.microsoft::GroupQueryAttention` f32 reference scores | `group_query_attention.rs` `workspace_requirement` | `batch·heads·sq·present_capacity·sizeof(f32)` only on the path that materializes scores | prompt/cache dependent | session-persistent | #795 + prepared workspace |
| `com.microsoft::GroupQueryAttention` packed QKV staging | `group_query_attention.rs` `workspace_requirement` | `align256(B·sq·num_heads·head_dim·elem) + 2·align256(B·sq·kv_num_heads·head_dim·elem)` only when a packed QKV tensor is split | prompt-dependent; ~24 MiB (f32) / 12 MiB (fp16) at B=1/H=32/hd=128/Hkv=8/sq=1024; `NONE` on unpacked/fused-decode routes | session-persistent | #736 + prepared composite workspace (shares the `WS_SCORES` slot) |
| `com.microsoft::GroupQueryAttention` BNSH transpose scratch | `group_query_attention.rs` `gqa_transpose_scratch` / `gqa_workspace_layout` | Q: `align256(B·sq·num_heads·head_dim·elem)` for `sq>1`, packed Q, or RoPE; output: same only for `sq>1`; packed `sq==1` Q overlays packed-Q staging | prompt/route dependent; derived 32 MiB f32 / 16 MiB fp16 at B=1/H=32/hd=128/sq=1024; unpacked non-RoPE decode charges 0, RoPE decode Q-only | session-persistent | #736 + existing prepared GQA composite |
| `Attention` (default domain) scores + staged dense K/V | `standard_attention.rs` `std_attention_workspace_layout` / `workspace_requirement` | scores `B·Hq·Sq·Sk·sizeof(f32)` always; staged K/V each `B·Hkv·present_seq·head_dim·elem` only for reachable dense aliased growth | prompt/cache/route dependent; derived scores ~512 MiB at B=1/Hq=32/S=2048; derived K+V staging 16 MiB f32 / 8 MiB fp16 at B=1/Hkv=8/S=2048/D=128; fixed-capacity append staging 0 | step-scoped (per-call prefill/batched) **and** session-persistent (capture-eligible single-token decode) | #736 + one prepared composite; route-sized through the shared helper |
| `MatMul`, `Gemm`, `MatMulNBits`, `FusedMatMulBias`, `FusedGemm` cuBLASLt scratch | `blas.rs` shared planner; `fused_gemm.rs`, `gemm.rs`, `matmul.rs`, `matmul_nbits.rs` | selected heuristic `workspaceSize`, bounded by the 32 MiB preference ceiling | measured 0–96 bytes on #799 shapes; algorithm dependent | session-persistent shared peak | #799 exact plan/execute helper + prepared workspace |

The Attention **fused** (flash) prefill path uses shared memory only and
allocates no device scratch, so `workspace_requirement` returns `NONE` for it —
the executor reserves nothing (#753). By contrast, the default-domain
`Attention` kernel (`standard_attention.rs`) has **no** flash/shared-memory
route: `attention_row` always stages the fp32 score matrix in global memory, so
every valid dispatch reserves it and none reports `NONE` (#736) — a genuine-use
result, not a missed optimization. Its staged K/V component is separate and
route-conditional: fixed-capacity append/no-past/no-present-output routes add
zero, while dense aliased growth adds disjoint key/value regions in the same
prepared composite. `attention.rs` remains outside #799's
shared cuBLASLt slot because its workspace is concurrent with the score matrix
inside the StepScoped Phase-2a composite; direct `execute` keeps the documented
self-owned compatibility/opt-out fallback (#753, #799).

KV layout and residency are a separate contract; use
[KV layout and residency](#kv-layout-and-residency) rather than restating them
in this workspace table (#791).

The authoritative site-by-site raw-allocation table, measured GEMM workspace
result, byte savings, and evidence-selected next slice live in
[CUDA raw allocation audit](#cuda-raw-allocation-audit-736-2026-08-10) above.
They are not repeated here so status and byte formulas have one source of truth.

---

## 3. Layer 2: Weight Residency (Per-Session)

Treats immutable model weights as a three-tier hierarchy within a single session.
This is the design from [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md), consolidated here.

> **The weight budget is what the KV reservation leaves behind, and it is never
> renegotiated (#857, open).** At load, `engine/load.rs` computes
>
> ```text
> weight_budget = resolved_device_budget − kv_bytes_per_token × max_context
> 6,116,140,442 = 7,726,753,178 − 1,610,612,736     (qwen14b-zp, exact to the byte)
> ```
>
> `max_context` is the model's **declared** `max_sequence_length` (8,192), not what
> the request uses, and `set_weight_residency_budget` is called once. So a 16-token
> run holds **1.611 GB** of device budget for KV it never commits (measured
> committed KV in that run: 402,653,184 bytes) while weights stream **3.94 GB per
> decode step** because they do not fit.
>
> Measured by changing only `max_sequence_length` 8192 → 1024, same weights
> (hard-linked, byte-identical), same binary, same prompt:
>
> | | ctx 8192 | ctx 1024 |
> |---|---:|---:|
> | weight `budget_bytes` | 6,116,140,442 | 7,525,426,586 |
> | `htod_bytes_per_token` | 3,943,690,240 | 1,850,486,784 (**2.13× less**) |
> | `page_ins_per_token` | 372 | 162 |
> | `hit_rate` | 57.09% | 81.31% |
>
> Token IDs byte-identical. So the reservation is genuinely recoverable and
> converts almost 1:1 into reduced streaming. **Shipping a smaller
> `max_sequence_length` is not the fix** — that configuration simply cannot serve
> longer sequences. The reservation exists to guarantee a sequence can reach its
> declared maximum, so the fix is an **elastic** weight budget with a guaranteed
> reclaim path, not a smaller reservation. Note the arena banner's "dynamic
> KV/weight lending" is accurate about the handle pool but overstates the lending:
> KV's *reservation* is deducted up front and never revisited.
>
> Note also the gap to the floor **widened** (1.78× → 2.30×) as the budget grew:
> policy efficiency (#837 item 3) and this budget split are independent and
> multiplicative levers.

### 3.1 Components

```text
ONNX loader / WeightStore
  owns read-only mmaps and validated WeightRef ranges
                 |
                 v
WeightRegionCatalog
  classifies shared tensors and expert subranges; records format/layout/alignment
                 |
                 v
WeightResidencyManager  <---- Resource Governor sub-budgets (Layer 3)
  cold mmap | warm host pages | hot device pages | LRU/heat | in-flight state
                 |
         +-------+--------+
         |                |
         v                v
ExpertStore facade    static layer placement
fused MoE kernels     dense/attention/embedding/lm-head bindings
```

### 3.2 Interfaces

```rust
struct WeightRegion {
    id: WeightRegionId,
    backing: ExternalRange,       // path identity + offset + length
    class: WeightClass,           // Shared or Expert { layer, expert, role }
    representation: WeightFormat, // f16, int4, MXFP4, IQ*, ...
    alignment: usize,
    transfer_page_bytes: usize,
}

trait WeightResidencyManager {
    /// Acquire a lease on a weight region. The lease pins the region in its
    /// current tier, preventing eviction until dropped.
    fn lease(&self, request: WeightRequest) -> Result<WeightLease>;

    /// Speculatively begin loading a weight region toward a warmer tier.
    fn prefetch(&self, request: WeightRequest);

    /// Report observed expert routing for heat-based admission.
    fn observe_routes(&self, layer: usize, experts: &[u32]);

    /// Current residency state for monitoring/debugging.
    fn usage(&self) -> WeightResidencySnapshot;
}

trait ExpertStore {
    /// Ensure the given experts are resident on `target` device.
    /// Returns a lease that pins them until dropped.
    fn ensure_resident(
        &self,
        layer: usize,
        experts: &[u32],
        target: WeightTarget,
    ) -> Result<ExpertLease>;
}
```

A lease contains stable mapped, host, or device views plus any readiness fence. Its
lifetime prevents eviction. Device leases remain live until stream completion, not
merely until kernel launch returns.

### 3.3 Tier Semantics

#### Cold: Read-Only mmap Backing

- Canonical bytes are ONNX external data and remain immutable.
- A cold hit returns a checked subrange of the existing mmap.
- CPU direct-compressed kernels may consume that range without a host copy.
- Clean mapped pages can be discarded after use; strict budget reporting must
  distinguish owned host-cache bytes from OS page-cache/RSS.

Inline initializers are acceptable for small shared tensors, but offloadable expert
pools must use external data.

#### Warm: Bounded Host RAM

Warm entries are optional derived copies of canonical packed pages:

- pageable aligned pages for CPU reuse;
- pinned pages for repeated H2D transfer;
- optional CPU-prepacked or dequantized panels only when their expanded byte cost is
  charged to the host budget and measured reuse justifies it.

The warm cache uses byte-based LFRU admission with hysteresis, not entry count. A miss
always falls back to mmap, so a zero-byte host cache remains functional.

#### Hot: Bounded Device VRAM

A device entry is an EP-owned allocation containing either canonical compressed bytes
or an explicitly versioned device-prepacked representation. It is keyed by
`(region, representation, device)` and charged at actual allocated bytes. Eviction is
legal only when no lease or transfer owns the entry. Failed speculative prefetch must
not displace a leased or demonstrably hotter entry.

On a fully resident plan, entries are pinned for the session and the manager collapses
to stable pointer lookup.

### 3.4 Expert Paging and Batching

For one admitted token batch:

1. Run the model-exact router and compute exact top-k IDs and aggregation weights.
2. Union selected expert IDs across all token rows.
3. Group rows by expert and compute token counts.
4. Ask `ExpertStore` for a residency plan under the current tier budgets.
5. Execute resident experts together; process the remainder in bounded waves/tiles.
6. Scatter and combine with the original aggregation weights.
7. Release CPU leases immediately and device leases after completion fences signal.
8. Record routes, bytes, stalls, and reuse for future admission/prefetch.

**Expert is the policy unit; page/tile is the capacity unit.** A whole expert is
convenient for heat and LRU decisions but may itself exceed free RAM/VRAM. Store each
expert contiguously, then divide its FC1/FC2/FC3, scale, zero-point, and bias ranges
into page-aligned transfer tiles.

- **Admission:** choose experts by heat/priority.
- **Transfer:** move bounded pages/tiles.
- **Compute:** consume direct compressed blocks or double-buffered panels.
- **Atomicity:** a logical expert lease groups all companion ranges required by the
  current kernel wave; it does not imply the whole expert is copied at once.

### 3.5 Residency Policy

- Shared attention, router, normalization, embeddings, and other dense weights have
  higher base priority than routed experts because they are touched predictably.
- Expert admission combines frequency, recency, bytes, measured load cost, and tokens
  served while resident. Use hysteresis to avoid ping-pong.
- A page used by an in-flight kernel or transfer is non-evictable.
- Derived dequantized/prepacked entries are disposable and never the sole copy.

### 3.6 Cache Policy and Prefetch

Track, per layer and expert:

- frequency and last-use step;
- bytes and current tier;
- load latency by source tier;
- in-flight/pinned/leased state;
- prefetch hit/miss/waste;
- tokens served while resident.

Default admission should be LFRU-like with hysteresis. Prefetch sources, ordered from
least speculative to most speculative:

1. exact routes already computed for the current fused op;
2. union of routes across the admitted batch;
3. recent per-layer heat;
4. predicted next-layer routes.

Prediction must be optional and budgeted. It must not evict a leased expert or a
demonstrably hotter resident expert merely to chase a weak prediction.

---

## 4. Layer 3a: DeviceGovernor (Per Compute Unit — Exclusive Memory)

The DeviceGovernor is the engine-level byte-budget authority for a single
accelerator's **exclusive** memory (GPU VRAM, NPU on-chip SRAM, etc.).
**CRITICAL: there is exactly one DeviceGovernor per DEVICE, not per session.**
It is shared across all sessions on that device.

### 4.0 Ownership Model

```rust
/// Process-level singleton that owns all governors.
/// Constructed once at process startup, lives for the process lifetime.
pub struct MachineRuntime {
    host: Arc<dyn HostGovernor>,
    devices: Vec<Arc<dyn DeviceGovernor>>,
    topology: MemoryTopology,
}

/// Process-unique identity for one physical allocation. IDs are allocated
/// monotonically with checked exhaustion and are never reused while any handle,
/// alias, or outstanding operation can reference them.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct PhysicalAllocationId(u64);

pub struct DeviceAllocation {
    pub id: PhysicalAllocationId,
    pub size: usize,
    pub device: LocalDeviceId,
    lease: AllocationLease,
}

pub struct HostAllocation {
    pub id: PhysicalAllocationId,
    pub size: usize,
    pub class: HostMemoryClass,
    lease: AllocationLease,
}

pub enum HostMemoryClass {
    Pageable,
    PinnedDma,
}

/// Sessions do not own governors. They hold budget leases.
pub struct SessionBudgetLease {
    device: LocalDeviceId,
    vram_reserved: usize,
    host_reserved: usize,
    /// Dropped on session end → budget returns to pool.
    _guard: LeaseGuard,
}
```

**Ownership rules:**

- `MachineRuntime` is a process-level singleton (one per server process). It owns
  all `DeviceGovernor` and `HostGovernor` instances directly.
- Sessions request a `SessionBudgetLease` from the governor, not direct access to
  the governor's allocation primitives. The lease represents a reserved portion of
  the device's budget.
- **Single-process deployment:** `MachineRuntime` owns everything directly. The
  existing `EngineResourceGovernor` becomes a facade that delegates to the
  `MachineRuntime`'s `DeviceGovernor` for the relevant device.
- **Multi-process (future):** Would require shared-memory coordination or a
  dedicated governor process — deferred to Phase 4.
- The governor enforces limits; sessions cannot exceed their lease. Every allocation
  is charged to both the physical pool (governor-level) and the session lease.

> **Mapping to DESIGN.md §26.11:** The `ResourceGovernor` described in §26.11
> corresponds to what is now called `DeviceGovernor`. The §26.11 interfaces,
> reconfigurability semantics, and error contracts remain canonical; this section
> refines scope to **device-exclusive memory only**. The `host_ram_limit` and
> `disk_spill_limit` fields of `ResourceLimits` are delegated to the
> `HostGovernor` (§5).

### 4.1 DeviceGovernor Scope

The DeviceGovernor owns **only** resources exclusive to one compute unit:

| Resource | Example | Owned by |
|---|---|---|
| Accelerator VRAM | GPU HBM, NPU SRAM | DeviceGovernor |
| Host RAM (shared) | DDR for offload, staging | HostGovernor (§5) |
| Disk spill (shared) | SSD cold tier | HostGovernor (§5) |

### 4.2 User-Facing Limit Model

The device-memory limit is expressible three ways — absolute bytes, fraction, or auto:

```rust
#[derive(Debug, Clone, Copy)]
pub enum ResourceLimit {
    Bytes(u64),
    Fraction(f32),   // of detected tier capacity
    Auto,            // sane default (90% VRAM)
}
```

The `ResourceLimits` struct splits across governor layers:

```yaml
serving:
  memory:
    limits:
      # DeviceGovernor (per device — exclusive memory)
      vram_limit: "8GiB"          # or fraction or auto

      # HostGovernor (per machine — shared across all devices)
      host_ram_limit: "16GiB"
      disk_spill_limit: null
```

### 4.3 Live Reconfigurability

```rust
impl DeviceGovernor {
    pub fn set_vram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn snapshot(&self) -> DeviceSnapshot;
}
```

Limits can change mid-session without restart. The governor holds limits behind
`ArcSwap<ResolvedLimits>` for lock-free reads on the hot admission path;
`reconfigure` serializes writers with a mutex.

### 4.4 Cross-Session Invariant

```rust
// Invariant checked on every reconfigure:
//   sum(session.max_pages or actual) ≤ budget.total_pages
//   interactive_reserve = round(reserve_fraction × budget.total_pages)
//   every per-session cap ≤ budget.total_pages − interactive_reserve
```

A single runaway session cannot blow the device's VRAM budget — all allocations go
through the same `can_allocate` gate, which the DeviceGovernor bounds in bytes.

### 4.5 Tiered Eviction on Lowering

When a VRAM limit is lowered below current usage, the DeviceGovernor drives
existing eviction tiers in order:

1. Drop **background** sessions' KV (cheap to re-prefill).
2. Offload **paused standard** sessions' KV to the warm tier — **requesting host
   RAM quota from HostGovernor** (§5) before copying.
3. Preempt **running standard** sessions (recompute from last checkpoint on resume).
4. **Interactive** sessions and `interactive_reserve` are touched last.

The eviction sequence follows a two-phase protocol:

**Phase 1 (reversible):** Mark candidate pages/experts for eviction, reduce soft
ceiling. Reserve destination-tier capacity (host RAM or disk) via HostGovernor. If
any reservation fails, unmark all candidates and restore the soft ceiling. No data
has moved yet.

**Phase 2 (commit):** Actually evict/copy marked data to the reserved destinations.
Once commit begins, partial progress is acceptable — some pages may be evicted while
others are still in flight. The invariant is that `sum(usage) ≤ ceiling` holds at
the END of the sequence, not at every intermediate point. After all evictions
complete, the new ceiling is published to sessions.

**Failure semantics:** Eviction is NOT transactional in the database sense. The
guarantee is: if Phase 1 (planning/reservation) fails, the ceiling reverts and no
side effects occur. If Phase 2 (execution) fails partway through, the old ceiling
remains advertised, but completed reclaim actions (KV already dropped, data already
offloaded) are irreversible and reported in the outcome. The API returns
`ReconfigureOutcome` which lists all completed reclaim actions regardless of
overall success or failure.

If the target cannot be met after exhausting all tiers, the governor returns
`ResourceError::CannotSatisfyLoweredLimit` with the list of actions already taken.

**Offload flow (DeviceGovernor → HostGovernor interaction):**

```text
GPU 0 VRAM full → DeviceGovernor: "need to evict to host"
    → MemoryTopology: transition_to_host(source_allocation, priority, deadline)
    → discrete: await charged HostAllocation, copy_async, publish, release source
    → unified: reclassify the same PhysicalAllocationId, no copy
    → caller receives the committed HostAllocation
```

### 4.6 VramBreakdown

The DeviceGovernor decomposes device memory usage into trackable components:

```rust
pub struct VramBreakdown {
    pub model_weights_bytes: u64,      // dense weights
    pub hot_expert_cache_bytes: u64,   // hot expert cache (from ExpertStore)
    pub kv_cache_bytes: u64,           // KV cache pages
    pub activations_bytes: u64,        // peak activation working set
    pub ort_overhead_bytes: u64,       // arena / session / EP overhead
    pub total_bytes: u64,
}
```

**Constraint:** `dense_weights + hot_expert_cache + kv_cache + activations + overhead ≤ ceiling`

### 4.7 Sub-Budget Coordination: KV vs Expert LRUs

Independent KV and expert LRUs must not race for the last bytes. The DeviceGovernor
assigns coordinated sub-budgets and can rebalance them with hysteresis:

```text
VRAM ceiling = resident shared weights
             + hot expert/device cache
             + KV cache
             + activations and routing scratch
             + EP/runtime overhead
```

Both the `WeightResidencyManager` and KV cache manager receive sub-budgets from
the DeviceGovernor and return usage. On lowering a live limit: cancel speculative
reservations, evict unleased weight pages, demote KV, reduce batch/scratch, and
return an actionable minimum-working-set error if still impossible.

> See [DESIGN.md §26.11](./DESIGN.md) for the full governor design including
> config surfaces (YAML + Rust API + Python), error experience
> (`ResourceError` with what/why/how), and implementation status.

> See [DESIGN.md §43.2](./DESIGN.md) for the declaration that expert weights
> are "not KV cache" and the rationale for separate APIs with shared concepts.

### 4.8 Config Surface

**YAML** (device-specific limits in the `memory:` block):

```yaml
serving:
  memory:
    limits:
      vram_limit: "8GiB"            # absolute; or "0.9" (fraction); or "auto"
      allow_runtime_override: true   # permit live reconfigure via API
    interactive_reserve_pct: 20
    eviction_policy: priority_then_lru
```

**Rust:**

```rust
let engine = GenAiEngine::load(model, EngineConfig { limits, .. })?;
engine.device_governor(device_id).set_vram_limit(ResourceLimit::Bytes(6 << 30))?;
let snap = engine.device_governor(device_id).snapshot();
```

**Python:**

```python
engine.set_vram_limit("6GiB")          # default device
snap = engine.resource_snapshot()       # dict: per-tier used / limit / headroom
```

### 4.9 Error Experience

```rust
pub enum ResourceError {
    VramOverBudget {
        requested_bytes: u64,
        limit_bytes: u64,
        available_bytes: u64,
        breakdown: VramBreakdown,
        tier: Tier,
        suggestions: Vec<Remedy>,
    },
    CannotSatisfyLoweredLimit {
        requested_limit_bytes: u64,
        floor_bytes: u64,
        breakdown: VramBreakdown,
        reclaimable_bytes: u64,
        suggestions: Vec<Remedy>,
    },
    SessionLimitExceedsGlobal {
        session: SessionId,
        requested_pages: usize,
        global_pages: usize,
    },
    HostQuotaDenied {
        device: LocalDeviceId,
        requested_bytes: u64,
        host_available_bytes: u64,
        suggestions: Vec<Remedy>,
    },
}
```

---

## 5. Layer 3b: HostGovernor (Per Machine — Shared Memory)

The HostGovernor is the machine-level authority for **shared** memory resources
that all devices on a single physical host contend for. There is exactly **one
HostGovernor per machine**, regardless of how many devices it has.

### 5.1 Why a Separate Governor?

Host RAM and disk are **shared across all devices** on the same machine:

- When GPU 0 offloads weights from VRAM → host RAM, it uses the same physical
  DDR that GPU 1-7 also use for offload.
- If 8 independent DeviceGovernors each manage `host_ram_limit` independently,
  each thinking it has 25% of host RAM, they collectively claim 200% and OOM.
- Pinned memory pools, DMA staging buffers, and disk spill paths are
  machine-global OS resources.

The HostGovernor provides a **single source of truth** for shared memory.

### 5.2 HostGovernor Interface

```rust
trait HostGovernor: Send + Sync {
    /// Request host RAM pages for offload (VRAM → host).
    /// `device` is a local ordinal — HostGovernor is per-machine, so local
    /// identification suffices. In distributed contexts, ClusterCoordinator
    /// maps `GlobalDeviceId` → `(ClusterNodeId, LocalDeviceId)` before dispatching
    /// to the appropriate node's HostGovernor.
    ///
    /// This call briefly locks the ledger and returns immediately. The Future
    /// resolves either to an allocation already charged to the ledger or to an
    /// error; a wakeup without a reservation is never exposed to callers.
    fn request_host_pages(
        &self,
        request: HostPageRequest,
    ) -> PressureTicket;

    /// Release previously granted host pages.
    fn release_host_pages(&self, alloc: HostAllocation) -> Result<()>;

    /// Current host RAM limit.
    fn host_ram_limit(&self) -> ResourceLimit;

    /// Current disk spill limit (None = disabled).
    fn disk_spill_limit(&self) -> Option<ResourceLimit>;

    /// Reconfigure host RAM limit live.
    fn set_host_ram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome>;

    /// Reconfigure disk spill limit live.
    fn set_disk_spill_limit(&self, limit: Option<ResourceLimit>) -> Result<ReconfigureOutcome>;

    /// Global view: per-device host RAM usage breakdown.
    fn snapshot(&self) -> HostSnapshot;
}

/// A pending-or-granted request. Implements:
/// `Future<Output = Result<HostAllocation, ResourceError>>`.
pub struct PressureTicket {
    request_id: PressureRequestId,
    generation: PressureGeneration,
    governor: Weak<HostGovernorInner>,
    /// Cleared when Future output is claimed or cancellation is linearized.
    armed: bool,
}

pub struct PressureRequestId(pub u64);
pub struct PressureGeneration(pub u64);

pub struct HostPageRequest {
    pub device: LocalDeviceId,
    pub bytes: usize,
    pub class: HostMemoryClass,
    pub priority: Priority,
    pub deadline: Instant,
}

/// Canonical identity types are defined in DISTRIBUTED_RUNTIME.md §7.1.
/// HostGovernor consumes only `LocalDeviceId` because its scope is one node.

/// Snapshot of machine-wide shared memory usage.
pub struct HostSnapshot {
    pub host_ram_limit_bytes: u64,
    pub host_ram_used_bytes: u64,
    pub host_ram_headroom_bytes: u64,
    /// Per-device breakdown of host RAM usage.
    pub per_device_host_usage: Vec<(LocalDeviceId, u64)>,
    pub disk_spill_limit_bytes: Option<u64>,
    pub disk_spill_used_bytes: u64,
    pub pinned_memory_bytes: u64,
}
```

### 5.3 Host Allocation Lifecycle

When a DeviceGovernor needs to offload data from device memory to host RAM:

1. **Request:** DeviceGovernor calls
   `host_governor.request_host_pages(HostPageRequest { ... })` and receives a
   `PressureTicket`.
2. **Arbitrate:** HostGovernor checks total host RAM usage across all devices.
   If the request fits, it charges an allocation and returns a ready ticket.
3. **Pressure:** If over budget, HostGovernor can:
   - Ask other DeviceGovernors to release their host pages (cross-device pressure).
   - Cascade to disk spill (if enabled): move cold host pages to SSD.
   - Deny the request with `HostQuotaDenied` error.
4. **Grant:** Resolve the ticket only after `HostAllocation` is charged.
5. **Release:** DeviceGovernor calls `release_host_pages()` when data is
   promoted back to VRAM or no longer needed.

### 5.3.1 Deadlock Prevention: Ticketed Non-Blocking Pressure Protocol

**INVARIANT: No thread ever WAITS while holding a governor lock.**

The earlier "lock ordering" approach (Host → Device[0] → Device[1]) is
insufficient because: a HostGovernor lock serializing requests means a waiting
request holds the lock, but a reclaiming device needs that same lock to release
pages — deadlock.

Instead, requests use a ticketed, non-blocking state machine:

```rust
enum PressureState {
    Pending(HostPageRequest),
    /// The allocation is already charged before the ticket is woken.
    Granted(HostAllocation),
    /// The Future claimed the allocation; ticket drop is now a no-op.
    Claimed,
    Cancelled,
    Failed(ResourceError),
}
```

**Linearization and ownership rules:**

1. `request_host_pages()` rejects zero/oversized or impossible pinned-pool
   requests, then briefly acquires the HostGovernor ledger lock. If
   capacity is available, it creates and charges `HostAllocation` immediately
   and returns an already-ready ticket. Otherwise it inserts one `Pending`
   request with a unique `PressureRequestId`, releases the lock, and enqueues
   non-blocking reclaim notices.
2. Reclaim workers never run under the HostGovernor lock. They evict at their
   own pace and call `release_host_pages()`, which briefly updates the ledger.
3. After every release, the governor applies deterministic priority/FIFO
   arbitration. For each satisfiable request it first reserves bytes and stores
   `Granted(HostAllocation)` under the ledger lock, then wakes the ticket.
   Fresh requests cannot steal those bytes. FIFO applies within a priority;
   bounded aging raises the effective priority of an older satisfiable request,
   so continuous high-priority arrivals cannot starve it indefinitely.
4. Awaiting a ticket holds no governor lock. A successful poll atomically
   replaces `Granted(allocation)` with `Claimed`, disarms the ticket, and returns
   that allocation. A ticket can yield a grant at most once.
5. Ticket drop sends a non-blocking, lossless
   `cancel(request_id, generation)` command to the governor's cancellation
   mailbox. Because the queue does not own the caller's cancellation capability,
   drop is observable. The cancellation slot is reserved when the request is
   created, so Drop does not allocate or discard cancellation under backpressure.
   Cancel, timeout, grant, and reconfigure are serialized under the ledger lock:
   if grant wins, cancellation releases that exact allocation; if cancellation
   wins, a later grant is forbidden.
6. `PressureGeneration` identifies the HostGovernor configuration generation,
   not an individual request. Reconfigure increments it and explicitly
   revalidates or fails pending requests from the prior generation.

The invariant is therefore stronger than "no lock across await": **no caller is
woken successfully until capacity is atomically charged to an owned
allocation**.

The grant/cancel/timeout/reconfigure linearization points and capacity ledger
are modeled in
[`PressureProtocol.tla`](../specs/tla/PressureProtocol.tla).

**Test scenario — two devices simultaneously requesting under pressure:**

1. GPU 0 receives pending ticket A for 10 GB; GPU 1 receives pending ticket B
   for 8 GB. Neither task holds a lock while awaiting.
2. Reclaim workers release 12 GB. Under the ledger lock, arbitration grants one
   ticket according to priority/FIFO and charges its allocation before wakeup.
3. A fresh 12 GB request cannot consume the reserved bytes.
4. A later 6 GB release permits the second ticket to be granted and charged.
5. A timeout racing either grant has one ledger-ordered winner; no allocation
   is leaked and no waiter proceeds without ownership.

### 5.3.2 Implementation Refinement and Ledger Audit

The implementation must conform to the action mapping and trace contract in
[`specs/tla/REFINEMENT.md`](../specs/tla/REFINEMENT.md). In particular, the
following are one ledger-locked transition each, not a sequence of observable
partial updates:

- charge exact bytes and publish `Granted` before wakeup;
- claim the granted `PhysicalAllocationId` and disarm cancellation;
- cancel or time out a grant and return that exact allocation;
- increment configuration generation and resolve every prior-generation
  pending request; and
- credit reclaimed bytes before reconsidering queued tickets.

Test traces identify tickets by `PressureRequestId` and allocations by
`PhysicalAllocationId`; queue indices and addresses are not stable identities.
The trace records checked byte extents, owner `LocalDeviceId`, configuration
generation, previous/new ticket state, and the ledger counters after the
transition.

Debug and conformance builds independently recompute:

```text
host_ram_used
  = reclaimable allocations
  + granted-but-unclaimed allocations
  + claimed live allocations
  + other explicitly classified host allocations
```

The recomputed total must equal the sum of authoritative
`PhysicalAllocationId` entries and remain within the configured limit. A
counter matching its own previous value is not sufficient evidence. Overflow,
duplicate physical identity, negative headroom, wakeup without a charge, and a
terminal ticket retaining an allocation are immediate failures.

The deterministic test campaign covers multiple variable-sized tickets per
device, exact-capacity admission, cancellation-mailbox saturation,
grant/claim/cancel/timeout/reconfigure races, priority aging, and reclaim by the
requesting device itself. Each failing schedule records a replayable scheduler
decision trace.

### 5.4 Config Surface

**YAML** (machine-wide shared limits in the `memory:` block):

```yaml
serving:
  memory:
    limits:
      # HostGovernor (per machine — shared across all devices)
      host_ram_limit: "16GiB"       # or fraction of detected host RAM; or "auto" (25%)
      disk_spill_limit: null         # null = disabled (default)
      allow_runtime_override: true
    offload_to_cpu: true             # enables warm tier offload via HostGovernor
```

**Rust:**

```rust
engine.host_governor().set_host_ram_limit(ResourceLimit::Bytes(16 << 30))?;
let snap = engine.host_governor().snapshot();
println!("Host RAM: {} / {} bytes across {} devices",
    snap.host_ram_used_bytes, snap.host_ram_limit_bytes,
    snap.per_device_host_usage.len());
```

**Python:**

```python
engine.set_host_ram_limit("16GiB")
snap = engine.host_snapshot()  # dict: used / limit / per_device_usage
```

### 5.5 Cross-Device Arbitration

With multiple devices, the HostGovernor must decide **which device gets host RAM**
when the pool is contested:

- **Priority-based:** Interactive sessions' offload requests outrank background.
- **Proportional:** Each device gets a fair share by default, but can borrow
  from idle devices.
- **Pressure cascade:** When host RAM is full, the HostGovernor can trigger
  disk spill for the coldest host-resident data across any device.

```text
8×GPU system, 256GB host RAM, host_ram_limit = 200GB:
  GPU 0: 40GB host usage (weight offload)
  GPU 1: 35GB host usage (KV offload)
  ...
  GPU 7: 25GB host usage
  Total: 180GB / 200GB → 20GB headroom

  GPU 3 requests 30GB offload → only 20GB available
  → HostGovernor: pressure GPU 0 to spill 10GB coldest pages to disk
  → or: deny with HostQuotaDenied + suggestion to raise host_ram_limit
```

---

## 6. Layer 4: ClusterCoordinator (Cross-Node, genai-server)

### 6.1 When Is This Needed?

- **Single-machine, single-session:** Not needed. DeviceGovernor + HostGovernor (Layers 3a/3b) are sufficient.
- **Single-machine, multi-session:** DeviceGovernor enforces per-device budgets; HostGovernor
  arbitrates shared host RAM. No additional coordination layer is required for correctness.
- **Single-machine, cross-session optimizations:** The ClusterCoordinator (running locally)
  provides shared weight dedup, KV prefix sharing, and expert migration.
- **Multi-node distributed deployment:** The ClusterCoordinator coordinates across
  HostGovernors on different machines.

**Key clarification:** For single-machine deployments, the DeviceGovernor (§4) handles
per-device budgeting and the HostGovernor (§5) handles shared memory arbitration. The
ClusterCoordinator adds value only for cross-session *optimizations* or multi-node coordination.

### 6.2 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     genai-server                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ClusterCoordinator (global, cross-session)                │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │ Weight Dedup  │  │ KV Pool      │  │ Budget Arbiter  │  │  │
│  │  │ (shared mmap) │  │ (prefix      │  │ (rebalance      │  │  │
│  │  │               │  │  sharing)    │  │  sub-budgets)   │  │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │  │
│  └───────┬───────────────────┬───────────────────┬────────────┘  │
│          │                   │                   │               │
│    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐       │
│    │ Session 0 │       │ Session 1 │       │ Session N │       │
│    │           │       │           │       │           │       │
│    │ Governor  │       │ Governor  │       │ Governor  │       │
│    │ Residency │       │ Residency │       │ Residency │       │
│    │ KV cache  │       │ KV cache  │       │ KV cache  │       │
│    └───────────┘       └───────────┘       └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 ClusterCoordinator Interface

```rust
/// Global memory coordinator across sessions (single-machine optimizations
/// or multi-node coordination).
///
/// Sits above DeviceGovernors and HostGovernors, adjusting their budgets
/// and providing cross-session optimizations.
trait ClusterCoordinator: Send + Sync {
    // ── Shared Weight Deduplication ──

    /// Register a weight region for deduplication. Returns a handle
    /// that multiple sessions can use without each allocating a copy.
    /// Uses CUDA IPC / mmap for zero-copy sharing.
    fn register_shared_weight(
        &self,
        region: &WeightRegion,
        device: GlobalDeviceId,
    ) -> Result<SharedWeightHandle>;

    /// Acquire a read-only view of a shared weight. Ref-counted;
    /// the weight stays resident as long as any session holds a view.
    fn acquire_shared_view(
        &self,
        handle: &SharedWeightHandle,
        session: SessionId,
    ) -> Result<WeightView>;

    // ── Cross-Session KV Cache ──

    fn request_kv_pages(
        &self,
        session: SessionId,
        num_pages: usize,
        priority: PagePriority,
    ) -> Result<Vec<PageHandle>>;

    fn release_kv_pages(&self, pages: Vec<PageHandle>);

    fn lookup_prefix(
        &self,
        token_hash: u64,
        num_tokens: usize,
    ) -> Option<PrefixCacheHit>;

    // ── Expert Migration ──

    fn migrate_expert(
        &self,
        expert: ExpertId,
        from: SessionId,
        to: SessionId,
    ) -> Result<()>;

    fn report_expert_heat(
        &self,
        session: SessionId,
        layer: usize,
        activations: &[(ExpertId, u32)],
    );

    // ── Budget Arbitration (drives Layer 3 governors) ──

    fn memory_pressure(&self) -> MemoryPressure;

    /// Rebalance sub-budgets across sessions.
    /// Pushes adjustments down to each session's DeviceGovernor
    /// via `governor.reconfigure()`.
    fn rebalance(&self) -> Vec<BudgetAdjustment>;

    fn set_session_limit(
        &self,
        session: SessionId,
        limit: ResourceLimit,
    ) -> Result<ReconfigureOutcome>;
}

struct BudgetAdjustment {
    session: SessionId,
    new_kv_budget_bytes: usize,
    new_expert_cache_bytes: usize,
    reason: AdjustmentReason,
}

enum AdjustmentReason {
    KvPressure { requesting_session: SessionId },
    ExpertHeatShift,
    GlobalPressure,
}
```

### 6.4 How ClusterCoordinator Calls Down Into Governors

```text
ClusterCoordinator.rebalance()
  │
  ├── reads: device_governor[0].snapshot() → {used: 120GB, limit: 141GB, headroom: 21GB}
  ├── reads: device_governor[1].snapshot() → {used: 139GB, limit: 141GB, headroom: 2GB}
  │   └── GPU 1 under pressure!
  │
  ├── decides: GPU 1 needs 15GB for KV. GPU 0 has 21GB headroom.
  │   Migrate cold expert 742 (3GB) from GPU 1 → GPU 0.
  │   Lower GPU 1's expert sub-budget by 3GB, raise KV sub-budget.
  │
  ├── calls: device_governor[1].reconfigure({vram_kv: +3GB, vram_expert: -3GB})
  │   └── DeviceGovernor triggers tiered eviction on expert cache
  │
  └── calls: device_governor[0].reconfigure({vram_expert: +3GB})
      └── DeviceGovernor admits the migrated expert
```

The coordinator never sets a per-session limit that would violate the global ceiling.
If it tries, the DeviceGovernor rejects with `ResourceError::CannotSatisfyLoweredLimit`
and the coordinator rolls back.

### 6.5 Three Progressive Strategies

#### Strategy 1: Static Isolation

Each session gets a fixed budget. No cross-session coordination. The per-device
`DeviceGovernor` operates unchanged. Identical to running independent processes.

#### Strategy 2: Shared Weights + Shared KV Pool

Deduplicate shared weights (attention/router/embed stored ONCE via CUDA IPC);
unify KV cache pool. This is where `register_shared_weight()` and
`request_kv_pages()` become active.

```text
8×H200, 1128 GB total:
  Shared weights (attention/router/embed): 50 GB (stored ONCE via CUDA IPC)
  Expert weights (per-session shard): 700 GB
  KV cache (global pool): 350 GB  ← was 8×43=344 GB, now unified
  Scratch: 28 GB
  Savings: 7 × 50 GB = 350 GB freed from weight duplication
```

#### Strategy 3: Dynamic Expert Migration + Replication

The coordinator monitors expert heat and actively rebalances — replicating hot
experts across GPUs and evicting cold ones. Extends the `observe_routes()`
mechanism from the per-session residency manager.

### 6.6 Cross-Node Memory Coordination

For multi-node (e.g., Mac Studio cluster), the coordinator splits into:

- **LocalCoordinator** per machine — handles CUDA IPC / mmap sharing.
- **ClusterCoordinator** in genai-server (global role) — handles cross-node expert
  migration (via Communicator), cross-node prefix cache lookup, and global budget
  arbitration.

```text
┌─ Node 0 ─────────────────┐    ┌─ Node 1 ─────────────────┐
│ LocalCoordinator          │    │ LocalCoordinator          │
│ ├── Session 0 (GPU 0)     │    │ ├── Session 2 (MLX)       │
│ └── Session 1 (GPU 1)     │    │ └── Session 3 (MLX)       │
└───────────┬───────────────┘    └───────────┬───────────────┘
            │                                │
            └───────── ClusterCoordinator ──────┘
                       (in genai-server)
```

Cross-node expert migration transfers weights via the Communicator (§7). Within a
node, shared weights use zero-copy IPC. The `ClusterCoordinator` delegates intra-node
sharing to the `LocalCoordinator`.

---

## 7. Communication Layer

The `Communicator` trait is the runtime-level communication abstraction for
distributed inference. It lives alongside EPs in the runtime — EPs produce
tensors; the Communicator moves them between devices.

> See [DISTRIBUTED_RUNTIME.md §3.1](./DISTRIBUTED_RUNTIME.md) for the canonical
> `Communicator` trait definition (including `CommHandle`, `all_to_all_v`,
> `exchange_counts`). This section summarizes the interface and focuses on how
> communication interacts with memory governance.

### 7.1 Core Trait (Summary)

The canonical `Communicator` trait (defined in DISTRIBUTED_RUNTIME.md §3.1)
provides:

- **Collectives:** `all_reduce`, `all_to_all`, `all_to_all_v`, `all_gather`,
  `broadcast`, `reduce_scatter`, plus ticketed `exchange_counts`
- **Point-to-point:** `send`, `recv` with `CommHandle` for async completion
- **Synchronization/failure:** asynchronous `barrier`, communicator-wide `abort`
- **Metadata:** world `rank()`, ordered `members()`, `group_size()`,
  `group_id()`, `backend_name()`

### 7.2 Memory Integration Requirements

Backend inventory, capability, and performance claims live only in
[DISTRIBUTED_RUNTIME.md §4](./DISTRIBUTED_RUNTIME.md#4-communicator-backends).
From the memory architecture's perspective:

- Direct-device transports register complete read and write
  `PhysicalAllocationId` lease sets before enqueue and retain them until the
  local `CommHandle` is terminal. Read/read aliasing is legal; a write lease
  excludes all other access to that allocation.
- Host-staged transports obtain staging capacity through `PressureTicket`; an
  enqueue cannot begin with uncharged staging memory.
- Unified-memory transports alias one ledger entry and do not create a second
  host charge.
- Communicator abort transitions outstanding handles to terminal errors before
  their retained allocation leases are released.
- Test builds emit the lossless lease lifecycle required by
  [`specs/tla/REFINEMENT.md`](../specs/tla/REFINEMENT.md); allocator reuse before
  terminal release is rejected even when a new view has a different address.

### 7.3 Communicator Supersedes DispatchTransport

> **DEPRECATED:** The `DispatchTransport` trait from
> [MOE_EXPERT_PARALLELISM.md §8](./MOE_EXPERT_PARALLELISM.md) is superseded by
> `Communicator`. `DispatchTransport` was MoE-specific (send/recv/all_reduce/all_to_all
> scoped to expert dispatch). `Communicator` generalizes this to support tensor
> parallelism, pipeline parallelism, and expert parallelism through a single interface.
>
> **Use `Communicator` for all new work.** `DispatchTransport` remains in
> MOE_EXPERT_PARALLELISM.md for historical reference only.

The key differences:

| Aspect | DispatchTransport (deprecated) | Communicator |
|---|---|---|
| Scope | MoE expert dispatch only | All distributed patterns |
| Buffer type | `Tensor` (opaque) | `DeviceBuffer` with explicit dtype/len |
| Sub-groups | None | `CommGroup` for hybrid TP+EP strategies |
| Device awareness | Implicit | Explicit `TransportCapability` for staging |
| Backends | 3 (CUDA IPC, Host, Network) | 5 (NCCL, Gloo, TB5, RDMA, InProcess) |

### 7.4 Buffer Location Awareness

`TransportCapability` is defined only in
[DISTRIBUTED_RUNTIME.md §3.3](./DISTRIBUTED_RUNTIME.md#33-buffer-location-awareness).
Its staging location is a `GlobalDeviceId`; conversion to `LocalDeviceId`
occurs only after rank-local dispatch.

---

## 8. Heterogeneous Device Support

Because communication lives outside EP, different EP types coexist naturally.
Full design in [DISTRIBUTED_RUNTIME.md §5](./DISTRIBUTED_RUNTIME.md).

### 8.1 Format Negotiation at Boundaries

> See [DISTRIBUTED_RUNTIME.md §5.2](./DISTRIBUTED_RUNTIME.md) for the canonical
> `TensorFormat` definition, including shape, strides, logical/wire dtype,
> quantization parameters, alignment, and ownership.

The runtime inserts format conversion nodes at EP boundaries automatically.
Conversion placement (`convert_on: GlobalDeviceId`) is determined by the graph
partitioner based on bandwidth and compute cost heuristics.

### 8.2 Mixing Scenarios

| Scenario | Devices | Communicator | Use Case |
|---|---|---|---|
| Multi-GPU single node | 8× H200, CUDA EP | NCCL | TP + EP for large models |
| Mac Studio cluster | 4× M3 Ultra, MLX EP | Thunderbolt | EP across Macs |
| Hybrid GPU+Mac | H200 + Mac Studio | Gloo (TCP) | Overflow to Mac for cold experts |
| NPU + GPU | NPU EP + CUDA EP | InProcess | NPU handles attention, GPU handles FFN |
| Multi-vendor GPU | ROCm EP + CUDA EP | Gloo/RDMA | Rare but architecturally possible |
| Dev/test | Multiple CPU EPs | InProcess | Verify distributed logic locally |

---

## 9. Hardware Topology Variants

Different hardware configurations require different governor topologies. The engine
selects the appropriate topology at startup based on hardware probing.

### 9.1 MemoryTopology (Trait-Based, Not Enum)

The topology is a **struct with trait-object fields**, not a closed enum. This
ensures new hardware configurations (future NPU architectures, CXL-attached memory,
etc.) can be added without modifying upper-layer code.

```rust
/// Engine constructs at startup based on hardware probing.
/// Upper layers access only via trait methods — never match on TopologyKind.
struct MemoryTopology {
    /// Per-device governors. Empty = CPU-only.
    devices: Vec<Arc<dyn DeviceGovernor>>,

    /// Per-machine shared memory. Always present.
    host: Arc<dyn HostGovernor>,

    /// Informational only — for logging, metrics, config validation.
    /// Never use for control flow.
    kind: TopologyKind,
}

/// Descriptive, not prescriptive. New variants added freely.
#[non_exhaustive]
enum TopologyKind {
    CpuOnly,
    SingleGpu,
    MultiGpuDiscrete,
    GpuWithNpu,
    UnifiedMemory,
    // future variants added without breaking changes...
}
```

**Key design points:**

- **Upper layers use allocation methods for new storage and
  `MemoryTopology::transition_to_host()` for existing storage**, never match on
  `TopologyKind`.
- **`TopologyKind` is for logging/metrics/config validation only.** Adding a new
  variant is not a breaking change thanks to `#[non_exhaustive]`.
- **Unified memory:** `UnifiedGovernor` implements *both* `DeviceGovernor` and
  `HostGovernor` traits. The same `Arc` is placed in both `devices` and `host`:

```rust
// Apple Silicon / DGX Spark construction
let unified = Arc::new(UnifiedGovernor::new(total_mem, recommended_working_set));
MemoryTopology {
    devices: vec![unified.clone()],  // it IS a DeviceGovernor
    host: unified.clone(),           // it IS also a HostGovernor
    kind: TopologyKind::UnifiedMemory,
}

// 8×H200 construction
MemoryTopology {
    devices: gpu_governors,  // 8 independent DeviceGovernors
    host: host_governor,     // 1 shared HostGovernor
    kind: TopologyKind::MultiGpuDiscrete,
}

// CPU-only construction
MemoryTopology {
    devices: vec![],         // no accelerator
    host: host_governor,     // HostGovernor manages everything
    kind: TopologyKind::CpuOnly,
}
```

### 9.2 Variant 1: CPU-Only

**Example:** Inference on a server with no GPU.

- No `DeviceGovernor` — there is no exclusive device memory to manage.
- `HostGovernor` manages **all** memory (host RAM for weights, activations, KV cache).
- The warm and hot tiers collapse: everything lives in host RAM.
- Disk spill provides the cold tier.

```text
MemoryTopology::CpuOnly
└── HostGovernor (manages host RAM as both "device" and "host" memory)
    ├── host_ram_limit: "32GiB"
    └── disk_spill_limit: "100GiB"
```

### 9.3 Variant 2: Single GPU + CPU

**Example:** Desktop with one discrete GPU.

- 1 `DeviceGovernor` for the GPU's VRAM.
- 1 `HostGovernor` for host RAM offload and disk spill.
- The simplest discrete topology. No cross-device arbitration needed.

```text
MemoryTopology::Discrete
├── HostGovernor (host RAM + disk)
└── DeviceGovernor[GPU 0] (VRAM)
```

### 9.4 Variant 3: Multi-GPU Discrete

**Example:** 8×H200 server.

- N `DeviceGovernor`s, one per GPU, each managing exclusive VRAM.
- 1 `HostGovernor` arbitrating shared host RAM across all N devices.
- HostGovernor prevents 8 GPUs from collectively over-committing host RAM.
- `ClusterCoordinator` optional for cross-session optimizations (weight dedup,
  expert migration).

```text
MemoryTopology::Discrete
├── HostGovernor (256GB DDR shared across all GPUs)
├── DeviceGovernor[GPU 0] (141GB HBM)
├── DeviceGovernor[GPU 1] (141GB HBM)
│   ...
└── DeviceGovernor[GPU 7] (141GB HBM)
```

### 9.5 Variant 4: GPU + NPU

**Example:** Intel Core Ultra with Arc GPU + NPU, or Qualcomm with Adreno GPU + Hexagon NPU.

- Each accelerator gets its own `DeviceGovernor`.
- NPU device memory is typically tiny (a few MB of on-chip SRAM); it relies heavily
  on host DMA for weight streaming.
- The NPU requests `HostMemoryClass::PinnedDma` through HostGovernor. These
  leased pages **must not be evicted** by GPU offload pressure.
- HostGovernor tracks pinned and pageable pools separately; a pinned request
  never silently degrades to pageable memory.

```text
MemoryTopology::Discrete
├── HostGovernor (host RAM, with pinned vs pageable tracking)
├── DeviceGovernor[GPU] (VRAM: 8GB)
└── DeviceGovernor[NPU] (on-chip SRAM: 4MB, relies on host DMA)
```

### 9.6 Variant 5: Unified Memory (Apple Silicon, DGX Spark)

**Example:** M4 Ultra Mac Studio, NVIDIA DGX Spark (Grace Blackwell).

On unified memory architectures, "device memory" and "host memory" are the **same
physical DRAM**. The GPU/NPU and CPU share a single memory pool with hardware
coherence. Separate DeviceGovernor and HostGovernor would create a false dichotomy.

- `UnifiedGovernor` replaces both DeviceGovernor and HostGovernor.
- Manages **logical partitions** within unified memory:
  - Device working set (what the GPU/NPU is actively using)
  - Host working set (what the CPU is actively using)
  - Shared weight pages (accessible by both without copying)
- No copy between "host" and "device" — just pointer sharing.
- Apple's `recommendedMaxWorkingSetSize` provides the device partition hint.

**Double-accounting prevention via one physical-allocation ledger:**

Every governor allocation handle carries a `PhysicalAllocationId`. On discrete
memory, copying VRAM to host creates a new physical ID and retires the old one
after copy commit. On unified memory, both trait interfaces resolve to the same
ledger and a residency transition preserves the ID:

```rust
pub struct UnifiedGovernor {
    total_capacity: usize,
    allocations: HashMap<PhysicalAllocationId, AllocationEntry>,
    /// Physical bytes are charged exactly once per ledger entry.
    physical_used: usize,
    device_wired: usize,
    host_pageable: usize,
    shared_coherent: usize,
}

#[derive(Clone, Copy)]
pub struct ResidencyCounters {
    device_wired: usize,
    host_pageable: usize,
    shared_coherent: usize,
}

pub struct AllocationEntry {
    pub size: usize,
    pub residency: UnifiedResidency,
    /// Access leases do not create another physical charge.
    pub cpu_readers: u32,
    pub device_readers: u32,
    pub writer: Option<AccessOwner>,
    pub owners: HashSet<AllocationOwner>,
}

#[derive(Clone, Copy)]
pub enum UnifiedResidency {
    DeviceWired,
    HostPageable,
    /// Coherent pages intentionally active from both CPU and accelerator.
    SharedCoherent,
}

impl UnifiedGovernor {
    /// Transactional ledger transition; no copy and no new allocation.
    pub fn reclassify(
        &mut self,
        id: PhysicalAllocationId,
        target: UnifiedResidency,
    ) -> Result<()> {
        let (size, source) = self.lookup(id)?;
        // Compute and validate the complete next state before publishing any
        // counter or entry mutation.
        let next = self.counters()
            .checked_transition(source, target, size)?;
        self.validate_counters(next)?;
        self.device_wired = next.device_wired;
        self.host_pageable = next.host_pageable;
        self.shared_coherent = next.shared_coherent;
        self.allocations.get_mut(&id).unwrap().residency = target;
        Ok(())
    }

    pub fn snapshot(&self) -> UnifiedSnapshot {
        // Every mutation checked these invariants before publication.
        UnifiedSnapshot {
            total: self.total_capacity,
            device_wired: self.device_wired,
            host_pageable: self.host_pageable,
            shared_coherent: self.shared_coherent,
            free: self.total_capacity - self.physical_used,
        }
    }
}
```

**Key invariants:**

- `physical_used == sum(unique allocation sizes)` and never exceeds capacity.
- Residency counters are a disjoint classification:
  `device_wired + host_pageable + shared_coherent == physical_used`.
- CPU/device aliases add access leases and owners to the same entry; they never
  create another physical charge and do not change residency based on a vague
  "dominant accessor" heuristic.
- Reclassification validates the destination sub-budget before mutating any
  counter and is atomic under the ledger lock.
- Release names the physical ID and owner. The entry is removed only after all
  owners and access leases are gone.

Upper layers do not call `HostGovernor::request_host_pages()` to offload an
existing allocation. They call one topology operation carrying the source
handle:

```rust
impl MemoryTopology {
    pub async fn transition_to_host(
        &self,
        source: DeviceAllocation,
        priority: Priority,
        deadline: Instant,
    ) -> Result<HostAllocation>;
}
```

- **Discrete topology:** obtain a ticketed host allocation, copy into its new
  physical ID, await the copy completion fence, publish the host handle, then
  release the device allocation. Failure before publish releases the host
  reservation and preserves the source.
- **Unified topology:** atomically `reclassify(source.id, HostPageable)` and
  transfer the consumed source lease into a host handle for the same physical
  ID. The source destructor is disarmed; no allocation or copy occurs.
- **Shared coherent use:** transition to `SharedCoherent`; CPU and device access
  leases independently protect the same entry.

```text
MemoryTopology::Unified
└── UnifiedGovernor (192GB unified pool)
    ├── device_partition: 160GB (GPU working set)
    ├── host_partition: 24GB (CPU working set)
    └── shared: 8GB (weights readable by both, no copy)
```

### 9.7 Variant 6: Multi-Node Cluster

**Example:** 4× Mac Studio cluster via Thunderbolt 5, or multi-node DGX.

- Each node runs its own governor topology (any of variants 1–5).
- `ClusterCoordinator` sits above per-node `HostGovernor`s.
- Cross-node expert migration and prefix sharing via `Communicator` (§7).

```text
┌─ Node 0 ───────────────────┐    ┌─ Node 1 ───────────────────┐
│ UnifiedGovernor (M4 Ultra)    │    │ Discrete (8×H200)           │
│ └── 192GB unified             │    │ ├── HostGovernor (256GB DDR) │
│                               │    │ └── 8× DeviceGovernor       │
└───────────────┬───────────────┘    └───────────────┬───────────────┘
                │                                │
                └──────── ClusterCoordinator ────────┘
                           (in genai-server)
```

### 9.8 Topology-Agnostic Upper Layers

Upper layers access `MemoryTopology` via trait methods on the contained governors.
They never match on `TopologyKind` — the trait dispatch handles routing:

```rust
/// Upper-layer code — topology-agnostic.
async fn load_weights(
    topo: &MemoryTopology,
    device: LocalDeviceId,
    size: usize,
) -> Result<()> {
    if let Some(dev) = topo.device(device) {
        // Accelerator present — allocate on device
        dev.request_device_memory(size)?;
    } else {
        // CPU-only — route through host governor
        topo.host.request_host_pages(
            HostPageRequest {
                device: LocalDeviceId::cpu(),
                bytes: size,
                class: HostMemoryClass::Pageable,
                priority: Priority::Normal,
                deadline: Instant::now() + DEFAULT_ALLOCATION_TIMEOUT,
            },
        ).await?;
    }
    Ok(())
}

/// Existing allocation identity is mandatory for offload.
async fn offload_to_host(
    topo: &MemoryTopology,
    allocation: DeviceAllocation,
) -> Result<HostAllocation> {
    topo.transition_to_host(
        allocation,
        Priority::Normal,
        Instant::now() + DEFAULT_ALLOCATION_TIMEOUT,
    ).await
}

/// Combined snapshot across all governors.
fn topology_snapshot(topo: &MemoryTopology) -> TopologySnapshot {
    TopologySnapshot {
        devices: topo.devices.iter().map(|d| d.snapshot()).collect(),
        host: topo.host.snapshot(),
        kind: topo.kind,
    }
}
```

This means `WeightResidencyManager`, session scheduling, and `ParallelStrategy`
never branch on hardware type — they call the same trait methods regardless of
whether the system is CPU-only, discrete multi-GPU, unified, or a heterogeneous
mix. On unified memory, the `DeviceGovernor` and `HostGovernor` trait calls both
route to the same `UnifiedGovernor` instance, which internally manages logical
partitions within the single memory pool.

---

## 10. Decision Log

Key architectural decisions and their rationale:

### D1: Governor splits into DeviceGovernor (per device) and HostGovernor (per machine)

**Decision:** One `DeviceGovernor` per physical device manages exclusive device memory
(VRAM). One `HostGovernor` per machine manages shared host RAM and disk spill.

**Rationale:** A per-session governor cannot enforce `sum(session.usage) ≤ device_capacity`.
The DeviceGovernor is the single source of truth for device byte budgets. Host RAM and
disk are shared across all devices on a machine; per-device governors managing these
resources independently would contend over the same physical memory. The HostGovernor
provides a single machine-wide authority for shared resources.

### D2: Communication outside EP, not inside

**Decision:** The `Communicator` trait lives alongside EPs in the runtime, not inside
the EP trait.

**Rationale:** EPs produce tensors; the Communicator moves them. This separation
enables heterogeneous deployment (CUDA EP + MLX EP in the same distributed graph)
and keeps EP implementations focused on compute.

### D3: Expert weights are not KV cache

**Decision:** Expert weights get a separate `ExpertStore` / `WeightResidencyManager`
API, not storage in `onnx-genai-kv`.

**Rationale:** Expert weights are immutable model data with different access patterns
(heat-based LRU, expert-major layout, read-only). KV cache is mutable, sequence-keyed,
and copy-on-write. They share *concepts* (tiering, leases, LRU, page tables) but not
identity, keys, or mutability semantics.

### D4: DispatchTransport → Communicator (superseded)

**Decision:** The MoE-specific `DispatchTransport` trait is deprecated in favor of
the general `Communicator` trait.

**Rationale:** `DispatchTransport` was scoped to MoE expert dispatch. Tensor
parallelism and pipeline parallelism need the same primitives. One trait covers
all distributed patterns with sub-groups for hybrid strategies.

### D5: Single-machine uses DeviceGovernor + HostGovernor; ClusterCoordinator only for multi-node

**Decision:** For single-machine deployments, the per-device `DeviceGovernor` manages
device memory and the `HostGovernor` arbitrates shared host resources. The
`ClusterCoordinator` adds value only for cross-session optimizations (shared weight
dedup, KV prefix sharing) or multi-node coordination.

**Rationale:** Avoid adding a coordination layer where the governor pair already
enforces all invariants. The ClusterCoordinator is an optimization layer, not a
correctness requirement for single-machine deployments.

### D6: ONNX multi-device annotations are hints, not execution constraints

**Decision:** The ONNX IR v11+ multi-device spec (`DeviceConfigurationProto`,
`ShardingSpecProto`, `NodeDeviceConfigurationProto`) is preserved in the IR as
optional annotations. The runtime reads them as **placement hints** for the graph
partitioner, but the `ParallelStrategy` makes the actual placement decision.

**What the ONNX spec provides:**
- `DeviceConfigurationProto` — model-level declaration of available device groups
  and their sizes.
- `NodeDeviceConfigurationProto` — per-node annotation of which device config it
  belongs to.
- `ShardingSpecProto` — per-tensor description of how axes are sharded across
  devices (shard vs replicate per dimension).

**How it interacts with our layers:**

```text
ONNX model (with optional sharding annotations)
    │
    ▼
Loader: parse DeviceConfigurationProto → NodeDeviceHints
    │
    ▼
IR: Node.device_hints (optional, informational)
    │
    ▼
ParallelStrategy: reads hints as ILP seed placement
    │  Hint says "TP on dim=0, 8 devices"
    │  → validate annotation feasibility (device count matches,
    │    EP supports the required ops, memory budget accommodates
    │    the sharding, communication is achievable), then generate
    │    strategy from hint — skipping the optimization *search*
    │    but not the *validation*. If validation fails, emit a
    │    diagnostic warning and fall back to automatic placement.
    │  No hint → fall back to automatic graph analysis
    │
    ▼
Communicator: executes communication (hint-agnostic)
    │
    ▼
EP: execute() (unaware of sharding)
```

**What we store in IR:**

```rust
struct NodeDeviceHints {
    /// Which device configuration this node prefers.
    pub config_name: Option<String>,
    /// Sharding specs for inputs/outputs.
    pub input_sharding: Vec<Option<ShardingSpec>>,
    pub output_sharding: Vec<Option<ShardingSpec>>,
}

struct ShardingSpec {
    /// Device IDs across which this tensor is sharded/replicated.
    pub devices: Vec<String>,
    /// Per-axis sharding description.
    pub sharded_dims: Vec<ShardedDim>,
}
```

**Rationale:**
- ONNX annotations are declarative ("SHOULD be sharded this way"), not imperative.
  They don't specify communication — that's the `Communicator`'s job.
- If Mobius or other exporters annotate models with sharding specs, the partitioner
  can skip expensive graph analysis and use the hints directly.
- The `onnx-std` crate already validates these annotations (`MultiDeviceConfigurationRule`)
  but the runtime IR (`onnx-runtime-ir`) currently drops them after parsing. The
  loader should preserve them into `NodeDeviceHints` when present.
- Without annotations, the runtime falls back to automatic placement — no regression.

**Stale annotation detection:** Annotations from a previous model version (e.g.,
model modified post-export) are detected when device counts or tensor shapes don't
match the current runtime topology. The runtime MUST NOT silently produce incorrect
results from stale hints — validation catches mismatches and falls back to automatic
placement with a diagnostic warning.

**Current status:** `onnx-std` validates; IR/loader do not yet propagate. Low priority
until real models with sharding annotations exist.

### D7: Host RAM and disk are per-machine shared resources, not per-device

**Decision:** `host_ram_limit` and `disk_spill_limit` are owned by the HostGovernor
(one per machine), not by individual DeviceGovernors.

**Rationale:** Host RAM and disk are physically shared across all devices on a machine.
If each of N devices independently manages a `host_ram_limit`, they collectively risk
claiming N× the available memory. A single HostGovernor with a global view prevents
this contention and provides fair cross-device arbitration.

### D8: MemoryTopology is trait-based, not a closed enum

**Decision:** `MemoryTopology` is a struct with trait-object fields
(`Vec<Arc<dyn DeviceGovernor>>` + `Arc<dyn HostGovernor>`) plus a `#[non_exhaustive]`
`TopologyKind` for logging/metrics. Upper layers never match on topology kind for
control flow — they use trait methods exclusively.

**Rationale:** A closed enum forces all match sites to update when a new hardware
topology appears. Trait objects let new topologies (e.g., `UnifiedGovernor`
implementing both `DeviceGovernor` and `HostGovernor`) slot in without changing
upper layers. The enum approach would break on every new device class (CXL memory,
new NPU architectures, disaggregated memory pools).

### D9: Distributed rendezvous is opt-in with token auth

**Decision:** Multi-device interconnect is off by default (`distributed.enabled: false`).
When enabled, genai-server acts as rendezvous server with pre-shared token
authentication. Default bind to `127.0.0.1`; multi-machine requires explicit
network configuration (`listen_addr` + `allowed_cidrs`).

**Security model:**

**Control plane (rendezvous, rank registration, topology exchange):**
- Requires TLS or mTLS for multi-machine deployments
- PSK token is transmitted only over encrypted channel
- Rank identity bound to session ID + topology epoch (prevents replay/replacement)
- Localhost binding for single-machine skips TLS (trusted loopback)

**Data plane (tensor transport):**
- NCCL/CUDA IPC: inherently local, no encryption needed
- Thunderbolt 5 DMA: physical link, no encryption needed
- TCP/RDMA over network: explicitly restricted to trusted isolated network segment
- If running on untrusted network, data plane requires transport-level encryption
  (performance tradeoff documented)

**Threat model explicitly covers:** interception, replay, rank replacement,
topology mismatch, and tensor-data exposure.

**Defense in depth:** opt-in + TLS/mTLS + token + binding + CIDR + epoch binding

```yaml
distributed:
  enabled: false  # default off
  rendezvous:
    listen: "127.0.0.1:18801"
    auth_token: null  # auto-generate if null when enabled
  # Multi-machine requires explicit configuration:
  # listen: "0.0.0.0:18801"
  # allowed_cidrs: ["10.0.0.0/24"]
  # auth_token: "user-provided-secret"
```

**Rationale:** Exposing a rendezvous endpoint without auth would allow unauthorized
rank registration and potential tensor injection. An attacker joining as a fake rank
could corrupt all-reduce results or exfiltrate model weights.

### D10: Model metadata hint namespace (`com.nxrt.hint.*`)

**Decision:** Runtime-advisory hints are stored in ONNX `metadata_props`
(`map<string, string>`) on `NodeProto` using a structured namespace:

```
com.nxrt.hint.{category}.{specific}
```

**Categories:**
- `placement` — device affinity, shard axis/count, pipeline stage
- `memory` — tier preference (hot/warm/cold), offload priority, residency hint
- `expert` — affinity group, activation frequency, prefetch window
- `compute` — preferred precision, kernel selection hints

**Per-config specialization (optional):**
```
com.nxrt.hint.config.{config_name}.{category}.{specific}
```

Allows the same model to carry different hints for different deployment scenarios.

**Example — same model, multiple configs:**
```
# Generic (default)
com.nxrt.hint.placement.shard_axis: "0"
com.nxrt.hint.placement.shard_count: "8"
com.nxrt.hint.expert.affinity_group: "routing_cluster_0"
com.nxrt.hint.expert.activation_frequency: "0.73"
com.nxrt.hint.memory.tier: "hot"

# 4×Mac Studio specialization
com.nxrt.hint.config.mac_cluster.placement.shard_count: "4"
com.nxrt.hint.config.mac_cluster.memory.tier: "unified"

# Single-GPU specialization
com.nxrt.hint.config.single_gpu.expert.prefetch_window: "2"
com.nxrt.hint.config.single_gpu.memory.offload_priority: "1"
```

**Resolution logic:**
```rust
fn resolve_hint(
    node_metadata: &HashMap<String, String>,
    key: &str,
    active_config: Option<&str>,
) -> Option<String> {
    // 1. Specialized config hint takes priority
    if let Some(cfg) = active_config {
        let specialized = format!("com.nxrt.hint.config.{cfg}.{key}");
        if let Some(v) = node_metadata.get(&specialized) {
            return Some(v.clone());
        }
    }
    // 2. Fallback to generic hint
    node_metadata.get(&format!("com.nxrt.hint.{key}")).cloned()
}
```

**Relationship with ONNX native sharding (D6):**
- ONNX `ShardingSpecProto` → structured TP/PP placement (protobuf fields)
- `com.nxrt.hint.*` → runtime-specific scheduling/memory/precision (metadata_props)
- Both coexist; ONNX spec handles what it standardizes, `com.nxrt.hint.*` handles
  everything else (expert affinity, precision hints, tier preferences, config
  specialization)

**Rationale:** Model exporters (Mobius, converter tools) already know deployment-
relevant information from profiling data. A structured namespace prevents ad-hoc
key proliferation, supports cross-scenario specialization without model duplication,
and keeps all hints advisory (runtime always has final say).

---

## 11. Phased Implementation

Unified across all design documents:

### Phase 1: Single-Session Weight Residency

*Maps to WEIGHT_OFFLOAD.md Phases 1-2.*

- `WeightRegionCatalog` classifies model regions (shared vs expert).
- `WeightResidencyManager` with cold/warm/hot tiers.
- `ExpertStore` facade for fused MoE kernels.
- Heat-based LRU admission for experts.
- Lease/pin lifecycle with completion fences.
- DeviceGovernor sub-budgets (KV vs expert) with hysteresis.
- DeviceGovernor is the first priority (already partially wired as `ResourceGovernor`).

### Phase 2: Governor Wiring + HostGovernor

*Maps to DESIGN.md §26.11.*

- Connect real EP/model weight usage, activation/scratch high-water marks, and
  ORT/EP allocations to the DeviceGovernor.
- `hot_expert_bytes` component in `VramBreakdown`.
- Coordinated KV + expert sub-budget rebalancing.
- Lowering-triggered live eviction (tiered: background → paused → running → interactive).
- Auto mode with real capacity detection from EP device queries.
- **HostGovernor wiring:** host RAM quota management, per-device usage tracking,
  cross-device arbitration for offload pages.
- DeviceGovernor → HostGovernor integration for VRAM eviction → host RAM offload flow.
- Exhaustively check `PressureProtocol.cfg`, then run deterministic
  grant/cancel/timeout/reconfigure/reclaim schedules through the independent
  refinement checker. Every successful ticket owns its exact charged bytes,
  every abandoned grant is released, and physical usage never exceeds the
  limit.
- Include multiple variable-sized tickets per device, fixed non-reclaimable
  charges, exact-capacity requests, and cancellation-mailbox saturation.
- Test simultaneous reclaim and allocation to prove no governor lock is held
  across await and priority/FIFO arbitration cannot starve an older request.
- Test discrete offload creates a new physical ID only after reservation, while
  unified offload preserves the ID and changes exactly one residency class.

### Phase 3: Multi-GPU Single-Node

- NCCL `Communicator` for multi-GPU collective ops.
- Shared weights via CUDA IPC (zero-copy across sessions).
- `ClusterCoordinator` Strategy 2 (shared weights + shared KV pool).
- Expert migration between GPUs based on heat.
- InProcess `Communicator` for testing.
- `MemoryTopology::Discrete` with multi-device HostGovernor arbitration.

### Phase 4: Cross-Node

- Thunderbolt 5 `Communicator` for Mac Studio cluster.
- RDMA `Communicator` for data center.
- `ClusterCoordinator` above per-node `HostGovernor`s (global cross-node role).
- Cross-node expert migration via Communicator.
- Cross-node prefix cache lookup.

> **Canonical naming:** `ClusterCoordinator` is the sole name for the
> cross-session/cross-node coordination layer. This document is canonical for
> memory ownership, governor hierarchy, and coordination policy.
> `DISTRIBUTED_RUNTIME.md` is canonical for communicator contracts, execution plan
> structure, and collective semantics.

---

## 12. Resolved Questions

All questions consolidated from source documents, with decisions.

### From weight residency / governor (WEIGHT_OFFLOAD.md, DESIGN.md)

1. **Auto mode completeness.** Auto mode must not be considered complete until real
   free/total RAM, filesystem, and device capacity are reported by the EP.
   **Decision:** Implementation detail. EP interface adds `query_device_info()`.
   Not an architectural question.

2. **Budget reporting fidelity.** Clean mapped pages (cold tier) are OS page cache,
   not owned bytes. How to distinguish in budget reporting?
   **Decision:** Implementation detail. Track clean mapped pages separately in
   budget snapshots. Not an architectural question.

### From distributed coordination (DISTRIBUTED_RUNTIME.md)

3. **Rendezvous mechanism.** How do distributed ranks discover each other?
   **Decision:** genai-server acts as rendezvous server. Off by default. Token
   auth + localhost binding + CIDR allowlist for multi-machine. Fallback to env
   vars for compatibility. See D9.

4. **Fault tolerance.** What happens when a rank crashes mid-collective?
   **Decision:** Phase 1–3: abort all ranks + restart. Optimize for fast reload:
   coordinator-owned immutable host weight mappings may survive, while device
   state and failed-group-owned allocations are rebuilt. Partial degradation is
   deferred to Phase 4+. The canonical execution failure state machine and
   ownership are defined in DISTRIBUTED_RUNTIME.md §8.5.

5. **Dynamic rank membership.** Can ranks join/leave a live session?
   **Decision:** Not needed. Topology fixed at startup. Elastic scaling via
   session group restart. **Closed.**

6. **Communicator selection.** When multiple backends are available, auto-select
   based on topology or user-configured?
   **Decision:** Auto-select based on hardware detection (NVLink → NCCL, TB5 →
   Thunderbolt, same process → InProcess). User can override via config.

7. **Quantized communication.** Send FP8/INT8 and up-cast at receiver to halve
   bandwidth?
   **Decision:** Phase 1 full precision. Phase 3+ add optional quantized wire
   formats. The sole `WireTensorSpec` and codec contract are defined in
   [DISTRIBUTED_RUNTIME.md §3.1](./DISTRIBUTED_RUNTIME.md#31-core-trait).
   Wire format is frozen during plan compilation; unsupported codecs fail
   compilation. Full precision uses `WireCodec::Identity` and zero error bound.

   Highest value for cross-node (TB5 40Gb/s bottleneck).

8. **CUDA IPC ownership semantics.** When session 0 allocates shared weights and
   sessions 1–7 map via IPC, who owns the lifecycle?
   **Decision:** genai-server manages shared weight lifecycle (reference counting).
   Session crash doesn't affect weights because genai-server outlives sessions.

9. **KV cache sharing granularity.** Different sessions may quantize KV differently
   (FP16 vs FP8). Enforce uniform format or support conversion at share boundaries?
   **Decision:** Enforce uniform format within a shared KV prefix pool. Different
   formats → different pools. Conversion at every cache hit is too expensive.

10. **ClusterCoordinator placement.** In genai-server process or separate daemon?
    **Decision:** Module inside genai-server. Separate daemon adds operational
    complexity with no benefit.

### From MoE / expert parallelism

11. **Expert-aware scheduling across sessions.** When multiple sessions share a device,
    should the governor prefer expert affinity?
    **Decision:** Model export tools (Mobius) annotate expert affinity groups at
    export time via `com.nxrt.hint.expert.affinity_group` (see D10). Runtime uses
    hints to seed scheduling; no hints → LRU fallback. Phase 3+.

12. **Prefetch speculation budget.** How many speculative prefetch bytes before the
    cost of wrong predictions exceeds the benefit?
    **Decision:** Start with 5–10% of device memory. Dynamic adjustment based on
    hit rate. Algorithm-managed, not user-configured.

### From governor split / topology (§4, §5, §9)

13. **HostGovernor pinned vs pageable allocation.** Should HostGovernor allocate pinned
    vs pageable host memory separately?
    **Decision:** Bounded pinned pool + pageable overflow. HostGovernor maintains
    a configurable pinned pool (default: 10% of host RAM).
    `HostMemoryClass::PinnedDma` is strict and fails rather than silently
    returning pageable memory; ordinary offload requests use `Pageable`.
    The pool is user-configurable and dynamically adjustable at runtime.

14. **Unified memory working set size.** How to define the GPU budget on unified
    memory devices?
    **Decision:** Three-tier fallback: (1) OS API available (Apple
    `recommendedMaxWorkingSetSize`) → use it; (2) no OS API →
    `total_memory * 0.75`; (3) user override in config → user wins.

15. **NPU DMA pinning.** How to prevent host pages being DMA’d by NPU from being
    evicted by GPU offload pressure?
    **Decision:** NPU requests `HostMemoryClass::PinnedDma`; awaiting the ticket
    returns a charged `HostAllocation` pin/lease. NPU holds it during DMA,
    eviction skips it, and release occurs after the DMA completion fence.
    **Closed.**

16. **CPU-only mode.** Should we instantiate a DeviceGovernor for CPU?
    **Decision:** No empty-shell DeviceGovernor. `MemoryTopology.devices` is empty.
    Upper layers check `devices.get(id)` — `None` means route through `host`.
    `TopologyKind::CpuOnly` is informational only. See D8.

---

## 13. References

- [DESIGN.md §26.11](./DESIGN.md) — Resource Governor: canonical design (stays in place)
- [DESIGN.md §43.2](./DESIGN.md) — MoE Expert Weights: "not KV cache" declaration
- [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md) — Three-tier weight residency (redirects here for §4)
- [MOE_SUPPORT.md](./MOE_SUPPORT.md) — First-class MoE support (redirects here for §7)
- [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) — Session-per-GPU MoE architecture (DispatchTransport deprecated)
- [DISTRIBUTED_RUNTIME.md](./DISTRIBUTED_RUNTIME.md) — Communicator abstraction & multi-device inference
- [SCHEDULING.md](./SCHEDULING.md) — Adaptive scheduling, EP negotiation protocol
- `crates/onnx-runtime-ep-api/src/provider.rs` — ExecutionProvider trait
- `crates/onnx-genai-scheduler/src/governor.rs` — DeviceGovernor implementation (originally ResourceGovernor)