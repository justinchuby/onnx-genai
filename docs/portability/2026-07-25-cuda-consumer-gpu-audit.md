# CUDA EP consumer-GPU portability audit — 2026-07-25

**Author:** Deckard (Squad worker) · **Requested by:** @justinchuby
**Branch:** `audit/cuda-consumer-gpu-portability` (off `origin/main` 6a64ee3c)
**Scope:** `crates/onnx-runtime-ep-cuda/src`

## Directive

> "Our optimizations must not target ONLY H200 — most people run consumer chips."
> Every CUDA EP kernel must (a) **run correctly** on consumer NVIDIA GPUs
> (RTX 30-series = sm_86, RTX 40-series = sm_89, plus sm_80/sm_75) and (b) tune
> from **device properties** (SM count, compute capability, L2, shared-memory
> limits), not hardcoded H200 (sm_90, 132 SM, ~227 KB opt-in shared memory)
> constants.

## Headline finding — the build is NOT H200-only

**The CUDA EP ships no ahead-of-time cubin and hardcodes no target architecture.**
Every device kernel is a CUDA-C source string compiled at runtime through NVRTC,
targeting **the compute capability of the GPU actually present**:

* `crates/onnx-runtime-ep-cuda/src/runtime.rs:60-66` derives the compile target
  from the live device: `ptx_arch = compute_{major}{minor}` (virtual, JIT
  forward-compatible) and `cubin_arch = sm_{major}{minor}`.
* `runtime.rs:211-253` queries `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR`
  and the shared-memory / SM-count attributes from the driver at init.
* `runtime.rs:502-556` compiles each module with
  `--gpu-architecture=compute_XX` (PTX, so the driver JITs forward to the real
  SM) and only falls back to a device-native `sm_XX` **cubin** if the installed
  NVRTC emits a PTX ISA the driver rejects (`nvrtc_cubin_fallback`).

There is therefore **no `build.rs`, no `-arch`/`-gencode` flag list, and no
embedded sm_90 binary**. A consumer sm_86/sm_89/sm_80/sm_75 card compiles the
exact same source to its own ISA. This is the single most important portability
property and it is already correct.

## Priority 1 — correctness portability

### 1. sm_90-only device features — NONE present

Grepped every kernel source string for `wgmma`, `cp.async`/`cp.async.bulk`/TMA,
thread-block clusters (`cluster`, `__cluster_dims__`, `cudaLaunchKernelEx`),
`mbarrier`, `tcgen05`, `st.async`, `setmaxnreg`, and `__CUDA_ARCH__ >= 900`.
**Zero hits in kernel code.** `qmoe.rs:2109` even asserts the QMoE GEMM source
does not contain `sm_90`. The inline PTX asm used (`lop3.b32`, `sub.f16x2`,
`fma.rn.f16x2`, `prmt.b32` in `matmul_nbits.rs:811-925`) is available from
sm_53 upward, so it compiles and runs on every target card.

### 2. Tensor-core / dtype guards — present with fallbacks

| Path | Guard | Fallback on consumer / old card |
|------|-------|--------------------------------|
| Flash-attention fp16 tensor-core | `compute_capability().0 >= 7` (`flash_attention.rs:528`) | non-tensor-core fp16/fp32 flash path |
| Fused Attention softmax half | `compute_capability().0 >= 7` (`attention.rs:605`, `group_query_attention.rs:1231`) | fp32 softmax path |
| QMoE GEMM tile | `qmoe_gemm::tile_for(compute_capability, …)` (`qmoe.rs:1585`) | smaller tile / general path |

### 3. Shared-memory limits — reduction & QMoE already clamp; 3 GEMV sites hardened here

Non-opt-in dynamic shared memory is capped at ~48 KB on **every** architecture;
the opt-in ceiling is device specific (~100 KB sm_86/sm_89, ~163 KB sm_80,
~227 KB sm_90). A launch that requests >48 KB without setting
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` fails on **any** GPU; one that
requests more than the device's opt-in ceiling fails on that specific card.

| Kernel | Shared-mem request | Clamped to device opt-in? | Status |
|--------|--------------------|---------------------------|--------|
| Reduction launches (`reduction_launch_config`, `runtime.rs:421`) | `threads × bytes_per_thread`, threads chosen from `max_shared_memory_per_block_optin` | yes, + opt-in attribute set | already correct |
| QMoE GEMM (`qmoe_gemm::tile_for`) | tile picks largest that fits `max_shared_memory_per_block_optin` | yes, with fallback to smaller tile | already correct |
| GQA decode f32/f16 (`gqa_decode.rs:386`, `gqa_decode_fp16.rs:459`) | `warps×(2+head_dim)×4` ≈ ≤ 8 KB | n/a (well under 48 KB) | safe |
| **Fused RMSNorm/gate-up/int8 decode GEMV** (`matmul_nbits.rs:3890`, `:4343`, `:4740`) | **`K × sizeof(f16)`** (stages the normalized activation) | **was NOT — fixed in this change** | **hardened** |

The three fused GEMV launches stage the whole `K`-vector in dynamic shared
memory. For `K ≤ 24576` this is ≤ 48 KB and launches everywhere; for larger `K`
it exceeded the non-opt-in budget and would have **launch-crashed on any GPU**,
and beyond the opt-in ceiling it is unsatisfiable on a consumer card even though
an H200 could serve it. These sites previously set no opt-in attribute and did
no bounds check.

### 4. Compute-capability floor

`runtime.rs:227-231` rejects a device reporting compute-capability major 0. The
kernels themselves compile for whatever cc the device reports; the effective
floor for the inline-asm/f16 paths is sm_53, comfortably below the sm_75 target.

## Priority 2 — perf portability (overfit check)

Every launch-config / kernel-selection heuristic was classified:

| Heuristic | Location | Device-adaptive? |
|-----------|----------|------------------|
| Down-projection columns-per-CTA (8/4/2) | `select_down_columns` `matmul_nbits.rs:2852` | **yes** — target = `multiprocessor_count × DOWN_FILL_CTAS_PER_SM` |
| QMoE grouping saturation | `qmoe.rs:1814,1831` | **yes** — `multiprocessor_count × 16` |
| QMoE GEMM tile | `qmoe_gemm::tile_for` | **yes** — cc + opt-in shared memory |
| GQA-fp16 split-K grid | `gqa_decode_fp16.rs:446` | **yes** — `multiprocessor_count()` |
| Tensor-core / softmax dtype route | flash/attention/GQA | **yes** — `compute_capability()` |
| Split-K factor `K_SPLIT = 2` | `matmul_nbits.rs:632,1386` | fixed 2× grid multiply — helps low-SM cards too (more CTAs), not H200-overfit; gated on shape only |
| Block sizes (256 threads), GEMM tile widths | various | architecture-neutral constants (occupancy-safe on sm_75+); not SM-count tied |

The literal `132` appears **only in test assertions** (`matmul_nbits.rs:6235+`),
never in production dispatch. The `DOWN_FILL_CTAS_PER_SM = 12` comment cites an
H200 measurement, but the value multiplies the live SM count, so the grid scales
down correctly on a 46-SM RTX 4070 or 28-SM RTX 3060. **No production heuristic
was found hardcoded to H200 geometry**, so no Priority-2 refactor was required —
the `#148` `select_down_columns` pattern is already used pervasively.

## Fixes applied in this change

1. **New shared helper** `CudaRuntime::configure_dynamic_shared_memory`
   (`runtime.rs`): validates a dynamic shared-memory request against the device
   budgets, opts the function into >48 KB when the card allows it, and returns a
   **loud error (never launches)** when the request exceeds the device opt-in
   ceiling, so a consumer card fails safely instead of launch-crashing.
   Backed by the pure, unit-tested core `dynamic_shared_memory_optin`.
2. **Applied the clamp** at the three fused-GEMV launch sites in
   `matmul_nbits.rs` (int8 RMS-norm-prologue GEMV, gate/up SwiGLU RMS-norm GEMV,
   fp16 RMS-norm-prologue GEMV). For the existing tested regime (`K ≤ 24576`)
   the helper returns "no opt-in needed" and the launch is **bit-identical** to
   before — this is a grid/launch-attribute-only guard, accumulation order is
   untouched.
3. **Unit test** `dynamic_shared_memory_optin_respects_device_budgets` covering
   fits-default, opt-in-needed, and reject-on-consumer cases.

## Needs real consumer-GPU validation

The audit was run on H200 (the only hardware available). The following want a
consumer card (RTX 30/40) to confirm at runtime:

* **Opt-in shared-memory path on sm_86/sm_89 (100 KB ceiling).** The new clamp
  logic is unit-tested and the H200 path is exercised, but the >48 KB opt-in and
  the >100 KB reject branch have not been executed on a real 100 KB card. Needs a
  model with `K > 24576` on a gate/up/qkv projection to hit it.
* **`DOWN_FILL_CTAS_PER_SM = 12` (~2-wave target) on low-SM cards.** Tuned on
  H200 (132 SM); the grid scales with SM count, but the *optimal* wave target on
  a 28–46 SM card is unverified. Documented in `matmul_nbits.rs:2828`.
* **`K_SPLIT = 2` split-K** benefit on low-SM cards (expected positive, unmeasured).

## Blocking gaps

**None.** No kernel is sm_90-only, and no path lacked a runtime guard that would
let it launch-crash on a consumer card after this change. The one real exposure
(fused-GEMV shared-memory overrun for very large `K`) now fails loudly and routes
the caller to error handling rather than crashing the launch.

## Validation performed (H200 / GPU0)

* `cargo build --release -p onnx-runtime-ep-cuda --features cuda` — green.
* `cargo build --release -p onnx-genai-bench --features bench-native,bench-ort,cuda --bin profile_native` — green.
* `cargo test --release -p onnx-runtime-ep-cuda --features cuda --lib` — **215 passed, 0 failed** (incl. the new smem test).
* `cargo test --release -p onnx-runtime-ep-cuda --features cuda` — only the two
  **pre-existing** `int8_block32` GPU tolerance failures
  (`..._default_zero_point_matches_cpu`, `..._explicit_zero_points_match_cpu`),
  which also fail on clean `main`. **No new failures introduced.**
* No-regression on H200: the clamp is a launch-attribute-only guard that is a
  no-op for every currently-tested shape (`K ≤ 24576`), so decode output is
  bit-identical (the block-32 capture/replay bit-exact tests pass unchanged).
