# Cross-arch decode-kernel optimizations — learnings from TensorRT-LLM, FLA, causal-conv1d

**Authors:** research agents `sota` (TensorRT / TensorRT-LLM deep-dive) + `flakernels`
(flash-linear-attention / causal-conv1d survey); synthesized by Squad coordinator.
**Date:** 2026-08-15
**Purpose:** Persist the adoptable kernel techniques and the per-architecture
(A100 / Hopper / Blackwell) optimization catalog so future work — including agents
on other machines and future hardware tiers — can pick this up without re-deriving it.
**Hardware reality at time of writing:** development + all live benchmarks on 8×H200
(Hopper, `sm_90`). **No A100 or Blackwell hardware available** — the A100/Blackwell
sections below are built from *known/published* optimizations; they are arch-guarded,
compile-validated on H200, and ready to benchmark when hardware lands.

> **Headline (read first).** Our M=1 int4 decode kernels are **issue/latency-bound,
> not bandwidth-bound** (ncu: L1TEX scoreboard ~61%, DRAM ~2% of peak). That means
> the highest-value lever is **reducing instruction count in the dequant path**, not
> adding bandwidth/occupancy tricks. TensorRT-LLM's weight-only int4 GEMV does exactly
> this and is **measurably beyond ORT**: an offline weight interleave + bias-baking
> that lets the runtime dequant 8 int4→fp16 in **~9 instructions vs our ORT-style ~14**
> (~35% fewer). This technique is **architecture-independent** — a win carries to
> A100, Hopper, and Blackwell alike — so it is the correct *first* lever, before any
> per-arch specialization.

---

## 0. Binding constraint (do not violate)

Beat stock ORT under **equal, non-speculative base-decode conditions first**.
Speculative decoding is additive-on-top only and may never be used to *claim* an
ORT win. Every technique below is evaluated against base decode with byte/fp16
parity as a hard correctness gate.

---

## 1. The core insight: dequant instruction count is the bottleneck

Prior incremental levers (GEMV occupancy, GQA attention warp/tile/cp.async probes)
hit an **ncu-proven floor** because they attacked the wrong axis. The M=1 decode
GEMV reads little DRAM (2% of peak) and stalls on issue slots / long-scoreboard
global-load latency. The arithmetic that *does* run — int4→fp16 dequant — is
therefore the thing to shrink.

### 1.1 TensorRT-LLM offline weight preprocessing (the secret)

Source: `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp`
(`add_bias_and_interleave_int4s_inplace`) — a host-side, run-once transform:

- **Bias-baking:** add `+8` to every int4 nibble at load time, mapping signed
  `[-8,7]` → unsigned `[0,15]`. The kernel then **never subtracts 8 at runtime**.
- **Nibble interleave:** rearrange a 32-bit word from natural order
  `[e7 e6 e5 e4 | e3 e2 e1 e0]` to `[e7 e5 e3 e1 | e6 e4 e2 e0]` (even nibbles →
  low 4-bit positions, odd → high). This pre-positions elements so the runtime
  dequant needs **no `prmt.b32` activation reorder**.

### 1.2 Runtime converter — `FastInterleavedAndBiasedNumericArrayConverter<half, uint4b_t, 8>`

Source: `cpp/tensorrt_llm/cutlass_extensions/.../interleaved_numeric_conversion.h`.
Converts 8 interleaved uint4 → 8 fp16 from **one** 32-bit register:

```
1 shift  (top_i4s = i4s >> 8)
4 lop3.b32   (immLut 0xaa; masks 0x000f000f / 0x00f000f0; magic 0x64006400)
2 sub.f16x2  (0x64086408)   — even-nibble lanes: remove bias + fp16 exp offset
2 fma.f16x2  (0x2c002c00, 0xd480d480) — odd-nibble lanes: ×(1/16) position fix
= 9 instructions for 8 fp16 values
```

The per-group fp16 **scale is applied separately** (hmul2) after dequant, so the
instruction-count win is independent of the quantization scale. Our ORT-style path
spends ~14 instructions per 8 values (extra `prmt.b32` + an extra subtract) — the
offline interleave is what removes them.

**Status:** being ported behind an opt-in flag (`ONNX_GENAI_INTERLEAVE_DEQUANT`),
byte-parity gated, benchmarked on H200 (`matmul_nbits` GEMV, ~32% of decode).
Kept opt-in until proven — glm base perf is a knife-edge.

---

## 2. Per-architecture optimization catalog

The TRT-LLM / FLA design lesson is that decode kernels are **templated per SM
version** — one dispatch, different primitives and tile sizes per arch. Our plan
mirrors that with a runtime `sm_XX` query → kernel/tile selection.

| Arch | SM | Async-copy | Tensor/MMA | FLA decode tiles | Bandwidth | Notes |
|---|---|---|---|---|---|---|
| **A100 (Ampere)** | `sm_80` | `cp.async` (LDGSTS) — **no TMA** | 3rd-gen (fp16/bf16, no fp8) | BS=32, BV=128 | HBM2e ~2.0 TB/s | See §2.1 |
| **Hopper (H200)** | `sm_90` | TMA + `cp.async` | 4th-gen wgmma, fp8 | BS=64, BV=256 | HBM3e ~4.8 TB/s | Have hardware; §2.2 |
| **Blackwell DC (B200/GB200)** | `sm_100` | bigger TMA, 2-SM MMA | 5th-gen `tcgen05`, native NVFP4/FP6 | larger | HBM3e ~8 TB/s | §2.3 |
| **Blackwell consumer (RTX 50/GB202)** | `sm_120` | TMA | 5th-gen, **no 2-SM MMA** | needs own tuning | GDDR7 (≠HBM) | §2.4 |

### 2.1 A100 (`sm_80`) — known Ampere optimizations

- **`cp.async` KV-prefetch is expected to be a NET WIN here** (unlike Hopper).
  We tested a multi-stage `cp.async` KV pipeline on Hopper and it was NO-GO —
  the register/smem cost offset the latency-hiding *because TMA was a cheaper
  alternative*. **A100 has no TMA**, so `cp.async` (LDGSTS) is the *only* async
  path and its latency-hiding should pay off. Guard the pipeline to `sm_80`.
- Smaller decode tiles (BS=32/BV=128 per FLA) to fit Ampere's smaller L2 / smem.
- No PDL (see §3) — Ampere lacks programmatic dependent launch.
- 3rd-gen tensor cores: fp16/bf16 only; no fp8/fp4, so int4-weight × fp16-act
  SIMT GEMV (with the §1 interleave dequant) remains the decode path.

### 2.2 Hopper (`sm_90`) — have hardware, floor reached on current kernel

- TMA + wgmma already used; GQA decode at an ncu-proven register/smem/occupancy
  Pareto frontier (see `docs/research/decode-remaining-levers-feasibility.md`).
- **Untapped: PDL launch-overlap** (§3) and the §1 interleave dequant.

### 2.3 Blackwell datacenter (`sm_100`) — known optimizations (no hardware)

- **5th-gen tensor cores + `tcgen05` MMA + native NVFP4/FP6.** The large lever is
  native FP4: to use the 5th-gen path for decode we would need an **FP4 activation
  quant** path (our weights are int4, activations fp16). Until then the likely
  Blackwell win is bigger TMA tiles + `tcgen05` for the prefill/GEMM path.
- **2-SM MMA** (thread-block-cluster pairs sharing an MMA) — a `sm_100`-only
  capability; scaffold behind arch guard, do not assume on consumer Blackwell.
- Bigger HBM3e bandwidth (~8 TB/s) shifts the roofline; if the §1 dequant win
  pushes us toward bandwidth-bound, Blackwell benefits more than Hopper.

### 2.4 Blackwell consumer (`sm_120`) — known optimizations (no hardware)

- Same 5th-gen ISA family as `sm_100` but **no 2-SM MMA**, and **GDDR7 instead of
  HBM3e** — a very different bandwidth/latency profile. Decode tiles need
  **separate tuning** from datacenter Blackwell; do not reuse `sm_100` tile sizes.
- Relevant to the existing `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`
  consumer-GPU work — cross-reference when that lands on hardware.

---

## 3. PDL — Programmatic Dependent Launch (`sm_90+`)

Source: TRT-LLM GEMV kernels use `cudaGridDependencySynchronize()` /
`cudaTriggerProgrammaticLaunchCompletion()`. PDL lets the **next** kernel's launch
overlap with the **current** kernel's tail execution. Decode is a long chain of
tiny per-layer kernels, so this directly attacks **launch/latency** on the critical
path — the same axis our per-op profiling flags as dominant. Available on Hopper
**and** Blackwell (not Ampere). Testable on our H200 today.

---

## 4. FLA — linear attention: perf upgrade to a LIVE kernel

**Linear attention is a current optimization target in this repo, not future work.**
We already run the Qwen3.5 / Qwen3-Next hybrid family via
`crates/onnx-runtime-ep-cuda/src/kernels/linear_attention.rs` —
`com.microsoft::LinearAttention`, a gated delta-rule (Gated DeltaNet) kernel with a
per-head recurrent state matrix, a faithful CUDA port of ORT's `linear_attention.cc`
(design note: `.squad/decisions/inbox/cohaagen-linear-attention-design.md`).

**Important:** the current kernel is a *correctness-first port* — **one thread owns
one state column**, keeping the whole recurrent scan in per-thread f32 to reproduce
ORT's `float` kernel exactly. That is the right correctness baseline but leaves
performance on the table. FLA is the **perf reference** for the upgrade:

- `fla/ops/attn/decoding.py` (`naive_attn_decoding_kernel`) — single-pass
  online-softmax decode with **arch-tiered tiles** (Hopper BS=64/BV=256, Ampere
  BS=32/BV=128) and a running `(m, acc, o)` triplet over KV-cache tiles. The
  arch-tiering pattern applies directly to our decode kernels.
- `fla/ops/common/fused_recurrent.py` / `fla/ops/kda/fused_recurrent.py` — chunked /
  tiled recurrent (Gated-DeltaNet-style) kernels with per-dim gating, `T=1` decode
  path, and speculative-decode/continuous-batch hooks (`ssm_state_indices`,
  `num_accepted_tokens`). These are the structural template for a **higher-occupancy
  linear-attention decode** than our current one-thread-per-column scan — while
  preserving the f32-state numerics as the parity gate.
- FLA has **no dedicated MLA decode kernel** (its MLA layer delegates to
  FlashAttention and caches full expanded KV — see the `# TODO: only cache
  compressed_kv` note in `fla/layers/mla.py`), so it does not solve DeepSeek-V2 MLA;
  our own compressed-KV path is still needed there.

---

## 5. causal-conv1d — perf upgrade to our LIVE short-conv kernel

**Also a live path, not future work.** The hybrid linear-attention models pair the
gated delta-rule with a causal depthwise short conv, which we already run via
`crates/onnx-runtime-ep-cuda/src/kernels/causal_conv_with_state.rs` —
`com.microsoft::CausalConvWithState` (Mamba / linear-attention "short conv"), a
faithful CUDA port with a rolling `[B, C, K-1]` state cache. Like §4 it is a
correctness port: **one thread per `(b, c)` row**, f32 accumulation.

`Dao-AILab/causal-conv1d`'s `csrc/causal_conv1d_update.cu` is the **perf reference**:
a decode-step kernel with **64 threads/channel** (vectorized `kNElts` loads),
sliding `(kWidth-1)` state buffer, both linear and **circular-buffer** state layouts,
continuous-batching via `conv_state_indices_ptr`, and optional fused SiLU. Its
`ConvParamsBase` struct (`csrc/causal_conv1d.h`) is the canonical parameter layout to
mirror. Arch-independent; a direct upgrade to our existing kernel with the f32
pre-activation values as the parity gate.

---

## 6. Adoption status & sequencing

1. **[in progress]** §1 interleave + bias-baked dequant — arch-independent, opt-in
   flag, byte-parity gated, benchmarked on H200. Highest value; carries to all archs.
2. **[queued]** SM-version dispatch scaffolding (§2) — the templated-per-arch pattern.
3. **[queued, no hw]** §2.1 A100 `cp.async` KV pipeline — expected net-positive on
   Ampere; compile-validate on H200, benchmark when hardware lands.
4. **[queued]** §3 PDL launch-overlap — Hopper + Blackwell; testable on H200.
5. **[queued, no hw]** §2.3/§2.4 Blackwell `tcgen05`/NVFP4 — feasibility + scaffold
   from docs; needs FP4 activation path to fully exploit.
6. **[live kernels, perf upgrade]** §4/§5 linear-attention (Gated DeltaNet,
   Qwen3.5/Qwen3-Next) and causal short-conv — already ported for correctness
   (one thread per column/row, f32 state); FLA chunked-recurrent + causal-conv1d's
   64-thread/channel update kernel are the perf upgrades, gated on f32 parity.

---

## References

- Adoption plans (session artifacts): `trt-llm-port-plan.md`,
  `flakernels-adoption-plan.md` (research agent full reports).
- `docs/research/decode-remaining-levers-feasibility.md` — the ncu-proven decode floor.
- `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md` — consumer-GPU audit.
- TensorRT-LLM: `github.com/NVIDIA/TensorRT-LLM`
  (`kernels/weightOnlyBatchedGemv/`, `cutlass_kernels/fpA_intB_gemm/`,
  `cutlass_kernels/cutlass_preprocessors.cpp`, `cutlass_extensions/.../interleaved_numeric_conversion.h`).
- flash-linear-attention: `github.com/fla-org/flash-linear-attention`
  (`fla/ops/attn/decoding.py`, `fla/layers/mla.py`).
- causal-conv1d: `github.com/Dao-AILab/causal-conv1d` (`csrc/causal_conv1d_update.cu`).
