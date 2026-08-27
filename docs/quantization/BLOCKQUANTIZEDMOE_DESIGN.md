# `pkg.nxrt::BlockQuantizedMoE` — Operator Design Note (mixed-projection ABI)

**Status:** Mixed-projection ABI — CPU oracle and native CUDA decode/prefill
execution implemented for the verified GLM-5.2 UD-IQ1_S formats. This
supersedes the frozen single-`format` v1 ABI: each projection (`fc1`/`fc2`/`fc3`)
now carries its **own** native block format, because real GLM-5.2 GGUF routed
experts pack gate/up and down at different qtypes and block widths. There is no
backward-compatibility layer — the old single-`format` attribute is removed.
**Author:** Ripley (architect); mixed-projection revision by Sebastian (systems/perf)
**Date:** 2026-07-18; revised 2026-08-27
**Scope:** ABI, dispatch semantics, decode reuse, determinism, shape inference,
CPU+CUDA staging plan. This op is P1 #6, the largest practical GLM blocker
([GLM_READINESS_GAPS.md:259-268](../models/GLM_READINESS_GAPS.md#L259-L268)).

This note deliberately does **not** invent contracts. Where a detail is
genuinely undefined it is deferred to
[Decisions for Justin](#decisions-for-justin). A prior agent was rejected for
guessing `GatherBlockQuantized` shapes; that failure mode is respected here
([custom_ops.rs:1-6](../../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L1-L6)).

---

## 0. Context and positioning

### 0.1 Verified UD-IQ1_S checkpoint inventory

Inventory was read from the six official GGUF shards in place on 2026-08-27;
no model bytes were copied into the repository. The checkpoint has 1809 tensors:
`F32=709, IQ1_S=106, IQ2_XXS=44, IQ3_XXS=71, IQ4_XS=4, Q2_K=2,
Q3_K=1, Q4_K=2, Q5_K=312, Q6_K=82, Q8_0=476`.

Routed MoE layers are 3–78 with `E=256`, `H=6144`, and `I=2048`.
Gate/up tensors have GGUF dimensions `(6144, 2048, 256)` and down tensors
`(2048, 6144, 256)`. Their exact layer combinations are:

| gate/up | down | layer count |
|---|---|---:|
| IQ1_S | IQ3_XXS | 53 |
| IQ2_XXS | IQ3_XXS | 18 |
| IQ2_XXS | IQ4_XS | 4 |
| Q2_K | Q3_K | 1 (layer 78) |

Dense layers 0–2 use Q5_K gate/up and Q6_K down. Shared experts use Q5_K
gate/up and Q6_K down except layer 8, which uses Q6_K gate/up and Q8_0 down.
The authoritative serialized block geometries used by both oracles are
IQ1_S `(256,50)`, IQ2_XXS `(256,66)`, IQ3_XXS `(256,98)`,
IQ4_XS `(256,136)`, Q2_K `(256,84)`, Q3_K `(256,110)`,
Q5_K `(256,176)`, Q6_K `(256,210)`, and Q8_0 `(32,34)`.

The opt-in real-block tests preserve representative evidence with absolute
GGUF file offsets: layer 56 IQ1_S/IQ3_XXS in shard 5 at
`2668820832`/`1425373536`; layer 74 IQ2_XXS/IQ3_XXS in shard 6 at
`3071455584`/`1828008288`; layer 8 IQ2_XXS/IQ4_XS in shard 2 at
`17617400576`/`15892755200`; and layer 78 Q2_K/Q3_K in shard 6 at
`16942690656`/`15548248416`. These are `GGUFReader` tensor `data_offset`
values and are already absolute file offsets.

GLM's routed MoE tensors are dominated by codebook/IQ block formats
(IQ1_M / IQ2_XXS / IQ3_XXS), which the affine-integer `QMoE` kernel **cannot
represent**, while the portable float export evaluates **every** expert
([GLM_READINESS_GAPS.md:261-263](../models/GLM_READINESS_GAPS.md#L261-L263)). This op
fills exactly that gap: QMoE's expert structure, but with the native block
formats already decoded by `BlockQuantizedMatMul`, plus **selected-expert
dispatch** (evaluate only routed experts).

It is implemented on the CPU reference backend and native CUDA backend and is
the sole boundary the lazy initializer/offload seam already recognizes
([GLM_READINESS_GAPS.md:166](../models/GLM_READINESS_GAPS.md#L166),
[weight.rs:94-104](../../crates/onnx-runtime-ep-api/src/weight.rs#L94-L104)).
Expert-indexed device residency remains the missing part of the ordered
smoothness plan:
`BlockQuantizedMoE → selected-token IndexShare DSA → liveness planner → MTP`
([GLM_READINESS_GAPS.md:344-347](../models/GLM_READINESS_GAPS.md#L344-L347)).

Three existing contracts are the load-bearing precedents this ABI mirrors:

1. **QMoE CPU kernel** — input ordering, activation/SwiGLU semantics, and the
   route-first offload path
   ([qmoe.rs:103-160](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L103-L160),
   [qmoe.rs:307-420](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L307-L420)).
2. **`BlockQuantizedMatMul`** — the native IQ/MXFP4 block decoders to reuse
   verbatim ([block_quantized_matmul.rs:51-141](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L51-L141)).
3. **The lazy weight seam** — `LazyWeightBoundary::BlockQuantizedMoe` and the
   `LazyDeviceWeightBinder` trait
   ([weight.rs:94-104](../../crates/onnx-runtime-ep-api/src/weight.rs#L94-L104),
   [weight.rs:205-241](../../crates/onnx-runtime-ep-api/src/weight.rs#L205-L241)).

---

## 1. Operator ABI (mixed-projection)

The CPU reference oracle implements this schema. The positional inputs are
unchanged from the frozen v1 layout; the **attribute surface changed**: the
single `format` attribute is replaced by three per-projection format attributes
`fc1_format`, `fc2_format`, `fc3_format` (see §1.3). This is a breaking change
with no compatibility shim.

- **Domain:** `pkg.nxrt`
- **Op type:** `BlockQuantizedMoE` (exact string the seam matches at
  [weight.rs:101-105](../../crates/onnx-runtime-ep-api/src/weight.rs#L101-L105))
- **Version:** 1

### 1.1 Weight-representation deviation from QMoE (the key structural change)

`QMoE` carries **separate `*_scales` (and optional `*_zero_points`) inputs**
because affine-int dequant is `w = (q - zp) * scale`
([qmoe.rs:120-150](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L120-L150)).
IQ/MXFP4 blocks are **self-describing**: each serialized llama.cpp block embeds
its own scale bytes, and the codebook grids are **compile-time constants**, not
tensors ([block_quantized_matmul.rs:11-13](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L11-L13),
[lib.rs:7](../../crates/onnx-runtime-quantization/src/lib.rs#L7)). `BlockQuantizedMatMul`
therefore has **no scales input and no codebook input** — only `packed_B` with
layout `[N, blocks, block_bytes]`
([block_quantized_matmul.rs:202-207](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L202-L207)).

**Consequence for v1:** `BlockQuantizedMoE` **drops** the QMoE `*_scales`,
`*_zero_points`, and any codebook inputs. Each expert weight tensor is a single
packed `uint8` blob per projection, expert-major. This is the deliberate,
justified deviation from the QMoE precedent — its reason is the block format
itself, not a stylistic choice.

### 1.2 Input list

Ordering mirrors QMoE positions 0/1/2/(bias)/(down)/(bias)/(gate)/(bias) so a
graph author can transcode a QMoE node by swapping weight tensors and dropping
scales. `packed_*` tensors are `uint8`; all float tensors are `float32`
(f32 is the shortest native profile,
[GLM_READINESS_GAPS.md:172-174](../models/GLM_READINESS_GAPS.md#L172-L174)).

| Idx | Name | Type | Req? | Shape | Mirrors QMoE |
|---:|---|---|---|---|---|
| 0 | `input` (hidden states) | f32 | **required** | `[rows, H]` or `[B,S,H]` | idx 0 ([qmoe.rs:107](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L107)) |
| 1 | `router_logits` | f32 | **required** | `[rows, E]` | idx 1 ([qmoe.rs:108,298](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L108)) |
| 2 | `fc1_experts_weights` (gate/up, packed) | u8 | **required** | `[E, fc1_out, blocks1, block_bytes]` | idx 2 |
| 3 | `fc1_experts_bias` | f32 | optional | `[E, fc1_out]` | idx 4 (bias) |
| 4 | `fc2_experts_weights` (down, packed) | u8 | **required** | `[E, H, blocks2, block_bytes]` | idx 5 |
| 5 | `fc2_experts_bias` | f32 | optional | `[E, H]` | idx 7 |
| 6 | `fc3_experts_weights` (separate gate, packed) | u8 | optional | `[E, inter, blocks3, block_bytes]` | idx 8 |
| 7 | `fc3_experts_bias` | f32 | optional | `[E, inter]` | idx 10 |
| 8 | `router_weights` (aggregation weights) | f32 | optional | `[rows, E]` | idx 14 ([qmoe.rs:293-294](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L293-L294)) |

Notes:

- **Contiguous packing chosen over QMoE's sparse layout.** QMoE interleaves
  bias/scales/zero_points across indices 2–14 with FP4/FP8 modes reserved at
  15+ ([qmoe.rs:130-160](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L130-L160)).
  Because IQ formats carry no scales/zero_points, v1 has no such reserved gaps;
  a dense 0–8 ordering is clearer. This is a deviation flagged for sign-off
  (Decision 1).
- `fc1` follows the QMoE/MoE convention: when SwiGLU is **fused**,
  `fc1_out = 2 * inter`; otherwise `fc1_out = inter`
  ([moe.rs:287-298](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L287-L298)).
  `fc3` is present only for **unfused** SwiGLU / gated-GLU, exactly as QMoE
  gates it ([qmoe.rs:254-291](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L254-L291)).
- Each projection carries its **own** native block format: `fc1_format`
  (gate/up), `fc2_format` (down), and `fc3_format` (the separate gate of an
  unfused SwiGLU / gated-GLU, present iff `fc3_experts_weights` is wired). The
  formats are independent. This is required by real GLM-5.2 UD-IQ1_S routed
  experts (arch `glm-dsa`, 1809 tensors), whose per-layer combinations are, from
  the on-disk GGUF header metadata:

  | gate/up (`fc1_format`) | down (`fc2_format`) | layers |
  |---|---|---|
  | IQ1_S | IQ3_XXS | 53 |
  | IQ2_XXS | IQ3_XXS | 18 |
  | IQ2_XXS | IQ4_XS | 4 |
  | Q2_K | Q3_K | 1 |

  Gate always equals up; down always differs from gate/up. The single-`format`
  v1 ABI could not represent any of these layers. The lone `Q2_K`/`Q3_K` routed
  layer and shared/dense `Q5_K`/`Q6_K`/`Q8_0` projections are supported by the
  same property-driven decoder.
- Each projection's packed shape is `[E, N, blocksN, block_bytesN]` where
  `blocksN = KN / qk(fmt)` and `block_bytesN = block_bytes(fmt)` for that
  projection's own `fmt`. Partial block tails are rejected.
  ([block_quantized_matmul.rs:84-112](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L84-L112)),
  where `KN` is the input-feature width of that projection
  (H for fc1/fc3, inter for fc2). All byte offsets, per-row and per-expert
  strides are derived from one property-typed `ProjectionLayout` contract shared
  by the schema validator, the CPU oracle, and the CUDA claim gate.

### 1.3 Attributes

All attributes reuse the QMoE/`MoeAttributes` parse contract
([moe.rs:57-107](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L57-L107))
except the per-projection format attributes and `block_layout_version`, which
reuse `BlockQuantizedMatMul`'s `BlockFormat::parse`
([block_quantized_matmul.rs:155-169](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L155-L169)).

| Attribute | Type | Default | Source precedent |
|---|---|---|---|
| `fc1_format` | string enum | *required* | `BlockFormat::parse` — IQ formats plus `q2_k, q3_k, q5_k, q6_k, q8_0` ([block_quantized_matmul.rs](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs)) |
| `fc2_format` | string enum | *required* | same enum as `fc1_format` |
| `fc3_format` | string enum | required **iff** `fc3_experts_weights` wired; forbidden otherwise | same enum as `fc1_format` |
| `block_layout_version` | int | 1 | must equal 1 ([block_quantized_matmul.rs:157-162](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L157-L162)) |
| `k` (top_k) | int | 1 | `MoeAttributes.k`, must be `>0` and `<= E` ([moe.rs:59-62](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L59-L62), [qmoe.rs:192-197](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L192-L197)) |
| `activation_type` | string | `relu` | `relu, gelu, silu, swiglu, identity` ([moe.rs:63-80](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L63-L80)) |
| `normalize_routing_weights` | int(bool) | 0 | ([moe.rs:81](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L81)) |
| `swiglu_fusion` | int | 0 | `0..=2`; nonzero only with swiglu ([moe.rs:87-97](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L87-L97)) |
| `activation_alpha` | float | 1.0 | ([moe.rs:103](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L103)) |
| `activation_beta` | float | 0.0 | ([moe.rs:104](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L104)) |
| `swiglu_limit` | float | `f32::MAX` | ([moe.rs](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs)) |

The claim validator rejects any node carrying the obsolete single `format`
attribute (it is no longer in the attribute whitelist) — there is no silent
fallback to a uniform format.

**Intentionally NOT present** (justified): `expert_weight_bits`, `block_size`,
`quant_type`, and `use_sparse_mixer`. The first three are affine-int-only knobs
that IQ/MXFP4 self-description makes meaningless
([qmoe.rs:67-89](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L67-L89));
sparse-mixer normalization is lowered outside the op (§7).
`expert_dim`/`inter_dim` are **not** attributes: like QMoE they are inferred
from weight shapes (inter from fc2's packed width, hidden from input), not
declared ([qmoe.rs:174-222](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L174-L222)).
Adding them as redundant attributes is offered as Decision 6.

### 1.4 Output

Single output `output : f32`, shape **equal to `input`**
(`[rows,H]` or `[B,S,H]`), identical to MoE/QMoE
([qmoe.rs:168-173](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L168-L173),
[custom_ops.rs:15-22](../../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L15-L22)).

---

## 2. Selected-expert dispatch semantics (the crux)

### 2.1 Routing

Routing reuses `routing_weights` unchanged
([moe.rs:394-434](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L394-L434)):
top-`k` selection over `router_logits`; when `router_weights` (idx 8) is present
it supplies aggregation weights, otherwise a softmax over logits is used;
`normalize_routing_weights` renormalizes over the selected `k`.

### 2.2 Evaluate only routed experts

The portable/QMoE-eval-all path loops every expert per row
([moe.rs:244-278](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L244-L278)).
v1 adopts QMoE's **route-first grouping** as the *always-on* semantic (not just
the offload path): compute all routes, bucket tokens by selected expert into a
`BTreeMap<expert, Vec<(row, slot, weight)>>`, then iterate experts in ascending
index, decoding each selected expert's weights **once** and accumulating all its
routed tokens ([qmoe.rs:311-402](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L311-L402)).
Experts with no routed token are never decoded — this is the win over QMoE,
which still materializes all experts unless offload is enabled.

### 2.3 Lazy selected-slice binding (weight lease seam)

The offload boundary is recognized as `LazyWeightBoundary::BlockQuantizedMoe`
([weight.rs:94-105](../../crates/onnx-runtime-ep-api/src/weight.rs#L94-L105)), but
the current binder cannot lease a selected expert slice. The contract between
the op and the lazy weight handle:

1. **Negotiation.** The executor delivers each expert weight as a
   `WeightHandle`. An EP advertising `NXRT_WEIGHT_PAGING_CAPABILITY` receives
   `NegotiatedWeight::Lazy`; otherwise it transparently materializes resident
   ([weight.rs:171-203](../../crates/onnx-runtime-ep-api/src/weight.rs#L171-L203)).
   A non-paging EP is therefore always correct, just resident.
2. **Required per-expert lease seam extension.** Today
   `try_bind_device` passes only the whole `&LazyWeight` to
   `bind_block_quantized_moe`; that method has no expert index, projection, or
   byte-range argument ([weight.rs:205-227](../../crates/onnx-runtime-ep-api/src/weight.rs#L205-L227)).
   `LazyWeight` exposes a region list and a whole-weight resident materializer,
   not per-expert materialization ([weight.rs:121-161](../../crates/onnx-runtime-ep-api/src/weight.rs#L121-L161)).
   Before live selected-expert paging, extend
   `bind_block_quantized_moe` with an expert-indexed slice descriptor
   (projection plus byte range), and make `LazyWeight` resolve that descriptor
   against its regions. Only then can the kernel request each selected
   expert's projection slice, analogous to QMoE's host-cache lease of one
   expert window ([qmoe.rs:356-402](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L356-L402)).
3. **Current granularity = whole weight.** Until that extension lands, a lazy
   weight can only use its resident materializer for the entire weight
   ([weight.rs:158-161](../../crates/onnx-runtime-ep-api/src/weight.rs#L158-L161));
   per-expert page-in is not available through this seam.
4. **Phase staging of the binder.** v1 CPU ships with the
   `Phase3aHostOnlyBinder` semantics: the device binder returns
   `Unsupported` and callers use the host materialization route
   ([weight.rs:229-241](../../crates/onnx-runtime-ep-api/src/weight.rs#L229-L241)).
   Live device paging (`try_bind_device` returning real CUDA bindings) is
   Phase 3b and is **out of v1 scope** (Decision 7).

**Determinism note on grouping:** the `BTreeMap` gives ascending expert
*processing* order, but QMoE assigns slots during top-k route discovery and
reduces them in that route-slot order
([qmoe.rs:312-335,356,404-420](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L312-L335)).
The ascending-expert reduction rule for this op is specified separately in §4.

---

## 3. Block-format decode reuse

Every format maps to the shared `BlockQuantizedMatMul` decoder surface. This
slice extends that surface with the required GGML K-quants and Q8_0. The kernel
calls `BlockFormat::parse` then
`BlockFormat::decoder()` / `scalar_decoder()`
([block_quantized_matmul.rs:114-141](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L114-L141)):

| Format | Decoder fn (reused) | Codebook grid (compile-time) |
|---|---|---|
| `mxfp4` | `decode_mxfp4_block` (+AVX2 dispatch) | E2M1/E8M0 via `decode_e2m1`/`decode_e8m0_scale` ([block_dequant.rs:14-24](../../crates/onnx-runtime-ep-cpu/src/kernels/block_dequant.rs#L14-L24)) |
| `iq4_nl` | `decode_iq4_nl_block` (+AVX2) | `IQ4_NL_CODEBOOK` ([block_quantized_matmul.rs:45-47](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L45-L47)) |
| `iq4_xs` | `decode_iq4_xs_block` (+AVX2) | `IQ4_NL_CODEBOOK` |
| `iq3_s` | `decode_iq3_s_block` | `IQ3S_GRID` ([lib.rs:1219](../../crates/onnx-runtime-quantization/src/lib.rs#L1219)) |
| `iq3_xxs` | `decode_iq3_xxs_block` | `IQ3XXS_GRID` ([lib.rs:915](../../crates/onnx-runtime-quantization/src/lib.rs#L915)) |
| `iq2_s` | `decode_iq2_s_block` | `IQ2S_GRID` ([lib.rs:655](../../crates/onnx-runtime-quantization/src/lib.rs#L655)) |
| `iq2_xs` | `decode_iq2_xs_block` | `IQ2XS_GRID`/`IQ2XS_SIGNS` ([lib.rs:523,950](../../crates/onnx-runtime-quantization/src/lib.rs#L523)) |
| `iq2_xxs` | `decode_iq2_xxs_block` | `IQ2XXS_GRID` ([lib.rs:960](../../crates/onnx-runtime-quantization/src/lib.rs#L960)) |
| `iq1_s` | `decode_iq1_s_block` | `IQ1S_GRID` ([lib.rs:7](../../crates/onnx-runtime-quantization/src/lib.rs#L7)) |
| `iq1_m` | `decode_iq1_m_block` | `IQ1S_GRID` + `IQ1_M_DELTA` ([block_quantized_matmul.rs:38-39](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L38-L39)) |
| `q2_k` | `decode_q2_k_block` | embedded 4-bit scales/mins |
| `q3_k` | `decode_q3_k_block` | embedded 6-bit signed scales |
| `q5_k` | `decode_q5_k_block` | embedded 6-bit scales/mins |
| `q6_k` | `decode_q6_k_block` | embedded signed 8-bit scales |
| `q8_0` | `decode_q8_0_block` | per-block f16 scale |

**Genuinely new vs. reused:**

- **Reused:** existing IQ/MXFP4 decode math, grids, `qk`/`block_bytes`
  tables, AVX2 dispatch, E2M1/E8M0 scale decode.
- **Added from authoritative GGML layouts:** Q2_K/Q3_K/Q5_K/Q6_K/Q8_0 scalar
  CPU oracles and matching property-driven CUDA cases.
- **New (thin):** (a) an *expert-major, per-expert* iteration wrapper that
  slices `[E, N, blocks, block_bytes]` into one `[N, blocks, block_bytes]`
  weight per selected expert and feeds `dequantize_weight_kn`
  ([block_quantized_matmul.rs:255-260](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L255-L260));
  (b) fusion of the decoded gate/up→activation→down pipeline (reuse
  `MoeAttributes::swiglu` / `accumulate_expert`,
  [moe.rs:109-113](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L109-L113),
  [qmoe.rs:931](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L931));
  (c) the lease integration of §2.3. No new numeric kernel.

To avoid cross-crate churn, the shared decode routines should be lifted to a
`pub(crate)`/module-visible surface reachable by both kernels; today
`BlockFormat` and its decoders are private to `block_quantized_matmul.rs`
([block_quantized_matmul.rs:51-141](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L51-L141)).
That refactor is mechanical and behavior-preserving.

---

## 4. Determinism contract (hard repo requirement)

Bit-reproducible CPU-side and CPU↔CUDA parity-testable. Rules:

1. **f32 accumulation everywhere.** GEMM and expert accumulation run in f32,
   matching QMoE/`BlockQuantizedMatMul` (`vec![0.0f32; …]`, f32 `gemm`)
   ([block_quantized_matmul.rs:238-239](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L238-L239),
   [qmoe.rs:355](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L355)).
2. **Fixed reduction order.** v1 introduces (rather than inherits from QMoE)
   an ascending-selected-expert-index reduction rule: after routing, it orders
   each token's selected contributions by expert index before assigning fixed
   slots and sums those slots in that order, independent of routing/threading
   order. QMoE processes its `BTreeMap` in ascending expert order but assigns
   slots in top-k route order and reduces in slot order
   ([qmoe.rs:312-335,356,404-420](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L312-L335)).
   The route-first design has all selected routes before evaluation (§2.2), so
   it can impose this canonical order; the tradeoff is the per-token
   reordering step. Within a projection, dot-product accumulation follows the
   fixed `gemm` iteration order.
3. **Deterministic top-k tie-break rule (stated).** When two experts have equal
   router scores, the **lower expert index wins**. This is the existing rule:
   `sort_unstable_by(|&a,&b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)))`
   — descending score via `total_cmp` (NaN-total-ordered, no UB), ties broken by
   ascending index ([moe.rs:400-402](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L400-L402)).
   v1 inherits this verbatim; it must be documented in the schema so CUDA
   reproduces it.
4. **CPU is the parity oracle.** Phase 1 CPU output is the golden reference;
   CUDA must match it bitwise on the same inputs, following the QMoE staging
   ([GLM_READINESS_GAPS.md:363-365](../models/GLM_READINESS_GAPS.md#L363-L365)). Decoders
   are already vendored byte-for-byte from a pinned llama.cpp commit, which
   anchors cross-backend agreement
   ([block_quantized_matmul.rs:44-49](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L44-L49)).

---

## 5. Shape-inference contract

Mirror the `moe` handler style — the single output preserves the activation
tensor ([custom_ops.rs:15-22](../../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L15-L22)).

```
pub fn block_quantized_moe(ctx):
    require_outputs(ctx, 1)
    if let Some(input_type) = ctx.input_type(0):
        ctx.set_output_type(0, input_type)   # dtype + full shape = input
```

Registered alongside the others
([custom_ops.rs:297-313](../../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L297-L313)):
`reg.register("pkg.nxrt", "BlockQuantizedMoE", 1, block_quantized_moe)`.

**Derived by inference:** output rank/dims/dtype = `input`.
**Deliberately left unresolved (kernel-time only):** `E`, `inter`, and
`block_bytes` consistency across experts/projections. Like QMoE, these come from
weight tensor shapes the shape-inference pass does not fully constrain
([qmoe.rs:205-222](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L205-L222));
encoding them in shape inference would require guessing packed-layout invariants
and is intentionally avoided (the `GatherBlockQuantized` lesson,
[custom_ops.rs:1-6](../../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L1-L6)).

---

## 6. CPU + CUDA kernel plan (high level, no kernel code)

Staged exactly like QMoE Phase 1/2
([GLM_READINESS_GAPS.md:344-347,363-365](../models/GLM_READINESS_GAPS.md#L344-L347)):

**Phase 1 — CPU reference (parity oracle).**
- Register `BlockQuantizedMoEFactory` in the CPU kernel registry.
- Validate ABI (§1), infer `E`/`inter`, reuse `MoeAttributes`.
- Route-first grouping (§2.2) as the always-on path.
- Per selected expert: slice expert-major weights → reuse
  `BlockQuantizedMatMul` decoders (§3) → fused gate/up/activation/down via
  `accumulate_expert`.
- Host-materialization lease path only (`Phase3aHostOnlyBinder` semantics).
- **Interface:** `Kernel::execute(inputs, outputs)` +
  `set_constant_inputs` for constant packed weights (as
  `BlockQuantizedMatMul` caches constant weight KN,
  [block_quantized_matmul.rs:182-232](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L182-L232)).

**Phase 2 — CUDA correctness.**
- Add `BlockQuantizedMoE` to the CUDA registry next to `qmoe`/`qmoe_gemm`/
  `qmoe_grouping` (existing CUDA QMoE surface). Reuse the CUDA
  `block_quantized_matmul` decoder.
- All-f32. Each projection launch carries its own format id, elements/block,
  bytes/block, row stride, and expert stride. Mixed IQ/K-quant/Q8 combinations
  are claimed and executed directly from packed bytes.
- The warmed launch path has no trailing host synchronization, allocation,
  transfer, or format scan, so fixed-shape execution is CUDA-graph capturable.
- Gate merges on parity vs. the Phase 1 CPU oracle (§4), including opt-in
  official-checkpoint block tests.

**Phase 3b (out of v1 scope) — live device paging.**
- First extend `LazyDeviceWeightBinder` and `LazyWeight` for an
  expert-indexed slice descriptor (§2.3); their current interface binds only a
  whole `LazyWeight` ([weight.rs:121-161,205-227](../../crates/onnx-runtime-ep-api/src/weight.rs#L121-L161)).
- Then implement a real CUDA binder that pages selected expert slices to device
  on demand.

The current CUDA kernel reads only the routed experts selected by the route
kernel, but the production paging API still binds/commits a whole projection
bank. Therefore selected-expert residency/page-in and byte-hit-rate claims
remain incomplete; this implementation makes no per-expert residency claim.
`admit_block_quantized_moe_banks` is the sealed ownership boundary: it validates
full-block geometry and reserved/non-finite scale codes before immutable upload,
binds the result to one provider context/device, and exposes no device pointer.
Its `no_residency_traffic` control reports one-time full-bank load bytes and
unique selected-expert projection bytes, with `page_ins=0` and
`byte_hit_rate=None` until the expert-indexed paging dependency lands.

### 6.1 A100 validation snapshot

On 2026-08-27, the release-mode opt-in probe ran on idle physical A100 GPU 7
(`GPU-ef3c1a70-d297-933c-0c37-dbaad2136a57`) with
`CUDA_VISIBLE_DEVICES=7` and `ONNX_GENAI_CUDA_DEVICE=0`. Each shape used an
8-second continuous ramp, CUDA-graph batches of 128 replays, and three timed
batches. The first decode shape was remeasured last. Logical bytes are unique
selected projection bytes, not measured DRAM transactions.

| checkpoint projection pair | stage | rows | fixed load / selected read bytes | median µs | full range µs |
|---|---|---:|---:|---:|---:|
| IQ1_S / IQ3_XXS | decode | 1 | 7,274,496 | 413.041 | 412.644–413.177 |
| IQ1_S / IQ3_XXS | prefill | 8 | 7,274,496 | 3,319.706 | 3,319.499–3,320.734 |
| IQ2_XXS / IQ3_XXS | decode | 1 | 8,060,928 | 271.792 | 271.753–271.834 |
| IQ2_XXS / IQ4_XS | decode | 1 | 9,928,704 | 261.997 | 261.997–262.005 |
| Q2_K / Q3_K | decode | 1 | 9,535,488 | 318.347 | 318.337–318.355 |
| IQ1_S / IQ3_XXS (repeat) | decode | 1 | 7,274,496 | 412.951 | 412.781–413.071 |

Every row reported `captures=1`, `fallbacks=0`, `page_ins=0`, and
`byte_hit_rate=None`. The probe presents one exact-geometry expert
(`H=6144`, `I=2048`) so fixed load and selected-read bytes are equal; it does
not claim expert-indexed paging. Base
`aa417ad372e6c0c1d2df154ceb236ab1c3bea73e` refuses these mixed-format nodes
and lacks the required K-quant/Q8 decode cases, so no comparable base latency
exists and no speedup claim is made.

---

## 7. `sparse_mixer` interaction (P1 #9)

The QMoE/MoE Phase-1 CPU reference **rejects** `use_sparse_mixer=1`
([moe.rs:82-86](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L82-L86)).
sparse_mixer is a *routing-normalization* transform, orthogonal to expert
arithmetic.

**Recommendation:** keep routing normalization **outside** this op, expressed
as an explicit portable gate (softmax / sparse-mixer jitter+renorm) feeding
`router_logits`/`router_weights`. `BlockQuantizedMoE` v1 omits the
`use_sparse_mixer` attribute; a QMoE transcode must lower it explicitly.
Rationale: (a) it keeps the determinism surface small and CUDA-parity-simple;
(b) the schema does not freeze an attribute that every v1 implementation
rejects; and (c) native fused sparse_mixer can be added in a later version once
P1 #9 freezes its semantics. This is offered for sign-off as Decision 4.

---

## Decisions for Justin

Each item lists a recommended default. Sign-off requested before kernel work.

1. **Weight-input ordering — dense 0–8 vs. QMoE-index-preserving.**
   *Recommend:* the dense 0–8 layout in §1.2 (no scales/zero_points, so QMoE's
   reserved gaps are meaningless). *Alt:* keep QMoE's exact indices for
   mechanical transcode. **Default: dense 0–8.**

2. **Router input: logits vs. pre-normalized weights.**
   *Recommend:* name the required input `router_logits` and apply softmax
   internally, with optional `router_weights` overriding aggregation — identical
   in semantics to QMoE without the misleading `router_probs` name
   ([qmoe.rs:293-322](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L293-L322)).
   **Default: logits in, optional aggregation weights.**

3. **Block-format enum encoding: string attribute vs. int enum.**
   *Recommend:* reuse `BlockQuantizedMatMul`'s **string** `format`
   ([block_quantized_matmul.rs:66-82](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L66-L82))
   for consistency and readability. *Alt:* int enum for compactness.
   **Default: string `format`.**

4. **sparse_mixer placement (P1 #9).**
   *Recommend:* normalization **outside** the op (explicit portable gate); omit
   `use_sparse_mixer` from v1 (§7). **Default: outside / absent from schema.**

5. **Single uniform `format` vs. per-projection / per-expert formats.**
   **RESOLVED (2026-08-24): per-projection formats implemented.** The original v1
   recommendation of one uniform `format` was **wrong for real GLM-5.2 GGUF** —
   its routed experts pack gate/up and down at different qtypes on every layer
   (see §1.1). The ABI now carries independent `fc1_format`/`fc2_format`/
   `fc3_format`; the uniform `format` attribute is removed with no compatibility
   shim. `mxfp4`/IQ names still mean the current `BlockQuantizedMatMul` layout,
   not an unpublished K3/Moonshot format. **Per-expert** (as opposed to
   per-projection) formats remain out of scope — every expert in a layer shares
   that layer's per-projection formats. `Q2_K`/`Q3_K`/`Q5_K`/`Q6_K`/`Q8_0`
   now use the same property-driven per-projection dispatch. Q4_K remains
   unsupported because the two checkpoint Q4_K tensors are outside this FFN
   production slice.

6. **`inter`/`expert_dim` inferred vs. declared attributes.**
   *Recommend:* **inferred from weight shapes** (QMoE precedent,
   [qmoe.rs:205-222](../../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L205-L222)),
   not redundant attributes. **Default: inferred.**

7. **v1 lazy-offload requirement: resident-only first, or require paging?**
   *Recommend:* v1 supports **resident-only** correctness on any EP via
   `WeightHandle::negotiate` fallback
   ([weight.rs:171-203](../../crates/onnx-runtime-ep-api/src/weight.rs#L171-L203)),
   while device paging is deferred to Phase 3b
   (`Phase3aHostOnlyBinder`, [weight.rs:229-241](../../crates/onnx-runtime-ep-api/src/weight.rs#L229-L241)).
   **Prerequisite for selected-expert device paging:** extend the binder and
   `LazyWeight` as §2.3 specifies; the current whole-`LazyWeight` binder
   signature cannot select an expert slice
   ([weight.rs:121-161,205-227](../../crates/onnx-runtime-ep-api/src/weight.rs#L121-L161)).
   **Default: resident-first correctness for the GLM/IQ profile; require that
   seam extension before paging or production-scale K3 claims.**

8. **Op version: ship as `pkg.nxrt::BlockQuantizedMoE` v1 now?**
   *Recommend:* yes — freeze this ABI as v1 for the currently verified GLM/IQ
   and BlockQuantizedMatMul layouts so the CPU parity oracle and the `weight.rs`
   seam are backed by a stable contract. Native K3/Moonshot formats remain out
   of scope until official layouts and activation semantics are available; use
   a namespaced format or new op version if they differ. **Default: scoped v1.**
