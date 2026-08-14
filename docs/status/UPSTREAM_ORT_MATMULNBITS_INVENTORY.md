# MatMulNBits CPU Upstream Inventory

**Author:** Resch (Intel CPU Optimization Engineer)
**Date:** 2026-08-11
**Status:** Analysis only — no code, no PR

---

## 1. Our MatMulNBits / Block-Quantized CPU Paths

### 1a. `matmul_nbits.rs` — ORT-compatible `com.microsoft::MatMulNBits`

| Component | ISA | Block sizes | Bit widths | Notes |
|-----------|-----|-------------|------------|-------|
| `int4_matmul_m1` | AVX2, AVX-VNNI, AVX-512 VNNI, NEON | block-32 (VNNI int4-direct), all (int8 route) | 4-bit | M=1 decode; symmetric only for VNNI direct |
| `int8_matmul` | AVX2, AVX-VNNI, AVX-512 VNNI, NEON, AMX | all | 4-bit (dequant to int8) | `accuracy_level=4`; activation quantized per-block to int8 |
| `dot_u8_i8` | AVX2 (`vpmaddubsw`), AVX-VNNI (`vpdpbusd`), AVX-512 VNNI (`vpdpbusd zmm`), NEON (`sdot`) | n/a (inner dot) | u8 × i8 | Core dot-product micro-kernel |
| `block_dot_u8_i16` | AVX-512 BW, AVX2, scalar | n/a | u8 × i16 | Grouped dot for non-standard quant |
| MLAS SQNBit sharding | via `mlas-sys` FFI | all MLAS-supported | 4-bit, 2-bit | Prepacks weights, uses MLAS `MlasQNBitGemmBatch` for M>1 |

**Source:** `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs`

### 1b. `simd_quant.rs` — AVX-512 activation quantizers

| Function | ISA | Output type | Notes |
|----------|-----|-------------|-------|
| `quantize_block_i8` | AVX-512 F/BW/VL (runtime-gated), scalar fallback | i8 | Symmetric, scale = max_abs/127 |
| `quantize_block_u8_offset` | AVX-512 F/BW/VL (runtime-gated), scalar fallback | u8 (=i8+128) | For VNNI `u8 × i8` consumption |
| `quantize_block_i16` | AVX-512 F/BW/VL (runtime-gated), scalar fallback | i16 | For grouped dot path |

**Rounding:** Uses `round_half_away_avx512` — round-half-away-from-zero, matching Rust `f32::round()`.

**NaN handling:** Detects non-finite lanes via ordered compare and falls back to scalar for exact bit-identity.

**Dependencies:** **Zero** nxrt/IR/EP imports. Pure `std::arch` + `std` only. This is the most portable module in our codebase.

**Source:** `crates/onnx-runtime-ep-cpu/src/kernels/simd_quant.rs`

### 1c. `block_quantized_matmul.rs` — GGUF block formats

GGUF-specific (MXFP4, IQ series). Depends on `onnx_runtime_quantization` crate, our custom `BlockQuantizedMatMul` op domain, GGUF layout constants. **Not upstreamable** — entirely our custom op with our memory format.

---

## 2. Upstream Comparison

### 2a. MLAS SQNBit Activation Quantization

| Upstream file | ISA | Function | What it does |
|---------------|-----|----------|-------------|
| `sqnbitgemm_kernel_avx512.cpp:349` | AVX-512 F | `QuantizeARow_CompInt8_avx512` | Per-block i8 quant + block-sum; 16-wide `_mm512_*` |
| `sqnbitgemm_kernel_avx512.cpp:452` | AVX-512 F + F16C | `QuantizeARow_CompInt8_Fp16_avx512` | Same but from fp16 input |
| `sqnbitgemm_kernel_avx2.cpp:1279` | AVX2 | `QuantizeARow_CompInt8_avx2` | Per-block i8 quant + block-sum; 8-wide `_mm256_*` |
| `sqnbitgemm_kernel_avx2.cpp:1365` | AVX2 + F16C | `QuantizeARow_CompInt8_Fp16_avx2` | Same but from fp16 input |
| `sqnbitgemm_kernel_avx512vnni.cpp:398` | AVX-512 VNNI | Delegates to avx512 version | Same quantizer, VNNI dot kernels |
| `sqnbitgemm_kernel_avx2vnni.cpp` | AVX2 + VNNI | (verified: has `QuantizeARowComputeBlkSum`) | Same quantizer, VNNI dot kernels |

**Verdict on our AVX-512 activation quantizer (`simd_quant.rs`):**

Upstream **ALREADY HAS** AVX-512 activation quantization at `sqnbitgemm_kernel_avx512.cpp:349-446`. The upstream version:
- Uses `_mm512_roundscale_ps(v, _MM_ROUND_NEAREST)` = **round-half-to-even** (IEEE banker's rounding)
- Also computes `AScaledBlkSum` (block sum for zero-point correction) inline
- Is tightly integrated into the MLAS dispatch table (`QNBitGemmDispatch`)

Our version:
- Uses `round_half_away_avx512` = **round-half-away-from-zero** (Rust `f32::round`)
- Does NOT compute block-sum inline (done separately)
- Has explicit NaN-safety fallback (upstream does not)
- Is a standalone function with zero framework dependencies

**The rounding semantics differ.** For exact half-integer values (rare in practice), our codes and upstream's codes will differ by 1. This is a real semantic gap but:
- Upstream chose banker's rounding deliberately to match `_MM_ROUND_NEAREST` (the hardware default)
- Changing upstream's rounding to match ours would break existing ORT users
- Changing ours to match upstream is possible but irrelevant for upstream contribution

**Overall verdict: ALREADY-COVERED (with minor rounding difference)**

### 2b. MLAS SQNBit Dot Products

| Upstream path | ISA | Block sizes | Notes |
|---------------|-----|-------------|-------|
| `sqnbitgemm_kernel_avx2.cpp` | AVX2 | blk 16/32/64 (int8), all (fp32) | `vpmaddubsw`/`vpmaddwd` dot |
| `sqnbitgemm_kernel_avx2vnni.cpp` | AVX2 + VNNI | blk 16/32/64 (int8) | `vpdpbusd ymm` |
| `sqnbitgemm_kernel_avx512.cpp` | AVX-512 F | blk 16/32/64/128 (int8), all (fp32) | 512-bit dot |
| `sqnbitgemm_kernel_avx512vnni.cpp` | AVX-512 + VNNI | blk 16/32/64/128 (int8) | `vpdpbusd zmm` |
| `sqnbitgemm_lut_kernel_avx2.cpp` | AVX2 | (LUT-based path) | Alternative approach |
| `sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h` | AVX2 | blk 32/64 (M=1 sym) | Specialized M=1 |

**Our dot products** (`dot_u8_i8` in `matmul_nbits.rs`):
- AVX2: `vpmaddubsw` + `vpmaddwd` — **same technique as upstream**
- AVX-VNNI: `vpdpbusd ymm` — **same as upstream**
- AVX-512 VNNI: `vpdpbusd zmm` — **same as upstream**
- NEON `sdot` — **same as upstream ARM path**

**Verdict: ALREADY-COVERED** — upstream has the same ISA dispatch and instruction selection.

### 2c. int4-direct VNNI path (block-32 symmetric M=1)

Our `int4_matmul_m1` for block-32 symmetric on VNNI streams packed int4 weights directly into VNNI dot without dequanting to int8. Upstream's `sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h` already implements a similar M=1 symmetric specialization.

**Verdict: ALREADY-COVERED / PARTIAL** — upstream has M=1 sym blk-32 specializations; whether they avoid the int4→int8 unpack is an implementation detail, but the performance-critical path exists.

---

## 3. Issue #23004 Verdict

**Issue:** [Performance] MatMulNBits Performance — reports int4 MatMulNBits ~10x slower than int8 DynamicQuantizeMatMul on CPU.

**Status:** Open, assigned to `@fajin-corp` (Microsoft contributor). Last activity 2024-12-06.

**Root cause per comments:** The reporter was comparing different quantization granularities (int4 weight-only vs int8 dynamic with activation quantization) and also quantizing Gather ops to int4. Microsoft's `@fajin-corp` explained that int4 being ~1.6x slower than int8 is expected (int4 must convert to int8 during compute), and the 10x slowdown in LLMs was likely due to the reporter's setup (quantizing Gather). The issue appears to be a **user configuration problem, not a kernel performance bug**.

**Relevance to upstream:** This issue does not identify a missing kernel or optimization. The MLAS int4 kernels already exist for `accuracy_level=4`. The performance gap is the inherent cost of int4→int8 conversion. **No actionable upstream contribution from this issue.**

---

## 4. In-Flight Work Check

| PR/Issue | Title | Status | Relevance |
|----------|-------|--------|-----------|
| PR #29842 | [MLAS] Quantize fp16 activations directly on AVX-512 MatMulNBits CompInt8 | **Closed/merged** | Upstream already added fp16→int8 direct AVX-512 quant |
| PR #29064 | [MLAS] Add AVX512 (+VNNI) 2-bit weight CPU kernels | **Closed/merged** | 2-bit AVX-512 kernels already landed |
| Issue #29853 | MLAS AVX2 M=1 CompInt8 SQNBit wrong results for asymmetric weights | **Open** | Bug, not perf; we also have a workaround (line 1326-1330 in matmul_nbits.rs) |
| Issue #27251 | [Feature Request] MatMulNBits faster for fp16 input | **Open** | Microsoft tracking; PR #29842 addressed AVX-512 part |
| Issue #29849 | MatMulNBits accuracy_level=4 wrong argmax on some LLMs | **Open** | Accuracy bug in CompInt8 path; not perf |

**No open PRs touching sqnbitgemm activation quantization for x86.** But the existing codebase already covers AVX2 and AVX-512.

---

## 5. Ranked Shortlist of Upstreamable Candidates

### Candidate A: NaN-safe activation quantizer (simd_quant.rs)
- **What:** Our AVX-512 quantizer detects non-finite lanes and falls back to scalar, preserving bit-identical results. Upstream's `QuantizeARow_CompInt8_avx512` does NOT handle NaN/inf — it will propagate garbage through `_mm512_max_ps` and produce wrong codes.
- **Upstream gap:** PARTIAL — upstream has AVX-512 quant but no NaN guard.
- **Impact:** Low in practice (NaN activations shouldn't occur in healthy inference), but it's a correctness hardening.
- **Portability:** HIGH — the NaN check is ~3 extra instructions (one ordered compare per 16 lanes). No nxrt dependencies. Pure C++ intrinsics.
- **Testability on this host:** ❌ NO — this host is AMD EPYC 9V74, AVX2 only, **no AVX-512**. Cannot test.
- **Implementation cost:** ~20 lines of C++ change.
- **Likelihood of acceptance:** LOW — Microsoft would likely argue NaN activations are UB and not worth guarding.
- **Score: 2/10** — too niche, untestable here.

### Candidate B: round-half-away-from-zero quantizer
- **What:** Our quantizer uses round-half-away-from-zero; upstream uses banker's rounding. This is a deliberate semantic choice, not a bug.
- **Upstreamability:** NONE — changing upstream's rounding would break backward compatibility for all ORT users.
- **Score: 0/10** — not a gap, it's a design choice.

### Candidate C: Standalone activation quantizer library
- **What:** Extract our `simd_quant.rs` as a reusable C/C++ activation quantizer with runtime ISA dispatch (AVX2 fallback, AVX-512 when available).
- **Upstream gap:** Upstream's quantizer is tightly coupled to the MLAS dispatch table. A standalone version could be useful.
- **Portability:** HIGH — simd_quant.rs has zero framework dependencies.
- **Testability:** ❌ AVX-512 path untestable on this host.
- **Implementation cost:** Medium — needs C++ rewrite from Rust, plus MLAS integration.
- **Likelihood of acceptance:** LOW — MLAS already has working quantizers; refactoring them for standalone use isn't a priority.
- **Score: 1/10** — not motivated by a real perf gap.

---

## 6. Explicit Non-Candidates

| Component | Reason |
|-----------|--------|
| `block_quantized_matmul.rs` | GGUF-specific formats, depends on `onnx_runtime_quantization`, our custom op domain. Not ORT-compatible. |
| `int4_matmul_m1` (full kernel) | Tightly integrated with our Rayon thread pool, decode-pool model, MLAS sharding. Upstream has equivalent M=1 kernels. |
| `int8_matmul` (full kernel) | Same — our kernel orchestration (sharding, thread pool, crossover logic) is runtime-specific. The inner dots are identical to upstream. |
| AMX `int8_matmul_amx` | AMX tile management tied to our runtime; upstream would need its own AMX integration. |
| `dot_u8_i8` micro-kernels | Identical instruction selection to upstream MLAS. No gap. |
| `block_dot_u8_i16` | Upstream doesn't have an i16 grouped-dot path, but this serves our specific non-standard quantization — not a standard ORT need. |
| Decode thread pool / sharding | Runtime orchestration, not a kernel. Not upstreamable. |

---

## 7. Top Candidate Scope Sketch

**There is no viable top candidate.**

The honest conclusion is that upstream MLAS already covers:
- AVX2 and AVX-512 activation quantization (both fp32 and fp16 input)
- AVX2, AVX-VNNI, AVX-512, AVX-512 VNNI dot products
- Block sizes 16/32/64/128 for CompInt8
- M=1 symmetric specializations
- 2-bit and 4-bit weight kernels

Our implementations are **functionally equivalent** at the ISA/instruction level. The differences are:
1. **Rounding semantics** (round-half-away vs banker's) — not a gap, a design choice
2. **NaN safety** — too niche to upstream
3. **Runtime orchestration** (decode pools, sharding) — not kernel code, not portable

---

## 8. Hardware Testability Warning

**This host: AMD EPYC 9V74 (Bergamo)**
- ✅ AVX2, FMA, F16C
- ❌ No AVX-512 of any kind
- ❌ No VNNI (AVX or AVX-512)
- ❌ No AMX

Any AVX-512, VNNI, or AMX candidate **cannot be validated on this machine**. This is a serious constraint: all of our most differentiated code paths (AVX-512 quantizer, VNNI dots, AMX matmul) are untestable. Only the AVX2 scalar-fallback paths can be verified here.

---

## 9. Verification Status

| Claim | Status |
|-------|--------|
| Upstream has AVX-512 QuantizeARow | **VERIFIED** — `sqnbitgemm_kernel_avx512.cpp:349` |
| Upstream has AVX2 QuantizeARow | **VERIFIED** — `sqnbitgemm_kernel_avx2.cpp:1279` |
| Upstream uses banker's rounding | **VERIFIED** — `_mm512_roundscale_ps(v0, _MM_ROUND_NEAREST)` at line 403 |
| Our quantizer has zero nxrt deps | **VERIFIED** — grep found no matches in `simd_quant.rs` |
| Issue #23004 is a user config issue | **VERIFIED** — comments show fajin-corp explaining expected behavior |
| PR #29842 (fp16 AVX-512 quant) merged | **VERIFIED** — state: closed |
| Our dot_u8_i8 uses same instructions as upstream | **VERIFIED** — both use `vpmaddubsw`/`vpdpbusd` |

| Claim | Status |
|-------|--------|
| Upstream M=1 sym blk-32 avoids int4→int8 unpack | **UNVERIFIED** — would need deeper read of `sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h` |
| Our VNNI int4-direct path is faster than upstream's | **UNVERIFIED** — no comparative benchmarks exist; our numbers measured our Rust runtime |

---

## 10. Open Questions for @justinchuby

1. **Should we close this line of investigation?** Four of four prior candidates died on inspection. This fifth (MatMulNBits CPU) shows the same pattern: upstream MLAS already has comprehensive ISA coverage. The differences are in runtime orchestration (our Rayon pools, decode sharding) which are not portable.

2. **Is the NaN-safety hardening worth a standalone micro-PR?** It's ~20 lines of C++ and genuinely missing from upstream, but it's a correctness edge case, not a performance improvement, and we can't test it on this host.

3. **Should we pivot to non-kernel contributions?** The most impactful CPU int4 work might be at the graph/session level (better threading defaults, MatMulNBits session config guidance) rather than kernel-level — which is where issue #23004's real problem lay.

4. **Is there value in upstreaming our benchmark harness** (`ONNX_GENAI_PROFILE_MM` split timing) as a diagnostic tool for ORT's MatMulNBits path? That's not kernel code but could help the community diagnose issues like #23004.

---

## Recommendation

**Do not proceed to implementation.** There is no viable kernel-level gap between our MatMulNBits CPU code and upstream MLAS. The instruction selection is identical, the ISA dispatch is equivalent, and the activation quantizers are functionally equivalent (differing only in rounding tie-breaking). Our value-add is in runtime orchestration (decode pools, sharding, crossover tuning) which is inherently tied to our runtime and not portable to ORT's threading model.

A well-evidenced "no viable gap" is the correct outcome. Five honest negatives are better than one wasted implementation.
