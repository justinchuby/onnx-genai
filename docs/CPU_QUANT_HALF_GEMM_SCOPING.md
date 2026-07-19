# CPU INT4 vs. half GEMM MLAS scoping

**Scope date:** 2026-07-20.  This is a source-tree audit, not a benchmark
claim.  The recommended next slice is to replace the existing CPU
`MatMulNBits` hot path with MLAS SQNBitGemm where MLAS reports that the exact
format is available.

## Current state (ep-cpu)

### Quantized operations and `MatMulNBits`

The CPU EP already has a registered `com.microsoft::MatMulNBits` factory; it
is not a missing-op problem.  Its implementation accepts f32 A/scales/output,
uint8 packed B, optional packed uint8 zero points, and the standard
`[N, ceil(K/block_size), block_size*bits/8]` B layout
(`crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs:102-134`).  It
supports 2- and 4-bit weights (`matmul_nbits.rs:53-91`).

This is a mixed implementation, not merely dequantize-then-f32-GEMM:

* default int4 and all int2 first dequantize; M=1 uses an f32 GEMV and M>1
  invokes the shared f32 GEMM (`matmul_nbits.rs:236-275`,
  `matmul_nbits.rs:347-425`);
* `accuracy_level=4` int4 pre-packs constant weights as signed int8 and
  quantizes A to int8 (`matmul_nbits.rs:203-235`, `292-344`);
* its most important decode case--M=1, block-32, no ZP/g_idx--retains the
  original packed nibbles and uses AVX-VNNI/AVX-512-VNNI dot products
  (`matmul_nbits.rs:156-202`, `516-531`, `551-623`).

The registry also contains private `pkg.nxrt::BlockQuantizedMatMul` and
`BlockQuantizedMoE` alongside `MatMulNBits`
(`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:235-247`), and
`com.microsoft::QMoE` (`kernels/mod.rs:372-373`).  Standard registered
quantization ops are `QuantizeLinear`, `DequantizeLinear`, and
`DynamicQuantizeLinear` (`kernels/mod.rs:209-211`, `547-557`).  There is no
`MatMulInteger` registration.  Thus, the relevant missing performance work is
an MLAS implementation behind an already registered op, not adding a
quantized op.

`MatMulNBits` is *not* in `PHASE1_OPS`: that list is default-domain milestone
coverage (`kernels/mod.rs:101-213`), while its contrib registration is
separate (`kernels/mod.rs:235-239`).  It is already included in the additive
registry count: the test explicitly describes the extra contrib/private
entries and asserts `reg.len() == PHASE1_OPS.len() + 85`
(`kernels/mod.rs:1443-1479`).  Retrofitting its implementation must not change
that invariant; a genuinely new registered op would require updating `+85`.

### f16/bf16 today

The dtype contract deliberately stores f16/bf16 as 2-byte values but computes
them in f32 (`crates/onnx-runtime-ep-cpu/src/dtype.rs:12-22`).  `MatMul` uses
`to_dense_f32_widen` for both operands/cache materialization
(`kernels/matmul.rs:63-81`) and then invokes only an f32 GEMM backend
(`kernels/matmul.rs:108-167`).  The same f32-widen/narrow pattern is used by
Gemm (`kernels/gemm.rs:1-13`).  There is no native f16/bf16 GEMM invocation in
ep-cpu.

The conversion/storage support is real rather than bit reinterpretation:
`half::f16`/`half::bf16` are described as f32 round trips
(`dtype.rs:17-20`), and MatMul's tests explicitly describe f16 accumulation in
f32 (`kernels/matmul.rs` search result at `matmul_f16_accumulates_in_f32`).
This makes half GEMM an internal dispatch optimization, not a dtype
representation project.

### Model/export impact

The evidence shows that CPU-relevant LLM exports do emit this op.  The
quantized GLM-5.2 engine E2E fixture says block-32 asymmetric int4 projections
and expert MLPs emit `com.microsoft::MatMulNBits`
(`crates/onnx-genai-engine/tests/glm_tiny_quant_e2e.rs:1-18`), and progress
records 34 such nodes in the GLM graph
(`docs/PROGRESS.md:39-44`).  Fused GLM/DeepSeek expert exports can instead
emit `QMoE` (`docs/PROGRESS.md:32-35`), so SQNBitGemm is primarily the dense
projection and unfused-expert lever, not a QMoE replacement.

The checked-in engine/config layer recognizes Qwen architecture metadata and
f16/bf16 KV dtypes (`crates/onnx-genai-genai-config/src/lib.rs:51-79`,
`237-244`), and an existing Qwen2.5 int4 CPU-versus-CUDA smoke loads an
int4 ONNX model (`crates/onnx-genai-engine/src/native_decode.rs:1004-1024`).
This tree has no checked-in real Gemma/Qwen/DeepSeek/GLM ONNX artifact from
which to generalize every export.  The affirmative model-format evidence is
GLM int4 (and the Qwen int4 smoke); the fp32 DeepSeek synthetic E2E in
`docs/PROGRESS.md:41-43` should not be represented as proof of half dense
MatMul use.

## Vendored MLAS capability

### INT4 / SQNBitGemm

The vendor has the public C++ QNBit API in
`crates/mlas-sys/vendor/mlas/onnxruntime/core/mlas/inc/mlas_qnbit.h:20-225`:
it exposes availability, workspace-size, B-pack-size/B-pack, and batched
GEMM calls.  It explicitly models f32 A/f32 accumulation and int8 activation
compute (`mlas_qnbit.h:27-35, 43-71`).  The generic driver supports W4
block lengths 16/32/64/128/256 and f32/int8 compute
(`lib/qnbitgemm.cpp:45-82`).

Relevant sources are present: `lib/qnbitgemm.cpp/.h`,
`lib/sqnbitgemm_kernel_avx2.cpp`, `lib/sqnbitgemm_kernel_avx512.cpp`,
`lib/sqnbitgemm_kernel_avx512vnni.cpp`, the associated `*_int8_blklen*`
headers, and `lib/q4gemm.cpp/.h`, `q4gemm_avx512.cpp`,
`qgemm.cpp/.h`, and `qgemm_kernel_amx.cpp`.  More importantly, the current
build already compiles the generic QNBit driver
(`crates/mlas-sys/build.rs:52-84`), AVX2 SQNBit kernels
(`build.rs:94-106`), AVX-512 and AVX-512-VNNI SQNBit kernels
(`build.rs:120-139`), and AMX qgemm (`build.rs:141-145`).  No additional
vendor source is needed for option A.

There is not yet a Rust-callable QNBit function: the current shim exports only
SGEMM/packing and threading functions (`crates/mlas-sys/vendor/shim.cpp:101-181`;
`crates/mlas-sys/src/lib.rs:18-75`).  A narrow C shim is therefore required;
Rust should not bind MLAS C++ templates/data structures directly.

On Sapphire Rapids, MLAS selects AVX-512 then AVX-512-VNNI SQNBit dispatch
when the CPUID/OS-state predicates pass
(`lib/platform.cpp:572-595`).  That VNNI dispatch installs f32 M=1 Q4,
f32 dequant-B, and int8 Q4 kernels (`lib/sqnbitgemm_kernel_avx512vnni.cpp:508-539`).
AMX in this vendor changes U8S8 qgemm dispatch
(`lib/platform.cpp:621-630`), not the shown SQNBit dispatch, so it must not
be promised as the W4 SQNBit path.

### Half precision

MLAS declares `MlasHalfGemmBatch`, f16 acceleration probing, and generic and
native B packing (`inc/mlas.h:1826-1830, 1933-2071`).  The source driver
`lib/halfgemm.cpp/.h`, `fp16_common.h`, and f16 conversion assembly are
vendored.  However, `build.rs` does **not** compile `halfgemm.cpp`
(the generic source list is `build.rs:52-78`), so it is unavailable today.
The only half-GEMM dispatch object defined in this vendored x86 build is the
default template dispatch (`lib/halfgemm.cpp:717-724`); the header chooses
Neon/RVV only on those targets and default otherwise
(`lib/halfgemm.h:583-605`).  This inventory contains no x86 AVX512-FP16 or
AVX512-BF16 half-GEMM kernel source to compile.

Consequently, although the host ISA has AVX512-FP16/BF16, this vendor snapshot
does not supply an x86 MLAS HalfGemm path that would select those instructions.
`MlasFp16AccelerationSupported` additionally depends on
`MLAS_F16VEC_INTRINSICS_SUPPORTED` (`lib/halfgemm.cpp:51-60`).  bf16 is not a
separate MLAS dense-GEMM API here; the QNBit header has BF16 as a compute enum
(`inc/mlas_qnbit.h:29-35`), but that is not evidence of an x86 bf16 dense
kernel.  Option B therefore requires an upstream vendor expansion or a new
independent x86 kernel, plus compiler/target-feature gating; merely compiling
`halfgemm.cpp` is not a Sapphire-Rapids native-half solution.

## Option A: MLAS SQNBitGemm for existing `MatMulNBits`

**Wiring.** Add C shim functions for (1) availability, (2) packed-B/workspace
sizes and packing, and (3) f32 Q4 batch execution.  Expose safe Rust wrappers
that own aligned packed B/workspace and reject unavailable combinations.  Use
the existing standalone thread hooks, passing the same configured Rayon bridge
as SGEMM.  In `matmul_nbits.rs`, cache MLAS-packed B for constant
B/scales/ZP; dynamically pack only nonconstant inputs.  Keep the current path
as fallback for `g_idx`, unsupported bit/block shapes, unavailable MLAS, and
non-f32 contracts.

**Layout.** ONNX already uses LSB-first B rows and packed per-block ZPs, with
an absent ZP default of midpoint (`matmul_nbits.rs:4-12, 124-134`,
`429-470`).  This is exactly the affine W4 input that MLAS's pack API accepts:
its pack documentation consumes quantized B, scales, optional ZPs, and
computes the block correction (`mlas_qnbit.h:185-225`).  For standard
block-32 symmetric ONNX (`zp=8`, whether implicit or explicit), pass the
ONNX bytes/scales/ZPs to MLAS packing; do **not** pre-subtract 8 or reuse the
project's signed-int8 cache.  MLAS-owned packed B is a separate cache format,
so repacking is required once per immutable initializer.

**Tests and performance.** Add shim-level availability/shape tests; unit tests
against the existing dequantized oracle for block-32 implicit/explicit ZP,
asymmetric ZP, tail K, M=1 and M>1, bias, strided A, and cache reuse; and
engine GLM Q4 decode/prefill regression.  Benchmark ORT and native with
matching threads, separately measuring cold pack, warm M=1 decode, and M>1
prefill.  Expect the largest win in the 34-node GLM-style int4 decode path,
where MLAS replaces handwritten per-output VNNI work with its tuned dispatch.

This requires no new op registration and leaves `+85` unchanged.  It must
not route through the just-merged f32 direct-output/PackedB code: those are
`MatMul` f32 buffers (`kernels/matmul.rs:44-90, 125-136`), whereas QNBit needs
its own packed B and normally writes f32 C directly.  Preserve direct f32 C
to avoid an intermediate allocation.

## Option B: native f16/bf16 GEMM

**Wiring.** A minimal f16-only experiment would add `halfgemm.cpp` to
`build.rs`, a C shim around `MlasHalfGemmBatch`/pack APIs, Rust RAII packed-B,
and a MatMul/Gemm dtype dispatch that avoids `to_dense_f32_widen`.  It needs
separate f16 C output and f32 accumulation/rounding policy tests.  bf16 cannot
be bolted onto that signature: it needs a real BF16 kernel/API (or a distinct
backend), validation of ONNX accumulation semantics, and runtime ISA gates.

**Risk/impact.** This is harder and currently low-confidence for performance:
the vendor's generic half dispatch is not proven to exploit the host
AVX512-FP16, no x86 bf16 dense kernel is present, and adding the driver may
only replace explicit widening with a generic implementation.  It adds no ONNX
op (so no registry-count change), but it has wider correctness surface than A:
all dense f16/bf16 MatMul/Gemm layouts, output narrowing, batched/broadcast
handling, and constant cache representation must be preserved.  It also cannot
reuse f32 `PackedB`; cache f16/bf16 packed weights independently.

## Recommendation and reviewable plan

Build **Option A first**.  It targets demonstrated int4 LLM exports, the
vendor code and Sapphire-Rapids VNNI dispatch are already built, and it
requires a contained wrapper plus replacement of an existing hot path.
Option B is a vendor/kernel acquisition project before it is a wiring task.

1. Extend `crates/mlas-sys/vendor/shim.cpp` with a POD f32-QNBit API and add
   matching extern/safe owner types in `crates/mlas-sys/src/lib.rs`; make size,
   alignment, availability, pack, and execute explicit.
2. Add a focused `mlas-sys` test/probe for W4 block-32 f32 and verify the
   Sapphire Rapids VNNI availability result.
3. In `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs`, introduce an
   `OnceLock` MLAS packed-weight cache keyed by the existing constant-input
   contract.  Dispatch MLAS first only for exact supported standard layouts;
   retain every current fallback.
4. Add CPU-kernel numerical/cache tests and the ignored GLM Q4 engine
   regression; extend the existing kernel benchmark harness with warm/cold
   M=1/M>1 ORT comparisons.
5. Do not touch `kernels/mod.rs` registrations or the `+85` invariant.  If a
   later fallback needs a separate public op, register it deliberately and
   update the invariant/test comment in the same change.
