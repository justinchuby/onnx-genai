# Sub-4-bit weight quantization and sparse MoE execution

**Status:** design plus correctness-first CPU increments for affine int2
`MatMulNBits` and native `BlockQuantizedMatMul` with MXFP4 and IQ4_NL. The
IQ1/IQ2/IQ3 grids, fused sub-4-bit MoE execution, offload, and CUDA kernels
remain follow-ups.

## 1. Motivation

GLM-5.2 has 744B total parameters but activates 40B per token. Unsloth reports
that its dynamic 1-bit package fits in about 223 GB total memory and its 2-bit
package in about 245 GB; the full model is roughly 1.5 TB. The guide explicitly
targets CPU/GPU memory offload and demonstrates `UD-IQ1_S` and `UD-IQ2_M`
packages [U1]. Dynamic GGUF does not mean that every tensor has the same type:
Unsloth chooses a quantization format per layer and preserves important layers
at higher precision [U2].

The runtime therefore needs two distinct capabilities:

1. ordinary linear 2-bit block quantization, which fits the existing
   `com.microsoft::MatMulNBits` affine contract; and
2. native block formats such as IQ and MXFP4, whose encoded values are not
   affine integers and must not be silently interpreted as `(q-zp)*scale`.

The latter is especially important for sparse MoE. Loading or expanding every
expert defeats the memory benefit even though only top-k experts execute.

## 2. Exact llama.cpp block formats

The descriptions below follow llama.cpp commit
`b15ca938ad00aa6b3ee6c2edda7363fd02826b18`. The IQ1/IQ2/IQ3 formats below use
a 256-weight super-block (`QK_K=256`); IQ4_NL uses 32 weights. “bpw” includes
scale and metadata bytes, not just nominal index bits. The serialized field
order is the C structure order in `ggml-common.h`; consumers must not invent a
different packing.
The corresponding row dequantization and importance-quantization entry points
are declared in `ggml-quants.h` [L0].

### 2.1 Summary

| GGUF type | Serialized block for weights | Bytes | Effective bpw | Scale / offset layout |
|---|---:|---:|---:|---|
| `IQ1_S` | `fp16 d; u8 qs[32]; u16 qh[8]` | 50 | 1.5625 | One `d`; each 32 weights has a 3-bit odd multiplier and one ±0.125 grid shift |
| `IQ1_M` | `u8 qs[32]; u8 qh[16]; u8 scales[8]` | 56 | 1.75 | Global fp16 `d` is bit-sliced into the high nibbles of `scales`; two 3-bit odd multipliers per 32 weights; ±0.125 shifts |
| `IQ2_XXS` | `fp16 d; u16 qs[32]` | 66 | 2.0625 | One `d`; one 4-bit scale per 32 weights is packed with grid/sign metadata |
| `IQ2_XS` | `fp16 d; u16 qs[32]; u8 scales[8]` | 74 | 2.3125 | One `d`; two 4-bit scales per 32 weights, one per 16 |
| `IQ2_S` | `fp16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]` | 82 | 2.5625 | One `d`; two 4-bit scales per 32 weights, one per 16 |
| `IQ3_XXS` | `fp16 d; u8 qs[96]` | 98 | 3.0625 | One `d`; one 4-bit scale per 32 weights is packed with sign metadata |
| `IQ3_S` | `fp16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]` | 110 | 3.4375 | One `d`; one 4-bit odd multiplier per 32 weights |
| `IQ4_NL` | `fp16 d; u8 qs[16]` for 32 weights | 18 | 4.5 | One fp16 scale and a fixed 16-entry non-linear scalar codebook |
| `MXFP4` | `u8 e; u8 qs[16]` for 32 weights | 17 | 4.25 | One E8M0 power-of-two scale and 32 E2M1 values |

These sizes and layouts are declared directly by llama.cpp [L1]. None of the
IQ formats stores an affine minimum. Their apparent “offset” is instead encoded
by a sign mask or, for IQ1, by a ±`IQ1S_DELTA` grid shift. Treating the bytes as
linear int1/int2/int3 corrupts the tensor.

### 2.2 IQ grid mechanism

Importance quantization selects a short vector from a fixed, format-specific
grid, applies signs and a small block scale, and reconstructs multiple weights
at once. The imatrix influences grid selection during quantization, but the
matrix is not needed at inference.

- **`IQ2_XXS`:** each 32-weight group contains four 8-weight grid vectors.
  Each vector has an 8-bit index into a 256-entry grid and a 7-bit index into a
  sign table. The remaining high nibble supplies the 32-weight scale. The
  decoder uses
  `db = d * (0.5 + scale4) * 0.25` [L2].
- **`IQ2_XS`:** each 16-bit word stores a 9-bit index into a 512-entry
  8-weight grid and a 7-bit sign-table index. One scale byte per 32 weights
  contains two 4-bit scales, one per 16 weights; the same
  `d * (0.5 + scale4) * 0.25` rule applies [L3].
- **`IQ2_S`:** each 8-weight vector uses an 8-bit low grid index plus two high
  bits from `qh`, selecting one of 1024 grid entries. It stores an explicit
  8-bit sign mask per vector and the same two nibble scales per 32 weights
  [L4].
- **`IQ3_XXS`:** each 8-weight vector is made from two 4-value vectors selected
  by 8-bit indices into a 256-entry grid. Four 7-bit sign-table indices and a
  4-bit scale share four metadata bytes per 32 weights. Its scale is
  `db = d * (0.5 + scale4) * 0.5` [L5].
- **`IQ3_S`:** each 4-value vector uses a 9-bit index into a 512-entry grid.
  Explicit sign bytes cover each 8 weights. Each 32 weights uses
  `db = d * (1 + 2*scale4)` [L6].
- **`IQ1_S`:** each 8-weight vector uses an 11-bit index into the 2048-entry
  `iq1s_grid`. A 16-bit `qh` word per 32 weights holds four 3-bit high index
  fragments, a 3-bit scale, and a shift-sign bit. Reconstruction is
  `d * (2*scale3+1) * (grid + delta)`, where `delta` is ±0.125 [L7].
- **`IQ1_M`:** uses the same 2048-entry grid. `qh` contributes three high index
  bits plus a shift-sign bit for every 8 weights. The global fp16 scale is
  reconstructed from four high nibbles in the `scales` byte array, and each
  32-weight group has two independent 3-bit odd scale multipliers [L8].

The fixed grid table sizes, E2M1 value table, and `IQ1S_DELTA=0.125` are in
`ggml-common.h` [L9]. A runtime implementation should import or generate these
tables from one audited source and test byte-exact vectors from llama.cpp.

### 2.3 MXFP4

OCP MXFP4 uses blocks of 32 elements. Each element is FP4 E2M1
(sign, two exponent bits, one mantissa bit), and the block shares one unsigned
E8M0 scale. Storage is `32*4 + 8 = 136` bits, or 4.25 bpw [O1].

The E2M1 finite magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. llama.cpp stores
an equivalent integer table `{0,1,2,3,4,6,8,12}` and uses its half-scaled E8M0
conversion [L9][L11]. Its 17-byte block places the E8M0 byte first. In each payload
byte, the low nibble encodes weight `j` and the high nibble weight `j+16`, not
the immediately adjacent weight [L10]. That llama.cpp/GGUF byte layout is a
serialization detail; an ONNX op may use separate E8M0 scales, but Mobius must
then transcode and test the permutation.

E8M0 byte `0xff` is the OCP NaN encoding. The CPU implementation propagates it
as NaN; llama.cpp's GGUF quantizer does not emit it and its optimized helper
explicitly omits NaN handling. Bytes `0x00..0xfe` decode exactly as
`2^(e-127)`, including the `2^-127` subnormal at zero.

### 2.4 IQ4_NL

IQ4_NL is included as the first concrete IQ/codebook implementation because it
is the smallest independently auditable llama.cpp non-linear format. Each
32-weight block is an fp16 little-endian scale followed by 16 bytes. As with
MXFP4, byte `j` stores weight `j` in the low nibble and weight `j+16` in the
high nibble. The exact llama.cpp codebook is
`{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}` and each
reconstructed value is `d * codebook[q]` [L12].

This extends the initial schema sketch beyond the sub-4-bit IQ1/IQ2/IQ3 list:
IQ4_NL is not itself sub-4-bit, but landing it validates the format decoder,
native-block validation, fp16 scale handling, and explicit unsupported-format
gating without importing a large vector-grid table prematurely.

## 3. Existing ONNX contracts and the CPU prototype

### 3.1 `com.microsoft::MatMulNBits`

The contrib schema computes:

```text
B_dequant = (B_quant - zero_point) * scale
Y = A @ B_dequant
```

For attributes `K`, `N`, `bits`, and `block_size`, standard packed `B` has shape
`[N, ceil(K/block_size), block_size*bits/8]`. Bits are packed from least
significant to most significant within each byte. Scales have shape
`[N, ceil(K/block_size)]`. Packed zero points have shape
`[N, ceil(ceil(K/block_size)*bits/8)]`; when absent, the default is
`2^(bits-1)` [R1].

Although one sentence in the generated documentation says “2 to 8”, the
attribute and current CPU implementation support the discrete set `{2,4,8}`;
ORT main enforces that set [R1][R2]. There is no schema-compatible
`bits=1` today.

Before this increment, our CPU EP only accepted `bits=4`. Its optimized
symmetric block-32 decode path interprets each nibble as
`w=(q-8)*scale`, optionally quantizes the activation to int8 for
`accuracy_level=4`, and uses VNNI where available. Batched f32 work reaches the
shared GEMM seam and can use oneDNN.

This increment adds the standard **linear** `bits=2` layout:

- block size remains any supported power of two ≥16; the proof uses 32;
- four 2-bit codes occupy each byte, low bits first;
- absent zero point means `zp=2`;
- dequantization is `(q-2)*scale`; and
- computation is correctness-first f32 GEMV/GEMM after dequantization.

The int4 VNNI/int8 paths are deliberately gated to `bits=4`. A 2-bit model can
therefore never be misread by the nibble kernel. Unit tests cover a partial
final K block, batched matmul parity against an independently dequantized f32
reference, and explicit low-bit-first unpacking.

### 3.2 What fits and what does not

| Format | `MatMulNBits`? | Reason |
|---|---|---|
| Linear symmetric/asymmetric int2 | **Yes** | Exact affine integer model; use `bits=2` |
| Linear int1 | **Not yet** | Schema and ORT kernels do not accept `bits=1` |
| IQ1/IQ2/IQ3 | **No** | Grid indices, sign tables, odd subscales, and IQ1 delta are not affine scalar integers |
| MXFP4 | **No, not as integer NBits** | E2M1 values with E8M0 scale are floating-point microscaling |

## 4. Recommended operator design

Use both an existing affine op and a native-block op; do not overload one
attribute until “bits” changes meaning.

### 4.1 Linear path: retain `MatMulNBits`

Mobius should emit `com.microsoft::MatMulNBits(bits=2)` for true linear int2
weights when the target EP advertises it. This reuses the published schema,
shape inference, tooling, and ORT kernels. Add `bits=1` only through an upstream
schema revision with an agreed default zero point and packing; a private
interpretation would not be portable.

For performance, use ORT/MLAS packing and kernels as the reference. In our CPU
EP, the current f32 dequant path is the oracle. Prefill can feed the existing
oneDNN f32 GEMM after dequantization; decode ultimately needs a direct packed
2-bit GEMV because materializing f32 weights is memory-bandwidth hostile.

### 4.2 Native path: `BlockQuantizedMatMul`

Incubate a private op, then propose it upstream:

```text
com.github.onnxruntime.genai::BlockQuantizedMatMul(
    A, packed_B, optional_bias
) -> Y

attributes:
    K: int
    N: int
    format: string
      # iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s,
      # iq3_xxs, iq3_s, iq4_nl, mxfp4
    block_layout_version: int = 1
```

`packed_B` is an opaque `uint8` tensor shaped
`[N, ceil(K/QK), block_bytes]`, where `(QK, block_bytes)` is fixed by `format`.
For IQ1/IQ2/IQ3, `QK=256`; IQ4_NL uses `(QK, block_bytes)=(32,18)`; GGUF
MXFP4 uses `(32,17)`. Keeping the exact native block makes external-data slices
mmap-able and avoids separating/recombining embedded metadata. The kernel
defines unused values in the final native block as padding, validates the exact
shape and byte count, and owns the codebook.

The op name intentionally says “block quantized”, not “NBits”: MXFP4 and IQ
indices are semantic formats, not merely bit widths. A future schema can add
standardized layouts without changing the meaning of old models.

The CPU v1 implementation now registers this op in
`com.github.onnxruntime.genai`, accepts f32 `A`, native uint8 `packed_B`, and an
optional f32 bias, then dequantizes to f32 and uses the shared CPU GEMM. MXFP4
and IQ4_NL are implemented. IQ1/IQ2/IQ3 and IQ4_XS are recognized but fail
kernel creation with a clear unsupported-format error; no incomplete decoder
can silently produce weights.

For MXFP4 interoperability, also support a lowering between this GGUF-native
layout and ORT's current `QMoE(quant_type="fp4")` representation, which uses
packed E2M1 weights, separate float8e8m0 block-scale tensors, and per-expert
global scales [R4]. The lowering must be covered by byte-level and numeric
tests.

## 5. Fused sparse-MoE design

ORT has `com.microsoft::MoE` for floating weights and `com.microsoft::QMoE`
for quantized expert-major weights [R3][R4]. Current `QMoE` supports integer
2/4/8-bit weights. ORT main also defines `quant_type="fp4"` and
`"wfp4afp8"` for MXFP4, including E8M0 block scales and global scales [R4].
It does not define the llama.cpp IQ grids.

Mobius already models the right high-level contract: keep model-specific router
math explicit, then use a fused MoE op for top-k selection, dispatch, expert
FFNs, and weighted combination. The float CPU reference kernel in this
repository follows the current `MoE` positional contract.

For IQ and GGUF-native MXFP4, add a `BlockQuantizedMoE` sibling rather than
expanding hundreds of expert `BlockQuantizedMatMul` nodes:

```text
BlockQuantizedMoE(
    hidden, selection_scores,
    fc1_packed, fc2_packed, optional_fc3_packed,
    optional_aggregation_weights,
    optional_biases
) -> output

attributes:
    top_k
    fc1_format, fc2_format, fc3_format
    activation_type, swiglu_fusion
    normalize_routing_weights
    block_layout_version
```

Weights are expert-major:
`[experts, output_features, ceil(input_features/QK), block_bytes]`.
Selection and aggregation remain separate tensors so DeepSeek-style
bias-corrected/noaux routing does not accidentally change combine weights.

Execution for one admitted token batch is:

1. compute exact top-k expert IDs from `selection_scores`;
2. union selected IDs across all token rows;
3. sort/group token rows by expert;
4. acquire only those immutable expert weight slices from the expert store;
5. run grouped FC1/gate, activation, and FC2 directly from compressed blocks;
6. scatter-add outputs using `aggregation_weights`; and
7. release expert leases after the stream/event completes.

For decode, grouped GEMV over selected experts is the primary kernel. For
prefill or a continuous batch with many rows per expert, dequantized panels may
feed oneDNN grouped/batched GEMM on CPU; ORT's QMoE and MLAS kernels should be
the implementation reference. CUDA should stage selected block ranges into a
stream-ordered expert cache and use a direct IQ/MXFP4 kernel, not expand entire
experts to fp16/f32.

This fusion is also the memory boundary. A decomposed graph encourages the
executor to map or upload every expert initializer. The fused kernel can page
expert-major external-data ranges from disk to host RAM to VRAM, share one load
across all routed rows, and expose hit/miss/bytes metrics without changing
routing or precision.

## 6. Mobius EP-capability sketch

Mobius already centralizes EP behavior in `EpCapabilities`, makes it available
during graph construction through `ep_capabilities()`, and uses
`supports_fused_moe` plus `default_int4_accuracy_level` instead of scattering
EP-name checks [M1][M2]. Extend that mechanism with semantic format
capabilities:

```python
@dataclass(frozen=True)
class WeightQuantCapabilities:
    matmul_nbits_bits: frozenset[int] = frozenset({4})
    block_quant_matmul_formats: frozenset[str] = frozenset()
    block_quant_moe_formats: frozenset[str] = frozenset()
    mxfp4_layouts: frozenset[str] = frozenset()
    # e.g. {"gguf_interleaved_v1", "ort_separate_e8m0_v1"}

@dataclass(frozen=True)
class EpCapabilities:
    ...
    weight_quant: WeightQuantCapabilities = WeightQuantCapabilities()
```

Emission policy:

1. preserve a source IQ/MXFP4 tensor only when the active EP advertises the
   exact format and layout;
2. use fused `BlockQuantizedMoE` only when all expert formats are supported;
3. otherwise use supported `MatMulNBits(bits=2/4/8)` after a tested,
   semantics-preserving conversion;
4. otherwise dequantize or reject according to an explicit export policy;
5. never silently relabel grid bytes as linear int2.

The package metadata should record the required op domain/version, format, and
layout version. Runtime capability negotiation can then reject an incompatible
variant before allocating hundreds of gigabytes.

## 7. Delivery phases

1. **Landed:** CPU `MatMulNBits(bits=2)` f32 correctness baseline and parity
   tests.
2. Add Mobius capability flags and a linear-int2 export/e2e fixture.
3. **Landed:** private `BlockQuantizedMatMul` v1 CPU baseline with GGUF-native
   MXFP4 and IQ4_NL, exact block validation, optional bias, ONNX IR fixture, and
   numeric/reference tests.
4. Import llama.cpp golden blocks and audited grid tables for IQ1/IQ2/IQ3 and
   IQ4_XS; add GGUF-to-ORT MXFP4 layout parity.
5. Add fused CPU `BlockQuantizedMoE`, expert-major external-data slicing, and
   grouped decode GEMV; use oneDNN for sufficiently large dequantized GEMMs.
6. Add direct CUDA IQ/MXFP4 kernels and a stream-ordered expert residency cache.
7. Upstream the schema/kernels to ONNX Runtime and wire Mobius's GLM-5.2 and
   DeepSeek-V4-Flash exports.

## 8. Draft ONNX Runtime issue — do not file

### Title

Native IQ1/IQ2/IQ3 and standalone MXFP4 MatMul + sparse QMoE support for
memory-offloaded huge MoE models

### Body

Large sparse-MoE models now exceed practical single-machine memory at fp16 and
even int4. For example, GLM-5.2 has 744B total parameters / 40B active and its
community GGUF distributions rely on importance-quantized IQ1/IQ2/IQ3 formats
to fit roughly 223–360 GB instead of about 1.5 TB. Efficient inference must
preserve those compressed expert blocks and load only top-k selected experts;
expanding all experts to fp16/fp32 or lowering them to hundreds of ordinary
MatMuls defeats both capacity and bandwidth goals.

`com.microsoft::MatMulNBits` is the right contract for affine integer
quantization and already supports 2/4/8-bit weights. It cannot represent
llama.cpp IQ formats: IQ blocks use fixed vector grids/codebooks, sign tables,
sub-block scales, and IQ1 delta shifts. They are not
`(integer-zero_point)*scale`. MXFP4 is likewise E2M1 data with E8M0
microscaling, not integer NBits.

`com.microsoft::QMoE` already provides the desired fused routing/dispatch/FFN
shape and current main includes integer 2/4/8 plus MXFP4 quant types. The
remaining gaps are:

- a standalone MXFP4 weight-only MatMul contract and CPU/CUDA kernels;
- native IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, and IQ3_XXS/IQ3_S block formats;
- IQ support in fused sparse MoE, preserving separate selection and aggregation
  router tensors;
- efficient CPU 2-bit and IQ decode kernels rather than full-weight
  dequantization;
- expert-major packed layouts that allow mmap/range loading and top-k-only
  residency; and
- documented, versioned layouts plus reference vectors so exporters cannot
  silently disagree on bit packing.

Proposed direction:

1. keep affine int2 in `MatMulNBits`;
2. add a format-explicit block-quantized MatMul schema for IQ/MXFP4, with opaque
   fixed-size blocks and audited codebooks; and
3. extend `QMoE` or add a sibling block-quantized MoE op that consumes the same
   formats and performs top-k gather, grouped GEMV/GEMM, and weighted combine
   without materializing inactive experts.

We can contribute byte-exact llama.cpp reference vectors, exporter fixtures,
and a correctness-first CPU implementation. ORT's MLAS `MatMulNBits` and QMoE
kernels should remain the performance reference; oneDNN is applicable to
larger CPU GEMMs, while decode requires direct compressed GEMV.

### Suggested acceptance criteria

- documented schemas and shape inference for standalone and MoE use;
- CPU reference tests for every format against llama.cpp dequantization;
- packed-layout validation that rejects undersized/mismatched tensors;
- parity tests for partial/final blocks and separate router combine weights;
- no full expert-pool dequantization or upload on sparse decode; and
- CPU and CUDA performance follow-ups with stable prepacking/cache interfaces.

## References

- **[L0]** llama.cpp IQ/MXFP4 quantize and dequantize API:
  [`ggml-quants.h` lines 20-105](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.h#L20-L105)
- **[L1]** llama.cpp block structures:
  [`ggml-common.h` lines 210-438](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-common.h#L210-L438)
- **[L2]** IQ2_XXS decoder:
  [`ggml-quants.c` lines 2488-2513](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2488-L2513)
- **[L3]** IQ2_XS decoder:
  [`ggml-quants.c` lines 2516-2540](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2516-L2540)
- **[L4]** IQ2_S decoder:
  [`ggml-quants.c` lines 2543-2572](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2543-L2572)
- **[L5]** IQ3_XXS decoder:
  [`ggml-quants.c` lines 2575-2604](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2575-L2604)
- **[L6]** IQ3_S decoder:
  [`ggml-quants.c` lines 2607-2647](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2607-L2647)
- **[L7]** IQ1_S decoder:
  [`ggml-quants.c` lines 2650-2672](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2650-L2672)
- **[L8]** IQ1_M decoder:
  [`ggml-quants.c` lines 2675-2722](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2675-L2722)
- **[L9]** llama.cpp grid tables, E2M1 values, and IQ1 delta:
  [`ggml-common.h` lines 550-1140](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-common.h#L550-L1140)
- **[L10]** llama.cpp MXFP4 quantize/dequantize:
  [`ggml-quants.c` quantize lines 350-380](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L350-L380) and
  [dequantize lines 569-588](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L569-L588)
- **[L11]** llama.cpp half-scaled E8M0 conversion:
  [`ggml-impl.h` lines 475-498](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-impl.h#L475-L498)
- **[L12]** llama.cpp IQ4_NL block, codebook, and decoder:
  [`ggml-common.h` lines 446-452](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-common.h#L446-L452),
  [`ggml-common.h` lines 1119-1122](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-common.h#L1119-L1122),
  and [`ggml-quants.c` lines 2725-2741](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-quants.c#L2725-L2741)
- **[O1]** OCP,
  [Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **[R1]** ONNX Runtime `MatMulNBits` schema:
  [`ContribOperators.md` lines 3127-3228](https://github.com/microsoft/onnxruntime/blob/16ebc1d98b8a6d8b31823b42ee0b4b8b97ff1dac/docs/ContribOperators.md#L3127-L3228)
- **[R2]** ONNX Runtime CPU bit-width validation:
  [`matmul_nbits.cc` lines 125-136](https://github.com/microsoft/onnxruntime/blob/16ebc1d98b8a6d8b31823b42ee0b4b8b97ff1dac/onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc#L125-L136)
- **[R3]** ONNX Runtime `MoE` schema:
  [`ContribOperators.md` lines 3440-3548](https://github.com/microsoft/onnxruntime/blob/16ebc1d98b8a6d8b31823b42ee0b4b8b97ff1dac/docs/ContribOperators.md#L3440-L3548)
- **[R4]** ONNX Runtime `QMoE` schema:
  [`ContribOperators.md` lines 4895-5075](https://github.com/microsoft/onnxruntime/blob/16ebc1d98b8a6d8b31823b42ee0b4b8b97ff1dac/docs/ContribOperators.md#L4895-L5075)
- **[U1]** Unsloth,
  [GLM-5.2 — How to Run Locally](https://unsloth.ai/docs/models/glm-5.2.md)
- **[U2]** Unsloth,
  [Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs.md)
- **[M1]** Mobius commit `7a72880e31868f385e9598a74d95468069e1d8aa`,
  `src/mobius/_execution_providers.py`
- **[M2]** Mobius commit `7a72880e31868f385e9598a74d95468069e1d8aa`,
  `src/mobius/_build_context.py`
