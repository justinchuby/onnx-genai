# CPU EP activation and norm gaps against ORT

This EP claims every node it supports. When it is selected it does not hand
work back to ORT's CPU EP, so a range where it is slower than ORT is a **kernel
to optimize**, not a node to give away.

This file is the list of those ranges. It is a work list, not a set of
thresholds — nothing here changes what the EP claims.

## How these were measured

Session-level interleaved A/B on one host: the same model run through ORT's CPU
EP and through this plugin in the same process, alternating, `ORT_DISABLE_ALL`
so no fusion rewrites the graph, p50 and p90 of whole-`Run` latency. Session
latency rather than kernel time, because per-node dispatch overhead is part of
what the user pays.

Host: AMD EPYC 9V74, 32 vCPU, `avx2 f16c fma`, AVX-512 masked off. Ratios are
**ours ÷ ORT**, so **below 1.00 means we are slower** and the row is a gap to
close.

## Open gaps

### Elementwise float32, single thread

| op | 1 K | 4 K | 64 K | 1 M | note |
|---|---|---|---|---|---|
| `Tanh` | 0.60 | 0.71 | 0.80 | 0.82 | AVX2 poly landed in #1037 |
| `Sigmoid` | 0.62 | 0.70 | 0.78 | 0.81 | AVX2 poly landed in #1037 |
| `Gelu` (tanh) | 0.64 | 0.72 | 0.77 | 0.79 | |
| `Gelu` (none, exact) | 0.67 | 0.70 | 0.74 | 0.76 | erf poly landed in #1074 |
| `Erf` | 0.66 | 0.69 | 0.72 | 0.74 | was 0.022–0.25 before #1074 |
| `Exp` | 0.74 | 0.75 | 0.88 | 0.92 | MLAS poly ported in #1093 |
| `Log` | 0.68 | 0.60 | 0.60 | 0.64 | still scalar `libm` |
| `Softplus` | 1.03 | 1.13 | 0.96 | 0.92 | wins small, loses large |
| `Relu` | 0.76 | 0.76 | 0.87 | 1.03 | |
| `Sqrt` | — | — | 1.1–1.9 | 1.1–1.9 | wins at 1 thread, **0.30 at 16** |

### Elementwise float16

ORT has no native f16 kernel for most of these; it casts to f32 around its own
kernel and still wins, which means the gap is in our conversion path as much as
in the math.

| op | 1 K | 4 K | 64 K | 1 M |
|---|---|---|---|---|
| `Exp` | 0.84 | 0.84 | 0.77 | 0.22 |
| `Log` | 0.75 | 0.66 | 0.60 | 0.59 |
| activation family (Tanh/Sigmoid/Gelu/Erf/Sqrt) | 0.59–0.96 across sizes | | | |

The `Exp` f16 0.22 at 1 M is the worst single cell in the table and is
conversion-bound, not math-bound.

### `com.microsoft` activations, float32

| op | range |
|---|---|
| `FastGelu` | 0.57–0.75 |
| `BiasGelu` | 0.72–0.74 |
| `QuickGelu` | 0.75–1.02 |

### `com.microsoft::RotaryEmbedding`, float32

Loses across the whole measured grid (decode and prefill shapes).

### Matmul family

`QLinearMatMul` with unsigned activations is 1.13× slower at one thread and
2.65× slower at sixteen (K=N=3584). The signed path was translated onto the
same MLAS `u8 × u8` kernel and wins 1.7–33×, so the residue is thread scaling,
not packing. Full matrix in `CPU_MATMUL_ASSIGNMENT.md`.

## The dominant root cause

Two effects explain most of the table, and neither is the transcendental math:

1. **Our elementwise kernels are single-threaded; ORT's are not.** ORT splits
   an elementwise tensor across its intra-op pool. This is why the float32
   activation family sits on a flat ~0.7–0.8 plateau that barely moves when the
   polynomial gets better, and why `Sqrt` wins 1.9× at one thread and loses at
   0.30× at sixteen. Closing this is worth more than any further polynomial
   work.

2. **A fixed per-node plugin overhead of roughly 1.2 µs.** Visible as a flat
   ~0.75–0.8 ratio at `n = 1`, where neither side does arithmetic. Below
   roughly 10 K elements no elementwise kernel can earn it back through
   arithmetic alone, so the small-size fix is dispatch cost, not the kernel.

## Closed

| gap | was | now | how |
|---|---|---|---|
| `Erf`, `Gelu(none)` f32 | 0.022–0.25 | 0.66–0.76 | ported MLAS's erf polynomial to AVX2 (#1074) |
| `Sqrt` f32 | 0.03 | 1.1–1.9 at 1 thread | routed through the bulk SIMD path (#1048) |
| `Exp` f32 | 0.14 | 0.82–0.92 | ported `MlasComputeExpVector` to AVX2 (#1093) |
| bf16 ⇄ f32 conversion | scalar | 1.6–2.8× | bulk AVX2 conversion (#1041) |
| `com.microsoft` activations unreachable | never ran | reachable | shape-inference table entries (#1082) |

## Activations with no kernel at all

Distinct from the sections below: these are not declines, they are missing
features. ORT runs them because this EP has nothing to run.

- **`Celu`** — ONNX opset 12. `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
- **`Mish`** — ONNX opset 18. `x * tanh(softplus(x))`.

Both are cheap to add on top of the existing `exp`/`tanh` AVX2 primitives, and
`activation_and_norm_ops_clear_every_capability_filter` has a comment naming
them so they are added to that test the moment a kernel lands.

## Ops that still reach ORT despite being supported

Removing the performance-based decline is not by itself enough to guarantee
that selecting this EP keeps work off ORT's CPU EP. `GetCapability` runs
**three** independent fail-closed filters, and a claim has to clear all of
them. Deleting the policy addressed one; the other two were each found by
review, after this document had already claimed the job was done.

**Filter 1 — the assignment policy.** Removed. This was the performance-based
decline.

**Filter 2 — the shape table.** Drops any claim containing a node whose
`ShapeInference::for_node` returns `Declined`
(`onnx-runtime-ep-plugin/src/ep.rs`). That table matches on op name and ends in
`_ => Declined`, so **an op the CPU EP registers a kernel for, but which is
absent from the table, is silently handed to ORT** — whatever `supports_op`
answers. This is the same mechanism that made the `com.microsoft` activations
unreachable until #1082. The activation, trigonometric and hyperbolic families
were in that gap and are now covered: `Sin`, `Cos`, `Tan`, `Asin`, `Acos`,
`Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`, `ThresholdedRelu`, `Swish`,
`Silu`, `PRelu`, `GroupNormalization`.

**Filter 3 — the dtype filter.** `node_passes_dtype_filter` looks the node's op
up in the plugin's `KernelRegistryEntry` list and returns `false` when there is
no entry. That list came from `build_cpu_registry_with_descriptors`, which
recorded keys as they were registered — but `register_cnn_ops` takes `&mut
OpRegistry` and wrote *past* the recorder. Eighteen ops were therefore in the
registry and absent from the descriptors: `PRelu`, `BatchNormalization`,
`InstanceNormalization`, `GroupNormalization`, `Conv`, the pooling family,
`Resize`, `GridSample`, `AffineGrid`, `Col2Im`, `CenterCropPad`,
`SpaceToDepth`, `ConvTranspose`. Every one was claimed by `supports_op` and
then dropped at capability time.

`PRelu` is the sharpest case: it cleared the shape filter (added above) and was
still declined by the dtype filter, so the pure-Rust inventory tests passed
while real ORT ran the node. Descriptors are now derived from
`OpRegistry::keys()` rather than a parallel recorded list, which makes the two
sets identical by construction, and
`every_registered_op_has_a_kernel_registry_entry` holds them together.

The lesson is that an inventory test is only as good as its source of truth.
Two rounds of review passed on tests that enumerated the wrong set.

**Filter 3, second failure mode — the dtype *union* is per-op, the kernel's rule
is per-slot.** An entry advertises one set of dtypes for the whole op and every
input slot is tested against it, so a mixed-dtype op is decided by whichever
constraint is written last. `MatMulNBits` showed the union being too *wide*.
The attention family shows it being too *narrow*, which is worse because it is
silent: the ops map to `FLOAT_DTYPES`, but `RotaryEmbedding`'s `position_ids` is
int64 and `GroupQueryAttention`'s `seqlens_k` / `total_sequence_length` are
int32, so the integer slots failed the float test and **both ops were handed to
ORT on every real session** while every pure-Rust test passed.
`input_dtype_constraints_for_op` (`onnx-runtime-ep-cpu/src/kernels/mod.rs`) now
carries per-slot tables for `RotaryEmbedding` in both domains — their slot
orders differ — plus `GroupQueryAttention` (including `position_ids` at slot
**9**, which a first pass missed because it only listed the two slots the
fixtures happened to exercise), `MultiHeadAttention`, `com.microsoft::Attention`,
`PackedMultiHeadAttention` (int32 `token_offset` / `cumulative_sequence_length`)
and `QMoE` (uint8-packed experts and zero points).

The union can also be too narrow without any mixed-dtype slot at all: `MoE`
advertised **float32 only** while its kernel accepts float16 and bfloat16 too,
so the f32 fixture passed and every production half-precision mixture was
declined. A single dtype's worth of coverage is not coverage.

Four of those six were found only after review asked for one real-ORT fixture
per rescued op. **An inventory test that never opens a session cannot see this
filter at all**, which is the same lesson as above arriving a third time.

**Filter 4 — the kernel factory, which runs after assignment.** Clearing all
three filters only gets as far as `get_kernel`. ORT stamps **schema defaults**
onto a node before an EP sees it, so a factory that rejects an attribute it does
not support must use ORT's default, not ONNX's zero: the contrib default for
`smooth_softmax` is `-1`, and testing it for `!= 0` rejected every
`GroupQueryAttention` node ORT ever resolved. A rejection here is a hard session
failure rather than a fallback, so it is the most expensive of the four — and
that cuts both ways. Making an op *reachable* can break a model that used to
work: once `QMoE` was claimed, a column-wise (`block_size` absent) node that ORT
had been running fine reached our factory and killed `CreateSession`. Review
then found the same defect on `use_sparse_mixer=1` (Phi-3.5-MoE, GRIN-MoE) and
on `smooth_softmax=1` (Gemma-style attention sink) — both of which ORT's CPU EP
runs today — so it is a class, not three bugs.

**Any capability limit a factory enforces must be mirrored in `supports_op`**,
where a decline is still recoverable. The mirror is now structural rather than
duplicated: each kernel's attribute validation lives in one function and the
claim-time guard is that same function's error, so a new factory limit cannot
fail to appear at claim time. `provider::tests::every_factory_attribute_rejection_is_mirrored_at_claim_time`
asserts both halves for eleven hostile nodes and fails if they diverge.

The attention, MoE and KV-cache ops are now covered end-to-end by
`plugin_ort_e2e`'s `ASSIGNMENT_FIXTURES` (38 graphs, all on our EP with
`session.disable_cpu_ep_fallback=1`) and by
`rope_and_gqa_execute_on_our_ep_and_match_ort_numerics`, which checks the two
recovered ops against ORT's own kernels rather than only checking placement.

**64 registered ops remain in the shape-table gap.** The list is not prose: it is asserted
exactly by `every_registered_op_has_a_shape_rule_or_is_a_known_gap`
(`crates/onnx-runtime-ep-cpu-plugin/tests/shape_inference_coverage.rs`), which
enumerates `OpRegistry::keys()` — the same set `supports_op` consults — and
fails if an op is registered without a shape rule *or* a shape rule is added
without updating the list. Read that test for the current contents; the summary
below is the shape of it.

### 1. Data-dependent — correctly declined (20)

The output shape is a function of an input's *values*, not its shape, so it
cannot be inferred at capability time: `Compress`, `NonZero`, `Unique`,
`NonMaxSuppression`, `ConstantOfShape`, `Expand`, `Range`, `Tile`, `OneHot`,
`Pad`, `TopK`, `Split`, `Unsqueeze`, `Resize`, `AffineGrid`, `Col2Im`,
`CenterCropPad`, `BlackmanWindow`, `HammingWindow`, `HannWindow`.

Most carry a **constant initializer** in practice — `Unsqueeze`'s `axes`,
`Pad`'s `pads` and `Resize`'s `sizes` are almost always constant-folded inputs
in a real graph. A pass that resolves initializer values at capability time
would let us claim them. Until one exists, declining is the honest answer, and
`Unsqueeze` in particular is a frequent op we are handing over. `Split` is only
*partly* undecidable: the equal-split case follows from `num_outputs` alone.

### 2. Internal fusion ops — never candidates (10)

Emitted by our own fusion passes, so they are created *after* capability and
never appear in an input graph: `FusedGemm`, `FusedAttention`,
`FusedMatMulBias`, and the `pkg.nxrt` ops `BlockQuantizedMatMul`,
`BlockQuantizedMoE`, `CompressedSparseAttention`, `IndexShare`,
`PackedVarlenAttention`, `SparseKvGather`, `VarlenAttention`.

### 3. Inferrable but unwritten — this is the work (34)

Real ops, in real input graphs, that we have a kernel for and hand to ORT
anyway because nobody wrote the rule.

**Shape-preserving; one line each.** `BitwiseNot`, `CastLike`, `CumProd`,
`CumSum`, `DequantizeLinear`, `EyeLike`, `QuantizeLinear`, `ScatterElements`,
`ScatterND`, `Trilu`.
`QuantizeLinear`/`DequantizeLinear` are the ones that matter: they bracket
every operator in a quantised model, so handing them over also fragments the
partition around everything between them.

**Pooling and CNN geometry.** `MaxPool`, `AveragePool`, `GlobalMaxPool`,
`GlobalAveragePool`, `GlobalLpPool`, `LpPool`, `ConvTranspose`, `GridSample`,
`SpaceToDepth`. All inferrable from `kernel_shape`, `strides`, `pads`,
`dilations` and `ceil_mode` — exactly what `build_conv` already does for
`Conv`, which *is* covered.

**Inferrable from attributes or a fixed rule.** `ArgMax`, `ArgMin`, `Constant`,
`DFT`, `DynamicQuantizeLinear`, `Flatten`, `GatherElements`, `Size`.

`QLinearMatMul` left this group when `main` gave it a rule; the coverage test
caught the stale entry on the merge, which is the drift check working.

**Contrib and model ops needing a real rule.** `com.microsoft::Attention` —
packed QKV with a different signature from `ai.onnx::Attention`, so the
opset-23 arm deliberately does not cover it; this is the single highest-value
entry, because it is *the* attention op in exported GenAI models.
`com.microsoft::MoE`, `com.microsoft::QMoE` and
`com.microsoft::PackedMultiHeadAttention` are likewise read from exported
models. `LinearAttention` (both domains) and `com.microsoft::CausalConvWithState`
are the Qwen3.5 / Qwen3-Next hybrid linear-attention primitives — **ORT has no
kernel for these at all**, so declining them does not get a faster
implementation, it gets a load failure.

Groups 1 and 3 are why this EP cannot yet claim that no supported node reaches
ORT.
