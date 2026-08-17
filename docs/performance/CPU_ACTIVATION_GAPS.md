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

## Ops that still reach ORT despite being supported

Removing the performance-based decline is not by itself enough to guarantee
that selecting this EP keeps work off ORT's CPU EP. There is a second,
independent mechanism.

`GetCapability` runs a fail-closed filter that drops any claim containing a node
whose `ShapeInference::for_node` returns `Declined`
(`onnx-runtime-ep-plugin/src/ep.rs`). That table matches on op name and ends in
`_ => Declined`, so **an op the CPU EP registers a kernel for, but which is
absent from the table, is silently handed to ORT** — whatever `supports_op`
answers. This is the same mechanism that made the `com.microsoft` activations
unreachable until #1082.

The activation, trigonometric and hyperbolic families were in that gap and are
now listed: `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`,
`Asinh`, `Acosh`, `Atanh`, `ThresholdedRelu`, `Swish`, `Silu`, `PRelu`.

**66 registered ops remain in the gap.** They fall into three groups:

1. **Genuinely undecidable, correctly declined** — the output shape is
   data-dependent and cannot be inferred: `NonZero`, `Unique`, `Compress`,
   `NonMaxSuppression`.
2. **Internal ops that never appear in an input graph** — produced by our own
   fusion passes, so they are not candidates at `GetCapability` time:
   `FusedGemm`, `FusedAttention`, `FusedMatMulBias`, `LinearAttention`,
   `CausalConvWithState`, `MoE`, `QMoE`.
3. **Inferrable, but not yet written — this is the work.** `Split`, `Tile`,
   `TopK`, `Pad`, `Resize`, `Expand`, `Flatten`, `ArgMax`, `ArgMin`, `MaxPool`,
   `AveragePool`, the `Global*Pool` family, the whole `Reduce*` family,
   `QuantizeLinear`, `DequantizeLinear`, `DynamicQuantizeLinear`,
   `QLinearMatMul`, `ScatterND`, `ScatterElements`, `GatherElements`, `Range`,
   `Size`, `OneHot`, `CumSum`, `CumProd`, `Trilu`, `Constant`,
   `ConstantOfShape`, `CastLike`, `SpaceToDepth`, `Col2Im`, `ConvTranspose`,
   `GridSample`, `GroupNormalization`, `CenterCropPad`, `EyeLike`, `DFT`, the
   window functions, `AffineGrid`, `LpPool`, `BitwiseNot`.

Group 3 is the reason this EP cannot yet claim that no supported node reaches
ORT. Each entry needs real shape inference, and `QLinearMatMul` and
`GroupNormalization` are the ones most likely to matter for a transformer
workload.
