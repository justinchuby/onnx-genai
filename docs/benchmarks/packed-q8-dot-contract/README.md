# Grouped Q8-dot arithmetic contract

This records the original test-only study. The subsequent opt-in production
integration has its own [arithmetic and workspace contract](../../quantization/PACKED_ACTIVATION_ARITHMETIC.md).
The study's test-owned Vecs and numerical results remain historical evidence,
not production allocation or performance measurements.

The cold packed projection currently decodes a bounded f32 weight tile and
reuses it across the expert's tokens. A quantized-domain dot can remove that
weight representation, but quantizing its f32 activation is a separate numerical
choice. This experiment defines and tests that boundary before adding a backend.

The prototype is private and compiled only in the CPU crate's unit tests. It
adds no production dispatch, public API, option, owner, cache, worker pool or
quota callback. Its Vecs are test-owned storage, not an implemented workspace
contract. It is neither a throughput measurement nor ggml numerical parity.

## Primary-source basis

At the fixed PolyStrata revision
[`3e16421`](https://github.com/simplexai-labs/PolyStrata/blob/3e1642161198c50ff1abed4046f4e4f476099dbb/src/poly/cpu_experts.cpp),
`cpu_experts` chooses activation types from each weight type's CPU dot traits,
quantizes the input rows before the channel products, and reuses gate/up
activation storage when their dot types agree. Its worker/storage implementation
is not imported. This is an independently written Rust arithmetic experiment
using nxrt's existing Q8_0 weight representation.

The current nxrt baseline is the real shared row-streamed projection in
`block_quantized_matmul::linear_packed_grouped`, including the native or compiled
MLAS serial product. This test helper reaches the same decoded projection
machinery used by standalone packed MatMul and cold grouped experts; it does not
measure full operators, routing, activation functions or inference tokens.

## Explicit experimental arithmetic

For each activation block of at most 32 values, compute `d = max(abs(x))/127`,
round `x/d` to the nearest integer (ties away from zero), and store a half-precision
scale with 32 signed codes. Padding is zero. A channel computes i32 dot products
of those codes with the existing signed Q8_0 weight codes, then combines block
scales and partial sums in f32. Finite nonzero scales that cannot be stored as
finite nonzero f16 and non-finite activations are explicitly rejected by this
prototype with row/block context and an f32-route remedy.

The grouped schedule quantizes each input block once outside the output-channel
loop. Test-local work counters verify `rows * ceil(K/32)` successful activation
conversions and `rows * outputs * ceil(K/32)` integer block dots. Changing the
output-channel count does not repeat activation conversion. These are work
counts, not performance or grant/refund measurements.

## Numerical evidence

Four tests cover literal rounding and padding, grouped/partial-block arithmetic,
unsupported activation ranges and empty outputs, and a concrete output-winner
change. The independent oracle reconstructs each logical operand in f64 and
accumulates element-wise, rather than reusing the prototype's block integer dot.
It checks both f32 rounding error against quantized operands and the separate
activation quantization error bound against the original f32 input.

The adversarial fixture uses one 32-value input block, with `x[0] = 1` and
`x[1] = 2^-10`. One Q8_0 weight row selects `x[1]` at scale one; another selects
`x[0]` at scale `2^-11`:

| Route | Output 0 | Output 1 | Winner |
|---|---:|---:|---|
| Current f32 activation projection | `2^-10` | `2^-11` | 0 |
| Experimental Q8 activation projection | 0 | positive, approximately `2^-11` | 1 |

The small activation rounds to zero because the large value sets the block's
scale. Thus even finite ordinary values can change the winner, rather than just
rearrange floating-point accumulation noise. This is a projection-level
counterexample, not a claim that an end-to-end token test ran. Automatically
replacing current f32 activation math would violate the existing route's
numerical contract.

## Integration decision and next gates

Keep production execution unchanged. A future quantized-dot path needs an
explicit activation-arithmetic contract understood by graph/EP claiming, not an
implicit consequence of compressed weight storage. An EP that does not implement
that contract must decline it with a reason. Do not add a CPU-only attribute that
another EP can ignore while claiming the same graph.

The implementation boundary remains the shared packed projection recipe used by
MatMul and MoE. Prepare activation storage once for the projection/group and
reuse it across channels; equal gate/up dot types can later share their prepared
activation representation. Account the real aligned lane through the existing
whole-operation workspace, using the maximum of mutually exclusive decoder and
quantizer demands. Do not add per-row/per-expert admissions, another cache or a
worker pool. Mixed FC formats must resolve independently while preserving the
existing gate/bias/scatter order.

Before production adoption, require caller-lane/refusal tests, explicit mixed
format eligibility, independent numerical tests across the supported formats,
SIMD capability/portable controls, and same-binary wall/CPU measurements for both
single-token and grouped inputs. Hold the shared host lock across all benchmark
and null arms; report actual source/build provenance and complete output checks.
No performance or latency conclusion is established by this experiment.

## Reproduction

List and run the four named tests (a listing alone does not prove execution):

```sh
cargo test --offline -p onnx-runtime-ep-cpu --lib packed_dot_contract:: -- --list
cargo test --offline -p onnx-runtime-ep-cpu --lib packed_dot_contract::
cargo test --offline -p onnx-runtime-ep-cpu --features mlas --lib packed_dot_contract::
```

Use the coordinated host lock for the full CPU test matrix. The MLAS feature is
only the existing research reference, not a new runtime dependency. The prototype
has no CUDA path and these tests establish no GPU behavior.
