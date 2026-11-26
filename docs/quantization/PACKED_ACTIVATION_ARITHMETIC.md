# Packed activation arithmetic

`pkg.nxrt::BlockQuantizedMatMul` and `pkg.nxrt::BlockQuantizedMoE` carry
`activation_arithmetic` independently of their compressed weight formats.
This is graph numerical semantics: it cannot change when a resource grant is
refused or when a dense weight cache becomes available.

| Arithmetic | ABI tag (`block_layout_version`) | CPU weights | CUDA |
| --- | --- | --- | --- |
| `float32` (default) | 1; omission means 1 | Existing supported formats | Existing supported formats |
| `q8_0` | Explicit integer 2 required | Q8_0 (`q8_0` or `gguf:q8_0`), `gguf:q4_0` | Declined before device initialization |

The explicit version-2 tag makes version-1 factories reject these graphs even
if they ignore unknown attributes. Version 2 with `float32`, version 1 with
`q8_0`, unknown modes and non-string modes are errors. Shared IR validation is
used by CPU, CUDA and shape inference. For MoE, every effective FC1/FC2/FC3
format must implement the selected arithmetic; projections may mix Q4_0 and
Q8_0. No exporter enables the new arithmetic automatically.

For example, a CPU packed MatMul with Q4_0 weights uses:

```text
format = "gguf:q4_0"
activation_arithmetic = "q8_0"
block_layout_version = 2
K = <reduction dimension>
N = <output channels>
```

Each activation row is split into blocks of at most 32 values. A block computes
`d = max(abs(x))/127` in f32, stores `d` in f16, rounds `x/d` to the nearest
integer with ties away from zero, and clamps codes to [-127, 127]. The stored
scale is used for reconstruction; codes are computed with the original f32
scale. Partial blocks have zero codes in every unused lane. All-zero blocks
have zero scale and zero codes. Non-finite activations, nonzero activation
scales that round to zero or infinity in f16, and non-finite weight scales
produce actionable errors. Q4_0 uses low 16/high 16 nibbles offset by -8;
Q8_0 uses signed byte codes. Each i32 block dot combines its f32 scale product
with the f32 sum; bias is added after all blocks.

Activation conversion occurs once per row/block per projection, outside the
output-channel loop. MoE retains its existing gather, activation/gate, bias,
routing and scatter semantics. Its separate FC projections currently convert
their activation independently, including gate and up; there is no cross-FC
conversion cache.

Execution uses the existing whole-operation Host workspace loan. A projection
needs `round_up(rows * ceil(K/32) * 34, 4)` activation bytes and, when both rows
and output channels exceed one, `rows * N * 4` column staging bytes. Empty rows
or output channels require neither region. MoE reserves the maximum complete
serial projection demand plus its existing separate FC banks and accumulator.
Standalone MatMul adds a disjoint `rows * N * 4` final-output staging region to
that complete projection layout and a separate packed-weight byte lane for
noncontiguous weights; all three regions are simultaneously live. Dense
hits initialize only final-output staging, while admission retains enough
projection and packed-byte capacity for a subsequent streamed fallback.
The byte lane is `round_up(N * ceil(K / block_elements) * block_bytes, 4)`
when packed_B needs copying, and zero for contiguous weights or empty rows.
Nonempty metadata preparation is Deferred: each launch resolves actual strides
before one complete admission. Cold hashing reads the validated logical source
directly; dense builders and streaming/Q8 consume the same admitted byte lane.
Dense-cache hits leave that lane untouched. No execution path materializes a
packed-weight Vec.
No per-token, per-row, per-block or per-expert quota requests are added.
The existing initialized f32 scratch proof covers the byte loan, including its
rounded tail; every exposed activation block is overwritten before use.

Contiguous, aligned F32 input, router logits, optional router weights and FC
biases in BlockQuantizedMoE are borrowed through their final read. Strided or
unaligned sources still materialize. Runtime metadata validation retains the
op's F32-only interface; reuse of the widening helper does not enable half or
F64 MoE inputs. Projection banks and the final accumulator keep publication
after every source read and successful expert, including when Y overlaps one of
these sources. These input borrows add no cache, pool or quota operations.

Q8 validates its complete workspace before materializing input payloads.
Missing, short or misaligned loans fail; precision never changes to recover.
Arithmetic failures can dirty private scratch/staging but do not publish a
partial tensor. A new prepared execution can reuse the same storage safely.
Q8 never expands weights into the dense-f32 cache, even with a large budget.
Default `float32` retains the existing dense-cache and decoded-row fast paths.

This opt-in mode can change model predictions: the
[numerical study](../benchmarks/packed-q8-dot-contract/README.md) includes a
two-channel argmax reversal. It is not ggml numerical parity, a lossless
optimization or a measured speedup. The initial integer dots are portable Rust
loops using the existing Rayon execution pool. Complete scratch demand may
increase for large groups; no universal memory or latency improvement is
claimed. CUDA execution and model-quality evaluation require separate evidence.
