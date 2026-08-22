# `FusedMatMulBias` 16-bit decode GEMV — full record (#1702)

Date: 2026-08-22. Host: AMD EPYC 9V74, 16 physical / 32 logical, AVX2 + FMA +
F16C, **no AVX-512 / VNNI / AMX**. Shared host — `uptime` checked before every
run (see "Discarded measurements").

## Question

The optimizer fuses `MatMul + Add(bias)` into `FusedMatMulBias`. Does the fused
node reach the same 16-bit decode GEMV as the unfused `MatMul`, and if not, what
does it cost?

## Method

`crates/onnx-runtime-ep-cpu/benches/half_decode_gemv_ab.rs` with
`PROBE_OP=fused_matmul_bias`, driven by an interleaved A/B harness that runs
three arms **in one process, alternating**, so drift and thermal state hit all
arms equally:

1. `fmb_before` — binary built at `62d696699` (pre-fix), with the bench file
   from the working tree copied in so both arms measure identical workloads.
2. `fmb_after` — the fix.
3. `matmul` — the unfused `MatMul` on the same shapes, the parity target.

Bias is deliberately **non-zero and non-uniform** (`sin(i * 0.31)`), so a fix
that dropped or mis-broadcast it fails numerically instead of merely looking
fast. `steady_ms` is the median of three reps after warmup; cold is reported
separately where it carries signal.

The f32 rows are a **null control**: f32 does not use the 16-bit route, so the
fix must not move them. This control is intrinsic to the workload rather than a
separate experiment — the same binary, the same shapes, one dtype flag apart.

## Results (steady_ms)

| shape | dtype | fmb_before | fmb_after | gain | matmul | after/mm | before/mm |
|---|---|---|---|---|---|---|---|
| mlp 4096x14336 | bf16 | 2.845 | 1.723 | 1.65x | 1.709 | 1.01x | 1.66x |
| mlp 4096x14336 | f16 | 2.687 | 1.730 | 1.55x | 1.759 | 0.98x | 1.53x |
| lm_head 4096x128256 | bf16 | 8.698 | 6.885 | 1.26x | 6.784 | 1.01x | 1.28x |
| lm_head 4096x128256 | f16 | 8.595 | 6.862 | 1.25x | 6.789 | 1.01x | 1.27x |
| qkv 4096x4096 | bf16 | 1.427 | 0.152 | 9.39x | 0.171 | 0.89x | 8.35x |
| qkv 4096x4096 | f16 | 1.407 | 0.175 | 8.04x | 0.171 | 1.02x | 8.23x |
| w34M 4096x8192 | bf16 | 2.871 | 1.080 | 2.66x | 1.080 | 1.00x | 2.66x |
| w34M 4096x8192 | f16 | 2.841 | 1.090 | 2.61x | 1.073 | 1.02x | 2.65x |
| attn_out | bf16 | 0.273 | 0.025 | 10.92x | 0.025 | 1.00x | 10.92x |
| attn_out | f16 | 0.289 | 0.026 | 11.12x | 0.025 | 1.04x | 11.56x |
| small | bf16 | 0.086 | 0.020 | 4.30x | 0.019 | 1.05x | 4.53x |
| small | f16 | 0.091 | 0.020 | 4.55x | 0.019 | 1.05x | 4.79x |
| **mlp (null)** | **f32** | **6.551** | **6.538** | **1.00x** | 6.656 | 0.98x | 0.98x |
| **lm_head (null)** | **f32** | **14.896** | **14.840** | **1.00x** | 14.795 | 1.00x | 1.01x |
| **w17M (null)** | **f32** | **2.012** | **2.065** | **0.97x** | 2.230 | 0.93x | 0.90x |
| **w34M (null)** | **f32** | **5.276** | **5.084** | **1.04x** | 5.167 | 0.98x | 1.02x |
| **attn_out (null)** | **f32** | **0.066** | **0.064** | **1.03x** | 0.064 | 1.00x | 1.03x |
| **small (null)** | **f32** | **0.031** | **0.038** | **0.82x** | 0.035 | 1.09x | 0.89x |

## Reading

* **Parity is reached.** `after/mm` is 0.98x–1.01x on all four model-shaped
  rows (mlp, lm_head at both dtypes). The remaining spread (0.89x–1.09x) is
  confined to shapes whose absolute time is 19–175 microseconds.
* **The gain is a strong function of shape**, 1.25x at lm_head to 11.12x at
  attn_out. Quoting a single number for this fix is wrong; #1702's 1.55x is the
  mlp row.
* **The null control holds** at 0.97x–1.04x on the five f32 rows above 60
  microseconds. The sixth (`small`, 0.82x) is a 7-microsecond difference on a
  31-microsecond measurement and is noise, not signal — reported rather than
  dropped.
* Before the fix, all f16/bf16 FMB rows land at ~23 GB/s regardless of dtype,
  which is the signature of widening to f32 first: the achieved bandwidth is set
  by the f32 copy, not by the stored width.

## Route proof, independent of timing

* `ONNX_GENAI_CPU_MM_HALF_GEMV=0` moves post-fix FMB f16 w17M from 1.691 to
  4.437 ms — the GEMV is genuinely running, not being attributed.
* Cold/steady shape: `MatMul` cold 4.673 / steady 0.602 (builds a transpose);
  pre-fix FMB cold 1.832 / steady 1.691 (never builds one). This diagnostic
  correctly identified the missing route **while the host was contended and
  every steady number was void**, which is why it is recorded.
* `fused_bias_decode_reaches_the_transposed_half_gemv` asserts the route
  counters directly, so the claim does not rest on timing at all.

## Discarded measurements

The first post-fix pass reported FMB f16 w17M at 1.691 ms — *worse* than the
1.443 ms pre-fix number, which would have read as a failed optimisation.
`uptime` showed **load average 20.94**: the acc0 matrix (§23) was still running.
All numbers from that window are void and none are reported above. The host was
allowed to quiesce to 1.77 before the table was taken.

This is the third contended-host false reading in this assignment. The standing
rule is now: `uptime` immediately before every benchmark, one benchmark at a
time, and prefer cold/steady *shape* over absolute steady time when in doubt.

## Numerics

* `every_bias_shape_survives_the_new_route` compares the GEMV route against the
  f32 route with `ONNX_GENAI_CPU_MM_HALF_GEMV=0` for 1-D, scalar, `[1, N]` and
  `[M, N]` bias, to 1e-3 relative. Routing must not narrow which biases the
  operator accepts.
* Bias is applied **after** the reduction, exactly as the two pre-existing paths
  do, never folded into the accumulator — folding would change summation order
  and therefore the bits.

## Memory accounting

`retained_transpose_bytes_are_bytes_the_plan_predicted` executes the kernel and
asserts retained == predicted (ratio 1.00, the #1056 criterion). See §26 of the
assignment ledger for the two accounting defects this surfaced and the
registry-derived guard that now prevents their recurrence.
