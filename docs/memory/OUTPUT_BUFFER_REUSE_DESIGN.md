# CPU MatMul output-buffer reuse design

## Decision

Implement **Option A: direct f32 output** first.  When the supplied MatMul output
is a validated, row-major-contiguous `Float32` CPU tensor, pass its existing
storage as the `&mut [f32]` C operand to the existing GEMM dispatch.  This removes
the per-execution intermediate result allocation and the dtype writer's f32
conversion/copy.  Keep the current intermediate path for `Float16`, `BFloat16`,
`Float64`, and non-contiguous outputs.  Do not add a reusable output scratch
buffer in this slice.

This is backend-agnostic: `gemm` / `gemm_with_backend` already accept a mutable
C slice, and MLAS, SimdX86, and Generic all overwrite that slice.  It is also the
only approach that removes memory traffic rather than merely amortizing its
allocator cost.

## Investigation: current f32 path

For the common contiguous, non-batched `A[m,k] @ B[k,n] -> C[m,n]` Float32 case,
`MatMulKernel::execute` calls `matmul_dense_prepacked`, which calls
`matmul_dense_impl`.  The latter creates
`vec![0.0f32; batch_count * m * n]`, then calls `gemm(&a_dense, &b_dense, &mut
out, m, k, n)`.  `gemm_with_backend` forwards that same C slice to MLAS,
SimdX86, or Generic.  `execute` then calls `write_dense_f32_narrow`; even for
Float32 this builds `narrowed: Vec<f32>` with `data.iter().map(...).collect()`,
then `write_strided` copies that vector into the actual output tensor element by
element.

Thus, for contiguous f32 inputs and output, every `execute` currently performs:

| Work | `32x512x512` (`C=16,384`) | `32x1024x1024` (`C=32,768`) | Avoidable by direct f32 output? |
|---|---:|---:|---|
| `out = vec![0.0; C]`: heap allocation plus zero-fill | 64 KiB | 128 KiB | yes |
| `narrowed: Vec<f32>`: heap allocation plus full copy of `out` | 64 KiB | 128 KiB | yes |
| `write_strided`: full store to the destination (contiguous here) | 64 KiB | 128 KiB | yes (GEMM stores C there) |
| writer index vector | rank-sized, two `usize` here | rank-sized | yes on direct path |

The initial zero-fill is redundant for MLAS and SimdX86, whose C operation is
overwrite, and is additionally repeated by Generic's `gemm_block` before its
accumulation.  The output tensor itself is allocated by the executor outside this
kernel timing loop and is reusable across executions.

There are also small metadata allocations in `matmul_dense_impl`: promoted
`a_shape` and `b_shape`, `batch_shape`, and the contiguous batch-stride vectors.
Batched calls allocate `bidx` (one `usize` per output batch axis).  These are tiny
relative to C, but remain after Option A.  A batched direct path can write each
`b_out * c_mat..(b_out + 1) * c_mat` slice exactly as the current path does.

Input materialization is independent of the output problem:

* A contiguous Float32 input returns `Cow::Borrowed` from `to_dense_f32_widen`:
  no activation-data allocation.  A contiguous Float32 constant likewise has no
  `MatMulPrepack` dense cache allocation.
* A strided Float32 or any f16/bf16/f64 input calls `to_dense_float`, allocating a
  dense `Vec<T>`, then maps it into a second owned `Vec<f32>`.  Each buffer spans
  the input element count (2/2/8 bytes then 4 bytes per element for f16/bf16/f64;
  two 4-byte buffers for strided f32).  Non-constant inputs pay this each call.
  A constant materialized input pays it once, retains the final f32 buffer in its
  `OnceLock`, and borrows that cache thereafter.
* For f16/bf16/f64 output, `write_dense_f32_narrow` necessarily allocates a
  `Vec<T>` of the full C size (2/2/8 bytes per element), converts every f32 value,
  then `write_strided` stores it.  This remains the correct fallback.

## Feasibility and required API shape

`TensorMut` exposes `dtype`, `is_contiguous()`, `numel()`, validation, and
`data_ptr_mut::<T>()`.  Therefore, after validating a Float32 contiguous output,
the CPU kernel can safely construct a mutable slice of `output.numel()` f32
values from its data pointer.  The output view carries its byte offset, so
`data_ptr_mut` already selects the correct origin.  A squeezed 1-D MatMul result
has the same *element count* as the promoted matrix result, so no reshape/copy is
needed.

No change is needed to `gemm`: it already has the desired C-slice API.  Refactor
`matmul_dense_impl` into an internal `matmul_dense_into(..., out: &mut [f32]) ->
Result<()>` that computes dimensions first, verifies `out.len() == batch_count *
c_mat`, handles zero-sized results, and sends one C sub-slice per batch to
`gemm`.  Preserve a allocating `matmul_dense_impl` wrapper for current callers
such as fused attention, which need an owned result.  Add a corresponding
prepacked `..._into` entry point for `MatMulKernel::execute`.

At execute time, select the direct path only when `outputs[0].device` is CPU,
`outputs[0].dtype == Float32`, and `outputs[0].is_contiguous()`.  First validate
the output and check its logical length against the computed result *before* any
GEMM write.  Otherwise retain the present `Vec<f32>` result plus
`write_dense_f32_narrow` path.  This preserves strided-output and non-f32
semantics without widening the unsafe surface beyond the existing validated
typed-pointer convention.

## Alternative: reusable scratch buffer

A per-kernel scratch `Vec<f32>` would avoid repeated capacity allocation but not
zeroing/initialization requirements, the f32 narrowing vector, or the final copy.
It also does not solve the output memory bandwidth.  `RefCell<Vec<f32>>` is not a
safe default: `execute` takes `&self`, and a kernel can be scheduled concurrently;
`RefCell` would make sharing unsafe/invalid.  A `Mutex<Vec<f32>>` serializes
concurrent executions or needs pool/lease complexity.  Thread-local scratch avoids
contention but makes capacity per-worker, can retain large buffers indefinitely,
and is not naturally tied to a kernel instance.  A pool is more infrastructure
than this optimization requires.

Option B is a fallback only if the executor cannot guarantee a direct mutable
output slice.  In that case, use a leased pool rather than `RefCell`, return the
buffer after narrowing, and explicitly test concurrent execution.  It should not
be used in preference to Option A.

## Interactions and correctness constraints

* **Backends:** keep all MLAS/SimdX86/Generic selection exclusively in `gemm`.
  The new code changes only C ownership.  Generic still zeroes its assigned C
  blocks; MLAS and SimdX86 overwrite C as today.
* **Packed B:** the in-flight `mlas_sys::PackedB` constant-B prepack can replace
  the B operand later without changing this output API.  The direct C slice must
  remain separate from that immutable packed weight.  Preserve the existing
  `MatMulPrepack::dense` behavior until the PackedB cache lands.
* **Batches/broadcast/1-D:** use the existing offsets and batch loop verbatim,
  changing only `out[b_out * c_mat..]` from an owned vector to the supplied C
  slice.  Verify all existing matrix, batched, broadcast, vector×matrix, and
  matrix×vector tests on both paths.
* **Strides:** direct write requires row-major-contiguous output only.  A strided
  Float32 output must use the fallback so `write_strided` remains the sole place
  that applies arbitrary (including negative) output strides.
* **Aliasing:** direct GEMM must not write an output that overlaps either input
  while that input is being read.  The executor should enforce the normal
  non-aliasing contract for MatMul outputs; document/assert that contract at the
  kernel boundary if it is not already guaranteed.  If aliasing is permitted by
  the EP API, direct write must be disabled when ranges may overlap (or inputs
  must first be materialized).  The present late write accidentally tolerates
  some aliasing and must not be silently weakened.
* **Zero dimensions:** retain the pre-GEMM empty-result return.  It writes no
  elements, matching the current result-vector and narrowing behavior.

## Reviewable implementation plan

1. In `crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs`, factor geometry and the
   existing batch loop into an `into` helper accepting `&mut [f32]`; retain the
   owned-vector wrapper for `matmul_dense` and fused-attention callers.
2. Add a prepacked `into` helper and make `MatMulKernel::execute` choose it for a
   validated, contiguous Float32 `TensorMut`; otherwise call the retained owned
   path and `write_dense_f32_narrow`.
3. Add focused unit tests for direct f32: 2-D, batched/broadcast, each 1-D
   promotion, zero-sized result, and a strided Float32 output fallback.  Run the
   existing f16/bf16/f64 cases to prove narrowing remains unchanged.  Add a test
   that a mismatched output length errors before write if the test tensor harness
   can observe it.
4. Run `cargo test -p onnx-runtime-ep-cpu`, the numeric-regression integration
   test, and the MLAS-gated MatMul tests.  No registry changes are needed; retain
   the `reg.len() == PHASE1_OPS.len() + 85` invariant.
5. Re-run the Criterion f32 MatMul MLAS, SimdX86, and Generic rows in dedicated
   1- and 8-worker Rayon pools.  Also re-run the `mlas-sys` isolated probe to
   distinguish remaining GEMM cost from kernel overhead.

## Measurements and expected impact

The benchmark harness times full `Kernel::execute` and creates dedicated Rayon
pools of 1 and 8 workers.  This investigation ran:

```text
cargo bench -p onnx-runtime-ep-cpu --features mlas --bench kernels -- \
  'matmul/(medium|large)/mlas/f32'
cargo test -p mlas-sys --release -- --ignored --nocapture perf_sgemm_multithread
```

On this shared host, full execute medians/ranges were 185.84 us (184.27--188.24)
and 125.18 us (124.75--125.65) for `32x512x512` at 1/8 workers, and 875.11 us
(868.98--882.40) and 342.92 us (340.96--344.66) for `32x1024x1024`.  The isolated
MLAS probe covers only `32x512x512`; it reported 123.0 us at one worker and 34.1
us at eight, reusing C.  It does not provide a 1024 isolated row.  The existing
recorded controlled Criterion run reports 178/147 us full execute for 512 and
808/306 us for 1024, while its isolated 512 MLAS result is about 123/32 us.
Host contention/frequency makes the new full-execute rows unsuitable as a
replacement baseline, but both runs show the large 8-worker non-GEMM gap.

The prior controlled investigation isolated fresh `Vec` allocation/zeroing at
roughly 50--90 us for 512 at eight workers.  Direct output also removes the f32
narrowing allocation/copy and destination walk.  A realistic first target is to
remove about 50--100 us from the 147 us 512/8-worker end-to-end row (subject to
allocator and CPU-frequency variance), bringing it materially closer to the
32--34 us isolated GEMM, not claiming exact parity until remeasured.
