# `QLinearMatMul` decode: the integer GEMM branched on operand signedness in its innermost loop

Date: 2026-08-21. Host: AMD EPYC 9V74, 32 vCPU (16c x 2 SMT), AVX2 + FMA + F16C, **no AVX-512 /
VNNI / AMX**, L1d 32 KiB per core, L2 1 MiB per core (16 MiB total), L3 64 MiB, 75.8 GB/s DRAM.
Shared box; every number below is a median of repeated paired runs with the two arms alternated.
rustc 1.98.0. All builds are the **default (no `mlas`) native build**.

## 1. What was measured, and why it had never been measured before

The ledger's `QLinearMatMul` rows were taken on a `--features mlas` research build, with a caveat
saying the shipped build was 11.8x–12.0x behind ORT. Since then #1194 landed a native integer GEMM
(`qgemm_native.rs`) and nothing re-measured the shipped build, because **there was no
`QLinearMatMul` generator in `scripts/ort_ab/`** — the op could not be driven through the standard
A/B harness at all. `scripts/ort_ab/gen_qlinear.py` (new, in this change) fixes that: u8 x u8 and
i8 x i8, decode and prefill, production projection geometry, plus a `fork_gate` set that puts rows
immediately below, at and above both of the kernel's own dispatch gates (`PARALLEL_MIN_WORK` and
the `m <= MR` fused/packed split).

First run of the new generator against a real ORT CPU session, one thread, arms interleaved,
parity `PASS` on every cell:

| cell | native | ORT | native/ort |
| --- | ---: | ---: | ---: |
| `qwen3_8b_square` u8 x u8, M=1, 3584x3584 | 1.073 ms | 0.361 ms | **2.97** |
| `llama3_8b_qkv` u8 x u8, M=1, 4096x6144 | 2.651 ms | 0.827 ms | **3.21** |
| `qwen3_0p6b_qkv` u8 x u8, M=1, 1024x3072 | 0.200 ms | 0.084 ms | **2.39** |
| `qwen3_8b_square` u8 x u8, M=128 | 28.638 ms | 25.324 ms | 1.13 |
| `qwen3_8b_square` **i8 x i8**, M=1 | 1.336 ms | 12.619 ms | **0.11 (we are 9.4x faster)** |

So the ledger's 11.8x caveat is **stale in both directions**: the native kernel is 1.13x at M=128
and 9.4x *ahead* at i8 M=1, and the real remaining loss is specifically **u8 x u8 at M=1**, the
decode shape. ORT's own i8 x i8 path is 35x slower than its u8 x u8 path at the same shape, which
is why that row is a win rather than anything we did.

## 2. Localising it: not the wrapper, not the working set

**Not the wrapper.** Timed alone (`--native-only`) the 3584x3584 u8 M=1 cell is 0.721–0.743 ms
against 0.635 ms for `qgemm()` called directly with the same operands — about **12%** outside the
kernel, not the 44% an interleaved-arm comparison suggested. (The interleaved 1.073 ms is the
paired number: each arm evicts the other's 12.85 MB weight from L3. Both arms pay it, so the
*ratio* is sound — 2.40x and 3.94x solo versus 2.97x and 3.21x paired — but the *absolute* is not
a wrapper cost, and it would have been a wasted PR to go hunting one.)

**Not the working set.** Kernel-only, one thread, footprint held at 12.85 MB and the aspect ratio
swept:

| shape (m x k x n) | footprint | GB/s of B |
| --- | ---: | ---: |
| 1 x 3584 x 3584 | 12.85 MB | 20.0 |
| 1 x 896 x 14336 | 12.85 MB | 19.4 |
| 1 x 1792 x 7168 | 12.85 MB | 20.4 |
| 1 x 7168 x 1792 | 12.85 MB | 19.0 |
| 1 x 14336 x 896 | 12.85 MB | 17.4 |
| 1 x 512 x 2048 | **1.0 MB (L2)** | 19.5 |
| 1 x 4096 x 14336 | 58.7 MB (> L3) | 7.2 |

A 1 MB weight that fits L2 runs at the same 19.5 GB/s as a 12.85 MB one, and every aspect ratio at
equal footprint lands within 17–20 GB/s. **At m=1 and L2/L3 residency the kernel is
instruction-bound, not memory-bound.** Only the 58.7 MB cell (bigger than L3) is bandwidth-bound,
and that one is a different problem.

## 3. The defect

`widen16(signed: bool, src)` — the u8 -> i16 widen that every byte of `B` passes through — took
signedness as a **runtime argument** and branched on it inside the innermost loop, once per 16
bytes of `B`:

```rust
let raw = _mm_loadu_si128(src.cast());
if signed { _mm256_cvtepi8_epi16(raw) } else { _mm256_cvtepu8_epi16(raw) }
```

`Operand::signed` is fixed for the whole call — it comes from the input's dtype — so this is a
loop-invariant branch that the compiler cannot hoist through the `#[target_feature]` boundary. At
m=1 the fused kernel spends about 8 vector uops per 32 bytes of `B` (2 widen, 2 interleave, 2
`vpmaddwd`, 2 `vpaddd`), so an extra predicted branch and the register pressure of keeping both
arms live is a measurable fraction of the whole loop.

Confirmed by hacking the condition to a constant `false` and re-measuring the kernel: 20.0 ->
23.7 GB/s at 3584x3584, and the same ~1.15x–1.18x at every L2/L3-resident cell.

## 4. The change

`widen16`, `fused_strip`, `accumulate_fused` and `pack_panel` take signedness as a
`const SIGNED: bool`. The one runtime `match` moved out to the block dispatcher, which already
matched on `m`. No arithmetic changed: the accumulation is still wrapping `i32`, so the kernel is
bit-identical to before, which the existing `check(m, k, n, a_signed, b_signed)` oracle asserts
across both sign domains.

### Kernel A/B (`bench_qgemm_ab`, two prebuilt test binaries alternated, `portable` control)

`portable` is the harness's drift control — the same arithmetic with none of the blocking, so it
must not move when a kernel detail changes. Across all six runs below it stayed within
3.80–3.86 GMACS (1.6%), so the box was quiet enough for the native arm to mean something. Three
paired repetitions, 41 iterations per sample, one thread, arms alternated `base, new, base, new, …`:

| shape | arm | rep 1 | rep 2 | rep 3 | median | ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1x3584x3584 | base | 19.97 | 19.92 | 20.07 | 19.97 GMACS | |
| 1x3584x3584 | **new** | 22.52 | 22.91 | 21.41 | **22.52** | **1.13x** |
| 1x1024x3072 | base | 18.03 | 17.24 | 17.52 | 17.52 | |
| 1x1024x3072 | **new** | 19.96 | 19.73 | 19.89 | **19.89** | **1.14x** |
| 128x3584x3584 | base | 62.78 | 61.16 | 61.92 | 61.92 | |
| 128x3584x3584 | **new** | 64.15 | 64.00 | 63.94 | **64.00** | **1.03x** |
| *`portable` control* | base | 3.82 | 3.86 | 3.82 | 3.82 | |
| *`portable` control* | **new** | 3.82 | 3.82 | 3.83 | **3.82** | *1.00x* |

The two `m = 1` ranges do not overlap between arms — every repetition of `new` beats every
repetition of `base` — and the packed `m = 128` path moves 1.03x, consistent with `pack_panel`
being a small share of a path that then re-reads its panel many times. An independent check agrees:
hacking the runtime `if signed` to a constant `false` (correct only for the unsigned probe, but
enough to price the branch) reproduced 1.15x–1.18x on the same shapes.

### End-to-end: **not resolvable on this host, and an earlier claim retracted**

A first attempt built two `bench_generic` binaries from the two sources and alternated them
(`--native-only`, medians of 3–5 repetitions of 10–20 runs). It read **1.41x–1.46x at M = 1**, and
that number is **withdrawn**. A 1.13x kernel cannot produce a 1.43x call: if the kernel is fraction
`f` of the call, the end-to-end ratio is `1 / (f/R_k + (1 - f))`, which is bounded above by `R_k`
and reaches it only at `f = 1`. Solving for `R_e = 1.43` at `R_k = 1.13` needs `f = 2.6`. The claim
was arithmetically impossible against my own controlled measurement, and I published it before
checking that.

What it actually is:

- **Null control.** The *same* baseline binary run against itself, 5 paired repetitions of 20 runs:
  ratios 0.91, 0.99, 1.04, 1.01, 1.02. The paired end-to-end protocol has a **±10% noise floor** on
  this box, so 1.43x is not run-to-run noise — but it is also not attributable to a 1.13x kernel.
  The remaining suspect is build-to-build code layout in the *unchanged* wrapper, which the kernel
  A/B is immune to because its `portable` arm proves the two builds were otherwise equivalent.
- **Re-run against ORT, both binaries measuring their own ORT reference back to back**, 3
  repetitions, 15 runs each: base `native/ort` 2.43, 2.26, 2.50 (median **2.43**); new 1.71, 2.73,
  2.58 (median **2.58**). **The win does not reproduce end to end at all.**

So the honest statement is: **the change is a 1.13x kernel improvement at `m = 1` with a drift
control, and its effect on the whole call is below what this host can resolve.** The earlier
"2.97x -> 1.61x against ORT" pairing is withdrawn with it; a base binary re-measured today is 2.43x
against ORT at that cell, and the new binary is not measurably different from it.

Kept because the mechanism is not in doubt and the risk is nil: the branch is provably
loop-invariant (`Operand::signed` comes from the input dtype and cannot change inside a call), the
output is bit-identical, and three controlled repetitions separate the arms cleanly at the level
where the change acts.

## 5. What is still open (measured, not merged)

1. **The u8 M=1 loss against ORT is essentially untouched by this change** — 2.4x–3.2x before,
   2.43x re-measured after. Everything below is still open.
2. **Parallel scaling at m=1 is the biggest remaining loss.** 3584x3584 u8 M=1 goes 0.541 ms at one
   thread to 0.411 ms at eight — **1.3x from 8 threads**, while ORT gets 5.8x (0.336 -> 0.058 ms).
   Eight threads yields ~31 GB/s where one thread already reaches 23.5 GB/s. The fused path splits
   *columns* (`block_width = n / threads`), so each worker walks a 448-byte slice of every row: eight
   strided streams whose stride exceeds a 4 KiB page, which is exactly the access the L2 stride
   prefetcher cannot follow. At m=1 there is no reuse of `B` to protect, so splitting `k` instead —
   private accumulators over contiguous row ranges, reduced at the end over an `n * 4` byte
   output — is the shape that streams. Not attempted here; it is a separate mechanism with its own
   numerics burden (the reduction reorders `i32` adds, which is still exact under wrapping, but the
   oracle has to say so).
3. **The ~2.4x at one thread is an instruction budget.** Exact full-range 8-bit needs
   `vpmaddwd`, which costs ~0.25 vector uops per byte of `B`. `vpmaddubsw` would halve that, but it
   saturates its `i16` pair sum: it is exact only when one operand stays within +/-64
   (`255 * 64 * 2 = 32640` fits `i16`, `255 * 65 * 2 = 33150` does not; this is why
   ORT's quantizer has `reduce_range` for non-VNNI AVX2). A 32-column formulation that byte-
   interleaves two `k` rows with 256-bit unpacks before widening gets to ~0.22 uops/byte on paper —
   about 1.14x, and worth trying — but matching ORT at m=1 almost certainly needs a prepacked `B`
   plus a saturation-safe `maddubs` scheme, which is a correctness contract, not a tuning change.
4. **The 58.7 MB DRAM-bound cell (7.2 GB/s of 75.8)** is untouched and is not an instruction
   problem.

## Reproducing

```bash
python3 scripts/ort_ab/gen_qlinear.py --out /path/to/models
cargo build --release -p onnx-genai-bench --features bench-native,cuda-13000 --bin bench_generic
./target/release/bench_generic --model /path/to/models/qlinear_qwen3_8b_square_u8_k3584_n3584_t1.onnx \
  --runs 10 --warmups 3 --native-threads 1 --ort-intra-threads 1

QGEMM_AB_THREADS=1,8 QGEMM_AB_SHAPES=1x3584x3584,1x1024x3072,4x3584x3584,128x3584x3584 \
  cargo test --release -p onnx-runtime-ep-cpu --lib bench_qgemm_ab -- --ignored --nocapture
```
