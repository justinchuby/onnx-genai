# Native ARM64 integer GEMM: borrowed-byte register reuse

This change adds a pack-free NEON route to the native `QLinearMatMul` integer
GEMM. It is a partial MLAS absorption step. It does **not** graduate or remove
the MLAS reference, measure an end-to-end model, or compare against MLAS.

## Mechanism

The vendored MLAS ARM64 `QgemmU8X8KernelNeon.S` reuses loaded operands across
four output rows. Our x86 native GEMM already uses row reuse; ARM64 previously
used the compiler-vectorized portable byte loop. The new implementation keeps
up to four rows of eight i32 output accumulators in registers and shares each
borrowed B load across those rows. It does not import MLAS source or its packed
layout, add a weight copy, or create a new workspace/cache/thread pool.

The tile height is bounded by rows per worker, keeping at least one task per
worker. Single-row, narrow-column and tiles with fewer than two contributing
rows retain the portable route. Signed and unsigned bytes widen before i32
multiplication; accumulation wraps exactly as before. Row sums, B zero-point
correction and requantization are unchanged.

## Measurement and limits

Apple M1 Max, 10 CPU cores, 32 GiB unified memory; macOS 27.0.1; rustc
1.98.0 (88d9e12ae). The actual CPU crate's release test binary was used with its
existing `codegen-units = 1` profile, based on `7eb6386a9` plus this patch.
The standalone exploratory probe is excluded from these results.

Each cell has 21 samples per arm, with rotating `native`, `portable` and
`native_control` order. Every arm runs in the **same explicit Rayon pool** and
performs exactly `sum_k (a - za) * b`. Initial buffers and untimed warm-up output
comparisons are identical. The table reports process CPU time; elapsed latency
and all reported cells are in [the CSV](2026-11-21-qgemm-neon-results.csv).

| shape (M × K × N), u8 × u8 | workers | native CPU ms | portable CPU ms | portable/native |
|---|---:|---:|---:|---:|
| 128 × 5120 × 5120 | 1 | 294.414 | 320.846 | 1.09× |
| 128 × 5120 × 5120 | 4 | 311.809 | 423.975 | 1.36× |
| 128 × 1024 × 14336 | 1 | 154.890 | 182.309 | 1.18× |
| 128 × 1024 × 14336 | 4 | 168.060 | 236.617 | 1.41× |

For 16/128 × 2048 × 2048, all four signedness combinations measured about
1.25–1.27× at one worker and 1.54–1.58× in process CPU time at four workers.
Large-cell A/A elapsed medians differed by less than 1%. Small cells include
noisy A/A results (up to 31%); those rows do not establish a speedup. The
single-row 2048 × 2048 fallback measured essentially equal in both arms. No
general ARM64, x86, throughput, MLAS parity or RSS improvement is claimed.

The Linux `scripts/hostlock.sh` explicitly rejects Darwin. Each local sweep
instead held an exclusive kernel-managed `fcntl.flock` on the persistent inode
`/tmp/onnx-genai-hostlock-darwin.flock`, never unlinked or reaped. The wrapper
recorded load averages and whole-child CPU/RSS before releasing it. This
coordinates only participants using that Darwin lock; Linux foreign-CPU and
SMT-sibling monitoring is unavailable here. A/A controls qualify these local
measurements; the lock does not prove absence of other applications. Whole-test
peak RSS is not per-arm memory evidence.

The old diagnostic had two confounders: portable could use the ambient global
pool while native used the selected worker count, and only native included
row sums and zero-point correction. The revised harness fixes both. The first
exploratory run using that old harness is invalid evidence and is not included.

## Reproduction

First prove the selector exists:

```sh
cargo test -p onnx-runtime-ep-cpu --release --lib bench_qgemm_ab -- --list
```

Hold the host measurement lock for the complete sweep. On Linux use
`scripts/hostlock.sh run --owner <owner> --reason <reason> -- <command>`.
On this Darwin host the wrapper held `fcntl.flock` around each command below;
the environment bootstrap only supplied the existing target directory and ORT
prebuilt location and did not select a different arithmetic backend.

```sh
QGEMM_AB_ITERS=21 QGEMM_AB_THREADS=1,4 \
QGEMM_AB_TYPES=u8xu8,u8xi8,i8xu8,i8xi8 \
QGEMM_AB_SHAPES=1x513x257,2x513x257,3x513x257,4x513x257,9x513x257,16x2048x2048,128x2048x2048 \
cargo test -p onnx-runtime-ep-cpu --release --lib bench_qgemm_ab -- --ignored --nocapture

QGEMM_AB_ITERS=21 QGEMM_AB_THREADS=1,4 QGEMM_AB_TYPES=u8xu8 \
QGEMM_AB_SHAPES=1x2048x2048,4x5120x5120,128x5120x5120,128x1024x14336 \
cargo test -p onnx-runtime-ep-cpu --release --lib bench_qgemm_ab -- --ignored --nocapture
```

Correctness checks cover all byte signedness combinations, ragged row/column
tiles, arbitrary i32 zero points, seeded accumulators, wrapping overflow,
zero-activation rows, and multiple worker counts. This closes a native ARM64
capability gap while retaining the optional MLAS comparison route for further
work on packing, requantization and low-bit GEMM.
