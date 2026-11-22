# Native SiLU absorption: measured gap and rejected rational

Measured on Apple M1 Max (10 physical/logical cores, 32 GiB), macOS 27.0.1
arm64, rustc 1.98.0, release defaults, `mlas` enabled, debug info and incremental
compilation disabled. No CUDA, non-Apple ISA or whole-model claim.

This is a historical baseline. The later [repair-set implementation and
measurements](2026-11-21-silu-density-repair.md) supersede its ordinary-vector
performance conclusion. The numerical counterexample and raw baseline data
remain valid and reproducible.

## Decision

Keep the existing native exp-based implementation and the MLAS research route.
The native ordinary long-vector path is still slower here. Directly absorbing
the MLAS logistic rational also violates the existing native numerical guard,
so that experimental replacement was reverted. No graduation or runtime
shape-based backend selection is introduced.

The [metadata and per-cell medians](2026-11-21-silu-native.results.json) and
[raw samples](2026-11-21-silu-native.samples.csv.gz) support these findings.
The exact measured baseline source is commit `edd930a63` (activation file
SHA-256 `5a874ab3ba15c3c3cce17886d1dbfaf6a0bb0e55e3c78923ae701e3a7e4097ff`).
The candidate is commit `1ee82c3e2`, reverted by `44f3fbbdc`. Final production
activation code and SIMD implementation match validated `d4d0b01a9`; later
benchmark type naming, analysis, documentation and licensing do not change them.

## Method

One binary interleaves MLAS, native, and repeated-native A/A calls, rotating
order across 21 samples per cell. Each sample batches `ceil(2^20/n)` calls,
clamped to 1..16384; raw data retains batch lengths. Wall time spans synchronous
calls and black-box barriers, including scheduling. Process CPU time includes
all threads and clock bookkeeping. Allocation, validation, warmup and printing
are outside the timing loop. Inputs/output allocations are reused.

Each of three sweeps starts fresh 1/4/8-thread processes, rotating width order.
Both implementations use the same actual runtime pool width; serial mode
forces the same scheduler guard for both. Rayon is set to one thread and is
not the measured pool. Eligible runtime cells assert actual dispatch for
both implementations. Short cells remain serial by the existing policy.

The harness validates numerical results before timing: stable staged f32
semantics bitwise for out-of-band/non-finite/zero inputs, independent f64 error
for ordinary inputs. This is the selected three input populations, not proof
that MLAS meets the native error budget everywhere in `[-18,18]`.

There are 17,010 raw samples and 270 cells (nine processes, 30 cells each),
with 21 samples for each of three arms in every cell. All nine exact-selector
invocations ran and passed the named diagnostic. The Darwin `fcntl.flock`
lock `/tmp/onnx-genai-hostlock-darwin.flock` was held for the whole sweep;
repository `hostlock.sh` does not support Darwin. Load and per-child wait4 CPU,
wall and RSS receipts are retained. No affinity or exclusive-machine proof is
claimed. No invocation was discarded: CPU/wall utilization was consistent
within each width (about 0.998–0.999, 1.137–1.150 and 1.159–1.165 at 1/4/8).
A/A wall ratios across all cells span approximately 0.982–1.007; the ordinary
long-vector deficit is much larger. Favorable near-1.0 cells are inconclusive.

Ratios below are MLAS/native wall-time medians: greater than 1 favors native.
Ranges include all widths, serial/runtime modes and three sweeps; each
per-cell median and CPU ratio remains available in the metadata.

| Input population | n | MLAS/native wall ratio range |
|---|---:|---:|
| ordinary `[-12.05,12.05]` | 3 | 1.061–1.072 |
| ordinary | 17 | 0.937–0.940 |
| ordinary | 4,096 | 0.673–0.678 |
| ordinary | 65,537 | 0.678–0.682 |
| ordinary | 1,048,581 | 0.692–0.708 |
| sparse repairs (one in 31) | 4,096 | 0.973–1.022 |
| sparse repairs | 1,048,581 | 1.016–1.066 |
| dense finite negative tails | 3 | 1.614–1.623 |
| dense finite negative tails | 1,048,581 | 0.908–0.921 |

The ordinary 4,096-element native call costs about 3.99 us versus 2.70 us for
MLAS in the first one-thread runtime sweep. Both paths retain identical
thread budgets. Results do not establish an inference speedup, RSS savings,
a universal kernel ceiling or an end-to-end ceiling without a model profile.

## Why the rational candidate was rejected

The candidate reused the already absorbed AVX2 sigmoid coefficients in an
ARM64 NEON rational, fused the SiLU multiply in registers, and shared the
bounded repair reduction. A 131,072-element vector-lane differential test
matched MLAS bitwise (NaNs checked by classification). This proves the port
followed its reference; it does not prove the native accuracy requirement.

The existing independent 36,001-point native f64 guard failed:

- Input f32: `-16.434999465942383`.
- Candidate/MLAS output: about `-3.4286081e-6`.
- Independent f64 SiLU: `-1.1971266100662218e-6`.
- Absolute error: about `2.23148e-6`; allowed error here is `2e-6`.

The failure occurred before candidate timing. Three other contract tests
passed; the independent guard failed. No failing guard was relaxed, removed
or ignored. The candidate was reverted. Future absorption needs a stable
negative-shoulder formulation and measured cost, alongside the existing
negative-tail, IEEE, lane/tail and parallel invariance guards.

The already absorbed SIMD coefficients retain Microsoft's complete MIT notice
in the CPU crate's `MLAS_LICENSE`, independently of the removable vendor crate.

## Reproduce

Recompute and validate every reported median from the raw archive:

```sh
python3 scripts/analyze_silu_native.py docs/benchmarks/2026-11-21-silu-native.results.json
```

Build the measured source commit before measuring. On a host holding the
shared benchmark lock for the entire width/sweep loop, repeat each width
three times in rotating order:

```sh
ONNX_GENAI_CPU_TASK_THREADS=4 RAYON_NUM_THREADS=1 \
  cargo test --release --locked -p onnx-runtime-ep-cpu --features mlas --lib \
  kernels::activations::silu_bench::silu_native_vs_mlas_rotating \
  -- --exact --ignored --nocapture --test-threads=1
```

The Rust diagnostic's timers exclude Cargo/build time. The retained receipts
used the already-built test executable directly, so per-child process usage
also excludes build time. Restore the candidate commit only in an isolated
checkout to reproduce its expected independent-guard failure; it is not the
final production implementation.
