# Real Apple NBits execute: legacy KN versus reuse-gated native policy

This diagnostic compares actual factory-created `MatMulNBitsKernel::execute`
calls. A scoped, thread-local, test-only selector makes the NT capability
unavailable, taking the existing KN fallback. It does not duplicate the old
kernel or restore unchecked BLAS entry. Both routes use the current validated
NN/NT boundary. No production selector or environment knob is added.

The [initial source](APPLE_NT_EXECUTE_AB_INITIAL.md) also tried transient NT.
Its three qualified wall-clock sweeps at M=128,K=N=1024 were slower than KN
(legacy/native ratios 0.955–0.982). Those results remain archived separately,
including both failed CPU A/A cells; no sample was replaced. The final source
selects Apple NT only when the existing retention rule admits NK reuse. Zero
retention now takes the same KN fallback arithmetic in both arms, with normal
eligibility overhead still timed. Initial and final corpora are not pooled.

The native NT extension and its non-dyadic f64 numerical contract are covered
separately in [Apple NT GEMM](APPLE_NT_GEMM.md). This timing corpus does not
replace that numerical validation, benchmark MLAS, graduate a family, or measure
model tokens or full-session throughput.

## Controlled population and clocks

Five synthetic grouped-int4 geometries are measured: (M,K,N) = (2,33,3),
(8,257,65), (32,1024,1024), (128,1024,1024), (16,2048,2048). For each, the
legacy and native kernels have identical immutable inputs, constant-metadata
flags, retention ceilings and Host capacities. Retention is either disabled or
set to one complete f32 payload; each independent governor allows twice that
payload for admission. The arms have independent caches/books; repeated-native
A/A arms use the same native kernel, with separate caller-owned output buffers.

Inputs, outputs, kernel construction, f64 oracle, warm validation, post-validation,
checksums and printing are outside timing. Each timed block contains repeated
synchronous `execute` calls, TensorView/TensorMut construction, black-box and loop
overhead. Wall time uses `Instant`; CPU uses process `getrusage(RUSAGE_SELF)` and
includes all worker activity, framework/Rayon background spin and clock-reading
overhead during that interval. It is not an energy or CPU-savings measurement.

Each block has an untimed warm call. Retained native NK is warm during timing;
legacy KN always prepares a transient matrix. Zero retention forces both to
prepare KN each call, serving as a fallback-policy null control. Twenty-four rounds use six cyclic orders, balancing arm
positions and directed predecessor pairs across cycles. Every sample is kept.
A cell qualifies separately per clock only when the native-A/native-B median
ratio in either direction is at most 1.05. Failed A/A records remain in raw data
and summaries; they are excluded only from qualified summaries.

The recorder holds the stable Darwin shared flock throughout all three process
sweeps. Source, binary, Cargo hashes, environment, runtime Rayon size and load
readings are recorded. Framework threading is its default and is not inferred
from Rayon size. No affinity, foreign-core isolation, fixed frequency or other
hardware-tier equivalence is claimed. Weights span 396 bytes to 16 MiB of f32
payload; this is not a full LLM residency or out-of-core workload.

## Output and ownership witnesses

The timing fixture uses A values with a 1/32 quantum and quantized weights with
1/128 scale quanta. At the largest K, the absolute numerator sum stays below
2^24 at the common 1/4096 quantum. Every element therefore equals the independent
f64 oracle numerically across reduction orders. Checksums are computed from
actual post-execute output, canonicalizing signed zero. Exact numeric checks
before and after every timed block are outside the clocks. This does not imply
bit identity for the separate broad, non-dyadic numerical corpus.

Per cell the recorder verifies the actual complete-weight build count and cache
membership. Legacy builds once per warm/timed call and retains zero bytes;
retained native builds once total. With zero retention the two native A/A arms
build once per call each. Protocol 2 additionally checks actual successful NT
calls: twice the round/call count with retention, zero without it. Post-operation Host charges are checked against cache
payload ownership, and both books return to zero after their kernels drop.
These are ledger charges and build counts, not allocated-byte totals or RSS.
Framework-internal scratch and the caller's original encoded operands are outside
this observation. The matching limit on the legacy arm can falsify the claim
that a positive cache knob alone caused the retention difference.

Libtest's test-only cache accounting and route-selector read are included in
measurements; shipped builds omit them. Especially at tiny geometries, these
observations do not establish production latency ratios. The selector's nested
scope, thread independence and unwind restoration have a runnable control.

## Final source results

All 2,160 samples / 30 cells are retained. CPU A/A qualifies 27/30; wall
qualifies 29/30. The retained (8,257,65) CPU cells fail in all three sweeps;
its CPU timing has no qualified result. The nonretained (8,257,65) wall cell
fails in sweep 1. Their exact ratios remain in aggregate.json. No retry replaced
these cells. The initial protocol-1 corpus remains separately archived.

The table gives qualified independent-sweep ranges of legacy/current median
ratios; larger is faster. It is an instrumented execute comparison, including
libtest bookkeeping, not a production or whole-model latency guarantee. For the
three retained geometries with K>=1024, wall ratios are 5.067–8.206. Nonretained
null cells use the same KN arithmetic; qualified wall ratios span 0.992–1.022.
Process CPU includes worker spin and must not be read as CPU savings.

| M,K,N | Retained | CPU ratio range | Wall ratio range | CPU/wall qualified sweeps |
|---|---|---|---|---|
| 2,33,3 | 0 | 1.002–1.017 | 1.000–1.022 | 3/3; 3/3 |
| 2,33,3 | 1 | 520.297–530.484 | 74.033–78.723 | 3/3; 3/3 |
| 8,257,65 | 0 | 0.994–1.009 | 1.007–1.010 | 3/3; 2/3 |
| 8,257,65 | 1 | unqualified | 19.486–22.342 | 0/3; 3/3 |
| 32,1024,1024 | 0 | 0.993–1.002 | 0.993–1.009 | 3/3; 3/3 |
| 32,1024,1024 | 1 | 62.181–63.023 | 8.004–8.206 | 3/3; 3/3 |
| 128,1024,1024 | 0 | 1.000–1.004 | 0.997–1.012 | 3/3; 3/3 |
| 128,1024,1024 | 1 | 20.051–20.641 | 5.160–5.200 | 3/3; 3/3 |
| 16,2048,2048 | 0 | 0.995–1.008 | 0.992–1.001 | 3/3; 3/3 |
| 16,2048,2048 | 1 | 33.729–34.224 | 5.067–5.160 | 3/3; 3/3 |

Final validation:

| Build | CPU library passed / ignored | Session library passed / ignored | Strict all-target Clippy |
|---|---|---|---|
| native | 1969 / 20 | 274 / 0 | both crates pass |
| mlas | 1997 / 34 | 275 / 0 | both crates pass |

The 16 reader controls pass normally and under `-O`; source/binary/Cargo
fingerprints agree across all three final processes. No production selector or
compatibility API is retained.

## Reader and reproduction

The protocol-2 reader requires the complete population, exact six-order sequence,
iteration counts, checksums, build counts, Host charges and actual NT calls. Its 16 controls run
normally and under Python `-O`, including incomplete/duplicate/reordered records,
invalid dimensions/timing values/witnesses and retained clock-specific A/A
failures. Validation does not rely on Python assertions.

```sh
cargo test --locked -p onnx-runtime-ep-cpu --release --lib \
  kernels::matmul_nbits::tests::bench_apple_dense_execute_kn_nt_ab \
  -- --exact --ignored --list
# Hold the shared platform host lock, then use --nocapture instead of --list.
# Preserve every ANT_SAMPLE/ANT_WITNESS record from one process, without merging.
python3 -O scripts/bench_apple_dense_execute.py run-1.csv
python3 -m unittest discover -s scripts -p test_bench_apple_dense_execute.py -v
python3 -O -m unittest discover -s scripts -p test_bench_apple_dense_execute.py -v
```

Final raw transcripts, all records, regenerated summaries, fingerprints and
validation logs are preserved in [`apple-nt-execute-ab`](apple-nt-execute-ab/).
Wrapper command-lifetime CPU/RSS readings are diagnostics for the whole command,
not kernel performance evidence. The archived lock wrapper is the exact one used
on Darwin; its stable lock file must never be unlinked while coordinating runs.
