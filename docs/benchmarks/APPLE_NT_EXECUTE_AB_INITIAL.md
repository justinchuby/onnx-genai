# Initial Apple execute comparison: NT with and without retention

This initial source (`afe9cc8ed`) selects NT even when retention is disabled.
All data is preserved separately from the later reuse-gated source. For
M=128,K=N=1024 with zero retention, three qualified wall-clock sweeps show
legacy/native ratios 0.955–0.982: the transient NT route is slower in this
instrumented corpus. Retained NK shows a different result. This motivates the
shape-independent reuse policy, not a model/size exception or a production
latency guarantee. These records are not pooled into the final source's results.
The initial protocol-1 reader is archived as `reader-v1.py.txt`; the current
reader deliberately follows the final protocol instead.


This diagnostic compares actual factory-created `MatMulNBitsKernel::execute`
calls. A scoped, thread-local, test-only selector makes the NT capability
unavailable, taking the existing KN fallback. It does not duplicate the old
kernel or restore unchecked BLAS entry. Both routes use the current validated
NN/NT boundary. No production selector or environment knob is added.

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
prepare each call. Twenty-four rounds use six cyclic orders, balancing arm
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
build once per call each. Post-operation Host charges are checked against cache
payload ownership, and both books return to zero after their kernels drop.
These are ledger charges and build counts, not allocated-byte totals or RSS.
Framework-internal scratch and the caller's original encoded operands are outside
this observation. The matching limit on the legacy arm can falsify the claim
that a positive cache knob alone caused the retention difference.

Libtest's test-only cache accounting and route-selector read are included in
measurements; shipped builds omit them. Especially at tiny geometries, these
observations do not establish production latency ratios. The selector's nested
scope, thread independence and unwind restoration have a runnable control.

## Reader and reproduction

The protocol-1 reader requires the complete population, exact six-order sequence,
iteration counts, checksums, build counts and Host charges. Its 16 controls run
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

Initial raw transcripts, all records, regenerated summaries, fingerprints and
validation logs are preserved in [`apple-nt-execute-ab`](apple-nt-execute-ab-initial/).
Wrapper command-lifetime CPU/RSS readings are diagnostics for the whole command,
not kernel performance evidence. The archived lock wrapper is the exact one used
on Darwin; its stable lock file must never be unlinked while coordinating runs.

## Initial results

All 2,160 samples / 30 cells remain archived. CPU A/A qualifies 28/30; wall
qualifies 30/30. CPU sweep 2 and 3 at (8,257,65), retained, are unqualified;
their A/A ratios are retained in aggregate.json, not replaced by new samples.

| M,K,N | Retained | Qualified CPU ratio range | Qualified wall ratio range | CPU/wall qualified sweeps |
|---|---|---|---|---|
| 2,33,3 | 0 | 1.716–1.730 | 1.645–1.657 | 3/3; 3/3 |
| 2,33,3 | 1 | 529.463–536.076 | 78.245–79.573 | 3/3; 3/3 |
| 8,257,65 | 0 | 1.051–1.071 | 1.046–1.050 | 3/3; 3/3 |
| 8,257,65 | 1 | 81.282–81.282 | 19.629–20.000 | 1/3; 3/3 |
| 32,1024,1024 | 0 | 1.012–1.018 | 0.969–1.001 | 3/3; 3/3 |
| 32,1024,1024 | 1 | 62.192–62.831 | 8.097–8.125 | 3/3; 3/3 |
| 128,1024,1024 | 0 | 1.006–1.016 | 0.955–0.982 | 3/3; 3/3 |
| 128,1024,1024 | 1 | 20.171–20.405 | 5.001–5.215 | 3/3; 3/3 |
| 16,2048,2048 | 0 | 1.019–1.025 | 1.061–1.070 | 3/3; 3/3 |
| 16,2048,2048 | 1 | 33.396–34.001 | 4.995–5.108 | 3/3; 3/3 |
