# Eight-row ARM64 NT tiles with the original arithmetic order

Apple M1 Max, macOS 27.0.1, one thread, Rust 1.98.0 / LLVM 22.1.8,
release aarch64-apple-darwin. Increasing the register tile from four to eight
output rows lowers CPU time by about 20–22% in qualified M=8/16/32, K=512,
N=64 serial fixtures. The M=8, K=4096, N=4 fixture improves about 9.6%.
This is a scoped native kernel optimization. The tiny 4x1x4 case costs about
0.75 ns more per call (8.07 to 8.82 ns); no universal zero-cost claim is made.

## Implementation and correctness

The eight-row loop reuses four B columns transposed in registers across eight
independent output-row chains. A remaining four-row tile, four-row column tails,
and portable bottom rows cover the other extents. Every output consumes the
original K order with separate multiply/add rounding: no FMA, horizontal K
reduction, packed weight copy, heap scratch, resource operation, worker pool or
public API is introduced. Existing runtime NEON detection and portable dispatch
remain. The B register transpose is factored into one private macro.

Tests pin finite output bits for full/partial tiles and K tails, public-dispatch
selection of the eight-row tile with a two-ULP argmax fixture, FMA/reassociation
adversaries, nonfinite classes, signed zero, subnormals and guarded outputs.
NaN payload identity is not promised. Measured candidate disassembly has 84
fmul / 132 fadd versus 44 / 92 before; both have zero fmla/fmadd/faddp.
The inspected q-register stack load/store pattern count is zero. This is a
compiled-path witness, not a general proof about every compiler's register use.
All full default and focused output hashes/row argmaxes agree within each shape.
The serial probe also checks an independent f64 oracle (1e-4+1e-5*abs(reference)).
These fixtures are not a whole-model token stream or quality evaluation.

## Serial default-build evidence

CPU microseconds/call: median of three process medians, nine batches/process.
`Generic` selects the built-in dispatcher, not a forced portable implementation.
Both default executables contain zero MLAS symbols. Original 90 processes:

| M x K x N | Before us | After us | After/base | Null/base | Controls |
|---|---:|---:|---:|---:|---|
| 1 x 4096 x 1 | 3.731968 | 3.697674 | 0.9908 | 0.9910 | pass |
| 1 x 4096 x 64 | 61.849907 | 61.852378 | 1.0000 | 1.0000 | pass |
| 4 x 1024 x 64 | 15.519350 | 15.515789 | 0.9998 | 0.9999 | pass |
| 8 x 512 x 64 | 15.486227 | 12.246054 | 0.7908 | 1.0146 | reject |
| 16 x 512 x 64 | 31.420074 | 24.515180 | 0.7802 | 0.9858 | pass |
| 32 x 512 x 64 | 61.886044 | 48.884123 | 0.7899 | 1.0062 | reject |
| 9 x 35 x 7 | 0.284414 | 0.274434 | 0.9649 | 0.9927 | pass |
| 4 x 4096 x 4 | 3.871733 | 3.881157 | 1.0024 | 1.0003 | reject |
| 8 x 4096 x 4 | 7.783832 | 7.040031 | 0.9044 | 0.9998 | pass |
| 4 x 1 x 4 | 0.008070 | 0.008822 | 1.0932 | 0.9999 | pass |

The original M=8/32 wide cells and M=4 narrow cell fail the +/-5% unchanged
ReLU control drift check and are excluded from conclusions. Their raw results
remain intact. One focused repeat of just those three cells uses the exact
same executable SHAs, common repetitions and nine batches/process (27 processes):

| M x K x N | Before us | After us | After/base | Null/base | Controls |
|---|---:|---:|---:|---:|---|
| 8 x 512 x 64 | 15.982513 | 12.786134 | 0.8000 | 1.0004 | pass |
| 32 x 512 x 64 | 64.190141 | 50.418694 | 0.7855 | 0.9949 | pass |
| 4 x 4096 x 4 | 3.994845 | 4.006162 | 1.0028 | 1.0000 | pass |

All focused duration/control checks pass. M=4 narrow has overlapping process
ranges and no established change. No second repeat is used to seek a better
result. The original qualified M=16 wide and M=8 narrow cells are not rerun.
Full ranges, wall measurements, control ratios and raw batch arrays are in
serial/summary.json, serial/focused-summary.json and corresponding runs.json.

## Same-binary optional MLAS reference

Native and explicit MLAS alternate order in nine paired batches; three isolated
processes per shape. The ratio is the median of within-process paired batch
ratio medians. Absolute times below are marginal medians, in microseconds.

| M x K x N | Native us | MLAS us | Paired native/MLAS |
|---|---:|---:|---:|
| 8 x 512 x 64 | 12.255761 | 8.319578 | 1.4801 |
| 16 x 512 x 64 | 24.563935 | 13.473839 | 1.8340 |
| 32 x 512 x 64 | 48.897485 | 24.079734 | 2.0372 |

Both routes satisfy the f64 error bound; bit identity versus MLAS is not required.
The native/reference gap remains about 1.48–2.04x in these shapes. This does not
graduate GEMM or justify removing the optional research reference. The serial
probe enforces median CPU/wall <=1.2 for native and reference to detect extra
process CPU activity; the actual global MLAS threadpool limit is explicitly one.

## Actual public session follow-through

The fixture executes strict InferenceSession prepare -> run, including output
destruction, with MXFP4/f32 BlockQuantizedMoE, four experts, top-k=2 and SiLU.
Every row selects the same two experts. Dense M=16, H=I=512 retains 4 MiB of
derived weights; streamed M=4, H=256, I=512 retains none. The 64 MiB Host domain
funds actual derived payloads and workspace, not all process allocations.
Conversions, routing/intermediate/output vectors remain outside this scratch
cap. Requirement equals physical workspace (73,728 and 16,384 bytes), usage
remains stable warm, and drop + idle trim returns ledger usage to zero.
Stable used-byte counts alone do not prove zero issuer callbacks; existing
resource tests pin that contract and the math patch adds no resource operations.

The original dense CPU ratio is 0.7906, but its identical-baseline null's ReLU
control drifts +5.99%, so the original dense timing is excluded. The streamed
control passes: 0.951368 -> 0.953217 ms, ratio 1.00194, null ratio 1.00159,
overlapping process ranges and no established endpoint change. A single focused
dense repeat (nine processes) uses the same binaries/repetitions:

| M x K x N | Before us | After us | After/base | Null/base | Controls |
|---|---:|---:|---:|---:|---|
| 16 x 512 x 512 | 1150.122222 | 912.500000 | 0.7934 | 0.9875 | pass |

The focused repeat passes all duration/control checks: 1.150122 -> 0.912500 ms,
20.7% less CPU time, with disjoint process-median ranges (1.134–1.153 versus
0.906–0.926 ms). The identical-baseline null ratio is 0.98751. All 27
original/focused session outputs and argmaxes match within shape, including the
rejected original dense timing. This is a synthetic session fixture, not a full
model, CUDA execution or a universal throughput result.

## Protocol and reproducibility

Baseline: 911fcbd78ee1eb0a82b471b164aa02cff2d33c32.
Measured source: 077746f0329a410f862245701816fb8f4acdc0c8.
Both baseline and candidate were freshly built at the same source path after
cleaning affected release packages; the baseline was not reused from another
experiment. Each driver restores original source before timing. ReLU(16384)
is a separate noinline unchanged control (nine batches x 100,000 calls).
Excluded CPU warmup is 100 ms; excluded pilots select common repetitions targeting
100 ms, using batched pilots for serial tiny calls. Formal batches must take
>=20 ms; both after/base and null/base median controls must stay within 5%.
Baseline/after/identical-baseline null rotate order over three processes.

All heavy work holds the shared stable flock
/tmp/onnx-genai-hostlock-darwin.flock, coordinating with the llama.cpp chat.
Host process snapshots are supplementary context, not an idle-host proof.
Serial thread overrides: RAYON_NUM_THREADS, MLAS_NUM_THREADS, OMP_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and ONNX_GENAI_MLAS_THREADPOOL_THREADS are all one.
Session uses an explicit one-thread Rayon pool; focused.py additionally applies
the same five overrides. Absolute peak RSS is retained; no RSS savings claim.

This directory commits exact drivers/probes, provenance, raw batches, summaries,
disassembly, validation logs, full-buffer hash verification and a normalized
measured-source.patch. source-reconstruction.json proves applying that patch to
the baseline reconstructs both measured source SHA-256s exactly. Original raw
logs, emitted buffers, process snapshots and full symbols remain in
/tmp/nxrt-neon-eight-perf and /tmp/nxrt-neon-eight-session-perf; consumed executable
copies may be removed after SHA verification. Reproduce in an isolated checkout
using cached ORT 1.32.0 and the shared lock. Drivers intentionally preserve exact
original paths and modify/rebuild their checkout; do not run in a peer's tree.

## Validation and limits

Native CPU: 2,016 passed / 20 ignored; session: 295 passed. Optional MLAS CPU:
2,058 passed / 35 ignored; session: 297 passed. Strict all-target CPU/session
Clippy passes in both modes. Exact commands and outputs are retained in
validation.py and validation-*.log. No CUDA hardware, x86 execution,
Linux/Windows ARM64 execution or whole-model generation was tested here.
