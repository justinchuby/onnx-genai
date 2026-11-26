# Packed activation arithmetic: first controlled CPU study

All four timing cells are **diagnostic**. None cleared the frozen quality gates,
so this study publishes no Q8 speedup or slowdown estimate and makes no change
to default arithmetic. It does establish the complete-output and managed
workspace invariants below through the real optimized CPU execution path.

## Measured scope

Runtime source: `7a012a5fc7dda0dd214932a27fa54e8f115e7f30`.
Apple M1 Max, 10 physical/logical cores, 32 GiB RAM. The release executable uses
the default full CPU EP, one Rayon/decode thread, and the production CPU package's
single-codegen-unit override. All arithmetic arms live in the same binary,
SHA256 `556621dbad80baae203de3186fbf34dc59ebde574fd1e96e69ad41519a79dfbb`.
Compiler, OS, environment and source hashes are in
[provenance](results/provenance.json).

Four cells cover Q4_0 and Q8_0 weights, rows 1 and 8, K=1024, N=256. These
synthetic weights fit comfortably in CPU cache. The Float32 arithmetic baseline
explicitly forces row streaming; it does not stand for the warm dense-weight
cache path. A separate real dense-expansion positive control checks that the
expansion counter is live. Q8 arithmetic bypasses dense-f32 expansion even with
a sufficient configured cache budget. No model, MoE routing, GPU, quality or
memory-bandwidth extrapolation follows from this projection study.

The fixture is deliberately exactly quantizable: every 32-value activation
block has an integer 127 anchor and bounded integer lanes, so its stored Q8
scale is exactly one. Weight scales and bias are dyadic. Integer bounds keep
every floating-point prefix exactly representable. An independent integer
coordinate oracle checks **every output bit**, outside each timed window;
the oracle never calls the production decoder. Q4 high/low halves have distinct
codes. K=1003 is a separate partial-block correctness precheck. This fixture
does not remove the general numerical difference demonstrated by the
[argmax-reversal study](../packed-q8-dot-contract/README.md).

## Deterministic resource result

The complete cohort ran 136 fresh processes and 320 windows, containing
1,036,816 calls to the prepare/checkout/compute-or-setup/return seam. Every
window had zero real issuer claim and return callbacks, including zero-byte
callbacks. An explicit zero-byte positive control demonstrates that the
observer can detect them. Every output, fixed workspace identity, no-expansion
condition, and final managed-memory settlement assertion passed.

Both weight formats have these complete declared projection demands:

| Rows | Float32 scratch bytes | Q8 scratch bytes | Shared retained physical bytes |
| ---: | ---: | ---: | ---: |
| 1 | 16384 | 1088 | 16384 |
| 8 | 24576 | 16896 | 24576 |

Each process first admits one real provider-owned lane at the maximum arm
demand, then all arms borrow that same physical allocation. The smaller Q8
logical demand therefore **does not reduce the retained physical peak in this
mixed-arm study**. Grouped scratch includes its full column staging as well as
encoded activations. The initial funding and dense positive control occur
before the warm observation window. Trim/settlement occur afterward.

These are the declared managed workspace bytes. Existing activation/bias
materialization and final output staging remain ordinary Host allocations;
this is not total process memory accounting. Zero issuer callbacks proves the
absence of admission/refund calls in the warm seam, not a measured zero-cycle
cost for provider checkout or inference.

## Timing qualification and retained failures

[The frozen protocol](results/protocol.json) uses three repetitions, ABBA and
BAAB, two Float32 copies for A/A nulls, complete output validation, and fresh
process work/copy/setup/double-call CPU controls. Calls were calibrated once
and then frozen. The confirmatory cohort held the same shared Darwin flock
for its entire run. No measured request was repeated based on its outcome.

All cells exceeded the predeclared foreign **process-average** CPU gate.
Grouped cells also failed the whole-process setup-fraction control. This
fraction includes fixture/oracle construction, positive controls and warmup;
it requests longer control windows rather than isolating workspace checkout
overhead. [The summary](results/summary.json) retains every failing gate and
suppresses timing comparisons for failed cells. Raw samples and all controls
remain in [processes.json](results/processes.json).

The host monitor aggregates `host_processor_info` counters and subtracts the
child's complete `wait4` CPU time. Its process average includes a minimum
250 ms process lifetime with an **untimed idle tail** for short processes.
It cannot bound competition inside an individual warm window or identify
the cores on which the child ran. A/A, range and CPU/wall checks are additional
controls, not proof of zero interference. Neither helper changes thread affinity.

Two pilots are retained in [pilots.json](results/pilots.json). The first stopped
before paired measurements when `host_statistics` returned repeated snapshots.
Apple's [pinned kernel source](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/host.c)
contains a cached/rate-limited user statistics path; the replacement API reads
processor counters directly. The second pilot was interrupted by failed
pre-start idle admission. Its already completed samples were retained across
an attempted continuation and excluded from the later confirmatory cohort.
The final cohort kept the same executable, calls, workloads and timing gates;
only the maximum **pre-start** idle wait was extended. No timing result was
repeated until favorable. A background Crash Reporter was independently
observed consuming approximately one core during this investigation.

## Reproduction

The standalone probe's nested workspace pins dependency resolution in its
Cargo.lock and explicitly repeats the production CPU codegen override.
For exact reproduction, use a checkout of the recorded runtime commit and copy
this study directory into that tree before building; path dependencies resolve
the runtime source in that checkout:

```sh
ORT_ROOT=/path/to/existing/ort \
CARGO_TARGET_DIR=/tmp/nxrt-q8-study-target CARGO_BUILD_JOBS=1 \
cargo build --offline --release --manifest-path \
  docs/benchmarks/packed-activation-study/probe/Cargo.toml
```

The current dependency build needs local ORT headers/runtime preparation;
the measured executable's dynamic dependency list contains Accelerate and
system libraries, with no ORT dylib. This study does not invoke ORT inference.
[Build evidence](results/build.json) records the actual dependency list.

`run.py` requires an inherited descriptor 3 holding the stable shared flock at
`/tmp/onnx-genai-hostlock-darwin.flock`. Use the existing host coordinator; keep
that descriptor open for the complete batch and never unlink/replace the lock.
System hardware/process reads require permission on a sandboxed Darwin host.

```sh
python3 docs/benchmarks/packed-activation-study/with-lock.py \
  /absolute/probe-binary /fresh/results-directory
python3 docs/benchmarks/packed-activation-study/analyze.py /fresh/results-directory
python3 docs/benchmarks/packed-activation-study/verify.py /fresh/results-directory
```

The first invocation acquires the existing shared lock and passes FD 3 through.
Use the recorded calibration-protocol file's absolute path when supplying
`--calls-from`; it reuses the
original counts only for the identical executable/geometry. `--resume` continues
an interrupted directory without rerunning completed requests; analysis rejects
cells spanning multiple lock epochs. Fresh eligible timing evidence requires a
quiet, uninterrupted full cohort and all frozen controls to pass. Preserve
failed cohorts when improving the experimental setup.

`processes.json` embeds every original JSON stdout as `result`; console-file
hashes and the common stderr text are in
[console-manifest.json](results/console-manifest.json). Private process inventories
and the host nodename are omitted from the public export. `verify.py` checks
request/window identities, every fixture digest using a separate Python integer
oracle, quota/physical-capacity invariants and the measured source hashes.
