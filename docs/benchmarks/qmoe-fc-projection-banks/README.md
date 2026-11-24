# QMoE FC projection banks

QMoE now uses the private two-bank MoE driver for grouped input, FC1/FC3,
activation and expert output. Dense transient and row-streamed experts fill
borrowed destinations. One StepScoped lane is sized from all input rows and
reused serially, without a top-k/expert multiplier or new projection grants.
Only the main compute thread borrows it; prefetch retains independent weights.

Preparation and dispatch share geometry from input0 and FC2's packed K axis
multiplied by 8/bits. A singleton still borrows its original input row but now
funds FC capacity. Rows0 have empty banks; hidden0 keeps validation and can
still need FC1 capacity. Missing FC2 metadata now fails explicitly.

For R rows, H hidden and I intermediate: nongated A=R*H, B=R*I; gated
A=R*max(H,I), B=R*max(FC1+separate_FC3,H), in f32 elements. No compatibility
interface is kept. The owned expert driver is test-only, a separate reference
against which dirty-bank/guard and gate/quantization cases are checked.

This is an ownership/accounting change, not an RSS-saving or process-cap claim.
The former FC Vecs were outside the lane; their new retained capacity increases
its declared requirement and can expose an insufficient cap. Contributions,
final accumulator/output, widening, dynamic row decoder and column-major
decode staging remain separate. Weight admission order and final contribution
accumulation/publication order are unchanged. Warm workspace request/return is
callback-free; existing positive transient weight requests can still call their
issuer and are not described as zero-cost.

## Measurement protocol

Apple M1 Max, 10 logical CPUs, 32 GiB, macOS; native optimized builds, no MLAS
symbols. The same source-controlled public InferenceSession strict
prepare/run/output-drop probe is built before/after the change. null is a new
baseline process. Separate binaries, not one-address-space A/B. A stable Darwin
flock covers the complete build/matrix. Per-core foreign CPU monitoring is
unavailable; no idle-host claim is made.

One com.microsoft QMoE graph has four deterministic integer-affine 4-bit
experts, block16, top-k2, alternating routes, nonzero input/output and optional
separate SwiGLU. Inline weights exclude mmap offload and prefetch: their
correctness remains covered by existing serial/prefetch tests, but this matrix
does not measure prefetch performance or CUDA/model throughput.

`dense=false` is established by lowering the SAME shared authority to that
binary's prepared workspace bytes before any inference. The active workspace
fills it, forcing positive transient expert requests to decline. A zero dense
cache-retention knob alone would not do this and is not used as path evidence.
Actual funding/zero availability/zero oversubscription are asserted, and a final
traced dispatch must contain `qmoe.streamed_rows` exactly in streamed cases.
`dense=true` allows 64 MiB for complete transient expert expansion; no retained
expert cache is used. Final retained derived payload is zero in both cases.
Each binary's own workspace bound determines its streamed cap, so the baseline
gets no spare capacity from the candidate's larger bound.

Parent wait4 collects TERMINAL child user+system CPU. It includes model/thread
pool setup, allocation, five fixed warm calls, fixed repetitions of strict
prepare/run/output drop, one final traced dispatch, output checking/writing,
retirement and trim. No short-window process clock is used. RSS is absolute
process peak and is not attributed to banks. Every receipt must close against
10 logical CPUs plus one-CPU tolerance, record its exact workload/funding, and
trim to zero. Complete f32 outputs and argmax must match across every arm and
work amount; finite and nonzero output is required.

144 processes = three repetitions * six shapes * (setup/work for
baseline/candidate/null plus double work for baseline/candidate). Normal calls
are floor(268435456/(R*H*I*2*projection_count)), with two/three projections.
Setup repeats zero calls; double repeats exactly twice the normal count. Arm
and work order rotate. Each candidate gets one complete matrix. All samples,
including noisy or unqualified cases, are retained.

Predeclared aggregate qualification: baseline/null median CPU within5%, setup
median <=25% of work for each arm, setup-adjusted doubled work in[1.7,2.3], and
every work process >=50ms CPU. The reader accepts a complete synthetic matrix,
rejects ten malformed/vacuous variants and retains failed A/A cells unqualified.

## Complete results

| Rows / H / I | Width | Weights | Gate | Candidate / baseline CPU | Full paired range | A/A | Qualified | Workspace bytes |
|---|---:|---|---|---:|---|---:|---|---|
| 1 / 32 / 64 | 1 | row streamed | none | 1.0036 | 1.0023–1.0086 | 0.9824 | yes | 0 → 384 |
| 1 / 64 / 128 | 1 | dense transient | none | 1.0037 | 1.0015–1.0059 | 0.9999 | yes | 0 → 768 |
| 1 / 64 / 128 | 1 | row streamed | separate | 1.0095 | 0.9791–1.0169 | 0.9881 | yes | 0 → 1536 |
| 8 / 128 / 256 | 1 | row streamed | none | 0.9844 | 0.9770–1.0212 | 0.9795 | yes | 4096 → 12288 |
| 8 / 128 / 256 | 4 | row streamed | separate | 0.9759 | 0.8883–1.0272 | 1.0317 | yes | 4096 → 24576 |
| 8 / 128 / 256 | 4 | dense transient | separate | 0.9994 | 0.9630–0.9994 | 1.0031 | yes | 4096 → 24576 |

These are medians of three paired whole-process CPU ratios with every pair
retained. Aggregate qualification does not establish that each repetition was
quiet, universal zero overhead, pure kernel speed or full-model throughput.
Unqualified cells remain visible and do not support a speed claim. Existing
transient weight admission/rejection costs are included in both arms.

Validation: native CPU2059/21 ignored, MLAS2101/37 ignored; decoded layout14
in each configuration; session373; strict CPU/plugin/session Clippy in both
configurations. The CUDA package's host-only CPU oracle runs without CUDA;
GPU-feature QMoE fixtures are compiled/listed only, never represented as device
execution. The initial focused log retains three error-context/order failures
and the subsequent57-pass correction, before final validation.

All data and code payloads are hashed by manifest.json. study/ contains the
protocol, source patch/snapshots, probe/manifest/lock, both build logs, binary
hashes, full process receipts, stdout/stderr, complete output bytes and dispatch
traces. reader-controls.json contains only synthetic reader checks. Validation
logs are compressed separately. Executable bytes are not committed; their
hashes bind the frozen builds. Paths in the historical runner are provenance,
not a portable invocation recipe.

Run `python3 verify.py` in this directory; pass a checked-out source root to
also verify the measured candidate kernel sources. No CUDA device was available.

The three single-row candidate medians are +0.36%, +0.37% and +0.95%.
A small CPU cost is not excluded; these do not establish universally free
workspace borrowing. Four-width streamed repetition2 has null/baseline1.139
and candidate/baseline1.027; it remains in the full pair range despite aggregate
qualification. No general speedup is claimed for negative point estimates.

Initial postprocessing accidentally read a prior experiment directory. Its
independent workspace formula rejected that data (baseline1280 vs expected0).
The default path was corrected and the same frozen144-process data read
successfully, with no timing rerun or sample replacement. The diagnostic is
retained in study/reader-path-diagnostic.json.
