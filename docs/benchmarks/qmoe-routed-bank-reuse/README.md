# QMoE contributions and final reduction share the FC lane

QMoE's one StepScoped composite now contains FC bank A, FC bank B and
route-ordered contributions. Every expert first consumes its borrowed FC banks
and writes its original contribution slots. Once all experts and the prefetch
scope end, bank A is dead and is large enough for R*H output values. Its output
prefix is cleared to +0, then receives the original row/route-slot += sequence.
Only after successful reduction is the caller's real tensor written.

This removes the independent contribution and final-output Vecs without adding
an accumulator region, allocator, public lifecycle API, pool or per-expert
quota. FC capacities remain A/B from the shared driver; only contributions
add R*k*H f32 capacity. Each row selects exactly k unique experts after existing
k<=experts validation. Contributions need top-k capacity; FC banks do not.
Counts, individual f32 layouts and their composite sum are checked.

Zero rows/hidden do not gain an early return or bypass weight/routing checks.
Prefetch owns weights only. Original +0 initialization, signed-zero/NaN/Inf
arithmetic, expert issue/read/join order, route accumulation, strided outputs
and input/output aliases remain intact. Test backing is explicitly poisoned;
metadata/product/composite overflow and real alias/canary regressions are covered.
Warm workspace loan/return has zero issuer calls while retained capacity remains
charged until trim. Positive transient weight requests keep their old protocol.

Retained workspace increases because contributions were previously funded
outside the lane. This does not claim lower RSS or a full-process cap. Routing,
widening, dynamic decoded-row/column scratch and caller output allocation remain
outside this bound. Final accumulator capacity is reused from A, not charged
again as a separately live region.

## Baseline and historical gate

The older [draft #2451](https://github.com/justinchuby/onnx-genai/pull/2451)
used gather/contributions/a separate accumulator while FC results remained
owned, and kept its single-row acceptance unresolved. Its historical data is
preserved on that branch; its short-window RUSAGE_SELF tables are not current
terminal-process CPU evidence and do not establish attribution.

This independent candidate starts from merged #2653, where single-row FC banks
already require one workspace checkout. It adds contributions to that existing
lane and reuses A after FC compute. It does not adopt the earlier experimental
active-scope API, skip singleton accounting, or infer that old measured costs
have disappeared. It is a different before/after comparison.

## Protocol

Same Apple M1 Max, 10 logical CPUs,32GiB/native release CPU fixture and public
strict InferenceSession prepare/run/output-drop probe as the
[FC bank study](../qmoe-fc-projection-banks/README.md). Probe source and Cargo.lock
are byte-identical. Baseline reuses that study's frozen candidate executable,
with its SHA256 and candidate source hashes verified against this change's
base. Candidate is built at the same original probe path with the same lock.
Both builds exclude MLAS symbols. This is separate source-controlled binaries;
null is a duplicate baseline invocation, not one-address-space A/B.

One stable Darwin flock covers the entire build/matrix. Parent terminal wait4
collects child user+system CPU and wall/RSS. All setup, five warm calls, strict
prepare/run/output drop, final traced dispatch, full output validation/write,
retirement and trim are included; there is no short-window CPU clock. No
per-core foreign monitor is available and no idle-host claim is made.

The graph uses inline integer-affine 4-bit weights, four experts, top-k2,
block16, alternating routes and optional separate SwiGLU. Streamed cases cap
that binary's actual shared authority at its own prepared workspace capacity,
so active workspace leaves no room for positive transient expert grants. Dense
cases allow64MiB and expand transient sets. Both retain zero derived payload.
Final trace must witness qmoe.streamed_rows exactly for streamed cases.
No mmap-prefetch, CUDA or full-model performance is measured here.

144 processes:3 repetitions*6 shapes*(baseline/candidate/null setup+work plus
baseline/candidate double work). Normal calls=floor(268435456/(R*H*I*2*P))
where P=2 or3 projections; setup repeats0, double exactly2x. Work/arm order
rotate. One complete matrix per candidate, retaining every repetition.
Predeclared qualification: aggregate A/A within5%, setup<=25% of work, adjusted
double in[1.7,2.3], each work process>=50ms CPU, physical CPU<=wall*11,
correct independently computed actual funding, finite nonzero bit-identical
complete outputs/argmax, observed path, and trim0. Synthetic reader controls
accept a complete matrix and reject malformed/vacuous evidence.

## Complete results

| Rows / H / I | Width | Weights | Gate | Candidate / baseline CPU | Full paired range | A/A | Qualified | Workspace bytes |
|---|---:|---|---|---:|---|---:|---|---|
| 1 / 32 / 64 | 1 | row streamed | none | 0.9893 | 0.9872–1.0248 | 0.9900 | yes | 384 → 640 |
| 1 / 64 / 128 | 1 | dense transient | none | 0.9965 | 0.9952–1.0156 | 1.0050 | yes | 768 → 1280 |
| 1 / 64 / 128 | 1 | row streamed | separate | 0.9910 | 0.9879–1.0008 | 0.9821 | yes | 1536 → 2048 |
| 8 / 128 / 256 | 1 | row streamed | none | 1.0004 | 0.8785–1.0091 | 1.0151 | yes | 12288 → 20480 |
| 8 / 128 / 256 | 4 | row streamed | separate | 0.9921 | 0.9899–0.9993 | 0.9957 | yes | 24576 → 32768 |
| 8 / 128 / 256 | 4 | dense transient | separate | 0.9974 | 0.8698–1.0001 | 1.0011 | yes | 24576 → 32768 |

Ratios are medians of three paired whole-process CPU ratios, with all pairs
retained. Aggregate qualification does not make every repetition quiet or prove
universally zero overhead. A small cost is not excluded. These are synthetic
CPU session observations, not pure kernel latency, model throughput or CUDA.

Validation: full native2061/21 ignored, MLAS2103/37 ignored, session373,
MLAS decoded layout14 and strict all-target CPU/plugin/session Clippy both
configurations passed. The additional IEEE regression was then added with
runtime-prefix SHA256 unchanged, and run by full identity in both configurations,
with final Clippy both and the host-only CUDA-package CPU oracle. The runtime
hash receipt records that distinction. No GPU tests were executed.

All source/data payloads are hashed by manifest.json. study/ retains the fixed
protocol, patch/source snapshots, probe/manifest/lock, build logs and binary
hashes,144 terminal receipts, full stdout/stderr, complete f32 output bytes and
actual path traces. Baseline executable bytes are inherited by verified hash
from #2653; its metadata receipt is retained. Binaries themselves stay outside
the repository. Historical runner paths are provenance, not a portable recipe.
Run python3 verify.py here; optionally pass the measured source checkout root.

Single-row paired medians are -1.07%,-0.35%,-0.90%; aggregate baseline/null
ratios are0.9900,1.0050,0.9821. These point estimates do not establish a stable
speedup. A small cost remains possible. Multirow point estimates are near flat.
All six aggregate qualification cells pass, but individual noisy pairs remain:
case3 repetition0 has null/baseline0.8847 and candidate/baseline0.8785;
case5 repetition1 has null/baseline0.8705 and candidate/baseline0.8698. Their
apparent gains are not claimed as improvements, and no sample was replaced.
