# Block-format execution-context candidate: not yet adopted

This experimental branch makes BlockQuantizedMatMul and BlockQuantizedMoE select
their decode plan during execution refinement, then consume that same plan once
for layout and calculation. Metadata planning uses the current pool and #2661's
provider context cache key. A constructor-frozen plan can cap construction1→run4
at one shard. The candidate adds no public API, allocator, pool or per-shard quota.
The existing native tile4/reference tile1 and numerical algorithm remain unchanged.

**Do not merge this candidate on functional results alone.** It passes all
functional checks, but this first complete performance matrix leaves regression
questions unresolved. No sample was discarded or replaced and no speedup is
claimed. The next action is to explain or remove the same-pool/single-worker
cost while preserving actual cross-pool parallelism and the preparation contract.

## Functional checks

CPU2068 passed/21 ignored, MLAS2110/37, decoded-layout15, Session376, and strict
all-target CPU/plugin/Session Clippy in both configurations. Actual kernel tests
cover selected-pool capacity, consumed/failed plans, unchanged outputs and scratch
guards. Existing dense untouched-scratch, zero axes, mixed formats, gates,
bitwise references and warm loan/return zero-issuer tests remain green. Session
checks prove same-context memo reuse and1→4→1→4 reprepare for QMoE and both block
operators.

Two GPU parity CPU-oracle helpers were also found bypassing execution refinement.
They now use the existing shared execute_cpu_kernel contract instead of direct
execute or duplicate naked scratch. Two new CPU-only block oracle tests pass;
28 GPU cases compile/list with gpu-tests, and targeted fixture Clippy and the
CUDA honesty source scan pass. **No GPU tests ran.** Oracle fixture sources and
validation are recorded separately from the timed runtime sources because these
test-only changes were made after the performance source was frozen. The oracle
fix was then isolated as commit5b3a0f6fbc4cb735ed01286448cbedf5685c02f3 and
revalidated against its unchanged main runtime, with both host tests, GPU listing,
Clippy and the source guard passing again. That separate source set and all logs
are retained under standalone-oracle-validation/.

## Fixed protocol

Apple M1 Max,10 logical CPUs,32GiB, native release CPU without MLAS symbols.
Baseline61c47ee27c19978a441443ed4e42995a7a172e15 and candidate are fresh builds of
one identical public-session probe. Construction builds and prepares in its pool;
execution prepares/warms/runs in its pool. Baseline remains constructor-frozen;
candidate follows execution width. Initial demand, execution demand and actual
retained peak are independently validated. After trimming replaced idle backing,
streamed paths cap their own authority at physical peak, while dense paths allow
64MiB and retain the independently predicted decoded weight sets. Final traces
must witness the requested path. Warm backing is stable and final trim returns0.

One stable shared Darwin flock covers both builds and the complete matrix;
other chat work coordinates on that lock. No per-core foreign monitor is available
and no idle host is claimed. Terminal parent wait4 measures complete child
user+system CPU, wall/RSS, including setup, five warms, strict prepare/run/output
drop, final trace/validation/output writes and retirement. This is not isolated
kernel latency. No short-window CPU clock is used.

240 processes:3 repetitions*10 cases*(baseline/candidate/null setup+work and
baseline/candidate double). Work budget536870912 MACs; normal calls are budget
floor-divided by R*K*N for MatMul or R*H*I*2*P for top-k2 MoE, P=2/3.
Setup repeats0; double repeats exactly2x. Orders rotate/reverse; one matrix per
candidate. CPU and wall independently require aggregate A/A within5%,
setup<=25%, adjusted double[1.7,2.3], work>=50ms; CPU<=wall*11. Reader controls
also reject13 malformed matrices and preserve failed CPU/wall A/A cells.

## Complete results

Each fixture's complete outputs and argmax agree bitwise across all arms and
matching streamed pool-width cases. All240 processes, outputs, traces and funding
receipts are retained. Ratios are medians of three paired whole-process ratios.

|Operator / R,H,I|Construct→execute|Path|CPU ratio|Wall ratio|CPU / wall qualified|
|---|---|---|---:|---:|---|
|MoE /1,64,128|1→1|streamed separate gate|1.0441|1.0467|yes /yes|
|MoE /8,128,256|4→4|streamed separate gate|0.9913|0.9944|no /no: A/A|
|MoE /8,128,256|4→4|dense separate gate|1.0219|1.0320|yes /yes|
|MoE /8,128,256|1→4|streamed separate gate|1.4681|0.5134|yes /yes|
|MoE /8,128,256|4→1|streamed separate gate|0.9748|0.9743|yes /yes|
|MatMul /1,64,128|1→1|streamed|1.0107|1.0114|yes /yes|
|MatMul /8,128,256|4→4|streamed|1.1417|1.1028|no /no: A/A|
|MatMul /8,128,256|4→4|dense|0.9935|1.0050|no /no: setup/work floor|
|MatMul /8,128,256|1→4|streamed|1.7872|0.6906|yes /no: double scaling|
|MatMul /8,128,256|4→1|streamed|1.0814|1.0787|yes /yes|

Qualification is7/10 CPU,6/10 wall. Qualified does not mean regression-free.
MoE1→4 shows restored parallelism reducing wall while consuming more CPU; this
is a scheduling tradeoff, not a free speedup. MoE singleton and MatMul4→1 cost
remain unresolved, so the candidate is withheld. Other cells are noisy or too
short; their point estimates cannot establish an effect. No CUDA, full-model,
lower-RSS or universally zero-overhead result is claimed.

## Diagnostic sampling

Separate two-second macOS sample runs target only our own baseline/candidate
singleton MoE processes under the same host lock. Both predominantly sample MXFP4
decode and projection; planner frames are rare. This does **not** attribute the
regression to quota/refinement, and does not justify blindly inlining or adding
new heuristics. Instrumented process times are excluded from the matrix. The
raw profiles, exact script and process stdout/stderr are retained under profile/.

The manifest hashes every payload. study/ retains timed source snapshots, patch,
probe/manifest/lock, build logs and binary hashes, receipts, full outputs, logs
and actual dispatch events. validation/ and oracle-validation/ retain separate
source receipts and logs. Binaries remain outside git. Absolute historical runner
paths are provenance rather than a portable recipe. Run python3 verify.py here;
optionally pass the candidate checkout root to check both source sets.
