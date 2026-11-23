# In-place serial expert activation and minimum bank capacity

Runtime candidate: `af268c19e3cb6e93c4901db7a5b11d3fdc5464d7`.
Prior static-bias bank kernels: `f0a2cd115`; old modules:
`509144567` (Block uses owned FC results; Float already uses admitted phase banks). This single native Apple M1 Max synthetic CPU experiment covers the
same prior eight cases, not model/CUDA throughput, RSS, a full-process memory cap,
or an adoption audit of a candidate rebased onto newer main.

Serial gated activation now compacts already-biased FC1 in its own initialized
prefix. Separate FC3 is a disjoint slice; fused gate/linear pairs are copied
before writing an earlier/equal destination. Forward row order preserves unread
rows. A uniformly holds gather/output; B holds projections/activation. The extra
activation clear and bank-role switch disappear. The unused applied-bias marker
and private bias trait are deleted; grouped activation retains one pending-bias
value with its existing optional +0 arithmetic. There is no public EP interface,
new allocation/lease/cache, per-expert issuer call, backwards alias or change to
projection/accumulation order.

Serial metadata now declares A=hidden and B=FC1+optional FC3. Float adds its old
A>=inter and B>=hidden staging maxima only when the parallel grouped driver is
possible; its parallel scheduling/storage remains unchanged. Max routed rows in
an actual group never exceeds metadata total rows at the same fixed work
threshold; grouped slots remain bounded by rows*k. Thus the smaller declaration
still covers every possible serial path. The independently reviewed source
lifetime/bound proof found no blocker, but does not substitute for performance.
Retained backing remains charged until verified release.

Native and MLAS-reference final validation: BlockMoE 29 pass each, FloatMoE 26
pass plus one existing ignored each, BlockMatmul 38 pass plus one existing ignored
each; strict CPU all-target Clippy in both configurations, format and diff checks
pass. The named compaction test covers separate/interleaved/split gates, zero
rows/intermediate width, multiple rows, future-source preservation and untouched
capacity; signed zero/pending +0 still pass. Existing asymmetric/empty-axis,
forced serial/grouped, full numerical/reference, missing/short-workspace,
admitted-failure-before-publication, exact charge, first-warm zero workspace
issuer callback (including zero-byte calls), and trim-to-zero guards remain.
No claim about all transient weight issuer callbacks is made.

The fixed unsampled protocol uses four public sessions in one executable/CPU
crate: old kernels, prior static-bias banks, compacted minimum banks, and identical
old A/A. The CPU thread-budget/allocator/decoder/Rayon globals are shared; factory
choice is cold only, with positive/stable creation witnesses. Each arm has an
independent funded physical pool and books. Four construction/batch rotations
and eight interleaved batches balance all arms. Excluded >=50 ms CPU pilots target
250 ms batches; observer overhead may shorten actual batches. Strict public
prepare/run/output drop lies within the compute timer. An unchanged Relu control
precedes each arm. CPU time sums all threads; wall brackets the loop. The stable
shared Host flock covers build and sweep, with process snapshots retained.
There is no CPU-affinity or scheduler/DVFS stability claim.

Every formal process and excluded smoke compares full output bytes, shape and
finite values before and after timing; exact declared/funded workspace and
selected dense payload bytes; stable warm ownership; final trim-to-zero. The
script checks actual compact-vs-prior capacity reduction and compares compact
capacity against the old driver using independently stated width arithmetic.
All 32 formal processes and eight smokes validate within each case. No source
changes occur during the sweep; copied old module prefixes are reconstructed
exactly in `copy-proof.json` (tests omitted, own Block helper imports redirected,
rustfmt only).

Gates are predeclared: all CPU/control batches >=20 ms, every compute CPU/wall
ratio >=0.9, each process's paired-median Relu and old A/A CPU ratios within 5%.
A case qualifies only if all four processes pass. Ratios below are median process
paired-median wall / CPU changes, retained even for failed cells. Qualification
alone does not prove a small effect distinguishable from A/A spread; consult all
per-process ratios. No failed cell is discarded or rerun to get favorable results.
Shape order R,H,I,threads,dense,gate,op; gate 0 identity, 1 separate FC3,
2 interleaved SwiGLU, 3 split SwiGLU.

| Shape | All gates | Compact/prior banks | Compact/old driver | Old A/A | Prior → compact funded workspace bytes |
|---|---|---:|---:|---:|---:|
| `[1, 32, 64, 1, 0, 0, 'block']` | no; p1,p2 | -0.250% / -0.251% | -2.349% / -2.340% | -0.215% / -0.198% | 1536 → 1536 |
| `[8, 128, 256, 1, 0, 0, 'block']` | no; p0,p1,p2 | -0.594% / -0.710% | -0.820% / -0.740% | +0.110% / +0.105% | 26624 → 26624 |
| `[1, 64, 128, 1, 1, 0, 'block']` | yes | -0.885% / -0.883% | -3.140% / -3.139% | -0.027% / -0.037% | 3072 → 3072 |
| `[8, 128, 256, 4, 0, 1, 'block']` | yes | -1.230% / +0.360% | +3.396% / +0.605% | -1.311% / +0.078% | 49152 → 45056 |
| `[8, 128, 256, 4, 1, 2, 'block']` | yes | +15.428% / +7.747% | -2.699% / -1.275% | +2.968% / +2.168% | 53248 → 49152 |
| `[8, 128, 256, 4, 0, 3, 'block']` | yes | -0.259% / +1.083% | +4.835% / +2.077% | +0.811% / +1.219% | 53248 → 49152 |
| `[1, 64, 128, 1, 0, 0, 'float']` | no; p3 | -0.145% / -0.148% | -0.009% / +0.179% | +0.076% / +0.119% | 1024 → 1024 |
| `[8, 128, 256, 4, 0, 1, 'float']` | no; p0,p1,p2,p3 | -1.673% / +1.801% | -7.926% / -4.256% | +7.061% / +4.303% | 28672 → 24576 |

This record does not automatically adopt #2469. Remaining regressions or failed
controls must be considered at the actual candidate head, with the original
resource scope and lifetime safeguards intact. The temporary factory selector
and old module copies are archived only as a reconstruction patch and removed
from product sources after the sweep.

`runtime-fix.patch` records the reviewed change; `instrumentation.patch` applies
to the candidate for reproduction. Sources, manifests, lock protocol, complete
raw batches, failed controls, source/binary hashes, native/MLAS build/test/lint and
Host logs are retained. Manifest paths point to the isolated worktree and need
adjustment elsewhere. The measured binary remains
`/tmp/nxrt-bank-inplace-gate/inplace-gate-probe`. All scope limitations still apply
when comparing this report with earlier paired/static-bias studies.

Decision: **do not adopt this candidate or update #2469 to this runtime head**. Four of eight cells qualify. The qualified dense interleaved case regresses +15.43% wall/+7.75% CPU versus the current static-bias bank draft, despite a 4,096-byte lower retained lane. Streamed split still costs +4.84% wall/+2.08% CPU relative to the old driver. The qualified separate-FC3 streamed case is +3.40% wall/+0.61% CPU versus old, so the original issue is not resolved. Smaller physical retention and correct coarse ownership do not outweigh the observed performance tradeoffs. Failed controls remain, including the unqualified Float grouped-size case with large A/A drift; no speed result is claimed for them. Keep the previous draft at `244fda6c2`; investigate the activation/storage implementation rather than change quota granularity.
