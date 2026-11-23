# Static expert activation-bias state

Runtime fix: `f0a2cd115` on candidate FC banks `71b105629`; old allocation-based
expert modules: `509144567`. This is a focused native Apple M1 Max CPU experiment
on four affected gated cases, not a complete adoption audit of #2469. No CUDA,
model throughput, RSS saving, whole-process cap, or universal performance claim.

[CPU samples and compiled assembly](https://github.com/justinchuby/onnx-genai/blob/d4bb7d2a2/docs/benchmarks/block-moe-fc-bank-cpu-sample/README.md)
identified a runtime bias-state discriminant inside the old row helper's element
loop. The fix replaces the private enum with two static types and a private
monomorphized value operation: Applied is identity; Pending retains optional bias
addition, including +0 when absent. There is one shared activation algorithm.
The new signed-zero/layout test pins separate, interleaved and split gate layouts
with absent/present bias. No backward alias, public EP API, extra pool/lease,
quota operation, shape policy, arithmetic reorder or bank-capacity change is added.
The recorded Applied specialization has no runtime bias-state discriminant or
pending-bias payload loads; ordinary layout and bounds checks remain.

Local native and MLAS-reference validation: BlockMoE 29 pass each, FloatMoE 25 pass
plus one existing ignored each, BlockMatmul 38 pass plus one existing ignored each;
strict CPU all-target Clippy in both configurations, format and diff checks pass.
The exact resource regression still checks complete funded banks/accumulator,
zero warm workspace issuer callbacks including zero-byte calls, failure before
output publication, and trim-to-zero. These are not claims about all weight
issuer callbacks. Named filters were listed and the new signed-zero test ran.

One unsampled sweep, fixed before measurement, uses four independent public
sessions in each process: old kernels, original FC banks, static-bias FC banks,
and identical old-kernel A/A. All share one CPU crate/executable and thread-budget,
Rayon, decoder and allocator globals; factory selection is cold only. Each has
its own physical pool and books. Four construction/batch rotations and eight
interleaved batches balance the four arms. Excluded >=50 ms CPU pilots target
250 ms batches; pilot observer overhead may shorten actual batches. Strict
prepare/run/output drop lies inside each compute timer; a shared unchanged Relu
control precedes each arm. CPU time sums all threads; wall brackets the loop.
The stable shared host flock spans build and sweep, with Host snapshots retained.
No affinity, scheduler or DVFS stability is asserted.

Every process checks complete output bytes/finite values/shape before and after
timing, exact retained workspace and selected dense bytes, positive/stable factory
creation counters, and final trim-to-zero. Original and static banks must have
identical funded capacity. All 16 formal processes and four excluded smokes compare full outputs within each case.
The two old module versions are reconstructed exactly in `copy-proof.json`, with
only omitted test modules, Block helper import redirection and rustfmt.

Gates are fixed: all CPU/control batches >=20 ms, every compute CPU/wall >=0.9,
each process's paired-median control and old A/A ratios within 5%. A case qualifies
only if all four processes pass. All failed cells remain; no favorable reruns.
Table entries are median process paired-median wall / CPU changes, retained even
when disqualified. Shape order is R,H,I,threads,dense,gate,op. Gate 1 is separate
FC3, 2 interleaved SwiGLU, 3 split SwiGLU.

| Shape | All gates | Static/original banks | Original banks/old | Static banks/old | Old A/A |
|---|---|---:|---:|---:|---:|
| `[8, 128, 256, 4, 0, 1, 'block']` | no; p1 | +1.256% / +0.717% | +10.706% / +1.738% | +11.914% / +2.286% | +0.174% / -0.318% |
| `[8, 128, 256, 4, 1, 2, 'block']` | yes | -7.052% / -3.705% | -12.822% / -5.776% | -17.657% / -9.641% | +2.265% / +0.287% |
| `[8, 128, 256, 4, 0, 3, 'block']` | yes | +0.361% / +0.240% | +4.008% / +1.009% | +3.682% / +0.687% | -0.274% / +0.204% |
| `[8, 128, 256, 4, 0, 1, 'float']` | yes | -10.884% / -7.907% | +3.152% / +1.038% | -9.437% / -6.413% | +2.438% / +0.315% |

No gate failure is evidence of a performance effect. Interpret the qualified
comparisons and residual differences separately; eliminating a compiled branch
alone does not establish an end-to-end benefit. #2469 remains draft until the
complete candidate's tradeoffs are resolved. This experiment does not authorize
adoption of its temporary factory selector or old module copies.

`runtime-fix.patch` records the private change; `instrumentation.patch` applies to
the fixed commit for reproduction. The temporary interface is removed after the
sweep. Sources, manifests, lock protocol, source/binary hashes, compiled Applied
assembly, full raw batches, failed controls and compressed build/test/Host logs
are archived. Local paths in the manifest require adjustment when reconstructing.
The measured binary remains `/tmp/nxrt-bank-static-bias/static-bias-probe`.

The streamed separate-FC3 case fails one process control gate, so its new ratios are not attributable performance evidence. Among qualified cases, static-vs-original-bank median wall/CPU changes are −7.05%/−3.71% for dense interleaved, +0.36%/+0.24% for streamed split, and −10.88%/−7.91% for Float separate FC3. The qualified streamed split case still costs +3.68% wall/+0.69% CPU relative to the old allocation driver. The static fix removes a proven branch and helps some paths here, but does **not** resolve the streamed candidate acceptance problem. Retain draft status and do not auto-merge #2469.
