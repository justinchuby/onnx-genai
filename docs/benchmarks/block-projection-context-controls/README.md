# Execution-context follow-up: shared-code control and Ready-cache fix

This supplements, rather than replaces, the frozen
[first240-process candidate study](../block-projection-execution-context/README.md).
The block execution-context candidate remains experimental. Original noisy and
regressing results are unchanged. This follow-up makes two distinct observations:
shared-code controls weaken attribution of the singleton regression to plan state,
and source review found a separate, concrete ORT Ready-cache hot-path regression.
Neither observation proves a universally zero-overhead runtime.

## Shared executable diagnostic

The first study linked separate baseline/candidate programs. The sampled MXFP4
decoder and prepared projection have81 and273 instructions respectively, with
the same arithmetic/control instructions after accounting for relocations.
The64-byte E2M1 table payloads also match exactly, although table/code addresses
differ. These facts do not prove that placement caused the measured cost.

A diagnostic build temporarily adds the old constructor-frozen kernel types
alongside the execution-selected types in one CPU module. A factory-only selector
chooses the old variant; no selector is read on warm dispatch. Both variants call
the same format decoders, prepared projection and GEMM implementation. The image
has exactly one decode_mxfp4_block symbol. The temporary source was restored by
byte hash after compilation. **These diagnostic types and selector are not part
of production source or a compatibility API.** Instrumented source, exact builder,
build logs, executable hash and symbol receipt are retained.

90 terminal wait4 processes use one immutable executable:3 repetitions*3 cases*
(baseline/candidate and each arm's own null control for setup/work, plus double
work for baseline/candidate). Calls use4294967296 MACs,8x the first study's budget.
Orders rotate/reverse. CPU/wall qualify independently using each arm's A/A within
5%, setup<=25%, adjusted double[1.7,2.3], work>=50ms and CPU<=wall*11. One shared
Darwin flock covers the matrix; no per-core foreign monitor or idle-host claim.
Both variants use each one's own independently checked logical demand and actual
retained physical peak. Streamed caps, dense retained weights, final trace and
trim0 are checked. Every sample is retained and no cell was rerun to pass.

|Case|CPU candidate/baseline|Wall candidate/baseline|CPU /wall qualified|
|---|---:|---:|---|
|singleton MoE1→1, streamed separate gate|1.0005|1.0005|yes /yes|
|MatMul4→1, streamed|0.9927*|0.9927*|no /no|
|MatMul4→4, dense control|0.9963|0.9948|yes /yes|

*MatMul4→1 candidate A/A=0.9482 CPU,0.9471 wall fails the5% bar. Its ratios are
only descriptive: this case remains **inconclusive**, and the original8% cost
has not been causally explained. Singleton MoE's original4% cost does not recur
in the qualified shared-image control; this supports code/layout sensitivity as
an inference, not a proven root cause or a production speedup. A small cost is
still possible. All90 full outputs/argmax match across arms and match the
corresponding original240-process fixtures. Raw traces/logs/outputs are retained.
Reader controls reject11 malformed matrices and keep failed A/A unqualified for
both arms and both metrics. No CUDA or full-model throughput result is implied.

## Ready-cache hot path

The default execution-refinement hook already reuses a cached Ready requirement.
The custom context-plan refiners had discarded it and repeated metadata planning,
even though the caller had matched kernel, tensor metadata and provider context.
They now capture a fresh current-pool plan and reuse that validated requirement.
None/Deferred still perform checked planning; the old plan is cleared first,
and execution consumes the new plan and retains actual-view/layout validation.
No public API, owner, per-row quota or production counter is added.

Test-only thread-local observation proves0 metadata queries for Ready and1 for
None/Deferred on QMoE and both block kernels. The tests also cover correct output,
selected execution width, consumed-state rejection and failure clearing. Invalid
metadata failure tests now correctly use None: a Ready from different metadata
violates the existing cache-origin contract. No actual execution guard was removed.

Final candidate checks pass: CPU2068/21 ignored, MLAS2110/37, layouts15, Session376,
plugin cache19, CPU-only CUDA-package oracles2+1, strict all-target CPU/plugin/
Session Clippy native+MLAS. Sources and logs are retained separately. This fix
addresses **ORT cached Ready work**. Native currently supplies None, so it cannot
explain or offset the native matrix's regression. The shared-image executable was
built from the earlier candidate, not this cache fix, and is not timing evidence
for the new Ready path. A read-only resource-design review found no cache/state
contract blocker.

The manifest hashes all payloads. Run python3 verify.py here to validate the
complete90-process diagnostic, exact outputs and validation receipts. The optional
checkout argument checks only the recorded cache-fix source set. The first study
has its own immutable manifest and source set. Binary bytes remain outside git;
absolute historical runner paths are provenance, not a portable recipe.
