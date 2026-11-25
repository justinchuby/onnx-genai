# Initialize only live decoded-projection prefixes

BlockMatMul, BlockMoE and QMoE bind one admitted scratch owner per dispatch.
Previously the first streamed projection cleared its entire bound lane, even
when that lane was sized for a larger expert group or another FC projection.
The owner now tracks a valid initialized f32 prefix, and the canonical plan's
split method requests only columns-plus-decoded storage for the actual projection.
Only a newly requested suffix is zeroed. Subsequent smaller requests neither
re-clear old values nor touch the unused bound. A new dispatch starts at zero.

Prepared projection helpers accept the scratch owner instead of an eagerly
initialized full slice. Returned column/decoded slices borrow only that short
mutable owner borrow. Capacity and checked element arithmetic precede any write;
refusal preserves state and storage. Decoder failure can retain valid initialized
storage, without treating its contents as a successful result. Gather/zeroed-prefix
operations keep their previous separate semantics and do not advance this lazy
initialization proof. No compatibility method remains.

The declared maximum workspace, retained pool capacity, physical charge, quota
source, request/refund policy, logical shard bound, cache eligibility, schedules,
GEMM arithmetic and output publication are unchanged. Dense hits still never
initialize decoder-only scratch. There is no per-row/expert/tile admission or new
public EP API. This is less unnecessary initialization, not a smaller admitted
budget, measured RSS saving, latency claim or complete inference-memory cap.

## Checks and controls

- The state-transition test checks zero demand, small/large/small requests,
  preserved prior values, capacity refusal without writes/state change, untouched
  suffixes, and fresh-dispatch initialization over dirty reused backing.
- A real plan/project test uses actual group rows smaller than a 64-row bound,
  multiple K/N geometries including empty rows/zero K, exact outputs, a decoder
  error followed by successful reuse, and poisoned unused suffixes.
- The existing QMoE Host-pool resource test retains full-row routing and adds
  sparse routing whose largest expert group has two of three rows. It executes
  both construction/execution pool transitions, three preparation states and
  warm loans. The actual decoder suffix remains poisoned beyond the peak live
  projection. Workspace loans/returns still make zero warm issuer callbacks;
  positive transient weight requests still reach and are declined by the full
  authority. Pool identity, exact charges, output and trim-to-zero guards remain.
- All old geometry, invalid destination, numerical, ownership and cache guards
  remain. The standalone integration test still compiles the actual private
  implementation through #[path]. A cfg(test)-only binding helper ties test
  owners to real mutable backing; it is absent from production.

The negative control changes only the plan split to initialize all available
scratch before returning the requested slices. It must compile and execute the
exact QMoE resource test, then fail on the unused decoder suffix. A compiler
failure or zero tests does not count. Complete original/mutant source and logs
are retained; production source is restored byte-for-byte in finally.

Two intermediate migration diagnostics are retained: a missed raw test caller
and old field name, then the standalone integration's old method/path calls.
They were fixed without restoring the removed API. Final evidence is separately
pinned to the current five runtime/test source hashes.

No timing benchmark or CUDA execution was performed. Checks use the shared host
flock coordinated with the peer chat. Driver paths retain the original checkout
and ORT 1.32 cache locations and need deliberate adaptation for reproduction.
Run `python3 -B docs/benchmarks/decoded-scratch-prefix/verify.py .` to verify sealed
payloads and the current checkout; omit the checkout argument after later source
changes. The MLAS adapter reads the retained native driver at its original /tmp
path; the negative driver reads environment-source.py's prelude through its
original /tmp/nxrt-prefetch-groups-check.py path. queued.py holds the shared flock.

Final native and optional-MLAS configurations each pass projection8,
BlockMatMul39 (one existing ignored timing benchmark), BlockMoE31, QMoE68
(one ignored) and standalone layout17, with strict all-target CPU Clippy.
Native host QMoE oracle1 passes too. The exact negative-control QMoE resource
test compiles and fails with zero passed/one failed at the unused-suffix guard.
Both configurations' source hashes agree with the restored final sources.
