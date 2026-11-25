# QMoE route statistics from the existing task groups

QMoE used to increment a second `BTreeMap<expert, count>` while appending the same
assignments to `tasks`. Every count was exactly the corresponding task-vector
length. This second map was built even when the route-first metrics branch was
not used. Remove it and let route statistics borrow the existing task-group map
before tasks are consumed.

The map's key type preserves unique experts and the existing ascending order.
The method is generic only over assignment contents and reads vector lengths;
no assignment or count map is cloned. The statistics method retains its active
sum, unique count, atomic updates, saturated global/per-layer maps and lock order. It drops the totals
lock before taking the layer lock. The call remains after successful mapped
region reporting and before route-first weight admission/prefetch. Persistent
process/layer statistics remain; only the redundant temporary diagnostic index
is removed.

There is no allocator, pool, lock, quota API, grant/refund, task-order,
contribution-order, cache or prefetch-policy change. The prerelease internal
method signature is updated cleanly; no compatibility forwarding method is
retained. Nothing about grouping requires a new sort or copying expert payloads.

## Behavior checks

Two new metrics tests compare complete route snapshots against explicit expected
values, including empty batches, nonconsecutive experts, repeated experts over
multiple layers, zero-count members and saturated accumulated fields. The
best-effort Linux process memory snapshot is independent of route counters and
is copied into the expected snapshot rather than asserted as a stable value.

Existing QMoE tests cover cached/uncached numerical identity, exact IEEE route
order, route-first range reads, host capacity, independent reader grants,
pressure reclaim and serial/prefetch byte-and-stat equivalence. Nonempty test
listings are retained with the validation logs.

## Allocation observation boundary

The diagnostic appends a temporary allocator observer and an exact frozen copy
of main's old statistics method to the actual test library. Both paths run in
one native unoptimized executable. The old path rebuilds its temporary count
map in the same order as a deterministic sequence of selected expert IDs and
calls the frozen method; the new path borrows lengths from the existing task
groups. Persistent metrics maps are primed outside the counted loops. Both
paths perform all statistics updates inside the observed loop.

A real Box allocation/free positive control proves the TLS observer is armed.
There are 15 cases spanning 1,7,65 experts, 1,8,64 rows, and top-1/top-min(3,E),
with 2048 observations per case. The final complete statistics snapshots must
match (excluding unrelated dynamic Linux process counters), with explicit
expected layer execution, active, unique and routed totals. Group/task creation,
input events, metric priming/snapshotting and cold persistent-map allocations
are outside the counted loop.

This is a private diagnostic-index allocation count, not an elapsed-time,
whole-QMoE allocation or inference-throughput measurement. No CUDA execution
is involved. Temporary instrumentation is restored byte-for-byte and checked
against the final validated production hashes; it adds no shipped observer or
reference implementation. Drivers retain original local paths and require
adaptation before reproduction on another checkout.

## Recorded results and verification

On macOS arm64/Rust 1.98.0, all 15 paired cases passed with
identical route-statistics snapshots. Each contains 2048 calls. The original
index produced 2048 allocation/free calls per case for 1–10 distinct selected
experts, and 20480 for the 64/65-distinct-expert cases; the borrowed-count path
produced zero allocation/free calls in every counted loop. The positive control
observed one allocation and one free. There was one observation per API version. All original counts, fixtures,
source snapshots, executable hashes and logs are retained.

The exact final sources passed five metrics tests, native QMoE62 tests (one
existing ignored benchmark), MLAS research-reference QMoE62 (one ignored), the
host QMoE CPU oracle, and strict CPU all-target Clippy in both configurations.
Formatting and diff checks pass. Test lists explicitly select both new snapshot
cases and the existing prefetch/stat/bit-identity cases. Self-review and read-only
resource-design review found no ownership, lock-order or statistics blocker.
No CUDA execution or inference speedup is claimed.

Run `python3 -B docs/benchmarks/qmoe-route-counts/verify.py .` from the repo root to
verify the sealed logs, all 15 case identities, old main method body, restoration
and final validation hashes. These checks verify only this candidate's source
seal when a checkout path is supplied; archived evidence can be checked without
that argument after future source evolution. Heavy work used the shared host
flock and was coordinated with the peer chat.

The first prototype accepted an exact-size cloneable iterator, and passed its
behavior/allocation checks. Final review replaced that interface with a direct
borrow of `BTreeMap<usize, Vec<T>>`: unique expert identity remains guaranteed by
the type, instead of depending on an iterator contract, and the caller no longer
needs a mapping closure or Clone bound. The prototype observation is retained
in `iterator-prototype`; the reported final validation/source seal refers to
the direct-map interface.
