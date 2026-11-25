# CPU routing scratch consolidation

MoE, QMoE and BlockQuantizedMoE share one dispatch-local `RoutingScratch` with
reusable index, exponential and selected-pair vectors. Each row replaces all
live entries and returns a borrowed mutable selection. The driver consumes that
selection before routing the next row. BlockQuantizedMoE retains its additional
expert-order sort; the next call rebuilds score order from fresh indices.
The scratch is dropped before expert weight admission/computation or prefetch.
It is not retained in the kernel, exposed as a new public EP API, or charged by
per-row/per-token quota requests.

QMoE no longer retains a separate vector for every row's route. Those copies
were used after expert execution only for their length. The routing contract
returns exactly `min(k, experts)` pairs, with no value-dependent filtering, and
the existing `k <= experts` validation remains. Each row therefore has exactly
k consecutive contribution slots. The final row/slot loops preserve the original
score-order additions, including zero and IEEE special values. Route-first
weight admission, prefetch ordering, expert tasks and contribution ownership are
unchanged. There is no second per-row count authority.

## Numerical and failure invariants

Keep the original complete sort with `total_cmp` and index tie-break, truncation,
max reduction, exp evaluation per expert, full and selected sum order,
aggregation zero-denominator handling and normalization. No top-k approximation,
new nonfinite cleaning, arithmetic reassociation or shape/format guard removal
is included. Old callers are updated; the former Vec-returning production helper
is removed. A `cfg(test)` frozen copy named `reference_routing_weights` supplies
independent expected values; it matches the old main body except the name.

New tests compare expert indices and every f32 bit with that independent
reference across empty/growing/shrinking geometry, k=0/1/half/all/oversized,
aggregation transitions, normalized/unnormalized weights, ties, signed zeros,
positive/negative NaNs and infinities. Explicit hand-written expectations also
pin tie order and a selected aggregation sum of zero. Every borrowed result is
mutated to expert order and the next call must reconstruct the original score
order. Existing three-kernel numerical, IEEE, streamed/dense, workspace,
route-first/prefetch, cache and failure tests remain in the validation scope.

## What the allocation probe observes

A temporary drop-free TLS allocator counter is appended to the actual test
library, forwarding all pointer/layout operations unchanged to System. A real
Box allocation/free positive control verifies observation. The frozen old
algorithm and new scratch run in the same executable, with the same finite
logits and optional aggregation inputs. The new scratch is primed outside each
observed loop; output values and expert order are checked by bit equality on
all observed calls. Each result is then expert-sorted to exercise reuse after
BlockQuantizedMoE's mutation. The old returned Vec is dropped inside its loop;
the new borrowed storage survives until the end of routing.

This observes 2048 repeated rows per case for expert counts 1,7,65,257, k=1/all,
both normalization modes and both aggregation modes (28 unique cases). It is
an allocation/free-call observation, not elapsed time, whole-kernel allocation,
inference throughput or a CUDA result. Cold scratch creation, capacity growth,
inputs, grouping/task storage, output allocation and final scratch destruction
are outside the observed warm row loop. Initial one-row dispatches need not
benefit. Routing metadata remains outside the governed projection lane; this
change does not claim full routing-memory governance or zero allocations per
dispatch. All temporary instrumentation is restored byte-for-byte before merge.

## Recorded results and validation

On macOS arm64/Rust 1.98.0, the native unoptimized same-image probe observed:

| Routing mode | Cases | Old allocation/free calls per row | New calls after priming |
| --- | --- | --- | --- |
| Softmax | 14 | 3 / 3 | 0 / 0 |
| Explicit aggregation | 14 | 2 / 2 | 0 / 0 |

Each case contains 2048 rows; every output pair matches the independent reference
by index and f32 bits. The positive control observed one allocation and one
free. This was one observation attempt, not a timing sample selected by retries.
Complete logs, test executable hash and exact source snapshots are retained.

The final sources passed native CPU2071 tests (21 ignored), MLAS research-reference
CPU2113 (37 ignored), host block oracles2 and QMoE oracle1, and strict CPU
all-target Clippy in both feature configurations. The test list confirms both
new routing cases were selected. The initial compile log records a mutable-slice
iteration type mismatch, corrected to explicit immutable iteration before the
successful validation. No runtime or numerical guard was relaxed. Read-only
design review and self-review found no resource/prefetch ownership blocker.

Run `python3 -B docs/benchmarks/cpu-routing-scratch/verify.py .` from the repo root
to verify sealed logs, the complete paired case set, reference-body identity,
source restoration and final validation. Reproduction drivers retain original
local paths and need path adaptation; builds and observations used the shared
host flock. No GPU execution was available or attempted.
