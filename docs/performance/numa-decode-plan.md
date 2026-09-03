# NUMA-aware native decode plan

## Current safe increment

`ONNX_GENAI_CPU_DECODE_THREADS` bounds only M=1 CPU `MatMulNBits` work in a
dedicated Rayon pool. It is opt-in, does not change M>1 prefill, and leaves
placement to `numactl`, `taskset`, or the process supervisor.

Invalid, empty, zero, and negative values behave as if the variable were
unset. Positive values are capped at the logical CPUs available to the process.

This avoids a hardware-specific default. On the measured 2x48-core host,
Qwen2.5-0.5B INT4 peaks at six node-local workers, while the 96-worker default
is about half as fast. A larger model may have a different optimum.

## Why automatic NUMA placement is deferred

A portable runtime policy needs to distinguish available process affinity
from machine topology and coordinate weight placement with worker placement.
Merely limiting workers does not guarantee that Linux keeps them on one node,
and merely pinning to one node does not prevent too many Rayon tasks.

## Proposed larger design

1. Discover CPUs allowed by process affinity and group them by NUMA node.
2. Create one decode pool per selected node with explicit worker affinity.
3. First-touch or replicate immutable packed weights on the node that consumes
   them; do not migrate the existing shared cache implicitly.
4. Dispatch a whole projection to one node, or shard only sufficiently large
   projections across node-local replicas.
5. Keep M>1 prefill on the global/full-machine pool unless separate benchmarks
   justify changing it.

Before enabling any automatic policy, benchmark small and large INT4 models,
M=1 decode and representative prefill sizes, one- and multi-socket machines,
and restricted container/cgroup affinity. The fallback must remain the current
global Rayon behavior when topology or affinity information is unavailable.

## Implemented increment: opt-in worker affinity

`ONNX_GENAI_CPU_DECODE_AFFINITY` pins the bounded M=1 decode pool workers to the
CPUs of a single NUMA node, realizing steps 1--3 of the design above without any
hardcoded socket or core counts. Topology is queried at runtime from
`/sys/devices/system/node/node*/cpulist`; the switch is inspectable and opt-in
(Rule 5), and a single-node or non-Linux host, or a cgroup that rejects the
pinning request, transparently falls back to the unpinned global behavior.

Modes:

- unset / `off` -- no pinning (default; leaves placement to `numactl`/`taskset`).
- `compact` -- pin the workers, one per CPU, to the smallest-index NUMA node
  whose CPU count covers `ONNX_GENAI_CPU_DECODE_THREADS`, so the per-op
  fork-join barrier and the first-touched packed int4 weights stay node-local.
- `node:<index>` -- pin to a named NUMA node; an unknown index is a clear error.
- `numa-split` -- explicitly build node-local sub-pools across all usable NUMA
  nodes.

For the persistent SPMD pool, explicit `off` performs one production
`allowed_cpus()` capacity-discovery read so the worker count still respects an
external `taskset`/cgroup restriction and leaves dispatcher headroom. It does
not invoke the per-worker affinity setter or read-back API, and does not replace
the creating thread's mask. Worker kernel affinity masks therefore remain the
inherited mask.

Because the packed int4 decode weights are lazily first-touched inside the
`with_decode_pool_scope` installation (on a pinned worker), they land on the
selected node, so both the barrier traffic and the weight stream are node-local.

Measured (Sapphire Rapids Xeon 8480C, 2x48 cores, 2 NUMA nodes,
Qwen2.5-Coder-7B int4, 32 decode threads, steady M=1, 5 runs x 3 rounds):

| Affinity | steady decode median | best | run-to-run spread |
| --- | --- | --- | --- |
| `off` | 13.1 tok/s | 14.4 | 12.6--14.4 (jittery) |
| `compact` | 16.3 tok/s | 16.4 | 16.3--16.4 (stable) |

`compact` is ~+25% on the median and, just as important, removes the OS-migration
jitter that makes the unpinned pool swing run to run. Greedy token ids are
bit-identical with and without pinning (it only changes placement, not math).

### Placement inside the chosen CPU set: `ONNX_GENAI_CPU_DECODE_PLACEMENT`

`ONNX_GENAI_CPU_DECODE_AFFINITY` selects *which* CPUs the decode pool may use.
`ONNX_GENAI_CPU_DECODE_PLACEMENT` selects how the workers are arranged inside
that set. The two are orthogonal and can be combined.

- unset / `compact` -- **shared-host default**. SMT siblings are filled before
  the next core is used, so an N-worker pool occupies about `N / siblings`
  physical cores. The automatic worker count is capped after reserving
  scheduling capacity for the inline dispatcher, so the active decode threads
  leave at least half the process's logical CPU capacity available to a
  co-tenant. On an exact 2-core/4-thread SMT mask, the sole worker and dispatcher
  are bound to the two siblings of one physical core for the decode scope, so
  the other physical core is wholly available to a co-tenant. If that topology
  or the required worker pin cannot be proved, the pool fails safe to one
  dispatcher-owned lane instead of retaining an unpinned resident worker. On an
  exact two-CPU mask the dispatcher likewise owns the sole compute lane and no
  resident worker is spawned. This also holds for a one-CPU-per-core `taskset`.
- `spread` -- explicit dedicated-host opt-in. Worker `i` takes a distinct
  physical core for as long as cores last, only then doubling up on SMT
  siblings; its automatic width may use every allowed physical core (subject to
  the existing dispatcher/service-core reserve).

Neither is universally better, which is why it is a selector and not a constant.
Measured with four DRAM-bandwidth hogs on a 16-core/32-thread host, `compact`
ran 4.54 ms/token against `spread`'s 5.03--6.26; with eight hogs covering both
halves the ranking inverts (`compact` 15.92 against `spread` 6.08). The full
table is in `crates/onnx-runtime-ep-cpu/src/core_topology.rs`'s module docs.

An unrecognized value is a configuration error. CPU EP initialization fails
with a diagnostic that names the variable, rejected value, and accepted modes;
it never silently selects either policy.

Because the ordering is only a ranking of a permitted set, both values are
permutations of the same CPU list: no placement policy can widen or shrink the
CPUs a cgroup or `taskset` allows.

Not yet implemented (steps 4--5, deferred): sharding a single projection across
node-local replicas on both sockets. A naive dual-node pool regresses badly --
a 64-thread pool spanning both sockets with interleaved memory measured
11.1 tok/s vs 16.3 for single-node `compact` -- because every per-op fork-join
barrier then pays a cross-socket cache-coherency round trip. Reaching both
sockets' bandwidth requires eliminating that cross-socket barrier (per-node
sub-pools joined by a two-level barrier), which is the remaining lever.

## Implemented increment: persistent SPMD decode pool (barrier fusion)

The persistent SPMD pool is the **default** CPU decode path (`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL`
unset or `=1` selects it deterministically; `=0` opts out to the flat Rayon
legacy path; `=auto` explicitly enables load-adaptive calibration). It replaces
the per-op rayon fork-join region -- there are ~141 `MatMulNBits` ops per decoded
token, each previously a separate parallel region whose *join barrier* dominates
once the int4 kernel itself is L3-resident -- with **one persistent worker set**
that stays hot for the whole decode loop and is driven by a lightweight reusable
barrier instead of re-forking rayon tasks. This targets the fork-join/barrier
bound directly rather than the kernel math (Rule 4: the actual GEMV still runs the
existing packed int4 / MLAS SQNBit kernels; only the orchestration changes).

Design (mirrors the `numa-split` two-level structure so it inherits its
node-local placement and exact reduction order):

- On first decode use, spawn the resolved worker budget. Shared-host `compact`
  fills SMT siblings first and includes dispatcher capacity in its automatic
  limit; explicit `spread` takes distinct physical cores first. Pin targets come
  from the same runtime topology probe as `ONNX_GENAI_CPU_DECODE_AFFINITY` (no
  hardcoded socket/core counts, Rule 2).
- Each op is broadcast by bumping a `sequence` counter the spinning workers watch;
  completion is tracked with **per-node** counters so the dispatcher reads mostly
  node-local cache lines and never pays a cross-socket coherency round trip on the
  hot path (this is exactly the cross-socket-barrier cost that sank the naive
  dual-node pool). Workers spin briefly, then park; the dispatcher only unparks
  workers actually parked, so the steady hot loop issues zero syscalls.
- Weights are first-touched by each pinned worker on its own row-shard, so the
  packed int4 stream is node-local, same as `compact`/`numa-split`.

Bit-parity: output rows are sharded, and each output row is an independent
full-K dot product, so any row partition is exactly associative -- greedy token
ids are byte-identical with the flag ON vs OFF. The regression test runs several
sequential real packed-int4 M=1 MatMulNBits kernels in fresh subprocesses, compares
every output byte, asserts that the persistent pool actually dispatched the ON
case, and uses 31 workers to cover uneven node/row sharding.

Generality / fallback (Rule 2, Rule 5): **default-on with opt-out**. Unset or
`=1` deterministically uses the pool, `=auto` opts into live calibration, and
`=0` or `THREADS=0` uses the flat path. A single-CPU cpuset also falls back
because it cannot host both a spinning worker and the dispatcher safely. On a
single-node host, a non-NUMA machine, or when NUMA topology is undiscoverable,
the pool collapses to a single worker group instead of the two-level node split;
it does not switch back to the bounded Rayon pool. That single group is **not**
unconditionally unpinned: `node_shards` passes the
process's allowed CPU set (`allowed_cpus()`, i.e. the cpuset/taskset mask) into
the fallback shard, and `SpmdDecodePools::build` **best-effort pins** every
worker to those CPUs on platforms that support pinning (Linux, Windows). Only
when the allowed set is unknown, or on a platform without pinning support
(macOS / unsupported OS), or when the OS refuses the request (a restricted
cgroup, logged once) do the workers run truly unpinned. The barrier primitive
itself is portable `std` atomics + `thread::park`; only the optional CPU pinning
is platform-specific and best-effort.

Default worker count (Rule 2, topology-derived): when
`ONNX_GENAI_CPU_DECODE_THREADS` is unset, shared-host `compact` uses at most
`floor((logical_capacity - 1) / 2)` workers (at least one): after adding the
inline dispatcher, at least half the logical capacity remains free. This is
*not* the flat pool's eight-worker ceiling. Explicit `spread` may instead use one
worker per allowed physical core. The flat pool caps at eight because its per-op
fork/join regresses beyond that; the persistent pool replaces that fork/join
with one hot broadcast barrier, so it keeps scaling with cores until the
memory-bandwidth knee. The compact cap leaves scheduler capacity free for
prefill's global pool and co-tenants; a *fully*-subscribed spinning pool starves
the dispatcher and collapses (measured below). An explicit
`ONNX_GENAI_CPU_DECODE_THREADS` is always
honored.

Decode strategy precedence is explicit (Rule 5): **explicit `numa-split` env >
persistent SPMD (default, unless an explicit non-`numa-split` affinity defers it)
> flat + auto-`compact`**. When persistent SPMD is active it does its own per-node
worker pinning, and the flat pool's auto-`compact` affinity is never built (the
two are mutually exclusive by construction, verified: the auto-`compact` log does
not appear in the default path). When both
`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split` and a *forced*
(`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1`) pool are in play, `numa-split` wins if
its two-level topology can be built, and the runtime reports that choice once; if
`numa-split` cannot build, the persistent SPMD pool remains eligible and that
fallback is also reported once.

Explicit-affinity defer (Rule 5, Rule 2): because the persistent pool is now the
Auto default, an explicit `ONNX_GENAI_CPU_DECODE_AFFINITY` request would otherwise
be silently ignored (the flat pool that honors it is never built). To preserve
that user control, when the mode is **Auto** (`PERSISTENT_POOL` unset) *and* the
user explicitly set `ONNX_GENAI_CPU_DECODE_AFFINITY` to a non-`numa-split` value
(`off`, `compact`, `node:<n>`, or a malformed value), the persistent default
**defers to the flat path** so `plan_decode_affinity` honors and validates the
request exactly as before (`off` = unpinned, `compact`, `node:<n>`, malformed
still errors). An explicit affinity request thus opts the user out of the SPMD
default, and the defer is logged once. `=1` (Forced) overrides this: the
persistent pool wins regardless of the affinity env and applies its own per-node
pinning. With no affinity env set at all (the true out-of-box case) the SPMD
default is used unchanged. The decision lives in one place,
`decode_spmd::auto_defers_to_flat` (consumed by `build_from_env`/`pools`).

Measured (Sapphire Rapids Xeon 8480C, 2x48 cores, 2 NUMA nodes,
Qwen2.5-Coder-7B int4, steady M=1, 128 tokens, 5-run median, interleaved A/B,
shared/noisy host):

Out-of-box default flip (nothing set):

| Path | steady decode median |
| --- | --- |
| before (flat + auto-`compact`, 8 workers) | ~11.1 tok/s |
| after (persistent SPMD default, 48 workers) | **~28.8 tok/s** |
| `PERSISTENT_POOL=0` (restores legacy) | ~11.1 tok/s |

Beats onnxruntime-genai 0.14.1 (21.30) and raw ORT (~26.9) out of the box.

Persistent-pool worker-count sweep (why half, and why not all cores):

| workers | tok/s |
| --- | --- |
| 8 (old flat default) | 12.95 |
| 16 | 19.78 |
| 24 | 23.86 |
| 32 | 26.18 |
| 40 | 27.90 |
| 48 (= half of 96, the default) | 28.71 |
| 64 | 28.67 |
| 96 (all logical CPUs) | **1.36** (dispatcher starved by spinning workers) |

Throughput plateaus at ~half the logical CPUs and *collapses* at full
subscription, which is why the shared-host default accounts for the dispatcher
inside a half-capacity active-thread budget and why an explicit override is
clamped to the host.
