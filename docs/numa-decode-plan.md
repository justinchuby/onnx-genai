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

Not yet implemented (steps 4--5, deferred): sharding a single projection across
node-local replicas on both sockets. A naive dual-node pool regresses badly --
a 64-thread pool spanning both sockets with interleaved memory measured
11.1 tok/s vs 16.3 for single-node `compact` -- because every per-op fork-join
barrier then pays a cross-socket cache-coherency round trip. Reaching both
sockets' bandwidth requires eliminating that cross-socket barrier (per-node
sub-pools joined by a two-level barrier), which is the remaining lever.

