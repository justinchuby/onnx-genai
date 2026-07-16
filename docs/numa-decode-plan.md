# NUMA-aware native decode plan

## Current safe increment

`ONNX_GENAI_CPU_DECODE_THREADS` bounds only M=1 CPU `MatMulNBits` work in a
dedicated Rayon pool. It is opt-in, does not change M>1 prefill, and leaves
placement to `numactl`, `taskset`, or the process supervisor.

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
