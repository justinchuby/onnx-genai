# Decision: Use Session_GetEpGraphAssignmentInfo for direct EP assignment assertions

**Author:** Resch  
**Date:** 2026-08-11  
**Status:** Implemented

## Context

The test suite relied on two mechanisms to prove node assignment:
1. `session.disable_cpu_ep_fallback=1` (indirect — prevents fallback, doesn't prove assignment)
2. Device lookup assertion (proves registration, not node-level assignment)

A false comment claimed ORT 1.27 had no per-node provider attribution API. This was incorrect — `Session_GetEpGraphAssignmentInfo` has existed since ORT 1.24.

## Decision

Use `Session_GetEpGraphAssignmentInfo` + `EpAssignedSubgraph_GetEpName` + `EpAssignedNode_GetOperatorType` to directly query which nodes are assigned to "cpu_ep" vs other EPs.

- Enable `session.record_ep_graph_assignment_info=1` in `conformance_setup` unconditionally.
- Assert specific ops are assigned to "cpu_ep" in 8 conformance tests.
- For `mixed_partition` (fallback enabled): assert our EP never gets unsupported ops.
- For `shape_f32`: ORT may constant-fold Shape before EP assignment; soft-check only.

## Trade-off: `unwrap_or(0)` → `DIM_UNKNOWN` sentinel

The `ExecutionProvider::get_kernel` trait takes `&[Vec<usize>]` — it cannot express optional dims. Rather than changing a public trait (which would break all EP implementations), we:
- Introduced a named `DIM_UNKNOWN = 0` constant with a loud invariant comment
- Documented that kernels MUST NOT pre-allocate from compile-time shapes
- The `shapes_opt` (with full `Option<usize>` fidelity) is passed separately to `ShapeInference::for_node`

The risk (future kernel sizing from sentinel 0) is mitigated by: (a) valid dims are ≥1, (b) runtime shapes come from OrtKernelContext, (c) the constant name makes the sentinel visible in code review.
