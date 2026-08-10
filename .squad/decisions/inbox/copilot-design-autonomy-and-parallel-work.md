# Design Autonomy and Parallel Work

## Decision

The coordinator may independently make architecture and design optimizations
when evidence supports them. Direction-changing decisions must update durable
design documentation with the measurement, falsifier, limitations, and
rollback or override path.

When work can be separated without overlapping files, shared mutable state, or
an unresolved common contract, prefer parallel agents in separate git
worktrees. Keep changes to the same core contract or continuous call chain
serial until the shared dependency lands.

## Rationale

Repeated approval for each design adjustment slows evidence-driven work, while
undocumented decisions quickly become stale or contradictory. Parallel
worktrees improve throughput for independent tasks, but overlapping work
creates rebases, duplicated investigation, and inconsistent contracts.

## Source

- User: "如果需要设计上的优化 你可以自行裁决 更新设计文档。"
- User: "还有如果可以最好并行多worktree推进"
