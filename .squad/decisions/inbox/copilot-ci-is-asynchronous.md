# CI Is Asynchronous

## Decision

Unless the user explicitly requests otherwise, agents and the coordinator do
not wait for CI before continuing, reporting, or merging.

Required local targeted tests, Clippy checks, builds, and hardware probes remain
blocking. CI runs asynchronously; failures found later are fixed forward.

## Consequences

- Do not poll or hold a turn open for CI.
- Do not keep completed worktrees solely for CI.
- Report which local validation was run.
- Fix platform- or matrix-specific CI failures in follow-up commits or PRs.
- An explicit user instruction may make a particular CI result blocking.

## Source

User: "我看到了它在等ci。我觉得我们等ci这件事让很多事变慢了。现在规定所有人除非明确指令否则不要等ci。本地测试。ci看到有问题再修。可以fix forward"
