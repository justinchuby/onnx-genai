# AGENTS.md

Entry point for AI coding agents working on **nxrt / onnx-genai**.

## Start here: the rules are normative

**[`RULES.md`](RULES.md) binds every human contributor and every AI agent.** Read it before making changes, and treat a rule violation as review-blocking even when the code works.

The rules, in short — `RULES.md` is authoritative, this table is only a map:

| # | Rule |
| --- | --- |
| 1 | Error messages must say what failed, why, and how to fix it |
| 2 | Stay model-, vendor-, and EP-agnostic — no hardcoded architecture |
| 3 | Make pre-release changes cleanly — no compatibility shims for our own APIs |
| 4 | Do not rewrite what already works — measure before replacing |
| 5 | Prefer explicit, inspectable behavior over hidden heuristics |
| 6 | Use Rust types to enforce invariants |
| 7 | Use the canonical names |
| 8 | Ship stable-ABI Python wheels |
| 9 | Tests track behavior and APIs |
| 10 | Keep history linear and review independent |
| 11 | Run portably across hardware tiers |
| 12 | **Reduce entropy: derive the rule from first principles** |

Rule 12 is the one most often skipped under time pressure. When a new case does not fit an existing predicate, dispatch table, or allowlist, find the property that actually decides it and state that **once** — do not append another special case. If two code paths answer the same underlying question, collapse them. Gate on the operand, topology, or capability that determines correctness, not on op names or model identities. A rejecting rule should return *why*, not just `false`.

## Working agreements

These are enforced in review alongside `RULES.md`:

- **Never wait for CI.** Validate locally with the smallest command that covers the change, then fix forward.
- **Work in a git worktree**, never the main checkout.
- **Separate measured from inferred** in every claim, PR body, and issue. If an A/B difference is smaller than the spread within one arm, it is unmeasured — say so.
- **When a result surprises you, suspect the probe first.** Publishing a wrong conclusion costs more than re-measuring.
- **Correct published mistakes in the open.** Retract in a comment on the same issue or PR; do not silently edit.
- English for all code, comments, commits, PRs, and issues.

## Skills

Task-specific discipline lives in skills, loaded on demand:

| Skill | Use when |
| --- | --- |
| [`measurement-discipline`](.github/skills/measurement-discipline/SKILL.md) | Making any performance or memory claim |
| [`test-discipline`](.github/skills/test-discipline/SKILL.md) | Changing an API or public interface |
| [`git-workflow`](.github/skills/git-workflow/SKILL.md) | Branching, review, and merge flow |
| [`profiling`](.agents/skills/profiling/SKILL.md) | Profiling the native CUDA/CPU EP |

Squad coordination and routing are described in [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

## Orientation

| Document | Contents |
| --- | --- |
| [`README.md`](README.md) | What the project is, how to build and run it |
| [`docs/architecture/ORT2.md`](docs/architecture/ORT2.md) | Runtime architecture; most rules cite a section here |
| [`docs/status/PROGRESS.md`](docs/status/PROGRESS.md) | What is implemented, in progress, and planned |
| [`docs/benchmarks/windows-cuda-runbook.md`](docs/benchmarks/windows-cuda-runbook.md) | Measured Windows CUDA setup, failure modes, and metric traps |
