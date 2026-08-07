---
name: "agent-collaboration"
description: "Standard collaboration patterns for all squad agents — worktree awareness, decisions, cross-agent communication"
domain: "team-workflow"
confidence: "high"
source: "extracted from charter boilerplate — identical content in 18+ agent charters"
---

## Context

Every agent on the team follows identical collaboration patterns for worktree awareness, decision recording, and cross-agent communication. These were previously duplicated in every charter's Collaboration section (~300 bytes × 18 agents = ~5.4KB of redundant context). Now centralized here.

The coordinator's spawn prompt already instructs agents to read decisions.md and their history.md. This skill adds the patterns for WRITING decisions and requesting help.

## Patterns

### Worktree Awareness
Use the `TEAM ROOT` path provided in your spawn prompt. All `.squad/` paths are relative to this root. If TEAM ROOT is not provided (rare), run `git rev-parse --show-toplevel` as fallback. Never assume CWD is the repo root.

### Worktree Cleanup
**Delete your worktree's `target/` directory when your work is done** — after the PR is open and validated, before you report back.

This repository's `target/` directories reach 30–45 GB each. With parallel agents in separate worktrees they have totalled ~245 GB and filled the disk, which corrupts session-event persistence (`I/O error: There is not enough space on the disk`) and can silently break other agents mid-build. Cleaning up is not optional politeness; it is what keeps concurrent agents working.

```powershell
Remove-Item <your-worktree>\target -Recurse -Force -ErrorAction SilentlyContinue
```

Do not delete another agent's worktree or `target/`. Do not delete your own before your validation commands have passed and your branch is pushed.

Note for disk-reduction attempts: on `x86_64-pc-windows-msvc`, Cargo's `split-debuginfo` is **silently ignored** — `packed`, `unpacked` and `off` all produce identical output, with no error or warning. The knob that works is `debug = "line-tables-only"` (~37% smaller than `debug = 2`, keeps backtrace line numbers).

### Decision Recording
After making a decision that affects other team members, write it to:
`.squad/decisions/inbox/{your-name}-{brief-slug}.md`

Format:
```
### {date}: {decision title}
**By:** {Your Name}
**What:** {the decision}
**Why:** {rationale}
```

### Cross-Agent Communication
If you need another team member's input, say so in your response. The coordinator will bring them in. Don't try to do work outside your domain.

### Reviewer Protocol
If you have reviewer authority and reject work: the original author is locked out from revising that artifact. A different agent must own the revision. State who should revise in your rejection response.

## Anti-Patterns
- Don't read all agent charters — you only need your own context + decisions.md
- Don't write directly to `.squad/decisions.md` — always use the inbox drop-box
- Don't modify other agents' history.md files — that's Scribe's job
- Don't assume CWD is the repo root — always use TEAM ROOT
