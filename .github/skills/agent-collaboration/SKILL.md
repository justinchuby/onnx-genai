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

### Worktree Cleanup — the coordinator's job, not yours

**Do not clean up build artifacts yourself.** Leave your worktree's `target/` in place when you finish; the coordinator sweeps stale ones on a schedule. An agent cannot see whether another worktree is mid-build, so a well-meaning cleanup can break a peer — and it is one more step for every agent to get wrong.

Context, so the sweep is not surprising: this repository's `target/` directories reach 30–45 GB each. With parallel agents in separate worktrees they have totalled ~245 GB and filled the disk, which corrupts session-event persistence (`I/O error: There is not enough space on the disk`) and can silently break other agents mid-build. If a build fails on disk space, **say so and stop** rather than deleting anything.

**Coordinator:** sweep at least every two days, and whenever free space drops below ~50 GB. Take `target/` only from worktrees whose agent has finished and whose branch is pushed:

```powershell
git -C <repo> worktree list          # find worktrees
Remove-Item <finished-worktree>\target -Recurse -Force -ErrorAction SilentlyContinue
```

Never sweep a worktree belonging to a running agent.

### CI Is Asynchronous
**Do not wait for CI unless the user explicitly asks you to.** Run the exact
local tests, Clippy checks, builds, and hardware probes needed to validate the
changed behavior, then report or merge based on that evidence.

CI is asynchronous feedback, not a blocking gate. If CI later exposes a
platform-specific or matrix-only problem, fix it forward in a follow-up commit
or PR. Do not spend an agent turn polling jobs, and do not keep a completed
worktree alive solely to wait for CI.
Local validation remains mandatory. "Do not wait for CI" never means "do not test."

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
