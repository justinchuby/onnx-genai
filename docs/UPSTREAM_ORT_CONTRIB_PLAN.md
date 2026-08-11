# Upstream ORT Contribution Plan

> **Status:** QUEUED — planning only. No code, no forks, no upstream PRs.
>
> **Owner:** Roy (Lead) · **Requested by:** Justin Chu · **Date:** 2026-08-11

---

## 1. Entry Gate

This phase **does not begin** until ALL of the following are satisfied:

| Gate | Testable condition |
|------|--------------------|
| EP compatibility merged | PR #762 (`squad/ep-plugin-parity-cuda` → `main`) is **merged** or **approved with no outstanding blocking reviews** |
| Known open items resolved | The two currently-unimplemented items — (a) native nxrt dynamic ABI and (b) real CUDA EP with device-pointer/data-transfer — are either **closed** or **explicitly deferred with a tracking issue** |
| Teammate inventories landed | `docs/UPSTREAM_ORT_CPU_KERNEL_GAPS.md` (Resch), `docs/UPSTREAM_ORT_CUDA_KERNEL_GAPS.md` (Batty), and `docs/UPSTREAM_ORT_CONTRIB_METHODOLOGY.md` (Sebastian) exist on `main` |
| Justin gives explicit go | A comment, issue, or message from @justinchuby authorizing the start of this phase |

**"Stable" means:** someone can check these four boxes objectively — no vibes.

---

## 2. Separation from Current Work

| Concern | Rule |
|---------|------|
| Repository | Upstream contributions live in a **fork of `microsoft/onnxruntime`** (not this repo). |
| Branches | In the fork, use: `contrib/{issue-or-slug}` (e.g. `contrib/avx512-softmax-f16`). This repo's `squad/` prefix per `.github/skills/git-workflow/SKILL.md` does NOT apply to the upstream fork. |
| PRs | Each upstream PR is self-contained: one kernel or one small family. No coupling to PR #762. |
| Language | Upstream is **C++/CUDA**. This repo is Rust. The contribution lives entirely in C++ in the upstream fork; no Rust code is upstreamed. |
| Back-integration | If an upstream contribution is accepted, this repo benefits by upgrading its ORT dependency — no code duplication. |

---

## 3. Routing — Who Owns What

### Kernel Work (C++/CUDA)

| Candidate class | Primary owner | Notes |
|-----------------|---------------|-------|
| CPU kernels — Intel/AVX2/AVX-512/VNNI | **Resch** | See `docs/UPSTREAM_ORT_CPU_KERNEL_GAPS.md` |
| CPU kernels — Apple Silicon/NEON/AMX | **Iran** | aarch64-apple-darwin paths |
| CPU kernels — ARM/SVE/NEON (non-Apple) | **Luba** | Edge/Windows-on-ARM |
| CUDA kernels (attention, quant, decode) | **Deckard** (systems lead), **Leon** (buffers), **Batty** (engine) | See `docs/UPSTREAM_ORT_CUDA_KERNEL_GAPS.md` |

### Quality Gates

| Gate | Owner |
|------|-------|
| Numeric parity verification | **Chew** |
| Benchmark methodology & model-level validation | **Sebastian** (per `docs/UPSTREAM_ORT_CONTRIB_METHODOLOGY.md`) |
| "Was the feature actually on?" / reachability | **Challenger** |
| Security / FFI / unsafe review | **Holden** |
| Unit & integration tests | **Pris** |

### Language Skills Caveat

This repo is Rust; upstream is C++/CUDA. The kernel engineers (Resch, Iran, Luba, Deckard, Leon, Batty) are chartered for systems/engine work in Rust. Upstream contributions require **C++ fluency and familiarity with ORT's internal APIs** (`gsl::span`, `InlinedVector`, ORT kernel registration macros, CUDA EP provider interfaces). This implies:

- A different working mode: reading ORT source directly, matching ORT style (Google C++ style, 120-char lines, `clang-format`), using ORT's build system (CMake + `build.py`).
- Possible slower velocity per-PR than in our Rust codebase.
- Review will come from Microsoft's ORT team, not our internal reviewers.

---

## 4. Per-Contribution Workflow

```
┌─────────────────┐
│ Candidate Select│  ← from Resch/Batty gap inventories
└────────┬────────┘
         ▼
┌─────────────────┐
│ Feasibility Spike│  ← read ORT source for the existing impl, assess delta
└────────┬────────┘
         ▼
┌─────────────────┐
│ Port / Implement │  ← in fork of microsoft/onnxruntime, branch contrib/{slug}
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tests            │  ← reachability (Challenger) + numeric parity (Sebastian protocol)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Model-level Bench│  ← per Sebastian's methodology doc
└────────┬────────┘
         ▼
┌─────────────────┐
│ Internal Review  │  ← Chew (numerics), Sebastian (bench validity), Challenger ("on?")
└────────┬────────┘
         ▼
┌─────────────────┐
│ Upstream PR      │  ← in microsoft/onnxruntime, following their process
└────────┬────────┘
         ▼
┌─────────────────┐
│ Review Iteration │  ← respond to ORT team feedback; may require multiple rounds
└─────────────────┘
```

### Upstream Prerequisites (Verified)

The following are **verified** from `microsoft/onnxruntime` `CONTRIBUTING.md`, `docs/PR_Guidelines.md`, and `docs/Coding_Conventions_and_Standards.md` (accessed 2026-08-11):

| Requirement | Citation | Status |
|-------------|----------|--------|
| **CLA** — Microsoft Contributor License Agreement required; CLA-bot checks automatically | `CONTRIBUTING.md` § "Licensing guidelines" | ✅ Verified |
| **Feature request issue first** for non-trivial/API changes | `CONTRIBUTING.md` § "Contribute a code change" | ✅ Verified |
| **Unit tests mandatory** for all PRs | `docs/PR_Guidelines.md` rule 6 | ✅ Verified |
| **Google C++ style** (120 char max), `clang-format` | `docs/Coding_Conventions_and_Standards.md` | ✅ Verified |
| **Small PRs** (≤10 files ideal); separate cosmetic from functional | `docs/PR_Guidelines.md` rules 8, 9 | ✅ Verified |
| **Build locally** before PR; CI triggered by ORT team member | `docs/PR_Guidelines.md` rule 7; `CONTRIBUTING.md` | ✅ Verified |
| **Lintrunner** for linting (`lintrunner -a`) | `docs/Coding_Conventions_and_Standards.md` § "Linting" | ✅ Verified |
| **Performance PRs** must describe measurement methodology | `docs/PR_Guidelines.md` rule 2 | ✅ Verified |
| ORT containers: `InlinedVector`, `InlinedHashMap`, `TensorShapeVector`; no raw `absl::` | Coding standards | ✅ Verified |
| Pre-commit hook via `git config core.hooksPath .githooks` recommended | `CONTRIBUTING.md` § "Git hooks" | ✅ Verified |

### Unverified / To Confirm Before Starting

| Item | Action needed |
|------|---------------|
| Whether kernel-only perf PRs require a feature-request issue or can go straight to PR | Ask in ORT discussions or check past kernel PRs |
| Specific kernel test infrastructure (how to add a new kernel unit test, test data format) | Read `onnxruntime/test/providers/` before first PR |
| CUDA kernel CI availability (do they have GPU CI for contributors?) | Check PR CI logs on existing CUDA PRs |
| Whether ORT accepts optimizations for ops they already implement (improvement vs. replacement) | Gauge from existing accepted perf PRs |

---

## 5. Sequencing

### Phase A — Learning (Serial, 1 PR)

**Start with one small, high-confidence CPU kernel candidate** end-to-end:
- Purpose: learn upstream's review cadence, CI expectations, and style nits.
- Pick from Resch's inventory: ideally something with a clear benchmark win, small diff (≤5 files), no API change.
- This PR teaches us the real turnaround time and review friction before committing resources.

### Phase B — CPU Batch (Parallel after Phase A lands)

Once the first PR is merged:
- Resch, Iran, Luba can work **in parallel** on independent CPU kernel candidates.
- Each candidate is a separate branch and PR upstream.
- Chew/Sebastian/Challenger gate each before submission.

### Phase C — CUDA (Serial gating, then parallel)

CUDA contributions are **blocked until**:
1. We have validated CUDA on real hardware (no GPU/toolkit on current host).
2. Batty's `UPSTREAM_ORT_CUDA_KERNEL_GAPS.md` confirms which candidates are **kernel-level** (portable) vs **runtime-level** (not portable).
3. At least one CPU PR has landed, proving the process works.

After unblocking, Deckard/Leon/Batty can parallelize on independent CUDA kernels.

### Serial vs Parallel Summary

| Serial | Parallel |
|--------|----------|
| Phase A (learning PR) before anything else | Phase B CPU candidates (independent kernels) |
| CUDA hardware validation before Phase C | Phase C CUDA candidates (independent kernels) |
| Internal review gate before each upstream PR | Multiple upstream PRs in flight simultaneously |

---

## 6. Risks and Honest Caveats

| Risk | Mitigation |
|------|------------|
| **Upstream may decline** — ORT team can reject contributions for any reason (roadmap misalignment, maintenance burden, style) | Start small to learn. Accept "no" gracefully. |
| **Runtime-level vs kernel-level** — Batty is assessing this. Much of our perf advantage may come from runtime orchestration (scheduling, KV management, graph capture) that cannot be expressed as a kernel contribution. If so, the contribution surface is smaller than hoped. | Wait for Batty's analysis before committing to CUDA. |
| **Maintenance burden** — upstreamed code becomes Microsoft's to maintain, but they may expect the contributor to respond to bugs/regressions. | Scope contributions narrowly; don't upstream complex coupled systems. |
| **Licensing/provenance** — any code derived from third-party sources or vendored libs needs clean provenance. Our Rust code is original, but the C++ port must be verified. | Holden reviews provenance before each upstream PR. |
| **CUDA hardware gap** — no GPU on current host means we cannot validate CUDA kernels locally. | Defer CUDA phase until hardware is available; do not submit unvalidated CUDA code. |
| **Language gap** — team is primarily Rust-focused; C++ upstream work may be slower. | Explicit acknowledgment; possibly longer timelines. |
| **Review latency** — ORT is a large project with many contributors; review may take weeks. | Pipeline multiple PRs; don't block on single review. |

---

## 7. Decision Points for Justin

Before this phase starts, @justinchuby must decide:

| Decision | Options |
|----------|---------|
| **How many candidates?** | 1 pilot only? 3–5 CPU? Full inventory? |
| **CPU or CUDA priority?** | CPU-first is lower-risk (no hardware gap); CUDA has higher potential impact but more blockers. |
| **Who is allocated?** | Full-time on upstream, or part-time alongside ongoing onnx-genai work? |
| **Does a fork already exist?** | If not, who creates it and under which GitHub org/account? |
| **Acceptable "no" threshold?** | If the first 2 PRs are declined, do we stop? |
| **Timeline expectation?** | Is this a Q3 2026 goal or an open-ended best-effort? |

---

## References

- Teammate inventories (may not yet exist on this branch):
  - `docs/UPSTREAM_ORT_CPU_KERNEL_GAPS.md` (Resch)
  - `docs/UPSTREAM_ORT_CUDA_KERNEL_GAPS.md` (Batty)
  - `docs/UPSTREAM_ORT_CONTRIB_METHODOLOGY.md` (Sebastian)
- Upstream docs (verified 2026-08-11):
  - `microsoft/onnxruntime/CONTRIBUTING.md`
  - `microsoft/onnxruntime/docs/PR_Guidelines.md`
  - `microsoft/onnxruntime/docs/Coding_Conventions_and_Standards.md`
- This repo: PR #762, `.squad/decisions.md` § "Extension contract standing directive"
