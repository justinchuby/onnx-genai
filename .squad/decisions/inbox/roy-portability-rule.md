# Decision: Make hardware-tier portability an explicit project rule (RULES.md §11)

**By:** Roy (Lead) · **Requested by:** Justin (@justinchuby) · **Date:** 2026-08-13
**Branch:** `squad/rules-portability` · **PR:** docs(rules): make hardware-tier portability an explicit project rule

## WHAT

Added **Rule 11 — "Run portably across hardware tiers"** to `RULES.md`, matching the
existing rule style (numbered `## N.`, bold one-sentence principle, tight bullet list,
`See …` link line). Governance doc only — **no kernel/code changes.**

The rule codifies four things:

1. **Runtime capability detection with graceful fallback** — detect CPU instruction sets
   (AVX-512/AVX2/NEON/SVE) and take the fast path when present, with a correct portable
   scalar/generic fallback; a missing fast ISA slows execution, it never fails to run.
   GPU kernels compile/JIT to the device actually present (no assumed SM/arch, tune from
   queried device properties).
2. **Hardware-tier awareness** — memory-bandwidth and VRAM tier shape the bottleneck;
   a feature needing more VRAM/bandwidth than the tier has must degrade or clearly opt
   out, not crash (30B int4 ~15 GB fits H200, not an 8–12 GB consumer GPU → fail clearly
   or offer a smaller-footprint path, never silently OOM).
3. **Perf claims are tier-scoped** — state the device/EP/tier a benchmark or "ceiling"
   was measured on; never generalize one device (e.g. H200) into a universal conclusion.
4. **No hard runtime dependency** on a specific vendor toolkit/driver/arch beyond the
   declared minimum; consistent with Rule 2 (agnostic) and Rule 5 (fail clearly).

## WHY

Portability was previously only *implied* by Rule 2 (model/vendor/EP-agnostic,
architecture-gated kernels) and Rule 5 (fail clearly). The substance lived only in docs
(consumer-GPU audit, benchmark notes) with no binding rule. Justin has directed this
repeatedly: "fix any errors that limit portability"; "CPU must use fast instructions when
available and gracefully fall back"; "different machine configs have different bottlenecks —
we must consider them all." Making it an explicit rule gives reviewers a citable gate.

The tier-scoped clause is grounded in measured evidence: the lower-bit-quant NO-GO is an
**H200** finding (decode is latency-bound on the serial node chain there, per PR #885) — it
is explicitly *not* a universal conclusion, and may still be a real win on a
bandwidth-starved / VRAM-limited consumer tier. Rule 11 requires that qualification rather
than letting a single-device measurement become a blanket ceiling.

## Trade-offs / scope

- Kept **terse and non-duplicative**: cross-references Rules 2/4/5 instead of repeating them.
- No new prescriptive engineering mandates beyond Justin's directives — the rule is a
  governance contract, not a design spec.
- Cited only docs verified to exist: `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`,
  `docs/CROSS_PLATFORM.md`, `docs/benchmarks/2026-07-25-gqa-decode-avx512.md`,
  `docs/research/lowbit-quant-feasibility.md`.
