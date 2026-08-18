---
title: Documentation Guide
aliases:
  - Docs Reading Guide
  - Source Precedence
tags:
  - documentation
  - onboarding
  - evidence
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Documentation Guide

> [!summary] Question answered
> Which document should I trust, and how do I navigate overlapping design, status, research and benchmark notes?

## Source precedence

Use this order when sources disagree:

1. Current code plus a reproducible test or measurement.
2. A document explicitly named authoritative for the question.
3. Accepted design decisions and current implementation plans.
4. Status notes tied to a dated revision.
5. Research/investigation notes.
6. Wiki explanations.
7. Old issue descriptions and unverified AI conversations.

“Authoritative” means the project intends to maintain that document, not that it
can never be wrong. Contradicting measurements require fixing the document.

## Directory meanings

| Directory | Use it for |
|---|---|
| `docs/architecture` | Project/runtime structure and cross-cutting contracts |
| `docs/memory` | Memory evidence, design, VMM, KV and offload |
| `docs/execution` | EPs, kernels, graph capture and placement |
| `docs/genai` | Scheduling, pipelines, metadata and model packages |
| `docs/quantization` | Quantized formats, kernels and MoE |
| `docs/performance` | Performance methodology and focused investigations |
| `docs/benchmarks` | Dated measurement records with conditions |
| `docs/ep-plugin` | Plugin export ABI, gaps, security and conformance |
| `docs/distributed` | Communication, collectives and multi-device runtime |
| `docs/status` | Current progress and upstream inventories |
| `docs/research` | Exploratory work; useful but not automatically accepted |
| `wiki` | Learning paths, maps and plain-language explanations |

## Start by question

| Question | Read first |
|---|---|
| What is the product architecture? | [`docs/architecture/DESIGN.md`](../../docs/architecture/DESIGN.md) |
| What is the native runtime architecture? | [`docs/architecture/ORT2.md`](../../docs/architecture/ORT2.md) |
| What is implemented now? | [`docs/status/PROGRESS.md`](../../docs/status/PROGRESS.md) and code |
| What does memory do now? | [`docs/memory/MEMORY_ARCHITECTURE.md`](../../docs/memory/MEMORY_ARCHITECTURE.md) |
| What memory model is proposed? | [`docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md) |
| What is the weight-offload north star? | [`docs/memory/WEIGHT_OFFLOAD.md`](../../docs/memory/WEIGHT_OFFLOAD.md) |
| How is CUDA execution progressing? | [`docs/execution/CUDA_EP_STATUS.md`](../../docs/execution/CUDA_EP_STATUS.md) |
| What do measurements show? | [`docs/benchmarks/README.md`](../../docs/benchmarks/README.md) and a dated run |

## Reading dated evidence

A trustworthy performance note names:

- code revision;
- model and exact artifact;
- hardware/driver/platform;
- backend and EP;
- precision and quantization;
- batch, prompt/context and generation length;
- concurrency/contender conditions;
- warmup and repetition method;
- what changed and what stayed fixed.

> [!important] A number without conditions is not a result
> Do not copy a tok/s, latency or memory number into a decision without its
> measurement conditions.

## Wiki versus docs

Wiki notes should answer:

- “What does this term mean?”
- “How do these components connect?”
- “Where should I start?”
- “Which formal source owns the truth?”

Formal docs should answer:

- “What contract did we accept?”
- “What exactly was measured?”
- “What must an implementation satisfy?”
- “What is the current support/status matrix?”

If a wiki note accumulates normative requirements or benchmark evidence, move
that material into `docs/` and link to it.

## Related notes

- [[start/Repository Map]]
- [[meta/Using this Wiki]]
- [[development/Testing and Verification]]
