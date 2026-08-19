---
title: onnx-genai Knowledge Base
aliases:
  - Home
  - Wiki Home
tags:
  - wiki
  - home
status: maintained
created: 2026-08-18
updated: 2026-08-19
---

# onnx-genai Knowledge Base

Understand how the runtime is structured, how an inference request moves
through it, and which contracts keep execution, memory and plugins correct.

This wiki is written for people first. Each path starts with an explanatory
note; links to source and formal documents provide evidence without becoming
prerequisite homework.

## Choose a learning path

### Understand the repository

Start with [[start/Repository Map]], then follow:

1. [[architecture/Crate Architecture]]
2. [[architecture/Inference Request Lifecycle]]
3. [[execution/Execution Backends]]
4. [[execution/Execution Provider Contract]]

### Understand execution providers

- [[execution/CPU Execution Provider]]
- [[execution/CUDA Execution Provider]]
- [[execution/Plugin Execution Providers]]

### Understand memory

[[memory/Memory Management for Beginners]] explains allocation, virtual
backing, shared mappings, governors, holders and the provider's stream/context
responsibilities from first principles.

[[memory/MoE Router Skew and Always-On Experts]] explains how expert-selection
skew was measured for a trained-router MoE model, and why it decides whether a
residency policy has anything to exploit.

[[memory/Virtual Memory for KV Cache]] (中文) walks through what the virtual
memory manager actually does when a KV cache grows, when a prefix is shared,
when one physical handle is mapped at several virtual addresses, and when a
release or a teardown synchronization fails.

### Understand model prompt formats

[[prompting/Chat Templates]] (中文) explains how a message array becomes the token
sequence a model sees: template structure and conventions, why generation
starts at the position right after the rendered assistant prefix, multimodal
placeholder variants, recipient-scoped channels (`to=self` / `to=user` /
`to=<tool>`), and how tool calls are rendered and fed back.

[[performance/Chunked Prefill]] (中文) explains why a prompt can be prefilled in
pieces without recomputing the KV a chunk attends over, what capping the query
width buys, and why the leftover final chunk had to be padded onto a fixed
ladder before the kernel cache could stop recompiling the graph on every
request.

### Change or measure the runtime

- [[development/Testing and Verification]]
- [[performance/Performance Engineering Playbook]]
- [[performance/Chunked Prefill]]
- [[observability/Tracing and Profiling]]
- [[api/API Design Principles]]

### Understand contracts and models

- [[contracts/Runtime Contracts]]
- [[contracts/Formal Verification with TLA+]]
- [[metadata/Metadata Driven Runtime]]
- [[metadata/Model Packages and Variants]]

## Source of truth

> [!important] Explanations are not specifications
> Current code and reproducible measurements take precedence, followed by
> authoritative documents under `docs/`, accepted design decisions, and then
> these explanatory notes.

The wiki distinguishes shipped behavior from proposed design. If a note and
the implementation disagree, treat that as a documentation bug and verify the
current source.

## Reading in Obsidian

The complete `wiki/` directory is an Obsidian-compatible vault. Notes use
stable English paths, YAML properties, wikilinks and callouts while remaining
ordinary Markdown in GitHub and the published site.

See [[README|Wiki index and conventions]] for every note and
[[meta/Using this Wiki]] for authoring details.
