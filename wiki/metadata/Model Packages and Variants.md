---
title: Model Packages and Variants
aliases:
  - Model Package
  - Package Variant Selection
tags:
  - metadata
  - model-package
  - deployment
status: proposed
created: 2026-08-17
updated: 2026-08-17
---

# Model Packages and Variants

> [!summary] Question answered
> Why is a model package more than a directory of ONNX files, and how should the runtime select hardware-specific variants safely?

> [!warning] Proposal status
> The formal model-package document is a proposal and explicitly does not imply
> full implementation. Verify current tooling before relying on a command or
> format feature.

## Deployment problem

A deployable GenAI model may include:

- one or more ONNX graphs;
- external weights;
- tokenizer and chat template;
- inference metadata or compatibility config;
- image/audio processors;
- adapters and speculative draft models;
- compiled EP contexts;
- multiple hardware/device variants.

Loose relative paths provide no common identity, integrity inventory or
deterministic explanation of which variant ran.

## Package goals

1. Portable, offline distribution.
2. Reproducible graph/weight/tokenizer/compiler identity.
3. Zero-copy-compatible external weights.
4. Hardware-specific variants under one logical model.
5. Compiled EP context reuse.
6. Inspectable validation and selection.
7. Explicit trust boundaries and path confinement.

## Conceptual layout

```text
package_root/
├── manifest.json
├── decoder/
│   ├── component.json
│   ├── cpu/
│   └── cuda/
└── shared_assets/
    └── sha256-<digest>/
        ├── tokenizer.json
        └── chat_template.jinja
```

The package is conceptually a directory with a manifest, components, variants
and content-addressed shared assets—not necessarily a compressed archive.

## Variant selection

A variant may declare:

- EP/provider identity;
- device class;
- compatibility string;
- executor-specific information;
- model/context/external-data paths.

Selection should:

1. filter by requested/available EP and device;
2. ask the provider to validate opaque compiled compatibility;
3. rank deterministically;
4. explain the chosen/rejected candidates;
5. never silently run a hash/compatibility mismatch.

## Trust and integrity

- Portable layouts remain confined to the package root.
- Installed layouts require explicit host trust.
- Symlinks and path traversal need deliberate policy.
- Shared assets use content-addressed identity.
- Hashes cover both file names and bytes.
- Loading should not fetch missing content implicitly.

## Relationship to inference metadata

The package answers **where artifacts are and which variant is compatible**.
Inference metadata answers **what the model means and which runtime capabilities
it requires**. A package may carry metadata, but it does not replace semantic
validation.

## Formal sources

- [`MODEL_PACKAGE.md`](../../docs/genai/MODEL_PACKAGE.md)
- [`MODEL_METADATA.md`](../../docs/genai/MODEL_METADATA.md)
- [`onnx-model-package`](../../crates/onnx-model-package/src/lib.rs)
- [[metadata/Metadata Driven Runtime]]
- [[architecture/Inference Request Lifecycle]]
- [[contracts/Runtime Contracts]]
