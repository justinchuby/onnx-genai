# Decision: Collapse identical clippy branches in nxrt-host loader

**Date:** 2026-08-12
**Author:** Iran
**PR:** #762

## Context

Clippy `-D warnings` rejected identical `if`/`else if` blocks in `loader.rs:263`. Both branches returned `String::from("unknown")`.

## Decision

Merged the two conditions with `||`, keeping the `struct_size < name_end` check on the left so it short-circuits before any field access. Added a comment explaining the ordering requirement to prevent future regressions.

## Rationale

The struct_size guard is security-relevant: it prevents reading past the end of an older plugin's smaller struct. Collapsing with `||` preserves the guard while satisfying clippy.
