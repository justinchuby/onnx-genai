# Decision: env var verifier excludes filename references

**Date:** 2026-08-11
**Author:** Gaff (Code Reviewer / Quality)
**Status:** Implemented

## Context

`verify_documented_env_vars.py` uses a regex to find `ONNX_GENAI_*` / `NXRT_*`
names in `docs/` and ensures each is read in `crates/`. The pattern also matched
documentation filenames (e.g. `NXRT_ABI.md`) when they appeared as
cross-references. This is a false-positive class: any doc named with the prefix
trips the gate when linked from another doc.

## Decision

Added a negative lookahead — a match immediately followed by `.md`, `.rst`, or
`.toml` is treated as a filename, not an environment variable. A genuine env var
is never written with a file extension suffix.

## Why not `KNOWN_UNIMPLEMENTED`?

That list means "a documented env var deliberately not wired up." `NXRT_ABI` is
not an env var at all — listing it would be a false statement in a file whose
purpose is honesty.

## Rule

`ENV_PATTERN = re.compile(r"\b(?:ONNX_GENAI|NXRT)_[A-Z0-9_]+\b(?!\.(?:md|rst|toml)\b)")`
