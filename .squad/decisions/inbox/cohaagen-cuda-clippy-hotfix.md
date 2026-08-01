# CI hotfix — make `DeviceOffloadPolicy` `Default` derivable (clippy `-D warnings`)

- Date: 2026-08-01
- By: Cohaagen
- Branch: `squad/fix-cuda-clippy-derivable-default`
- Lineage: unblocks origin/main "CUDA compile (Linux+Windows)"; regression introduced by #572

## Root cause

#572 flipped `DeviceOffloadPolicy::async_pagein`'s default to `false`. That made
the hand-written `impl Default` fully equal to a derivable default (all three
fields = their type default: `bool` → `false`, `Option` → `None`), so
`clippy::derivable_impls` fires under the CI's `-D warnings`, turning every PR's
CUDA-compile job RED (weight_paging.rs:117).

## Fix (behavior-identical)

Replaced the hand-written `impl Default for DeviceOffloadPolicy` with
`#[derive(Default)]` on the struct. Verified byte-identical: the three field
defaults are exactly `enabled: false`, `device_budget_bytes: None`,
`async_pagein: false` — which is precisely what the derive produces. Doc comments
on fields preserved. `from_env()` and all runtime behavior unchanged. Net diff:
1 insertion, 11 deletions (one file).

## Green confirmation

- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` → **exit 0**
  (was the failing gate; no other `derivable_impls`/`-D warnings` hits in the crate).
- CI first step `cargo check --locked -p onnx-runtime-ep-cuda -p onnx-runtime-python
  --features onnx-runtime-python/cuda` → **exit 0**.
- `cargo fmt --all --check` → clean.
- Guard test `weight_paging::tests::device_policy_defaults_to_disabled` (already
  present, :678) asserts `enabled=false`, `device_budget_bytes=None`,
  `async_pagein=false` → **passes**, locking the derived defaults against field
  default-drift. No new test needed.

## Verdict

Trivial, provably behavior-identical lint hotfix. Unblocks main (and #574).
