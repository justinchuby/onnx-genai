# Decision: cfg(test) does not reach integration tests — use feature gates

**Date:** 2026-08-12
**Author:** Deckard
**Status:** Applied (PR pending)

## Context

Rust integration tests (`tests/*.rs`) are compiled as separate crates that link
the library as a regular dependency. The `cfg(test)` attribute is only set on
the test binary itself, **not** on the library it imports. Therefore a module
gated with `#[cfg(any(test, feature = "gpu-tests"))]` in `lib.rs` is invisible
to integration tests unless `gpu-tests` is explicitly enabled.

Cargo self dev-dependencies (`foo = { path = ".", features = ["gpu-tests"] }`)
are silently ignored for feature resolution, so they cannot be relied upon to
auto-enable features for integration tests.

## Decision

- Gate test-only helper modules purely on a feature flag (`#[cfg(feature = "gpu-tests")]`),
  never on `cfg(test)` alone.
- Integration tests that import such modules must `#[cfg(feature = "gpu-tests")]`
  the import and conditionally compile any body that references the gated types.
- The test function signature remains unconditional so the test binary and its
  inventory entry exist in both feature configurations (required by the honesty
  checker).

## Consequences

- No test-only code leaks to production builds.
- The honesty checker (`verify_cuda_test_honesty.py`) sees matching inventories
  in both configs.
- Future test helpers must follow this pattern; `cfg(test)` on a `pub mod` in
  lib.rs is only useful for *unit* tests inside the same crate.
