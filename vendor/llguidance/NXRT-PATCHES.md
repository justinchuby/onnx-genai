# llguidance source provenance and local patches

This directory contains the published llguidance 1.9.1 crate, licensed under
[MIT](LICENSE). `UPSTREAM.json` records the archive SHA-256 from Cargo.lock and
hashes of every original file. The archive is the authority for this snapshot;
the publisher's `.cargo_vcs_info.json` records a dirty source tree.

The engine uses this explicit versioned path dependency as its canonical compiler.
Python source distributions therefore include and resolve the audited source. `src/json/compiler.rs` changes object-key lexemes to
recognize legal JSON quoting consistently and reuses exactly those lexemes for
known-name exclusion from pattern/additional properties. Existing raw key spelling
is retained. No schema rewriting or model-specific normalization is introduced.

`src/json/compiler.rs` also makes its existing `UnsatisfiableSchemaError` type
public, and `src/lib.rs` reexports it. Fields remain private, and compiler error
construction and branch pruning are unchanged. The engine can therefore classify
only a backend-proven empty schema language without matching error strings or
silently treating malformed, unsupported or resource-limited schemas as empty.

Remove these local patches after an upstream release provides the same behavior
and typed distinction, and the repository's key-escape and schema-satisfiability
regressions pass against it.

`python3 scripts/check_llguidance_vendor.py` checks the original file hashes,
allowed patch footprint, and exact audited patch hashes without network access.

Repository builds and bundled source distributions use this path dependency.
Crates.io publication removes dependency paths; upstream or publish the compiler
patches before publishing a registry-only engine that requires the typed export.
Release-order validation recognizes only explicit vendored provenance with an unchanged
published manifest identity/version; it still rejects missing workspace crates.

`Cargo.toml.orig` is an explicitly recorded omission: it is a generated registry
manifest copy whose reserved filename prevents Cargo source packaging.
