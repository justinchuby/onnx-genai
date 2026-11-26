# Maintained grammar compiler and upstream provenance

This is the single source tree for the workspace-owned
`onnx-genai-llguidance` package (Rust name `onnx_genai_llguidance`). Its release
version follows the workspace, and its implementation derives from the published
llguidance 1.9.1 crate under the original [MIT license](LICENSE). It is an
explicit maintained fork, not an unchanged registry copy or an import alias.

`UPSTREAM.json` retains the original archive SHA-256 and every original file
hash. `UPSTREAM-Cargo.toml`, `UPSTREAM-Cargo.lock` and `UPSTREAM-VCS.json` preserve
published metadata byte-for-byte under names distinct from Cargo's generated
package metadata. The publisher's VCS record identifies a dirty upstream tree;
the published archive is the authority for the audited base. The archive's
`Cargo.toml.orig` remains an explicitly recorded omission because Cargo reserves
that filename for the maintained package's own source manifest.

The audited compiler changes:

- Object-key lexemes recognize legal JSON quoting consistently and reuse those
  lexemes for known-name exclusion from pattern/additional properties. Raw
  spelling is retained; schemas and model-specific text are not rewritten.
- The existing `UnsatisfiableSchemaError` is public and reexported, with private
  fields. Existing error construction/pruning is unchanged. The engine can
  distinguish backend-proven empty languages from malformed, unsupported or
  resource-limited schemas without matching error strings.

Two single-range token vectors use `std::iter::once(...).collect()` rather than
ambiguous single-range `vec!` syntax, preserving their exact values while passing
the current toolchain's strict Clippy gate.

The managed manifest preserves upstream Rust 2021 edition, minimum Rust version,
features, runtime dependency requirements and build/header behavior. The archive's
integration/benchmark sources depend on unpublished upstream test utilities; they
are retained as provenance, not selected as executable acceptance. Source unit
tests, the explicit `registry_surface` integration target and the engine/server
conformance suites test this maintained compiler directly.

`python3 scripts/check_llguidance_vendor.py` verifies source provenance, explicit
relocations, the exact code-patch footprint and owned source hashes. Its
`--root EXTRACTED_PACKAGE --packaged` mode additionally binds Cargo's generated
manifest to `Cargo.toml.orig`, checks the managed identity and preserves features.
No original compiler source is duplicated or selected by configuration fallbacks.

The release workflow publishes `onnx-genai-llguidance` before the engine. Engine
packages depend on this versioned package, so Cargo's normalization no longer
substitutes an unpatched `llguidance` registry package. Registry-only publication
is validated using the exact packaged archive and a temporary local registry;
see [verification instructions](https://github.com/justinchuby/onnx-genai/blob/main/docs/GRAMMAR_COMPILER.md). This does not
claim that a release has already been uploaded.

Remove the fork only when a published upstream version supplies all required
behaviors and the key-escape, typed-satisfiability and root-key encoding regressions pass against it;
update every caller and release edge together without compatibility shims.

### Third patch: root named-key generation policy

The XML generation followup adds the opt-in typed `RootObjectKeyPolicy`.
It generates only names explicitly declared in the normalized root object's
`properties` or `required`, with decoded UTF-8 bytes inside the configured
bounds. Values still use the original property/pattern/additional intersection;
dynamic root-name generation is disabled by this explicit generation policy.
A separate normalized-root reference cache leaves nested JSON objects, default
compilation, URI scopes and anchors unchanged. Named-field min/maxProperties
combinations use a bounded, iterative grammar construction, not subset
enumeration or a new JSON Schema solver. Backend-proven absence of values under
this policy uses the distinct typed `RootObjectKeyPolicyError`; callers must
not classify it as semantic `UnsatisfiableSchemaError` or silently prune tools.
It does not assert that the original unrestricted schema is unsatisfiable.
Original value schemas, including dynamic values and pending referenced
schemas, retain their invalid/unsupported/resource errors before any policy-only
capability error is returned. Default options serialize unchanged.
This is independent of the initial package-publication fork.

Before applying the policy, the original compiler fully compiles the same
normalized root and all pending definitions on the same builder. Original
empty-language, unsupported, invalid and resource errors retain their original
types. Only after that succeeds can restricted generation fail with the policy
error. This reuses normalized URI resolution and caches, does not re-fetch schema
references, and consumes the caller's actual builder/fuel limits. No second
byte-token parser is constructed. Unsupported original cardinality combinations
remain unsupported rather than being enabled by the policy.
