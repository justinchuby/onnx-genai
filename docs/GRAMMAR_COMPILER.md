# Maintained grammar compiler

The GenAI engine uses `onnx-genai-llguidance`, a workspace-owned publishable
package whose canonical Rust name is `onnx_genai_llguidance`. Its only compiler
source tree remains in `vendor/llguidance`; there is no parallel implementation,
registry/path feature switch or import alias. Package versions track the
workspace and engine dependencies are exact-pinned like other owned crates.

The code derives from the published MIT-licensed llguidance 1.9.1 archive.
[Source provenance and patches](../vendor/llguidance/NXRT-PATCHES.md) describe the
retained archive checksum, immutable original metadata, original source hashes,
exact compiler patch footprint and owned manifest/test hashes. The original
Rust 2021 edition, features and runtime dependency requirements are preserved.

Two compiler behaviors are required: escaped known-object keys keep their schema
ownership, and backend-proven unsatisfiability has a typed public error distinct
from invalid/unsupported schema failures. At the 2026-11-25 audit, the
[crates.io release metadata](https://crates.io/api/v1/crates/llguidance) identified
1.9.1 as latest, and its published API did not expose that typed distinction.
Using an unchanged upstream registry dependency therefore cannot build the
engine's current typed classification API or supply its key-escape fixes.

The maintained package is a normal dependency in the release graph. The publish
workflow uploads it before the engine. No vendored-source exemption treats code
patches as if they already existed in an upstream registry release. Python source
distributions follow the same workspace dependency/source tree. Actual release
publication remains a separate authorized operation; configuring and testing the
release graph does not itself publish a crate.

## Checks and registry-only artifact proof

Run all Cargo work under the host's shared workload lock. No performance claim is
made by these checks. Use the local build configuration and external target
folder appropriate to the host; packaging and compilation can consume substantial
resources.

```sh
python3 scripts/check_llguidance_vendor.py
python3 scripts/test_check_llguidance_vendor.py
python3 scripts/test_check_publish_order.py
python3 scripts/check_publish_order.py
cargo test -p onnx-genai-llguidance --lib
cargo test -p onnx-genai-llguidance --test registry_surface
cargo test -p onnx-genai-engine --lib json_schema_satisfiability
```

After reviewing and committing a clean source state, create the exact crate
without uploading it:

```sh
CARGO_TARGET_DIR=/tmp/nxrt-grammar-package cargo package \
  -p onnx-genai-llguidance --locked --no-verify
python3 scripts/verify_llguidance_registry.py \
  --archive /tmp/nxrt-grammar-package/package/onnx-genai-llguidance-0.1.0-dev.6.crate \
  --output-dir /tmp/nxrt-grammar-registry-proof \
  --cargo-config /path/to/local-cargo-config.toml
```

Use the current workspace version in the archive filename. The output directory
must be new and outside the repository. This script:

1. Safely extracts the archive and requires its provenance to equal the reviewed
   canonical source record. All preserved upstream/code/owned hashes are checked,
   including Cargo's source and normalized manifests and feature semantics.
2. Serves that exact checksum-bound `.crate` through an ephemeral localhost sparse
   registry. Index dependencies/features come from the normalized package manifest.
3. Creates a fresh external consumer with a versioned registry dependency, with
   no path dependency or patch override. Initial lockfile/metadata resolution may
   read crates.io and fetch registry dependencies; it does not publish anything.
4. Inspects Cargo's resolved graph to require this exact maintained package from
   the local registry, reject path/git/unpatched compiler substitution, then runs
   both public API/key-ownership regressions with `--locked --offline`.
5. Requires exactly two passed, zero failed/ignored tests and retains commands,
   logs, the resolved lockfile, package checksum and effective Cargo-input context.
   Timeout/interruption uses the same owned process-group containment as the
   behavioral conformance runner; no detached/remote-process guarantee is made.

Follow this with the engine/server key-escape, grammar masking and Native wire
regressions. A passing registry consumer proves the packaged compiler's required
surface, not general model/backend compatibility or completed publication.

Archived upstream integration sources are not passing acceptance results: their
unpublished helper dependency was omitted from the original published manifest.
The maintained manifest selects source unit tests and its own public
`registry_surface` integration. Nothing relabels upstream test source as a current
run of this fork.

The registry-surface ownership regression uses a legal raw DEL object key to
exercise the actual patched known/additional lexemes. Arbitrary Unicode escape
spellings are normalized by the engine before compiler recognition; this test
does not claim that the standalone compiler directly admits all such spellings.
