# Decision: harden ort-sys ORT download against transient network flakes

**Author:** Isidore (Mobile & Bindings Engineer)
**Date:** 2026-08-12
**Branch:** squad/ep-cpu-win-arm64 (PR #829)

## Context
Build-only CI run 31623032180 VALIDATED the new Windows ARM64 wheel config: the
`windows-11-arm` runner provisioned cp311 CPython (pythonarm64.3.11.9), the
rustup aarch64 toolchain, and cargo, reaching the exact same build step as
Windows AMD64. Both Windows jobs then failed on a **transient** network flake
(unrelated to the packaging change): `onnx-genai-ort-sys` `build.rs` downloads
prebuilt ONNX Runtime from GitHub releases via `curl`, which returned exit 52
(`CURLE_GOT_NOTHING`, "Empty reply from server"). Linux/CUDA/macOS passed.

## Root cause
In `crates/onnx-genai-ort/ort-sys/build.rs`, `download_prebuilt()` used
`--retry 3` but NOT `--retry-all-errors`. curl exit 52 is not in curl's default
retryable set — plain `--retry` only retries HTTP 5xx/408/429, timeouts, and
connection-refused. Empty-reply / connection-reset / TLS-handshake flakes are
skipped, so a single blip fails the whole job.

## Change
`crates/onnx-genai-ort/ort-sys/build.rs`, curl args in `download_prebuilt`:
- `--retry 3` → `--retry 5`
- added `--retry-all-errors` (curl >=7.71.0, present on all GitHub runners) so
  empty-reply / connection-reset / TLS flakes are retried too
- added `--connect-timeout 30` and `--max-time 300` so a hung connection can't
  consume the entire job before a retry fires
- `-sSL`, `-w %{http_code}`, and `-o <path>` kept intact and in order

Verified with `cargo check -p onnx-genai-ort-sys` (build scripts are compiled)
and `cargo fmt --all`.

## Risk
Low. Purely additive curl flags; no behavior change on the happy path. `--max-time
300` (5 min) comfortably exceeds a normal ORT archive download; if a legitimately
slow link ever exceeds it, curl will still retry up to 5 times.
