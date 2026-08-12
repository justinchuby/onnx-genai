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
`crates/onnx-genai-ort/ort-sys/build.rs`, `download_prebuilt`:
- **Dropped `--retry-all-errors`** — it is a curl >=7.71.0 flag, but the
  manylinux_2_28 (AlmaLinux 8) build container ships **curl 7.61.1**, which
  rejects the unknown flag (exit 2). This broke the Linux x86_64 CPU and CUDA
  wheel builds in CI run 31624588420 (Windows AMD64/ARM64 passed because those
  runners have modern curl). The flag is therefore NOT portable to the
  manylinux build environment.
- **Added a portable Rust-level retry loop** around the curl invocation
  (4 attempts). An attempt is a success only if `output.status.success()` AND
  the captured `http_code == "200"`. On a non-final failed attempt it logs a
  warning (attempt #, curl exit, http_code, truncated stderr) and backs off
  3s / 6s / 12s before retrying. This portably covers the transient failures
  plain `--retry` skips on ANY curl version (exit 52 empty-reply, exit 56
  connection-died — the ones that originally flaked the Windows jobs).
- **Kept the portable curl flags** curl 7.61 supports: `--retry 5`,
  `--retry-delay 2`, `--connect-timeout 30`, `--max-time 300`, plus the
  existing `-sSL`, `-w %{http_code}`, `-o`.
- **Preserved the existing panic messages EXACTLY** — the `if !output.status.success()`
  panic and the `if http_code != "200"` "upstream does not publish" guidance —
  so a genuine 404 / missing asset on the final attempt still yields the clear
  diagnostic. `verify_archive_checksum` / `verify_archive_magic` run only after
  a successful attempt.

Verified with `cargo check -p onnx-genai-ort-sys` (build scripts are compiled)
and `cargo fmt --all`.

## Portability note
`--retry-all-errors` (curl 7.71.0) is unusable here precisely because the oldest
build environment (manylinux curl 7.61) predates it. A Rust loop needs no curl
feature detection and behaves identically across Linux/Windows/macOS runners.

## History
- v1 (commit c6a5753b) used `--retry-all-errors`; reverted because it broke the
  manylinux Linux builds (curl 7.61 rejects the flag). Replaced with this
  portable loop.

## Risk
Low. Purely additive; happy path unchanged. Worst case a genuine outage runs
4 attempts (≈21s of backoff) before the same panic as before.
