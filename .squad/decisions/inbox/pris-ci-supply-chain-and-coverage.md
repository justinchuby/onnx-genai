# CI supply-chain hardening and coverage policy

Date: 2026-07-27
Owner: Pris
Requested by: Justin Chu
Run: https://github.com/justinchuby/onnx-genai/actions/runs/30309111341
Commit verified: ca3c2a127f14d82f444c8f2fa586dc95a80a0328

## Policy recorded

- Be wary of personally-owned third-party GitHub Actions. Prefer commands we own directly when GitHub-hosted runners already provide the substrate.
- Every coverage-capable test lane should produce a coverage report and upload it to Codecov with a lane-specific flag. Upload failures remain non-blocking; test/coverage execution failures remain blocking.

## Replacements in scope

Changed `.github/workflows/ci.yml` and `.github/workflows/audit.yml` only.

- Replaced `dtolnay/rust-toolchain@stable` with direct `rustup toolchain install stable --profile minimal --component ...` plus `rustup default stable`.
  - Stable is deliberate: it follows Rust's stability promise without freezing security fixes. The cache key records the resolved `rustc` release.
- Replaced `Swatinem/rust-cache@v2` with `actions/cache@v4` over `~/.cargo/bin`, `~/.cargo/registry`, `~/.cargo/git`, and `target`.
  - `actions/cache` is GitHub-owned, which is acceptable under the new supply-chain rule.
- Replaced `taiki-e/install-action@cargo-llvm-cov` with direct `cargo install cargo-llvm-cov --version 0.8.7 --locked --force`, guarded by an installed-version check.
  - Recommendation implemented: install from crates.io instead of a personal install action. This costs build minutes on cache misses, but keeps the binary provenance in Cargo's registry/checksum path and avoids ad-hoc binary checksum plumbing in CI.

## Coverage flags added

- `offline` for the portable offline crate tests on Linux, Windows x86_64, and macOS.
- `mlas` for the Linux-only `mlas-sys` test.
- `cli-ort-linux` for CLI ORT Linux tests.
- `cli-ort-windows` for CLI ORT Windows tests.

`codecov.yml` now declares those flags with `carryforward: false`.

## Verification

Final CI run is green: https://github.com/justinchuby/onnx-genai/actions/runs/30309111341

- `a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer` executed in `CLI ORT coverage (Linux x86_64)` and passed.
- Codecov uploads were queued for `offline`, `mlas`, `cli-ort-linux`, and `cli-ort-windows`.
- Windows CLI ORT coverage explicitly stages `onnxruntime.dll` into both `target\\debug` and `target\\llvm-cov-target\\debug` paths and runs `cargo llvm-cov --no-clean` so coverage target layout does not lose the DLL.

## Measured time cost

Baseline: main CI run https://github.com/justinchuby/onnx-genai/actions/runs/30307881431
Final run: https://github.com/justinchuby/onnx-genai/actions/runs/30309111341

| Lane | Baseline | Final | Delta |
|---|---:|---:|---:|
| Rust quality | 2m12s | 3m39s | +1m27s |
| CUDA compile Linux | 0m42s | 1m31s | +0m49s |
| CUDA compile Windows | 2m31s | 1m40s | -0m51s |
| Rust Linux x86_64 | 3m32s | 5m26s | +1m54s |
| Rust Windows x86_64 | 5m21s | 8m50s | +3m29s |
| Rust macOS arm64 | 2m43s | 4m55s | +2m12s |
| Rust Windows ARM64 | 6m40s | 10m25s | +3m45s |
| CLI ORT Linux x86_64 | 1m22s | 6m36s | +5m14s |
| CLI ORT Windows x86_64 | 3m10s | 10m55s | +7m45s |

The largest added costs are the CLI ORT coverage lanes, especially Windows, where instrumentation plus ORT rebuild/staging is significant but still within a tolerable CI budget. The final critical path increased from about 6m40s to about 10m55s.

## Known exception

`aarch64-pc-windows-msvc` coverage is blocked by an upstream Rust/LLVM issue: `llvm-profdata` reports malformed `.profraw` files with `symbol name is empty` and cannot merge profiles (rust-lang/rust#150123). The Windows ARM64 runner still executes the same offline tests and clippy gate, but coverage upload is disabled there until upstream fixes coverage instrumentation.

## Remaining release workflow debt

Out of scope for this change, still present and should get a dedicated release-workflow review:

- `.github/workflows/publish.yml` uses `dtolnay/rust-toolchain@stable` at four call sites.
- `.github/workflows/wheels.yml` uses `dtolnay/rust-toolchain@stable` at two call sites.
