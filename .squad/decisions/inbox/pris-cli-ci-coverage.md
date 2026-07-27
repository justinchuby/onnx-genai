# Pris decision — CLI ORT CI coverage

Date: 2026-07-27

## Constraint found

`onnx-genai-ort-sys` resolves ONNX Runtime in this order: `ORT_LIB_DIR`, `ORT_ROOT`, `pkg-config`, then an automatic GitHub release download. The automatic path downloads ONNX Runtime 1.27.0 with `curl`, verifies a pinned SHA-256 for Linux x64, macOS arm64, Windows x64, and Windows arm64, extracts it under Cargo `OUT_DIR`, and reuses it only when the cached header and runtime match API version 27. Bindgen needs libclang; Linux CI should install `clang libclang-dev`, while Windows can use the hosted LLVM install.

`publish.yml` already pays the ORT-linked build cost for `onnx-genai` and `onnx-genai-server` wheels. Those wheels deliberately do not bundle libonnxruntime; runtime loading comes from the Python `onnxruntime` package. Their build-time headers/import library still come from the same `ort-sys` auto-download. `wheels.yml` builds `nxrt` wheels and leaves `onnx-genai-server` wheels to `publish.yml`.

## Design chosen

Add an isolated `cli-ort` CI job, separate from the offline allowlist, with Linux x86_64 and Windows x86_64 matrix entries. It intentionally permits the pinned native ORT download only for `onnx-genai-cli`, then runs:

- `cargo build --locked -p onnx-genai-cli`
- `cargo test --locked -p onnx-genai-cli`
- `cargo clippy --locked -p onnx-genai-cli --all-targets -- -D warnings`

Linux is mandatory because `repl_e2e.rs` contains Unix-only REPL/interrupt/contract tests, including `a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer`. Windows is included because the auto-download path supports `win-x64` and it catches platform drift on Justin's main development OS.

## CI cost

Pending observation from the first pushed workflow run on `ci/cli-test-coverage`.

## Residual coverage gap

The lane covers CLI build, unit tests, and integration tests that can run against checked-in fixtures. It still does not cover paths requiring a real external model, GPU execution, or an actual interactive TTY. The ratatui live view is inert when `stdout` is not a terminal, so piped CI cannot exercise the live terminal rendering path.
