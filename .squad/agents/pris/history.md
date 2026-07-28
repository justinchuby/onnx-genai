# pris — History

## Summary through 2026-07-28T21:15:00+0000
- Earlier detailed history was compacted by Scribe after exceeding the 15KB history gate. Durable team-wide decisions remain in `.squad/decisions.md` and `.squad/decisions-archive/`.
- Pris's sustained domains: test infrastructure, fixture quality, coverage hardening, metadata/schema validation, CPU/CUDA dispatch correctness, and reviewer-driven fix cycles.
- Delivered/maintained tiny-LLM and Mobius fixtures; KV/session and CPU-EP op coverage; macOS benchmark harness guards; CLI ORT CI; and dispatch reachability auditing.

## Retained durable outcomes
- **Dispatch correctness (2026-07-27):** Added reachability tests and `_TEST_HITS` counters for matmul/SDPA dispatch, with guard-break proof. Established the durable expectation that every new dispatch branch ships with a reachability test; this later became linted in CI.
- **CI hardening (2026-07-27–28):** Replaced personal setup actions with direct `rustup` and GitHub cache actions; adopted architecture-aware cache keys and workflow concurrency cancellation. CLI ORT coverage became a dedicated lane, and Windows path-safety remains a targeted fast-coverage exception.
- **CI shape (2026-07-28):** Full coverage remains required on PRs. A parallel uninstrumented Linux fast job offers early feedback but never substitutes for the full gate. Windows ARM64 retains tests/clippy but not llvm-cov reporting due to rust-lang/rust#150123; coverage signal remains supplied by Linux, Windows x86_64, and macOS.
- **Benchmark discipline (2026-07-28):** Per-PR benchmarks compare merge-base first and PR second, remain informational, and flag ≥15% / ≥30% deltas. Real regression gates remain profile throughput floors plus dispatch-reachability tests on representative hardware.
- **CLI test contracts:** Fixed `--decode-skip 0` accounting and stdout/stderr JSON routing; hardened REPL e2e tests against nondeterministic stderr and preserved stdout-specific behavioral assertions.
- **Scope update (2026-07-28):** #54 model-package and #299 LoRA are owned by another squad and remain out of scope.
