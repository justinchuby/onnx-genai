# Host workspace warm reuse mechanism

Conditions: Apple M1 Max, macOS arm64, Rust 1.98.0 / LLVM 22, release build, default native CPU implementation, no MLAS symbols. This probe does not execute a model or touch payload bytes. It measures one synchronous checkout/return cycle through the shared pool or the real CPU EP workspace bridge. Results are not inference throughput, physical RSS, CUDA, or memory-bandwidth measurements.

The candidate moves immutable holder/role attribution out of the settlement mutex and stores a context scope when the pool is constructed. Each checkout still enters a fresh context operation guard. Allocation/settlement, context retirement, capture pins, owner validation and quarantine contracts are preserved.

## Protocol

Baseline is main `c0d4fabd9c96070d06deb5624a93555d30bae6b2`; candidate source hashes and binary hashes are in `provenance.json`. The null arm is a byte-identical copy of the baseline executable. The same probe source is used for both builds. The driver restores the candidate sources in `finally` before measurements.

The entire build and sweep holds `/tmp/onnx-genai-hostlock-darwin.flock`, shared with the llama.cpp chat. Three independent processes per arm/case run in rotated arm order, with nine batches per process. Excluded baseline pilots choose iteration counts targeting 100 ms of process CPU time for both the measured loop and an independent integer recurrence control. No formal cells are retried. Each process records a starting process-load snapshot.

Gates: all measured and control CPU batches >=20 ms, measured CPU/wall ratios >=0.9, and median control CPU time within 5% of baseline for both candidate and null. Process snapshots and these checks reduce confounding; they do not establish exclusive physical-core scheduling on macOS.

Each run positively tests issuer counters with zero-byte claim/release calls outside timing, then asserts warm checkout/return has zero issuer calls (including zero bytes), backing identity/pointer reuse, exact retained capacity charge, and zero charge after trimming. No payload access is measured. Larger-backed/smaller-view reuse, foreign owners, retirement and failed release continue to be covered by the existing memory tests.

## Results

Numbers are medians of three process medians; brackets show the range of process medians. Units are nanoseconds per checkout/return cycle.

| Path | Capacity | Baseline ns | Candidate ns | Candidate change | Identical-baseline change | Gates |
|---|---:|---:|---:|---:|---:|---|
| pool | 64 | 45.04 [44.99, 45.09] | 38.25 [38.09, 38.44] | -15.07% | -0.05% | pass |
| pool | 768 | 45.83 [44.08, 46.01] | 38.51 [38.17, 39.12] | -15.97% | -1.84% | pass |
| pool | 1048576 | 45.07 [44.39, 45.24] | 38.28 [37.59, 38.28] | -15.07% | -0.87% | pass |
| ep | 64 | 53.50 [53.12, 53.88] | 46.60 [46.00, 46.63] | -12.89% | -0.65% | pass |
| ep | 768 | 53.13 [52.27, 53.20] | 46.58 [45.62, 46.65] | -12.31% | -0.04% | pass |
| ep | 1048576 | 53.23 [53.14, 53.25] | 46.60 [45.76, 46.87] | -12.44% | +0.52% | pass |

This bounds a small bookkeeping component only. It does not establish that the earlier QMoE routed-buffer regression is resolved, justify cross-run active-lane retention, or justify adding an execution-scope API. Reducing checkout frequency requires a separate end-to-end experiment. Preserve the independently performance-gated QMoE draft.

## Validation and reproduction

Existing memory API/governor/virtual-memory tests and doctests pass; strict all-target Clippy for those three crates passes. Session library: 296 passed. CPU QMoE module: 57 passed, one pre-existing ignored. Full logs are under `validation/`. No CUDA runtime validation was performed.

`probe.rs`, `driver.py`, `protocol.json`, `provenance.json`, `runs.json`, `summary.json`, raw process logs and gzip-compressed load snapshots are archived here. The standalone temporary Cargo package has path dependencies on memory-api, memory-governor, ep-api and ep-cpu plus libc 0.2. The driver records the temporary checkout and shared target used on this host; adjust those absolute paths when reproducing elsewhere. Recreate baseline/candidate source states from the recorded base and `candidate.patch`, hold the same host lock for the complete sweep, and retain failed gates rather than rerunning until favorable.
