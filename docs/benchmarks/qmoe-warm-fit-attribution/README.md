# QMoE routed buffers on the simplified warm-fit path

Conditions: Apple M1 Max, macOS arm64, Rust 1.98.0 / LLVM 22, release native CPU, no MLAS symbols, no CUDA device. These are synthetic public-session strict prepare + run + output-owner-drop measurements, not model throughput.

## Question and source identity

Does the earlier single-row routed-buffer comparison reproduce when both arms use the current common runtime, including immutable holder/role attribution and the cached context-scope handle from #2454? The separate pool/EP mechanism study saved approximately 7 ns per warm checkout/return. That magnitude cannot explain the earlier H64 difference of about 1.7 us. This experiment changes the common runtime; it is not a causal isolation of either optimization.

Base: `388c818906b49580870baee0a96bedf2d1dfd139`. Only `crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs` changes between arms. The candidate source is byte-identical to draft #2451 at `02ce323cc83606b03b4a4fd276460ab855963b21`. The probe source is byte-identical to the original routed-buffer study. `provenance.json` records compiler and source/binary hashes; `candidate.patch` records the exact delta. The null executable is a byte-identical baseline copy. Source switching is restored in `finally` before the sweep.

## Protocol

The complete build/sweep holds the stable `/tmp/onnx-genai-hostlock-darwin.flock` shared with llama.cpp work. Three independent processes per arm/shape use three rotated arm orders and nine timed batches each (27 formal processes). Excluded baseline pilots target 100 ms process CPU per batch. The unchanged Relu control runs nine batches of 100,000 calls. No formal run is retried. Per-process starting load snapshots are archived as gzip files.

Qualification requires all measured and control batches >=20 ms CPU, candidate and null control medians within 5% of baseline, and measured CPU/wall ratio >=0.9. A failed cell remains reported but cannot support a performance claim. The macOS lock and snapshots do not prove exclusive physical-core scheduling.

## Results

CPU values are medians of three process medians in microseconds per prepare/run/drop; brackets give their range.

| Hidden | Baseline us | Candidate us | Candidate delta | Identical-baseline delta | Minimum CPU/wall | Control gate | Qualified |
|---:|---:|---:|---:|---:|---:|---|---|
| 32 | 28.539 [28.514, 28.592] | 28.717 [28.624, 28.759] | +0.623% | -0.430% | 0.698 | FAIL | NO |
| 64 | 85.883 [85.444, 85.969] | 85.918 [85.719, 86.142] | +0.041% | -0.403% | 0.996 | pass | yes |
| 256 | 1194.095 [1194.000, 1194.476] | 1195.738 [1194.560, 1195.833] | +0.138% | +1.645% | 0.995 | pass | yes |

H32 fails both the independent control gate (null/baseline 1.05618) and the CPU/wall guard (minimum 0.69787). Its timing is not qualified. H64 qualifies, with about +0.041% candidate change versus -0.403% identical-baseline change; the earlier +1.99% point estimate does not reproduce in this updated build. This does not prove the warm-fit changes fixed it. H256 qualifies but the identical baseline copy shifts +1.645%, much larger than the candidate +0.138%; neither is evidence of a gain.

All 27 complete output byte strings and argmaxes match within each shape. All outputs are finite. Funded retained workspace reconciles exactly: baseline 0 bytes; candidate 384/768/3072 bytes for H32/H64/H256. The probe checks stable retained usage across warm calls and zero retained derived-weight bytes in its zero-cache fixture. It does not count issuer calls; the separate warm-fit mechanism and existing resource tests establish their narrower contracts.

## Decision

Keep #2451 as a performance-gated draft. Retain the failed cell and the original study; do not call this a universally free accounting path or an end-to-end speedup. No new execution scope, per-kernel pool, quota envelope or compatibility API is justified.

The existing executor borrows/returns at run boundaries and gives kernels views. Source review found that execution-time workspace refinement and QMoE execution each compute the routed layout; their distinct validation responsibilities remain. If further attribution is needed, measure layout/metadata preparation separately from the kernel body before considering a shared resolved-layout protocol. The combined compute timer cannot assign their cost by itself.

This branch republishes the existing QMoE candidate only for the experiment; it is not an adoption or a new CUDA correctness claim. No new unit/Clippy matrix was run for this source-identical diagnostic. Prior candidate tests remain recorded on #2451; the current evidence added here is the real public-session output, capacity and timing sweep.

## Reproduction

`driver.py` and `probe.rs` are included. The driver creates a standalone temporary Cargo package with path dependencies on ep-api, ep-cpu, ir, loader, memory-api, memory-governor and session, plus libc and rayon. It uses the absolute temporary checkout and cached ORT 1.32.0 paths from this host; adjust them on another host, reconstruct the recorded baseline/candidate, and hold the host lock for the complete sweep. Raw process logs, batches, controls, output hashes, compiler/source/binary provenance and failed gates remain archived. The three current shape cells were specified before their pilots.
