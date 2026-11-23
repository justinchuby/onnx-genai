# Shared FC banks for BlockQuantizedMoE: candidate study

Status: unadopted candidate; numerical/resource validation passes, performance remains inconclusive.

Conditions: Apple M1 Max, macOS arm64, Rust 1.98.0 / LLVM 22, native release CPU with zero MLAS symbols. The workload is synthetic public InferenceSession strict prepare + run + output-owner drop, not model throughput.

## Scope and ownership

Float and block expert execution share one private phase driver. Dense execution supplies the existing dense GEMM; streamed execution supplies the caller-destination packed projection. No public EP API, quota envelope, private pool or per-expert issuer request is added. The owned QMoE driver remains in use and is not migrated by this candidate.

Block execution binds four disjoint regions in one lane: existing decoded projection maximum, bank A, bank B, and the existing final accumulator. For a full-input bound R, hidden width H, intermediate width I and FC1 width F:

- Nongated: A=R*H, B=R*I. Gather may be overwritten by FC2 output only after its last read.
- Gated: A=R*max(H,I), B=R*max(F + separate_gate*I,H). FC1/FC3 finish before gather becomes activation; activation finishes before the projection bank becomes FC2 output.
- Accumulator: R*H, live across all experts. Decoded scratch remains independently bounded by the captured projection plan.

All dimensions are checked and complete layouts are validated before typed slices. Actual prefixes are initialized. Singleton rows inherit the existing float driver’s direct input-row borrow, while fixed bank capacity remains reserved. Larger windows do not gain a new borrowing heuristic. Expert order, K reduction order, bias application before activation, active weight readers and late tensor publication are preserved.

This admits FC1/FC3/activation/expert-output staging instead of allocating their result Vecs. Routing, input/bias conversions, compressed strided copies, original weights and other runtime allocations remain outside the workspace cap. Larger retained capacity is a deliberate cost and stays charged until idle reclamation; this study does not establish RSS savings or a complete process cap.

## Source and protocol

Base `509144567bc6f74d899385aacee5717d20811cc0` combines main f13b60689, the accumulator change #2458, the private destination change #2462 and an identical-precision math-lint correction. Only moe.rs and block_quantized_moe.rs change between measured builds. Hashes, compiler, reconstruction patch and zero-MLAS symbol checks are recorded. The null executable is a byte-identical baseline copy. The driver restores candidate source in finally before the timing sweep.

The entire build/sweep holds `/tmp/onnx-genai-hostlock-darwin.flock`, shared with the llama.cpp chat. Eight predeclared cases each have three arms and three independent processes, with rotated arm order and nine batches per process (72 formal processes). Excluded baseline pilots calibrate both loop iterations and an unchanged Relu control to 250 ms process CPU per batch. There is one attempt; no failed timing cell is retried.

Qualification requires every measured/control CPU batch >=20 ms, measured CPU/wall ratio >=0.9, and both candidate/null median control time within 5% of baseline. Raw controls, ranges, load snapshots (gzip) and failed gates remain included. The lock prevents cooperating jobs from overlapping, but does not guarantee quiet OS activity or physical-core placement on macOS. Float cases also exercise the changed shared per-expert path at the default grouping threshold; they are not an unchanged-code control. Relu is the independent off-path control.

## Results

CPU units are microseconds per prepare/run/drop. Values are medians of three process medians; brackets give their range. Gate 0=identity, 1=separate FC3, 2=interleaved SwiGLU, 3=split SwiGLU.

| Op | Rows/H/I | Threads | Dense | Gate | Baseline us | Candidate us | Delta | Identical-baseline delta | Qualified |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| block | 1/32/64 | 1 | False | 0 | 12.416 [12.301, 15.067] | 13.757 [11.911, 14.675] | +10.80% | +14.78% | NO |
| block | 8/128/256 | 1 | False | 0 | 501.154 [490.934, 561.471] | 491.258 [485.932, 549.258] | -1.97% | +0.90% | NO |
| block | 1/64/128 | 1 | True | 0 | 13.143 [12.920, 14.266] | 12.719 [12.545, 13.905] | -3.22% | -0.39% | NO |
| block | 8/128/256 | 4 | False | 1 | 1372.192 [1353.397, 1744.232] | 1701.543 [1364.536, 1872.338] | +24.00% | +23.95% | yes |
| block | 8/128/256 | 4 | True | 2 | 986.039 [804.388, 1113.397] | 988.914 [734.142, 1084.586] | +0.29% | -2.22% | NO |
| block | 8/128/256 | 4 | False | 3 | 1341.851 [1075.662, 1549.182] | 1543.786 [1303.870, 1625.065] | +15.05% | -12.27% | NO |
| float | 1/64/128 | 1 | False | 0 | 11.007 [10.811, 12.199] | 10.923 [10.767, 12.066] | -0.77% | -1.25% | NO |
| float | 8/128/256 | 4 | False | 1 | 727.663 [704.533, 1015.090] | 644.447 [627.925, 1076.367] | -11.44% | -6.59% | yes |

Six cells fail at least one timing/control guard. The two qualified multi-thread cells still have large identical-baseline shifts: separate-gate block candidate +24.002% versus null +23.945%, and float candidate -11.436% versus null -6.588%, with overlapping process ranges. Neither supports attribution to the source change. Preserve the data and keep adoption gated; no distinguishable end-to-end regression or gain is established by this noisy matrix, and unqualified cells cannot support a performance conclusion.

All 72 complete output byte strings and argmaxes agree within each case, and all values are finite. Workspace charge equals the metadata requirement in every process; retained use does not grow across warm loops. The expected dense payload charge matches the actual selected experts/projections, or zero for streamed/float fixtures. The scope is workspace plus eligible derived payloads, not process-wide memory.

| Op / geometry / mode | Baseline funded workspace bytes | Candidate funded workspace bytes |
|---|---:|---:|
| block 1/32/64 T1 dense=0 gate=0 | 1280 | 1536 |
| block 8/128/256 T1 dense=0 gate=0 | 18432 | 26624 |
| block 1/64/128 T1 dense=1 gate=0 | 2560 | 3072 |
| block 8/128/256 T4 dense=0 gate=1 | 28672 | 49152 |
| block 8/128/256 T4 dense=1 gate=2 | 32768 | 53248 |
| block 8/128/256 T4 dense=0 gate=3 | 32768 | 53248 |
| float 1/64/128 T1 dense=0 gate=0 | 1024 | 1024 |
| float 8/128/256 T4 dense=0 gate=1 | 28672 | 28672 |

Block capacity grows by R*(A_width+B_width-H)*4 bytes relative to the accumulator baseline; float requirements remain unchanged. Banks are not multiplied by expert count/top-k. Retention can be reclaimed only when idle; active work and ownership guards remain authoritative.

## Validation and rejected fixtures

Final native and MLAS-reference checks each pass BlockQuantizedMoE 29 tests, float MoE 24 tests (one existing ignored), and block MatMul 38 tests (one existing ignored). Strict all-target CPU Clippy passes in both configurations. Listings and named resource-test completion are verified. The shared driver preserves outputs across the existing 15-format/mixed-format, gating, bias, strided-input and empty-output tests.

The resource regression pre-admits a complete lane, checks zero issuer callbacks including zero-byte calls from the first warm computation/return, verifies retained physical capacity, and trims to zero. Missing/short backing and a computation-time plan mismatch must leave actual output unchanged. The initial fault fixture enlarged an invalid tile while keeping a full three-row frame; MLAS correctly refused its insufficient capacity before compute. The corrected fixture uses a smaller frame, explicitly proves it fits the already funded backing, and then reaches the intended tile mismatch. No production guard is weakened. The original fixture failure is retained.

The first debug link hit ENOSPC before any tests executed. A broad cache cleanup proposal was not executed; cleanup was narrowed to the two CPU test artifacts identified by this chat’s completed logs and the current failed native linker objects. The receipt records 382,307,976 logical bytes removed. Release checks and the final debug matrix then completed after disk recovery. Source, dependencies, other package/target caches and the stable flock inode were retained.

Main’s unrelated standard-math precision/grouping Clippy failures were corrected separately (#2466). Both f32/f64 coefficient bits remain identical (standalone compiler proof in constant-bits.log), and the isolated math branch passes its two module tests and strict all-target CPU Clippy. No CUDA runtime was executed.

## Decision and reproduction

Keep this candidate in draft with automatic merge disabled. The next performance experiment must address host/process-placement variance rather than infer a gain from this matrix or retry failed cells until favorable. A same-process, same-executable paired comparison is a possible diagnostic; it must retain a baseline/baseline control and its own declared gates.

Use the recorded base, candidate.patch, probe.rs and driver.py. The driver creates a standalone path-dependent Cargo package with ep-api, ep-cpu, ir, loader, memory-api, memory-governor, session, libc and rayon. Adjust the absolute temporary checkout, shared target and cached ORT 1.32.0 paths when reproducing elsewhere. Hold the stable host lock for the complete experiment and retain all failed controls.
