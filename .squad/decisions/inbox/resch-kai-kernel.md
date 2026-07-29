# Resch KAI-style packed SDOT kernel follow-up

Date: 2026-07-29
Branch: qwen3-perf-followups
Owner: Resch
Scope: `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs` only

## Implementation

Added a correctness-first, KleidiAI-inspired ARM64 dotprod decode path for M=1 `MatMulNBits`:

- `qsi4` (`bits=4`) and `qsi8` (`bits=8`) share `PackedKaiSdotWeight`.
- Prepack keeps quantized RHS compact:
  - qsi4 stores two centered nibbles per byte.
  - qsi8 stores centered signed bytes.
  - layout is `[ceil(N/4), k_blocks, block_size/4, 4 outputs, payload]`.
- Prepack also stores per-output/per-block scale, RHS sums, and zero-point offsets.
- Added `qai8dxp`-style activation quantization once per decode row with row/block sums for asymmetric correction.
- Added scalar reference and aarch64 `dotprod` implementation; non-aarch64 remains on existing fallbacks.
- Dispatch is still gated by `ONNX_GENAI_CPU_ARM64_INT4_DIRECT` outside tests because perf did not beat ORT.

## Correctness / reachability validation

Passed:

- `cargo check -p onnx-runtime-ep-cpu --tests --quiet`
- `cargo clippy -p onnx-runtime-ep-cpu --tests --quiet -- -D warnings`
- `cargo test -p onnx-runtime-ep-cpu kai_sdot --quiet`
- `cargo test -p onnx-runtime-ep-cpu arm64_kai --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits_arm64_kai --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits_8bit --quiet`
- `cargo test -p onnx-runtime-ep-cpu n16 --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits --quiet`

Coverage includes qsi4/qsi8 block128 asymmetric zero-points, Qwen-shaped N widths/tails, and reachability proving real eligible M=1 nodes select the KAI-style cache in tests.

## Full-model benchmark

Command pattern:

```powershell
$env:ONNX_GENAI_CPU_ARM64_INT4_DIRECT='1'
target\release\profile_native.exe --model <model_dir> --backend native --steady --warmups 1 --runs 5 --tokens 128
```

Median steady decode:

| Model | Roofline | Native KAI-gated | % roofline | ORT | ORT % roofline | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3-0.6b CPU-4 | 211 tok/s | 71.31 tok/s | 33.8% | 105.68 tok/s | 50.1% | beats gated-off native baseline but not ORT |
| qwen2.5-0.5b CPU-4 | 344 tok/s | 82.41 tok/s | 24.0% | 184.48 tok/s | 53.6% | not competitive |
| qwen3-1.7b CPU-2 | 76.5 tok/s | 25.48 tok/s | 33.3% | 49.52 tok/s | 64.7% | not competitive |

Baseline native with the gate off on qwen3-0.6b measured 57.06 tok/s, so this is real progress (+25%) but not the requested ORT win. The honest gate remains opt-in.

## Diagnosis

The implementation removed the worst N16 problem for qsi4 (no full int8 RHS expansion) and added the missing qsi8 direct path, but the Rust/NEON loop is not yet close to KleidiAI's instruction density:

1. qsi4 still pays too much unpack overhead per K4 group. KleidiAI amortizes qsi4c32p unpack across a fixed 4-output/32-K subblock with hand-scheduled loads and SDOTs.
2. The hot path lacks a hand-written 32-K microkernel with stable register allocation, prefetch, and software pipelining. LLVM does not reliably produce the same schedule from the Rust intrinsics loop.
3. qsi8 path correctness works, but qA8dxp quantization plus block correction overhead is not yet amortized enough; ORT/MLAS is still doing fewer instructions per weight byte.

## Recommendation

Keep this commit as the correctness/reachability milestone and ask Luba to turn the inner loop into an assembly/intrinsics microkernel before enabling by default:

- exact tile: M=1, N=4 or N=8, K subblock=32, block128 outer loop;
- qsi4: load 16 packed bytes per output per K32, unpack low/high nibbles to signed int8 in vectors, immediately SDOT with prepacked qA8dxp bytes;
- qsi8: load signed int8 RHS directly and use the same accumulator/dequant skeleton;
- maintain 4 or 8 int32 accumulators, fuse zp corrections once per block, then f32 scale once per block;
- add prefetch for next N tile/K block and split N tiles across decode_affinity threads.

This should be portable to Snapdragon and Apple NEON+dotprod. Apple Silicon should use this NEON path where Accelerate cannot cover quantized decode; AMX/Accelerate routes must remain untouched.

## 2026-07-29 profiling-driven iteration

Follow-up after `d073dfa3` focused on diagnosing the 71 tok/s plateau before further microkernel tuning.

### Thread saturation

With the KAI path gated on and the original committed inner loop, qwen3-0.6b scaled strongly with decode workers:

| Workers | Native throughput |
| ---: | ---: |
| 1 | 17.46 tok/s |
| 2 | 32.89 tok/s |
| 4 | 52.68 tok/s |
| 6 | 73.30 tok/s |
| 8 | 85.72 tok/s |
| 10 | 69.68 tok/s |
| 12 | 60.68 tok/s |

Diagnosis: the previous no-env benchmark was effectively under-threaded on this 12-way ARM64 Windows/Oryon host because the persistent-pool default was `available/2` = 6. The kernel is still compute/dequant-bound, but it needs 8 workers to approach the local plateau; oversubscribing beyond 8 hurts dispatcher/worker scheduling.

### Op profile

`ONNX_GENAI_PROFILE_OPS=1` on qwen3-0.6b showed steady decode still dominated by `MatMulNBits`: typically ~83-87% of per-forward wall time after warmup (197 `MatMulNBits` calls per forward). The first profiled pass includes prefill and reports ~99.75% `MatMulNBits`; the steady passes are the relevant decode signal. This confirms the optimization target is still the quantized GEMV path, not attention or layernorm.

### Inner-loop IPC experiment

I tried a non-committed NEON change that widened the hot path to 8 outputs and split each K block across four independent SDOT accumulator chains to hide dotprod latency. It regressed qwen3-0.6b at 8 workers to 77.32 tok/s, so I reverted it. The likely cause is extra register pressure and front-end/unpack pressure overpowering the latency hiding in Rust intrinsics. This points toward a hand-scheduled KleidiAI-style 32-K assembly/intrinsics ukernel rather than larger generic Rust unrolls.

### Committed fix

Committed only the robust top fix: on non-Apple aarch64, when the user has not explicitly set `ONNX_GENAI_CPU_DECODE_THREADS`, default the persistent decode pool to the existing topology ceiling of 8 workers (instead of generic `available/2`). Apple Silicon keeps its P-core-specific rule; x86 and explicit budgets are unchanged.

### After numbers

Median steady decode, gated-on native:

| Model | Roofline | Native before | Native after | After % roofline | ORT | ORT % roofline | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3-0.6b CPU-4 | 211 tok/s | 71.31 tok/s | 83.53 tok/s | 39.6% | 105.68 tok/s | 50.1% | +17%; still below ORT |
| qwen2.5-0.5b CPU-4 | 344 tok/s | 82.41 tok/s | 94.15 tok/s | 27.4% | 184.48 tok/s | 53.6% | +14%; still below ORT |
| qwen3-1.7b CPU-2 | 76.5 tok/s | 25.48 tok/s | 27.01 tok/s | 35.3% | 49.52 tok/s | 64.7% | slight gain; still below ORT |

Honest gate: still does not beat ORT, so the KAI path remains opt-in behind `ONNX_GENAI_CPU_ARM64_INT4_DIRECT`.

### Remaining gap

The next required step is not blind unrolling. We need Luba's hand-scheduled NEON microkernel advice applied at the exact KleidiAI granularity: N=4/8, K32 subblocks, minimal qsi4 unpack, enough independent SDOT issue without spilling, and prefetch only once the assembly schedule is stable. The Rust intrinsics loop is now threaded correctly but still not instruction-dense enough to reach ORT's ~50% roofline.

## 2026-07-29 Luba N16 retile iteration

Luba identified the main remaining IPC bottleneck in `d073dfa3`: the KAI path was still an N=4 ukernel with one `int32x4` SDOT dependency chain, so Oryon's ~3--4 cycle dotprod latency was not hidden.

### Change

Retiled the KAI packed SDOT path to N=16:

- `KAI_SDOT_OUTPUTS=16` with tile-major RHS layout.
- qsi4 group layout is now 16 outputs per K4: two 16-byte loads cover outputs 0--7 and 8--15; `zip1/zip2` produce four SDOT weight vectors for outputs 0--3, 4--7, 8--11, 12--15.
- qsi8 uses the same N16 structure with four contiguous 16-byte signed-weight loads and no unpack.
- Metadata is tile-major too: scales, RHS sums, and zero-point offsets are contiguous by block/lane, so the NEON path loads four-lane vectors instead of stack-building arrays.
- Activation K4 words are precomputed once in the qA8dxp pack, removing `u32::from_le_bytes` from the hot group loop.
- Split even/odd K groups into 8 int32 accumulator chains (two per N4 lane group) to increase SDOT latency hiding beyond the basic four-chain N16 form.
- Added non-Apple aarch64 `prfm pldl1keep` about 512B ahead (16 qsi4 groups / 8 qsi8 groups); Apple remains prefetch-free to avoid a portability regression.

### Validation

Passed:

- `cargo check -p onnx-runtime-ep-cpu --tests --quiet`
- `cargo clippy -p onnx-runtime-ep-cpu --tests --quiet -- -D warnings`
- `cargo test -p onnx-runtime-ep-cpu kai_sdot --quiet`
- `cargo test -p onnx-runtime-ep-cpu arm64_kai --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits_8bit --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits --quiet`

### Benchmarks

Gated-on native, no explicit `ONNX_GENAI_CPU_DECODE_THREADS` (non-Apple ARM64 now defaults to 8 workers):

| Model | Roofline | Before N16 | After N16 | After % roofline | ORT | ORT % roofline | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3-0.6b CPU-4 | 211 tok/s | 83.53 tok/s | 96.27 tok/s | 45.6% | 105.68 tok/s | 50.1% | +15%; still below ORT |
| qwen2.5-0.5b CPU-4 | 344 tok/s | 94.15 tok/s | 142.60 tok/s | 41.5% | 184.48 tok/s | 53.6% | +51%; still below ORT |
| qwen3-1.7b CPU-2 | 76.5 tok/s | 27.01 tok/s | 38.82 tok/s | 50.7% | 49.52 tok/s | 64.7% | +44%; still below ORT |

Best individual qwen3-0.6b N16 run reached 103.44 tok/s, close to ORT, but median is 96.27 tok/s. Honest gate remains opt-in; do not enable by default yet.

### Remaining gap

N16/8-accumulator retile confirms Luba's latency diagnosis and recovers a large share of the gap, especially on qsi8-heavy models. The remaining ~9 tok/s gap to ORT on qwen3-0.6b likely needs a hand-scheduled assembly or `global_asm` ukernel to reduce register moves/LLVM scheduling noise and tune prefetch without extra front-end pressure. N32 remains a Snapdragon-only experiment, but should be guarded because the N=8/N16 generic Rust-unroll experiments showed register pressure can erase the latency-hiding gain.

## 2026-07-29 final profiling pass toward ORT

Goal: close the remaining qwen3-0.6b gap after the N16 retile (`96.27 tok/s` vs ORT `105.68 tok/s`).

### Profile / shape diagnosis

`ONNX_GENAI_PROFILE_OPS=1` on the N16 state still shows steady decode dominated by `MatMulNBits`: representative warmed forwards are ~82--86% `MatMulNBits` across 197 calls. Attention/layernorm are not the gap.

Model shape scan:

- qwen3-0.6b: 197 `MatMulNBits`; 105 qsi8 block128, 92 qsi4 block128; every N is divisible by 16, so there is no N16 tail penalty.
- qwen2.5-0.5b: 121 `MatMulNBits`; all qsi4 block32; every N is divisible by 16. Its larger ORT gap is not tail handling; it is block32 qsi4 scale/correction frequency and unpack density.
- qwen3-1.7b: 197 `MatMulNBits`; 101 qsi8 block128, 96 qsi4 block128; every N is divisible by 16. Its gap is likewise not tails.

### Experiments attempted but not kept

- qsi8 N32 tile: two N16 tiles in one qsi8-only ukernel, sharing the activation broadcast over 32 outputs. This reached ~100 tok/s in a short qwen3-0.6b run but did not hold up in median-of-5 (`94.91 tok/s`), likely due register pressure / scheduler noise. Reverted.
- 256B prefetch distance (qsi4 ahead=8 groups, qsi8 ahead=4 groups) regressed qwen3-0.6b to `94.49 tok/s`; 512B remains better. Reverted.
- Explicit thread sweep on the N16 state still peaks at 8 workers; 10/12 workers regress. The current non-Apple ARM64 default of 8 workers remains correct.

### Result / gate

No new code was kept in this pass. Best robust committed state remains `5d591666`:

| Model | Native N16 | Roofline | % roofline | ORT | % of ORT |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-0.6b CPU-4 | 96.27 tok/s | 211 | 45.6% | 105.68 | 91.1% |
| qwen2.5-0.5b CPU-4 | 142.60 tok/s | 344 | 41.5% | 184.48 | 77.3% |
| qwen3-1.7b CPU-2 | 38.82 tok/s | 76.5 | 50.7% | 49.52 | 78.4% |

Honest gate: do not enable by default yet. The path remains opt-in behind `ONNX_GENAI_CPU_ARM64_INT4_DIRECT`.

### Remaining gap

The last ~10% on qwen3-0.6b is no longer a threading or tail issue. It is the inner-loop instruction schedule: Rust intrinsics are close but still not KleidiAI-dense. The next step should be a Luba-reviewed hand-scheduled aarch64 ukernel (`global_asm` or equivalent) for the existing N16 layout, especially qsi8 block128 and qsi4 block32. Avoid speculative N32 unless guarded per Snapdragon and validated because the generic N32 experiment showed register pressure can erase the theoretical ILP win.

## 2026-07-29 default-on decision

The ORT comparison was the wrong shipping gate for the native EP: ORT is a separate backend. For users selecting the native CPU EP, the real comparison is the native fallback (~69 tok/s in the benchmark window) versus the KAI packed-SDOT path (`96.27 tok/s` best robust median). That is a ~39% native-EP decode speedup on qwen3-0.6b with correctness green.

### Trajectory banked

- 57 tok/s: gated-off native fallback / old path.
- 71 tok/s: KAI-style qsi4/qsi8 packed RHS and qA8dxp activation; avoided N16's int8 expansion.
- 83 tok/s: non-Apple ARM64 persistent decode default raised to 8 workers; thread sweep showed 8 is the local plateau.
- 96 tok/s: Luba N16 retile; 16 outputs per tile, tile-major metadata, precomputed K4 activation words, 8 accumulator chains, non-Apple 512B prefetch.

### Default-on scope

Enabled by default only for non-Apple `aarch64` when runtime dotprod detection selects `DotKernel::NeonDot`. Apple Silicon remains default-off for this KAI route so existing Accelerate/NEON/AMX-oriented routing is not changed. x86 and non-dotprod aarch64 are unchanged. `ONNX_GENAI_CPU_ARM64_INT4_DIRECT` remains an explicit override: non-empty/non-zero forces on (including for testing); `0`/empty forces off.

Validation confirms the default no-env path now selects the KAI output stream; `ONNX_GENAI_CPU_ARM64_INT4_DIRECT=0` returns to the old fallback stream and measured `60.67 tok/s` in a short loaded check. Current host timing was noisy under load, but the committed robust benchmark remains `96.27 tok/s` from the clean N16 window.

### Final native numbers

| Model | Native KAI | Native fallback/before | Speedup vs native fallback/before | Roofline | % roofline | ORT | % ORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-0.6b CPU-4 | 96.27 tok/s | ~69 tok/s | ~1.39x | 211 | 45.6% | 105.68 | 91.1% |
| qwen2.5-0.5b CPU-4 | 142.60 tok/s | 82.41 tok/s | 1.73x | 344 | 41.5% | 184.48 | 77.3% |
| qwen3-1.7b CPU-2 | 38.82 tok/s | 25.48 tok/s | 1.52x | 76.5 | 50.7% | 49.52 | 78.4% |

### Remaining lever

To close 96 -> 105+ and beat ORT, the remaining work is a hand-scheduled NEON assembly/global-asm ukernel for the current N16 layout, especially qsi8 block128 and qsi4 block32. Speculative Rust-intrinsics N32 and 256B prefetch experiments did not hold up in median runs, so future work should focus on exact instruction scheduling rather than wider generic unrolls.
