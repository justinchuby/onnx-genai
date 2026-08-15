# MLAS int4 end-to-end decode benchmark

**Date:** 2026-07-20T05:20Z
**Host:** Intel(R) Xeon(R) Platinum 8480C (Sapphire Rapids)

## Result

The newly landed MLAS SQNBit path is **not an end-to-end decode throughput win**
for the tested Qwen2.5 0.5B int4 model. With otherwise identical binaries and
arguments, selecting MLAS approximately halved throughput.

| Model | New tokens | Baseline (SimdX86) | MLAS SQNBit | MLAS / baseline |
|---|---:|---:|---:|---:|
| Qwen2.5 0.5B int4 | 128 | 18.08 tok/s (55.308 ms/step) | 9.53 tok/s (104.963 ms/step) | **0.527x (-47.3%)** |
| Qwen2.5 0.5B int4 | 256 | 17.50 tok/s (57.144 ms/step) | 9.04 tok/s (110.645 ms/step) | **0.517x (-48.3%)** |

Each performance result used two warmups and three measured runs, performed
sequentially to avoid benchmark contention.

## Correctness

Both modes produced coherent text and answered with Paris, but token output was
not identical. The difference is consistent with the different quantized
compute paths (`accuracy_level=4` selects MLAS CompInt8), and neither output was
garbage.

- **Baseline:** “Paris. It is the largest city in the country and the capital
  of France. It is also the most populous city in the country...”
- **MLAS:** “Paris. It is the largest city in the world and the most populous
  city in the world... The answer is **Paris**...”

The secondary GLM artifact had both required files, but was not runnable with
the required prompt: tokenization failed with `WordLevel error: Missing [UNK]
token from the vocabulary`.

## Path confirmation

The profiler was built with the bench crate's `mlas` feature, which forwards
`onnx-genai-engine/mlas` and `onnx-runtime-ep-cpu/mlas`. ONNX inspection found
121 `com.microsoft::MatMulNBits` nodes in Qwen, all `bits=4`,
`block_size=32`, `accuracy_level=4`, with no `g_idx`. Therefore the baseline
uses auto-detected SimdX86, while `NXRT_CPU_GEMM_BACKEND=mlas` satisfies the
SQNBit dispatch gate. `RUST_LOG=debug` emitted no additional kernel-selection
line, so confirmation is from the build/runtime gates and graph attributes.

## Commands

```bash
cargo build --release -p onnx-genai-bench --features mlas --bin profile_native

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 256 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 256 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"
```

Generated token IDs were decoded with Python's `tokenizers.Tokenizer` using the
model's `tokenizer.json`.

## Caveat and interpretation

Autoregressive decode is dominated by small-`M` GEMV-like work. The existing
hand-written int4 path is specialized for `M=1`, while SQNBit's strongest
microbenchmark wins are expected at prefill or larger `M`. These results do not
invalidate larger-`M` SQNBit gains, but they show that enabling MLAS globally is
currently a substantial regression for end-to-end single-sequence decode on
this small model.

---

## After M-based routing (fix)

**Date:** 2026-07-20T05:40Z — same host (Xeon 8480C, 96 hardware threads).

`try_mlas_sqnbit` now gates on `M`: MatMulNBits int4 with `m <
NXRT_SQNBIT_DECODE_MIN` (default **16**) falls back to the specialized hand
int4 GEMV (`int4_matmul_m1`), and MLAS `MlasQNBitGemmBatch` is used only once `m`
reaches the threshold (prefill). The `NXRT_CPU_GEMM_BACKEND=mlas` f32 backend is
left untouched.

| Config (`--tokens 128`) | int4 M=1 route | f32 route | Decode |
|---|---|---|---:|
| Baseline (no env) | hand | SimdX86 | **18.14 tok/s** (55.138 ms/step) |
| MLAS + M-gate (default 16) | hand | MLAS | **18.37 tok/s** (54.444 ms/step) |
| MLAS, gate disabled (`NXRT_SQNBIT_DECODE_MIN=0`, old behavior) | MLAS | MLAS | 9.62 tok/s (103.936 ms/step) |

Decode with the gate **fully recovers** to the hand-path baseline (+1.3%). The
only difference between the recovered run and the regressed run is int4 routing
— f32 stays on the MLAS backend in both — which proves the SQNBit int4 M=1 path
was the entire regression and the MLAS **f32** GEMV at M=1 is **not** a material
drag here. No f32 M-routing change was needed.

Correctness: baseline and MLAS+gate produce **identical** token IDs (decode uses
the same hand kernel), decoding to “Paris. It is the largest city in the country
and the capital of France. …”.

### Why the isolated microbench disagreed (crossover)

The `matmulnbits_mlas_perf` microbench (1 and 8 threads only) reports MLAS int4
*winning* even at M=1:

```
int4 K=2048 N=2048  M=1  1t: hand=184.5us  mlas=97.6us   speedup=1.89x
int4 K=2048 N=2048  M=1  8t: hand=44.9us   mlas=30.4us   speedup=1.48x
int4 K=2048 N=2048  M=32 1t: hand=16786us  mlas=1761.7us speedup=9.53x
int4 K=2048 N=2048  M=32 8t: hand=2330us   mlas=390.9us  speedup=5.96x
int4 K=4096 N=11008 M=1  1t: hand=1791us   mlas=1098us   speedup=1.63x
int4 K=4096 N=11008 M=1  8t: hand=339us    mlas=194.6us  speedup=1.74x
int4 K=4096 N=11008 M=32 1t: hand=176645us mlas=19182us  speedup=9.21x
int4 K=4096 N=11008 M=32 8t: hand=23128us  mlas=2844us   speedup=8.13x
```

But the real decode loop runs on the **full 96-thread** process Rayon pool, not
1/8 threads. MLAS's batch GEMM dispatches its own tiles across that pool; for a
tiny M=1 GEMV the many-thread fan-out/sync cost dominates, so end-to-end MLAS
M=1 is ~1.9× **slower** (9.62 vs 18.37 tok/s). The hand GEMV deliberately caps
dispatch for small work (`output_chunk_len`, `MANY_THREAD_CUTOFF`). At M=32 MLAS
wins by 6–9× even at 8 threads, so prefill still belongs on MLAS.

**Crossover:** in the deployed many-thread regime, M=1 favors the hand path
(~1.9×), M=32 favors MLAS (6–9×). The default threshold **16** keeps decode and
tiny batches on the hand path while routing real prefill (typically ≫16 tokens)
to MLAS. Tune with `NXRT_SQNBIT_DECODE_MIN`.

### Commands

```bash
cargo build --release -p onnx-genai-bench --features mlas --bin profile_native

# baseline
env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu --prompt "The capital of France is"

# MLAS + M-gate (default 16)
NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu --prompt "The capital of France is"

# regression repro (gate off)
NXRT_CPU_GEMM_BACKEND=mlas NXRT_SQNBIT_DECODE_MIN=0 ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu --prompt "The capital of France is"

# int4 hand-vs-MLAS crossover microbench
cargo test -p onnx-runtime-ep-cpu --features mlas --release \
  matmulnbits_mlas_perf -- --ignored --nocapture
```

---

## 7B addendum — re-profiling M=1 at production scale (perf/cpu-ep-mlas)

**Date:** 2026-07-22 — host Xeon 8480C (Sapphire Rapids, AMX + AVX512-VNNI), 2
NUMA nodes (node0 cpus 0–47, node1 48–95), 32 decode threads. Model:
Qwen2.5-Coder-7B-Instruct int4 generic-cpu (all MatMulNBits `block_size=32`,
`bits=4`, `accuracy_level=4`).

The 0.5B measurements above were re-validated against the 7B decode shapes
because the earlier "hand beats MLAS at M=1" conclusion was measured on a small
model. **The conclusion holds at 7B, but the reason is different and the gap to
ORT is not a kernel-choice problem.**

### Real per-token MatMulNBits shapes (7B, extracted from the ONNX graph)

| Projection | K | N | ops/token |
|---|---:|---:|---:|
| lm_head | 3584 | 152064 | 1 |
| gate + up | 3584 | 18944 | 56 |
| down | 18944 | 3584 | 28 |
| qkv | 3584 | 4608 | 28 |
| o_proj | 3584 | 3584 | 28 |

~3.5 GB of int4 weights are streamed per token (141 MatMulNBits ops).

### Cold vs cache-hot micro-benchmark (the earlier win was a cache artifact)

The `matmulnbits_mlas_perf` probe reuses the same activation/weight buffers
across iterations, so weights stay L3-resident — MLAS reports a 1.7–1.97× M=1
"win". That is a fantasy for decode, where every op touches a **distinct**
DRAM-resident weight. The new `matmulnbits_mlas_decode_step` probe replays the
real 7B op sequence with distinct cold buffers:

| Path (cold, distinct DRAM weights, M=1) | Throughput | Bandwidth |
|---|---:|---:|
| hand int4 GEMV | ~26 tok/s | ~92.9 GB/s |
| MLAS SQNBit CompInt8 | ~25 tok/s | ~89.2 GB/s |

At production scale M=1 decode is **memory-bound and the two paths tie**. MLAS
CompInt8 does not win, and it would add int8 requantization rounding — so per
rules 4/5 the hand path is retained for M=1 `accuracy_level=4`.

### Where the 2.3× end-to-end gap actually is

`perf record` over end-to-end decode:

| Bucket | Share | Notes |
|---|---:|---|
| MatMulNBits compute | ~44% | the actual GEMM work |
| rayon / crossbeam-epoch fork-join | ~27% | threads idle-spinning at per-op join barriers |
| `to_dense_bytes` | ~7.5% | one-time weight materialization |
| `prepack_int8_weight` | ~4.5% | one-time, cached in OnceLock |

141 ops/token × up to 64 `par_chunks_mut` tasks each means the process pays a
fork-join barrier ~141 times per token. NUMA test: `numactl --cpunodebind=0
--membind=0` yields **+25% (~10 tok/s)** but plateaus at ~10 even with 48
threads, using only ~14% of memory bandwidth → the decode loop is
**latency/sync-bound**, not bandwidth- or kernel-bound.

**Conclusion:** switching M=1 int4 decode to MLAS does not close the gap to ORT
(20.12 tok/s); the gap is engine-level per-op fork-join overhead plus NUMA weight
placement, which is cross-crate (loader + decode pool + executor) and out of
scope for the kernel. Recommended follow-up: reduce per-op fork-join barriers
(ORT-style persistent worker pool / fewer join points per token), and
NUMA-aware weight placement + thread pinning.

### Change shipped on this branch

- Renamed the knob `NXRT_SQNBIT_PREFILL_MIN` → **`NXRT_SQNBIT_DECODE_MIN`**
  (default **16**). It gates the `M` below which int4 decode with a *fast* hand
  path stays on the hand kernel; at/above it, MLAS SQNBit is used (prefill).
- The M=1 gate is now dtype/accuracy-driven, not just `M`-driven: only
  `bits==4 && accuracy_level==4` (the fast `int4_matmul_m1`/`int8_matmul`
  hand paths) are kept on the hand path for small `M`. Slow dequant-to-f32 M=1
  cases (e.g. `bits==4, accuracy_level!=4`) now route to MLAS CompFp32, a
  genuine generic win, while `g_idx` / 2-bit cases MLAS can't serve still fall
  back to the hand path. No model identity is used (rule 2).

### Contiguous bulk-copy fast path for f32 kernel I/O

`to_dense_f32` / `write_dense_f32` walked tensor elements one at a time with
multi-dim index bookkeeping even for contiguous tensors. In M=1 decode that is
~2.5M serial (non-parallel) iterations/token across every op's activation read
and output write. Adding a contiguous row-major bulk-copy fast path
(`copy_from_slice`) cut the Qwen2.5-Coder-7B decode step from ~104 ms to ~88 ms
best-case (9.61 -> 11.39 tok/s best; ~+11% median) on the 32-thread run, with
bit-identical output. Non-contiguous views keep the strided walk. The remaining
gap to ORT (20.12 tok/s) is per-op Rayon fork-join, executor dispatch, and NUMA
locality — see the decision note for the ranked cross-crate follow-up.
