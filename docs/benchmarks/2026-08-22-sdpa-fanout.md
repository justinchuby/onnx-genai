# SDPA fan-out for the pure-native CPU EP (#1718)

**Date:** 2026-08-22
**Author:** Sebastian (Performance Engineer)
**Base:** `origin/main` `6b639983c`
**Host:** AMD EPYC 9V74, 16 physical / 32 logical cores, two 32 MiB L3 domains, single NUMA node. **No hardware PMU** — `perf` is limited to the `cpu-clock` software event, so there is no roofline in this document and no cycle/instruction/LLC denominator is quoted.

## 1. What was wrong

`sdpa_f32_simd` is the SDPA route a **default, pure-native, non-MLAS** build actually ships. Its body was a plain serial triple loop:

```rust
for b in 0..batch { for n in 0..num_heads { for i in 0..q_seq { /* one row */ } } }
```

There was no fan-out anywhere in it. #1718 recorded the consequence from an interleaved window: ORT's attention went 3.05 ms → 1.06 ms from t=1 to t=16 while native stayed flat. The absolute milliseconds in that window were contaminated by shared-host load, but the *scaling shape* and the *source evidence* both survive contamination, and the source evidence is unambiguous: a loop with no fan-out cannot scale.

## 2. The change

Flatten the three loops into a single row index

```
r = (b * num_heads + head) * q_seq + i
```

`Y` is `[b, head, q_seq, Dv]` and `Q` is `[b, head, q_seq, Dk]`, so `q_base == r * head_size` and `y_base == r * v_head_size` exactly, and each row writes a disjoint contiguous span of `Y`. **Nothing is reassociated**, so the output is bit-identical to the serial version — not merely within tolerance. Softmax numerics, masks, causal offsets, soft-cap, `Dv != Dk`, odd head counts and sequence tails are all untouched: they live inside the per-row body, which moved verbatim into `sdpa_simd_row`.

Routing is hybrid and keys off **scope, never op or model identity**:

- if the decode pool is live (`matmul_nbits::decode_pool_active()`), use `decode_parallel_output_row_blocks` — during decode the forward runs on the engine thread while decode-pool workers are resident and spinning, so standing up a second executor contends with them. This is the same rule `GroupQueryAttention` already follows;
- otherwise use `task_runtime::chunk_runs_mut`.

It must **not** call `decode_parallel_output_row_blocks` unconditionally: its else-branch is `par_chunks_mut` on **global Rayon**, which would resurrect the pool #1728 removed.

Nesting is safe by construction — `for_each_range` returns `Backend::Serial` when already inside a task, and `SpmdDecodePools::dispatch` claims its barrier with a compare-exchange so a re-entrant dispatch runs inline rather than deadlocking. That legal inline case is also why the per-thread scratch is a `Cell<Vec<f32>>` take/put-back and not a `RefCell`: a held `RefCell` borrow would **panic** when the body re-entered on the same thread.

The grain policy `sdpa_rows_per_task` is deliberately **pool-blind** — it does not call `task_runtime::width()`, because that would *construct* the native pool even on the decode route.

## 3. Deriving the threshold (this is the part that was measured, not guessed)

The first implementation borrowed GQA's `MIN_PARALLEL_ATTENTION_WORK` (160 KiB). On the production grid that **regressed two real cells at t=16**:

| cell | MACs | t=16 with 160 KiB floor |
|---|---|---|
| `llama_decode_kv128` | 1.05 M | **+71%** |
| `bert_base_decode_kv1024` | 1.57 M | **+47%** |

Sorting every production cell by total MACs, the regressions stop at **2.1 M** and the wins start at **8.4 M**, with no production-shaped cell in between. `MIN_PARALLEL_SDPA_WORK` is therefore set to **4 Mi MACs**, inside that gap. `MIN_SDPA_WORK_PER_TASK` is 64 Ki (the grain floor per task).

### A probe that was thrown away

To fill the 2.1 M – 8.4 M gap I generated nine synthetic work-size cells (0.52 M → 8.4 M MACs). The results were non-monotonic, and `perf` explained why: those shapes are dominated by the **BSNH→BNSH transpose** (`load_bnsh`), not by attention — `llama32x128_kv512` at 4.2 M MACs spent 7.67 ms, far more than its arithmetic justifies. They measure the wrong thing. They are **discarded and disclosed here rather than quietly dropped**, and the threshold is set from the production grid alone.

## 4. Production matrix

16 MHA cells from `scripts/ort_ab/gen_mha.py` × t = 1/4/8/16 × three arms (**base** = `origin/main`, **new** = this branch, **null** = base rebuilt and rerun as an A/A control). Pure-native binaries; `nm | grep -ci mlas` = **0** on both. Median of 3 trials × 15 runs. Delta is native-vs-native; ORT is present in every cell as the parity check (all **PASS**).

Sorted by the **null arm's** deviation, which is the credibility gate — a cell is only as trustworthy as its own A/A control:

| cell | t=1 | t=4 | t=8 | t=16 | max abs null |
|---|---|---|---|---|---|
| `bert_base_s384` | -1.8% | -60.5% | -73.7% | **-83.4%** | 0.2% |
| `llama_prefill_s512_causal` | -0.5% | -60.5% | -78.5% | **-80.6%** | 0.9% |
| `bert_base_b8_s128` | -0.8% | -60.4% | -72.0% | **-79.9%** | 1.9% |
| `llama_decode_b8_kv1024` | +0.1% | -9.3% | -8.5% | -20.8% | 2.2% |
| `whisper_cross_s1500` | -0.5% | -67.2% | -74.6% | **-78.5%** | 6.0% |
| `llama_decode_kv1024` | +0.6% | -9.7% | -20.3% | -8.5% | 9.3% |
| `phi35_prefill_s256_causal` | -0.8% | -58.2% | -70.0% | -78.2% | 11.6% |
| `llama_decode_kv128` | +0.2% | +11.7% | -11.8% | +10.7% | 12.1% |
| `llama_prefill_s128_causal` | +2.8% | -48.8% | -69.9% | -59.2% | 19.5% |
| `clip_l14_s257` | -2.0% | -73.4% | -73.6% | -75.4% | 20.3% |
| `bert_base_decode_kv1024` | -3.7% | +3.0% | +6.2% | -1.5% | 21.1% |
| `llama_decode_past1023` | -1.4% | -8.8% | -14.4% | +12.0% | 21.7% |
| `llama_chunk32_past992` | -2.3% | -53.5% | -60.7% | -56.5% | 24.0% |
| `llama_chunk8_past1016` | +6.4% | -28.7% | -51.1% | -5.1% | 25.0% |
| `llama_decode_kv4096` | +0.4% | -9.2% | -12.7% | -16.5% | 25.5% |
| `bert_base_s128` | -0.7% | -46.4% | -75.5% | -65.6% | 35.7% |
| `vit_b16_s197` | -0.5% | -48.3% | +0.3% | -7.0% | 42.5% |
| `bert_large_s128` | -0.3% | -57.3% | -61.4% | -71.2% | 46.9% |

**How to read this honestly.**

- **t=1 is unchanged everywhere** (worst cell +6.4%, and its own null arm moved +11.6% — i.e. that cell is noise, not a regression). At t=1 the decode budget is 1, `resolve_width` returns 1, `plan_tasks` returns `None`, and the route is genuinely serial. Budget semantics are preserved without any pool probe.
- **The prefill/encoder cells are the result.** Their effect sizes (-48% to -83%) exceed even the worst null arms by 2-4x, and the three tightest-null cells are all large prefill wins. This conclusion is robust to the noise.
- **The decode cells are not resolvable at 3 trials.** Where the effect is ~10% and the null arm is ~25%, the 3-trial reading carries no information. They were re-measured — see next section.
- Several null arms are large because this is a **shared host**. That is disclosed, not smoothed over. Intra-run spread was tight in several of the contaminated cells, which is exactly the trap Roy documented: a tight spread means contention was *steady*, not that the host was idle.

## 5. Focused decode re-measure (11 trials × 15 runs)

| cell | t=2 | t=4 | t=8 |
|---|---|---|---|
| `llama_decode_kv1024` | -5.9% (null +0.3%) | -9.4% (null -1.3%) | -8.6% (null -1.9%) |
| `llama_decode_kv128` | -10.2% (null +2.2%) | +2.5% (null +3.1%) | -0.6% (null -3.5%) |
| `llama_decode_past1023` | +0.6% (null +7.0%) | -6.3% (null +11.7%) | -21.7% (null +6.0%) |

`llama_decode_kv128` is at **1.05 M MACs**, below the 4 Mi floor — under the final threshold it is *provably serial*, compiling to the same path as base. Its readings (-10.2%, +2.5%, -0.6%) are therefore a **direct measurement of this host's noise floor for a decode cell**, and they track its own null arm. In matrix 2 the same provably-serial cell read **+26.7% "> noise"** at 3 trials. That is the calibration: any decode reading under ±10% on this host at 3 trials is not a result.

Two cells that looked like t=4 regressions in matrix 2 (+32%, +27.5%) inverted to **-14.6%** and **-17.0%** wins at 11 trials.

## 6. Where the time actually goes (`perf record -e cpu-clock`)

### `llama_prefill_s512_causal`, t=16 — the SDPA-dominated cell

~73% of samples are on the SDPA critical path: `sdpa_f32_simd` 26.8%, `dot_avx2_fma` 16.7%, `axpy_avx2_fma` 13.8%, `expf`/`__xflowf` 9.4%, mask/bias `at` 6.1%. Direct A/B, two repeats each, tight dispersion:

**233.6 ms → 30.7 ms (7.6x)**, process CPU 137% → 956%, parity PASS.

### `llama_decode_past1023`, t=16 — the issue's own headline cell

This one barely moves, and the profile says why: it is **51% `__memmove_avx_unaligned_erms`**, under `concat_cache` and `load_bnsh`. SDPA is ~5.5% of samples, roughly 23% of wall on the critical path. **Amdahl bounds any SDPA fix on this shape at ~1.3x**, so the -14% observed is close to the ceiling.

The flat t=1→t=16 scaling reported in #1718 was real, but on decode-with-past shapes the KV-cache concat is the larger term, and this PR does not address it.

**Follow-up, out of scope here:** `concat_cache` still reaches **global Rayon** in a pure-native build. That is a separate lane and is not claimed or fixed by this work.

## 7. Falsifiers

Every claim below was verified by deliberately breaking the code and confirming the suite goes red.

| # | injected defect | caught by |
|---|---|---|
| F1 | grain policy always returns `None` (never fans out) | `fanned_out == 5` counter assert |
| F2 | head/batch decomposition corrupted | **independent `sdpa_f32_scalar` oracle** |
| F3 | task run offset ignored | bit-identity vs `force_serial()` |
| F4 | `v_head_size == 0` guard removed | zero-width no-op test (panics in `chunk_runs_mut`) |

**F2 is the interesting one: it was initially NOT caught.** The serial control and the parallel route share the same per-row body, so bit-identity falsifies *partitioning* bugs but is blind to *addressing* bugs — a wrong row decomposition is wrong identically on both arms. The fix was to compare against `sdpa_f32_scalar`, a genuinely independent implementation, inside the same test. This is recorded because the first version of the test suite would have shipped green with a corrupted head mapping.

## 8. Validation

- `cargo test --locked` over the `offline-linux` package set — **4595 passed, 0 failed**
- `cargo test -p onnx-runtime-ep-cpu --lib` — 1601 passed, 0 failed
- `cargo fmt --all -- --check`
- `cargo clippy --locked --all-targets` over the offline-linux set, `-D warnings`
- `cargo clippy --all-targets --target aarch64-unknown-linux-gnu -p onnx-runtime-ep-cpu -- -D warnings`
- `cargo clippy -p onnx-runtime-ep-cpu --features mlas` and `--no-default-features`
- `cargo test -p onnx-runtime-ep-cpu --features mlas --lib sdpa` — 27 passed
- Miri: this change adds **no new `unsafe`**. The pointer split lives in `chunk_runs_mut`, which the existing `task_runtime` Miri lane already covers. Running Miri over the new SDPA tests is infeasible (~46 M MACs).

macOS is believed unaffected — `sdpa_f32_simd` is only reached there for `q_seq == 1 && kv_seq * max(d, dv) <= 8192`, i.e. ≤ 0.52 M MACs, always below the floor. That is **reasoned, not measured**; no Apple hardware was available.

## 9. Bottom line

- Large, robust win on every prefill/encoder shape at t ≥ 4, peaking at **7.6x** on the SDPA-dominated cell.
- Modest, noise-limited win on decode shapes, bounded by Amdahl because decode-with-past is memmove-dominated.
- t=1 unchanged; no new `unsafe`; bit-identical output; no global Rayon reintroduced.
- The issue's headline cell is only ~23% SDPA. The remaining scaling gap on decode belongs to `concat_cache`, not to this kernel.
