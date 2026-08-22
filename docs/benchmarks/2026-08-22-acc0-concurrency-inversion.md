# acc0 concurrency inversion: an ORT-configuration artifact over a single-session deficit

**Date:** 2026-08-22
**Author:** Sebastian (performance)
**Workload:** `int4_decode_loop_ab`, `PROBE_MODEL=qwen`, `PROBE_ACCURACY=0`, pure native CPU EP (no ORT fallback, no MLAS)
**Host:** AMD EPYC 9V74 (Azure VM slice), 32 vCPU = 16 physical cores, SMT siblings adjacent (physical set = even CPUs), two CCXs of 32 MiB L3 each, single NUMA node, 128 GB. ORT 1.28.0.

## The claim under investigation

acc0 was reported as inverting with concurrency: single session 0.44-0.71x of ORT, 2 sessions 1.16-1.54x, 4 sessions 1.59-2.35x. A result that *improves* with concurrency is the shape of a measurement defect, not a kernel property, so it was treated as one.

## Result

1. The inversion is **mostly an ORT-configuration artifact**. It disappears when both sides are given the same number of threads.
2. At **matched total work** there is no native concurrency collapse at all — 4 sessions are 14% *faster* than 1.
3. The genuine single-session deficit is 2.10x and **factorises exactly** into a runtime placement bug (1.66x) and a kernel per-block overhead (1.27x).

## No hardware PMU

`perf` counters are unavailable on this VM. This was verified as a virtualization limit rather than a permissions one by lowering `perf_event_paranoid` from 4 to 1, re-testing, and restoring it to 4. **No roofline or bandwidth-utilization percentage is quoted anywhere in this document with an unverified denominator.**

The one bandwidth number used is a *floor*, established by measurement: native at block 128 moves 123.8 MB in 1.870 ms = **66.2 GB/s**, and ORT reached 62.8 GB/s. This corrects the 56.6 GB/s figure in ledger section 22, which is an underestimate and should not be cited as a ceiling.

## 1. The artifact

`crates/onnx-runtime-ep-cpu/benches/ort_matmulnbits_baseline.py::make_session` sets `so.intra_op_num_threads = args.threads` for **every** session, and `run_concurrent` constructs N of them. `--threads 16 --sessions 4` therefore runs **64 spinning ORT threads on 16 physical cores**, while the native EP shares one bounded pool of 16.

Single interleaved, load-gated cycle; A/A null control passed at 2.3% and 2.9%. tok/s:

| arm | s=1 | s=2 | s=4 |
|---|---|---|---|
| ORT as configured (16 threads *per session*) | 416.3 | 120.4 | **69.6** |
| ORT matched (16 threads *total*) | 416.3 | 451.2 | 428.6 |
| native | 219.6 | 136.2 | 134.6 |

Against the oversubscribed arm the reported cells reproduce (0.53x / 1.13x / 1.93x). Against matched ORT they become 0.53x / 0.30x / 0.31x. **Native is worst at concurrency, not best.** This is the same failure mode as ledger section 12, where a margin that grew with concurrency was entirely artefactual.

ORT's oversubscribed arm is also *bimodal* — 2.3 ms or 5.1 ms, with the slow mode burning more CPU (27.7 s vs 16.5 s user). Any single-shot comparison against it is a coin flip.

## 2. Same-total-work control

Per-session token counts make s=4 do 4x the work of s=1. Holding **total** work constant instead (min wall over 3 gated cycles, A/A null 0.4%):

| configuration | total tokens | wall |
|---|---|---|
| 1 session x 400 tokens | 400 | 8089 ms |
| 2 sessions x 200 tokens | 400 | 7693 ms |
| 4 sessions x 100 tokens | 400 | **7090 ms** |

4 sessions are **14% faster**, while paying 4x the session-construction overhead — so the true parallel-efficiency gap is *better* than this table shows. There is no concurrency collapse. The deficit is **single-session under-utilization**.

## 3. The single-session deficit factorises

```
4.65 ms (default placement) / 2.21 ms (ORT) = 2.10x
                                            = 1.66x (runtime placement) x 1.27x (kernel per-block)
```

Both factors were measured independently:

- **1.66x placement** (A/A 0.30%): default 4.65 ms vs one-worker-per-physical-core 2.80 ms. Cause is #1680 — `node_shards` returned CPUs in kernel order and workers pin to `cpus[worker % len]`, so 16 workers packed onto 8 physical cores. A second defect compounds it: the dispatcher reserve counted *logical* CPUs, so it never fired and the spin-waiting dispatcher had to share a core with a worker, making that worker a barrier straggler. Unpinned w=16 -> 4.41 ms, w=15 -> 2.811 ms.
- **1.27x kernel**: see section 5.

## 4. Falsified hypotheses

Recorded so they are not re-spent.

- **"Scaling collapses from 8 to 16 threads": FALSE.** Quiet-host gated sweep, widths 1/2/4/8/16 = 39.990 / 20.447 / 10.414 / 5.315 / **2.831** ms — **88% efficiency at width 16**. The earlier "52% collapse" was co-tenant memory contention measured on a busy host.
- **CCX locality is not a factor.** 8 workers confined to CCX0 (5.99-6.04 ms) vs split across both CCXs (5.87-5.91 ms). The split was slightly *faster*.
- **Barrier straggler intolerance is not the contended-host mechanism.** `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` (merged, opt-in) gave no meaningful improvement under injected load (A/A 0.65%/1.1%). Separating mechanisms with purpose-built co-tenants: a memory streamer costs native **1.90x**; pure-ALU spinners cost **1.01x**. The penalty is DRAM/LLC bandwidth contention, not SMT issue slots and not barrier stragglers.
- **THP** (~78% resident in both fast and slow modes) and **adaptive calibration** (opt-in `=auto`, not the default) are both ruled out for the observed bimodality.

## 5. Kernel regime: per-block overhead, not traffic

At **width 1**, where bandwidth contention is arithmetically impossible (3.6 GB/s, ~5% of the measured floor), block size still moves the kernel by **1.55x** from block 32 to 128, with 0.5% sample spread. Bytes alone predict only 1.18x.

Fitting `t = a*blocks + c`: **a = 2.62 ns/block**, `c = 21.26 ms`. Per-block work is **47% of single-thread decode time at block 32**.

Multi-threaded, quiet host, minima:

| block | native | ORT | |
|---|---|---|---|
| 32 | 2.806 | 2.211 | 1.27x behind |
| 64 | 2.218 | 2.051 | 1.08x behind |
| 128 | **1.870** | 2.026 | **0.92x — native wins** |

ORT is nearly block-insensitive (-8.4% across the range). We are not slower in general, we are slower **per block**.

Target: `borrowed_int4_nblock4_avx2` (`matmul_nbits.rs:7097`). Its per-block tail does `scales.get(...)`, `layout.zero_point(...)` and `activation_sums[block]` per block per column, and the chunk loop re-indexes `packed_rows[c]` with bounds checks per chunk per column — the same pattern #1628 removed from the acc4 path for 1.61x.

## 6. Runtime defect found while investigating

`SpmdDecodePools::dispatch_output_rows` sized its serial-vs-parallel threshold from `output_chunk_len`, which reads `rayon::current_num_threads()`. The SPMD pool is not a Rayon pool, so that is an unrelated executor. With a narrower ambient pool the rule collapsed to "run it all serially" and every projection ran on the dispatcher thread while all 15 workers spun:

| `RAYON_NUM_THREADS` | before | after |
|---|---|---|
| 1 | **39.97 ms/token** | 4.669 |
| default | 2.82 | 4.669 |
| ratio | **14.2x** | 1.00x (A/A null 0.5%) |

`rayon::current_num_threads()` also *builds* the global pool, so the decode path was constructing an `available_parallelism()`-sized pool it never dispatched to. Process threads for a 16-core budget: **49 (15 SPMD + 32 Rayon + main) -> 17.**

Fixed by #1728.

## 7. Per-session lanes

Emulated by running 4 processes x 4 workers on disjoint core groups against 1 process x 4 sessions x 16 workers: **154.8 -> 271.6 tok/s (1.75x)**, distributions completely separated, A/A 1.0%. That restores exactly ORT's concurrency efficiency (0.92x of its 1-session figure).

**Caveat:** the emulation confounds lane partitioning with process isolation and core-group affinity. It justifies building an in-process implementation to confirm; it does not by itself justify shipping one.

## Methodology notes for this host

Hard-won, and most of these rejected results that looked publishable.

- **Round-robin interleaving, never sequential-per-arm.**
- **In-run A/A null control** — two identically configured arms. Reject the run if they disagree by more than 2-3%. This gate rejected roughly half of all runs and is what caught ORT's bimodality.
- **Load gating on the instantaneous runnable count** (`cut -d' ' -f4 /proc/loadavg | cut -d/ -f1`), not `loadavg` — the latter is a 1-minute EMA and stays elevated for minutes after a 64-thread arm exits.
- **Estimator:** contention is strictly additive, so use the **minimum** across replicates for latency and the **maximum** for throughput. Medians fail the A/A gate on this host (native medians disagreed by 11% where the minima agreed to 0.14%).
- **Single-threaded runs are dramatically noise-robust** (0.5% spread) — prefer them for mechanism isolation.
- Pin to physical cores with `taskset -c $(seq -s, 0 2 30)`.
- The bench runs **two** phases (`cold`, `steady`), so token-runs = `2 x PROBE_TOKENS x PROBE_SESSIONS`. This matters for any CPU-seconds-per-token arithmetic.
