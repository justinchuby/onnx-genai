# Same-process FC bank attribution experiment

Candidate: shared expert FC banks in draft #2469, commit `71b105629d3bb65e8c11a4eaba9f5e804143b92b`.
Baseline: `509144567bc6f74d899385aacee5717d20811cc0`.
This is one native Apple M1 Max CPU EP synthetic experiment, not model throughput,
CUDA validation, an RSS improvement, or a whole-process memory cap.

The previous cross-process experiment could not isolate implementation effects.
This experiment uses one executable and one CPU crate, sharing thread-budget,
allocator, decoder and Rayon globals. Cold provider construction replaces only
the MoE/BlockQuantizedMoE factories. No mode branch, counter or probe wrapper is
added to compute. Old production module prefixes are reconstructed exactly;
`baseline-copy-proof.json` records the transformations and verification.
Baseline and null are independent sessions using the identical old factory code.
Each arm has its own funded workspace pool and derived-weight books. Creation
order and nine interleaved measurement batches rotate over three processes.

Calibration checks process CPU each pilot iteration, so pilot observer overhead can shorten the subsequent batches; the recorded duration gates apply to the actual batches.

All eight smoke cases and all 24 measured processes compare complete output bytes,
finite values and output geometry before timing and again after timing. Retained
workspace equals its declared requirement; the two old arms agree, and the banked
arm has the declared capacity delta. Expected selected dense payload bytes also
match. Factory creation counters are positive after warmup and unchanged during
timing. Final teardown trims idle backing and verifies zero retained charge.
These observations do not establish zero transient weight issuer calls; the
existing resource unit test remains the evidence for warm workspace callbacks.

Each fresh process uses excluded >=50 ms process-CPU pilots to target 250 ms
batches, with the same iteration count across its three arms. Every arm batch
includes strict prepare, public session run and output-owner drop. An unchanged
Relu control precedes each arm. CPU time sums all process threads; wall time spans
the same loop. The shared stable host flock covers build and the complete sweep;
read-only process snapshots are retained. There is no CPU affinity claim and no
claim that the lock removes scheduler, thermal, DVFS or external host effects.

The predeclared gates require all CPU/control batches >=20 ms, every compute
CPU/wall ratio >=0.9, and each process's paired-median control and A/A ratios within
5%. A case qualifies only if all three processes qualify. Effects below are the
median of the three process paired-median ratios, retained even when disqualified.
No failed cell is discarded or rerun to obtain a favorable result.

Shape order: rows, H, I, threads, dense, gate mode, op. Gate modes: 0 identity,
1 separate FC3, 2 interleaved SwiGLU, 3 split SwiGLU.

| Case | Shape | Banked vs before | A/A vs before | Qualified | Gate evidence |
|---|---|---:|---:|---|---|
| 0 | `[1, 32, 64, 1, 0, 0, 'block']` | -1.773% | -0.013% | no | p2: control |
| 1 | `[8, 128, 256, 1, 0, 0, 'block']` | -1.048% | -0.035% | yes | all declared gates pass |
| 2 | `[1, 64, 128, 1, 1, 0, 'block']` | -2.421% | -0.196% | no | p0: control |
| 3 | `[8, 128, 256, 4, 0, 1, 'block']` | +1.290% | -0.266% | yes | all declared gates pass |
| 4 | `[8, 128, 256, 4, 1, 2, 'block']` | -4.174% | +0.837% | yes | all declared gates pass |
| 5 | `[8, 128, 256, 4, 0, 3, 'block']` | +0.908% | -0.021% | yes | all declared gates pass |
| 6 | `[1, 64, 128, 1, 0, 0, 'float']` | +0.309% | +0.422% | yes | all declared gates pass |
| 7 | `[8, 128, 256, 4, 0, 1, 'float']` | +1.954% | +1.651% | yes | all declared gates pass |

The archived ratios must be interpreted with their gate results; failed gates prevent attribution. The runtime PR remains draft pending an adoption decision based on the complete evidence, including its larger retained workspace. No new product resource API is adopted by this experiment.

`instrumentation.patch` applies to the candidate commit and contains the feature-gated cold factory selector and old module copies. The feature is never default-enabled. `probe.rs`, scripts, lock protocol, hashes, raw JSON and compressed logs preserve reproduction evidence. Paths in the archived manifest point to the isolated worktree and must be adjusted when reconstructing elsewhere. The live binary remains `/tmp/nxrt-bank-paired/paired-probe`; its hash is in provenance. Smoke timings are intentionally excluded. The initial sandbox failure happened before any measured process and is recorded separately.

The paired wall-time ratios are also retained in `summary.json`. For the qualified separate-FC3 streamed Block case, banked/before wall time increased **7.46%, 8.79%, 10.15%** across the three processes (median **+8.79%**), while its CPU median increased +1.29%. Its A/A wall median is -1.34%. The qualified dense interleaved case has median wall time **−11.26%** and CPU **−4.17%**. The outcome is mixed, rather than a uniform optimization: do not adopt the whole candidate on this evidence. #2469 remains draft while the streamed path is investigated.
