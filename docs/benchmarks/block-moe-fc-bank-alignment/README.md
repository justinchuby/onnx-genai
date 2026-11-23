# FC bank backing-alignment diagnostic

This single predeclared native Apple M1 Max experiment follows the
[paired FC bank study](https://github.com/justinchuby/onnx-genai/blob/c21707e05/docs/benchmarks/block-moe-fc-banks-paired/README.md).
Candidate is `71b105629d3bb65e8c11a4eaba9f5e804143b92b`; old kernels are from
`509144567bc6f74d899385aacee5717d20811cc0`. Both share a CPU crate, executable,
thread-budget/allocator/decoder globals, with factory choice only at construction.

Hypothesis: did backing-base misalignment cause the streamed shared-bank
regression? The experimental provider can request 128-byte alignment at the cold
allocation boundary and records the actual backing address modulo 128. No compute
wrapper, alignment branch or counter is inserted into the kernel. Five arms use
independent funded pools/books: old, banked, identical old A/A, old requesting 128,
and banked requesting 128. Actual capacity/derived charges, full output bytes,
finite values, positive/stable factory witnesses, and final trim-to-zero are
asserted in every process. No claim of zero transient weight issuer callbacks is
made. This experiment adopts no new product policy or resource API.

Five construction/batch rotations, ten batches per arm, two cases and one attempt
were fixed before measurement. Excluded >=50 ms CPU pilots target 250 ms batches;
per-iteration pilot observation overhead may shorten actual batches. Every batch
must actually exceed 20 ms CPU/control time; compute CPU/wall >=0.9; each process's
paired median A/A and control ratios must be within 5%. All five processes must
pass for a case to qualify. The stable shared host flock covers build and sweep.
Process snapshots are retained. CPU time sums all threads; wall brackets strict
public session preparation/run/output drop. This is synthetic, not model/CUDA/RSS
performance. No CPU-affinity or scheduler/DVFS stability claim is made.

The Host reports a **128-byte** cache line. Observed backing address moduli:

```json
{
  "before": [
    0
  ],
  "banked": [
    0
  ],
  "null": [
    0
  ],
  "before128": [
    0
  ],
  "banked128": [
    0
  ]
}
```

Ratios below are median process paired-median wall / CPU changes. Default and
requested-alignment arms have identical capacity within their implementation.
The A/A, per-process controls, duration gates and address observations are retained
in the raw records even for failed cells.

| Shape: R,H,I,T,dense,gate,op | All processes qualified | Banked/old | Banked128/old128 | Banked128/banked | Old128/old |
|---|---|---:|---:|---:|---:|
| `[8, 128, 256, 4, 0, 1, 'block']` | no | +8.034% / +1.312% | +8.410% / +1.451% | +0.318% / -0.425% | -0.195% / -0.238% |
| `[8, 128, 256, 4, 1, 2, 'block']` | no | -15.828% / -7.237% | -12.107% / -5.166% | +0.579% / -0.010% | -5.865% / +0.028% |

Every observed default backing was already 128-byte aligned. The cold request therefore delivered **no change in actual base alignment** for these cases. These results rule out an unaligned backing base as the explanation for these observed executions; they do not prove arbitrary alignment choices or cache behavior are irrelevant. In particular, do not interpret the requested-alignment arms as a delivered alignment treatment or as hardware false-sharing counters. No 128-byte product requirement is justified.

Draft #2469 remains unadopted. Investigate remaining scheduler/compute/layout effects with an instrument that distinguishes them, rather than introduce a new allocator, per-op resource request, or speculative alignment policy. This record retains failed controls and does not repeat cells to obtain favorable measurements. The experiment API and old-module copies are stored solely in `instrumentation.patch` and removed from product sources.

The patch applies to the candidate commit. `prepare.py` derives the experiment from the prior archived instrumentation; `probe.rs` is the final source. The manifest uses local isolated-worktree paths that must be adjusted for reconstruction. `protocol.json`, binary/source hashes, complete raw batches, native build/smoke logs and host snapshots preserve evidence. Binary remains `/tmp/nxrt-bank-alignment/alignment-probe` with its hash in provenance. Two reps=1 smoke runs are excluded from performance evidence.

Both aggregate cases fail the predeclared all-process control gate (one streamed-case process and two dense-control processes). Their displayed timing ratios are descriptive and do not establish an alignment performance effect. The address observation is independent of those timing gates: all 50 real workspace owners have base modulus zero. No measured cells are rerun and no threshold is relaxed.
