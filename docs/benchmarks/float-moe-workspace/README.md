# Float MoE grouped-input workspace

Float `com.microsoft::MoE` gathers routed inputs into the existing execution
workspace instead of allocating a gather Vec per expert or batched dispatch.
This funds gather backing only. Routing, dtype widening, FC intermediates and
the owned result remain outside this migration and the probe's Host cap.

## Contract

The kernel snapshots `ONNX_GENAI_MOE_GROUPED_MIN_WORK` at creation. Changes apply
to newly created kernels. Metadata planning and dispatch share this immutable
threshold; there is no process-global cache. The actual router still selects
between the two existing drivers using its largest expert group, and the
execution-time worker budget still excludes grouped execution at one worker.

One expert contains at most `rows` routed entries. If
`rows * hidden * (fc1_size + inter)` is below the threshold, no routing pattern
can choose grouped execution: reserve `rows * hidden * sizeof(f32)` (zero for
empty/singleton input). Otherwise reserve `rows * top_k * hidden * sizeof(f32)`.
The work estimate saturates; actual addressable element/byte spans are checked.
The planner never uses the preparing thread pool's width to exclude a route:
prepare and execution may enter different pools. Consequently a large shape
can reserve the grouped bound even when this particular run is single-threaded.
This is a deliberate bound, not a promise of exact routed occupancy.

Per-expert singleton groups borrow their original dense row. Other groups reuse
the same scratch prefix. Grouped execution retains the old serial threshold,
worker-budget cap and whole-row task partitioning. It initializes each gathered
value once, without a preliminary zero fill. Gather references cannot escape
execution. Missing, short or misaligned backing fails before output writes.
Neither driver introduces a pool, issuer protocol or per-expert quota operation.

## Correctness and resource checks

- Measured-source CPU: 2,020 passed, 20 ignored; session library: 296 passed.
  Integration on main after #2445: 2,018 CPU tests passed, 20 ignored; session
  library: 296 passed. The pending #2446 QMoE-specific tests are excluded.
- MoE focus: 21 passed, 1 performance test ignored, with native and optional MLAS.
  Existing comparisons preserve bit-identical drivers and scalar-reference
  coverage across activations, fusion, bias, row counts and half formats.
- Shared projection-layout target: 10 passed. New tests use uninitialized
  backing, repeated/noncontiguous rows, partial task chunks, invalid row indices,
  short capacity, and successful reuse after failure.
- New real Host pool test exercises both sides of the crossover, a forced
  grouped override, skewed and noncontiguous routes, and preparation in a
  one-thread pool followed by execution in one/four-thread pools. The independent
  unit-weight oracle agrees exactly. Warm checkout/execute/return makes zero
  issuer calls, including zero-byte calls; idle trim reconciles to zero.
- Strict CPU/session all-target Clippy and CPU MLAS all-target Clippy pass.
  `validation-*.log` preserves results. No CUDA device was used or validated.

## Public session comparison

Measured baseline: `d2165102d91a2d8b99ae886b3d29e272f78481cf`.
`candidate.patch` reconstructs the measured candidate from that commit.
`provenance.json` records source, probe and independently rebuilt binary SHA-256s.
This baseline includes the earlier QMoE migration, which this float probe does
not execute. Integration/rebases must not relabel these timings as a comparison
to a later main commit.

Host: Apple M1 Max, ARM64 macOS 27.0.1, rustc 1.98 / LLVM 22. Native CPU default
backend; symbol audits verify InferenceSession/main and absence of MLAS symbols.
The named thread count is the Rayon pool width, not a measured backend width.

The public InferenceSession probe times strict prepare, run, and output-owner
destruction. The graph has four float experts, top-2 normalized routing,
identity activation, deterministic weights, and alternating expert pairs.
Original inline weights and FC buffers are outside the scoped 64 MiB Host
workspace cap. Each workspace requirement exactly matches its retained charge.
Shutdown trims backing and checks zero remaining charge.

| Rows / hidden / inter | Rayon threads | Gate | CPU us, baseline median [range] | Candidate median [range] | Candidate change | Null change | Candidate workspace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / 64 / 64 | 1 | default, per-expert | 6.826 [6.787, 6.836] | 6.778 [6.768, 6.963] | -0.70% | +1.34% | 0 |
| 8 / 256 / 256 | 1 | default, per-expert | 135.141 [133.533, 137.896] | 133.532 [133.515, 138.063] | -1.19% | -1.31% | 8 KiB |
| 512 / 256 / 256 | 4 | forced grouped, threshold 0 | 7507.857 [7296.214, 7667.857] | 7575.643 [7495.643, 7866.571] | +0.90% | -0.37% | 1 MiB |
| 512 / 1024 / 1024 | 4 | default, grouped | 110361 [110194, 110918] | 110455 [110216, 111223] | +0.09% | -0.07% | 4 MiB |

Baseline/null workspace requirements are zero: their gather Vecs were ordinary
allocations, not absent physical memory. Parallel process CPU time sums workers;
`summary.json` also retains wall time. The whole-slot gather exists concurrently
with FC intermediates as before; funded backing can remain idle between runs.

There are three fresh processes per arm/cell, nine batches per process. Pilot
calls select common repetitions; pilots are excluded. Orders rotate baseline,
candidate and a byte-identical baseline null binary. Every measured batch and
independent Relu control exceeds 20 ms of process CPU time. All control median
ratios are within 5%. All 36 full outputs and argmax arrays agree within shape.
The process ranges overlap in every cell; these data establish neither a gain
nor a regression and do not prove universally zero overhead.

Builds and measurements hold `/tmp/onnx-genai-hostlock-darwin.flock`, shared with
the llama.cpp session. Supplementary process snapshots showed OS background
activity; this is not an idle-host claim or continuous contention measurement.
`host-context.json` records the observations and raw-file hashes. The checks are
limited CPU evidence, not model throughput or a CUDA performance claim.

Reproduction material: `probe.rs`, external `probe-Cargo.toml` / `probe-Cargo.lock`,
`driver.py`, `measure.py`, exact candidate patch and build logs. The build driver
initially stopped before any pilot when the sandbox denied `ps`; both source
arms were already rebuilt and restored. `measure.py` verifies source/binary
hashes before running those same binaries with process observation available.
Raw outputs/logs remain in `/tmp/nxrt-float-moe-perf`; `runs.json` preserves every
formal batch and output hash, with `output-verification.json` checking equality.

After verification, five probe executable copies/aliases (69,245,392 logical
bytes, not a measured physical-disk saving) were removed under the same lock.
`cleanup.json` records verified hashes. Sources, raw outputs/logs, shared target
dependencies and the stable lock inode remain.

## Main integration

The change was replayed onto `e127e6ceb` after #2445, without including the
pending #2446 runtime work. `integration-source.patch` reconstructs the final
CPU changes from that base; `integration-source.json` records file hashes.
Float MoE production code and the shared gather helper match the measured
candidate. Test-only refinements respect an MLAS worker budget below the Rayon
pool width and retain the warm callback baseline across routing changes. The
main branch's QMoE float-reference test now uses existing workspace preparation;
its initially failing integration run and corrected results are retained.
Historical timings continue to use their original d216 baseline.

Integrated checks: CPU 2,018 passed / 20 ignored; session library 296 passed;
projection layout 10 passed; MLAS MoE 21 passed / 1 ignored; strict native
CPU/session and MLAS CPU all-target Clippy passed. The named cross-pool resource
test also executes and passes with `ONNX_GENAI_CPU_DECODE_THREADS=1`.

A Windows CLI ORT failure in the earlier QMoE PR also exposed an independent
linked-model cache bug. This PR includes the same prerequisite correction:
physical file identity plus modification time replaces a timestamp-only
non-Unix cache key. One open handle supplies both facts; no weight bytes are
read/copied. The regression forces equal-size/equal-timestamp replacement.
The correction runs during model materialization, outside warm inference, and
does not change the measured MoE/projection production sources or timings.
