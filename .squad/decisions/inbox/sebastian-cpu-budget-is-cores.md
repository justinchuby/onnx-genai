# A CPU thread budget means physical cores, not logical CPUs

**By:** Sebastian (Performance Engineer) — 2026-08-18
**What:** `ONNX_GENAI_CPU_DECODE_THREADS=N` now confines the process to `N`
CPUs chosen **one per physical core** before any SMT sibling is used.
Previously it took the `N` lowest CPU indices of the chosen NUMA node, which on
every host we run on is `N/2` cores with both hyperthreads each.

**Why:**

- Measured on the 16-core/32-thread EPYC 9V74, int4 `MatMulNBits` 4096×6144
  with 128 tokens: a budget of 2 ran in 81.2 ms — *identical* to a budget of 1
  (79.4 ms) — while burning twice the CPU time. A budget of 3 (40.4 ms) beat a
  budget of 4 (53.9 ms).
- `taskset` isolates the cause with no code change: `0,2` (two cores) ran in
  50.1 ms against `0,1` (one core, two threads) at 81.8 ms, and used 36% less
  CPU. `0,2,4,6` ran in 33.5 ms against `0,1,2,3` at 54.3 ms.
- Across 42 A/B cells × 6 widths the fix is worth a geometric mean of 1.77× at
  a budget of 2 and 1.64× at 4 on the GEMM cells, with **zero regressions out of
  42 cells** at either width. A budget of 1 and a budget of 32 (the whole
  machine) are flat, which is the control.

**Consequences for everyone else:**

1. **Any benchmark that varies a thread count is now measuring something
   different.** Every `--threads N` grid published before 2026-08-18 with
   `N < logical_core_count` compared two arms that were both confined to `N/2`
   cores. The comparisons remain valid (both arms had the same mask) but the
   *absolute* scaling curves in those documents understate what the machine can
   do, and any conclusion of the form "we stop scaling past 8 threads" needs
   re-reading: 8 threads was 4 cores.
2. **Do not size a thread pool from `available_parallelism()` and assume
   cores.** Use `core_topology::CoreTopology` (`leaders_within`,
   `physical_cores_within`, `allowed_physical_cores`). It reads sysfs on Linux,
   `GetLogicalProcessorInformationEx` on Windows and `hw.physicalcpu` on macOS,
   and returns `None` rather than guessing when the platform exposes nothing.
3. **Spinning pools are the exception.** The persistent SPMD decode pool and the
   `numa-split` sub-pools are deliberately left compact. Spreading *spinning*
   workers one per core measured worse (0.133 ms vs 0.079 ms) — that experiment
   is recorded in `core_topology`'s module docs. The one-per-core rule is for
   fork-join workers that are handed arithmetic, not for workers whose job is to
   wait.
4. **The cpuset guarantee is unchanged.** The new code only *reorders* the
   candidate pool before truncating it, so the result is still a subset of the
   process's allowed CPU set. A container or `taskset` restriction is still
   honoured exactly, and a host with no discoverable SMT map falls back to the
   previous behaviour byte for byte.

**Where:** `crates/onnx-runtime-ep-cpu/src/decode_affinity.rs`
(`scatter_across_cores`, `smt_scaled_request`, `order_pin_targets`),
`crates/onnx-runtime-ep-cpu/src/core_topology.rs`. Evidence in
`docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §34.
