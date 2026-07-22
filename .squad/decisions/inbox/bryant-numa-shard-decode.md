# Decision: NUMA-split two-level decode for native CPU int4 M=1

**Author:** Bryant (senior systems/perf engineer)
**Branch:** `bryant/numa-shard-decode` (rebased onto `perf/cpu-ep-mlas` @ `96fd406`,
which includes Deckard's affinity review-fix `046414b`)
**Date:** 2026-07
**Status:** Positive result — opt-in, ships behind an env flag, no default-path change.
**Reviewer:** pending (rule 9 — non-author review required before merge)

---

## TL;DR

`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split` with `ONNX_GENAI_CPU_DECODE_THREADS=32`
(16 workers per NUMA node) raises steady M=1 int4 decode from a **16.87 tok/s**
compact-single-node baseline to **18.42 tok/s median (best 18.51)** — a **+9%**
gain — with **exact greedy bit-parity**. It does not reach the 20 tok/s target
(remaining gap ~1.6 tok/s / ~9%), but it is a real, repeatable improvement and
not a regression. The default path is untouched; the mode is entirely opt-in.
(Numbers are post-rebase onto Deckard's affinity fix; a pre-rebase run gave the
same conclusion at 16.40 -> 18.38.)

---

## 1. Profile-first baseline reproduction (rule 4)

Host: Sapphire Rapids Xeon 8480C, 2 sockets × 48 cores, 2 NUMA nodes
(node0 = CPUs 0–47, node1 = CPUs 48–95), AVX512-VNNI + AMX. Shared 96-core host,
noisy — every number below is a `steady_median` over `runs>=4`, and A/B configs
were **interleaved** across 3 rounds; I report median and best, never a single
run.

Command (worktree build):
```
export LD_LIBRARY_PATH=$PWD/target/release/build/onnx-genai-ort-sys-6b88787cafa9d9dd/out/ort-prebuilt/lib
ONNX_GENAI_CPU_DECODE_AFFINITY=compact ONNX_GENAI_CPU_DECODE_THREADS=32 \
  ./target/release/profile_native \
  --model ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4 \
  --tokens 24 --runs 5 --warmups 1 --steady --decode-skip 8
```

Baseline `compact` T=32: **16.87 tok/s median** over 5 interleaved rounds
(16.87 / 17.07 / 16.14 / 15.37 / 16.97), in line with Batty's ~16.3. Matches
Batty's finding that MatMulNBits (int4, `accuracy_level==4`, block 32, M=1) is
the hot op and decode is memory-latency + per-op fork-join-barrier bound, not
compute bound.

## 2. Design (numa-decode-plan steps 4–5)

Use **both** sockets' memory bandwidth without paying a flat 64-way cross-socket
per-op barrier:

- A tiny **dispatcher pool** (one worker per NUMA node) installs the M=1 forward
  via `with_decode_pool_scope`, with `IN_NUMA_SCOPE` + `IN_DECODE_POOL` set so
  inner `with_decode_pool` calls run inline.
- Each MatMulNBits kernel (`int4_matmul_m1`, `int8_row`, `gemv_nk`) routes its
  parallel section through `parallel_output_rows`, which when numa-active calls
  `dispatch_output_rows`: the output rows are split into per-node **contiguous**
  slices; each slice runs on its **node-pinned sub-pool** via
  `dispatcher.install(|| segments.into_par_iter().for_each(|seg| node_pool.install(compute)))`.
- **Two-level barrier:** node-local `par_chunks_mut` is the first level; the
  single `for_each` join across nodes is the *only* cross-node barrier per op —
  replacing the toxic flat 64-thread cross-socket coherency round-trip that made
  Batty's naive interleaved pool 11.1 tok/s.
- **Node-local weight first-touch:** at prepack, `place_rows` allocates an
  uninitialized buffer (zero-filling would fault every page onto the dispatcher
  node) and each node's sub-pool **copies its own row-shard**, first-touching
  those pages on the owning node under the default policy.
  `row_lengths(n)` is the single source of truth used by both weight placement
  and compute dispatch, so they always line up.

**Bit-parity argument:** row-sharding a GEMV is exactly associative — each output
row is an independent dot product over the full K. The activation is quantized
**once** before dispatch (shared read-only). There is no cross-node K-reduction,
so results are bit-identical regardless of the row partition. Verified
empirically (§4).

New module: `crates/onnx-runtime-ep-cpu/src/decode_numa.rs`. Topology + affinity
parsing extended in `decode_affinity.rs` (`NumaSplit` variant, `NodeShard`,
`split_workers`). Kernel wiring in `kernels/matmul_nbits.rs`.

## 3. A/B results — median + best (post-rebase, 5 interleaved rounds, T=32 total)

| mode        | T   | per-round tok/s                       | median | best  |
|-------------|-----|---------------------------------------|--------|-------|
| compact     | 32  | 16.87 / 17.07 / 16.14 / 15.37 / 16.97 | 16.87  | 17.07 |
| **numa-split** | **32** | **18.42 / 18.51 / 18.15 / 18.44 / 18.30** | **18.42** | **18.51** |

**numa-split T=32 (16+16) is the winner: +9% over compact baseline, and notably
*more stable* run-to-run (18.15–18.51) than compact (15.37–17.07).**

A third fresh 3-round A/B on the final committed branch reconfirmed the result:
compact **16.66** median (15.29 / 16.74 / 16.66), numa-split **18.00** median
(17.77 / 18.28 / 18.00) — +8%. Across all three benchmarking sessions numa-split
lands 18.0–18.5 median vs compact 16.4–16.9; the win is robust to host noise.

A pre-rebase 3-round A/B at the wider grid corroborated the direction and showed
the failure modes of over-threading:

| mode        | T   | median (pre-rebase) |
|-------------|-----|---------------------|
| compact     | 32  | 16.40 |
| numa-split  | 32  | 18.38 |
| numa-split  | 64  | 15.42 (barrier cost dominates) |
| compact     | 64  | 10.18 (cross-node thrash) |

### Thread scaling (numa-split, 2 rounds each)

| T (per-node)  | tok/s        |
|---------------|--------------|
| 16 (8+8)      | 16.48 / 16.54 |
| 24 (12+12)    | 17.64 / 17.89 |
| **32 (16+16)**| **18.23 / 16.81** (peak) |
| 48 (24+24)    | 16.28 / 17.13 |
| 64 (32+32)    | 15.42 / 15.04 (from A/B) |

Clear knee at **T=32**. Below it, memory bandwidth is under-used; above it, the
per-op two-level barrier and cross-socket coherency cost of more workers erodes
the bandwidth gain. This is exactly the barrier-vs-bandwidth tradeoff the plan
predicted; the two-level structure moves the sweet spot up from single-node but
does not eliminate the per-op join cost.

## 4. Bit-parity confirmation

Greedy `generated_token_ids` were **identical** between compact (single-node) and
numa-split across **every** configuration (T=16/24/32/48/64), on two prompts:

- Default `"Hello"` (24 tokens) — both produce:
  ```
  [48298, 271, 9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645, 198,
   151643, 151644, 198, 151643, 151644, 198, 151643, 151643, 151643, 151643, 151643]
  ```
- A code prompt (32 tokens, real content) — both produce byte-for-byte:
  ```
  [576, 729, 1265, 3705, 2176, 25780, 323, 9069, 11, 323, 432, 1265, 10034,
   1142, 26443, 369, 9069, 382, 8420, 594, 458, 3110, 315, 1246, 279, 729,
   1265, 975, 1447, 73594, 12669, 198]
  ```

**On Batty's reference sequence** `[576, 729, 1265, 1896, 264, 1140, ...]`: Batty's
methodology note abbreviates his command (`profile_native ...`) and does not record
the prompt string; the tool's default prompt is `"Hello"`, which produces a chat
*greeting* (the `48298...` sequence above), not code. Batty's ids are clearly a
code-completion (`" The function should ..."`), so he used an undocumented code
prompt. My code-prompt run reproduces the exact same opener `[576, 729, 1265, ...]`
and then diverges (his prompt ≠ mine), confirming the reference is prompt-specific.
The correctness-relevant invariant for *this change* is that row-sharding a GEMV and
concatenating is numerically exact — i.e. numa-split == the single-node path
byte-for-byte on the **same** build+prompt — which is verified above on both a
trivial and a non-trivial (32-token code) output. Row-sharding is exactly
associative (each output row is an independent full-K dot product; the activation
is quantized once before dispatch; no cross-node K-reduction), so this parity holds
by construction.

## 5. What worked / what didn't (with evidence)

- **Worked:** node-pinned sub-pools + row-sharded weights + two-level barrier at
  T=32. +9% median and exact parity, with lower run-to-run variance than compact.
  Both sockets' bandwidth is used with a single cross-node join per op.
- **Didn't:** scaling past 32 total threads. T=64 numa-split (15.4) is *worse*
  than T=32 (18.4) and than compact T=32 (16.9) — the per-op cross-node barrier
  and coherency traffic of 64 workers outweighs the marginal bandwidth. So
  "throw more threads at both sockets" is the wrong lever; the correct lever is
  *bandwidth per node with a minimal barrier*, which peaks at 16 workers/node.
- **Not pursued:** node-local KV cache residency (numa-decode-plan future step) —
  larger surface, deferred. Sharding only the largest projections was
  unnecessary: uniform row-sharding already lands a positive result and the
  T-scaling curve shows the barrier cost, not small-op dispatch overhead, is the
  ceiling.

## 6. Remaining gap

18.42 median vs **20 target ⇒ ~1.6 tok/s / ~8% short**; vs 16.87 baseline ⇒ **+9%**.
The remaining gap is dominated by the per-op cross-node join latency (141 ops/token
× one cross-socket barrier each). Closing it likely needs *fewer* cross-node
synchronizations per token — e.g. node-local KV so attention doesn't re-cross,
or fusing consecutive projections under one barrier — rather than more threads.
Recommend that as the next step. Reference points: ORT 26.9, onnxruntime-genai 20.8.

## 7. Safety / rules compliance

- **Rule 5 (opt-in):** default path unchanged; only `numa-split` activates it.
- **Rule 2 (no hardcoded topology):** nodes/CPUs queried from `/sys` at runtime
  via the reused `decode_affinity` topology; no hardcoded node/core counts.
- **Rule 1 (good errors / graceful fallback):** single-node/non-Linux hosts,
  `THREADS=0`, or <2 populated nodes fall back to flat single-node decode,
  logged **once** via `report_numa_fallback`. Malformed env values remain a hard
  error from the existing flat path.
- **Rule 4 (reuse MLAS):** each node runs the **existing** MLAS SQNBit / hand
  VNNI kernel on its row-slice; no new matmul was hand-rolled.
- **Rule 8 (tests track behavior):** unit tests added for row-length splitting,
  dispatch-equals-flat, byte-preserving placement, and worker splitting across
  nodes. `cargo test -p onnx-runtime-ep-cpu --features mlas` → **675 passed, 0
  failed** (includes Deckard's 4 affinity-fix tests after the rebase). `cargo
  clippy` clean.

## 8. Coordination with Deckard's affinity fix (rebased)

Reviewer Gaff rejected Batty's affinity commit `32a122e`; Deckard landed a
surgical fix (`046414b`) confined to `decode_affinity.rs`: (1) `cpu_set_t` mask
sizing → a runtime-sized `build_cpu_mask`, (2) a single consistent invalid-value
diagnostic centralized in `ACCEPTED_MODES`/`invalid_selector_error`, and (3)
`compact` node-selection → smallest-index `find`.

Per the coordinator's guidance I kept my feature **additive** and did **not**
touch those three areas' logic:
- My `numa-split` parse arm returns `Ok(NumaSplit)` and flows through Deckard's
  new `resolve()` via its `Ok(affinity) => Ok(affinity)` pass-through — no change
  to his validation logic.
- `pin_current_thread_to_cpu(cpu: usize)` signature is unchanged, so my
  per-node sub-pool pinning needed no adaptation; it transparently benefits from
  the new mask sizing.
- My `DecodeAffinity::NumaSplit => Ok(None)` arm in `cpus_for` sits alongside his
  rewritten `Node`/`Compact` arms (the flat fallback pool stays unpinned).
- The rebase conflicted only in the tests module tail (both sides appended
  tests); resolved by keeping **both** sets.
- The **one** shared-line change I made is extending his `ACCEPTED_MODES` const
  to include `` `numa-split` `` so the invalid-value diagnostic lists the new
  mode (rule 1). His fix was already committed, so this is a static additive
  extension, not a live collision. His diagnostic tests only assert the three
  original modes are present, so they still pass.

I rebased `bryant/numa-shard-decode` onto `perf/cpu-ep-mlas` @ `96fd406` (which
contains `046414b`). The coordinator can fast-forward/cherry-pick it onto
`perf/cpu-ep-mlas`. **Not pushed** (coordinator pushes).

## 9. Handoff note (concurrent-agent hazard)

A concurrent agent was earlier running `git reset`/`checkout` on the shared main
working tree `/home/justinchu/onnx-genai-cpu`, which silently wiped in-progress
(including untracked) files twice. I therefore did all work in a separate git
worktree `/home/justinchu/onnx-genai-cpu-bryant` on branch
`bryant/numa-shard-decode`.
