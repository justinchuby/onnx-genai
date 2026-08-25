# acc0 width-16 straggler: separating "slow lane" from "slow chunk"

**Status: instrument landed; experiment run, result in §5.**

Sections 1–4 were written and merged (#2030, #2041) *before* the experiment ran
and are unedited below, including the predictions in §4. That ordering is the
point: publishing the instrument and its verdict together would make it
impossible to tell which of the two was designed first. §5 was appended
afterwards and scores the result against those predictions as written.

## 1. The ambiguity this exists to break (#2017)

The width-16 decode straggler is real, persistent within a process, and worth
~0.31 of the dispatch window. Six candidate selectors have been measured and
rejected: work assignment (`ops_spread` = 0.0000 in 24/24 launches), lane index,
CPU placement (one lane→cpu map across 24 launches while the victim moved),
virtual address layout (`setarch -R`), physical page backing
(`prctl(PR_SET_THP_DISABLE)`, ratio 1.023 against a required 0.60), and
mode-of-operation explained by placement (240 launches on verified one-per-core
placement, still bimodal ~50/50).

The candidate list is empty, and there is a structural reason no seventh
hypothesis can be tested with the EP as it stands:

* lane *i* always runs on cpu *2i* (static placement), **and**
* lane *i* always computes output chunk *i* (static assignment).

Both maps are fixed for the life of a process, so **"lane 7 is slow" and "chunk 7
is slow" predict identical observations in every experiment run to date.** Every
dataset either agent holds is consistent with both. This is not a sample-size
problem and more repetitions cannot fix it.

The two have different causes and different fixes. A slow *lane* is a
thread/core/hardware property. A slow *chunk* is a data property — a weight
region whose cache colouring, physical pages or NUMA interleave make identical
arithmetic take longer. `straggler_idx == slowest_idx` at 0.667 / 0.667 / 0.684
across 73 launches (chance 0.067) says the victim usually genuinely *computes*
longer rather than starting late, which is what makes a chunk explanation
plausible at all.

## 2. The instrument

`ONNX_GENAI_CPU_DECODE_CHUNK_PERMUTATION` permutes lane→chunk **while holding
lane→cpu fixed**:

| value | meaning |
|---|---|
| unset / `off` / `identity` | today's behaviour, exactly (default) |
| `rotate:<k>` | lane `i` computes chunk `i + k` within its node |
| `seed:<n>` | deterministic shuffle (splitmix64 + Fisher-Yates) |

An unparseable value falls back to the default rather than failing: this is a
diagnostic knob and a typo must not change numerics.

The experiment is then one line — hold placement fixed, permute the chunk map,
and ask whether the straggler follows the **lane index** or the **chunk index**.
Concentration on one vs the other separates the hypotheses. Both currently read
~0.208 against a 0.5 bar *only because they are the same number*.

### Why it is safe on the default path

Three properties, each asserted by a test rather than argued here:

1. **The segment set is unchanged.** The permutation reorders an already-computed
   table; it never recomputes boundaries. Alignment is applied to the *canonical*
   order and permuted afterwards, precisely so that turning the knob on cannot
   change which boundaries exist — only who receives them. Results are therefore
   bit-identical, and every output row is still computed exactly once.
2. **It is node-local.** The permutation runs within one node group at a time, so
   a row range never migrates to another NUMA node and `place_rows` first-touches
   it on the same node it would have anyway.
3. **Touch and compute stay together.** `place_rows` and dispatch read the same
   permuted table, so a lane first-touches and later computes the same rows. This
   matters for the experiment as well as for correctness: it keeps the knob a
   *relabeling*, not a change to where pages live.

The default is the identity function, not a permutation that happens to be
cheap, and the MLAS work-stealing pool records `Identity` explicitly because it
hands out tiles dynamically and has no static map to permute.

## 3. Correctness evidence

Five tests, and — more to the point — the negative controls that show they are
not decoration.

* `chunk_permutation_parses_env_values` — the parser, exhaustively, as a pure
  function. No `set_var`: this module forbids it (Rust 2024 data race against
  every other test thread's `getenv`).
* `chunk_permutation_is_always_a_permutation` — every mode, every count `0..=33`,
  must be a genuine permutation of `0..count`. This is the property coverage
  rests on; an off-by-one here would compute some rows twice and others never.
  Also asserts seed determinism (same seed ⇒ same permutation) so a reported
  `seed:<n>` is reproducible evidence rather than a description of a shuffle
  nobody can repeat.
* `chunk_permutation_preserves_the_segment_set_and_stays_node_local` — the
  permuted table is a reordering of the canonical one, coverage of `0..n` is
  exact, and no segment crosses a node boundary.
* `chunk_permutation_does_not_move_aligned_boundaries` — the aligned segment set
  is invariant under permutation.
* `the_chunk_permutation_env_string_reaches_dispatch` — the one that matters.

### The reachability test, and why it is written that way

This starts from the **documented env string**, in a child process, and asserts
that the lane→chunk map *changes*. That shape is a direct consequence of #2014:
`ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` was covered by tests that constructed
`DecodeSchedule::Steal` **directly**, so the implementation was verified while
its *reachability* never was — and the knob was inert in every shipped build for
as long as it existed. It is the third member of a family with #1792
(`ONNX_GENAI_CPU_DECODE_AFFINITY` verifiably inert) and the latched-`OnceLock`
A/B.

> **A knob is not verified until an observable changes when you turn it.**
> Cover the path from the user-settable input, not just from the internal enum.

The test asserts, across `unset` / `identity` / `rotate:3` / `seed:7`:

* unset behaves exactly as `identity`;
* `rotate:3` and `seed:7` each **change** the map (anti-vacuity — without this
  the test would pass against a build where the env read is dead code);
* the *sorted* maps are equal, so only the assignment moved;
* the result checksum is identical in every arm;
* and coverage is checked from the **dispatch itself** (one `AtomicUsize` per
  output row, asserted exactly 1) rather than from the table it was derived from.

### Negative controls (both run, both fired)

| control | injected defect | result |
|---|---|---|
| A | pool init hardcoded to `ChunkPermutation::Identity`, making the env read dead | ``rotate:3` did not change the lane->chunk map, so the env string never reached dispatch` |
| B | `Rotate` corrupted to duplicate an entry (no longer a permutation) | four independent failures: `not a permutation ([0, 0])`, `the segment set changed`, `permuting changed the aligned segment set`, and the child's live coverage assertion |

Control B is the important one: the coverage guard fires from the running
dispatch, not from re-reading the table, so it would catch a permutation that
corrupted output even if the table looked self-consistent.

Both controls reverted. 1727 lib tests pass, clippy clean, `--features mlas`
checks.

## 4. What is deliberately not claimed

No performance claim, in either direction. The knob is off by default and its
default is the identity, so the shipped path is byte-for-byte unchanged — that
is a correctness argument, not a measurement, and it is the only claim made here.

The experiment (does the straggler follow the lane or the chunk?) requires a
quiet host and has not been run. Two predictions worth recording **before** the
data exists, so the result cannot be read backwards:

* If the straggler follows the **lane**, it is a thread/core/hardware property
  and the remaining candidates are wake latency and per-core hardware state.
* If it follows the **chunk**, it is a data property and the next instrument is
  cache colouring / page interleave of that weight range.

A third outcome is possible and would be the most informative: the straggler
tracks *neither* under permutation, which would say the victim is selected by
something that moves when the assignment moves — i.e. an interaction, not a
property of either map alone.

---

## 5. Result (added 2026-08-24, after the experiment ran)

Three datasets on a quiet host, 832 pooled trusted samples (120 + 476 + 236). Every launch is a
fresh process; each launch runs all arms interleaved.

| dataset | arms (`k`) | launches | pooled samples |
|---|---|---|---|
| pilot | 0, 4, 8, 12 | 30 | 120 |
| confirmatory | 0, 4, 8, 12 | 120 | 476 |
| odd-`k` | 1, 3, 5, 7 | 60 | 236 |

Controls held in all three: CONTROL 1 (the binary reports the arm it was
launched as) retained 1.000 in every arm — the observable added in #2041 is what
makes this checkable at all. CONTROL 2 (placement fixed) found exactly **one**
lane→cpu map, `[0,2,4,…,28]`, 15 distinct cpus, across all launches.

### 5.1 RULE 1 (pre-registered primary) — NEITHER, in all three datasets

| dataset | lane conc. | lane floor | chunk conc. | chunk floor | verdict |
|---|---|---|---|---|---|
| pilot | 0.1417 | 0.1417 | 0.1083 | 0.1500 | NEITHER |
| confirmatory | 0.1429 | 0.1429 | 0.0903 | 0.1050 | NEITHER |
| odd-`k` | 0.1822 | 0.1822 | 0.0975 | 0.1356 | NEITHER |

This is the outcome §4 called "the most informative", and it was written down
before any data existed. No single index dominates in either frame. Note the
shape: the lane frame lands *exactly on* its own shuffle floor in all three,
while the chunk frame sits well below it. That is the signature of a frame that
carries information without being determinative, and it is what the
distributional rules below were added to characterise.

NEITHER is not "no effect". Within a process the victim is overwhelming: median
`straggler_share` **0.748** over 476 runs against a chance share of 0.0667, i.e.
**11.2×**. The straggler is real and stable inside one process; what moves is
*which* lane it is, from process to process.

### 5.2 RULE 2 — the lane frame is non-uniform, the chunk frame is not

`max cell / N` can only see "always the same index". It is blind to "drawn from
a biased subset", so RULE 2 adds a chi-square, with a different null per frame:
label-shuffle for chunk, uniform for lane.

| dataset | lane χ² | p | chunk χ² | p |
|---|---|---|---|---|
| pilot | 82.00 | <0.0001 | 22.00 | 0.78 |
| confirmatory | 135.66 | <0.0001 | 29.43 | 0.87 |
| odd-`k` | 79.38 | <0.0001 | 24.55 | 0.57 |

Three independent replications, same direction, and the dissociation is clean:
pooling in the lane frame **preserves** the structure, pooling in the chunk
frame **destroys** it. That asymmetry is only possible if the structure is
anchored to the lane. Confirmatory lane histogram (lane 0…14):

```
[49, 68, 27, 34, 10, 40, 39, 48, 50, 26, 9, 29, 21, 20, 6]
```

Lane 1 (cpu 2) is the modal victim in all three datasets.

### 5.3 RULE 3 — two structural regularities, pre-registered then replicated

Both were read off the pilot histogram and then tested on data that did not
generate them.

| hypothesis | null | pilot | confirmatory | odd-`k` |
|---|---|---|---|---|
| H1 odd lane index | 0.4667 | 0.667 (p=1e-5) | 0.557 (p=5e-5) | 0.585 (p=2e-4) |
| H2 lanes 0–7 (first 32 MiB L3) | 0.5333 | 0.750 (p<1e-5) | 0.662 (p<1e-5) | 0.636 (p=9.5e-4) |

H2 replicates three times: the straggler is materially more likely to be a lane
in the **first L3 domain** (cpus 0–14) than the second, even though placement is
one-per-core and verified.

### 5.4 RULE 4 — a defect in my own arm set, and the fix

While scoring RULE 3 I found that the four original arms are `k ∈ {0,4,8,12}`,
**all even**. Since chunk = (lane + k) mod 16, even `k` preserves parity: an odd
lane always computes an odd chunk. **H1 was therefore frame-ambiguous** in the
first two datasets — they cannot tell "odd lane" from "odd chunk". H2 was never
confounded, because k=4/8/12 move indices across the lanes-0–7 boundary.

Recording it rather than quietly fixing it, and then breaking it with odd `k`
(pre-registered before the odd-`k` run existed), n=236:

| test | null | observed | p | |
|---|---|---|---|---|
| odd **lane** | 0.4667 | 0.5847 | 0.00018 | significant |
| odd **chunk** | 0.5333 | 0.4153 | 0.99989 | not significant |

**Verdict A: the parity effect is lane-anchored.** With the confound broken the
chunk frame does not merely fail to reach significance, it falls *below* its own
null. The chunk/data frame carries no information in any test in this document.

### 5.5 What this closes, and the one thing it does not

The ambiguity §1 was built to break is broken. "Slow lane" and "slow chunk" made
identical predictions in every prior experiment; they no longer do, and the
answer is **lane**. That retires the entire data-property class — cache colouring
of the weight range, page interleave, NUMA placement of that chunk — as the
selector. It is the 8th rejected selector and the first to close a class rather
than a single candidate, and it also converts six earlier "lane index" results
from ambiguous to interpretable.

Positively: the victim is core-anchored, drawn from a biased distribution over
lanes rather than a fixed one, concentrated in the first L3 domain and on odd
lane indices (cpus ≡ 2 mod 4).

**The limit, stated plainly.** CONTROL 2 reports exactly one lane→cpu map, which
is what makes the experiment valid — but it also means this study separates lane
from *chunk*, not lane from *cpu*. "Lane 1 is slow" and "cpu 2 is slow" remain
the same claim here, exactly as "lane" and "chunk" were before #2017. The
instrument that would break *that* is a lane→cpu permutation, the direct analogue
of this one. It is named as the obvious next instrument, not as a hypothesis with
evidence behind it; per standing rule no ninth selector is nominated without
testing it.
