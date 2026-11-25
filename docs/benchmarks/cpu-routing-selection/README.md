# Exact CPU routing selection: candidate not adopted

Keep the default full ordering in `RoutingScratch`. An experiment partitioned
indices at k and sorted only the selected prefix. It preserved selected indices,
weights and reduction order, but did not establish a universal reduction in comparator calls
or any latency/throughput gain. The final change adds a permutation/boundary
regression test and this evidence; it does not ship the candidate or a threshold.

## Why exactness is possible

The comparator orders by descending `f32::total_cmp(score)` and ascending unique
expert index. This is a strict total order, including score ties, signed zeros
and NaN payload/sign ordering. For 0<k<E, partitioning at index k puts exactly
the original first k indices in the prefix. Sorting that prefix restores their
original order before the selected sum. k=0 skips partitioning, k>=E retains
full ordering, and every subsequent max/exp/full sum/selected sum/aggregation
operation remains unchanged.

The candidate passed native CPU2074 tests (21 ignored), MLAS research-reference
CPU2116 (37 ignored), host block oracles2 and QMoE oracle1, and strict CPU
all-target Clippy in both configurations. The new test exercises permuted f32
bits, explicit signed NaNs/infinities/zeros/ties, all k boundaries for short
rows and 0/1/2/half/E−1/E/oversized for larger rows. Existing reference-bit,
mutable-selection reuse and three-driver tests remain. This establishes the
recorded host's numerical behavior; it is not a CUDA or cross-ISA execution test.

## Comparator-invocation diagnostic

Both methods use the same scratch owner in one native unoptimized executable.
Temporary TLS counters increment once in each score comparator, leaving the
comparison result unchanged. The baseline method is copied from the exact main
source and renamed only for the diagnostic. The candidate's actual comparator
is instrumented, not replaced by a separately implemented selection algorithm.

The fixed grid contains E=0,1,2,8,64,256,1024 and four input patterns: deterministic
finite random, descending, all ties and a repeating nonfinite/zero/extreme set.
k values are the deduplicated set {0,1,2,E/2,saturating(E−1),E,E+3}. There are
160 cases, including helper boundaries that the kernel's k validation rejects.
No timing/CPU clock or allocation count was measured. This counts comparator
invocations in the recorded build; it is not proportional to total CPU work by
definition and must not be interpreted as latency or end-to-end throughput.

Every candidate selection is checked by index and f32 bits against both the
frozen borrowed method and the existing independent allocating reference.
One complete candidate output per case is retained; source assertions witness
equality to the other two outputs. The reader independently reconstructs all
input score bits and checks total-order prefix identities. Full-sort controls
(E>=2,k>=E) must have equal positive counts; k=0 candidates must make zero
comparisons. Instrumentation and all reference copies are absent from the final
production file.

| E=256 input | k | Full ordering comparisons | Partition + prefix ordering |
| --- | --- | --- | --- |
| random | 1 | 2096 | 622 |
| random | 128 | 2096 | 1478 |
| random | 255 | 2096 | 2453 |
| descending | 1 | 255 | 638 |
| descending | 128 | 255 | 1512 |
| descending | 255 | 255 | 509 |
| all ties | 1 | 255 | 638 |
| all ties | 128 | 255 | 1512 |

Across the complete grid, the candidate uses fewer comparisons in 39 cases,
the same in 80 and more in 41. These are case counts, not a probability model for
real routes. At identical E/k, random versus ordered/tied scores point in
opposite directions. These data do not justify choosing the candidate from
E/k alone as an improvement across both inputs. This does not prove a latency
regression; it does reject an unconditional claim that partitioning always
reduces comparator invocations.

## Adoption decision and final verification

Do not add a shape/data heuristic or replace the default without a bounded
release latency study and evidence that routing matters in the target execution.
No runtime switch, pool, quota operation, compatibility interface or private
candidate path is shipped. The new regression test remains to support future
native optimization. Source comparison proves that removing only this test from
the final file reproduces the base main file exactly.

The candidate's complete functional validation is retained separately from the
final test-only source validation. Full original comparator logs, 160 complete
outputs, frozen/candidate/instrumented source snapshots and executable hashes
are kept. Reproduction drivers retain original local paths and need adaptation;
all heavy work used the shared host flock, coordinated with the peer chat. The
single diagnostic attempt is retained in full; no case or adverse result was
removed. No CUDA execution or inference speedup is claimed.

The final default-plus-regression-test file passed all three routing scratch
tests in native and MLAS-reference configurations, and strict CPU all-target
Clippy in both configurations. This final validation is distinct from the
candidate's broader functional matrix. Run
`python3 -B docs/benchmarks/cpu-routing-selection/verify.py .` from the repo root
to verify every sealed payload, independently reconstructed total-order prefix,
full-sort controls, instrumentation edits, candidate/final checks and exact
production-code preservation. Omit the checkout argument to verify historical
evidence after future source changes.
