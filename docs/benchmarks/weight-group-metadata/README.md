# Inline derived-weight group control metadata

CPU derived-weight cache groups commonly contain one matrix or an expert's two
or three projections. Store their short control collections inline using the
existing SmallVec dependency instead of creating heap vectors on every hit.
BlockQuantizedMoE also inlines its projection/source/key/size/spec lists.
Capacity three is a storage optimization, not a supported-group limit: larger
groups retain the dynamic path. Actual weight payloads remain separately owned
and exactly charged; they are not moved into the inline metadata.

## Resource and failure contracts

The change preserves one whole-group preflight for missing construction peaks,
zero builder calls after grant refusal, publication only after every builder
succeeds, individual payload grants, active-reader ownership beyond cache
membership, single-flight cold fills and accountant callbacks outside the
payload lock. Warm hits neither rebuild nor call
the issuer. Eviction and pressure retirement containers remain unchanged.

The new four-member behavioral test exercises the dynamic path: 63 bytes cannot
admit a 64-byte group; failure in the last builder refunds the earlier payloads
and publishes no partial group; warm hits preserve Arc identity without issuer
callbacks; readers retain their exact charge after the cache is dropped.
Existing partial-hit, duplicate-key, build-peak, invalid-buffer, pressure,
reclamation and concurrent-reader tests are retained.

## Validation

The exact candidate sources passed CPU lib tests: native 2069 passed, 21 ignored;
MLAS research-reference configuration 2111 passed, 37 ignored. Strict CPU
all-target Clippy passed in both configurations. The test list explicitly
contains the new four-member case. The initial compilation identified a test
assertion comparing SmallVec directly with an array; it now compares the slice,
preserving the same expected statuses. That initial log is retained alongside
the final successful logs. There are no CUDA execution or ISA-wide performance
claims.

## Allocation observation boundary

The diagnostic appends an allocator observer and a test to the actual production
library temporarily. It uses constant, drop-free thread-local counters and
forwards allocation, zeroed allocation, reallocation and free to System with
unchanged pointers/layouts. A positive Box allocation/free control verifies that
the observer is armed. Only the private `WeightCache::load_group` warm-hit loop
and result destruction are observed; key/spec construction, cold fill, assertions
about final accounting and logging occur outside the counted loop.

Each arm tests groups of 0, 1, 2, 3, 4 and 8 members for 2048 warm calls. The old
main revision and candidate use identical probe code. Every warm call verifies
payload values, cache status and Arc payload-owner identity, with no builder activity.
Issuer request/refund counters are unchanged across the loop, and final payload
charge is zero after all readers drop. The dynamic-path groups are positive
controls against accidentally imposing a three-member cap.

This counts allocation/free calls, not elapsed time, bytes, end-to-end throughput,
whole-MoE allocations, or stack cost. It does not measure the five inlined
BlockQuantizedMoE setup lists separately. Routing, activation/result allocation,
weight payload construction, all other ISA/platforms, and GPU behavior are
outside this observation. No inference speedup or universally zero allocation
claim follows from it. The temporary observer is restored out of the production
tree in a finally block, with exact source hashes checked afterward. Drivers
retain original local paths and require adaptation to reproduce elsewhere.

The first candidate removed all observed heap calls for 1–3 members, but the
4/8-member controls rose from two allocations/frees per call to three. This
exposed loss of the spilled buffer during the owning conversion. The final
helper handles inline and spilled storage explicitly: spilled values transfer
through Vec's owning map and back into SmallVec, while inline values collect
without a heap buffer. That conversion is shared by warm hits and successful
cold fills. Buffer reuse is verified for the recorded Rust toolchain, not
promised as a cross-version compiler guarantee. The initial adverse observation
and instrumented sources are retained in `initial-probe`; the unchanged baseline
is reused in the final probe instead of measured again.

On macOS arm64, Rust 1.98.0, the final unoptimized native allocation probe
observed these totals (2048 calls per row, free counts equal allocation counts):

| Group members | Baseline allocations | Final allocations | Issuer request/refund delta |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 1 | 4096 | 0 | 0 |
| 2 | 4096 | 0 | 0 |
| 3 | 4096 | 0 | 0 |
| 4 | 4096 | 4096 | 0 |
| 8 | 4096 | 4096 | 0 |

Both arms' positive controls observed one allocation and one free. All payload
values/owner identities matched their cold group, and all final charges were
zero. Run `python3 -B docs/benchmarks/weight-group-metadata/verify.py .` from the
repository root to verify sealed logs, counts, source snapshots, baseline source
identity and exact restoration against the final validated production sources.
