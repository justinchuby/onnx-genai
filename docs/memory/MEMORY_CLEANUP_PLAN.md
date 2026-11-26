# Memory contract cleanup without CUDA hardware

Status: source cleanup completed. GPU-dependent redesign remains deferred.
This plan covers source and host-verifiable cleanup only.

## Scope and order

1. Remove the governor's compatibility allocator module and mechanism re-exports.
   Import mechanism contracts from `onnx-runtime-memory-api` everywhere, including
   tests, examples, manifests and documentation. Keep budget contracts in the governor.
2. Rename the reservation-level `VirtualBacking` trait to `ReservationBacking`.
   Keep the allocator capability named `VirtualBacking`; update implementations,
   callers and documentation without an old-name alias.
3. Move weight-paging diagnostic and experimental configuration into a private
   configuration module. Preserve environment names, defaults, parsing, lazy
   initialization and call sites. Move the relevant host-only tests with it.
4. Correct current-architecture documentation: the built-in CUDA allocator is VMM;
   legacy policy flags do not restore the removed eager allocator. Document the
   mechanism/capability distinction and the verification boundary.
5. Run formatting, relevant host tests, CUDA source compilation and workspace
   compilation where supported. Review the complete diff for accidental behavior
   changes, commit the result, and open a PR with exact validation evidence.

## Invariants

No changes to allocation/adoption order, physical-handle pooling, map/unmap,
budget grants/refunds, eviction selection, stable weight addresses, KV growth,
CUDA Graph lifetime, or deferred-release ordering. Keep existing correctness,
context/lifetime, partial-failure and quarantine guards.

## Deferred until real CUDA validation is available

- Construct the final memory authority before any device allocation and reduce
  startup-ledger adoption states, preserving external EP integration contracts.
- Bind charging behavior to the selected allocation domain rather than callers.
- Consolidate resource ownership and release settlement across memory holders.
- Separate dense streaming and MoE execution policies; evaluate CPU cold experts.

These changes require real-device tests for KV preservation across growth,
CUDA Graph capture/replay with stable addresses, compute/copy stream ordering,
partial map/release failure, physical versus mapped accounting, budget pressure
and token equivalence. CPU tests and compilation do not establish these properties
or performance on a CUDA device.

## Validation and self-review

Completed source work:

- Removed governor mechanism exports and migrated imports/manifests across callers.
- Renamed the reservation trait and its implementations/tests to `ReservationBacking`.
- Extracted paging configuration into `weight_paging/config.rs`, keeping per-knob
  lazy initialization and the dynamically read refill cadence. Four existing
  parser/default tests moved with their code. Public eviction-probe API paths
  remain in `weight_paging`; `config` is a private implementation module.
- Corrected current allocator-policy descriptions, separated the two backing
  contracts, and repaired documentation links discovered by strict rustdoc.

Initial validation on macOS arm64, without CUDA hardware:

| Command | Result |
|---|---|
| `cargo test --offline -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory --no-fail-fast` | Passed: 188 unit/integration tests, plus four doctests (one import example and three removed-path checks); no ignored tests |
| `cargo clippy --offline -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory --all-targets -- -D warnings` | Passed |
| `cargo check --offline -p onnx-runtime-cuda-memory --lib` | Passed; driver is dynamically loaded, not exercised |
| `cargo check --offline -p onnx-runtime-cuda-memory --lib --features gpu-tests` | Passed; compiles the feature's library code, does not execute GPU tests |
| `RUSTDOCFLAGS='-D warnings' cargo doc --offline --no-deps -p onnx-runtime-memory-api -p onnx-runtime-memory-governor -p onnx-runtime-virtual-memory -p onnx-runtime-cuda-memory` | Passed |
| `cargo fmt --all -- --check` and `git diff --check` | Passed |
| Four extracted configuration tests, compiled from the production source as below | Passed; test listing confirmed all four names, execution reported four passed |
| `cargo check --offline --workspace --all-targets` | Blocked before completion by the existing ORT 1.29.0 dependency download; network-enabled retry returned HTTP 404 for the macOS arm64 archive |
| Full CUDA EP test enumeration and all-target/gpu-tests compilation | Also blocked by the transitive `onnx-genai-ort-sys` dependency; not claimed as passed |

The configuration module uses only `std`, so its actual source and existing tests
can be checked independently of the EP's ORT dependency. From the repository root:

```sh
printf '#[path = "%s/crates/onnx-runtime-ep-cuda/src/weight_paging/config.rs"]\nmod config;\n' "$PWD" > /tmp/memory-config-host-tests.rs
rustc --edition=2024 --test -D warnings -A dead_code /tmp/memory-config-host-tests.rs -o /tmp/memory-config-host-tests
/tmp/memory-config-host-tests --list
/tmp/memory-config-host-tests
```

`-A dead_code` applies to this standalone harness because production callers live
in the EP, outside the harness. No parser or GPU implementation is replaced with
a test stub.

Self-review completed before publication:

- Compared executable text for 99 migrated Rust files and the paging
  implementation, normalizing imports, the intended trait/path rename and
  optional trailing commas. No other executable changes were found.
- Compared the extracted configuration function bodies, defaults, local statics
  and all four moved test bodies with the base revision; they are preserved.
- Reviewed governor export removal, dependency declarations and documentation
  separately. Added compile-fail doctests for both removed governor paths and the
  old reservation-trait name; existing allocation/reservation tests remain.
- Allocation/adoption, map/unmap, pooling, grants/refunds, eviction, capture and
  release algorithms are unchanged. Full EP/workspace compile coverage was
  initially incomplete because ORT 1.29.0 was unavailable; see the follow-up
  verification below for the resolved checks.

No throughput or GPU-runtime correctness claim is made by this cleanup.


## Follow-up: PyPI runtime update and full compilation (2026-11-20)

The user authorized upgrading ORT to PyPI's latest stable release. Both
`onnxruntime` and `onnxruntime-gpu` report 1.32.0. Automatic native bootstrap now
uses SHA-256-pinned official PyPI wheels and the matching Microsoft C API 32
headers from NuGet, bundled with license and provenance. GitHub availability no
longer blocks default bootstrap. Python package floors and the producer CI pin
are updated to 1.32 as well. See the [bootstrap documentation](../../crates/onnx-genai-ort/ort-sys/README.md).

Full compilation exposed missing imports/visibility for the extracted
`PIN_PROBE_GRANULE_BYTES`, eviction-order parser and test refill environment name.
Those are corrected without changing their values or behavior. ORT 1.32's new
optional factory capacity callback is left null, as permitted by the upstream
contract: the generic EP cannot report device-global capacity.

| Follow-up command | Result |
|---|---|
| `cargo check --offline --workspace --all-targets` | Passed; existing unrelated unused-import/dead-code warnings remain |
| `cargo check --offline -p onnx-runtime-ep-cuda --all-targets --features gpu-tests` | Passed; compilation only, no CUDA device |
| `cargo test --offline -p onnx-genai-ort-sys` | Passed: seven bootstrap tests and one real-library C API roundtrip |
| `cargo test --offline -p onnx-genai-ort --test runtime_upgrade` | Passed: actual CPU Identity inference, without optional external fixtures |
| `cargo test --offline -p onnx-runtime-ep-plugin --lib` | Passed: 384 tests, none ignored |
| `cargo test --offline -p onnx-runtime-ep-cuda --lib weight_paging::config::tests::` | Passed: all four named tests, first confirmed with `--list` in the real EP crate |
| `cargo clippy --offline -p onnx-genai-ort-sys --all-targets -- -D warnings` | Passed |

Network access was enabled for commands whose build scripts needed to populate
new artifact caches; Cargo dependency resolution remained offline. Source checks
and CPU execution are now verified; CUDA device behavior remains deferred.

Follow-up self-review matched all five wheel URLs and digests against official
PyPI metadata, and checked all three bundled headers byte-for-byte against the
verified Microsoft NuGet package. Bootstrap rejects digest mismatches and unsafe
archive paths before extraction. The EP capacity callback remains explicitly
unsupported; no device-global capacity is inferred from a process budget.

## Pre-merge self-review

Reviewed the complete PR again before merging, including allocation-contract
imports, reservation trait users, configuration initialization, artifact hashes,
ZIP extraction, cache completeness, loader version selection and API 32's optional
factory callback.

- Fixed the runtime smoke test to select CPU explicitly. A default EP could be
  influenced by ambient GPU/MLX settings and would not prove CPU-only execution.
- Scoped non-macOS EP test imports/helpers to their test platform and removed a
  test-only forwarding import. This resolves the strict macOS lint findings
  without changing runtime selection or skipping any runnable test.
- `cargo clippy --offline -p onnx-genai-ort --all-targets -- -D warnings` passed.
- `cargo test --offline -p onnx-genai-ort --test runtime_upgrade --test execution_provider`
  passed: CPU inference and CPU-provider enumeration both executed successfully.
- Publish-order, documented-environment-variable, documented-command and
  feature-gate checks passed. Existing GPU-dependent validation remains deferred.

Merge is conditional on the required checks for the final reviewed commit;
branch protection is not bypassed.


### CI follow-up review

- Fixed a real bootstrap regression: PyPI wheels contain versioned Unix runtime
  files only. Publish both major and unversioned relative links, including repair
  of already-populated caches; regression coverage exercises Linux/macOS names,
  idempotence, incomplete-cache errors and old-cache repair.
- Full CPU ORT integration execution exposed three unary assignment tests missing
  the shared plugin lock. Add the lock at their test boundaries so rejection
  tests retain their zero-compile/zero-execution counter assertions.
- ORT 1.32 now executes the 512-wide quantization block previously rejected by
  older runtimes. Require numerical parity for this case and correct the docs.
- Remove Windows ARM64's stale Visual Studio 2022 generator override; retain the
  native build script's existing Ninja/installed-toolchain discovery.
- Validation: eight bootstrap tests plus one real C API test; strict sys Clippy;
  all 65 runnable CPU plugin ORT integration tests passed with
  `NXRT_REQUIRE_ORT_TESTS=1` (one pre-existing ignored test). Workflow integrity,
  required-tier coverage and whitespace checks passed. Windows verification
  remains delegated to the actual CI runner, not inferred from this Mac.

The repo-wide fixture guard caught opset 21 in the new Identity smoke fixture.
Updated it to the required opset 24, retaining IR 11. Validate the actual CPU
inference and the complete fixture guard together before the final CI rerun.


Expanded local ORT-backed execution exposed a pre-existing host assumption in
`append_chain_error_reports_the_mid_chain_provider_that_failed`: it passed a null
session-options handle while assuming CoreML was unavailable. The official macOS
wheel advertises CoreML, so the test entered the real C API and crashed. Replace
that setup with owned valid CPU session options and a deliberately unsupported
second provider. Preserve the chain index/name/source assertions without relying
on accelerator availability. Complete ORT/server tests now pass: 612 passed,
zero failed, two existing ignored tests across 17 suites; strict ORT all-target
Clippy also passes. No runtime algorithm changed in this follow-up.


## Follow-up: shared weight contracts and external design comparison

The CPU EP's `weight_offload::weight_handle` module was only a compatibility
re-export of `onnx-runtime-ep-api`, with a second re-export at the CPU crate root.
Both surfaces are removed. Shared weight negotiation, leases, binding and provider
capabilities are owned by `onnx-runtime-ep-api`; consumers import them there.
The existing executor and CUDA consumers already use that authority. No allocator,
budget, placement, kernel or release behavior changes in this removal.

Upstream comparison, inspected at fixed revisions:

- [Linger `fb6812d`](https://github.com/simplexai-labs/linger/tree/fb6812dcfdf079c76c3cf9a2a500e0697bab3c45)
  is the application layer; its README identifies PolyStrata as the inference
  engine. Application descriptions and reported performance are not runtime
  validation for nxrt.
- [PolyStrata cache `3e16421`](https://github.com/simplexai-labs/PolyStrata/blob/3e1642161198c50ff1abed4046f4e4f476099dbb/src/core/expert_cache.cpp)
  has an optional CUDA VMM path (`cuMemAddressReserve`, `cuMemCreate`,
  `cuMemMap`, `cuMemSetAccess`) and a `cudaMalloc` fallback. Its design therefore
  does not justify removing our VMM. Mapping policy and allocation mechanism
  remain separate decisions.
- [PolyStrata CPU expert interface](https://github.com/simplexai-labs/PolyStrata/blob/3e1642161198c50ff1abed4046f4e4f476099dbb/src/poly/cpu_experts.hpp)
  uses ggml CPU quantized dot products, groups tokens sharing an expert, and
  exposes overlap/readiness callbacks. These are distinct capabilities; a host
  implementation does not establish GPU overlap correctness.

| Capability | Current nxrt position | Next integration boundary |
|---|---|---|
| Reuse an expert across a token batch | CPU `BlockQuantizedMoE` already groups routed tokens and resolves each projection once per expert group | Preserve grouping and numerical tests; do not add a parallel batching authority |
| Compute cold experts from compressed weights | CPU `BlockQuantizedMoE` decodes rows when a complete expert f32 projection set exceeds the existing dense-cache budget; fitting sets retain dense GEMM/cache | All 15 formats reuse canonical decoders; this bounds decoded-weight scratch per call. Quantized-domain dots and GPU overlap remain separate work |
| Use CPU and GPU experts within one layer | Static engine placement is advisory; native sessions still own one EP | Requires an executable partition and synchronization contract, plus real-device numerical and failure tests |
| Overlap expert loading and compute | CPU QMoE has a guarded route-first prefetch path; CUDA has separate stream/fence and residency authorities | Reuse their ownership and budget guards; do not copy callbacks into a second unaccounted scheduler |
| Choose a memory tier | Governor owns capacity; placement chooses residency within its grant | Keep byte budgets and explainable eligibility canonical; use measured cost-model rates rather than model or GPU identity guesses |

This comparison adds no upstream code dependency and makes no throughput claim.
The real-CUDA validation requirements above still apply to changes in mapping,
GPU/CPU execution splits, capture lifetime and asynchronous release ordering.


### CPU bounded expert working set

`BlockQuantizedMoE` now admits dense execution only when the complete selected
expert projection set (including an optional third projection) fits
`ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES`. Otherwise it decodes a bounded tile per
Rayon shard and reuses it for the expert's tokens. Portable native compute uses
up to four rows; the existing AVX2/MLAS routes retain one.
Routing, activation, gates and accumulation share the existing implementation;
format metadata and the canonical decoder remain the only format authority.
Zero budget selects streaming without hashing constant identities or building
dense cache entries. Contiguous packed weights are borrowed; strided inputs copy
only the selected expert's compressed bytes. Mutable inputs are read each call.

For a projection with input width K and N output features, at most
`min(Rayon workers, N) * (FEATURE_TILE * K + 256)` f32 values are live as
decoded-weight storage. FEATURE_TILE is four for portable native compute and one
for AVX2/MLAS; actual allocations shrink to each shard's feature count. This
remains call-owned scratch, not a public-workspace or complete process grant.
This excludes grouped activations, output-sized intermediate storage, compressed
strided copies and other concurrent calls. The cache budget is an eligibility
rule, not a process-wide working-memory grant; evicted Arcs can still be held by
other calls. No second memory governor or environment knob was added.

[CPU measurement and reproduction](../benchmarks/CPU_BLOCK_MOE_ROW_STREAMING.md)
records the measured memory/time effect and unchanged warm-cache control. This
is streamed f32 compute, not ggml integer-domain dot products or LLM throughput.

The measurements also exposed an independent startup cost: the existing host
allocation-cache floor calibration touches approximately 512 MiB when the floor
is not configured. Every measurement arm explicitly sets the same 1 MiB floor
to isolate expert memory. Startup calibration remains unchanged and should be
addressed in a separate allocator-lifecycle change.


### Follow-up: defer host-cache calibration until a retention decision needs it

`LargeAllocCache` construction captures explicit configuration and otherwise
leaves its automatic floor pending. The existing process-wide calibration stays
the sole measured answer. Disabled retention, invalid layouts, requests above
the maximum and requests below every possible automatic floor skip calibration.
The lower bound comes from the same probe ladder and fallback used to resolve
the floor; explicit configured floors, including smaller values, remain honored.

`floor_bytes()` now returns `Option<usize>` without performing a probe.
`resolve_floor_bytes()` explicitly permits calibration. There is no old-signature
shim. Pending/resolved state is encoded with `OnceLock`; automatic resolution
still runs once per process, and initialized instances take the resolved path.
Exact layouts, retained-byte admission, counters, poisoning, drain and concurrent
ownership guards stay in place. Zero-byte layouts are explicitly ineligible even
with a configured floor of zero, matching the documented allocation contract.

[Startup process evidence](../benchmarks/HOST_CACHE_LAZY_FLOOR.md) compares the
unchanged cache implementation with the lazy path and an explicit-floor control.
This avoids needless construction/small-request work; it does not remove the
probe from eligible large workloads or turn a retention cap into a peak-memory
guarantee. The separate global grant and active-allocation work remains open.


### Bounded standalone block projections

`BlockQuantizedMatMul` now applies the same dense-set eligibility rule as the
expert path: use full f32 expansion/GEMM only when the complete matrix fits its
existing cache budget. Otherwise reuse `linear_packed_grouped`, including all
format decoders and the inner serial/MLAS scheduling contract. No new allocator,
cache, format table or scheduler is introduced. Empty batches avoid dense
expansion while keeping argument validation. Fitting constant matrices retain
their existing cached fast path. The per-kernel cap and decoded scratch bound
remain distinct from a model-wide governor grant.

The block projection and MoE tests share one packed-format fixture. IQ1_M now
uses a finite nonzero scale so it exercises arithmetic rather than zero weights.
Half-storage output comparisons decode the actual dtype rather than interpreting
pairs of half values as f32 words. Existing cache/format guards remain intact.


### Owned QMoE host-cache payloads

Cache retention is an immutable policy ceiling, not a standing governor lease.
Each retained or temporary complete expert owns an exact grant from the same
Host source as block expansions. The grant survives cache eviction/provider
teardown while an active `Arc` still owns the expert, and returns only after
payload destruction. An empty cache consumes no Host capacity and can coexist
with host KV that was allocated before any expert is expanded.

`CpuMemoryResources::new(source, limits)` seals the cache and block/QMoE grant
source together. Its private fields prevent foreign-source cache pairing.
`CpuWeightCacheLimits` holds dense/expert retention ceilings, and the consuming
`with_cache_retention` builder captures the legacy planned-family verdict.
Standalone defaults are ungoverned; engine components clone their engine-owned
resource handle. The unused late-adoption API and the separate cache holder have
been removed without aliases. The optional environment ceiling is resolved once
at the cache's first operation; invalid configuration remains a typed error.

Cold publication has a single fill gate. Policy selection and publication hold
the short payload mutex; governor callbacks, dequantization and retired payload
drops do not. Ready hits skip the cold gate and issuer. This removes the old
adoption mutex, budget `RwLock` and operation-lifecycle `RwLock` while preserving
single-flight fills and the bank registry's independent prefetch proof.

Tests cover idle-zero ownership, KV competition and grant refusal, exact partial
refunds, active owners after cache drop, concurrent same-key fills, hard-failure
rollback and observers on reserve/refund. The existing frequency/pinning,
foreign-bank, routing, numerical and prefetch-statistics guards remain.


### Per-tier authority identity

`MemoryGovernor::authority_id(tier)` resolves the same books as a lease or
mapped allowance for that tier. The engine composes shared Device books with
worker-group-local Host/Disk books; a single Device identity incorrectly made
separate Host partitions look interchangeable when engines shared a device.
CPU weight-source anchors validate Host identity, virtual buffers compare their requested
tier, and process registration uses the declared device key's tier. CUDA pools
validate their context against Device identity, while sharing/admission/reclaim
use the physical location's tier: Device for VRAM and Host for Host-NUMA. All governor implementations and callers
use the explicit tier; no old-signature shim is retained. Regressions reproduce
the original identity/lease mismatch and independent Host books behind a shared
Device authority, then check clones and mapped allowances. This fixes
identity routing, not the separate physical-capacity relationship between CPU
hot-tier and Host books.


### Shared-cache prefetch capacity proof

The layer-only bound `experts * expanded_bytes <= budget` was insufficient:
foreign-layer payloads can occupy the rest of the shared partition. A reproduced
case admits the next expert in serial execution by evicting the current expert,
while prefetch's live `Arc` prevents that victim and changes the retained set.

A routed operation now registers its complete expert bank using the canonical
source/layout key and expert cardinality. Active operations and retained entries
hold that bank's metadata; the partition keeps weak registrations and prunes dead
ones on cold registration. Positive-budget prefetch requires the sum of every
live bank's complete expanded set to fit. A shared registry read guard freezes
that upper bound through compute and worker completion: new banks wait, while
operations on known banks remain concurrent. Workers borrow the existing proof
and never reacquire either reader lock behind a pending writer. Zero-budget
streaming needs no registry lock, and retains its single-ahead pipeline.

This is deliberately a conservative eviction proof, not a new memory budget or
a scheduler. Existing source identity, admission frequencies, pinning, exact
output and counter tests remain. Tests cover the foreign-bank counterexample,
exact shared-capacity boundary, retained metadata, expiry/pruning, concurrent
known-bank operations, cold-registration exclusion and worker progress. Full expert payload grants now share the Host source; row scratch and other
allocation classes still need model-wide grant convergence.


### Owned grants for block-quantized CPU derived weights

Native CPU registry construction now consumes one `CpuMemoryResources` bundle:
QMoE host-cache ownership, block-derived Host grants, the per-kernel dense
retention ceiling and the immutable CPU cache-retention verdict. This replaces the
combinatorial provider constructors; all engine and pipeline callers use the
same resource bundle without compatibility aliases. The engine constructs the
Host/Weights source before compiling kernels and shares it across components.
Native decode CUDA also receives the composite governor so Host-NUMA grants use
its Host books rather than the Device-only authority.

BlockQuantizedMatMul and BlockQuantizedMoE distinguish retention policy from
physical payload ownership. A zero-byte anchor supplies sibling grants without
reserving an unused standing allowance. Missing projections are admitted as one
group before any dense allocation, then `MemoryLease::split_off` transfers exact
shares into each buffer without changing the ledger. Partial eviction returns
only the dropped buffers' shares; active owners remain charged after eviction.
Nonconstant dense expansions use the same grants. Capacity refusal uses bounded
row decoding; allocation/configuration errors preserve their typed causes.

A separate cold-fill gate preserves single-flight reuse while grants, decoding
and releases happen outside the cache payload lock. MoE decodes directly into
its final NK layout, avoiding an uncharged complete transpose temporary. Warm
cache hits do not invoke the accountant. Tests exercise shared Host pressure,
complete-group refusal, partial replacement at exact quota, active owners after
LRU eviction, failure rollback, concurrent fills and numerical equivalence
across formats, dtypes, mutable/constant weights and packed storage layouts.

This is a scoped convergence step. Other derived caches, scratch/activation buffers, the physical
relationship between CPU Device and Host books, process-wide capacity and GPU
asynchronous lifetimes remain open. Default standalone CPU resources remain
ungoverned. CPU tests and CUDA compilation do not prove CUDA execution behavior.


### QMoE expert working sets and prefetch

Complete FC1/FC2/(optional FC3) sets acquire their Host grant before allocation.
Cache fills and transients share the sealed resource source. The same owned grant
remains with a retained payload, so there is no standing cache entitlement or
double charge. Capacity refusal chooses NBits row decoding through the shared
per-shard projection driver; allocation/configuration errors retain typed causes.

Streaming-only prefetch obtains a scoped grant for the driver's two simultaneous
temporary experts. Positive-cache prefetch pins initially selected hits, obtains
one admission grant for the complete missing selected set and splits exact
payload shares. A retained share survives on its own; consumed transient shares
and unused shares return independently. The scoped admission bound may exceed
the currently allocated dense payloads and is not an RSS measurement. Failure to
preflight keeps the existing serial path, rather than changing representation
only because an extra overlapping payload is live. Fully cached operations keep
the prefetch path without querying the issuer.

Real mmap fixtures verify exact serial/prefetch outputs and cache statistics,
Host-book peaks at quota boundaries, complete-set refusal, all activation/gates,
partial selected-set refunds and warm issuer bypass. Other derived caches,
scratch/activation accounting, CPU Device/Host physical-capacity relationships,
process-global capacity and asynchronous GPU validation remain open. Host tests
and CUDA compilation do not establish CUDA runtime behavior.


### Provider-scoped CPU cache retention

`CpuMemoryResources::with_cache_retention` carries an immutable `CpuCacheRetention`
verdict captured before registry construction. MatMul, Gemm, FusedMatMulBias,
MatMulNBits, QLinearMatMul and Einsum use it for the cache families already
included in the memory plan. The six process-global admission setters have
been removed. Loading a second model cannot alter the first model's decisions.
Standalone providers default to admitted retention. ISA and compact native
packing eligibility remain governed by their existing format/shape contracts.

Session builders accept an initialized CPU `fallback_execution_provider`.
CUDA whole-session fallback, heterogeneous partitions, plugin bridges,
control-flow child plans and decode/verify siblings carry that configured
provider instead of recreating default CPU resources. Non-CPU fallback
providers fail with an actionable configuration error. Native engine and
workflow callers supply their configured provider; standalone sessions can
still auto-detect a default CPU provider.

This is admission policy, not a new physical quota or grant. Existing shared
transpose/MLAS stores and process scratch ceilings remain shared. Block-derived
and transient QMoE payloads retain their separate owned Host grants. Remaining
work includes actual grants for other legacy packed caches, and the CPU Device/Host physical-capacity relationship. No GPU
execution or throughput claim follows from these changes.

Load-time constant preparation is a provider hook over immutable host operands,
independent of unresolved activation shapes. Its returned state belongs to the
executor. CPU transpose warming uses the same prepack policy and memo mechanism
as concrete kernels, so both static and symbolic graphs retain their load-time
warm path and release its ownership with the session.


### NBits f32 expansions join owned dense weights

`MatMulNBitsFactory::new` captures `CpuMemoryResources`, rather than only a
retention boolean. Every complete f32 expansion, including nonconstant and KN
transient operands, uses the same generic owned-dense cache component as block
formats. Cache keys remain representation-specific; the NBits key is local to
one immutable kernel slot and is never a shared raw address. A retained payload
keeps its exact Host grant through its last Arc, including after kernel drop.
Idle cache ceilings reserve nothing. Cold fills are single-flight; warm hits
claim no new capacity, and disabled/oversize retention does not serialize
transient operations through the cache fill gate.

The common per-kernel f32 retention ceiling is `CpuWeightCacheLimits::dense_bytes`
(default 256 MiB, read from `ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES`). The old
block-only environment knob is removed without an alias; benchmark commands and
current documentation use the common name. Retention and execution grants remain
distinct: NBits may use an admitted temporary full expansion even when retention
is off. Existing packed native and fused GEBP routes keep their legality and
priority, with the NT retention decision derived from the actual ceiling.

A resolved Local/NUMA/SPMD placement owns the allocation-peak rule and the actual
placement operation. NUMA/SPMD first-touch copies need two simultaneously live
f32 matrices, so construction preflights 2B and leaves only B charged to the
final payload. The shared cache rejects underspecified peaks, nonexact final
buffers and failed builders without publishing a partial entry or leaking a
grant. This is a weight-buffer admission bound, not an RSS measurement.

Full-grant capacity refusal selects the shared row-decoded projection driver.
NBits retains its serial M=1 dot order; batched projection uses serial per-column
GEMM inside bounded worker shards. The driver accepts an explicit scheduler;
NBits M=1 reuses its deferred flat/NUMA/SPMD dispatch rather than bypassing the
configured budget with ambient Rayon. The isolated SPMD parity fixture now
includes a refused dense grant and asserts that it actually reaches the active
scheduler. Unit output axes avoid a redundant layout copy. One affine decoder
serves NK, KN and refused
grant paths for 2/4/8 bits and group indices. Contiguous packed/scales/zero-point
operands borrow independently, including offsets; only strided or widened
operands materialize. The byte borrowing helper is shared with QMoE.

Host tests cover actual KV competition, independent source/retention policy,
active payloads after kernel drop, construction-peak refunds, refusal before a
full builder, rollback on invalid builders/group indices, all bit widths and
zero-point modes, strided operands, lower precision, bias and multiple rows.
This does not finish compact native/MLAS packing grants, external pressure coordination,
row scratch/activation accounting, CPU Device/Host capacity reconciliation,
process-global capacity or asynchronous GPU validation. Standalone default
resources remain local and ungoverned. No throughput or GPU execution claim.

The NBits planning exports are now `nbits_decode_f32_retention_candidate`,
`planned_nbits_node_retained_bytes` and `planned_nbits_retained_bytes`; engine
callers and fixtures use those names without old aliases. Their comments state
the estimate's limits: compact native packs, transients, scratch and runtime
shape-instance counts are not included. They never authorize physical bytes.
The optional MLAS Apple route test also checks the resolved platform policy,
including the native int8 fallback, rather than assuming linkage guarantees
QNBit/KAI execution. The unchanged main branch reproduces its former failure
in an isolated build directory; the original numerical guard is retained.


### Idle derived-weight reclamation through the physical authority

A governed `DerivedWeightMemory` keeps its zero-byte Host anchor and weak
registrations for the dense and shared QMoE expert caches. Its responder enrolls
with the supplying governor's optional authority/tier `PressureRegistry`; the
registration keeps neither the callback nor its authority alive. Source clones
share ownership of this enrollment. `supports_pressure_reclaim()` distinguishes
a funded source with this capability from a third-party authority that only
funds payloads. Lack of this optional capability preserves normal grants and
bounded refusal, without inventing a second CPU admission policy.

Ordinary `reserve`, existing lease `grow`, and replacement `reserve_sibling`
requests now enter the same physical-accounting admission. On a real refusal,
the ledger drops its claim gate, snapshots weak responders in rotating order,
and makes one bounded pass for the reported shortfall. A holder releases idle
whole objects under its own policy, outside registration/data locks. Admission
retries the actual books even when a callback reports zero; a claimed release
never authorizes allocation by itself. Oversized requests above the reported
total ceiling and hard arithmetic errors cannot flush the caches pointlessly.
Same-authority/tier recursive pressure declines rather than recursing; a later
round remains enabled in release as well as debug builds.

The CPU responder retains LRU/frequency policy and protects active `Arc` readers,
partial-hit projection groups and selected-expert prefetch pins. Ordinary QMoE
frequency pins guide retention, but cannot make an inactive payload immune to
physical Host pressure. Different CPU sources explicitly enrolled with the same
physical Host authority can fund one another and KV growth. Distinct authorities
or tiers cannot reclaim one another's idle weights. Cache membership/eviction
metrics remain separate from the owned grants that return only on the final
payload owner. Warm hits enter neither admission nor pressure.

Process allocation transactions use that same registry when the governor exposes
it; they do not pre-drive or repeat a pressure-capable authority's bounded
physical admission. Ungoverned/third-party allocator contexts can still use their
existing process-level callbacks. Process-quota shortage is a separate constraint.
`PressureResponder` now returns a typed result. CPU callback failures retain their
`EpError` cause through `MemoryError::ReclaimFailed`; failures cannot be relabelled
as ordinary capacity. A failed process transaction refunds its new quota before
returning that cause. Existing KV ownership is unchanged on failed growth.

This is synchronous, bounded admission pressure, not the scheduler's ticketed
host arbitration protocol or a complete process-wide RAM quota. CPU Device/Host
physical-capacity reconciliation, cross-engine host-book unification, compact
native/MLAS owned-pack admission, scratch/activation accounting and process-wide
coverage remain work. CUDA callbacks/runtime lifetime have not been exercised on
this host. No throughput, RSS or NUMA-performance claim is made.


### Owned native NBits preparation

All eight native NBits cache slots now use the same `DerivedWeightMemory`
source as dense expansions and QMoE experts. A `NativePackedWeight` payload type
fixes its physical layout and construction recipe; admission cannot pair an
expanded/tiled payload with a cheaper wire-layout charge. The six layouts cover
wire bytes, direct int4, expanded int8, direct u8, N16 SDOT and KAI qsi4/qsi8.
Runtime geometry determines padding, values and metadata sizes with checked
arithmetic. Invalid dtype, inaccessible Host operands, unsupported layout
geometry and size overflow fail before payload preparation.

Before allocating, the authority grants a bound for the entire transformation:
input gathering, portable half-scale widening, simultaneous input/output buffers,
tile-padded scales/sums/zero-point offsets and first-touch placement overlap.
Once construction destroys its temporary inputs, the prepared value keeps only
its exact payload grant. Data fields drop before returning that grant. Dynamic
preparations release at call end; retained constants release at kernel drop.
Capacity refusal cannot allocate an uncharged copy. All native representations
share one cold-fill admission mechanism, with warm lock-free cache reads and a
single gate preventing duplicate retained preparations. Pressure responders do
not take that cold-fill gate.

Unchanged contiguous Host Uint8/Float32 wire operands still borrow directly,
including with zero Host headroom. Numeric accumulation, ISA selection,
quantization and placement algorithms are unchanged. Tests compare the actual
six payload allocations with their charged bytes over tail/tile geometries,
scales dtypes and symmetric/affine inputs, including failure/refund and warm
reuse under a full Host budget. Real u8 decode and wire 2-bit execution exercise
the governed factory path; raw transformed-layout builders also execute on hosts
without their corresponding instruction-set route. That does not prove x86 ISA
execution or multi-node NUMA placement on this Apple host.

The optional MLAS pack admission is described below. Scratch/activation
allocations still need their own integration. Retained native OnceLock entries
are not yet idle-reclaimable.
Physical Host/Device book reconciliation and process-wide limits remain work.
No speedup, placement performance or CUDA runtime claim is made.


### Owned MLAS SQNBit payloads share Host admission

The optional reference route now obtains actual `DerivedWeightMemory` grants
before allocating standard, N-sharded or imported-prepacked SQNBit weights.
A resolved input plan carries borrowed contiguous Host bytes/f32 scales or
explicit gather/widen work. The grant covers the real packed allocation and
all retained scale/zero-point metadata, plus any overlapping owned input and
portable half-widening buffers during construction. Contiguous wire inputs do
not require a second whole-weight copy. Scale/zero-point storage in `mlas-sys`
is boxed to make the owned payload size exact, including on CompInt8, which
currently retains these copies too. Each packed object keeps its grant until
the final Arc owner drops, after data and metadata have been freed.

Shared-store keys include the physical Host authority, wire versus serialized
source encoding, and scale dtype in addition to the existing operand identities
and quantization geometry. Independently constructed sources/holders on the
same books can pay once. A different authority or explicitly ungoverned source
cannot reuse that paid allocation. The source snapshots its governor identity
even when custom lease accounting has no optional stamp; `host_authority_id()`
exposes this provenance. Weak per-key cold-fill gates prevent duplicate packs
under a one-payload budget, without serializing unrelated keys or locking value
maps during builders. Error/capacity refusals are retryable. Removal, clearing
and losing-candidate destruction return grants outside value-map/registration
locks; clearing membership does not revoke an active Arc's ownership.

A standard-layout MLAS packing capacity refusal lets the existing native
bounded fallback run and leaves the static capability memo unset, so the same
kernel can try packing after capacity returns. Other typed failures still
propagate. A serialized `weight_prepacked=1` operand cannot use the wire decoder:
its capacity failure remains an error, and an incorrect byte length is rejected
before a grant or data allocation. No new compatibility interface, quantization
algorithm, ISA predicate or dispatch threshold was introduced.

This covers the packed payload and retained scale/zero-point copies. Mutable
MLAS GEMM workspace, including the existing eager M=1 workspace, and transient
tile/activation scratch are separate scopes and remain ungoverned. Native/MLAS
OnceLock retention still needs idle reclamation; process-wide quota, physical
Host/Device book reconciliation and CUDA runtime proof remain unfinished. No
RSS, throughput or multi-node placement claim is made.


### Generic f32 GEMM borrows column panels

The thin-M column-parallel fallback no longer creates per-call B element copies
and secondary output strips. B keeps the original matrix stride; workers own
disjoint mutable row slices of the caller output. Dense row blocks and column
strips share the original register tile and K-panel accumulation through a
monomorphized row-access interface. No unsafe aliasing, persistent cache or
additional quota policy is needed for those borrowed elements.

Row-reference descriptors still allocate `O(M * min(N, threads))` metadata;
this is not zero allocation or full scratch governance. A public-kernel
allocator probe marks B as a constant initializer, registers the caller and
every explicit pool worker, observes
an intentional weight-sized positive control, and proves no allocations above
64 KiB on the first and repeated calls for its test geometry. Restoring the
old copy path makes the named guard fail on the first call. Exact panel-order,
partial-tile, output-bound and special-value checks preserve the previous
arithmetic, including empty-K +0 initialization.

See the [scoped diagnostic and allocation proof](../benchmarks/2026-11-21-generic-gemm-borrowed-columns.md).
This removes one family of ungoverned matrix copies; mutable MLAS workspace,
other transient scratch, idle OnceLock reclamation, process quota and physical
Host/Device reconciliation remain unfinished.


### Native NBits node pack ownership

MatMulNBits native layout slots and their single cold-fill gate now form one
shared node owner. The existing executor constant-state hook passes that owner
between activation-shape variants; no process-wide address store, second cache
registry or compatibility interface is added. Both publication and adoption
consume one eligibility rule, and adoption verifies preparation attributes,
constant-input flags and exact `DerivedWeightMemory` source identity. Only a
clone of the same governed source is interchangeable, even behind identical
Host books and holder numbers. Ungoverned standalone peers can still share.

Every layout retains its existing pressure enrollment and owned payload grant.
Dropping the producer kernel does not remove enrollment while a peer lives;
pressure cannot evict active readers, idle slots can be reclaimed, and the next
reader rebuilds under fresh admission. Optional dynamic metadata still forces
call-owned preparation through the existing execution gate. Dense f32 caches
and optional MLAS memo/store owners are unchanged.

[Host evidence](../benchmarks/NBITS_NODE_PACK_SHARING.md) reproduces duplicate
ledger charging in a real executor before the change, then verifies three
shape variants, output bytes, session isolation, all eight slot lifetimes and
concurrent cold builders. This is a node ownership change, not a throughput or
GPU execution result. Other scratch/activation accounting, physical capacity
relationships and CUDA asynchronous validation remain open.


### NBits dense expansions share node ownership

The NBits constant-state payload now carries its f32 NK cache as well as the
native compact layout slots. Its existing cache object owns the cold-fill gate,
pressure enrollment, payloads and grants; sharing an `Arc` to that object
removes shape-local duplication without creating a second cache authority.
The immutable dense-byte retention limit is part of the same resolved sharing
parameters. A different limit, including disabled retention, cannot adopt a
foreign policy even when the grant source and preparation attributes match.

[Host validation](../benchmarks/NBITS_DENSE_NODE_SHARING.md) uses a real
unknown-rank ONNX input to exercise vector, matrix and batched-vector shapes on
every backend without assuming transposed-B GEMM support. Complete outputs and
Host ledger charges distinguish one retained expansion from duplicate copies.
Factory-level tests cover a warm state exported after filling, producer teardown,
active readers, idle reclaim/rebuild and concurrent distinct cold builders under
a single-payload quota. Mutable group indices and zero retention keep execution
transient, with fresh numerical outputs and no retained charge. Dense arithmetic,
placement, decoding, compact formats and optional MLAS memo/store policy remain
unchanged. No memory prediction substitutes for live payload grants.


### Apple dense prefill reuses NK ownership

The native Apple backend now implements transposed-B GEMM, so dense NBits
prefill can consume the same granted NK expansion that decode and node shape
variants already share. Its capability predicate is the existing NT rule,
not a model-specific gate. The cache ceiling and physical grant continue to
control retention and admission; the existing fallbacks remain on refusal.
Actual executor tests cover symbolic prefill/decode row counts and separate
sessions, and a kernel test checks that warm prefill builds only once.

[Validation](../benchmarks/APPLE_NT_GEMM.md) explicitly records NN/NT floating-point
reduction differences and f64-oracle error bounds. No token-equivalence, GPU
behavior or throughput claim follows. The shared BLAS boundary validates buffer
extents and ABI dimensions before unsafe entry; pure native GEMV keeps usize
shape extents. Memory placement and grant/reclaim algorithms are unchanged.


### Reuse-gated Apple dense prefill

The resolved NT capability declares its dense-prefill reuse policy. Apple uses
NT only when immutable weights and the existing dense-byte ceiling admit NK
retention. With disabled/insufficient retention or mutable metadata, it preserves
KN preparation; physical grant refusal still follows the existing fallbacks.
This does not add a budget, model identity gate or geometry tuning table.

[Actual execute evidence](../benchmarks/APPLE_NT_EXECUTE_AB.md) keeps the initial
transient-NT observations separate from a newly recorded reuse-gated corpus.
Build-count, Host-charge and successful-NT-call witnesses accompany every cell.
The source's thread-local diagnostic selector is absent from shipped builds;
its scope/unwind behavior is tested. Process CPU includes framework/worker spin,
and no model throughput, RSS or other-device behavior follows from the numbers.


### Node-scoped optional reference ownership

MLAS SQNBit reference layout maps join the existing per-node constant-state
owner. Shape variants share only after the same source and preparation/retention
policy checks; independent nodes and sessions retain separate maps, even if they
use identical operand addresses and Host books. The global and test-local stores,
installer eviction responsibility and executor-wide clear API are removed.
Dropping an executor releases its kernel and static-state owners before source
buffers, without touching another executor's caches. Active payloads continue to
own Host grants through their final reader. No primitive arithmetic or production
backend choice changes. See the
[reference ownership contract](../performance/CPU_MLAS_MIGRATION.md#node-scoped-ownership-for-optional-sqnbit-reference-packs).

Reference layout maps now participate in idle Host pressure reclamation and
per-kernel strong memos are removed, as described in the follow-up below.
Serialized prepacked sources retain their existing exclusion from node-state
adoption.


### Idle SQNBit layouts join Host pressure reclamation

The optional reference's node maps now enroll weakly with the same derived-weight
source as native caches. Per-call `Arc` readers replace permanent per-kernel
memo ownership. Lookup and idle exclusion use the same map mutex; removed
payloads and their grants drop after all map locks are released. Reclamation
never takes a construction gate, including when a pack's own admission visits
its cache. Real KV reservations/growth and other node admissions can now reclaim
an idle reference pack without clearing any live session globally.

The bounded factory regression includes sharded, full-width, serialized and
noncontiguous-scale preparation, active readers, concurrent execute/pressure,
retry and final-owner refund. Logical constant slots preserve strided preparation
reuse inside the explicit node owner. MLAS scratch/workspace is not covered by
this payload accounting, and serialized bytes still require capacity for a
reference pack rather than taking a wire fallback. No CUDA execution or new
performance claim is involved.


### Ordinary MatMul derived copies honor retention decline

The shared MatMul preparation layer previously retained MLAS f32 panels,
once-widened f16 panels and native contiguous-half copies even with retention
disabled. Both pack entries now require an admitted constant B. Strided f16
copying returns an explicit grant-carrying reader, so a declined caller can
compute through the existing rescue path and release its transient copy without
populating the kernel cache. Existing admitted reuse and borrowed
column-major execution remain available.

The three old-behavior regressions demonstrate the bypass independently. New
bounded real-kernel controls check both verdicts and unchanged expected results,
while avoiding shared test-counter interference. This is retention enforcement,
not physical Host funding by itself. Ordinary f32 constant materialization now
joins Host admission below, followed by native half-copy payloads. SGEMM panels join Host admission below; transposes remain work; ordinary node ownership is consolidated below.


### Ordinary constant materialization joins the Host authority

All four MatMul-prepack consumers (MatMul, Gemm, FusedMatMulBias and FusedGemm)
now receive the provider's immutable resources rather than a bool/default-only
prepack. Both constant operand slots obtain actual Host payload grants before
conversion, retaining them in an executing reader or idle cache. The common
dense-cache authority handles singleflight construction and idle pressure.
Contiguous f32 borrowing remains zero-copy; dynamic activations are a separate
scratch scope. A declined retention policy changes lifetime, not the need to
fund a required constant materialization. Admission refusal stays typed and
retryable rather than allocating an uncharged fallback matrix.

The dtype converter writes directly into an admitted f32 destination and keeps
the existing SIMD conversion helpers. Strided/scalar conversion does not first
allocate a full source-dtype copy. A sealed reader prevents sibling kernels from
forging an uncharged materialized constant through the preparation API.
Address-keyed legacy transpose memos retain their converted source reader until
their key retires, preventing source-address recycling after pressure. This pins
those dependencies; funding/reclaiming transpose payloads themselves remains
separate, as do activation/output scratch and ordinary node ownership is consolidated below. Native contiguous-half copies and SGEMM panels are funded below.

The same provider-level test fails against the parent implementation with Host
usage 0 versus 24 expected bytes. New factory, active-reader, capacity/refusal,
retry, concurrent-fill and transpose-source controls exercise actual ownership.
The planner also includes Gemm and FusedGemm dense preparation, previously
omitted despite using the same prepack, and preserves domain gates.


### Native contiguous-half copies use the same grant lifecycle

Noncontiguous constant f16 B materialization now uses the same dense-weight
cache as f32 expansion, parameterized by its flat payload element (`u16` bits
or `f32`). Both retained and transient copies reserve their actual 2-byte-per-
element payload before allocation. Copying writes directly into that admitted
slice, preserving the existing 2-D parallel and N-D strided routes; no f32
weight matrix is needed. Source validation precedes admission and pointer reads.
Empty payloads return before parallel chunking.

The returned reader owns the grant through computation. Kernel destruction or
cache eviction cannot revoke a live reader; idle retained copies can donate
capacity to KV through the existing weak pressure registration. Retained cold
fills share one payload, while transient calls each require their own capacity.
Refusal propagates a typed, contextual error through both Apple rescue callers
and leaves preparation retryable. Column-major zero-copy execution is unchanged.

The unchanged public-provider regression fails on parent 3e2588329 with Host
usage 0 versus 306 expected bytes. Controls cover both retention verdicts,
negative strides/offsets, concurrent construction, active-reader protection,
KV eviction/refusal/retry, final refund, invalid host sources and empty layouts.
This step funds the native contiguous copy only. Ordinary SGEMM panel funding
follows below; transpose payloads, activation/output scratch remain work; shape-variant ownership is consolidated
below.


### Check SGEMM panel allocation geometry before funding it

The optional ordinary MLAS SGEMM packer now exposes an immutable `PackedBPlan`
whose private fields bind logical dimensions, source extent, native padding and
allocation layout. Planning allocates no payload. The native shim checks column
rounding, products and buffer alignment against MLAS's own constants before its
otherwise-unchecked size query. A disagreement with the standalone generic
layout is an explicit refusal rather than a potentially undersized allocation.
Rust additionally requires addressable source and destination layouts.

`PackedB::new` and `PackedBPlan::pack` return typed errors for geometry, source
length, layout and allocation failure; the old infallible interface is removed.
The plan and resulting pack report actual owned allocation bytes, including
padding and a one-byte empty sentinel. Empty operands do not call the packer.
Ordinary MatMul propagates construction errors and leaves its memo retryable;
a separate cold-fill gate prevents concurrent half conversions/packs while
warm hits avoid that gate. Warm f32 memo hits validate source extent and the
shape-specific packed dimensions before reuse.

This is a prerequisite for panel admission, not panel funding itself. The
next step below adds actual Host grants and idle pressure integration to
ordinary SGEMM panels and the half-to-f32 packing temporary. Existing f32 and
contiguous-half weight funding remains in place. No new GEMM primitive, backend default or performance
claim is introduced by checking the reference packer's allocation geometry.


### Ordinary SGEMM panels share the derived-weight authority

The shared cache now accepts owned weight payloads rather than assuming every
payload is a flat f32 vector. `WeightCache`, `GrantedWeight` and `WeightPayload`
replace the old dense-only names; vector construction remains a specialized
method and every caller uses the new types directly. Flat f32/u16 vectors and
aligned MLAS SGEMM panels share admission-before-build, exact logical/allocation
byte checks, serialized cold fills, active-reader ownership and weak idle
pressure registration. The dense-expansion environment limit remains a separate
policy and is not imposed on ordinary MatMul panels.

Both ordinary f32 panels and once-widened half panels retain their actual native
allocation grant in an executing Arc reader. Removing cache membership or
dropping the kernel cannot revoke that reader. Independent idle panels can
donate capacity to KV, and a subsequent call can rebuild. Keys bind the source,
dtype and K/N geometry, so a different source/shape never receives stale packed
values. Equal-source warm hits skip native layout planning and the fill gate.

Half preparation preflights the packed allocation plus the entire simultaneously
live f32 conversion before either buffer is created. Conversion writes directly
into one fallibly allocated destination. The scratch portion is refunded after
packing; only native panel bytes stay charged. Packing is optional: an admission
refusal invokes no conversion/packing builder, records no failed cache entry and
uses the existing unpacked execution route. Invalid layouts, bad source extents
and actual construction failures remain contextual typed errors.

This closes actual funding for ordinary SGEMM panel payloads and their half
conversion peak. It does not fund unrelated activation/output/workspace scratch,
legacy transpose payloads, and adds no shape sharing by itself; the next step consolidates node ownership.
Forecast changes are not a substitute for these actual grants; future resident
planning can use the checked native layout query.


### One preparation owner per ordinary MatMul node

`MatMulPrepack` keeps each kernel's retention/constant policy and grant source,
while one Arc-owned `MatMulWeights` holds its derived preparations. MatMul, Gemm,
FusedMatMulBias and FusedGemm expose/adopt that owner through the existing
node-scoped Kernel constant-state interface. The executor already exchanges this
state when compiling shape variants, so its generic cache needs no op-specific
logic. Shape-specific geometry, attributes and FLOP metadata stay on the kernel.

Adoption verifies immutable operand flags and the exact DerivedWeightMemory
source. A different grant source is an error, including a distinct holder on the
same Host ledger. Retention-off and runtime-only variants export/adopt no state;
unrecognized state is declined. Each new node/session starts with its own owner,
even when it uses the same provider and original initializer.

Dense f32 copies, contiguous half copies, native SGEMM panels and existing
transpose memos now survive an individual shape kernel's eviction under this
owner. Installed global transpose keys retire at the last node-owner drop; an
executing funded reader still keeps its payload/grant alive independently. The
legacy transpose payloads themselves still need actual grants and reclamation.

Dense-materialization prediction counts each constant non-f32 operand once per
node. Remove the fixed prefill/decode shape multiplier. Required bytes follow
the retained ownership unit and stay constant as activation shapes vary. The
estimate still excludes native panels, transpose payloads and other scratch,
and same-half paths may retain less; actual required copies remain governed.

Public factory controls cover all four consumers and both constant operand
slots under one-payload capacity with pressure reclamation unavailable. The
real executor control varies activation widths, shares within a node, keeps
distinct nodes/sessions separate, and verifies final refunds. It also disables
pressure reclamation so denied optional reference packs cannot distort the
resident-copy measurement on different CPU backends.


### Hand static preparation ownership to runtime kernels

Load-time provider preparation now lives in the existing kernel cache, indexed
by node together with its immutable-input flags. A newly compiled kernel first
tries a sibling owner, then compatible static preparation, and only performs
ordinary preparation if neither is recognized. Unknown or incompatible static
owners remain alive until executor teardown; kernels and these owners are
released before source buffers. This applies to fully static materialization
and symbolic-shape specialization without CPU-specific session dispatch.

The provider API receives immutable-input flags separately from available host
constant descriptors. CPU transpose prewarming returns the same shareable
MatMul owner consumed by runtime kernels, including when another immutable
operand has no load-time host descriptor. No compatibility signature remains.

Direct initializer immutability has one predicate: ONNX constant-initializer
semantics plus no producer. Writable producer-backed default bytes never enter
constant preparation or retained operand conversion. Allocation still follows
storage presence; view dispatch keeps its existing immutable-source path. Static
and runtime flag disagreement declines static adoption and prepares normally.
This does not redesign view provenance or fund legacy transpose payloads.


### Materialized operand identity includes logical layout

Immutable backing bytes can feed runtime views with different logical layouts.
The former dense-f32 and contiguous-half keys used only address/type/element
count, so equal-size views at the same origin could reuse a stale copy. Both
now use one typed copy key containing effective source address (including byte
offset), source dtype, full shape and signed element strides. Output element
width is a type parameter; payload bytes still use checked shape arithmetic.
No compatibility key or alternate classification remains.

The existing node owner and WeightCache admission/pressure/reader rules remain
in force. Distinct views own distinct funded copies; repeated identical views
reuse the same allocation, including after owner adoption by a sibling kernel.
Direct initializer predictions still count one immutable layout per operand;
custom runtime views are not represented by WeightRef and rely on actual grants.
This corrects materialized-copy identity, not transpose funding or native panel
source-lifetime policy.


### Keep derived-weight accounting out of steady cache hits

A warm reader consumes the payload's existing grant. It must not request a
sibling lease, return capacity or enter the accountant/pressure machinery merely
because a kernel executes again. Ordinary MatMul dense and contiguous-half
copies now use the existing single-entry cache lookup before the cold group
builder, avoiding group result/status containers on the warm path. Their full
logical-layout keys store common ranks inline; arbitrary ranks spill correctly
rather than introducing a rank limit. LRU updates and live-reader protection
remain in place, and the cold transactional builder is unchanged.

A real public-provider MatMul control verifies a cold 24-byte dense grant
(and the checked optional research SGEMM panel when that backend is active),
4,096 warm executions with zero further claims, including zero-byte requests,
or releases, and exact payload refunds at teardown. A retention-off control observes a claim and
refund for every transient copy, so a disconnected observer cannot pass. Warm
reuse does not remove admission for real allocations or hide transient costs.

The `weight_admission_ab` release benchmark interleaves real kernel executions
with local/governed resources, a duplicate local A/A arm, and an unaffected
contiguous-f32 control. It separates retained and transient policies, excludes
warm-up, retains per-round samples, and reports both wall and process CPU time.
The host-lock wrapper must cover the whole comparison. Zero warm callbacks are
a structural invariant; timing alone cannot establish universal zero overhead.


### Packed weights retain source identity without pinning source payloads

An owned conversion buffer can leave its dense cache while a packed panel is
still useful. Its allocation address is not a durable identity: the next
conversion can reuse the same storage for different immutable source values.
Keep the typed DenseOperand through the shared MatMul compute helper instead
of erasing its ownership to a slice before optional SGEMM preparation.

A materialized source now identifies its Arc owner through a weak key. The key
reserves the owner's control block, so that identity address cannot be recycled
while the cache entry lives. It keeps neither the vector's heap elements nor
their Host grant alive. The identity type accepts flat-vector owners, whose
payload allocation is separate from the control block. Live panel readers keep
the independent packed payload and its own grant. Original borrowed sources
use the complete logical-view key; this also fixes half-packing collisions
between equal-size strided views of the same immutable backing storage.

A call-owned, ungoverned conversion lacks a proven immutable identity and uses
the existing unpacked computation. The old raw-slice packed-B interface is
removed rather than retained as a compatibility path. No global identity
counter, registry or cache scan is needed. The native production build adds no
identity bookkeeping: weak identities are compiled with the optional reference
packing consumer. The native build retains its original slice compute interface;
the optional packing build carries typed ownership to its consumer.

Controls explicitly move one conversion allocation into a new owner, establish
identical data addresses without allocator assumptions, and verify different
packed numerical results. They also verify that source grants retire while
old panel readers remain valid, identical owners reuse a panel, full half-view
identity separates layouts, and call-owned conversion falls through correctly.
Legacy process-global transpose payload funding/lifetime policy remains work.

### Node-owned, funded half transposes

Native-width f16/bf16 transpose payloads now use the same node-owned
`WeightCache` as other prepared MatMul weights. Their actual `2*K*N` Host grant
is acquired before writing into the admitted output allocation; the transpose
algorithm allocates no second matrix. Identity includes the original contiguous
source, interpretation and complete matrix geometry. Shape variants and fused
operators share the node owner. Idle entries participate in Host pressure
reclamation; executing Arc readers independently keep their payload and grant.
A declined optional transpose runs no builder and can be retried later.

Apple f16 decode can read the original row-major half storage directly after
optional preparation is declined, without requesting a larger widened weight.
MatMul and FusedMatMulBias use the same fallback arithmetic. x86 keeps its
existing untransposed half GEMV. Cold preparation and pressure transitions may
call accounting; 4,096 warm provider executions make no claim/refund calls,
including zero-byte calls. This callback invariant does not assert zero cache
lookup overhead or universal inference latency.

The global half store and its diagnostic/clear APIs are removed. Multiturn
benchmark diagnostics report model Host ledger usage, explicitly including all
model holders. The remaining process-global **f32** transpose store is still
legacy and needs actual funding/reclamation; its installed keys continue to be
retired by the final node owner. Forecasts remain distinct from admission.

On Apple M1 Max, native CPU 1,992 tests / 20 ignored and optional MLAS CPU
2,032 / 35 ignored pass; session passes 277 / 279 tests respectively, with
strict all-target Clippy in both configurations. The public provider regression
fixture, injected with only a test observer into parent `95de9fef1`, compiles,
selects one test, and fails at the first funding assertion: actual Host usage
is 0 instead of 70 bytes. The current fixture also validates inference after
reclamation and exact final refunds. No CUDA or other-ISA execution is claimed.

### Fund the remaining f32 B transpose payloads

The f32 B transpose now uses node-owned `WeightCache` storage too. Identity is
the original logical view (effective address, dtype, shape, signed strides) plus
requested matrix geometry. It never uses a recyclable widening allocation as
identity and does not pin that allocation. Widened sources and transposes carry
independent grants; an executing transpose remains valid after the source copy
is reclaimed. Construction writes directly into the admitted transpose output.

The global f32 store, installation/eviction memos, global clear/byte diagnostics,
and the conversion-owner lookup are removed. `weight_transpose.rs` contains
only the shared transpose algorithm. Node owners share preparation across
shape kernels, while separate nodes/sessions cannot inherit entries at recycled
original addresses. Empty matrices retain nothing.

MatMul's optional f32 transpose can be refused and uses existing raw-layout
arithmetic. Gemm with `transB` requires a row-major B copy for its backend;
disabling retention now funds that copy transiently, and exhaustion returns an
actionable admission error before building. Other call-owned layout/scratch
allocations, including Gemm's A transpose, remain separate work. This step also
does not fuse source widening into transpose construction: a converted B may
still be materialized before looking up an existing transpose.

Forecasts now include the fused MatMul+bias+Relu f32 route on Apple and promoted
vector B operands. The registry forecast check describes declared routes;
actual provider and Gemm tests independently reconcile their Host grants. Source
validation, geometry/overflow, dtype isolation and pure transpose correctness
checks remain. Old global-map storage/clear/installation tests are replaced at
the actual node/cache seam: deterministic address reuse across node lifetimes,
grow/shrink/equal extents, concurrent equal/distinct geometries, shared owners,
pressure refusal/retry, independent readers and exact final refunds.

On Apple M1 Max, native CPU 1,984 / 20 ignored and optional MLAS CPU 2,024 /
35 ignored pass; sessions pass 277 / 279, with strict all-target Clippy in both
configurations. Public MatMul, FusedMatMulBias, Gemm and FusedGemm tests cover
matrix B; the MatMul family also covers vector B. Each admitted warm case makes
zero further claims/releases over 4,096 executions. The old public provider
fixture on parent `43055b185` compiles, selects one test and fails at its first
funding assertion: actual 0 versus required 140 Host bytes. No CUDA or other-ISA
execution or model-throughput result is implied.

A subsequent warm-path optimization uses only a weak last-hit slot for a
single-layout f32 cache. It cannot keep a payload or grant alive after idle
reclamation; multi-layout nodes retain canonical lookup/recency. The initial
full-key lookup added about 100 ns to a tiny case; the final coordinated run
reduces the overall increment to about 13 ns for f64 and 20 ns for f32 on this
Apple host. This remains measurable bookkeeping cost even though warm quota
requests/refunds are zero. CPU_MLAS_MIGRATION.md records the control arms and
ranges; no end-to-end model or other-tier performance is inferred.

### Prepare f32 transposes from original views

B preparation no longer accepts a separately materialized f32 slice. Its only
source is the validated original matrix/vector view. Conversion and transposition
share the tiled transpose traversal and write directly into the admitted final
f32 payload. This avoids a second matrix-sized allocation and removes the API
state in which supplied dense values could disagree with their identity view.
Contiguous aligned f32 keeps the existing tiled copy fast path; strided, reversed
and other float formats use original coordinates with unaligned numeric reads.

The shared prepacked compute path tries the prepared B before asking for a dense
B. Mandatory activation materialization still precedes optional B admission, so
an optional transpose cannot consume memory needed by A. Direct-output execution
keeps its original destination without a result-vector staging allocation. The
low-level slice compute context again carries only the prepack; optional MLAS
packing keeps its typed operand, unchanged.

Decode-only widened B now needs one `4*K*N` derived representation. A session
that also uses ordinary prefill may retain its independent dense B as well, but
KV pressure can reclaim that idle copy without decode rebuilding it or making
new quota calls. Gemm transB also builds its required retained/transient layout
directly from the original view. Other scratch and conservative graph forecasts
remain separate work.

Public factory controls cover MatMul, both fused MatMul forms and Gemm: a one-
representation budget succeeds from cold, 4,096 warm executions issue no further
claims/refunds, and decode continues with no quota calls after KV pressure
reclaims prefill's dense copy. The same test-only fixture compiles and selects
one test on parent `404fd04c4`, then fails at cold requests 3 versus expected 2
(including the setup anchor). All 65,536 bit patterns of f16 and bf16 are compared
bit-for-bit against the old conversion followed by an independent transpose;
f32/f64 special values and before-write extent checks are included too.

## Current design decision: execution workspace domains

Status: design reviewed; the CPU resource-domain foundation and first Gemm
transpose-workspace migration are implemented. Other scratch migration remains
pending. The unpublished per-node mutable-layout
pool prototype was withdrawn. The existing node-owned immutable weight preparation remains the
baseline. The next implementation is a CPU vertical through the existing
executor workspace contract, not another kernel-specific resource pool.

### Resource and ownership boundaries

| Resource | Owner / allocation unit | Authority interaction |
|---|---|---|
| Immutable prepared weights | Node preparation owner; independently reclaimable payloads or build groups | Cold build, growth and actual retirement |
| Temporary scratch | Provider/session resource domain; one exclusive workspace per concurrent execution slot | Slot creation, capacity growth, idle trim and destruction |
| KV and semantic persistent state | Their explicit state owners | Actual growth, shrink and release |
| Kernel-local scratch regions | Aligned borrowed slices of the prepared workspace | No claims or refunds |

Coarse allocation means pooling a real backing allocation with its existing
charge. It does not mean reserving the sum of every forecast at model load, or
introducing a second bank of unused budget credit. Stable-shape sequential
nodes reuse the graph's scratch peak; independent concurrent runs need separate
execution slots. A larger shape grows capacity at a preparation boundary.

Retain object-level weight reclamation: placing all model weights into one
session slab would let a small active reader pin an otherwise reclaimable slab.
A weight build group can preflight its live construction peak once and split
ownership afterward. `MemoryLease::split_off` avoids another claim, but each
child lease still refunds through the issuer on drop; it is not a hot scratch
suballocator.

### Reuse the existing workspace seam

`onnx-runtime-ep-api/src/kernel.rs` already supplies `workspace_requirement`,
`workspace_requirement_for_execution` and `execute_with_workspace`. Execution
returns Ready or Bind; only Bind uses the optional host-only
`workspace_requirement_for_bound_execution` hook. The shared resolver uses the
same caller owner for prefix and final admission, and permits no planning loop.
`onnx-runtime-session/src/executor/bindings.rs` collects workspace peaks and
`executor/dispatch.rs` passes `WorkspaceView`. Keep this single execution seam.
An EP describes checked size/alignment/layout and consumes borrowed regions;
the common resource layer owns admission, pooling, concurrency and release.
Direct kernel users should use the same public runner for workspace preparation.
Provide one common checked layout/partition helper for overflow, alignment,
capacity and non-overlap, so each EP supplies region requirements rather than
reimplementing raw byte-offset validation and allocator fallback.

Proposed lifecycle over the existing prepared-workspace allocation:

```text
run admission -> borrow exclusive execution slot -> dispatch with WorkspaceView
              -> run completion -> return backing + charge to idle storage
idle pressure / shutdown -> release backing -> settle its charge
```

Separate content lifetime from storage retention. Step-scoped contents expire
at run completion, while their backing may remain idle and funded for reuse.
Do not merely relabel temporary contents as `SessionPersistent`. State whose
values must survive calls remains independently owned and cannot be overwritten
by graph scratch reuse. An asynchronous slot enters pending until its completion
event; capture pinning and quarantine also prevent idle reuse and refunds.

The peak reflects overlap: mutually exclusive branches use a maximum; live
parent regions add to the child's peak; concurrent branches need the sum of
simultaneously active regions. Reuse is legal only at an established completion
or ordering boundary. Do not add per-kernel synchronization to obtain reuse.

Prepared-resource selection and scratch planning must agree. Either acquire and
pin the selected prepared readers before computing scratch, or declare a peak
that covers cache miss/refusal fallback. Workspace admission must not evict a
weight whose presumed presence was used to omit required fallback storage.

### CPU backing and accounting prerequisite

The CPU provider currently inherits the default workspace reservation returning
`None`. Merely advertising kernel workspace therefore does not prove a Host
grant. Wire CPU workspace once at the provider/resource-domain boundary using
`MemoryRole::Workspace` and the existing process memory manager's managed-owner
and physical-release transactions. `DerivedWeightMemory` has a weight-role
anchor; its sibling reservation must not misattribute scratch as weights.

The default CPU allocator includes `LargeAllocCache`. Returning an allocation
to that cache can retain its backing. An upper workspace pool must not refund
capacity while a lower, unaccounted cache still holds the block. Use one reuse
owner: trim either releases backing without reparking it, or transfers the same
managed owner and charge into the lower cache. Logical availability inside a
slot and physically releasable capacity are different facts, not two ledgers.

### Implementation sequence and evidence

1. Introduce borrow/return/idle-trim ownership for execution-slot workspace over
   the existing prepared allocation; centralize its policy in the resource
   domain, preserving active, pending, pinned and quarantined states.
2. Connect CPU Host workspace backing and actual Workspace-role charges, with
   one cache/release owner and an injectable authority for tests.
3. Move dynamic Gemm A layout scratch into this workspace seam. Keep immutable
   B preparation and checked original-view conversion; derive a single Gemm
   layout plan for its selected route and fallback requirements. Gemm already
   participates in workspace preparation; add no new op-name special case.
4. Migrate accumulator/TLS/Einsum scratch through the shared mechanism and
   remove their independent retention/accounting policies as each migrates.
   Replace `is_planned_workspace_node` with compiled workspace-plan facts.

The withdrawn prototype was useful evidence, not an implementation to publish:
per-call transient grants made a tiny dynamic A case roughly 288 -> 403 ns on
M1 Max. Retaining exclusive, refilled buffers eliminated warm issuer callbacks,
but still measured approximately 300--309 ns versus 289 ns; local and governed
arms were nearly equal. Zero issuer calls alone does not prove zero total cost.
The experiment's exact source patch, baseline fixtures, full validation drivers
and timing logs are preserved under `/tmp/nxrt-gemm-a-*`, with provenance in
`nxrt-gemm-a-candidate-provenance.json`. It was not merged into runtime code.

Required validation for the new implementation:

- Sequential heterogeneous kernels share one graph peak rather than a sum of
  node pools; simultaneous runs own independent slots.
- Stable-shape warm multi-node sessions perform no issuer calls, including
  zero-byte claims/refunds; growth admits only at the preparation boundary.
- KV pressure reclaims idle backing while active, pending and pinned backing
  remains funded. Cache refusal after route preparation remains executable
  within the declared live peak or fails before output writes.
- Allocation, execution and release failures preserve backing/charge ownership;
  lower allocator retention never becomes an unaccounted second cache.
- Shape growth, control-flow nesting, alignment and arithmetic overflow have
  explicit success/refusal tests using the public runner/session.
- Measure direct-kernel, common-runner and multi-node-session controls with the
  same host lock, so kernel savings cannot hide dispatcher overhead.

CPU tests and compilation may establish the Host contract. CUDA event ordering,
Graph replay, quarantine and actual physical release still require real-device
validation; no CUDA runtime or throughput claim follows from this design.

### Implemented foundation: managed Host execution workspace

`onnx-runtime-memory-governor::HostWorkspacePool` now borrows and returns whole
managed Host workspace owners. Clones share idle execution lanes. A warm fit
retains the original backing, authority and grant; capacity/alignment upgrades
retain their joint high-water marks and release the replaced idle backing
before admitting its replacement. Independent simultaneous executions receive
independent owners. The pool contains no active owner and rejects returns with
outstanding owned views. Physical release and issuer callbacks run outside pool
locks; failed release preparation preserves its owner in the error, and an
incomplete outcome remains charged/quarantined and is never reused.

The domain retains its pressure responder strongly; the responder weakly
references idle storage. Host authorities with a pressure registry can reclaim
idle slots for KV requests. Other authorities require explicit idle trim.
Private Host backing registrations retire after the scoped binding and verified
idle releases; outstanding/quarantined ownership conservatively keeps its
registration pinned. No empty budget-credit bank or per-kernel pool is added.

CPU workspace allocation now forwards to this domain and returns a managed
`DeviceBuffer` without an outer lease. Workspace backing uses uncached
`HostAllocator`; ordinary tensors keep their existing configured allocator.
Explicitly governed CPU resources require a matching workspace Host authority.
An ungoverned provider lazily creates an ungoverned managed domain reporting
unattributed physical bytes instead of inventing another budget.

Engine construction selects the final shared memory/manager before creating
CPU resources. Workers no longer construct CPU resources against temporary
books and subsequently replace only their governor fields. The Host ledger
identity now actually names the Host device, allowing canonical manager
registration against those same books. The executor records the capacity and
alignment returned by the provider, so a high-water backing is not needlessly
replaced when a later requirement fits it.

Native validation covers 77 governor unit tests (including six workspace
regressions), 36 manager transaction tests, 25 CPU provider tests, 277 session
tests, 15 engine governor tests and six authority tests, with strict all-target
Clippy. The warm checkout/return control observes every issuer callback and
performs 4,096 iterations without claims/refunds, including zero-byte calls.
Other controls cover concurrent slots, KV idle reclamation versus active
owners, alignment/capacity alternation, foreign or borrowed owners, delegated
process capacity, incomplete release and registration retirement. The worker
regression reconciles actual workspace and KV grants against the parent's Host
books. Optional MLAS configuration additionally passes 25 provider tests, 279
session tests and the exact worker regression; strict all-target Clippy passes
across governor, CPU EP, session and engine in both configurations. Raw
validation logs are `/tmp/nxrt-host-workspace-*`.

These foundation checks establish the resource domain; the first Gemm
migration below adds a real kernel consumer. Full control-flow scratch peak
validation, remaining kernel migration and CUDA runtime correctness still
require their own evidence.


### First kernel migration: Gemm transpose layouts

The requested subagent review confirmed the coarse allocation boundary: one
resource domain owns whole execution lanes, each kernel declares checked regions
and fills borrowed storage, and node-owned immutable weights remain separately
reclaimable. EPs keep the existing three workspace methods; no issuer, pressure
responder, mutable node pool or second execution API is added to Gemm.

`WorkspaceLayout<N>` performs fixed-size, stack-only layout arithmetic and binds
one workspace into disjoint raw subviews. Capacity, alignment, layout overflow
and address wrap are checked once. It deliberately does not return safe typed
references: the EP must establish Host/device accessibility, exclusive ownership
and lifetime before dereferencing. Fresh scratch is represented as
`MaybeUninit<f32>`; shared tiled traversal initializes every element once before
returning a readable slice. There is no preliminary zero fill or dense staging
matrix for a converted transpose.

Gemm's A transpose and dynamic/nonretained B transpose now use two simultaneous
regions within the execution slot. The required retained B transpose keeps its
existing node-owned grant and does not depend on an assumed cache hit. A
contiguous f32 unit-axis transpose can borrow the original linear storage;
strided unit-axis inputs still materialize, including negative strides and byte
offsets. Metadata-only planning covers unknown strides; actual views can shrink
the request, including to zero on the contiguous same-half computation route.
Missing or insufficient workspace fails before output writes. Direct kernel
callers must provide the same declared workspace, without a private allocator
fallback.

Immutable A's existing widening memo is preserved before filling its borrowed
transpose region, so a warm constant need not reconvert its source every call.
Dynamic/nonretained transposes fuse conversion directly from the original view.

This first migration intentionally leaves ordinary nontransposed widening,
output/bias buffers and half-kernel internal packing/scratch on their existing
paths. It also does not add an immutable A-transpose cache. Subsequent migration
must account for its own live peak rather than claim that all Gemm allocations
are now governed.

Native validation covers 99 EP API unit tests and 1,997 CPU unit tests (20
existing ignored tests), followed by the final 24 Gemm and three shared-transpose
controls. Final session validation covers 280 native tests; optional MLAS covers
2,038 CPU tests (35 existing ignored tests) and 282 session tests. Strict
all-target Clippy passes across all three crates in both configurations. The real governed Gemm control performs 4,096 workspace
checkout/execute/return cycles with zero issuer callbacks, including zero-byte
calls. A strict prepared two-Gemm graph uses one 32-byte peak allocation rather
than the sum of its 24- and 32-byte regions, and reuses the same managed identity
across calls. Other controls cover fresh uninitialized storage, simultaneous
regions, refill after source changes, half fallback, signed strides, offsets,
unit-axis borrowing, overflow and failure before output writes. These establish
Host ownership and execution behavior, not CUDA correctness or whole-model
throughput. Isolated timing evidence is recorded separately when available.


Nested executors now carry persistent and step workspace separately and select
by the requested lifetime. Forwarding prefers the inherited owner, matching
ordinary kernel dispatch, before considering an older local prepared slot.
Real If/Gemm regressions exercise both branches, smaller/larger persistent
backing, preserved persistent contents, and an older local step slot below or
above the inherited peak. The child creates no additional lane. This is Host
execution evidence; it does not establish asynchronous CUDA completion safety.

Isolated M1 Max native timing used the same stable host flock as the parallel
llama.cpp work, optimized builds, `RAYON_NUM_THREADS=1`, and alternating baseline
and candidate processes (three each). Each process reports the median of 15
batches; the table takes the median across processes. Baseline is `9a691aff5`,
whose tree equals merged foundation `8ff087186`. Kernel rows exclude preparation
and hold their workspace outside the timer; session rows include ordinary
public API dispatch, output collection and identical result checks. The shared
checked-out/returned-owner counter control above measures the issuer separately.

| Workload | Baseline | Workspace implementation |
|---|---:|---:|
| f32 A transpose, M=1 K=16 N=16 (linear-order equivalent) | 0.242 us | 0.218 us |
| f32 A transpose, M=4 K=16 N=16 | 0.335 us | 0.328 us |
| f32 A transpose, M=8 K=128 N=64 | 1.776 us | 1.614 us |
| f32 dynamic A+B transpose, M=2 K=32 N=16 | 1.097 us | 0.917 us |
| f32 no-transpose control, M=4 K=16 N=16 | 0.248 us | 0.253 us |
| Two-Gemm session run | 3.707 us | 3.909 us |
| Two-Gemm session prepare + run | 5.228 us | 5.285 us |

Zero issuer calls does not mean zero total runtime overhead. The tiny two-node
session still costs about 0.2 us more per run; this design amortizes slot handling
across the graph rather than paying it per kernel. Individual transpose wins
are not a whole-model throughput result. These timings cover the CPU/layout and
linear-session paths; nested lifetime selection is covered by the separate
functional regressions. Raw timing rounds, source/binary provenance and drivers
remain under `/tmp/nxrt-gemm-workspace-*`. Further slot/dispatcher improvements
must preserve these no-transpose and end-to-end controls.

### QLinearMatMul accumulator: one funded execution peak

The integer accumulator now belongs to the same execution workspace domain.
Its checked requirement is one logical `M * N * sizeof(i32)` tile, reused
synchronously across broadcast batches; it is not the full batched result size.
Empty batches or zero M/N need no accumulator. A nonempty zero-K output still
initializes its tile before requantizing the zero product. Geometry parsing
checks source spans, broadcast count, tile/output extents and byte capacity
before admission or execution. Common 2-D geometry uses inline shape storage
instead of allocating promotion copies and stride vectors for each query.

The TLS accumulator, independent process counter and public
`qlinear_accumulator_*` / `set_qlinear_accumulator_process_cap_bytes` interfaces
are removed without aliases. Engine resident-weight planning no longer adds a
fixed 128 MiB scratch forecast. Workspace backing and its existing charge are
reused independently of immutable-weight retention policy, and idle pressure
reclamation uses the common domain. The existing preparation inventory now
includes QLinearMatMul and its coverage guard; replacing that inventory with
compiled workspace facts remains separate work.

Native/nonfused accumulation initializes each tile before creating a readable
i32 slice. The MLAS fused reference accepts `MaybeUninit<i32>` scratch directly,
so it adds no memset and promises no initialized scratch afterward. Its zero-K
entry explicitly produces the requantized zero result; ordinary zero-K integer
GEMM overwrites stale accumulators with zero. Safe dense/packed/requantized
reference entry points check matrix element and byte spans before FFI, including
release-mode overflow. The packing constructor and size query also reject
unaddressable inputs.

Validation covers 2,001 native CPU tests and 282 native session tests, 2,042 CPU
tests and 284 session tests with optional MLAS, strict all-target Clippy across
CPU/session/engine in both configurations, ten integer-GEMM reference tests,
and a release-mode overflow refusal regression requiring no large allocation.
A two-node quantized graph uses a single 24-byte workspace peak instead of
summing 16 and 24 bytes. Concurrent calls hold separate funded lanes; a fourth
16-byte call refuses while three fill an actual 48-byte Host quota. Batch refill,
empty dimensions, vector promotion, zero contraction, nonfinite combined
scales and fused-to-native scratch reuse preserve the existing byte results.
The shape-variant pack test reconciles node-owned weight bytes separately from
the shared workspace charge and verifies both disappear through their owners.

The RSS example now uses an actual Host governor and shared workspace pool.
`QLINEAR_HOST_BUDGET_MIB` controls real admission; it reports admitted/refused
lanes and charged workspace bytes, not a fake per-thread retention total.
Scales, zero-point vectors, sign-domain copies, output staging and native/backend
packing scratch remain on their existing paths; accumulator migration is not a
claim that all quantized-kernel allocations or RSS are governed. The subsequent Einsum migration below removes the final consumer of the
legacy accumulator-budget helper.

Preparation now reuses dispatch's existing compiled-kernel bindings, then keeps
one successful metadata-only plan per executor. This memo holds the two
lifetime peaks, not physical storage or quota credit. Its signature uses exact
resolved symbols (including physical-capacity bindings), growing-axis
classification, artifact provider/executor/generation and kernel publication /
eviction counters. The graph, declared input dtypes/presence and immutable
weights are fixed for that executor. Input and external-binding validation
always precedes the memo lookup; unknown shapes that cannot be planned never
publish a memo. A hit still reserves both workspace slots, so a step return or
pressure eviction cannot turn a cached plan into uncharged execution. Metadata
requirements explicitly remain stable cold/warm upper bounds; runtime
refinement still runs at dispatch. Bound diagnostics deliberately bypass the
memo. Peak bytes and alignment are merged independently, including when a
smaller node needs stricter alignment.

Regression coverage includes changed row symbols, capacity refusal without
memo publication, dtype validation on a would-be hit, actual idle backing
reclamation followed by readmission, and 4,096 warm checkout/execute/return
cycles with retained constant B and **zero issuer callbacks**. This is a warm
admission property, not a claim of zero mutex, planning or dispatch cost. The
RSS smoke control admits four 16-byte lanes with a 1 MiB Host budget and
refuses all four with a zero budget; reclaimed charged bytes return to zero.

CPU-only performance controls on the same M1 Max / ARM64 macOS host compare
commit `9679414eb` (old TLS accumulator) with this change. Each mode runs in a
separate process; both arms warm 1,000 calls, collect 15 batches and use the
median of three process medians in alternating order. Both use the same
one-thread environment settings (`RAYON_NUM_THREADS`, `MLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`); these settings are not a measured
MLAS worker-count claim. The stable host lock covers the entire paired sweep.
Every case verifies output bytes. Direct-kernel timing reuses output and holds
its workspace outside the timed region; the real two-node session timings
include workspace checkout/return and optional constant-weight retention.

| Mode | Native before → after | Optional MLAS before → after |
| --- | --- | --- |
| Tiny kernel, M=1 K=32 N=16 | 0.920 → 0.894 us | 0.788 → 0.747 us |
| Prefill kernel, M=64 K=256 N=256 | 343.237 → 343.343 us | 22.601 → 22.368 us |
| Two-node session run | 5.645 → 5.880 us | 6.192 → 6.571 us |
| Two-node session prepare + run | 5.816 → 6.113 us | 6.294 → 6.662 us |

The first version regressed repeated prepare + run by roughly 3 us: it walked
the graph and rebuilt metadata/cache keys every call. Reusing compiled bindings
and the pure planning memo removes that regression. The residual roughly
0.24–0.38 us/run on this very small graph is real slot/dispatch cost, despite
zero warm issuer callbacks. The larger kernel controls are approximately flat;
none of these CPU microbenchmarks establishes whole-model or CUDA throughput.
Raw initial, prebinding-only and final rounds, drivers, binary/source hashes
and summaries remain under `/tmp/nxrt-qlinear-*`. Consumed baseline worktree,
one-shot mutation scripts and probe binaries are removed after provenance is
recorded; retained drivers can rebuild the documented base for reproduction.


### Einsum accumulator workspace and a plain retention policy

Einsum now borrows its typed GenericNative output accumulator, floating oracle
output and materialized-GEMM canonical f32 output from the same StepScoped
execution domain. Its requirement is a checked output-sized layout using the
actual accumulator width; empty output is NONE. Successful contraction trees,
optimized view routes and scalar-GEMM routes reserve no unused generic output
buffer. Floating GEMM keeps a conservative f32 output bound because the direct
route also depends on output layout and aliasing, which metadata-only planning
does not observe.

The factory compiles tree choices once for each supported accumulator width
(1/2/4/8 bytes). Planning and execution consume the same immutable candidate
index and byte fact, with exact compiled-shape/schema validation. This removes
per-dispatch candidate-map construction instead of adding another mutable EP
cache or resource protocol. The existing 32 MiB tree limit is now explicitly an
algorithm-selection ceiling: larger candidates select lower-memory generic
execution; it is not a physical grant, a process cap or a retention budget.

Generic accumulation writes each final sum into MaybeUninit storage, including
zero reductions, and reads only after all output elements have been written.
Oracle preserves its existing initialization pass. Materialized GEMM explicitly
initializes the destination before calling MatMul, whose current contract
forms initialized f32 references. This adds a real pass on that route; removing
it requires a properly proven overwrite seam in MatMul, not an unsafe cast in
Einsum. Alias, negative-stride, precision and byte-result conformance controls
remain in place.

The TLS weak registry, typed slot owners and process accumulator ledger are
removed, along with the final governed_accumulator_budget module. Removed
scratch live/cap/prediction APIs and the obsolete default-opset compatibility
helper have no aliases. CpuCacheRetention now lives with CpuMemoryResources as
a plain immutable boolean verdict; it owns no scratch, mutex, thread registry
or physical allocation. Engine resident-cache prediction no longer invents a
128 MiB Einsum scratch resident. The benchmark owns an exclusive prepared lane
outside timing, and conformance exercises fresh uninitialized backing instead
of consulting TLS capacity.

Real Host-budget regressions alternate f64/f32 execution and opposite weight
retention policies over the same allocation, observe 4,096 warm
checkout/execute/return cycles with zero issuer callbacks, reject a second
active lane when the true cap is full and return charged bytes to zero on idle
reclamation. A two-reduction graph uses max(24,8)=24 bytes; If branches borrow
the enclosing 8-byte peak rather than reserving a second lane.

This removes the retained accumulator pool, not every Einsum allocation.
Contraction-tree/scalar-GEMM intermediates, operand materializations and layout /
coordinate vectors still use their existing allocations. Those transient allocations remain separate work. The following resource
propagation section addresses the lowered child MatMul source; no claim is
made that all Einsum transient bytes are already governed.

Validation: 1,986 native CPU and 284 native session tests, 2,027 optional-MLAS CPU
and 286 optional-MLAS session tests pass. Each configuration also passes all
three shared-corpus / parallel-tile / 128-operand fallback conformance tests and
the two public numeric/schema contract tests. Strict all-target Clippy passes
for CPU/session/engine in both configurations. The previous private-pool and
budget-helper tests are replaced by real-owner admission/reuse/reclamation
controls; existing arithmetic, precision, route and alias guards remain.

CPU-only paired release controls compare `e4130c5a8` (the previous TLS Einsum)
with this change on the same M1 Max / ARM64 macOS host. Each mode is a separate
process with 1,000 warmups and 15 timed batches; the table uses the median of
three process medians per arm in alternating before/after order. Both arms use
the same thread-count environment settings and hold the stable shared host
flock; those settings do not prove the backend worker count.
Direct-kernel scratch and outputs are prepared outside timing; public-session
rows include checkout/return, and prepare+run also includes preparation. Kernel controls verify nonzero output values and observe the actual execution
route; public-session controls verify their output values through the session
API;
these are CPU controls, not whole-model or CUDA throughput measurements.

| Mode | Native before → after | Optional MLAS before → after |
| --- | --- | --- |
| Tiny generic reduction (2×4 → 2) | 0.358 → 0.326 us | 0.348 → 0.317 us |
| Generic reduction (16×64 → 16) | 3.561 → 3.540 us | 3.456 → 3.430 us |
| Oracle reduction (16×64 → 16) | 4.966 → 4.923 us | 4.830 → 4.807 us |
| Direct GEMM (16×64×16) | 1.683 → 1.643 us | 1.723 → 1.733 us |
| Materialized/permuted GEMM (4,4,8,8 × 8,8,4,4) | 6.829 → 6.906 us | 6.997 → 6.970 us |
| Larger materialized/permuted GEMM (8,8,16,16 × 16,16,8,8) | 65.422 → 65.329 us | 65.380 → 65.266 us |
| Three-input contraction tree (8×16,16×16,16×8) | 36.712 → 19.956 us | 35.637 → 19.363 us |
| Two-reduction session run | 3.499 → 3.633 us | 3.486 → 3.536 us |
| Two-reduction session prepare + run | 3.654 → 3.791 us | 3.671 → 3.637 us |

Removing the private pool helps the tiny kernel. Compiling tree decisions moves
candidate planning out of execution, reducing this tree control by about 46%.
Its sampled native factory median increases from roughly 175 to 259 us; that
is a cold setup tradeoff, based on only three process samples, not a full model
startup result. The mandatory materialized-output initialization does not show
a material larger-case regression here. Tiny session run still costs roughly
0.05–0.13 us more despite zero warm issuer callbacks. Raw rounds, complete
sampling details, drivers, source/binary hashes and summaries remain under
`/tmp/nxrt-einsum-*`. Consumed probe binaries, one-shot generators and baseline
worktree are removed after provenance is recorded.


### Lowered MatMul keeps prepared ownership and the parent resource source

EinsumFactory now accepts CpuMemoryResources instead of only a cache-retention
boolean. Its child MatMul clones that source and policy, using the same
constructor as ordinary MatMul. It no longer creates a default, unattributed
domain inside a governed provider. The old boolean-only constructor is removed
and all callers are updated. Constant-state sharing/adoption delegates to the
existing MatMul node owner, retaining source and immutable-flag checks. Raw
constant bytes are not blindly forwarded to preparation with the wrong lowered
geometry.

Materialized lowering now passes the original ordered view to the existing
MatMul operand preparer, then keeps the returned DenseOperand until GEMM and
output publication finish. Immutable conversions reuse their node-owned grant
and complete original-view key. Optional MLAS packing consumes that same typed
operand: Funded retains its WeightIdentity, Borrowed retains the original view
identity, and call-owned conversion scratch has no retained packing identity.
The lowering no longer erases a granted operand into a raw f32 TensorView.
There is no second cache, scoped retention mask, policy clone or issuer protocol.
Dynamic input eligibility still comes from the immutable node flags and cannot
be promoted by the storage representation.

This also removes repeated immutable conversion work. Funded conversions remain
in the weight ledger instead of being counted again as scratch telemetry. An
explicitly declined retention policy continues to use the existing real
transient-grant semantics for constant conversion; that configuration is not a
zero-callback warm-retention claim. Activation conversion vectors still follow
their existing scratch path.

The typed dense entry checks complete input counts, f32 addressable spans,
checked matrix/batch/result products and output length before computation.
Empty results return before unused batch strides or packing; zero-K fills the
initialized output directly without packing or FFI. Empty Einsum GEMM also
returns before weight preparation. Original direct-view routes keep their
native preparation and backing metadata.

Real-source regressions require an owned lowered B materialization to obtain a
Host grant, preserve outputs on quota refusal and reject foreign-source or
constant-flag adoption. Explicit shape variants share preparation through
4,096 warm calls with zero issuer callbacks, yield idle cache to Host-budget
pressure and re-admit before reuse. The public session test covers several
prefill shapes and M=1 decode, whose GEMV transpose may legitimately add another
representation. Representations remain node-owned and stable across subsequent
shapes; independent sessions own independent preparations.

Prepared-reader tests exercise Generic, the host-selected backend and explicitly
forced MLAS when compiled. Concurrent readers preserve distinct values, active
grants survive cache pressure, and idle owners release all bytes. The recycled
conversion-address regression also invokes the new typed compute entry and
requires reuse of the existing funded pack identity. Malformed counts, byte
spans, broadcast overflow, empty and zero-K cases fail or return before packing.

This fixes actual resource propagation and prepared ownership. Other transient
vectors still need workspace migration, and advisory forecasts do not describe
every lowered weight artifact. Future forecasting should consume compiled
preparation facts rather than append operation-name exceptions to prediction
lists.

The resource-design review keeps the EP contract coarse: metadata declares a
workspace layout, the executor owns admission and checkout/return, and kernels
borrow slices. Immutable derived weights reuse the existing node owner and
source. Lowering passes that owner through instead of adding another cache or
per-operation quota interface. A shape variant shares immutable preparation,
while its mutable execution workspace remains exclusive to its admitted lane.

Validation of this stage: native CPU 1,993 and session 285 tests pass; optional
MLAS CPU 2,034 and session 287 pass. Both shared legal corpus/public-contract
suites pass (3 + 2 each). Strict all-target Clippy passes for CPU/session/engine
in both configurations and the optional reference adapter. Release-mode
prepared-operand tests (2) and geometry overflow/empty guards (1) pass. This is
CPU-only validation; CUDA behavior is untested.

Final paired release controls compare commit 1887ca137 with this stage on Apple
M1 Max / ARM64 / macOS 27. Native selects Accelerate; the optional MLAS sweep
explicitly forces MLAS. Both arms use the same 128 MiB Host cap, actual governed
source and funded workspace outside raw-kernel timing. The previous version's
zero attributed weight bytes do not mean zero physical allocation: its lowered
child used an unrelated default source. All constant controls declare B
immutable; the dynamic control declares neither input immutable.

Each mode runs in an isolated process, with 15 batches and the median of three
process medians per arm, ordered before/after/after/before/before/after. Small
controls warm 1,000 calls; large controls warm 20. Thread environment variables
are identical at 1, without claiming a measured backend worker count. The shared
stable host flock excludes the llama.cpp workload. Route counters and numeric
outputs are checked. These are warm CPU controls, not model/CUDA throughput or
cold-start measurements.

| Mode (all materialized outputs use dcab) | Native before → after (us) | Forced MLAS before → after (us) |
| --- | --- | --- |
| Dynamic direct (16×64×16) | 1.638 → 1.701 | 1.681 → 1.669 |
| Strided constant direct (16×64×16) | 1.785 → 1.812 | 1.676 → 1.690 |
| abxy,xycd (4,4,8,8 × 8,8,4,4) | 6.956 → 3.276 | 6.204 → 2.930 |
| abxy,cdxy (4,4,8,8 each) | 6.968 → 3.278 | 6.223 → 2.928 |
| abxy,cdxy (8,8,16,16 each) | 65.790 → 18.832 | 65.889 → 19.378 |
| abxy,xycd (2,2,32,64 × 32,64,64,64) | 23476.383 → 3350.467 | 22677.867 → 331.650 |
| abxy,cdxy (2,2,32,64 × 64,64,32,64) | 54751.950 → 3284.158 | 54419.042 → 323.383 |
| Tiny Einsum public session run | 3.444 → 3.453 | 3.745 → 3.766 |
| Tiny public session prepare + run | 3.652 → 3.685 | 3.953 → 3.920 |

Retaining original-view conversion and packing removes repeated constant work.
The larger native controls retain 32 MiB of attributed weights; forced MLAS
retains 64 MiB (conversion plus pack), in addition to the 64 KiB output
workspace. The 128 MiB cap admits this real combined peak in both arms. Small
materialized controls retain 4/8 KiB native/MLAS; the 8,8,16,16 case retains
64/128 KiB. This is an explicit memory-for-reuse tradeoff; real-pressure tests
verify reclaim and re-admission. Internal probe labels such as borrowed/owned
are case identifiers, not assertions about the returned storage ownership.

The earlier scoped-policy experiment regressed one forced-MLAS prefill control
by about 13%; it was discarded. The final typed pipeline reduces that control
from 65.889 to 19.378 us. Tiny direct controls still add about 14–62 ns depending
on backend/case. Tiny public run adds 9 ns native and 21 ns forced MLAS; prepare
+ run changes by +33/-33 ns. Zero warm issuer callbacks therefore do not imply
literally zero runtime overhead. Raw rounds, source/driver/binary hashes and
summaries remain under `/tmp/nxrt-einsum-weight-prepared-*`; the superseded
experiment is separately labeled under `/tmp/nxrt-einsum-weight-*`.


### MatMulNBits half activation conversion borrows the execution lane

All native and optional reference routes consume one read-only f32 activation.
For f16/bf16 A, that conversion now borrows one StepScoped region through the
existing WorkspaceLayout/Kernel contract instead of allocating a Vec per call.
The bound is the checked complete A element count times four bytes, with f32
alignment; metadata admission precedes conversion and packing. Kernels do not
call an issuer or retain a private activation pool. Sequential nodes share the
executor's maximum, while concurrently active lanes need distinct backing.

Contiguous Host f32 still borrows its input and declares NONE. Strided f32
keeps its existing call-owned gather: shape-only TensorMetadata cannot prove
its layout, and reserving a complete f32 copy for every ordinary f32 call would
waste the quota. No runtime-only larger request can undercut strict prepare.
Missing, short and misaligned workspace fails with MatMulNBits context before
conversion or output writes. The common widening helper preserves byte offsets,
negative strides and the existing half conversion semantics. Fresh workspace is
borrowed as MaybeUninit, initialized inside time_widen and then passed to the
existing initialized-f32 converter; the phase timer includes initialization,
matching its old scope. This does not claim to have removed the zero fill.

The empty-output regression also exposed a pre-existing invalid typed slice:
an empty Vec<u8> need not carry an f32-aligned dangling address. The kernel now
returns before constructing that output slice or packing. Leading zero extents
avoid irrelevant row-count products, and nonempty output counts/f32 spans are
checked before typed output access.

Real Host-cap tests alternate f16 and bf16 and opposite weight-retention
policies through 4,096 checkout/compute/return cycles with zero warm issuer
callbacks, including zero-byte callbacks. A second active lane cannot borrow
the first lane's grant; idle storage yields to actual KV pressure and must
re-admit before reuse. The executor graph uses max(128,64)=128 bytes for two
sequential quantized nodes, reuses one allocation across runs, and re-admits
after trimming. Separate route controls check every measured M/K/N and A dtype
against actual SQNBit counters in the optional reference build, plus exact
numeric outputs. The production timing binary has no test-counter overhead.

The output conversion Vec, strided f32 gathers, scale/bias conversion, quantized
activation tiles and optional reference-private workspace remain separate
migration work. Those regions can coexist with widened A; future layout plans
must sum simultaneous regions instead of pretending they are alternative
lifetimes. No new EP interface, cache, environment policy or MLAS dependency is
introduced by this change.


The cross-platform lane also exposed a missed caller from the earlier QLinear
workspace migration: the ARM64 compensation-allocation integration test still
called execute without scratch. It now declares and owns the prepared 108-byte
accumulator outside its allocation-counting window, then counts the complete
prepared compute call. The positive allocation control, numeric checks, both
zero-point cases and equal call counts for 1/2/17 batches remain. The explicit
108-byte assertion prevents accidentally reserving one tile per batch. This
changes the test caller, not the arithmetic or its allocation threshold.

Final local validation for this stage: native CPU 1,997 / session 286; optional
MLAS CPU 2,038 / session 288; all CPU integration targets pass (62 native / 63
MLAS), including the corrected ARM allocation counter. Strict all-target Clippy
passes for CPU/session/engine in both configurations, and CPU targets were
checked again after the integration-caller fix. The four activation-workspace
controls also pass in release; explicit forced-reference route proofs cover all
three fixture geometries across all three input dtypes. No CUDA execution is claimed.


Paired release controls compare 319323a104 with this stage on M1 Max / ARM64 /
macOS 27, using the same 64 MiB real Host cap in both arms. All weights are
standard-layout int4, block 32, accuracy 0, with immutable B/scales. Raw kernel
controls publish f32 output to isolate activation conversion; the public
session uses f16 input/output. Workspace backing is held outside raw-kernel
timing; public run includes checkout/return and prepare + run includes prepare.
Each mode has 15 batches/process and the median of three isolated process
medians/arm in before/after/after/before/before/after order. Warmup is 1,000 calls
for tiny/decode/session, 100 for prefill. Rayon/MLAS/OMP/VECLIB/decode thread
configuration is identical at 1; these knobs are not a measured worker-count
claim. The stable host flock excludes the llama.cpp workload throughout.

Separate counter controls establish the fixture routes in both source arms;
the native baseline proof uses release tests and the reference baseline proof
uses debug tests, neither is timed. Production release examples have no test
counter overhead. Reference timing explicitly forces MLAS/SQNBit, checks an
actual live SQNBit pack, and validates every numeric output. Resident reference
pack bytes are unchanged before/after: tiny 320, decode 327,680, prefill 163,840.
Thus the recorded activation charge is not a changed weight-pack policy.

| Mode | Native before → after (us) | Forced MLAS before → after (us) |
| --- | --- | --- |
| Tiny f32 (M=1,K=64,N=8) | 0.869 → 0.846 | 0.659 → 0.676 |
| Tiny f16 (1,64,8) | 0.905 → 0.869 | 0.712 → 0.684 |
| Tiny bf16 (1,64,8) | 1.043 → 1.053 | 0.865 → 0.851 |
| Decode f16 (1,2048,256) | 74.412 → 74.457 | 100.027 → 100.063 |
| Decode bf16 (1,2048,256) | 79.143 → 78.892 | 104.499 → 104.538 |
| Prefill f32 (32,1024,256) | 1195.621 → 1173.934 | 215.523 → 215.476 |
| Prefill f16 (32,1024,256) | 1201.039 → 1183.014 | 219.750 → 219.230 |
| Prefill bf16 (32,1024,256) | 1254.490 → 1263.907 | 291.298 → 292.312 |
| Tiny half public session run | 3.235 → 3.427 | 2.985 → 3.187 |
| Tiny half public session prepare + run | 3.449 → 3.628 | 3.214 → 3.356 |

The managed activation charge is 256 bytes tiny, 8 KiB decode and 128 KiB prefill;
f32 stays zero. Old zero managed bytes mean an unaccounted call-owned Vec, not
zero physical memory. Raw decode/prefill is broadly flat on these controls;
tiny raw deltas range from -35 to +17 ns depending on case/backend. The tiny
public session adds about 0.192 us native / 0.202 us reference (5.9% / 6.8%);
prepare + run adds 0.179 / 0.142 us. This is a measured residual, not a
zero-overhead claim. The warm issuer test establishes zero callbacks, not zero
cost for ownership wrappers, validation or pool synchronization. No whole-model,
RSS, CUDA or universal throughput claim follows from these controls.

Source inspection identifies a concrete follow-up: DeviceBuffer currently
boxes a ManagedAllocation on every checkout and unboxes it on return. Preserve
that existing Box through the pool instead of allocating a new wrapper for each
loan. The design review recommends changing the ownership seam together, with
no parallel API, while retaining context retirement, foreign-return, holder/role
and outstanding-view guards. Verify the Box address across warm cycles and
measure the public-session effect before attributing any timing to it.

Validated source hashes, exact driver variants, route controls, binary hashes,
raw rounds, first-failure evidence and summaries remain under
`/tmp/nxrt-nbits-activation-*`. Consumed build helpers, probe binaries and the
baseline worktree are removed after recording provenance; source probes and
reproduction/validation drivers remain.


### Preserve the workspace ownership box across warm loans

HostWorkspacePool now creates a Box<ManagedAllocation> only with new physical
backing. Idle storage, checkout, DeviceBuffer and completed return pass that
same box without unpacking or allocating another wrapper. This does not move
physical charging into a different domain or pin idle storage: capacity growth,
Host/KV pressure, verified release and refund keep their existing ownership path.
Foreign and outstanding-view failures retain the original box for recovery.

The common ownership handoff preserves DeviceBuffer's existing Box for ordinary
Binding owners too. Otherwise boxing only Managed would leave a 216-byte
Binding alternative beside an 8-byte Managed alternative; preserving both boxes
keeps the handoff compact without a lint exemption or a new allocation. Plain
binding allocation/release mechanics are unchanged. CUDA's cold managed
constructors explicitly box their fresh owner once, where DeviceBuffer used to
do it internally. Host-only queue tests and source compilation cover these
callers; they do not establish CUDA runtime behavior.

The duplicate into_bound_owner / into_bound_ownership and into_raw_with_owner /
into_raw_with_bound_ownership interfaces are reduced to the explicit ownership
names, with every caller and error/doc reference updated. No alias remains.
EPs still transfer the same complete owner and settlement token; there is no
second workspace protocol. Kernel arithmetic and workspace layouts are unchanged.

Warm checkout still enters the provider context, rejecting retirement or
termination. Return still validates binding, holder/role and outstanding views.
Those guards are required and are not traded for timing. Existing tests retain
active-lane isolation, pressure reclamation, failed physical-release accounting
and exact warm issuer-callback checks. Additional assertions track the owner
Box through smaller-capacity reuse, raw/owned EP handoff and recoverable errors.

A separate CPU integration counter covers 4,096 complete provider
allocate_workspace/deallocate_workspace pairs after cold setup. It observes
both heap allocation and free, with a positive Box allocation/free control, and
requires zero calls plus stable owner/backing identities. Checking only an
address would be insufficient because an allocator could recycle the address
of a wrapper freed and reallocated each time. This counter is not a claim that
whole inference or its unrelated parameter vectors make zero allocations.


The identical CPU allocation probe fails on 1c95431c6 with exactly 4,096
allocations and 4,096 frees, and passes with zero of each after the change in
both debug feature configurations and release. Pointer checks are collected
without aborting mid-loan, then asserted after storage is returned and trimmed;
this keeps a deliberately failing baseline from abandoning an active handle.
Before/after timing binaries also verify the carrier's actual size: 232 versus
16 bytes on this target, detecting an accidentally reused API artifact.

Final host validation passes: governor 77, EP API 99, plugin adapter 384,
native CPU 1,997 + 63 integration tests / session 286, optional MLAS CPU 2,038 +
64 integration tests / session 288. Three manual CUDA workspace queue tests
pass without a device. Strict all-target Clippy covers governor/API/CPU/session/
plugin/CUDA/engine, plus the reference feature configuration. The CPU heap
counter also passes release. Production code hashes remain unchanged across
the final allocation-test cleanup.

Paired M1 Max / ARM64 / macOS 27 release controls compare 1c95431c6 with this
stage. Both arms use the same real 64 MiB Host cap and inputs. The tiny session
is one int4 MatMulNBits (M=1,K=64,N=8, block 32, accuracy 0, constant B/scales),
with exact numeric outputs checked. f16 needs a 256-byte execution lane; f32
requests NONE and is a control for unrelated executor/code-generation drift.
Reference controls explicitly force MLAS/SQNBit and require the same actual
320-byte resident pack in both arms. The borrow-only row executes the public
CPU provider allocation/return pair, including its guards, without arithmetic.

Each process warms 1,000 calls and takes 15 batches; borrow/return uses 20,000
repetitions per batch, public session 2,000. Report the median of three isolated
process medians per arm, ordered before/after/after/before/before/after. Identical
Rayon/MLAS/OMP/VECLIB/decode settings of 1 are configuration, not a measured
worker-count claim. The shared stable host flock excludes the llama.cpp batch.

| Mode | Native before → after (us) | Forced MLAS before → after (us) |
| --- | --- | --- |
| Complete 256-byte borrow + return | 0.1105 → 0.0529 | 0.1093 → 0.0533 |
| Tiny f16 public session run | 3.4224 → 3.3991 | 3.2130 → 3.1082 |
| Tiny f16 public prepare + run | 3.6510 → 3.6045 | 3.3955 → 3.2800 |
| Tiny f32 public session run (no workspace) | 3.1819 → 3.1834 | 2.9502 → 2.9684 |
| Tiny f32 public prepare + run (no workspace) | 3.3941 → 3.3657 | 3.1556 → 3.1728 |

Complete borrow/return falls about 51–52% (roughly 110 to 53 ns). The native
f16 public run saves about 23 ns and prepare + run 47 ns; reference saves about
105 / 115 ns. f32 controls move by about -28 to +18 ns. Public-session deltas
are much smaller and backend-dependent; this is not a 52% inference-speedup
claim and does not eliminate the entire previous ~0.2 us tiny-session residual.
There is no whole-model, RSS, cold-start or CUDA performance claim.

Source-arm switching explicitly cleans the changed governor/API/CPU packages
for debug checks and governor/API/CPU/session for release timing builds. The
allocation counter caught one stale shared-target debug reuse before any timing;
that evidence and the clean rebuild result are retained. Exact source/driver/
probe/binary hashes, raw rounds, allocation falsification and summaries remain
under `/tmp/nxrt-workspace-boxes-*`. Consumed generators, benchmark executables
and baseline checkout are removed after provenance; reproduction sources and
drivers remain.


### Coarse resource ownership and the EP implementation boundary

The resource-design review keeps admission proportional to physical ownership
changes, not to tensor or operator count. Scratch uses one exclusive execution
lane per concurrently active invocation and lifetime. Sequential kernels borrow
the enclosing peak; simultaneously live regions add their requirements. Control
flow must account for parent/child overlap rather than assume all nested scratch
can alias. Concurrent executions never share mutable backing.

The current prepared dispatch already has this boundary: preparation reserves
peak slots; a fitting kernel receives a WorkspaceView without pool checkout,
context entry or a pool lock. Eager execution checks out only on the first use
or capacity growth. Step completion returns the disposable slot once; persistent
and capture-pinned slots remain owned. Pool microbenchmarks measure a lane
checkout/return, not a cost incurred independently by every node in a graph.

A returned slot remains physically allocated and funded. Warm checkout/return
must perform no issuer callback, including zero-byte calls. Only confirmed
physical release refunds backing; failed releases, pending device work, active
views and pinned addresses retain ownership and charge. CPU completion rules do
not establish CUDA event completion. Keep the existing context, binding, holder,
role and outstanding-view guards at ownership boundaries.

EP responsibilities stay small: declare the concrete workspace bound, borrow
its disjoint regions, and forward provider workspace ownership through the
shared resource domain. Kernels do not acquire leases or own pools. Immutable
converted/packed weights retain independently reclaimable prepared payloads;
KV admission follows actual capacity growth blocks. These resources have
different lifetimes and should not be forced into one session-wide slab.

WorkspaceRequirement now derives its ledger role from its lifetime through
role(). Kernels provide only bytes, alignment and lifetime. The independently
writable role field and both flat/nested planning mismatch checks are removed;
non-Workspace or conflicting classifications are no longer representable. CPU,
CUDA, shared mock, session, examples and test callers use the canonical API,
without a compatibility alias. The underlying ledger classification remains. No second execution API, per-kernel quota
protocol, additional allocator/cache, or speculative maximum-capacity credit is
needed. Continue migrating private temporary buffers into the existing lane
before considering size classes or more elaborate eviction policies.


Validation of this cleanup: governor 77, EP API 99, plugin adapter 384;
native CPU 1,997 unit + 63 integration and session 286; optional MLAS CPU
2,038 unit + 64 integration and session 288. Strict all-target Clippy passes
for the ownership consumers, CPU/session/engine, CUDA and shared mock, plus
the MLAS configuration. The new EP API example and existing doctests pass.
Seven CUDA metadata-only governance tests and three Host queue barrier tests
pass; CUDA sources/test targets compile, with no real device execution claim.
Real ORT 1.32.0 shared-plugin conformance runs with prerequisites required:
all 12 tests pass, including workspace shapes/lifetimes and whole-graph control
flow. Removing the mock's now-unused direct memory-api dependency is also
validated by its strict all-target Clippy and a fresh 12-test ORT run.
The first real-ORT attempt lacked the testkit-specific NXRT_ORT_LIB_DIR and
failed before loading ORT; the corrected run explicitly names the cached
library directory and does not accept skipped tests. Existing warm zero-issuer
callback, allocation-counting, concurrent-lane, inherited-workspace and pressure
regressions remain. All heavy validation holds the shared stable host flock.
This is an API/state simplification; no new throughput or RSS claim is made.
Raw logs and source/commit provenance remain under /tmp/nxrt-workspace-role-*.


The role cleanup keeps MemoryRole imports inside the CUDA test modules that
use them. The capture source audit treats the first test-only section as the
production boundary, so a test-only import above production functions hid
those functions from its scan. The exact old head reproduces both named
capture-audit failures (3 pass / 2 fail); the final source passes all five
capture_sync_contract guards without changing any guard or allowlist. CUDA
all-target strict Clippy with cuda,gpu-tests and the seven metadata tests also
pass on this Host-only setup. Records are /tmp/nxrt-workspace-role-capture-*.


### Portable output tiles preserve K order and keep resource operations coarse

The portable NT kernel uses independent 4x4 outputs with unchanged per-output
f32 K accumulation. BlockQuantizedMatMul/MoE reuse at most four decoded rows per
worker; AVX2/MLAS and NBits/QMoE retain one. The generic decoder binds one tile
and one decoder state per shard, reuses them across features, clamps capacity
to the actual shard width and checks size multiplication before allocation.
No extra issuer, cache, scheduler or public EP workspace API is introduced.
This is still call-owned scratch; it is not a complete memory-governance claim.

The initial sixteen-lane K reduction was rejected by a real argmax change.
The replacement pins that two-ULP boundary and matches scalar output bits across
micro-tile edges. All 96 default-build public operator runs retain exact output
bytes and argmaxes. The [full evidence and rejected controls](../benchmarks/portable-nt-microtiles/README.md)
record process CPU time and absolute RSS on M1 Max, the real 64 MiB derived-weight
Host cap, retained-grant/cache-hit checks, normal/reference build tests and
remaining MLAS gaps. Resource-bound review and self-review found no blockers.
All heavy work shares the stable flock with the llama.cpp chat. A source-path
confound and too-short control samples were corrected before interpreting the
final sweep; one remaining failed timing control is excluded, and a focused
same-binary A/A check resolves the noisy unchanged dense path without claiming
a speedup there.


### Block projection scratch and implementation declarations

`BlockQuantizedMatMul` now has a private fixed logical shard plan captured at
kernel creation. Each task carries an exclusive pair of output-column and
decoded-weight slices; a larger Rayon pool cannot increase the number of
scratch slots. Work stealing changes task placement, not scratch ownership.
The streamed branch binds two disjoint regions from the existing StepScoped
`WorkspaceLayout`, without adding an EP execute or quota protocol.

For `M` rows, `K` input features, `N` output features and stored maximum `W`
logical shards, let `P=ceil(N/W)`, `S=ceil(N/P)` for a nonempty output, and `T`
be the compute tile (four on the portable native route, one on the existing
reference/AVX2 geometry). The f32 workspace upper bound is:

- column-major output: `M*N` elements when both axes exceed one, otherwise zero;
- decoded rows: `S*min(T,P)*K` elements, zero for an empty output.

The last shard may have a shorter live tile. Uniform slot widths permit a
checked, allocation-free zip of output shards with decoded slots; all layout,
element and byte products are checked. The fixed decoder block stays on the
stack. Single-token output uses the final staging buffer directly, avoiding a
new transpose copy. Dense-cache hits do not reduce metadata requirements,
because admission may decline and require the streamed fallback.

Final output staging, dtype/strided input conversion, packed-input gathering,
and MoE's owned FC1/FC3 results remain distinct ownership work. This migration
is not proof of a complete process memory cap or a model throughput gain.

Preparation now consumes the selected implementation's `WorkspacePlanning`
declaration from the versioned kernel registry and `KernelMatch`, replacing the
session's operator-name inventory. `None` promises zero executor scratch for
all legal specializations; `Metadata` delegates the exact bound to the existing
kernel requirement. A symbolic-shape decline is conservatively `Metadata`.
A nonzero execution requirement contradicting an explicit `None` fails before
late scratch allocation. Nested unclaimed control flow is planned recursively;
claimed graph kernels are treated as opaque implementations. Diagnostic node
locations are cached at construction and borrowed during decode, with no new
capability scan or allocation per location query.

The existing `execute_kernel_sized` entrypoint now accepts the same optional
workspace view. Native ordinary and HostOwned dynamic-output dispatch share one
requirement/refinement/binding helper, including lifetime selection, inherited
capacity/alignment validation and existing growth/capture rules. The ORT
single-node and routed multi-node HostOwned paths use `prepare_workspace` before
calling the kernel. ORT still declines nonzero SessionPersistent requests with
`None`; kernels requiring that lifetime must reject the decline or provide an
explicit supported owned path. It is never served from short-lived ORT scratch.

The native C graph-ABI bridge also exposes CPU memory-info comparison and the
existing ORT scratch callback. `PluginExecutionPlan::compile` receives an
explicit Host execution provider; the session passes its configured CPU source.
No private allocator or quota source is created. Each foreign Compute retains
all loans until it returns. On the first nonempty scratch callback it borrows
one coarse lane using a successful prior Compute's aggregate high-water hint;
requests within that lane only advance a checked offset. Cold overflow retains
old addresses in separate funded spill loans. A capacity-refused large hint can
shrink to the current request; reclaim/context/ownership failures propagate.
The hint is not a grant and is published only after successful execution.
Zero-byte callbacks borrow nothing. Concurrent Computes share only the atomic
hint, with independent owners and mutex-protected offsets. Normal completion
and Compute failure explicitly return loans; Drop provides cleanup on earlier
failure. Return errors propagate without inventing a refund or idle owner.

Foreign C plugins reveal scratch sizes through callbacks, so their first cold
scratch growth is not a metadata-only admission guarantee. Warm backing reuse
uses the same Host EP pool and funding as native workspace. Returned dynamic
outputs own their payload independently of the retired scratch.

Validation includes strict native preparation with nonzero graph scratch,
changing branches, repeated backing reuse, retained-output independence,
short/misaligned inherited rejection, actual nonzero scratch consumption across
real ORT and both native plugin bridge paths, and Host callback tests for cold
spill address stability, warm single-loan use, zero requests, concurrency,
capacity refusal, optional hints, foreign memory-info rejection and failed
return funding. Final local checks pass: native CPU 2,002/session 291; optional MLAS CPU
2,044/session 293; plugin 384 and shared-mock 6 unit tests; API 106 unit tests,
authority-privacy/integration/doctests; and all 12 required real-ORT shared-EP
conformance tests on cached ORT 1.32. Strict all-target checks pass for the
native API/CPU/session/engine/plugin stack and CUDA (`cuda,gpu-tests`, compile
only). Before/after performance evidence is recorded in [the session workspace report](../benchmarks/session-workspace/README.md): 36 exact-output process runs, qualified controls and overlapping CPU ranges, with no distinguishable regression on this CPU tier.
CUDA device execution and full-model throughput are not established on the
current CPU-only host.


### Lazy inputs borrow the same execution workspace

The general weight-delivery hook `execute_with_inputs` now accepts the existing
borrowed `WorkspaceView`. Its resident adapter forwards to
`execute_with_workspace`; paging-aware implementations consume handles and
scratch together. The executor no longer rejects this combination after having
already prepared its workspace. There is no second resource context, allocator,
lease protocol or public compatibility signature. Resident dispatch keeps its
existing path without wrapping all views as general inputs.

An unresolved initializer remains logically present during workspace planning.
Refinement gets dtype/shape from the weight handle while its actual TensorView
stays absent. Metadata presence never authorizes reading a pointer; runtime
refinement must check the view before accessing input values. Lookup follows the
execution plan's complete ValueId slots, including appended captures, rather than
reconstructing the node descriptor's explicit inputs. This does not extend
which graph or leaf implementations accept lazy inputs or change dynamic-output
protocols.

The same lane owner still covers sequential work. Metadata/shape preparation,
capacity and alignment checks, inherited workspace validation, growth, completion
and return remain in the common executor path. Warm reuse introduces no new
pool checkout, claim/refund or zero-byte request. This is a functional admission
fix, not a measured speedup or a claim of zero total dispatch cost. API forwarding
and unresolved-input rejection tests plus session regressions cover eager and
prepared execution, logical geometry, execution-slot lookup, warm address reuse,
refusal before materialization and successful retry.

Validation on the CPU-only ARM64 host: API 107, session 295 and all 48 public
executor integration tests pass. Strict all-target Clippy passes for API/CPU/
session and for CUDA with `cuda,gpu-tests`; no CUDA device execution is claimed.
The authorized read-only resource-design review found the execution-slot issue
and confirmed its fix. Validation driver and logs remain
`/tmp/nxrt-lazy-workspace-check.py` and `/tmp/nxrt-lazy-workspace-*.log`; all
build/test work held the shared stable host lock.


### Grouped MoE input shares the existing funded lane

BlockQuantizedMoE no longer allocates a gathered input Vec for each expert. One
disjoint rows*hidden f32 region shares the already funded execution workspace
with the maximum serial FC projection scratch. Experts reuse it in sequence;
no expert-count/top-k multiplication, per-group lease or new EP API is added.
Gathering initializes only copied prefixes; unused capacity stays untouched.
The common private f32 binder validates the complete layout once and all typed
region geometry before forming MaybeUninit references. Dense hits still avoid
clearing projection scratch. Full metadata validation precedes an empty-output
fast path with zero scratch, also fixing previously failed zero-hidden decoding.

[Controlled public-session evidence](../benchmarks/moe-grouped-input-workspace/README.md)
records original and refined candidates, exact source/binary hashes, 54 final
outputs, the excluded four-thread control failure and one qualified nine-output
focused repeat. M1 Max single-thread point estimates are -1.08% to +1.30% CPU;
a possible approximately 0.103 us tiny-case cost is retained. The focused four-
thread medians are essentially unchanged with overlapping ranges. There is no
universal zero-overhead or isolated speedup claim. All full outputs/argmaxes
match; physical workspace equals the declared expanded bound and teardown/trim
returns Host usage to zero. The existing 4,096-cycle tests retain zero warm
issuer callbacks, including zero-byte calls. Original conversions, routing,
FC intermediates and output vectors remain outside this scratch migration.

Rebased integration with merged ARM64 math/lazy-workspace work passes native
CPU 2,016 / session 295, MLAS CPU 2,058 / session 297, all 48 public executor
integration tests and both strict CPU/session all-target Clippy configurations.
Measured source is preserved separately and reconstructed exactly by its patch;
those timing numbers are not reinterpreted as a fresh current-main experiment.

### Resource granularity and the minimum EP contract

The allocation unit is an independently owned execution lane or prepared
artifact. Kernel-local regions are byte views within that ownership, not new
quota requests. Serial nodes reuse a peak workspace; simultaneously live work
requires distinct storage. Keep capacity and alignment planning outside the
inner compute loops.

An EP kernel declares its workspace planning capability and a checked
`WorkspaceRequirement` (bytes, alignment and lifetime), then computes using
the supplied `WorkspaceView`. `WorkspaceLayout` partitions composite scratch
without allocating or charging. The executor owns graph preparation and
workspace lifetimes; the provider supplies the backing owner and completion
boundary. A provider bridge delegates the complete allocate/replace/deallocate
workspace lifecycle to that same provider, rather than rebuilding accounting
around its raw allocator.

`PluginExecutionProvider` now forwards those three workspace operations to its
supplied CPU provider. Previously, CPU fallback kernels used the trait's default
raw-allocation adapter and bypassed the CPU managed workspace pool. The actual
two-node QLinearMatMul fallback regression checks one 24-byte peak rather than
16+24 bytes, stable allocation identity over repeated runs, idle reclamation and
readmission, smaller shapes, oversized admission failure and final zero usage.
The graph is built on CPU before installing the actual plugin bridge for
planning/execution, so this regression requires no plugin compiler or dylib.

Pool return retains physical backing and its charge. A sufficient warm checkout
and return must make zero issuer callbacks, including zero-byte calls; actual
refund follows proven physical release. Existing pool tests pin that callback
contract. This is not a claim of zero pool-lock or execution overhead. CPU and
CUDA workspace already carry their charge in managed backing ownership; neither
needs a second outer lease on those paths.

Prepared immutable cache entries retain their own grants at the granularity at
which they can be published and reclaimed, including jointly admitted expert
groups. Cold preparation obtains ownership; stable warm hits reuse it. Do not
combine all weights into a session-wide slab that prevents independent idle
eviction, or introduce a second prediction/credit ledger.

One possible later optimization is retaining a lane during a continuously
active decode phase, returning it at a scheduling yield, phase transition or
request end. Measure the pool round-trip cost before changing that policy.
Do not reuse capture pinning for scheduler ownership, keep a prefill high-water
allocation throughout decode, or reclaim an active/asynchronous owner before
its completion boundary. No such scheduling-policy change is made here.

### Workspace has one physical and accounting owner

`WorkspaceAllocation` now wraps only `DeviceBuffer`; its constructor takes the
buffer, and `into_buffer` returns it unchanged. The optional outer `MemoryLease`,
`ExecutionProvider::reserve_workspace`, CUDA's unused reserve override and global
compatibility-lease quarantine are removed, without compatibility aliases.
CPU, CUDA and Host plugin scratch retain their actual managed backing owner;
quota, physical ownership and release settlement travel together.

The default workspace adapter retains `prepare_mapped_growth` and passes its
grant to `allocate_with_mapped_growth`, or uses the provider's raw allocator.
That adapter does not make a raw allocator governed. Managed providers supply
one allocation transaction or borrow a funded lane. There is no replacement
reserve hook, extra credit ledger or per-kernel quota request.

The default return delegates to the same provider's deallocator. Pool return
keeps backing charged; physical completion can refund it. A failed or abandoned
managed release quarantines the allocation with its charge intact. CUDA's
allocation-specific settlement barrier and replacement wait remain unchanged;
removing an outer lease must not make an accepted enqueue count as completion.
The old compatibility-lease failure test is replaced by real managed backing
and ledger/process-manager checks. Existing incomplete-release, warm zero-issuer
callback and CUDA host-side barrier tests remain applicable.

### QMoE input groups reuse one execution lane

This entry describes the engine grouped-input migration. The CUDA integer-affine
`com.microsoft::QMoE` kernel now also places all routing, expert intermediates,
grouping and floating-operand conversion in one SessionPersistent caller
composite; see [the current ownership contract](MEMORY_ARCHITECTURE.md#integer-affine-qmoe-shares-one-prepared-composite).
Its regions do not introduce separate grants or slot owners.

Integer QMoE now declares metadata-planned StepScoped grouped-input scratch.
The checked bound is rows*hidden*f32, reused across serial experts, without an
expert-count/top-k multiplier. A singleton group borrows its already contiguous
input row; zero- and one-row batches need no gather storage. Multirow groups
reuse ProjectionScratch::gather_rows. Only the main compute thread borrows that
lane; prefetch workers still own weights independently, retaining their ordering,
reader, capacity-proof and join contracts. No extra allocator or quota API is
introduced. Contributions, routing, FC intermediates, row decoder scratch and
transient weight admission remain outside this migration.

[Actual-session evidence](../benchmarks/qmoe-grouped-input-workspace/README.md)
retains both 27-process studies. The first candidate's qualified single-row
+2.20% point estimate motivated the singleton refinement. Final point changes on
M1 Max are -0.51%, +0.81% and +0.47%, with overlapping ranges and passing controls;
no distinguishable regression or gain is established on these fixtures. All 54
full outputs/argmaxes agree. Single-row gather storage is zero; eight-row H=256
storage is exactly 8,192 bytes at one or four threads. Warm workspace borrow and
return have zero recorded issuer callbacks; this is not a claim that existing
transient weight admission makes no callbacks. The scoped cap is not a full
process cap. The CUDA parity CPU oracle is also updated to prepare scratch,
with a real GPU-free f64 regression that caught the missing caller.

### Float MoE gather uses the shared execution lane

Float MoE now follows the same declared-workspace contract as the quantized
variants. Its per-expert driver reuses one input-sized region; the batched
route uses one whole-slot region. Metadata excludes batched execution only
when its largest possible expert group cannot cross the kernel's fixed work
threshold. The threshold is read at kernel creation, so environment or thread
pool changes cannot invalidate an already prepared bound. Execution still uses
actual routing and the original worker-budget/serial-threshold policy.

Singleton per-expert groups borrow the original dense row. All other gathered
values are copied once into the admitted lane, without preliminary clearing.
No per-expert admission, mutable kernel pool or additional EP interface is
introduced. FC intermediates, conversions, routing and the owned result remain
separate work; this is not complete inference-memory accounting.

The real Host pool regression verifies zero warm workspace issuer callbacks
across skewed/noncontiguous routing, threshold boundaries and one/four-thread
pool transitions, with exact output and trim-to-zero checks. Integrated native CPU 2,018,
session library 296, focused native/MLAS MoE 21 each and projection-layout 10
checks pass; strict native/MLAS all-target Clippy passes. The shared-lock public
session comparison covers four geometries and 36 processes, with equal complete
outputs. Candidate CPU medians change -1.19% to +0.90%, with overlapping ranges
and all null/control duration gates passing; observed OS activity limits the
performance evidence. See [the reproducible report](../benchmarks/float-moe-workspace/README.md).
No CUDA runtime or model throughput conclusion follows.


### Float MoE FC intermediates reuse two admitted banks

The former gather-only boundary now includes FC1, optional FC3, activated and
expert-output matrices. The two fixed banks follow their live ranges: nongated
A gather/output and B FC1/activation; gated A gather/activation and B
FC1+FC3/output. Actual prefixes are initialized without treating untouched bank
capacity as initialized. The final accumulation/result, routing, widening and
GEMM-internal storage remain separate. Fixed banks are a conservative explicit
bound; they do not introduce a per-phase allocator or new public EP API.

Native CPU 2,019, MLAS CPU 2,061, session 296 and projection-layout 11 tests pass.
The named prefix primitive passes Miri; the cross-pool resource test also passes
at MLAS budget 1. Strict native/MLAS all-target Clippy passes. Exact-capacity Host
controls reconcile funding, zero warm issuer callbacks and final zero usage.

All 126 formal full outputs and argmaxes agree across the original 72-process
study and one fixed 54-process diagnostic. The original forced split-SwiGLU
control fails its 5% gate and is retained. The diagnostic passes all three
control/duration gates, with overlapping CPU ranges for the two large forced
cells (+1.89%/+0.65%, null +1.38%/-0.70%); a small cost is not excluded. Neither
study proves universally zero overhead, an RSS saving or a model speedup.
[The report](../benchmarks/float-moe-fc-workspace/README.md) retains both protocols,
source/binary hashes, raw batches, conservative capacity costs and host limits.
Consumed probes and the private Miri target were removed under the stable lock,
with receipts; sources, results, shared dependencies and lock inode remain.


### BlockQuantizedMoE final accumulator shares its admitted lane

The final rows*hidden f32 accumulator now joins the existing projection maximum
and gather regions in one checked StepScoped layout. Its lifetime spans all
experts, so it has a disjoint input-sized region rather than aliasing scratch
that an expert still needs. The whole lane is admitted once; serial experts
reuse their regions without per-expert quota calls. Empty output needs no lane.

The accumulator initializes only its actual prefix, preserves accumulation
order, and publishes to the output only after successful expert execution.
Dense hits leave projection scratch untouched. Routing, input conversions, FC
intermediates, compressed strided copies and GEMM internals remain separate;
this is not full process-memory accounting. Retained capacity stays charged
until actual reclamation.

Native and MLAS-reference module tests pass (29 each), as does native strict
all-target Clippy. The resource test pins exact full-lane charges, zero warm
issuer callbacks including zero bytes, and trim-to-zero. The
[102-process report](../benchmarks/block-moe-accumulator-workspace/README.md)
retains the original failed controls and one fixed diagnostic. Qualified ranges
do not establish a distinguishable regression or a general speedup; small costs
are not excluded. No CUDA runtime or model throughput claim is made.

The initial public-session pilot also exposed a generic NodePlan bug: trimming
legal trailing optional inputs leaves the source node and EP metadata with
different positional cardinalities. The performance probe uses the equivalent
omitted-tail form; this unrelated runtime inconsistency requires a separate
plan fix and public-session regression without relaxing EP validation.


### Caller-owned packed projection destinations

The private prepared packed-projection primitive now writes into an initialized
caller destination and returns `Result<()>`. Standalone BlockQuantizedMatMul and
the owned expert driver retain their existing staging allocations and publish
actual tensor outputs only after success. This separates projection arithmetic
from output storage so the shared expert-bank driver can supply admitted regions
next; this step alone does not complete FC intermediate accounting or claim
memory/performance savings.

Packed extents and the canonical decoder resolve through one immutable row view;
prepared production storage and the dynamic test oracle share that representation
and GEMM operand/K order. Destination and scratch extents are validated before
computation. Invalid geometry leaves staging untouched; later compute failures
may modify staging but cannot publish a partial tensor through these callers.
There is no old-signature alias, new public EP interface, pool or issuer call.
Native and MLAS-reference block MatMul tests pass (38 each, one pre-existing
ignored per configuration), as do block MoE tests (29 each) and strict all-target
CPU Clippy in both configurations. Caller-owned poisoned targets, separate prior
outputs, guard regions, short/oversized destinations, malformed input/packed/bias
spans, short scratch, empty axes and partial blocks are covered.


### BlockQuantizedMoE FC intermediates reuse the float expert banks

The packed projection destination seam now feeds the same two-bank private
expert driver as float MoE. One checked StepScoped layout combines maximum
decode scratch, bank A, bank B and the final accumulator. Nongated A holds
gather/output and B holds FC1/activation; gated A holds gather/activation and B
holds FC1+optional FC3/output. Actual prefixes are initialized, and safe short
borrows prevent overwriting still-live inputs. Expert ordering, projection/bias
math, gate indexing and late tensor publication remain. A singleton group
borrows its input row. There is no per-expert grant, new allocator or public EP
interface.

Bank widths share the float rule and cover streamed fallback even on dense
hits. Their conservative retained capacity is now charged; this can exceed the
old declared workspace that omitted FC Vec storage. It does not establish an
RSS saving or a complete process cap. Routing, widening, strided compressed
copies, GEMM internals and final output storage remain separate.

The exact-bit regression compares the packed bank driver with the owned
activation path across two formats, asymmetric and zero axes, row gathering
and poisoned bank reuse. Existing dense/streamed parity remains. It also exposed
and fixed the old owned driver's fused zero-width chunks_exact(0) panic: empty
FC1 has no channel chunks, but zero-K FC2 still runs and adds its bias. Existing
QMoE owned callers retain their legitimate storage contract, with that fix.

[Two complete public-session matrices](../benchmarks/block-moe-projection-banks/README.md)
preserve the original gated CPU regression and a one-line inline-hint refinement.
Each matrix has 144 interleaved whole-process CPU receipts with baseline/null,
setup-only and double-work controls. All six aggregate cells qualify in each;
individual noisy repetitions remain visible. The final single-row gated point
change is +1.4%, so a small cost is not excluded; no universal zero-overhead or
model-throughput conclusion is made. Host resource tests retain exact full-lane
charges, zero warm issuer callbacks and trim-to-zero checks.


### Initialize decoded workspace according to live projections

The one admitted BlockMatMul/BlockMoE/QMoE decode lane now initializes only the
prefix requested by the current projection. A monotone dispatch-local prefix
replaces the full-lane bool: first use clears its live span, larger later phases
clear only new elements, and smaller phases retain initialized storage. A fresh
binding discards the prior dispatch's proof. Prepared helpers pass the owner to
the canonical checked plan split rather than clearing a complete maximum lane.

Capacity/overflow errors precede writes. Decoder failures preserve valid f32
initialization without publishing partial tensor output. Dense hits still leave
decoder scratch untouched. The maximum capacity, charge, logical shard plan and
all grant/refund operations remain unchanged; no finer quota is introduced.
The [state/resource checks and mutation control](../benchmarks/decoded-scratch-prefix/README.md)
cover sparse actual expert groups and pool changes. No inference latency, RSS,
CUDA runtime or complete-process memory conclusion follows from this change.


### Idle workspace pressure uses queue operations

Host workspace idle slots now use a deque. A pressure sweep pops the front
without shifting every remaining owner after each physical release. Warm
best-fit and growth selection still remove from the back by swapping, preserving
the previous Vec's logical order and boxed ownership. Physical release stays
outside the idle lock; an error stops the sweep with remaining slots intact.

The real-ledger regression holds eight independently funded execution lanes,
partially trims them, checks the remaining allocation identities, reuses the
best-fit box without issuer callbacks, and verifies the existing swapped release
order before settling to zero. Existing concurrent, borrowed/foreign return and
incomplete-release guards remain. This changes neither allocation granularity,
EP APIs nor quota operations. It removes repeated queue bookkeeping on pressure;
no inference latency, RSS or universally zero pool-overhead claim is made.


### Structural claim checks travel with their registered factory

`KernelFactory::unsupported_reason` is a pure metadata capability query on the
same versioned factory the CPU provider will use for construction. The seven
migrated Attention/MoE parser gates now live with that factory rather than in a
second operator-name synchronization table. The selected factory receives the effective opset, original
node, shapes and dtypes. The query does not call factory `create`, prepare device
storage, initialize a thread pool or obtain an allocation grant. Existing parsers
may produce small typed attribute values; those are metadata, not execution
preparation. Execution and its coarse workspace/weight ownership paths do not
call the gate.

Native and linked-reference Conv share one rank rule for claim and concrete
construction. Claim treats empty shapes conservatively as unknown rank, matching
the current IR's scalar/unknown ambiguity; nonempty symbolic shapes retain known
rank. Concrete construction still rejects scalar or unsupported/mismatched ranks.
This is a shared structural capability boundary, not a performance-based decline
or a new dynamic-rank execution guarantee.

Regression tests query different registered factory revisions, verify metadata
forwarding and zero kernel construction, preserve existing parser declines, and
cover Conv known/unknown rank behavior. The real ORT plugin regression retains
three unsupported MoE/attention assignments on ORT's fallback provider while their
sessions load successfully. No additional per-op quota or scheduler is introduced.


### Complete the factory-owned metadata boundary

The remaining fifteen kernel-specific CPU claim gates now travel with their
registered factories, including sparse attention/MoE, indexing, Gemm and
DeformConv. Shared ONNX schema gates stay in the provider. Registry resolution
also makes the default domain spellings (`""` and `ai.onnx`) use the same kernel
limits; versioned Scatter and DeformConv queries use the selected factory's
schema revision, matching construction.

CompressedSparseAttention shares a plain attribute parser between claim and
construction. Claim no longer constructs and immediately drops a kernel just to
validate attributes. A constructor observer with a positive control covers this
boundary. No execution buffers, pools or resource grants are prepared by it.

DeformConv's previously unreachable provider check is now a factory capability
query. The shared IR attribute parser retains its required-input and three-to-five-slot
arity checks, with errors now naming the missing slot or invalid count. Empty claim shapes defer rank validation
because the current IR also uses them for unknown rank; known invalid ranks are
still rejected, and concrete execution retains strict shape validation. Tests
cover domain aliases, BF16 schema versions and partially unknown shapes. These
are claim-time changes; they introduce no inference-path quota operations or
performance, RSS or CUDA execution claim.


### STFT frames borrow one execution workspace

STFT's four frame-length f32 buffers now occupy disjoint checked regions in the
existing StepScoped execution domain. One lane serves every frame and batch;
there is no per-frame resource admission or node-owned scratch pool. Existing
zero initialization and DFT/vDSP arithmetic remain unchanged. The provider's
registered planning fact declares metadata preparation.

A present window supplies the exact metadata bound. Without a window, a runtime
frame_length scalar leaves metadata preparation Deferred; execution reads the
Host-accessible scalar and admits precisely four frames before dispatch. No
signal-length-sized reserve, scalar-value memo or new EP API is introduced.
Scalar reads no longer allocate a dense one-element vector. Empty batches still
validate scalar parameters, signal/window views and accessibility, but use no
frame scratch. Missing/short/misaligned backing fails before output writes.

Controls cover poisoned reuse across changing frame lengths, independent f64
references, strided/half input, overflow, empty-batch guards and exact Host
refunds. A real two-node executor prepares max(48,64)=64 bytes, reuses one
managed identity, and separately exercises scalar-only deferred preparation and
48-to-64-byte growth. The real governor observer includes live zero-byte claim
and refund controls; 4,096 warm checkout/execute/return cycles make no further
issuer callbacks. This is an accounting invariant, not zero total runtime cost.

Signal/window materialization, the output staging Vec and Apple DSP setup's
internal allocations remain outside these four admitted regions. This does not
establish complete STFT/RSS accounting, inference latency or CUDA behavior.


### DFT mode correctness before scratch migration

Inspection of the remaining DFT allocations exposed a pre-existing semantic
error: inverse+onesided used a real input, a complex inverse and truncated
complex output instead of IRFFT. The resource migration is kept separate while
this contract is corrected. The shared IR mode now determines accepted input
components, implicit transform length, output length and real/complex width.
CPU execution, symbolic shape inference and plugin runtime sizing consume it.
The [ONNX schemas 17/20](https://onnx.ai/onnx/operators/onnx__DFT.html) and
[official reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_dft.py)
define the same real-inverse contract.

CPU IRFFT pads/truncates positive-frequency bins, ignores imaginary DC and even-
length Nyquist components, reconstructs the conjugate half and uses the existing
normalized inverse transform. Its default length is twice (input bins minus
one); explicit odd lengths are preserved. Output has one real component.
Complete output geometry is checked before writes. Forward and full inverse
arithmetic retain their existing compute-in-f32 contract.

The cuFFT adapter has no IRFFT pack/unpack implementation. Its factory and
claim query now explicitly reject that mode with a CPU selection instruction;
this is not a CUDA execution result. Independent f64 sums cover odd/even sizes,
padding/truncation and batches; a real encoded model exercises RFFT/IRFFT
roundtrips for both schemas. Existing scalar/control, widening, output staging
and per-batch DFT allocations still require their own workspace migration.


### DFT batches reuse the Fourier execution lane

DFT's real/imaginary input and output vectors now borrow four transform-length
regions from the same StepScoped Host execution domain as STFT. A private
TypedWorkspace<f32, 4> descriptor centralizes checked partitioning and initialization;
it owns neither backing nor quota. Both operators use its closure-scoped Host
borrows. DFT constructs one existing DftPlan per execution and reuses it across
all batches, instead of allocating vectors and DSP setup for each transform.
Forward/full inverse/RFFT and the newly corrected IRFFT keep their algorithms.

Metadata preparation is Ready only when axis and transform length follow shape
facts. Present runtime controls remain Deferred; execution requires readable
control views, permits metadata-only intermediate signals for sizing, and reads
rank-zero Host integer controls
without a one-element Vec. Planning and run share checked geometry. Actual
control values are read again before binding, so stale smaller backing fails
before DSP setup or output writes. Workspace is `4 * dft_length * sizeof(f32)` bytes, independent of signal
length, output size, batch count or the sum of graph-node requirements.
Empty batches retain view/control/output validation and skip scratch/DSP setup.

Every scratch region starts with valid +0 f32 values before references form.
Within one call the copied positive-frequency prefix is fixed; input padding is
never modified by the transform, and IRFFT mirroring writes only negative bins.
Thus padding stays zero across batches without another clear or admission.
Separate calls initialize only their requested prefix of a retained larger lane.
The existing ownership domain still handles concurrency, growth, pressure and
physical release; no per-frame/expert grant or kernel-owned pool is added.

Controls cover poisoned and fresh uninitialized backing, canaries, odd/even
IRFFT, truncation/padding, scalar changes at the same address, unreadable logical
controls, overflow, missing/short/misaligned backing, and unchanged outputs on
refusal. A live constructor positive control distinguishes one DSP plan across
three batches from a disconnected observer. The real governor control performs
4,096 preparation/checkout/execute/return cycles with zero further issuer calls,
including zero-byte callback controls; an active lane cannot fund another call.
The encoded two-node RFFT/IRFFT session uses one 128-byte peak under an exact
Host quota, refuses at 127 bytes, reuses its managed identity and re-admits after
idle reclamation, then settles to zero.

Only these four f32 buffers are funded by the declared requirement. Input
materialization, output staging, small rank/index metadata and vendor DSP setup
allocations remain separate. This is not complete RSS accounting, zero total
runtime overhead, a latency/throughput measurement or CUDA execution evidence.


The shipping ORT plugin test disables fallback and verifies that both DFT nodes
are assigned to cpu_ep. Full-complex and real-inverse roundtrips change runtime
length 8/3/9/8 at unchanged input shapes, with numerical truncation/padding
checks on both batches. It exposed an existing dtype-descriptor omission:
DFT advertised only floats despite integer controls. The exact edge union and
per-slot floating signal/output, integer length, and int64 axis declarations
now agree with the kernel. The cross-crate shape fixture also receives the
explicit forward direction required by the shared DFT shape interface.


### RMS schemas share admitted staging and one statistics pass

Standard RMSNormalization and both registrations of SimplifiedLayerNormalization
now use one kernel with an explicit schema kind. The allocating `rms_norm_dense`
interface and the duplicate Simplified execution path are removed. The shared
metadata geometry checks axis, normalization/group products, broadcastability,
contiguous stride arithmetic, floating storage and complete output shapes before
planning a two-region StepScoped Host layout. It contains `X.numel()` f32 values
for Y and, only when requested, one f32 inverse RMS per group. An absent optional
statistics slot consumes no storage. No kernel-owned pool or per-row grant is
introduced; the existing execution owner admits the graph's maximum requirement.

Both checked regions are initialized before f32 references form. All groups are
computed into staging before either output is published, preserving X/output,
scale/output and statistics/input aliases. The same inverse RMS supplies Y and
the optional statistic; the old second sum-of-squares pass is removed. The
existing SIMD exact-identity path and multiply order remain intact, while only
the general broadcast path builds rank-sized index strides. Empty outer groups
need no scratch but still validate Host accessibility, views and outputs; an
empty normalized width remains an error.

The regression set covers fresh and poisoned scratch, prefix canaries,
missing/short/misaligned backing, optional holes, invalid shapes/dtypes,
overflow, aliases, half/strided inputs and exact statistic bits. A real governed
lane executes 4,096 warm preparation/checkout/compute/return cycles without
further issuer callbacks, including zero-byte callback controls, and refunds
physical capacity only after idle trimming. An encoded two-schema session shares
one 40-byte peak, refuses a 39-byte Host quota and re-admits after reclamation.
The shipping ORT plugin test disables fallback, requires both RMS schemas on
cpu_ep, and changes shapes 2x3/3x9/2x3 while checking Y and inverse RMS values.

Input/scale widening and the general broadcast path's rank-sized stride vectors
remain outside the staging requirement. These checks prove ownership and
numerical behavior, not zero total allocations, inference latency, complete RSS
accounting or CUDA execution.


### One private initialized-float binding contract for CPU kernels

DFT, STFT and both RMS schemas now use `TypedWorkspace<f32, N>` instead of separate
Fourier and RMS raw-pointer initialization code. A kernel supplies named f32
region lengths, keeps its own geometry and declares the same existing
StepScoped requirement. The descriptor uses fixed-size stack arrays and the
existing `WorkspaceLayout`; lengths are derived from checked region byte sizes
when binding rather than retained as a second source of truth. It owns no
backing, quota, lock or pool. This is a CPU-private helper, not a new EP ABI.

Complete capacity/alignment/address validation precedes all memory writes and
the callback. Each region is initialized before its f32 slice exists, including
fresh backing; empty regions use aligned dangling pointers. A higher-ranked
closure and fixed return type keep scratch references within the synchronous
borrow. Callers still prove Host accessibility, live exclusive ownership and
disjointness from tensors. Their existing view, optional-output, overflow and
empty-work guards remain in the kernels. The helper introduces no per-frame,
row or expert admission. Scalar integer reads move to the dtype module,
independent of workspace ownership, with invalid-view errors naming the op and
control input.

Exact source comparison preserves the DFT/STFT execution and tests and the RMS
math core. The binding regressions retain fresh/poisoned four-buffer controls
and cover unequal region lengths, optional empty holes, no regions,
missing/short/misaligned/wrapping backing, no writes/callback on refusal and
reuse after a consumer error. Scalar controls cover signed values, unaligned
backing, byte offsets, invalid views and Host-accessible unified-memory metadata.
Existing per-kernel and governed-lane tests, public sessions and fallback-disabled
real ORT plugin tests remain the integration checks. Byte bounds, arithmetic,
retention and grant/refund ownership are unchanged; no timing or CUDA-runtime
claim is made.


### LayerNorm and Skip share admitted output/statistics staging

LayerNormalization (default and private fused domain) and SkipLayerNormalization
now share one explicit schema-kind kernel, checked metadata geometry and
`TypedWorkspace<f32, 4>`. The allocating `layer_norm_dense` return interface and
Skip's residual-sum Vec are removed. The regions are Y, requested mean,
requested inverse standard deviation, and Skip's full residual sum. Standard
LayerNorm has no sum region; omitted statistics occupy zero bytes. Skip keeps
its sum alongside Y because it is both the normalization input and an optional
published output. No new EP protocol, kernel pool or per-row grant is introduced.

The original f32 mean/variance/scale/beta arithmetic is preserved. Skip still
computes `(X + skip) + bias_or_positive_zero` before normalization, including the
final +0.0 when bias is omitted. Beta and bias remain independently optional.
All live operands are consumed and every group computed before any output is
published, preserving input, residual and parameter aliases. Skip Y/sum retain
activation storage dtype, while each statistic independently supports Float32
or activation dtype. Absent optional output holes are not validated or written.

Preparation checks required slots, normalized width, group and stride products,
parameter element counts, dtypes, complete output shapes and total layout
capacity. Empty groups retain parameter/view/Host/output checks and need no
scratch. Empty normalized width remains an error; axis==rank is now rejected in
accordance with the [ONNX axis range](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html).
The initial migration retained the full-normalized-parameter element-count
rule. The affine-broadcast follow-up below replaces that restriction for
standard LayerNorm; Skip retains its complete-parameter contract.

Regression controls cover exact optional-region bounds, independent beta/bias
holes, fresh and poisoned backing, prefix canaries, missing/short/misaligned
scratch, view/shape/overflow refusal, operand/output aliases, signed zero,
strided half input/output and existing bf16 literal goldens. A real governed
lane alternates 48-byte standard and 80-byte Skip calls for 4,096 warm cycles
without additional issuer callbacks (including zero-byte observer controls),
refuses concurrent reuse and refunds physical capacity only after idle trim.
An encoded two-node public session uses one 80-byte peak, refuses 79 bytes,
reuses its managed identity and re-admits reclaimed storage. The shipping ORT
plugin disables fallback, assigns both nodes to cpu_ep, changes shapes
2x3/3x9/2x3 and checks all six published outputs against f64 references.

Input widening remains outside these four staging regions. This is not complete
RSS accounting, a zero-total-overhead or inference-speed measurement, or CUDA
runtime validation.


### LayerNorm affine broadcasting without expanded coefficient storage

Standard LayerNorm now validates Scale and B by right-aligned, unidirectional
broadcast into X. Preparation classifies each operand as a complete normalized
row, scalar, or general broadcast. The complete-row route preserves direct
coefficient indexing; the general route derives indices from tensor geometry.
No expanded coefficient tensor, new staging region, owner, or quota operation
is introduced. Skip and the existing f32 normalization arithmetic are unchanged.

CPU-scoped decomposition fusion accepts only affine shapes proven not to expand
X: matching dimensions (including the same symbol) or literal singleton axes.
Unrelated symbols decline. The complete reduced suffix must be provably
nonempty, including positive lower bounds for symbolic widths. Immutable scalar
coefficients and axes resolve through PassContext for inline and mapped weights
in both matching and application; absent or malformed bytes and overridable
initializer inputs decline. This also supplies the shared scalar reader used
by GELU and attention fusion, without materializing mapped weights.

Regression coverage includes 486 independent f64 coordinate cases, equal-size
cross-group counterexamples, aliases, four float storage types, invalid geometry,
and a 4,096-cycle governed workspace test spanning complete/scalar/group/Skip
routes with no warm issuer calls. A public session changes affine shape classes
while retaining one 112-byte allocation (111 bytes refused); an encoded nine-node
decomposition fuses to one CPU node and uses 96 bytes (95 refused). Six real ORT
plugin fixtures disable fallback and check Y and both statistics twice each.
Mapped scalar/axes success and malformed/missing/overridable refusals have
optimizer regressions. These checks establish correctness and allocation
invariants, not measured inference speed or CUDA validation; f32 compute and
existing stash-type behavior are outside this change.


### LogSoftmax uses one admitted output-staging region

Both legacy (opsets 1–12, flattened suffix) and modern (opset 13+, one selected
axis) LogSoftmax factories now declare metadata workspace planning. The kernel
replaces its per-execution full-output Vec with TypedWorkspace<f32, 1> on the existing
StepScoped lane. Its bound is exactly four bytes per X element, zero for empty
input. Unresolved Y metadata derives the same bound from X; resolved Y must
match the complete input shape. No kernel pool, extra owner or per-slice grant
is introduced. Input widening remains outside this output-staging region.

Geometry validates arity, floating dtypes, required presence, axis, complete Y
shape, dense suffix products and addressable f32 layout capacity. Execution
validates both views and Host accessibility before binding scratch. Missing,
short or misaligned backing refuses before writes. Empty work still validates
views and requires no allocation. All X reads complete before publishing Y,
including overlapping input/output and strided half storage. The f32
log_softmax_slices function is byte-for-byte unchanged, preserving reduction
order and both opset semantics.

Regression coverage includes independent f64 coordinates for every positive and
negative axis under both semantics, all four float storage types, invalid views,
fresh/poisoned scratch, prefix canaries, geometry overflow, and 4,096 warm pool
cycles without issuer callbacks (including positive zero-byte observer controls).
Encoded public sessions cover two nodes sharing one 96-byte peak (95 bytes
refused), dynamic 2x3x4/1x2x5/2x3x4 shapes, identity reuse, idle trim and
re-admission for opsets 11 and 13. Real ORT plugin fixtures disable fallback,
require LogSoftmax assignment to cpu_ep and check both versions numerically
across those changing shapes. These are correctness and allocation invariants,
not inference-speed or CUDA-runtime measurements.


### BlockMoE checks actual numerical stages across cache-budget routes

The optional-MLAS mixed-format expert regression used a full-width GEMM as the
reference for a streamed one-column GEMM, then judged only their final residual.
On main `4eaa636`, case 0 element 329 was -0.10598877 versus -0.10596141; the
existing output-relative threshold rejected this cancellation-sensitive result.
That failure independently reproduced before the CPU output-safety change.

The test now observes actual FC1/optional FC3/FC2 boundaries within the shared
expert workspace. Every projection is checked against f64 dot+bias arithmetic on
its actual f32 input, using gamma_(2K+1) times the sum of absolute products plus
bias. Reference rounding and propagated input uncertainty are explicit. An
additional full two-projection affine reference propagates those error bounds
through the identity fixture and routing. The canonical decoder still produces
f32 weights; arithmetic is independent, while decoding has separate regressions.
Normal/zero operands are checked before f64 conversion. The fixture's underflow
allowance is not a general certification of arbitrary hardware DAZ modes.

Identity, ReLU and tanh-GELU activation results are bit-checked from independent
formulas. SiLU/SwiGLU use a fixed fixture-local eight-ULP contract, separate from
the GEMM bound; each case logs maximum ULP, position and operands. Expert-order
scatter is reconstructed bit for bit from observed FC2 outputs and supplied
normalized routing weights. This oracle deliberately targets the all-experts,
all-rows fixture; it does not silently assume those properties for arbitrary
routed models.

Negative controls exercise the actual checker: projection error above its
computed budget, wrong gather order, wrong bias slice, excessive activation ULP
(with FC2 recomputed to avoid an unrelated downstream refusal), and a one-bit
scatter change. Existing streamed/denied byte equality, whole-group admission,
no partial expansion and no retained streamed weights remain unchanged. Scoped
thread-local observations are non-Send, clear after unwinding and exist only
under cfg(test). Production arithmetic, routing, cache owners and grant/refund
paths are unchanged; no timing or CUDA validation is inferred.


### Softmax preserves direct output and admits fallback staging

Softmax and LogSoftmax share checked axis, output-shape and capacity geometry.
Softmax metadata preparation reports Deferred for nonempty f32 Y because tensor
metadata cannot prove address alignment or input/output disjointness. Execution
refinement uses actual Host views without reading tensor values: safe direct
output needs no scratch; other routes admit one StepScoped `4 * numel` f32 lane.
Missing actual output views retain the complete conservative staging bound.
Non-f32 Y has that known bound at metadata preparation; empty shapes need none.
Deferred preparation is deliberately not a completed zero-byte reservation.

The borrowing predicate is shared with input widening: a source materialized
into owned storage can alias direct Y because materialization finishes first.
Borrowed f32 input/output overlap instead uses the admitted lane. This removes
the ungoverned per-call output Vec without adding a pool, owner, or per-row grant.
Input widening allocations remain outside this output-staging budget. Reducer
arithmetic and ISA dispatch remain unchanged. Tests check direct execution under
zero workspace capacity, fallback refusal/reuse/trim, dynamic shapes in both
opset families, real ORT assignment with fallback disabled, and 4096 warm cycles
with no governor callbacks (including zero-byte callbacks). These correctness
checks do not establish zero latency overhead or CUDA behavior.


### Typed workspace and LpNormalization storage staging

The CPU-only borrowed workspace helper is now `TypedWorkspace<T, N>` for
`Copy + Default` elements. All existing f32 callers use the same checked layout,
Host lane, HRTB closure and whole-binding validation; no compatibility alias or
new allocation owner remains. Regions are initialized with valid `T::default()`
values before constructing typed slices. A generic Default is not assumed to be
all-zero bits. Zero-sized types are rejected; byte counts and alignment come
from `Layout::array::<T>`.

LpNormalization now requests one StepScoped `numel * sizeof(storage T)` lane.
It computes reductions and division in the original accumulator type (f64 for
f64 storage, f32 otherwise), narrows each completed result directly into staging,
and publishes after all source reads. The previous accumulator-output Vec and
converted-output Vec are removed. Input materialization remains outside this
output budget and preserves arbitrary input/output aliasing. Checked metadata
planning validates exact dtype/shape, axis and indexing capacity, including empty
shapes; actual execution validates Host views before binding scratch. This
introduces no per-slice grants or new public EP API and makes no timing/CUDA claim.


### LpNormalization reuses its lane for the complete input snapshot

LpNormalization now copies the original storage elements directly into its
admitted staging lane, then normalizes that snapshot in place. Every dense
coordinate belongs to exactly one `(outer, inner)` reduction group. Each group
finishes its norm before overwriting its own values and cannot read another
group's outputs. The real output is still published only after the complete
snapshot and all reductions. Accumulator type, reduction order, TINY floor and
final storage conversion stay unchanged. The last matrix-sized input Vec is
removed without adding a lane or changing the `numel * sizeof(T)` peak.

The storage-copy visitor is shared by allocating `to_dense` reads and borrowed
`to_dense_into` reads. It validates Host access, dtype, view geometry and checked
capacity before accessing elements; the borrowed route also validates complete
destination length before writes. Copies preserve raw storage bits, including
half/f64 and NaN payloads, rather than widening through f32. Rank-sized indexing
metadata remains separate from tensor payload admission; high-rank indexing may
still allocate. A real global-allocator probe uses the old allocating copy as a
positive control and verifies that kernel executions allocate no matrix-sized
payload for any of the four float storage types. Existing public peak/trim and
4096-cycle warm governor tests retain their original bounds and callbacks.


### MeanVarianceNormalization preserves direct output and governs snapshots

MeanVarianceNormalization now lives in its own CPU module within the existing
ops-cnn capability group. It reuses the shared MVN axes layout and NormAcc
computation domain. All actual views are validated for Host access, exact
metadata shape/dtype and computable origin/range overflow before any reads or
writes, including empty inputs. This closes the contiguous-pointer route's
missing Host guard and rejects mismatched actual bindings.

Nonempty metadata preparation returns Deferred because stride and alias safety
are not known from shapes. Execution refinement needs no scratch when both
actual views are contiguous and either disjoint or have the same element
origin. Raw unaligned scalar loads/stores preserve packed-address direct paths.
Shifted overlap and strided views use one StepScoped storage-T snapshot lane;
missing actual views keep the complete `numel * sizeof(T)` bound. Empty metadata
has no workspace requirement after geometry validation.

Both routes share the original mean/variance loop, including f64 accumulation
and `sqrt(variance)+epsilon`. Snapshot execution first copies every original
storage element into the admitted lane, normalizes disjoint groups in place,
and publishes the real output last. No matrix-sized input/output Vec remains;
no owner, pool or per-group grant is added. Scoped rank metadata can still
allocate. Direct zero scratch does not imply zero Host allocations or latency.


The MVN public regression uses actual fixed-stride persistent input/output
bindings, with physical padding and logical subshapes. Ordinary host Tensor
uploads currently borrow/copy a contiguous byte prefix; custom non-contiguous
layouts are now rejected during shared input binding for both prepare and run,
with a named input and an explicit materialization/fixed-stride-binding remedy.
They are no longer silently interpreted as dense storage. Canonical contiguous
strides remain accepted regardless of the layout format hint. General strided
host Tensor uploads require a separate explicit ownership/transfer design;
this change does not add an implicit gather allocation.


Real ORT testing also exposed a missing MVN entry in the shared plugin's
shape-preserving route table; it now uses the existing SameAsInput(0) mechanism.
ORT 1.32.0 (current PyPI release at validation) does not ship the ONNX MVN-29
schema. The integration fixture therefore supplies a test-local schema
descriptor in the standard domain and enables unreleased opsets only in that
session. This is schema transport, not an alternate implementation: the
schema-only CPU callback returns an error if invoked, a positive control checks
that guard, and actual inference must assign the node to cpu_ep with zero
schema-callback executions. This proves the real plugin compute path under the
provided descriptor, not built-in ORT MVN-29 schema compatibility. Native Session
tests exercise the unmodified MVN-29 attributes and computation directly.
Official schema: https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html


### LogSoftmax reuses staging for widening and streams narrowed output

LogSoftmax now widens non-borrowable input directly into its existing admitted
f32 lane, then reduces that snapshot in place. Each logical coordinate belongs
to one `(outer, inner)` group. A group's maximum and exponential sum finish
before its first write; no later group can read that group's results. The
arithmetic loop is shared with the unchanged contiguous aligned-f32 borrowing
route through statically dispatched source accessors. The original reduction
order, f32 computation (including f64 input conversion), and storage narrowing
are preserved for both opset families. Real output is published only after all
source reads, including shifted overlap.

The allocation witness also exposed a second payload: scalar output narrowing
collected a complete storage-T Vec before scattering. Typed writes and scalar
narrowing now share one private checked scatter visitor, which converts and
writes one element at a time. Existing f16/bf16 SIMD and f32 memcpy routes remain
unchanged. The source/output disjoint contract already required by those routes
is explicit; production callers were audited for owned or workspace-backed
results and direct-output bypasses. There is no new snapshot owner, pool,
per-group request, or workspace-bound increase: LogSoftmax still uses one
StepScoped `4 * numel` lane. Rank-sized indexing metadata may still allocate.

Real allocator tests cover all four float storage types, contiguous/strided
sources, padded strided outputs, every axis and both opset semantics. A real
allocating storage-copy positive control verifies the counter; 32 repeated calls
per combination allocate no matrix-sized payload. Immutable coordinate oracles
check storage-converted output bits and padding. Separate tests exercise both
directions of shifted overlap (including NaN/Inf), unaligned negative-stride
narrowing with exact byte canaries, and short-producer refusal before writes.
Existing warm governor, peak refusal/reuse/trim, public Session and real ORT
assignment tests remain in place. These are correctness and allocation results;
no latency or CUDA-runtime improvement is inferred.


### Softmax admits input snapshots within the same single-lane peak

Softmax no longer allocates a tensor-sized f32 Vec for input widening or layout
materialization. A non-borrowable source now admits its required `4 * numel`
payload, including when real Y can be written directly. This makes the previous
owned input copy part of the workspace bound. Metadata can return Ready(full)
for non-f32 X; only potentially borrowable f32 X with f32 Y remains Deferred.
Actual-view refinement retains NONE for safely borrowed contiguous aligned f32
X and eligible disjoint Y. Missing real views retain the complete bound.

The admitted lane first receives every converted source element. Eligible
contiguous f32 Y then uses the original out-of-place row/interior reducer
reading that snapshot, including when source tensor storage overlaps Y. Other
outputs normalize the snapshot in place before streaming narrowed output.
Borrowable input with an ineligible Y keeps the original out-of-place reducer
into the lane. Input/output storage conversion, arithmetic order and CPU ISA
selection are unchanged. The interior-axis loop is shared through static source
accessors; disjoint reduction groups finish max before overwriting, read each
original value once for exp, then normalize their own outputs.

Untransformed in-place worker runs now call the existing row kernel once for
the whole run, matching the out-of-place batching. The scale/mask driver keeps
its cache tiles when it actually transforms data. A static callback regression
checks actual call spans above the tile boundary: no-transform batches once,
while scale-only and mask routes retain both tiles. This avoids increasing
single-thread row-kernel prologues as a consequence of snapshot reuse; it is not
a latency measurement.

No additional owner, pool or per-row grant is introduced. One StepScoped f32
lane replaces the previous separate input/output temporary payloads; scalar
output conversion uses the shared streaming writer. Tests retain all original
materialized-source overlap cases, add unfunded/short refusal with unchanged
backing bytes, compare in-place/borrowed reducer bits at boundary widths with
NaN/Inf, and verify four storage types, both opset semantics, every axis, strided
outputs and direct f32 outputs with real allocation positive controls. Existing
4096 warm iterations also execute a widened source within the same retained
allocation identity without issuer callbacks, including zero-byte callbacks.
Rank-sized metadata can still allocate. No CUDA-runtime or latency claim is made.


### BatchNormalization uses one storage snapshot and read-only channel parameters

Inference BatchNormalization now has its own ops-cnn module and Metadata
workspace registration. It replaces the separate storage/Acc copies of five
inputs and the Acc/storage output Vecs with one StepScoped
`numel(X) * sizeof(storage T)` lane. The complete X snapshot is normalized in
place; real Y is published only after all input and parameter reads finish.
Four private validated rank-one readers consume the original channel parameters
without copying them. Parameters may overlap X, real Y or one another because
those tensors remain untouched throughout computation. There is no per-channel
grant, new owner, pool or public workspace API.

Metadata checks five present same-storage-dtype inputs, parameter shape `[C]`,
complete Y shape/dtype, rank, checked dense/spatial products and storage capacity.
N=0 needs no scratch while retaining parameter validation; C=0 or empty spatial
axes keep the previous refusal. Runtime proves Host access, rank/strides and
computable address extrema before binding scratch. Channel readers support
negative strides and unaligned origins, with singleton strides normalized to
zero. Checked i128 extrema bound pointer arithmetic to an addressable host span;
actual backing allocation bounds and lifetime remain the owning EP's contract.

Each value keeps the original `(x-mean)/sqrt(variance+epsilon)*scale+bias`
operation order. Computation stays f64 for f64 storage and f32 otherwise; each
completed result is converted to storage T exactly once in the snapshot. All
existing norm-family arithmetic tests remain, now using prepared execution.
There is no compatibility alias for the old BatchNorm factory/module path.

New tests check four storage types and exact result bits, reversed/strided
parameters, shifted X/Y overlap and each parameter aliasing both or each other,
Host/address guards before writes, whole-binding refusal/canaries and fresh
uninitialized scratch. Real allocator controls observe both former X and
channel-sized copies; repeated calls admit no tensor/channel-sized payload.
A retained allocation runs 4096 cycles without issuer callbacks, with positive
zero-byte controls and exact idle trim/refund. Public encoded-model tests cover
two nodes sharing one peak, dynamic channel counts, empty batch, one-byte-short
refusal and trim/readmission. Real ORT fixtures require cpu_ep assignment with
fallback disabled across dynamic shapes. These establish correctness and
allocation invariants, not inference latency or CUDA execution.


### Affine expert banks borrow validated storage without parameter copies

The standalone affine MoE and affine grouped projection now share one private
`affine_bank` module. It holds lifetime-bound, validated rank-three readers for
packed code bytes, f32 scales and optional packed zero-point bytes. Contiguous,
padded, reversed and unaligned storage are consumed directly; no bank-sized
scale Vec or non-contiguous codes/zeros materialization remains in this reader.
The corresponding bank operand slots advertise strided input support so the
executor does not insert a gather before the reader.

Every bank's dtype, shape, Host access, checked count and address span is
validated before any scale or packed-byte reads. This fixes the old contiguous
byte borrowing path's missing Host check. Checked pointer-domain extrema cover
negative strides and singleton dimensions; actual allocation bounds and lifetime
remain the owning EP's contract. Positive/finite scale validation retains the
complete scan, including experts not selected by routing, with row/block/value
context on failure.

Each dot product resolves its expert/row origin once outside the weight loop.
The dense route uses a flat row offset; strided storage resolves the two leading
axes once. Within the loop, nibble order, zero-point subtraction, multiplication
by scale and `sum + x * weight` order are unchanged. Affine grouped projection
uses that same dot routine rather than recomputing a flattened expert index per
weight. Both callers retain full Y staging and publish last, so parameter/X/Y
alias safety is preserved. No owner, cache, lock or quota request is added.

Bit-exact tests cover optional zeros, odd depths/block counts, padded/reversed
strides, packed f32 origins and NaN/Inf activation values. Guard-only cases prove
that even a late non-Host zero-point bank is refused before any scale read.
A real allocator witness makes each bank much larger than its activation,
route and output; separate allocating code/scale/zero controls verify the probe,
then repeated grouped executions allocate no bank-sized copies. This scope does
not claim zero allocations for the entire kernel or a measured latency gain.

### Affine MoE uses one whole-execution mixed workspace

The standalone affine MoE now advertises metadata workspace planning. Metadata
validation returns Deferred because row borrowing depends on actual strides and
pointer alignment. Execution refinement ignores stale cached hints and resolves
one StepScoped layout containing f32 Y staging, optional X/router row scratch,
selected logits, probabilities and activated projection, followed by usize
routing indices with checked alignment padding. For T tokens, hidden width H,
E experts, intermediate width I and top-k k, the float count is
`T*H + optional(H) + optional(E) + 2*k + I`, with E usize indices. Missing actual
views retain both optional rows as a conservative bound.

A row borrows aligned contiguous f32 values even when its outer axes are padded,
transposed or reversed. Unaligned origins and noncontiguous inner axes copy only
one row into the admitted scratch; the same regions are reused across all
tokens. All eight operand slots advertise strided support, avoiding an upstream
full-tensor gather. Expert banks retain their borrowed readers. Real Y is
published after all source reads, preserving overlapping X/router/bank bindings.

All Host/address proofs precede workspace binding. Complete capacity/alignment
validation precedes initialization of either typed sublayout and precedes any
scale or activation reads. The existing complete positive/finite scale scan and
routing/threshold/activation arithmetic remain intact. No token/expert request,
new owner or cache is introduced: the existing retained Host pool supplies the
whole lane. A recording-governor witness runs 4096 real executions with one
identity and no additional issuer callbacks, including zero-byte callbacks;
positive controls observe both zero-byte admission and release. Idle trim refunds
the complete retained charge. Public Session tests check exact-budget admission,
one-byte-short refusal, dynamic shrinking/reuse and trim/readmission.

Bitwise reference tests cover ties, nonfinite logits, threshold masking, all
k/E relationships, row strides and overlapping source/output bindings. A real
allocator witness independently detects each former payload allocation and
observes no allocation at least as large as its smallest tested payload during
execution. Small bookkeeping allocations are outside that assertion. These are
memory/accounting checks, not latency measurements; CUDA remains untested.

### Packed grouped projections prove Host access before borrowing bytes

The nonempty standalone `PackedGroupedMatMul` route now uses the same numeric
Host-span validator before reading routing indices, activations or packed bytes.
Its common metadata contract is device-agnostic, so it does not itself authorize
a CPU raw slice over a device pointer. Packed bank device access and address
wrap failures report the operator and `packed weights` input while leaving Y
unchanged. Contiguous byte offsets remain supported; empty executions keep the
existing read-free return. This adds no data copy, resource owner or admission.

### Affine grouped projection admits one mixed execution workspace

The CPU affine grouped projection now plans one StepScoped lane containing
f32 Y staging, one conservative K-element activation row for nonempty output,
and one usize per route, including alignment padding. Metadata returns Ready;
there is no new execution refiner or owner. The row remains in the bound even
when aligned contiguous X rows borrow directly. Outer padding/negative strides
do not prevent that borrow. Other X layouts copy one row into the reusable
scratch, rather than materializing the full activation matrix. All operand slots
advertise strided support so upstream gathers do not hide those paths.

The mixed layout and validated Host-row reader are extracted from affine MoE
into CPU-private helpers shared by both kernels. These helpers only compose and
borrow the existing lane: they allocate no backing and add no pool or admission
protocol. Full outer binding precedes initialization of either typed region.
MoE retains its previous exact execution refinement and numerical behavior.

A fallible form of the existing numeric storage walk converts INT32/INT64 routes
directly into the admitted usize region, retaining first-error position/value.
The infallible storage walk delegates to that same loop. Native/Packed grouped
callers use the same validated route source with their existing usize Vec;
their other payloads are outside this change. A route source carries checked
count/Host/address geometry, avoiding a second span validation when filled.

Affine geometry has a shared device-independent metadata path for concrete or
unresolved output shapes. The live-view wrapper still validates descriptors,
including for CUDA consumers; it adds no Host restriction there. CPU storage
proofs precede complete scratch binding and all value reads. All routes are
validated before the complete scale scan, and Y is published last for source
alias safety. Unlike Packed's early empty return, empty Affine still validates
route values and all scales. N=0 can therefore require route scratch even with
no output payload; an entirely empty route/output lane needs no admission.

Tests cover bitwise old-order results, optional zeros, INT32/INT64, odd depths
and block counts, nonfinite X, padded/reversed/unaligned bindings, source/Y
shifted aliases, untouched strided-output padding, and refusal before source
reads or scratch initialization. Fresh mixed backing is initialized before typed
references are formed. A retained-governor witness checks 4096 real executions,
one identity and no warm issuer callbacks, including zero-byte controls and
exact trim refunds. Public Session checks exercise exact/one-byte-short budgets,
dynamic shrinking/empty/re-admission and fixed-stride X bindings. Real allocator
controls detect the former activation, integer/usize route and output payloads;
repeated executions use the admitted lane without payload-sized heap copies.
Small bookkeeping allocations remain outside that assertion. These checks make
no latency or CUDA hardware performance claim.

Logical-prefix capacity planning now distinguishes declared static axes from
physical padding. A binding that exposes a logical shape keeps every Static(n)
axis at n after checking logical==n and physical>=n; symbolic axes still reserve
physical capacity. The same rule covers ordinary and fixed-stride logical-prefix
bindings. Unknown declarations remain conservative, and non-logical-exposing
bindings keep the previous physical-shape rule. Physical strides, backing span
and output capacity facts are unchanged. Shared-symbol capacity conflicts still
fail explicitly. This lets a [T,4] input plan against physical [3,8] with logical
[1,4] without pretending the contraction width can grow to eight.

Public aliased I/O coverage also exposed Identity's false non-overlap assumption.
Identity now uses checked raw byte sizes and validates both nonempty Host spans
before an overlap-safe byte move. It retains its explicit contiguous-layout
contract and copies packed storage bits unchanged. CPU provider byte copies use
the same overlap-safe semantics because distinct borrowed buffer descriptors may
name overlapping ranges. Exact/two-direction shifted overlaps, non-Host/wrapping
pointers and a real same-backing Identity Session binding are covered without
introducing workspace or quota requests.

### Packed grouped projection reuses the whole-operation workspace

For nonempty output, CPU PackedGroupedMatMul now admits one StepScoped mixed
lane for f32 Y staging, one conservative K-element X row and usize routes, with
checked alignment padding. Metadata returns Ready. It reuses the existing
ValueIndexWorkspace, Host Rows and validated RouteSource helpers: no new owner,
pool or per-token/expert/block grant is introduced. Aligned inner-contiguous X
rows borrow even with padded/reversed outer axes; other X rows and INT32/INT64
routes stream into admitted storage. Packed bytes remain a borrowed contiguous
bank, and the existing 256-float block decoder buffer stays on the stack. This
change does not add strided packed-bank support or claim that upstream gathers
for unsupported layouts are governed by this kernel lane.

Complete Host/view/address proofs and mixed binding precede all value reads.
The positive-K decoder and accumulation order remain unchanged, and true Y is
published after every X/route/packed read, including overlapping I/O bindings.
Strided Y publication visits logical elements without touching padding. Shared
packed metadata geometry derives concrete/unresolved output shapes while the
existing live wrapper keeps descriptor validation and CUDA device independence.

Packed retains its read-free return when output lanes are zero, with requirement
NONE even if routes are nonempty. This differs intentionally from Affine's data
validation on empty output. For K=0 with nonempty Y, the kernel still validates
Host backing and every route before publishing initialized +0. It neither forms
a zero-length packed slice nor computes an unnecessary expert*N intermediate;
large expert geometry cannot overflow a row offset for an empty reduction. Null
views still fail the existing descriptor contract.

Bitwise tests cover every supported packed format, INT32/INT64 routes,
padded/reversed/unaligned X/routes and contiguous packed-bank origins, all three
source/Y shifted overlaps, strided Y padding, zero lanes and zero K, invalid
routes and refusal before scratch initialization or data reads. A retained-owner
witness exercises 4096 real executions with no additional issuer callbacks,
including zero-byte controls, and exact trim refunds. Public Session checks use
exact/one-byte-short budgets, dynamic shrinking/empty/trim/readmission and fixed
physical strides with logical prefixes. Independent allocator controls detect
former X/route/output payload sizes; repeated executions allocate no tested
payload-sized buffers. Small bookkeeping allocations are outside that claim.
These checks establish memory/accounting behavior, not measured latency or CUDA
hardware performance.


### Native grouped projection uses the same mixed workspace

CPU GroupedMatMul now plans one StepScoped lane for Y staging, necessary X/W/bias
snapshots, and usize routes. The mixed helper is ValueIndexWorkspace<T, N>,
generalizing the existing f32 layout to f64 without another owner, pool or quota
protocol. Affine grouped, Packed grouped and affine MoE simply specify f32 on
this same helper. The Native multiply/add loop, transposed weight indexing and
output conversion remain unchanged. FLOAT16/BFLOAT16 with bias still round the
MatMul sum to storage precision before adding widened bias; DOUBLE stays f64.

Aligned contiguous FLOAT X/W/bias retain the existing zero-copy borrow. Metadata
therefore reports Deferred for nonempty FLOAT; execution checks actual views and
charges only snapshots actually needed. DOUBLE keeps its previous full snapshot
path. Half/bfloat conversion uses the existing bulk widening helpers once per
source, preserving architecture-selected conversion paths rather than widening
weights repeatedly per route. All supported source strides are admitted directly;
there is no upstream gather for these four operands.

The complete mixed lane binds before any value read. Host/span checks precede
binding; routes are copied and validated before float sources are consumed. Y is
published only after every source read and all arithmetic have finished, so all
four source/Y overlaps remain safe. Empty Y still checks all routes while leaving
X/W/bias untouched, with only the route region charged. Empty reductions resolve
zero-element sources before numel/conversion helpers can overflow an intermediate
product on shapes such as [large_group_count, N, 0]. Host validation remains in
place for nonempty Y.

Tests compare the original snapshot arithmetic for four floating types, both
transposes, optional/absent bias slots and dense/padded/reversed/unaligned inputs.
They also cover half rounding before bias, DOUBLE precision, shifted overlaps
with strided Y/padding, missing/short/misaligned scratch, invalid routes, stale
metadata, non-Host and unaddressable sources, zero output and zero reduction.
Exact and one-byte-short Session budgets exercise Deferred FLOAT versus Ready
other types, changing logical shapes, physical strides, trim and readmission.
Allocator positive controls cover the former payload sizes; a retained-owner
witness exercises 4096 calls with no additional issuer callbacks, including
zero-byte controls. These establish accounting/correctness, not measured latency.
Typed snapshot storage is initialized before copying; the former DOUBLE Vec
collector wrote each element once, so DOUBLE snapshot traffic needs measurement
before making any speed claim.
