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
`ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES`. Otherwise it decodes one weight row per
explicit Rayon shard and applies that row to every token routed to the expert.
Routing, activation, gates and accumulation share the existing implementation;
format metadata and the canonical decoder remain the only format authority.
Zero budget selects streaming without hashing constant identities or building
dense cache entries. Contiguous packed weights are borrowed; strided inputs copy
only the selected expert's compressed bytes. Mutable inputs are read each call.

For a projection with input width K and N output features, at most
`min(Rayon workers, N) * (K + 256)` f32 values are live as decoded-weight scratch.
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


### Owned compact wire-layout NBits preparation

Compact wire-layout NBits operands now use the same `DerivedWeightMemory`
source as dense expansions and QMoE experts when conversion, strided gathering,
or explicit first-touch placement requires ownership. Before preparing any whole
payload, sizing checks Host accessibility, supported operand dtypes and checked
byte arithmetic. Admission covers packed bytes, f32 scales and optional packed
zero points, plus a portable bound on half-scale widening or source/destination
placement overlap. The extra allowance returns after preparation; actual payload
ownership retains its exact grant until the transient operand or constant kernel
cache is destroyed. Payload fields drop before their grant. Capacity refusals
propagate instead of allocating an uncharged copy.

Unchanged contiguous Host Uint8/Float32 wire operands remain borrowed, including
with zero Host headroom. Warm prepared constants reuse their allocation without
another grant. A cold-fill gate prevents competing threads from building copies
under a one-payload budget; callbacks never acquire that gate. Existing numeric,
offset, strided, dtype-conversion and placement dispatch guards remain.

This covers the shared wire-layout operand used by native 2-bit and nibble
execution. Other ISA-specific transformed int8/int4/u8, N16/KAI and optional MLAS
packs still need owned-grant integration, and retained compact OnceLock entries
are not yet idle-reclaimable. Scratch/activation coverage, physical Host/Device
book reconciliation and process-wide limits remain separate work. No speedup,
NUMA placement performance or CUDA execution is claimed.
