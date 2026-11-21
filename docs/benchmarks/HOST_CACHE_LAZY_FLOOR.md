# Host-cache lazy floor calibration

`LargeAllocCache` previously measured the allocator floor in its constructor.
That probe touched memory even for zero retention budget and small-only users.
Construction now leaves an automatic floor pending; a request that can require
a retention decision resolves the existing process-wide calibrated value.
Explicit floors start resolved. `floor_bytes()` inspects an `Option<usize>`;
`resolve_floor_bytes()` requests the probe explicitly.

## Scope and method

Apple M1 Max, 32 GiB RAM, release profile, macOS `/usr/bin/time -lp`. Compare
baseline `267408c9df76c6859147fc9f38b07bc31c17bd44` with the lazy implementation.
The governor source at that baseline is also unchanged in the parent CPU-expert
PR. Copy the exact same `host_cache_startup.rs` example source to each tree and
build in separate target directories. Three repetitions alternate before/after
order for each of four modes, for 24 fresh processes. There is no manual floor
override: `new` uses automatic calibration; the explicit control uses
`with_floor`. All runs assert cache hit and retained-byte counts.

[Raw records](HOST_CACHE_LAZY_FLOOR.results.json) preserve process stdout, complete
time output, probe-source hash and baseline revision. CPU time has coarse
resolution here, so these data support a memory observation, not a throughput
or latency claim. No model, CUDA or MLAS execution is involved.

## Results

Median process peak RSS in MiB, including startup and all example work:

| Mode | Work | Baseline | Lazy floor |
|---|---|---:|---:|
| construct | Construct one automatic cache | 511.953 | 1.797 |
| disabled | Zero budget; 32 allocate/release cycles of 512 KiB | 511.953 | 1.875 |
| small | Automatic floor; 32 cycles of 4 KiB | 511.969 | 1.844 |
| explicit | Explicit 4 KiB floor; 32 cycles of 4 KiB | 1.828 | 1.844 |

The explicit control retains 4096 bytes and records 31 hits on both revisions.
The other modes retain zero bytes and have zero hits. The old automatic
constructor's peak is dominated by the allocator probe. The new cache skips it
when the floor cannot affect the request; a pending floor is not replaced with
a guessed threshold. Large candidate requests still resolve and use the actual
calibrated result. An eligible workload can therefore still touch the probe's
large ladder; this change is not a universal 512 MiB runtime memory reduction.

## Reproduction

In each source tree, with the same example source:

```sh
cargo build --release -p onnx-runtime-memory-governor --example host_cache_startup
/usr/bin/time -lp target/release/examples/host_cache_startup construct
/usr/bin/time -lp target/release/examples/host_cache_startup disabled
/usr/bin/time -lp target/release/examples/host_cache_startup small
/usr/bin/time -lp target/release/examples/host_cache_startup explicit
```

Alternate tree order over three repetitions. On macOS, time reports maximum
resident set size in bytes; these figures divide it by 1048576. Platform
allocator behavior differs, so do not extrapolate this probe peak to other hosts.
A sandbox that blocks resource sysctls cannot supply these statistics; the
measurement above ran with local resource-stat access.

## Validation and invariants

All 71 governor unit tests pass, including four new cases covering pending
automatic floors, disabled retention, explicit resolution and low configured
floors. Existing exact-layout reuse, concurrent ownership, retained-byte cap,
rejection statistics and drain tests remain intact. The existing calibration
time-envelope test is retained with a first-use name. Strict governor all-target
Clippy passes. No new environment variable, compatibility export or scheduler
was added.
