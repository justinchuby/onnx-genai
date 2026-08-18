//! Vectorised transcendental primitives for the activation family (`Tanh`,
//! `Sigmoid`, `Erf`, `Gelu` in both its exact and tanh forms, `FastGelu`,
//! `QuickGelu`, `BiasGelu`).
//!
//! # Why this exists
//!
//! The scalar kernels evaluate one `libm` transcendental per element. On this
//! class of hardware a dependent `tanhf` is ~13 ns and `f64::tanh` ~25 ns per
//! element, which is roughly two orders of magnitude above what ONNX Runtime
//! achieves for the same op. ORT's advantage is not algorithmic: MLAS ships
//! hand-written FMA3 kernels (`lib/x86_64/{Tanh,Logistic,Erf}KernelFma3.S`)
//! that evaluate a *polynomial* over a clamped range, eight lanes at a time,
//! with no branches and no libm call.
//!
//! This module reproduces those approximations in safe-ish Rust over
//! `core::arch::x86_64` intrinsics, so the default build (which does **not**
//! enable the `mlas` feature) gets the same throughput.
//!
//! # Numerical contract
//!
//! * The polynomials and their evaluation order are taken verbatim from
//!   `MlasTanhConstants` / `MlasLogisticConstants` (which MLAS in turn took
//!   from Eigen) and `MlasErfConstants`, so a build using this path tracks
//!   ORT's own CPU output.
//! * `erf` is the one member of the family whose scalar fallback is *more*
//!   accurate than its vector path: `libm::erf` is correctly rounded, MLAS's
//!   polynomial is only faithfully rounded (measured worst error 5.96e-8 =
//!   1 ulp below 1.0, over a 400 003-point sweep of `[-6, 6]` plus both branch
//!   boundaries). That is the same trade ORT itself makes, and 1 ulp is two
//!   orders of magnitude inside the conformance suite's `rtol=1e-4`.
//! * MLAS clamps its input to the polynomial's valid band and returns the
//!   polynomial value at the clamp point; this module *saturates* instead,
//!   substituting the exact limit outside the band. On this host the two
//!   agree bit-for-bit everywhere they were probed, `±Inf` included: the
//!   vendored `MlasComputeLogistic` returns exactly `0.0` for `sigmoid(-Inf)`
//!   and for every `x <= -18`. That is not an underflow: at the clamp point
//!   the rational evaluates to `-5.96e-8`, and `logistic.cpp`'s own *output*
//!   clamp (`std::clamp((p / q) + 0.5f, 0.0f, 1.0f)`) pins that negative
//!   value to `0.0`. Saturation is therefore an *equivalent* formulation that
//!   makes the endpoints exact by construction rather than by that accident —
//!   it is not a correctness win, and earlier revisions of this comment
//!   wrongly claimed clamping leaked `1.5e-8` at `-Inf`. It is still the
//!   formulation worth keeping: it is exact for any future constant set, and
//!   it costs nothing measurable. Note that saturation is *not* correctly
//!   rounded at the very edge of the `tanh` band: `sigmoid(18) = 1 - 1.523e-8`
//!   does round to `1.0f32`, but `tanh(9) = 1 - 3.046e-8` rounds to
//!   `0.99999994` (`0x3F7FFFFF`) because `3.046e-8` exceeds the `2.98e-8`
//!   half-ulp threshold below `1.0`. `tanh` only rounds to `1.0f32` from
//!   `|x| >= 9.010914` (the first f32 whose `tanh` does), so the substituted
//!   `±1` is one ulp high on `9 < |x| < 9.010914`. MLAS and ORT return `1.0`
//!   there too, the scaled error is `6.62e-9` against the module's asserted
//!   `4e-7` bound, and the alternative in that range is the rational's own
//!   out-of-range overshoot, so this is accepted rather than special-cased.
//! * `NaN` propagates unchanged. Both the clamp (`maxps`/`minps` with the
//!   value as the *second* operand, matching MLAS) and the saturation selects
//!   (ordered compares, which are false for `NaN`) preserve it. Measured
//!   against ORT 1.28.0 CPU, the payload and sign survive exactly
//!   (`0x7FC01234 -> 0x7FC01234`) and a signalling `NaN` is quieted with its
//!   payload kept (`0x7F800001 -> 0x7FC00001`), on both the vector and the
//!   scalar path.
//! * Signed zero is preserved: the numerator is odd, so `p * (-0.0) = -0.0`
//!   and `-0.0 / q = -0.0`.
//! * **Two deliberate divergences from ORT**, both verified by running ORT
//!   1.28.0 on identical inputs. Over 140 probed `(function, special value)`
//!   pairs these are the only two disagreements; in 32 of the remaining pairs
//!   this vector path matches ORT where the exact-libm scalar fallback does
//!   not.
//!     1. `tanh` is pinned to `[-1, 1]` (see the note at the pin site); ORT
//!        returns `1.0000001` at `x = 8.442762`.
//!     2. `tanh_gelu(-Inf)` and `quick_gelu(-Inf)` return `+0.0`, the
//!        mathematical limit; ORT evaluates `-Inf * 0` and returns `NaN`.
//!        ONNX does not specify either. The limit is pinned by
//!        `gelu_special_values`; the scalar fallback and the f64 references
//!        agree only because they carry the same explicit `-Inf` guard, so
//!        that agreement documents intent rather than corroborating it.
//!
//! # ISA dependence
//!
//! Dispatch is by runtime feature detection and only ever *adds* a path:
//! without AVX2+FMA the caller's exact scalar closure is used unchanged, so no
//! existing target regresses. As with ORT, this means results can differ by
//! ~1e-7 relative between an AVX2 host and a non-AVX2 host. Tests therefore
//! assert an error bound against an `f64` reference rather than bit equality.

#![allow(clippy::excessive_precision)]

use crate::dtype::{output_direct_write_eligible, slice_byte_range, write_dense_f32_narrow};
use onnx_runtime_ep_api::{Result, TensorMut};

// ---------------------------------------------------------------------------
// Direct-write plumbing
// ---------------------------------------------------------------------------

/// Apply `f` to `x` and land the result in `out`, writing straight into the
/// output tensor's storage whenever that is sound.
///
/// The obvious spelling — `let mut y = vec![0.0; n]; f(&x, &mut y);
/// write_dense_f32_narrow(op, out, &y)` — makes three passes over the data
/// (zero the scratch, compute into it, copy it out) plus an allocation. At
/// prefill sizes the activation itself is a handful of cycles per element, so
/// those extra passes, not the arithmetic, set the runtime: they cost more than
/// the kernel.
///
/// So when [`output_direct_write_eligible`] says the output is a contiguous,
/// host-visible, correctly-sized `f32` buffer that does *not* alias the input we
/// still have to read, `f` writes into it in place and the scratch disappears.
/// Any other case — f16/bf16/f64 output, a strided view, a device pointer, or
/// an in-place `y = act(y)` node where the ranges do overlap — falls back to the
/// owned buffer, which is exactly the situation `write_dense_f32_narrow` exists
/// to handle. Correctness never depends on which arm runs.
pub(crate) fn write_mapped<F>(op: &str, out: &mut TensorMut, x: &[f32], f: F) -> Result<()>
where
    F: FnOnce(&[f32], &mut [f32]),
{
    write_mapped_reading(op, out, x, &[], f)
}

/// [`write_mapped`] for closures that read a slice *besides* `x`.
///
/// The disjointness check has to cover every buffer the closure still reads
/// once we start writing, not just the primary input. FastGelu's bias is the
/// motivating case: it is a `Cow::Borrowed` view of the bias tensor whenever
/// that tensor is contiguous `f32`, so it is live borrowed storage, and the
/// fused kernel re-reads it for every row. Were it ever handed to us aliasing
/// the output, writing row 0 would corrupt the bias that every later row still
/// depends on — and we would be holding `&mut` and `&` over the same bytes.
/// Callers declare those extra ranges here and the direct-write arm is skipped
/// when any of them overlaps the output.
pub(crate) fn write_mapped_reading<F>(
    op: &str,
    out: &mut TensorMut,
    x: &[f32],
    also_read: &[core::ops::Range<usize>],
    f: F,
) -> Result<()>
where
    F: FnOnce(&[f32], &mut [f32]),
{
    let n = x.len();
    let eligible = if also_read.is_empty() {
        output_direct_write_eligible(out, n, &[slice_byte_range(x)])
    } else {
        let mut reads = Vec::with_capacity(1 + also_read.len());
        reads.push(slice_byte_range(x));
        reads.extend_from_slice(also_read);
        output_direct_write_eligible(out, n, &reads)
    };
    if eligible {
        out.validate()?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: `output_direct_write_eligible` confirmed a validated,
        // contiguous, host-accessible Float32 tensor holding exactly `n`
        // elements, and that its bytes are disjoint from every slice the
        // closure still reads (`x` plus `also_read`).
        let dst = unsafe { std::slice::from_raw_parts_mut(out.data_ptr_mut::<f32>(), n) };
        f(x, dst);
        return Ok(());
    }
    let mut y = vec![0.0f32; n];
    // Serial on purpose: see `serial_scope`. This arm always ends in a full
    // serial pass over `y`, so splitting the kernel only costs locality.
    serial_scope(|| f(x, &mut y));
    write_dense_f32_narrow(op, out, &y)
}

/// Smallest slice length for which the vector path is worth its dispatch
/// overhead. Below this the scalar loop wins (measured: crossover sits between
/// 8 and 32 elements; 32 is the conservative side of it).
///
/// Only the `x86_64` dispatch arms consult this, but the tests below use it as
/// a length unit on every architecture, so it stays compiled rather than
/// `cfg`-gated.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const SIMD_MIN_LEN: usize = 32;

/// Smallest slice length worth splitting across threads.
///
/// These kernels are memory-bound at length, so the payoff is real bandwidth,
/// not arithmetic: one core cannot saturate this socket's memory controllers.
/// Below the threshold the fork/join costs more than the work it hands out, so
/// the split has to earn its place.
///
/// **This threshold is set by what the split costs inside a real session, not
/// by what it costs in a benchmark loop.** A rayon worker parks when it runs
/// out of work, and a session hands the pool one short burst per node with
/// non-trivial gaps in between, so nearly every call pays to wake the pool
/// again. A tight benchmark loop never sees that, because it keeps the workers
/// spinning. Measured through a real ORT session, our EP against ORT's CPU EP
/// on the same graph and the same input:
///
/// | elements | `PAR_MIN_LEN` 16 Ki | `PAR_MIN_LEN` 1 Mi |
/// |---|---|---|
/// | 16384 | 0.19x | 1.20x |
/// | 32768 | 0.17x | 1.43x |
/// | 65536 | 0.16x | 1.60x |
/// | 131072 | 0.21x | 1.68x |
/// | 262144 | 0.98x | 1.80x |
///
/// (`Sqrt`, f32, ORT 1.28.0. Both sides at `intra_op_num_threads = 1`; ours
/// additionally at `RAYON_NUM_THREADS = 32`, so the comparison is our
/// *parallel* kernel against single-threaded ORT — which is what makes the
/// wake-up visible. p50 of 200 interleaved runs, our EP's assignment asserted
/// from ORT's profiler.) The wake-up costs ~50 us and the work below a megabyte is worth less
/// than that, so the old 16 Ki threshold turned a 1.2-1.8x win into a 5x loss
/// on exactly the sizes a decode step uses.
///
/// One threshold for every kernel is a simplification, and a measured one.
/// A later 16-thread sweep found the expensive kernels want it lower: dropping
/// `PAR_MIN_LEN` to 256 Ki made `Gelu` 1.26-2.35x faster over 256 Ki - 2 Mi,
/// `Erf` 1.23-1.78x and `FastGelu` 1.28-2.03x, while making `Sqrt` at 256 Ki
/// *2.3x slower*. Break-even scales with per-element cost, and `Sqrt` — the
/// cheapest, most bandwidth-bound kernel here — is the one this constant was
/// derived from, so it is conservative for the transcendentals. Splitting it
/// into a per-kernel cost class needs the class plumbed through four generic
/// entry points and is left to a follow-up; this value is the safe one,
/// because being too high costs throughput while being too low cost 5x.
///
/// Kept well above [`SIMD_MIN_LEN`] so that every chunk is still long enough to
/// take the vector path. That matters for more than speed: the scalar and
/// vector paths are not bit-identical for the approximated kernels, so a chunk
/// that fell back to scalar would make the result depend on the thread count.
pub(crate) const PAR_MIN_LEN: usize = 1_048_576;

/// Minimum elements per thread.
///
/// A chunk shorter than this spends more time waking the worker that runs it
/// than the worker saves. At the measured ~50 us wake-up and ~0.3 ns/element,
/// break-even is around 160 Ki elements per chunk; 256 Ki keeps a margin and
/// still splits a 4 Mi prefill sixteen ways.
const PAR_MIN_CHUNK: usize = 262_144;

thread_local! {
    /// Set while a caller has to keep the f32 kernel on one thread.
    ///
    /// The f16/bf16 activations are a sandwich: widen the input into an f32
    /// scratch, run the kernel, narrow the scratch back. Only the middle layer is
    /// parallelisable here, and splitting just that layer measured *slower* than
    /// leaving it alone — on a 2M-element prefill, Sqrt/f16 dropped to 0.59x and
    /// Tanh/f16 to 0.79x against the same binary at one thread. Spreading the
    /// scratch across sixteen private caches only to have a serial narrow pull all
    /// 8 MB of it back costs more locality than the arithmetic saves, and it makes
    /// the result wildly variable (bf16 prefill ranged 1.6-3.4 ns/element across
    /// repeats where f32 held to +/-5%).
    ///
    /// So the narrow-output arm of `write_mapped_reading` runs the kernel under
    /// this guard and takes the serial path, which is exactly what it did before.
    /// The fix for f16/bf16 is to fuse widen, compute and narrow into a single
    /// pass per chunk so each thread keeps its slice in L2 — a different change,
    /// and one that has to be measured on its own.
    static FORCE_SERIAL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Runs `f` with the elementwise kernels' parallel split disabled.
///
/// Restores the previous value on unwind as well as on return. A leaked flag
/// would only cost speed, never correctness -- serial and parallel results are
/// bit-identical -- but it would leave that thread quietly serialised for every
/// later activation, which is a hard bug to find afterwards.
pub(crate) fn serial_scope<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_SERIAL.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FORCE_SERIAL.with(|c| c.replace(true)));
    f()
}

#[inline]
fn force_serial() -> bool {
    FORCE_SERIAL.with(core::cell::Cell::get)
}

/// Chunk length for [`run_chunked`], or `None` to run the slice in one call.
///
/// Split out from the execution so the boundaries can be swept over thread
/// counts this machine does not have. Rounding to eight lanes keeps every
/// chunk on an AVX2 vector boundary, and the [`PAR_MIN_CHUNK`] floor keeps
/// each one long enough to stay on the vector path — a chunk that fell back to
/// scalar would round differently from its neighbours.
#[inline]
fn par_chunk_len(len: usize, threads: usize) -> Option<usize> {
    if threads < 2 {
        return None;
    }
    let chunk = len.div_ceil(threads).max(PAR_MIN_CHUNK).next_multiple_of(8);
    (chunk < len).then_some(chunk)
}

/// Chunks to cut `len` into when the *host* runtime owns the pool.
///
/// The rayon split asks the pool how wide it is and cuts exactly that many
/// pieces, because a rayon `par_chunks` costs a wake-up per piece and handing
/// it more pieces than threads is wasted scheduling. The host pool is the
/// opposite: it is already spinning, we cannot ask ORT how many intra-op
/// threads it was given, and `TrySimpleParallelFor` claims indices
/// dynamically. So we cut by *size* rather than by thread count and let the
/// host decide how many of its threads to point at the result.
///
/// Cutting by size rather than by pool width is also a **correctness**
/// property, not only a scheduling one: the chunk boundaries — and therefore
/// the exact bit pattern of the output — depend only on `len`, never on how
/// many intra-op threads the session happens to have. The same tensor produces
/// the same bits whether it is run at `intra_op = 1` or `16`, so a result can
/// never move when the session is reconfigured. Had we cut by pool width (as
/// the rayon path does), reducing `intra_op` would re-chunk the slice and
/// perturb the last-lane rounding of the split boundaries.
///
/// [`HOST_MIN_CHUNK`] is the floor, so a chunk is never too short to be worth
/// dispatching or to stay on the vector path; this cap is the other end, and
/// keeps a very long slice from turning into thousands of tiny tasks. Raising
/// it to 256 or 1024 measured within noise of 64 at every size swept, so 64 —
/// four tasks per thread on a sixteen-thread session, enough for the host to
/// balance with — is the one that dispatches least.
const MAX_HOST_CHUNKS: usize = 64;

/// Length below which the host split is not worth dispatching.
///
/// Sixteen times lower than [`PAR_MIN_LEN`], and for a concrete reason: that
/// constant pays for waking rayon's workers, and ORT's intra-op workers are
/// already awake — they spin. Measured at `intra_op = 16` against ORT's own
/// CPU EP, splitting a 256 Ki slice on the host pool instead of leaving it
/// whole moved `Gelu` from 0.15x to 0.94x, `FastGelu` 0.11x to 0.64x, `Erf`
/// 0.24x to 1.33x and `Sqrt` 0.19x to 0.91x. At 64 Ki the split still helps
/// (`Tanh` 0.40x to 1.36x, `Relu` 0.61x to 1.12x); at 16 Ki it stopped
/// mattering, every gate measuring within noise of every other, so this is the
/// last length where the evidence is one-sided.
const HOST_MIN_LEN: usize = 65_536;

/// Shortest chunk worth handing to one of the host's threads.
///
/// [`PAR_MIN_CHUNK`] is 256 Ki because a rayon wake-up costs ~50 us. A host
/// task costs a fraction of that, and the sweep is monotone in this direction:
/// at `intra_op = 16` over 64 Ki - 1 Mi, dropping the floor from 256 Ki to
/// 64 Ki to 16 Ki to 4 Ki improved almost every op at almost every size
/// (`Erf` at 256 Ki: 0.18x, 0.33x, 0.79x, 1.02x against ORT; `Gelu`: 0.16x,
/// 0.31x, 0.90x, 0.98x). 4 Ki is where it flattened.
const HOST_MIN_CHUNK: usize = 4_096;

/// Chunk length and count for the host-pool split, or `None` to stay whole.
#[inline]
fn host_chunk_len(len: usize) -> Option<(usize, usize)> {
    if len < HOST_MIN_LEN {
        return None;
    }
    let chunk = len
        .div_ceil(MAX_HOST_CHUNKS)
        .max(HOST_MIN_CHUNK)
        .next_multiple_of(8);
    (chunk < len).then(|| (chunk, len.div_ceil(chunk)))
}

/// [`host_chunk_len`] for the bias-fused kernels: whole multiples of `width`.
#[inline]
fn host_chunk_len_rows(len: usize, width: usize) -> Option<(usize, usize)> {
    if width == 0 || len < HOST_MIN_LEN {
        return None;
    }
    let rows = len
        .div_ceil(width)
        .div_ceil(MAX_HOST_CHUNKS)
        .max(HOST_MIN_CHUNK.div_ceil(width));
    let chunk = rows.checked_mul(width)?;
    (chunk < len).then(|| (chunk, len.div_ceil(chunk)))
}

/// `input`/`output` as raw pointers, so the host's threads can share them.
///
/// Rayon proves disjointness with `par_chunks_mut`; the host pool has no such
/// API, so the split is done by index arithmetic and the disjointness argument
/// moves into [`run_on_host`].
struct HostChunks {
    input: *const f32,
    output: *mut f32,
    len: usize,
    chunk: usize,
}

// SAFETY: the only thing shared across threads is a pair of pointers plus the
// two lengths needed to derive a chunk from an index. `run_on_host` gives each
// index a disjoint half-open range of both slices, and `HostParallel::run`
// promises each index runs exactly once, so no two threads ever hold
// overlapping references to the output.
unsafe impl Sync for HostChunks {}

impl HostChunks {
    /// Runs `body` on the `index`-th chunk of the input and output slices.
    ///
    /// Takes the body rather than returning the two slices so that the
    /// caller's closure captures the whole `&HostChunks` — which is `Sync` —
    /// instead of the raw pointer fields individually, which are not.
    ///
    /// # Safety
    ///
    /// `index` must be below the chunk count this was built for, and no two
    /// live calls may share one: the `&mut` handed to `body` is only unique
    /// because distinct indices give disjoint ranges.
    unsafe fn run_chunk<F>(&self, index: usize, body: &F)
    where
        F: Fn(&[f32], &mut [f32]),
    {
        let start = index * self.chunk;
        // The last chunk is short whenever `chunk` does not divide `len`.
        let this = self.chunk.min(self.len - start);
        // SAFETY: `start + this <= len` by construction, and the caller
        // guarantees no other thread holds this range.
        let (input, output) = unsafe {
            (
                core::slice::from_raw_parts(self.input.add(start), this),
                core::slice::from_raw_parts_mut(self.output.add(start), this),
            )
        };
        body(input, output);
    }
}

/// Runs `body` on the host runtime's pool if this session has one worth using.
///
/// Returns whether `body` was run at all; a `false` return leaves `output`
/// untouched and the caller free to pick another split.
///
/// `#[inline(never)]` on purpose. Everything here is cold -- the decision is a
/// relaxed load, and the split it guards only happens inside an ORT session
/// whose pool has been proven parallel -- while `run_chunked`'s callers are the
/// hottest elementwise kernels in the crate. Inlining it grew `run_chunked`
/// enough to repartition codegen units: at `intra_op = 1` it cost `Relu` 34% at
/// 1 Mi (236 -> 315 us) with no path change at all, the same failure mode as
/// the instantiation note on `clip_chunked`.
#[inline(never)]
fn try_host<F>(input: &[f32], output: &mut [f32], body: &F) -> bool
where
    F: Fn(&[f32], &mut [f32]) + Sync + Send,
{
    let Some(host) = onnx_runtime_ep_api::host_parallel::current() else {
        return false;
    };
    if !host.prefer_host() {
        return false;
    }
    match host_chunk_len(input.len()) {
        Some((chunk, count)) => {
            note_parallel_dispatch();
            run_on_host(host, input, output, chunk, count, body);
        }
        // Too short to split across the host's threads. Our own pool is not
        // the answer either: it would be a second pool on the same cores.
        None => body(input, output),
    }
    true
}

/// [`try_host`] for the row-shaped split. Outlined for the same reason.
#[inline(never)]
fn try_host_rows<F>(input: &[f32], output: &mut [f32], width: usize, body: &F) -> bool
where
    F: Fn(&[f32], &mut [f32]) + Sync + Send,
{
    let Some(host) = onnx_runtime_ep_api::host_parallel::current() else {
        return false;
    };
    if !host.prefer_host() {
        return false;
    }
    match host_chunk_len_rows(input.len(), width) {
        Some((chunk, count)) => run_on_host(host, input, output, chunk, count, body),
        None => body(input, output),
    }
    true
}

/// Splits `input`/`output` into `count` chunks of `chunk` and runs `body` on
/// each one on the host runtime's pool.
///
/// Bit-identical to calling `body` on the whole slice, for the same reason the
/// rayon path is: the kernels are elementwise, every chunk is a multiple of
/// eight lanes, and none is shorter than [`SIMD_MIN_LEN`], so no chunk takes a
/// different code path from its neighbours.
fn run_on_host<F>(
    host: onnx_runtime_ep_api::HostParallel,
    input: &[f32],
    output: &mut [f32],
    chunk: usize,
    count: usize,
    body: &F,
) where
    F: Fn(&[f32], &mut [f32]) + Sync,
{
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(chunk > 0 && count > 0);
    let shared = HostChunks {
        input: input.as_ptr(),
        output: output.as_mut_ptr(),
        len: input.len(),
        chunk,
    };
    let shared = &shared;
    host.run(count, &|index| {
        // SAFETY: `HostParallel::run` invokes every index in `0..count`
        // exactly once, so this call owns its range for its whole duration.
        unsafe { shared.run_chunk(index, body) };
    });
}

/// Chunk length for [`run_chunked_rows`], always a whole multiple of `width`.
#[inline]
fn par_chunk_len_rows(len: usize, width: usize, threads: usize) -> Option<usize> {
    if threads < 2 || width == 0 {
        return None;
    }
    let rows = len
        .div_ceil(width)
        .div_ceil(threads)
        .max(PAR_MIN_CHUNK.div_ceil(width));
    let chunk = rows.checked_mul(width)?;
    (chunk < len).then_some(chunk)
}

/// Split `input`/`output` across the rayon pool and run `body` on each chunk.
///
/// Elementwise ops are exactly chunk-independent, so this is bit-identical to
/// running `body` over the whole slice — provided every chunk takes the same
/// code path, which is why chunks are never shorter than [`PAR_MIN_CHUNK`] and
/// are rounded to a multiple of eight lanes.
///
/// Falls back to a single call when the pool has one thread, when the slice is
/// short, or when already inside a rayon worker: nesting a `par_chunks` inside
/// an outer parallel region only adds scheduling overhead, since the outer
/// region is already keeping the pool busy.
///
/// `pub(crate)` because it is not specific to this file: any elementwise f32
/// kernel wants it, and the MLAS-backed `SiLU`/`Relu`/`Clip` paths in sibling
/// modules were single-threaded for want of it.
#[inline]
fn run_chunked<F>(input: &[f32], output: &mut [f32], body: F)
where
    F: Fn(&[f32], &mut [f32]) + Sync + Send,
{
    let len = input.len();
    // Test the length *before* touching rayon. `current_num_threads` reaches
    // the global registry, and initialising it spawns the pool; paying that on
    // a 4096-element decode call cost ~1.4 us, which is more than the whole
    // call. Measured as a uniform 0.6-0.8x regression on every short case
    // until this check moved above it.
    if len < HOST_MIN_LEN || force_serial() {
        body(input, output);
        return;
    }
    // Inside a host task we are already occupying one of the host's threads;
    // splitting again would nest a pool inside the pool we were handed.
    if onnx_runtime_ep_api::host_parallel::in_host_task() {
        body(input, output);
        return;
    }
    // Prefer the host's pool over ours whenever the host has one. Ours would
    // be a *second* pool on the same cores: measured 1.5-3.1x slower than
    // staying serial at 1 Mi under an `intra_op = 16` session.
    //
    // A host pool of *one* thread is the opposite case. It is not using the
    // machine, so there is nothing to contend with and our own pool is the
    // right tool: at `intra_op = 1`, rayon beat borrowing ORT's single thread
    // by 2-9x over 1-4 Mi. `prefer_host` decides which of the two a session is
    // by watching whether the host's pool ever actually runs our chunks.
    if try_host(input, &mut *output, &body) {
        return;
    }
    if len < PAR_MIN_LEN {
        body(input, output);
        return;
    }
    // Nesting a `par_chunks` inside an outer parallel region only adds
    // scheduling overhead: the outer region is already keeping the pool busy.
    if rayon::current_thread_index().is_some() {
        body(input, output);
        return;
    }
    let Some(chunk) = par_chunk_len(len, rayon::current_num_threads()) else {
        body(input, output);
        return;
    };

    note_parallel_dispatch();

    use rayon::prelude::*;
    output
        .par_chunks_mut(chunk)
        .zip(input.par_chunks(chunk))
        .for_each(|(o, i)| body(i, o));
}

#[cfg(test)]
thread_local! {
    /// Times *this* thread has handed work to the pool from [`run_chunked`].
    ///
    /// Thread-local rather than a global atomic so that concurrently running
    /// tests cannot bump each other's count: the increment happens on the
    /// calling thread, before the slice is split.
    static PARALLEL_DISPATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline(always)]
fn note_parallel_dispatch() {
    #[cfg(test)]
    PARALLEL_DISPATCHES.with(|c| c.set(c.get() + 1));
}

/// Chunk a whole-slice kernel that needs no captured state.
///
/// Takes a `fn` pointer rather than `impl Fn` on purpose: every caller shares
/// one instantiation, and that instantiation lives in this module. See
/// `clip_chunked` for why that matters.
#[cfg(feature = "mlas")]
pub(crate) fn run_chunked_fn(input: &[f32], output: &mut [f32], body: fn(&[f32], &mut [f32])) {
    run_chunked(input, output, body);
}

/// Chunked `Clip`, instantiated here rather than at the call site.
///
/// `run_chunked` is generic, and instantiating it from `selection.rs` moved
/// enough code between codegen units that the AVX2 unary kernels in this
/// module stopped being vectorised: `Sqrt` went 21 -> 48 us and `Tanh`
/// 32 -> 55 us at n = 65536, with no runtime path change at all. Keeping every
/// instantiation inside this module keeps the partitioning stable.
#[cfg(feature = "mlas")]
pub(crate) fn clip_chunked(input: &[f32], output: &mut [f32], minimum: f32, maximum: f32) {
    // Take the serial decision here, so the common short case is a direct call
    // instead of one through a closure the optimiser can no longer see into.
    if input.len() < HOST_MIN_LEN
        || force_serial()
        || onnx_runtime_ep_api::host_parallel::in_host_task()
        || rayon::current_thread_index().is_some()
    {
        // Short, or already on a worker: one direct call, with no closure for
        // the optimiser to lose sight of.
        mlas_sys::compute_clip(input, output, minimum, maximum);
        return;
    }
    run_chunked(input, output, |i, o| {
        mlas_sys::compute_clip(i, o, minimum, maximum);
    });
}

/// How many times this thread has reached [`run_chunked`]'s parallel branch.
#[cfg(test)]
pub(crate) fn parallel_dispatches() -> usize {
    PARALLEL_DISPATCHES.with(std::cell::Cell::get)
}

/// [`run_chunked`] for the bias-fused kernels, whose element mapping is
/// `bias[i % width]`.
///
/// Chunks are whole multiples of `width`, so every chunk starts on a row
/// boundary and its elements keep the same bias offsets they had in the
/// un-split slice. A chunk cut mid-row would silently rotate the bias.
#[inline]
fn run_chunked_rows<F>(input: &[f32], output: &mut [f32], width: usize, body: F)
where
    F: Fn(&[f32], &mut [f32]) + Sync + Send,
{
    let len = input.len();
    if len < HOST_MIN_LEN || width == 0 || force_serial() {
        body(input, output);
        return;
    }
    if onnx_runtime_ep_api::host_parallel::in_host_task() {
        body(input, output);
        return;
    }
    if try_host_rows(input, &mut *output, width, &body) {
        return;
    }
    if len < PAR_MIN_LEN {
        body(input, output);
        return;
    }
    if rayon::current_thread_index().is_some() {
        body(input, output);
        return;
    }
    let Some(chunk) = par_chunk_len_rows(len, width, rayon::current_num_threads()) else {
        body(input, output);
        return;
    };

    use rayon::prelude::*;
    output
        .par_chunks_mut(chunk)
        .zip(input.par_chunks(chunk))
        .for_each(|(o, i)| body(i, o));
}

// ---------------------------------------------------------------------------
// MLAS constants
// ---------------------------------------------------------------------------

/// `MlasTanhConstants`, `onnxruntime/core/mlas/lib/tanh.cpp`.
///
/// Consumed exclusively by the AVX2 module below, so it follows that module's
/// gating: on a non-`x86_64` target these would be unreferenced constants, and
/// CI builds with `-D warnings`.
#[cfg(target_arch = "x86_64")]
mod tanh_c {
    pub(super) const LOWER: f32 = -9.0;
    pub(super) const UPPER: f32 = 9.0;
    pub(super) const ALPHA_13: f32 = -2.76076847742355e-16;
    pub(super) const ALPHA_11: f32 = 2.00018790482477e-13;
    pub(super) const ALPHA_9: f32 = -8.60467152213735e-11;
    pub(super) const ALPHA_7: f32 = 5.12229709037114e-08;
    pub(super) const ALPHA_5: f32 = 1.48572235717979e-05;
    pub(super) const ALPHA_3: f32 = 6.37261928875436e-04;
    pub(super) const ALPHA_1: f32 = 4.89352455891786e-03;
    pub(super) const BETA_6: f32 = 1.19825839466702e-06;
    pub(super) const BETA_4: f32 = 1.18534705686654e-04;
    pub(super) const BETA_2: f32 = 2.26843463243900e-03;
    pub(super) const BETA_0: f32 = 4.89352518554385e-03;
}

/// `MlasLogisticConstants`, `onnxruntime/core/mlas/lib/logistic.cpp`.
///
/// `x86_64`-only for the same reason as [`tanh_c`].
#[cfg(target_arch = "x86_64")]
mod logistic_c {
    pub(super) const LOWER: f32 = -18.0;
    pub(super) const UPPER: f32 = 18.0;
    pub(super) const ALPHA_9: f32 = 4.37031012579801e-11;
    pub(super) const ALPHA_7: f32 = 1.15627324459942e-07;
    pub(super) const ALPHA_5: f32 = 6.08574864600143e-05;
    pub(super) const ALPHA_3: f32 = 8.51377133304701e-03;
    pub(super) const ALPHA_1: f32 = 2.48287947061529e-01;
    pub(super) const BETA_10: f32 = 6.10247389755681e-13;
    pub(super) const BETA_8: f32 = 5.76102136993427e-09;
    pub(super) const BETA_6: f32 = 6.29106785017040e-06;
    pub(super) const BETA_4: f32 = 1.70198817374094e-03;
    pub(super) const BETA_2: f32 = 1.16817656904453e-01;
    pub(super) const BETA_0: f32 = 9.93151921023180e-01;
}

/// `MlasErfConstants`, `onnxruntime/core/mlas/lib/erf.cpp`.
///
/// MLAS took the algorithm and coefficients from the "efficient faithfully
/// rounded implementation of erff" reference cited at the top of that file.
/// `x86_64`-only for the same reason as [`tanh_c`].
///
/// The two polynomials split at `|x| = 0.921875`:
///
/// * below it, `erf(x) ≈ x·(1 + P(x²))` with `SMALL_P5_MINUS_ONE` folded so
///   the final step is a single `fma(r, x, x)`;
/// * above it, `erf(x) = 1 - exp(-R(|x|))` with `R` a degree-7 polynomial in
///   `|x|` (again with the leading `1` folded into `BIG_P6_MINUS_ONE`), and
///   `exp` evaluated by the standard range-reduce/`2^k` scheme whose constants
///   follow.
#[cfg(target_arch = "x86_64")]
mod erf_c {
    /// `erf` is within half an ulp of `±1` past this, so MLAS clamps `|x|` here
    /// and lets the big-branch polynomial return exactly `1`.
    pub(super) const UPPER_ABS_RANGE: f32 = 3.925;
    pub(super) const SPLIT_BOUNDARY: f32 = 0.921875;

    pub(super) const SMALL_P0: f32 = -5.99104969e-4;
    pub(super) const SMALL_P1: f32 = 4.99339588e-3;
    pub(super) const SMALL_P2: f32 = -2.67667342e-2;
    pub(super) const SMALL_P3: f32 = 1.12818025e-1;
    pub(super) const SMALL_P4: f32 = -3.76124859e-1;
    pub(super) const SMALL_P5_MINUS_ONE: f32 = 1.28379151e-1;

    pub(super) const BIG_P0: f32 = 1.72948930e-5;
    pub(super) const BIG_P1: f32 = -3.83208680e-4;
    pub(super) const BIG_P2: f32 = 3.88393435e-3;
    pub(super) const BIG_P3: f32 = -2.42545605e-2;
    pub(super) const BIG_P4: f32 = 1.06777847e-1;
    pub(super) const BIG_P5: f32 = 6.34846687e-1;
    pub(super) const BIG_P6_MINUS_ONE: f32 = 1.28717512e-1;

    // Independent `exp` parameters, used only by the big branch.
    pub(super) const EXP_LOWER_RANGE: f32 = -88.376_262_664_794_9;
    /// MLAS spells this `1.44269504088896341f`, which rounds to exactly
    /// `f32::consts::LOG2_E`.
    pub(super) const EXP_LOG2_RECIPROCAL: f32 = std::f32::consts::LOG2_E;
    pub(super) const EXP_LOG2_HI: f32 = -6.93145752e-1;
    pub(super) const EXP_LOG2_LO: f32 = -1.42860677e-6;
    pub(super) const EXP_P0: f32 = 1.38319808e-3;
    pub(super) const EXP_P1: f32 = 8.37550033e-3;
    pub(super) const EXP_P2: f32 = 4.16689515e-2;
    pub(super) const EXP_P3: f32 = 1.66664466e-1;
    pub(super) const EXP_P4: f32 = 4.99999851e-1;
    pub(super) const EXP_P5: f32 = 1.0;
    pub(super) const EXP_P6: f32 = 1.0;
    /// `1.5 · 2^23`: adding then subtracting it rounds a float to the nearest
    /// integer under the default rounding mode without a `roundps`.
    pub(super) const EXP_C: f32 = 1.25829120e7;
}

/// `MlasExpConstants` (`onnxruntime/core/mlas/lib/compute.cpp`), the parameter
/// set behind `MlasComputeExp`.
///
/// These are *not* the `erf_c::EXP_*` values above. `erf_c`'s copy is the older
/// polynomial extracted from `MlasErfConstants`, valid only on
/// `[-88.376, 0]` — enough for `erf`'s big branch, which never sees a positive
/// argument. A standalone `Exp` operator has to cover the whole `f32` line, so
/// MLAS uses a separately refined polynomial plus XNNPACK's two-piece exponent
/// reconstruction, which extends the representable output range down to
/// `-103.972` (subnormal results) and up to `88.776` (overflow to `+Inf`).
#[cfg(target_arch = "x86_64")]
mod exp_c {
    /// Below this every result is `0`; `-Inf` clamps here.
    pub(super) const LOWER_RANGE: f32 = -103.9720840454;
    /// Above this every result overflows `f32`; `+Inf` clamps here and the
    /// reconstruction below still evaluates to `+Inf`.
    pub(super) const UPPER_RANGE: f32 = 88.7762626647950;
    /// `1.5 · 2^23`, the round-to-nearest-integer magic constant. Also reused
    /// as the raw bit source for the exponent reconstruction.
    pub(super) const ROUNDING_BIAS: f32 = 1.25829120e7;
    pub(super) const LOG2_RECIPROCAL: f32 = std::f32::consts::LOG2_E;
    pub(super) const LOG2_HIGH: f32 = -6.93145752e-1;
    pub(super) const LOG2_LOW: f32 = -1.42860677e-6;
    pub(super) const P0: f32 = f32::from_bits(0x3AB4_A000);
    pub(super) const P1: f32 = f32::from_bits(0x3C09_2F6E);
    pub(super) const P2: f32 = f32::from_bits(0x3D2A_ADAD);
    pub(super) const P3: f32 = f32::from_bits(0x3E2A_AA28);
    pub(super) const P4: f32 = f32::from_bits(0x3EFF_FFFB);
    /// MLAS stores a single `poly_56` field because the degree-5 and degree-6
    /// coefficients are both exactly `1.0`. In `MlasComputeExpVector` — the
    /// variant ported here — only one of them appears in the Horner chain; the
    /// other is merged into the overflow-exponent multiply/add below, following
    /// XNNPACK. (The reduced-range helper further down `compute.cpp` instead
    /// applies `poly_56` twice, because it has no overflow term to fold into.)
    pub(super) const P56: f32 = 1.0;
    /// Exponent field clamps for the two-piece `2^m` reconstruction.
    pub(super) const MINIMUM_EXPONENT: i32 = 0xC100_0000u32 as i32;
    pub(super) const MAXIMUM_EXPONENT: i32 = 0x3F80_0000;
}

/// `√(2/π)` and the cubic coefficient of the tanh GELU approximation, rounded
/// to `f32`. Matches ORT's `contrib_ops/cpu/bert/fast_gelu.cc`.
const GELU_B: f32 = 0.7978845608028654;
const GELU_C: f32 = 0.044715;

/// `1/√2`, the inner scale of exact GELU, rounded to `f32`. ORT's
/// `Gelu(approximate="none")` CPU kernel scales by `M_SQRT1_2` in `float`
/// before calling `MlasComputeErf`, so this matches its evaluation order.
#[cfg(target_arch = "x86_64")]
const FRAC_1_SQRT_2_F32: f32 = std::f32::consts::FRAC_1_SQRT_2;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Returns `true` when the AVX2+FMA vector kernels in this module are live.
///
/// Deliberately answers on every architecture (`false` off `x86_64`) so tests
/// can branch on it portably; only the `x86_64` dispatch arms call it, hence
/// the conditional `dead_code` allowance.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
pub(crate) fn vector_path_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Dispatch helper: run `vector` when AVX2+FMA is present and the slice is
/// long enough to amortise it, otherwise map `scalar` element-wise.
///
/// `input` and `output` must have equal length; the caller guarantees it.
macro_rules! dispatch {
    ($input:expr, $output:expr, $scalar:expr, $vector:expr) => {{
        let input: &[f32] = $input;
        let output: &mut [f32] = $output;
        debug_assert_eq!(input.len(), output.len());
        #[cfg(target_arch = "x86_64")]
        {
            if input.len() >= SIMD_MIN_LEN && vector_path_available() {
                // The vector/scalar choice is made once, on the whole slice,
                // and every chunk then inherits it. Deciding per chunk would
                // let a short tail chunk take the scalar path, and the two are
                // not bit-identical for the approximated kernels — results
                // would start depending on the thread count.
                //
                // SAFETY: guarded by the runtime AVX2+FMA detection above.
                run_chunked(input, output, |i, o| unsafe { $vector(i, o) });
                return;
            }
        }
        let scalar = $scalar;
        run_chunked(input, output, |i, o| {
            for (o, &i) in o.iter_mut().zip(i) {
                *o = scalar(i);
            }
        });
    }};
}

/// Route an elementwise f32 kernel to MLAS when the `mlas` feature is on.
///
/// Only `Erf` and exact `Gelu` still take this route. `Tanh` and `Sigmoid`
/// used to and no longer do — see the note below this macro for why.
///
/// MLAS is the library ONNX Runtime's own CPU activation kernels call, so this
/// is not "a faster approximation" — it is *the same* polynomial ORT uses.
///
/// Bit-reproducibility is **host-specific**, not universal. MLAS runtime
/// dispatches by ISA, so an AVX-512 or non-AVX2 host may select a different
/// kernel than the AVX2+FMA one the pure-Rust path mirrors. On this host and in
/// CI (AVX2+FMA, AVX-512 masked) the two routes are bit-identical, pinned by
/// `mlas_ab::mlas_matches_rust_simd_on_special_values`; elsewhere they may
/// differ within the documented tolerance. Before this change an `mlas`-on and
/// an `mlas`-off build agreed bitwise for these ops on every host, and
/// that is what is being traded for the speed. Measured on this
/// host (AMD EPYC 9V74, AVX2+FMA, AVX-512 masked off), the pure-Rust AVX2 path
/// is compute-bound at ~0.78 ns/elem while MLAS reaches ~0.35 ns/elem, which is
/// memory bandwidth for a 4 B-in/4 B-out stream. The gap is the polynomial, not
/// the loop.
///
/// The pure-Rust path stays as the fallback: it is what builds without the
/// feature, what runs on non-x86, and what the dense numeric sweeps still test.
///
/// `$mlas` is only reached for lengths at or above [`SIMD_MIN_LEN`]. Below that
/// the FFI call and MLAS's own dispatch cost more than the arithmetic saved, so
/// short tensors keep the scalar loop.
macro_rules! dispatch_mlas {
    ($input:expr, $output:expr, $scalar:expr, $vector:expr, $mlas:expr) => {{
        #[cfg(feature = "mlas")]
        {
            let input: &[f32] = $input;
            let output: &mut [f32] = $output;
            debug_assert_eq!(input.len(), output.len());
            if input.len() >= SIMD_MIN_LEN {
                let f: fn(&[f32], &mut [f32]) = $mlas;
                $crate::dispatch_ledger::record_with(|| {
                    $crate::dispatch_ledger::Observation::elementwise(
                        $crate::dispatch_ledger::KernelFamily::Activations,
                        // `run_chunked` and the special-value repair are ours;
                        // only the inner transcendental is MLAS's.
                        $crate::dispatch_ledger::Backend::NativeOverMlas,
                        "f32",
                        input.len(),
                    )
                });
                // Through `run_chunked`, exactly like the pure-Rust routes.
                // Calling `f(input, output)` directly here left every MLAS
                // route single-threaded no matter how many threads the pool
                // had, which cost 3.4-4.3x at 16 threads once the pure-Rust
                // routes started parallelising.
                run_chunked(input, output, f);
                return;
            }
        }
        $crate::dispatch_ledger::record_with(|| {
            $crate::dispatch_ledger::Observation::elementwise(
                $crate::dispatch_ledger::KernelFamily::Activations,
                $crate::dispatch_ledger::Backend::Native,
                "f32",
                $input.len(),
            )
        });
        dispatch!($input, $output, $scalar, $vector)
    }};
}

// Why `Tanh` and `Sigmoid` do *not* have an MLAS route.
//
// They used to. MLAS's `MlasComputeTanh`/`MlasComputeLogistic` are not
// range-preserving — over a dense sweep of [-20, 20] tanh lands outside
// `[-1, 1]` for 2934 of 1048576 points (by up to 2 ulp) and logistic outside
// `[0, 1]` for 2. Our kernels guarantee the range and
// `monotonicity_within_documented_slack` asserts it, so the MLAS route had to
// re-read every block it had just written and clamp it.
//
// That fix-up pass is now more expensive than the polynomial it was buying.
// The two kernels used the *same* Eigen rational as MLAS even before this,
// and once the redundant saturation blends came out of them (#1121) the
// pure-Rust AVX2 path became outright faster than MLAS-plus-clamp. Measured on
// this host, `mlas`-off vs `mlas`-on p50, 1 thread:
//
// | op      |  64 Ki |  256 Ki |    1 Mi |    4 Mi |
// |---------|-------:|--------:|--------:|--------:|
// | Tanh    | 32.5 / 36.2 | 105.6 / 146.7 | 438.4 / 469.4 | 1896 / 2226 |
// | Sigmoid | 31.2 / 38.3 | 113.7 / 129.3 | 408.9 / 525.8 | 1887 / 2268 |
//
// The `mlas` route lost at every size — up to 1.39x on `Tanh` at 256 Ki — so
// it is gone and both builds now run the same kernel. `Erf` and exact `Gelu`
// keep theirs: `Erf` needs no fix-up at all and `Gelu`'s is a non-writing
// compare scan, and both still win (1.4-1.6x and 1.17x at 4 Mi).
//

/// Elements per MLAS + fix-up block.
///
/// The fix-up passes below re-read the block MLAS just wrote. Run over the
/// whole tensor that is a second trip to DRAM and costs more than the win; run
/// per block it lands in L1/L2 and is nearly free. 8192 f32 = 32 KiB, so a
/// block plus its input stays inside a 512 KiB L2 with room to spare.
#[cfg(feature = "mlas")]
const MLAS_FIXUP_BLOCK: usize = 8192;

/// MLAS's exact GELU, with its one special-value divergence repaired.
///
/// `MlasComputeGeluErf` evaluates `x·0.5·(1 + erf(x/√2))` without special-casing
/// the `-inf` limit, so it computes `-inf · 0` and returns NaN where the limit
/// is `0`. That is the *only* input on which MLAS and the pure-Rust path
/// disagree, across `Tanh`, `Sigmoid`, `Erf` and exact `Gelu` and every special
/// value — NaN (both signs), ±Inf, ±0, ±`MIN_POSITIVE`, the smallest
/// subnormals, `MAX`/`MIN` and the exp saturation thresholds. That claim is
/// pinned by `mlas_ab::mlas_matches_rust_simd_on_special_values`, which fails
/// if a future MLAS bump changes any of it.
///
/// Note this makes us *disagree with ORT*, which calls `MlasComputeGeluErf`
/// directly and therefore returns NaN here. Returning the limit is the better
/// answer — it is what the pure-Rust path already returned, and NaN would
/// propagate through the rest of the graph — so the divergence is deliberate
/// and is not something to "fix" by matching ORT.
///
/// A NaN in the output can only arise from a NaN input, on which both paths
/// agree, or from this case. So scanning the output — hot in cache the instant
/// MLAS wrote it — and repairing only the `-inf` lanes is exact, and costs one
/// vectorisable compare pass on the common all-finite path.
#[cfg(feature = "mlas")]
fn erf_gelu_mlas(input: &[f32], output: &mut [f32]) {
    // Blocked so the repair scan reads each block while it is still in L1/L2.
    for (xs, ys) in input
        .chunks(MLAS_FIXUP_BLOCK)
        .zip(output.chunks_mut(MLAS_FIXUP_BLOCK))
    {
        mlas_sys::compute_gelu_erf(xs, ys);
        // Branch-free OR-reduction rather than `any()`: the early exit in
        // `any()` is a loop-carried control dependency that stops LLVM
        // vectorising, and measured 0.7 ns/elem — more than the whole win.
        // Folding the NaN predicate into an accumulator keeps the scan a
        // straight AVX2 compare.
        let mut saw_nan = 0u32;
        for v in ys.iter() {
            saw_nan |= u32::from(v.is_nan());
        }
        if saw_nan != 0 {
            for (o, &i) in ys.iter_mut().zip(xs) {
                if o.is_nan() && i == f32::NEG_INFINITY {
                    *o = 0.0;
                }
            }
        }
    }
}

/// `y = tanh(x)`.
pub(crate) fn tanh_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, tanh_scalar, tanh_avx2);
}

/// `y = √x`.
///
/// Unlike every other kernel in this module this is **not** an approximation.
/// `vsqrtps` and `sqrtss` are the same correctly-rounded IEEE-754 square root,
/// so the vector body, the scalar tail and the pre-existing `f32::sqrt` kernel
/// this replaces all produce bit-identical results — `-0.0 -> -0.0`,
/// `x < 0 -> NaN`, `+Inf -> +Inf`, subnormals exact. Nothing here trades
/// accuracy for speed; the win is eight lanes per instruction plus the caller
/// no longer materialising an intermediate `Vec` per call.
///
/// Both instructions read the same `MXCSR`, so the equivalence also holds under
/// flush-to-zero / denormals-are-zero: if the host process has set `FTZ`/`DAZ`
/// then subnormals are flushed by the replacement exactly as they were by the
/// code it replaces. "Subnormals exact" above is a statement about the default
/// `MXCSR`, not a guarantee this kernel adds or removes.
pub(crate) fn sqrt_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::sqrt, sqrt_avx2);
}

/// `y = 1 / (1 + e^-x)`.
pub(crate) fn sigmoid_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, sigmoid_scalar, sigmoid_avx2);
}

/// `y = e^x`.
///
/// The vector path is a port of `MlasComputeExpVector`
/// (`onnxruntime/core/mlas/lib/compute.cpp`), the same polynomial MLAS uses
/// inside its own softmax and logistic kernels. The scalar fallback is
/// `f32::exp`, which is correctly rounded, so — exactly as for [`erf_f32_slice`]
/// — a value can differ by ~1 ulp depending on whether the tensor was long
/// enough to reach [`SIMD_MIN_LEN`]. `Exp` has no bit-exactness contract in
/// ONNX and ORT's own CPU kernel is a different (Eigen) approximation again, so
/// this seam is a documented accuracy property, not a correctness bug. Over a
/// 65536-point sweep of `[-110, 89]` the worst observed error against an `f64`
/// reference is **1 ulp**, including through the subnormal range.
///
/// Special values match `f32::exp` and ORT: `NaN` in gives `NaN` out (see
/// [`avx2::exp_full_ps`] — it falls out of the clamp's operand order rather
/// than a mask), `+Inf` gives `+Inf`, `-Inf` gives `+0`, arguments above
/// `88.7762626647950` overflow to `+Inf`, and arguments below
/// `-103.9720840454` flush to `+0` through the subnormal range.
pub(crate) fn exp_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::exp, exp_avx2);
}

/// `y = erf(x)`, the Gauss error function.
///
/// The vector path is a port of `MlasErfKernel` (`onnxruntime/core/mlas/lib/
/// erf.cpp`), which is what ORT's own CPU `Erf` and `Gelu(approximate="none")`
/// kernels evaluate via `MlasComputeErf`. It is a *faithfully rounded* `f32`
/// approximation, not the correctly-rounded `libm::erf` the scalar fallback
/// uses: see the module-level note on ISA dependence. Measured against an
/// `f64` reference over a dense sweep the worst observed error is 5.96e-8
/// (1 ulp below `1.0`), and against ORT 1.28.0 on identical inputs the two
/// agree bit-for-bit over 4M+ probed points — which is the point of porting
/// MLAS's coefficients rather than inventing a polynomial.
///
/// # The `SIMD_MIN_LEN` seam
///
/// Slices shorter than [`SIMD_MIN_LEN`] take the correctly-rounded scalar
/// fallback, so the *same* input value can differ by up to 1 ulp depending on
/// how many elements the tensor has — measured at 286 of 2000 random values
/// between a 31-element and a 40-element tensor. `Tanh` and `Sigmoid` have had
/// exactly this seam since they were vectorised, and ORT has it too (MLAS
/// dispatches its own scalar tail below the vector width). It is accepted for
/// the same reason: 1 ulp is three orders of magnitude inside the conformance
/// tolerance, and the alternative is making short tensors 20x slower to buy
/// bit-stability nothing depends on.
pub(crate) fn erf_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch_mlas!(input, output, erf_scalar, erf_avx2, mlas_sys::compute_erf);
}

/// `Erf` on the native route only, whatever this build linked.
///
/// Kept callable in every build so [`crate::backend_ab`] can hold the native and
/// MLAS routes against each other inside one process — the only way to measure
/// or differentially test an absorption honestly.
pub(crate) fn erf_f32_slice_native(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, erf_scalar, erf_avx2);
}

/// `y = 0.5·x·(1 + erf(x/√2))`, the exact (`approximate="none"`) GELU.
///
/// Fused rather than composed out of [`erf_f32_slice`] so the intermediate
/// `x/√2` is never written to memory.
pub(crate) fn erf_gelu_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch_mlas!(input, output, erf_gelu_scalar, erf_gelu_avx2, erf_gelu_mlas);
}

/// Exact `Gelu` on the native route only. See [`erf_f32_slice_native`].
pub(crate) fn erf_gelu_f32_slice_native(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, erf_gelu_scalar, erf_gelu_avx2);
}

/// Exact `Gelu` on the MLAS route only (repair pass included), for A/B.
#[cfg(feature = "mlas")]
pub(crate) fn erf_gelu_f32_slice_mlas(input: &[f32], output: &mut [f32]) {
    run_chunked(input, output, erf_gelu_mlas);
}

/// `Erf` on the MLAS route only, for A/B.
#[cfg(feature = "mlas")]
pub(crate) fn erf_f32_slice_mlas(input: &[f32], output: &mut [f32]) {
    run_chunked(input, output, mlas_sys::compute_erf);
}

/// `y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, the tanh GELU
/// approximation used by `FastGelu`.
pub(crate) fn tanh_gelu_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, tanh_gelu_scalar, tanh_gelu_avx2);
}

/// `y = x·sigmoid(alpha·x)`, the `QuickGelu` / Swish form.
pub(crate) fn quick_gelu_f32_slice(input: &[f32], output: &mut [f32], alpha: f32) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above.
            run_chunked(input, output, |i, o| unsafe {
                quick_gelu_avx2(i, o, alpha)
            });
            return;
        }
    }
    run_chunked(input, output, |i, o| {
        for (o, &i) in o.iter_mut().zip(i) {
            *o = quick_gelu_scalar(i, alpha);
        }
    });
}

/// `tanh_gelu(x[i] + bias[i % width])` for every element, without ever
/// materialising `x + bias`.
///
/// FastGelu's bias is a broadcast over the last dimension, so folding it into a
/// scratch row before the transcendental costs a full extra write *and* read of
/// the activation tensor — at prefill sizes that is more traffic than the GELU
/// itself. Adding it in-register instead keeps the whole op at one read of `x`,
/// one write of `y`, and repeated reads of a `width`-element bias row that stays
/// resident in L1.
///
/// `width` must be non-zero and `bias.len()` must equal `width`. A trailing
/// partial row is written too, consuming the matching prefix of the bias, which
/// is what ONNX's `bias[i % width]` broadcast means. Results are bit-identical
/// to folding `x + bias` first and mapping over it, because the in-register add
/// is the same IEEE `f32` addition in the same order.
pub(crate) fn tanh_gelu_bias_f32_slice(
    input: &[f32],
    bias: &[f32],
    width: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(bias.len(), width);
    debug_assert!(width != 0);
    #[cfg(target_arch = "x86_64")]
    {
        // Gated on total length, exactly as `tanh_gelu_f32_slice` is: whether a
        // FastGelu node carries a bias must not change which polynomial its
        // elements go through. `map_bias_ps` handles `width < 8` through its
        // masked tail, so narrow rows stay correct (if unexciting) here.
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above; the
            // debug asserts above are the caller's contract.
            run_chunked_rows(input, output, width, |i, o| unsafe {
                tanh_gelu_bias_avx2(i, bias, width, o)
            });
            return;
        }
    }
    run_chunked_rows(input, output, width, |i, o| {
        for (row_in, row_out) in i.chunks(width).zip(o.chunks_mut(width)) {
            for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(bias) {
                *o = tanh_gelu_scalar(v + b);
            }
        }
    });
}

/// `erf_gelu(x[i] + bias[i % width])`, the `BiasGelu` contrib op.
///
/// Same in-register bias fold as [`tanh_gelu_bias_f32_slice`], and the same
/// contract: `width` non-zero, `bias.len() == width`, a trailing partial row
/// consumes the matching bias prefix.
pub(crate) fn erf_gelu_bias_f32_slice(
    input: &[f32],
    bias: &[f32],
    width: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(bias.len(), width);
    debug_assert!(width != 0);
    #[cfg(target_arch = "x86_64")]
    {
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above; the
            // debug asserts above are the caller's contract.
            run_chunked_rows(input, output, width, |i, o| unsafe {
                erf_gelu_bias_avx2(i, bias, width, o)
            });
            return;
        }
    }
    run_chunked_rows(input, output, width, |i, o| {
        for (row_in, row_out) in i.chunks(width).zip(o.chunks_mut(width)) {
            for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(bias) {
                *o = erf_gelu_scalar(v + b);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Exact (bit-identical) elementwise kernels
// ---------------------------------------------------------------------------
//
// Unlike the rest of this module these are **not** approximations. Each is a
// sign-bit mask, an IEEE-754 division, or `vroundps` in an explicit rounding
// mode, so the vector body, the masked tail and the scalar fallback all produce
// bit-identical results — `-0.0`, `±Inf`, subnormals and NaN payloads included.
// There is no `SIMD_MIN_LEN` accuracy seam here, only a throughput one.
//
// They needed their own kernels because `UnaryMathKernel::execute_f32` used to
// hand `write_mapped` a closure that ran `MathOp::apply` — a 24-arm `match` —
// *inside* the element loop. LLVM could not hoist it, so even `Neg`, which is
// one `vxorps`, compiled to a per-element jump table. Measured against ORT that
// put `Neg` at 0.049x at 1M elements while the plugin EP was claiming the op.
// ---------------------------------------------------------------------------

/// `y = -x`. Flips the sign bit; exact for every input, `NaN` keeps its payload
/// with the sign flipped and `-0.0 -> +0.0`.
pub(crate) fn neg_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, |v: f32| -v, neg_avx2);
}

/// `y = |x|`. Clears the sign bit; exact, and `|NaN|` keeps its payload.
pub(crate) fn abs_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::abs, abs_avx2);
}

/// `y = 1 / x`.
///
/// `vdivps`, **not** `vrcpps`: the reciprocal approximation carries only ~12
/// bits and would break the bit-exactness this group promises. Division is
/// correctly rounded, so this matches `1.0 / x` exactly, including
/// `1/+0 = +Inf` and `1/-0 = -Inf`.
pub(crate) fn reciprocal_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, |v: f32| 1.0 / v, reciprocal_avx2);
}

/// `y = floor(x)`, `vroundps` mode 1 — identical to `f32::floor`.
pub(crate) fn floor_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::floor, floor_avx2);
}

/// `y = ceil(x)`, `vroundps` mode 2 — identical to `f32::ceil`.
pub(crate) fn ceil_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::ceil, ceil_avx2);
}

/// `y = round-half-to-even(x)`, `vroundps` mode 0.
///
/// ONNX `Round` is banker's rounding, which is `f32::round_ties_even` and
/// `_MM_FROUND_TO_NEAREST_INT` — *not* `f32::round`, which rounds halves away
/// from zero.
pub(crate) fn round_ties_even_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::round_ties_even, round_ties_even_avx2);
}

/// ONNX `Sign`: `-1` / `0` / `+1`, with `sign(±0) = 0` and `sign(NaN) = NaN`.
///
/// Built from two *ordered* compares, which are false for `NaN`, so a `NaN`
/// input falls through both selects and is returned unchanged — matching the
/// scalar `is_nan()` branch bit-for-bit, payload included.
pub(crate) fn sign_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, sign_scalar, sign_avx2);
}

/// `y = x / (1 + |x|)`, ONNX `Softsign`.
///
/// One `vandps` and one `vdivps`, both exact, so this is bit-identical to the
/// scalar form rather than an approximation of it.
pub(crate) fn softsign_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, softsign_scalar, softsign_avx2);
}

/// ONNX `Sign`: `-1` / `0` / `+1`, with `sign(±0) = +0` and `sign(NaN) = NaN`.
///
/// The `NaN` input is returned **unchanged**, payload and sign bit included.
/// This function used to return the canonical `f32::NAN` instead, which
/// silently rewrote `0xFFC01234` to `0x7FC01234`. ORT 1.28.0's CPU `Sign`
/// preserves the input bit pattern (verified on both signs and a non-default
/// payload), and so does the AVX2 path below — its ordered compares are all
/// false for `NaN`, so the lane falls through every select. Canonicalising was
/// the odd one out, and it is what made the two paths disagree.
#[inline]
pub(crate) fn sign_scalar(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// ONNX `Softsign`: `x / (1 + |x|)`.
#[inline]
pub(crate) fn softsign_scalar(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

// ---------------------------------------------------------------------------
// Scalar reference implementations
//
// These are the *exact* libm forms, not the polynomial. They are what runs
// when AVX2+FMA is unavailable, so a legacy target keeps today's accuracy and
// today's speed rather than inheriting a slow software-`fma` polynomial.
// ---------------------------------------------------------------------------

#[inline]
fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

/// Correctly-rounded `erf`, the fallback when AVX2+FMA is absent. `libm::erf`
/// is `f64`, so this is what makes the non-x86 path both exact and slow; see
/// [`erf_f32_slice`].
#[inline]
fn erf_scalar(x: f32) -> f32 {
    crate::kernels::elementwise::erf(f64::from(x)) as f32
}

/// Exact GELU on the scalar fallback, in `f64` throughout to match the
/// pre-existing `kernels::gelu::exact_gelu`.
#[inline]
fn erf_gelu_scalar(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = f64::from(x);
    (0.5 * xf * (1.0 + crate::kernels::elementwise::erf(xf * std::f64::consts::FRAC_1_SQRT_2)))
        as f32
}

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn tanh_gelu_scalar(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = x as f64;
    let inner = f64::from(GELU_B) * (xf + f64::from(GELU_C) * xf * xf * xf);
    (0.5 * xf * (1.0 + inner.tanh())) as f32
}

#[inline]
fn quick_gelu_scalar(x: f32, alpha: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    x * sigmoid_scalar(alpha * x)
}

// ---------------------------------------------------------------------------
// AVX2 + FMA kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GELU_B, GELU_C, erf_c, exp_c, logistic_c, tanh_c};
    use core::arch::x86_64::*;

    /// `[-1; 7] ++ [0; 8]`. Loading 8 lanes at offset `7 - rem` yields a mask
    /// with exactly `rem` active lanes for `rem` in `1..=7`.
    #[rustfmt::skip]
    static MASK_TABLE: [i32; 15] = [
        -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// Mask selecting the low `rem` lanes. `rem` must be in `1..=7`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn tail_mask(rem: usize) -> __m256i {
        debug_assert!((1..=7).contains(&rem));
        // SAFETY: `7 - rem` is in `0..=6`, so the 8-lane read stays inside the
        // 15-element table.
        unsafe { _mm256_loadu_si256(MASK_TABLE.as_ptr().add(7 - rem).cast()) }
    }

    /// MLAS's NaN-preserving two-step clamp. `maxps`/`minps` return their
    /// *second* operand when either input is NaN, so passing the value second
    /// lets NaN through.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn clamp_nan_preserving(v: __m256, lower: f32, upper: f32) -> __m256 {
        let v = _mm256_max_ps(_mm256_set1_ps(lower), v);
        _mm256_min_ps(_mm256_set1_ps(upper), v)
    }

    /// `tanh` over 8 lanes, following `MlasTanhKernel` but saturating to `±1`
    /// outside `[-9, 9]` instead of returning the polynomial at the clamp
    /// point.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tanh_ps(x: __m256) -> __m256 {
        unsafe {
            let v = clamp_nan_preserving(x, tanh_c::LOWER, tanh_c::UPPER);
            let v2 = _mm256_mul_ps(v, v);

            let mut p = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(tanh_c::ALPHA_13),
                _mm256_set1_ps(tanh_c::ALPHA_11),
            );
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_9));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_7));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_5));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_3));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_1));
            p = _mm256_mul_ps(p, v);

            let mut q = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(tanh_c::BETA_6),
                _mm256_set1_ps(tanh_c::BETA_4),
            );
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(tanh_c::BETA_2));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(tanh_c::BETA_0));

            // The rational overshoots `tanh`'s mathematical range. Sweeping
            // every f32 in `[8, 9]` through this exact FMA evaluation order,
            // `p/q` exceeds `1.0` for 57 437 of them, spanning
            // `[8.127431, 8.999997]` and peaking at `1.0000002` near
            // `|x| = 8.4755`. ORT ships the overshoot — ORT 1.28.0 CPU
            // `Tanh(8.442762)` returns `1.0000001` — and downstream code is
            // entitled to assume `|tanh| <= 1`, so we pin to `[-1, 1]`. This
            // is a deliberate, measured divergence from ORT in favour of the
            // mathematical range. (The counts are FMA-specific: the same
            // constants evaluated without fusion overshoot on only 26 503
            // points over `[8.052297, 8.999964]`.)
            // The `[-1, 1]` clamp is also the saturation. Because the input was
            // already clamped to `[-9, 9]`, the largest argument the rational
            // ever sees is `±9` — and there `p` and `q` come out bit-equal
            // (`0x3fcd33e9`), so `p/q` is *exactly* `±1.0` and the clamp,
            // being inclusive, returns it unchanged. The margin is equality,
            // not slack: the overshoot to `1.0000001` happens strictly inside
            // the range, near `|v| = 8.9999971`, and is what the clamp is
            // actually for. So every `|x| >= 9` — up to and including `±Inf` —
            // already leaves here as exactly `±1` without a separate
            // saturation step, and `NaN` survives because `maxps`/`minps`
            // return their second operand on an unordered compare.
            // `saturation_blend_is_redundant_exhaustively` proves this over
            // all 1 047 527 424 finite `f32` with `|x| >= 9`, under the
            // default rounding mode (the property also holds under the other
            // three, but only round-to-nearest is exercised in CI).
            clamp_nan_preserving(_mm256_div_ps(p, q), -1.0, 1.0)
        }
    }

    /// `sigmoid` over 8 lanes, following `MlasLogisticKernel` but saturating
    /// to `0` / `1` outside `[-18, 18]`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn sigmoid_ps(x: __m256) -> __m256 {
        unsafe {
            let v = clamp_nan_preserving(x, logistic_c::LOWER, logistic_c::UPPER);
            let v2 = _mm256_mul_ps(v, v);

            let mut p = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(logistic_c::ALPHA_9),
                _mm256_set1_ps(logistic_c::ALPHA_7),
            );
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_5));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_3));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_1));
            p = _mm256_mul_ps(p, v);

            let mut q = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(logistic_c::BETA_10),
                _mm256_set1_ps(logistic_c::BETA_8),
            );
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_6));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_4));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_2));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_0));

            let poly = _mm256_add_ps(_mm256_div_ps(p, q), _mm256_set1_ps(0.5));

            // As in `tanh_ps`, the `[0, 1]` clamp is also the saturation. The
            // rational only ever sees `±18`: at `+18` it gives exactly `1.0`,
            // which the inclusive clamp passes through, and at `-18` it gives
            // `-5.96e-8`, which the clamp pulls to `0.0`. Either way `|x| >=
            // 18` and `±Inf` leave here saturated without a separate step.
            // `saturation_blend_is_redundant_exhaustively` proves this over
            // all 1 039 138 816 finite `f32` with `|x| >= 18`.
            clamp_nan_preserving(poly, 0.0, 1.0)
        }
    }

    /// `0.5·x·(1 + tanh(B·(x + C·x³)))` over 8 lanes.
    ///
    /// `x = -Inf` is pinned to `0` to match the scalar kernel: the natural
    /// evaluation gives `0.5·(-Inf)·(1 - 1) = NaN`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tanh_gelu_ps(x: __m256) -> __m256 {
        unsafe {
            let x2 = _mm256_mul_ps(x, x);
            // B·x + B·C·x³, arranged so a single fma covers the cubic term.
            let inner = _mm256_mul_ps(
                _mm256_set1_ps(GELU_B),
                _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(GELU_C), x2), x, x),
            );
            let t = tanh_ps(inner);
            let y = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_set1_ps(0.5), x),
                _mm256_add_ps(_mm256_set1_ps(1.0), t),
            );
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
        }
    }

    /// `x·sigmoid(alpha·x)` over 8 lanes, with `x = -Inf` pinned to `0`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quick_gelu_ps(x: __m256, alpha: __m256) -> __m256 {
        unsafe {
            let s = sigmoid_ps(_mm256_mul_ps(alpha, x));
            let y = _mm256_mul_ps(x, s);
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
        }
    }

    /// `erf` over 8 lanes, following `MlasErfKernel` step for step.
    ///
    /// Both branches are evaluated for every lane and merged with `or`, which
    /// is what MLAS does: the inactive branch is forced to `+0.0` — the small
    /// branch by `andnot(split, ..)`, the big branch because zeroing its input
    /// collapses the polynomial to `1 - exp(-0) = 0` — so the `or` acts as a
    /// select. Being branch-free is why this beats a scalar `libm::erf` by far
    /// more than the 8× the SIMD width alone would give.
    ///
    /// The sign is stripped up front (`erf` is odd) and re-applied at the end
    /// with an `or`, so `erf(-0.0) = -0.0` and a negative `NaN` stays negative.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn erf_ps(x: __m256) -> __m256 {
        unsafe {
            let neg_zero = _mm256_set1_ps(-0.0);
            let sign = _mm256_and_ps(x, neg_zero);
            // `minps` returns its *second* operand when either is NaN, so a
            // NaN input survives the clamp — matching MLAS's operand order.
            let abs = _mm256_min_ps(
                _mm256_set1_ps(erf_c::UPPER_ABS_RANGE),
                _mm256_andnot_ps(neg_zero, x),
            );
            let sq = _mm256_mul_ps(abs, abs);

            // |x| <= 0.921875: erf(x) = |x|·(1 + P(x²)).
            let mut small = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::SMALL_P0),
                sq,
                _mm256_set1_ps(erf_c::SMALL_P1),
            );
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P2));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P3));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P4));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P5_MINUS_ONE));
            small = _mm256_fmadd_ps(small, abs, abs);

            // Ordered `>`, so a NaN lane is *false* here and therefore keeps
            // the small branch, whose polynomial already produced that NaN.
            let split = _mm256_cmp_ps(abs, _mm256_set1_ps(erf_c::SPLIT_BOUNDARY), _CMP_GT_OQ);
            let small = _mm256_andnot_ps(split, small);

            // |x| > 0.921875: erf(x) = 1 - exp(-R(|x|)).
            let abs = _mm256_and_ps(split, abs);
            let mut big = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::BIG_P0),
                abs,
                _mm256_set1_ps(erf_c::BIG_P1),
            );
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P2));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P3));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P4));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P5));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P6_MINUS_ONE));
            big = _mm256_fmadd_ps(big, abs, abs);

            let neg_big = _mm256_max_ps(
                _mm256_set1_ps(erf_c::EXP_LOWER_RANGE),
                _mm256_xor_ps(big, neg_zero),
            );
            let y = _mm256_sub_ps(_mm256_set1_ps(1.0), exp_ps(neg_big));

            _mm256_or_ps(_mm256_or_ps(small, y), sign)
        }
    }

    /// `exp` over 8 lanes for arguments already clamped to
    /// `[EXP_LOWER_RANGE, 0]`, using `MlasErfConstants`' `exp` parameters.
    ///
    /// Range-reduces `x = k·ln2 + f` with `k` obtained by the add-then-subtract
    /// round-to-integer trick, evaluates a degree-6 polynomial on `f`, then
    /// scales by `2^k` built directly in the exponent field. Only the `erf` big
    /// branch calls this, so it deliberately does *not* handle overflow, `Inf`
    /// or `NaN`: those lanes are masked off before they reach it.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_ps(x: __m256) -> __m256 {
        unsafe {
            let magic = _mm256_set1_ps(erf_c::EXP_C);
            let k = _mm256_fmadd_ps(_mm256_set1_ps(erf_c::EXP_LOG2_RECIPROCAL), x, magic);
            let k = _mm256_sub_ps(k, magic);

            let mut f = _mm256_fmadd_ps(k, _mm256_set1_ps(erf_c::EXP_LOG2_HI), x);
            f = _mm256_fmadd_ps(k, _mm256_set1_ps(erf_c::EXP_LOG2_LO), f);

            let mut p = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::EXP_P0),
                f,
                _mm256_set1_ps(erf_c::EXP_P1),
            );
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P2));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P3));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P4));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P5));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P6));

            _mm256_mul_ps(p, power_of_2_ps(k))
        }
    }

    /// Full-range `exp` over 8 lanes: a port of `MlasComputeExpVector`.
    ///
    /// Unlike [`exp_ps`], which the `erf` big branch calls with arguments that
    /// are already clamped to `[-88.376, 0]`, this covers the entire `f32`
    /// line. Two differences buy that:
    ///
    /// * `2^m` is reconstructed in **two** pieces (XNNPACK's refinement). One
    ///   exponent field cannot express the full `[-150, 128]` range a
    ///   general-purpose `exp` needs, so the biased exponent is split into a
    ///   clamped `normal` part and an `overflow` remainder, and the two are
    ///   applied at different points in the Horner chain. This is what lets
    ///   subnormal results come out right instead of flushing early.
    /// # `NaN` survives without a mask, and the clamp's operand order is why
    ///
    /// The reconstruction reinterprets `biased` as an integer, and for a `NaN`
    /// argument that integer is meaningless — so it is worth being explicit
    /// about why a `NaN` cannot come out finite. `MINPS`/`MAXPS` return their
    /// **second** operand when either input is unordered. Both clamp steps put
    /// the value being clamped second (`min(UPPER, x)`, then `max(LOWER, v)`),
    /// so a `NaN` argument passes through the clamp unchanged; the first
    /// polynomial `fmadd(P0, v, P1)` then carries it into `p`, and every later
    /// step multiplies or adds `p`, so the `NaN` reaches the result. It holds
    /// for every payload and both signs, quiet and signalling alike, because
    /// `NaN * anything` is `NaN` — including `NaN * 0` and `NaN * Inf`, the two
    /// values the integer path can produce.
    ///
    /// **The operand order in the clamp is therefore load-bearing.** Rewriting
    /// `min(UPPER, x)` as `min(x, UPPER)` would silently replace every `NaN`
    /// with `UPPER_RANGE` and turn `exp(NaN)` into `+Inf`.
    /// [`exp_tests::nan_propagates_instead_of_becoming_finite`] pins this.
    ///
    /// An explicit `_CMP_UNORD_Q` compare plus `blendv` was measured as the
    /// alternative and cost about 10% of throughput at 256 K elements for no
    /// behavioural difference.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_full_ps(x: __m256) -> __m256 {
        let mut v = _mm256_min_ps(_mm256_set1_ps(exp_c::UPPER_RANGE), x);
        v = _mm256_max_ps(_mm256_set1_ps(exp_c::LOWER_RANGE), v);

        let bias = _mm256_set1_ps(exp_c::ROUNDING_BIAS);
        let biased = _mm256_fmadd_ps(v, _mm256_set1_ps(exp_c::LOG2_RECIPROCAL), bias);
        let m = _mm256_sub_ps(biased, bias);

        v = _mm256_fmadd_ps(m, _mm256_set1_ps(exp_c::LOG2_HIGH), v);
        v = _mm256_fmadd_ps(m, _mm256_set1_ps(exp_c::LOG2_LOW), v);

        let max_exp = _mm256_set1_epi32(exp_c::MAXIMUM_EXPONENT);
        let min_exp = _mm256_set1_epi32(exp_c::MINIMUM_EXPONENT);
        let raw = _mm256_slli_epi32::<23>(_mm256_castps_si256(biased));
        let normal = _mm256_max_epi32(_mm256_min_epi32(raw, max_exp), min_exp);
        let overflow = _mm256_add_epi32(_mm256_sub_epi32(raw, normal), max_exp);
        let normal = _mm256_add_epi32(normal, max_exp);
        let overflow = _mm256_castsi256_ps(overflow);
        let normal = _mm256_castsi256_ps(normal);

        let mut p = _mm256_set1_ps(exp_c::P0);
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P1));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P2));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P3));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P4));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P56));

        v = _mm256_mul_ps(v, overflow);
        p = _mm256_fmadd_ps(p, v, overflow);
        p = _mm256_mul_ps(p, normal);

        p
    }

    /// `2^k` for integer-valued `k`, built by biasing and shifting into the
    /// exponent field (`MlasPowerOf2Float32x4`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn power_of_2_ps(k: __m256) -> __m256 {
        let e = _mm256_add_epi32(_mm256_cvttps_epi32(k), _mm256_set1_epi32(127));
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e))
    }

    /// `0.5·x·(1 + erf(x/√2))` over 8 lanes.
    ///
    /// `x = -Inf` is pinned to `0` for the same reason as [`tanh_gelu_ps`]:
    /// the natural evaluation is `0.5·(-Inf)·(1 - 1) = NaN`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn erf_gelu_ps(x: __m256) -> __m256 {
        unsafe {
            let e = erf_ps(_mm256_mul_ps(x, _mm256_set1_ps(super::FRAC_1_SQRT_2_F32)));
            let y = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_set1_ps(0.5), x),
                _mm256_add_ps(_mm256_set1_ps(1.0), e),
            );
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
        }
    }

    /// Apply an 8-lane kernel across a slice. The `< 8` remainder is processed
    /// through the *same* kernel via a masked load/store, so every element of
    /// the output — tail included — is computed identically.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    /// Like [`map_ps`], but adds a `width`-element bias row to each `width`-element
    /// slab of the input before applying `kernel`.
    pub(super) unsafe fn map_bias_ps(
        input: &[f32],
        bias: &[f32],
        width: usize,
        output: &mut [f32],
        kernel: impl Fn(__m256) -> __m256,
    ) {
        unsafe {
            let bptr = bias.as_ptr();
            // `chunks`, not `chunks_exact`: a tensor that is not a whole number
            // of rows must still get every element written. ONNX broadcasts the
            // bias as `bias[i % width]`, so a short final row consumes the
            // matching prefix of the bias — which is what the shorter row length
            // produces here.
            for (row_in, row_out) in input.chunks(width).zip(output.chunks_mut(width)) {
                let len = row_in.len();
                let body = len & !7;
                let rem = len - body;
                let src = row_in.as_ptr();
                let dst = row_out.as_mut_ptr();
                let mut i = 0;
                while i < body {
                    let v =
                        _mm256_add_ps(_mm256_loadu_ps(src.add(i)), _mm256_loadu_ps(bptr.add(i)));
                    _mm256_storeu_ps(dst.add(i), kernel(v));
                    i += 8;
                }
                if rem != 0 {
                    let mask = tail_mask(rem);
                    let v = _mm256_add_ps(
                        _mm256_maskload_ps(src.add(body), mask),
                        _mm256_maskload_ps(bptr.add(body), mask),
                    );
                    _mm256_maskstore_ps(dst.add(body), mask, kernel(v));
                }
            }
        }
    }

    pub(super) unsafe fn map_ps(
        input: &[f32],
        output: &mut [f32],
        kernel: impl Fn(__m256) -> __m256,
    ) {
        unsafe {
            let n = input.len();
            let src = input.as_ptr();
            let dst = output.as_mut_ptr();
            let body = n & !7;
            let mut i = 0;
            while i < body {
                _mm256_storeu_ps(dst.add(i), kernel(_mm256_loadu_ps(src.add(i))));
                i += 8;
            }
            let rem = n - body;
            if rem != 0 {
                let mask = tail_mask(rem);
                let v = _mm256_maskload_ps(src.add(body), mask);
                _mm256_maskstore_ps(dst.add(body), mask, kernel(v));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::tanh_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::sigmoid_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::exp_full_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sqrt_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| core::arch::x86_64::_mm256_sqrt_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn neg_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    // XOR the sign bit. `_mm256_sub_ps(zero, v)` would turn `-0.0` into `+0.0`
    // correctly but map `NaN` through an arithmetic op; the mask is exact.
    let sign = _mm256_set1_ps(-0.0);
    unsafe { avx2::map_ps(input, output, |v| _mm256_xor_ps(v, sign)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn abs_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    unsafe { avx2::map_ps(input, output, |v| _mm256_and_ps(v, mask)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn reciprocal_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    unsafe { avx2::map_ps(input, output, |v| _mm256_div_ps(one, v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn floor_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe { avx2::map_ps(input, output, |v| _mm256_floor_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ceil_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe { avx2::map_ps(input, output, |v| _mm256_ceil_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn round_ties_even_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        avx2::map_ps(input, output, |v| {
            _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v)
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sign_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let minus_one = _mm256_set1_ps(-1.0);
        avx2::map_ps(input, output, |v| {
            // `_CMP_GT_OQ`/`_CMP_LT_OQ` are *ordered*: both are false for NaN,
            // so a NaN lane selects neither `±1` nor `0` and `v` survives.
            let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(v, zero);
            let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(v, zero);
            // Zero (either sign) is ordered-equal to zero, so it takes this
            // arm and yields `+0.0`, which is what ONNX `Sign` specifies.
            let eq = _mm256_cmp_ps::<_CMP_EQ_OQ>(v, zero);
            let out = _mm256_blendv_ps(v, zero, eq);
            let out = _mm256_blendv_ps(out, one, pos);
            _mm256_blendv_ps(out, minus_one, neg)
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softsign_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        avx2::map_ps(input, output, |v| {
            _mm256_div_ps(v, _mm256_add_ps(one, _mm256_and_ps(v, abs_mask)))
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::erf_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_gelu_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::erf_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_gelu_bias_avx2(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    unsafe { avx2::map_bias_ps(input, bias, width, output, |v| avx2::tanh_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_gelu_bias_avx2(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    unsafe { avx2::map_bias_ps(input, bias, width, output, |v| avx2::erf_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_gelu_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::tanh_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quick_gelu_avx2(input: &[f32], output: &mut [f32], alpha: f32) {
    unsafe {
        let a = core::arch::x86_64::_mm256_set1_ps(alpha);
        avx2::map_ps(input, output, |v| avx2::quick_gelu_ps(v, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- f64 references, rounded once to f32 --------------------------------

    fn tanh_ref(x: f32) -> f32 {
        f64::from(x).tanh() as f32
    }

    fn sigmoid_ref(x: f32) -> f32 {
        let x = f64::from(x);
        if x >= 0.0 {
            (1.0 / (1.0 + (-x).exp())) as f32
        } else {
            let e = x.exp();
            (e / (1.0 + e)) as f32
        }
    }

    fn tanh_gelu_ref(x: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        let inner = f64::from(GELU_B) * (xf + f64::from(GELU_C) * xf * xf * xf);
        (0.5 * xf * (1.0 + inner.tanh())) as f32
    }

    fn quick_gelu_ref(x: f32, alpha: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        let z = f64::from(alpha) * xf;
        let s = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        };
        (xf * s) as f32
    }

    fn erf_ref(x: f32) -> f32 {
        if x.is_nan() {
            return f32::NAN;
        }
        libm::erf(f64::from(x)) as f32
    }

    fn erf_gelu_ref(x: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        (0.5 * xf * (1.0 + libm::erf(xf * std::f64::consts::FRAC_1_SQRT_2))) as f32
    }

    /// Error normalised by the documented contract: absolute error scaled by
    /// `max(1, |x|)`. `tanh`/`sigmoid` are bounded in magnitude so the scale is
    /// 1; the GELU forms multiply by `x`, so their error scales with `|x|`.
    fn scaled_err(got: f32, want: f32, x: f32) -> f64 {
        if (got.is_nan() && want.is_nan()) || got == want {
            return 0.0;
        }
        (f64::from(got) - f64::from(want)).abs() / f64::from(x).abs().max(1.0)
    }

    fn grid(lo: f32, hi: f32, n: usize, extra: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| lo + (hi - lo) * (i as f32) / (n as f32 - 1.0))
            .collect();
        v.extend_from_slice(extra);
        v
    }

    fn check(
        values: &[f32],
        got: &[f32],
        reference: impl Fn(f32) -> f32,
        bound: f64,
        name: &str,
    ) -> f64 {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        for (&x, &g) in values.iter().zip(got) {
            let e = scaled_err(g, reference(x), x);
            if e > worst {
                worst = e;
                worst_at = x;
            }
        }
        assert!(
            worst <= bound,
            "{name}: worst scaled error {worst:e} at x={worst_at:e} exceeds {bound:e}"
        );
        worst
    }

    // ---- accuracy sweeps ----------------------------------------------------

    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`.
    const TANH_BOUND: f64 = 4e-7;
    /// Documented bound: `|err| <= 2e-7` (output is in `[0, 1]`).
    const SIGMOID_BOUND: f64 = 2e-7;
    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`.
    const GELU_BOUND: f64 = 4e-7;

    #[test]
    fn tanh_dense_sweep_matches_f64_reference() {
        let extra = [
            -9.0,
            9.0,
            (-9.0f32).next_down(),
            9.0f32.next_up(),
            (-9.0f32).next_up(),
            9.0f32.next_down(),
            1e-3,
            -1e-3,
        ];
        let x = grid(-14.0, 14.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut out);
        let worst = check(&x, &out, tanh_ref, TANH_BOUND, "tanh");
        eprintln!("tanh worst scaled error: {worst:e}");
    }

    #[test]
    fn sigmoid_dense_sweep_matches_f64_reference() {
        let extra = [
            -18.0,
            18.0,
            (-18.0f32).next_down(),
            18.0f32.next_up(),
            (-18.0f32).next_up(),
            18.0f32.next_down(),
        ];
        let x = grid(-26.0, 26.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        sigmoid_f32_slice(&x, &mut out);
        let worst = check(&x, &out, sigmoid_ref, SIGMOID_BOUND, "sigmoid");
        eprintln!("sigmoid worst scaled error: {worst:e}");
    }

    #[test]
    fn tanh_gelu_dense_sweep_matches_f64_reference() {
        let x = grid(-25.0, 25.0, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        tanh_gelu_f32_slice(&x, &mut out);
        let worst = check(&x, &out, tanh_gelu_ref, GELU_BOUND, "tanh_gelu");
        eprintln!("tanh_gelu worst scaled error: {worst:e}");
    }

    #[test]
    fn quick_gelu_dense_sweep_matches_f64_reference() {
        for alpha in [1.0f32, 1.702, 0.5, -1.0, 2.0] {
            let x = grid(-25.0, 25.0, 200_003, &[]);
            let mut out = vec![0.0f32; x.len()];
            quick_gelu_f32_slice(&x, &mut out, alpha);
            // The sigmoid argument is `alpha * x`, so the error scales with
            // `|x|` only after accounting for the extra `alpha` factor.
            let bound = GELU_BOUND * f64::from(alpha.abs()).max(1.0);
            let worst = check(&x, &out, |v| quick_gelu_ref(v, alpha), bound, "quick_gelu");
            eprintln!("quick_gelu(alpha={alpha}) worst scaled error: {worst:e}");
        }
    }

    /// Documented bound: `|err| <= 3e-7` (output is in `[-1, 1]`). MLAS's own
    /// reference calls this polynomial "faithfully rounded", i.e. within one
    /// ulp of the correctly-rounded `f32` result; one ulp just below `1.0` is
    /// `5.96e-8`, so `3e-7` is a ~5x margin that still fails loudly if a
    /// coefficient is mistyped.
    const ERF_BOUND: f64 = 3e-7;
    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`, as for tanh GELU.
    const ERF_GELU_BOUND: f64 = 4e-7;

    #[test]
    fn erf_dense_sweep_matches_f64_reference() {
        // Both sides of the 0.921875 branch split, both sides of the 3.925
        // saturation clamp, and the near-zero region where the small
        // polynomial's leading term dominates.
        let extra = [
            0.921875,
            -0.921875,
            0.921875f32.next_up(),
            0.921875f32.next_down(),
            (-0.921875f32).next_up(),
            (-0.921875f32).next_down(),
            3.925,
            -3.925,
            3.925f32.next_up(),
            3.925f32.next_down(),
            1e-7,
            -1e-7,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        let x = grid(-6.0, 6.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_ref, ERF_BOUND, "erf");
        eprintln!("erf worst scaled error: {worst:e}");
    }

    /// The interesting band is `|x| <= 1`, where `erf` is steep and the small
    /// polynomial runs; sample it far more densely than the wide sweep can.
    #[test]
    fn erf_dense_sweep_near_origin() {
        let x = grid(-1.5, 1.5, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_ref, ERF_BOUND, "erf near origin");
        eprintln!("erf near-origin worst scaled error: {worst:e}");
    }

    #[test]
    fn erf_gelu_dense_sweep_matches_f64_reference() {
        let x = grid(-25.0, 25.0, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        erf_gelu_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_gelu_ref, ERF_GELU_BOUND, "erf_gelu");
        eprintln!("erf_gelu worst scaled error: {worst:e}");
    }

    /// `erf` saturates to exactly `±1` in `f32` well before the `3.925` clamp,
    /// so the clamp must not be observable: every input past it has to return
    /// exactly `±1.0`, not `1.0 - epsilon`.
    #[test]
    fn erf_saturates_to_exactly_one_past_the_clamp() {
        let mut x: Vec<f32> = vec![0.0; PAD];
        x.extend([
            3.925,
            3.93,
            4.0,
            5.0,
            10.0,
            1e10,
            f32::MAX,
            f32::INFINITY,
            -3.925,
            -3.93,
            -4.0,
            -5.0,
            -10.0,
            -1e10,
            f32::MIN,
            f32::NEG_INFINITY,
        ]);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        for (&v, &r) in x[PAD..].iter().zip(&out[PAD..]) {
            let want = if v > 0.0 { 1.0f32 } else { -1.0f32 };
            assert_eq!(
                r, want,
                "erf({v}) = {r}, expected exactly {want} (saturation is not exact)"
            );
        }
    }

    /// The scalar tail and the 8-lane body must agree, or a tensor's results
    /// would depend on its length modulo 8. `map_ps` routes the tail through
    /// the same kernel via a masked load/store, so this asserts bit equality
    /// rather than a tolerance.
    #[test]
    fn erf_tail_lanes_match_the_vector_body() {
        if !vector_path_available() {
            return;
        }
        let full: Vec<f32> = (0..SIMD_MIN_LEN + 8)
            .map(|i| -4.0 + 8.0 * (i as f32) / 39.0)
            .collect();
        let mut want = vec![0.0f32; full.len()];
        erf_f32_slice(&full, &mut want);
        for len in SIMD_MIN_LEN..full.len() {
            let mut got = vec![0.0f32; len];
            erf_f32_slice(&full[..len], &mut got);
            assert_eq!(got, want[..len], "erf length {len} disagrees with the body");
        }
    }

    // ---- special values -----------------------------------------------------

    /// Index of the first special value; the preceding lanes exist purely to
    /// push the slice over `SIMD_MIN_LEN` so the vector path is taken.
    const PAD: usize = SIMD_MIN_LEN;

    fn special_inputs() -> Vec<f32> {
        std::iter::repeat_n(0.0f32, PAD)
            .chain([
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                f32::INFINITY,
                f32::NAN,
                -f32::MAX,
                f32::MAX,
                -1e30,
                1e30,
            ])
            .collect()
    }

    /// `sqrt` is not an approximation: the AVX2 body, the scalar tail and a
    /// plain `f32::sqrt` must agree **bit for bit** on every input, including
    /// the ones where a "fast" reciprocal-sqrt implementation would not. Every
    /// length in `0..=64` is covered so the 8-wide body and the masked tail are
    /// both exercised at every offset, and lengths below `SIMD_MIN_LEN` prove
    /// the scalar dispatch arm agrees too. Negative inputs are interleaved so
    /// the NaN-producing lanes run through the vector body as well, not only
    /// through `sqrt_special_values`.
    #[test]
    fn sqrt_is_bit_identical_to_scalar_at_every_length() {
        let base: Vec<f32> = (0..64)
            .map(|i| {
                // A spread that includes exact squares, non-squares, subnormals
                // and values whose sqrt is not representable.
                let v = match i % 8 {
                    0 => i as f32,
                    1 => 1.0 / (i as f32 + 1.0),
                    2 => (i as f32) * 1e-30,
                    3 => (i as f32) * 1e30,
                    4 => f32::from_bits(i as u32 + 1), // subnormals
                    5 => (i * i) as f32,
                    6 => 2.0f32.powi(i - 32),
                    _ => (i as f32) + 0.5,
                };
                // Every third lane is negative, so the vector body has to
                // produce NaN in arbitrary lane positions.
                if i % 3 == 1 { -v } else { v }
            })
            .collect();
        for len in 0..=64usize {
            let x = &base[..len];
            let mut got = vec![0.0f32; len];
            sqrt_f32_slice(x, &mut got);
            for (i, (&g, &v)) in got.iter().zip(x).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    v.sqrt().to_bits(),
                    "sqrt mismatch at len={len} index={i} input={v:e}"
                );
            }
        }
    }

    #[test]
    fn sqrt_special_values() {
        let mut x = special_inputs();
        // `special_inputs` is signed-symmetric, which is what matters for
        // `sqrt`: every negative input must produce NaN.
        x.push(1.0);
        x.push(4.0);
        let mut o = vec![0.0f32; x.len()];
        sqrt_f32_slice(&x, &mut o);
        assert!(o[PAD].is_nan(), "sqrt(-Inf) is NaN");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "sqrt(-0) is -0, not NaN"
        );
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "sqrt(+0) is +0");
        assert_eq!(o[PAD + 3], f32::INFINITY, "sqrt(+Inf)");
        assert!(o[PAD + 4].is_nan(), "sqrt(NaN)");
        assert!(o[PAD + 5].is_nan(), "sqrt(-MAX) is NaN");
        assert_eq!(o[PAD + 6], f32::MAX.sqrt());
        assert!(o[PAD + 7].is_nan(), "sqrt(-1e30) is NaN");
        assert_eq!(o[PAD + 8], 1e30f32.sqrt());
        assert_eq!(o[x.len() - 2], 1.0);
        assert_eq!(o[x.len() - 1], 2.0);
    }

    /// A NaN input must come back as a NaN with its payload and sign intact —
    /// the same contract the transcendental kernels above hold.
    #[test]
    fn sqrt_preserves_nan_payload() {
        let mut x = vec![1.0f32; PAD];
        x.push(f32::from_bits(0x7FC0_1234));
        x.push(f32::from_bits(0xFFC0_1234));
        let mut o = vec![0.0f32; x.len()];
        sqrt_f32_slice(&x, &mut o);
        assert_eq!(o[PAD].to_bits(), 0x7FC0_1234);
        assert_eq!(o[PAD + 1].to_bits(), 0xFFC0_1234);
    }

    /// `y = sqrt(y)` is a legal graph, and ORT hands us the same buffer for
    /// both. `write_mapped`'s disjointness check has to reject the direct-write
    /// arm there, exactly as it does for `Tanh` — this pins that it does for the
    /// new `Sqrt` caller too, and that the narrowing (non-f32 output) arm agrees.
    #[test]
    fn sqrt_write_mapped_agrees_between_direct_and_aliased_outputs() {
        let n = 257; // not a multiple of 8, so the masked tail runs too
        let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.37).collect();

        let mut expected = vec![0.0f32; n];
        sqrt_f32_slice(&src, &mut expected);

        let mut disjoint = Owned::f32(&[n], &vec![0.0f32; n]);
        write_mapped("Sqrt", &mut disjoint.view_mut(), &src, sqrt_f32_slice).unwrap();
        assert_eq!(disjoint.to_f32(), expected);

        let mut aliased = Owned::f32(&[n], &src);
        {
            let mut view = aliased.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32; the slice is only
            // read while `write_mapped` decides which arm to take, which is
            // exactly the aliasing `output_direct_write_eligible` must detect.
            let borrowed: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), n) };
            let borrowed: &[f32] = unsafe { std::mem::transmute(borrowed) };
            write_mapped("Sqrt", &mut view, borrowed, sqrt_f32_slice).unwrap();
        }
        assert_eq!(aliased.to_f32(), expected);

        let mut narrowed = Owned::f16(&[n], &vec![0.0f32; n]);
        write_mapped("Sqrt", &mut narrowed.view_mut(), &src, sqrt_f32_slice).unwrap();
        let got = narrowed.to_u16_bits();
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(*g, half::f16::from_f32(*e).to_bits());
        }
    }

    #[test]
    fn tanh_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], -1.0, "tanh(-Inf)");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "tanh(-0) keeps sign"
        );
        assert_eq!(
            o[PAD + 2].to_bits(),
            0.0f32.to_bits(),
            "tanh(+0) keeps sign"
        );
        assert_eq!(o[PAD + 3], 1.0, "tanh(+Inf)");
        assert!(o[PAD + 4].is_nan(), "tanh(NaN)");
        assert_eq!(o[PAD + 5], -1.0);
        assert_eq!(o[PAD + 6], 1.0);
        assert_eq!(o[PAD + 7], -1.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn sigmoid_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        sigmoid_f32_slice(&x, &mut o);
        assert_eq!(o[PAD].to_bits(), 0.0f32.to_bits(), "sigmoid(-Inf) = +0");
        assert_eq!(o[PAD + 1], 0.5, "sigmoid(-0)");
        assert_eq!(o[PAD + 2], 0.5, "sigmoid(+0)");
        assert_eq!(o[PAD + 3], 1.0, "sigmoid(+Inf)");
        assert!(o[PAD + 4].is_nan(), "sigmoid(NaN)");
        assert_eq!(o[PAD + 5], 0.0);
        assert_eq!(o[PAD + 6], 1.0);
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn gelu_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];

        tanh_gelu_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], 0.0, "tanh_gelu(-Inf) is pinned to the limit 0");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "tanh_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "tanh_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "tanh_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "tanh_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "tanh_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX, "tanh_gelu(MAX)");
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);

        quick_gelu_f32_slice(&x, &mut o, 1.702);
        assert_eq!(o[PAD], 0.0, "quick_gelu(-Inf)");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "quick_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "quick_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "quick_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "quick_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "quick_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX);
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);
    }

    #[test]
    fn erf_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], -1.0, "erf(-Inf)");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "erf(-0) keeps sign"
        );
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "erf(+0) keeps sign");
        assert_eq!(o[PAD + 3], 1.0, "erf(+Inf)");
        assert!(o[PAD + 4].is_nan(), "erf(NaN)");
        assert_eq!(o[PAD + 5], -1.0, "erf(-MAX)");
        assert_eq!(o[PAD + 6], 1.0, "erf(MAX)");
        assert_eq!(o[PAD + 7], -1.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn erf_gelu_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        erf_gelu_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], 0.0, "erf_gelu(-Inf) is pinned to the limit 0");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "erf_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "erf_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "erf_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "erf_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "erf_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX, "erf_gelu(MAX)");
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);
    }

    /// `erf` is odd. The sign is applied by an `or` at the very end of the
    /// kernel rather than being carried through the polynomial, so exact
    /// antisymmetry is a property worth pinning: any lane where it fails means
    /// the sign mask leaked into the arithmetic.
    #[test]
    fn erf_is_exactly_odd() {
        let pos: Vec<f32> = (0..4096)
            .map(|i| 1e-4 + 6.0 * (i as f32) / 4095.0)
            .collect();
        let neg: Vec<f32> = pos.iter().map(|v| -v).collect();
        let mut po = vec![0.0f32; pos.len()];
        let mut no = vec![0.0f32; neg.len()];
        erf_f32_slice(&pos, &mut po);
        erf_f32_slice(&neg, &mut no);
        for ((p, n), x) in po.iter().zip(&no).zip(&pos) {
            assert_eq!(*n, -*p, "erf({x}) and erf(-{x}) are not antisymmetric");
        }
    }

    /// A `NaN` anywhere in a vector must not perturb its neighbours: the
    /// polynomial is branch-free, but `blendv` masks and `min`/`max` operand
    /// order both have to be right for this to hold.
    #[test]
    fn nan_does_not_contaminate_neighbouring_lanes() {
        let mut x: Vec<f32> = (0..SIMD_MIN_LEN + 8)
            .map(|i| (i as f32 - 16.0) * 0.5)
            .collect();
        let clean = x.clone();
        for poison in [3usize, 8, SIMD_MIN_LEN, SIMD_MIN_LEN + 5] {
            x.copy_from_slice(&clean);
            x[poison] = f32::NAN;
            let mut a = vec![0.0f32; x.len()];
            let mut b = vec![0.0f32; x.len()];
            tanh_f32_slice(&x, &mut a);
            tanh_f32_slice(&clean, &mut b);
            for i in 0..x.len() {
                if i == poison {
                    assert!(a[i].is_nan(), "poison lane {i} lost its NaN");
                } else {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "lane {i} perturbed by NaN at {poison}"
                    );
                }
            }
        }
    }

    // ---- lengths, tails, aliasing ------------------------------------------

    /// Every length from 0 up to four vectors past the dispatch threshold, so
    /// the masked tail is exercised at all eight residues and the short
    /// (scalar) path is compared against the same reference.
    #[test]
    fn all_lengths_match_reference() {
        let base: Vec<f32> = (0..SIMD_MIN_LEN + 40)
            .map(|i| (i as f32 - 30.0) * 0.37)
            .collect();
        for len in 0..base.len() {
            let x = &base[..len];
            let mut o = vec![0.0f32; len];

            tanh_f32_slice(x, &mut o);
            check(x, &o, tanh_ref, TANH_BOUND, &format!("tanh len={len}"));
            sigmoid_f32_slice(x, &mut o);
            check(
                x,
                &o,
                sigmoid_ref,
                SIGMOID_BOUND,
                &format!("sigmoid len={len}"),
            );
            tanh_gelu_f32_slice(x, &mut o);
            check(x, &o, tanh_gelu_ref, GELU_BOUND, &format!("gelu len={len}"));
            quick_gelu_f32_slice(x, &mut o, 1.702);
            check(
                x,
                &o,
                |v| quick_gelu_ref(v, 1.702),
                GELU_BOUND * 1.702,
                &format!("quick len={len}"),
            );
        }
    }

    /// The masked tail must not write past the requested length.
    #[test]
    fn masked_tail_does_not_overwrite_neighbours() {
        const GUARD: u32 = 0xDEAD_BEEF;
        for len in SIMD_MIN_LEN..SIMD_MIN_LEN + 16 {
            let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.0).collect();
            let mut out = vec![f32::from_bits(GUARD); len + 16];
            tanh_f32_slice(&x, &mut out[..len]);
            for (i, &v) in out[len..].iter().enumerate() {
                assert_eq!(v.to_bits(), GUARD, "len {len} clobbered slot +{i}");
            }
        }
    }

    /// Misaligned inputs and outputs (the kernels use unaligned loads/stores).
    #[test]
    fn unaligned_slices_match_aligned() {
        let backing: Vec<f32> = (0..SIMD_MIN_LEN + 32)
            .map(|i| (i as f32 - 20.0) * 0.31)
            .collect();
        let n = SIMD_MIN_LEN + 8;
        let mut aligned = vec![0.0f32; n];
        tanh_f32_slice(&backing[..n], &mut aligned);
        for off in 1..8 {
            let mut out = vec![0.0f32; n + off];
            tanh_f32_slice(&backing[..n], &mut out[off..][..n]);
            for i in 0..n {
                assert_eq!(
                    out[off + i].to_bits(),
                    aligned[i].to_bits(),
                    "offset {off} lane {i}"
                );
            }
        }
    }

    // ---- documented deviations ---------------------------------------------

    /// Monotonicity.
    ///
    /// Both functions are non-decreasing in exact arithmetic. In the interior
    /// of their range — where the derivative is large relative to the output
    /// resolution — the vector kernels reproduce that exactly. Near the
    /// asymptotes they do not, for two reasons that are both inherited from
    /// MLAS/Eigen and are present in ORT as well:
    ///
    /// * `sigmoid` is evaluated as `p/q + 0.5`. In the far negative tail that
    ///   sum cancels, so the result is quantised to multiples of
    ///   `ulp(0.5) = 6e-8` and can step backwards by a few of those.
    /// * `tanh`'s `p/q` rounds to exactly `±1` slightly before the `±9`
    ///   saturation point, so a one-ulp backwards step can occur there.
    ///
    /// Both deviations are bounded by the documented absolute error, which is
    /// itself two orders of magnitude inside the ONNX conformance tolerance
    /// (`atol = 1e-5`). This test pins that: strict monotonicity where the
    /// output is informative, bounded regression everywhere else.
    #[test]
    fn monotonicity_within_documented_slack() {
        /// Region where the output is far enough from its asymptotes that
        /// strict monotonicity is required.
        fn informative(v: f32, lo: f32, hi: f32) -> bool {
            let span = hi - lo;
            v > lo + span * 1e-2 && v < hi - span * 1e-2
        }
        fn check_pair(name: &str, x: f32, prev: f32, cur: f32, lo: f32, hi: f32, slack: f64) {
            if informative(prev, lo, hi) {
                assert!(cur >= prev, "{name} not monotone at {x}: {cur} < {prev}");
            } else {
                assert!(
                    f64::from(cur) >= f64::from(prev) - slack,
                    "{name} regressed beyond the documented bound at {x}: {cur} < {prev}"
                );
            }
            assert!(
                (lo..=hi).contains(&cur),
                "{name}({x}) = {cur} escaped [{lo}, {hi}]"
            );
        }

        let x: Vec<f32> = (0..60_001).map(|i| -30.0 + i as f32 * 0.001).collect();
        let mut t = vec![0.0f32; x.len()];
        let mut s = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut t);
        sigmoid_f32_slice(&x, &mut s);
        for i in 1..x.len() {
            check_pair("tanh", x[i], t[i - 1], t[i], -1.0, 1.0, 2.0 * TANH_BOUND);
            check_pair(
                "sigmoid",
                x[i],
                s[i - 1],
                s[i],
                0.0,
                1.0,
                2.0 * SIGMOID_BOUND,
            );
        }
    }

    /// Known deviation: `tanh`'s numerator is `x * poly(x^2)`, and for
    /// subnormal `x` that product underflows, so the result is a signed zero
    /// rather than `x`. The absolute error is bounded by `f32::MIN_POSITIVE`
    /// (1.2e-38) and the sign is preserved. `sigmoid` is unaffected.
    #[test]
    fn subnormal_inputs_underflow_to_signed_zero() {
        let d = f32::from_bits(1);
        let x: Vec<f32> = std::iter::repeat_n(d, PAD)
            .chain([d, -d, f32::MIN_POSITIVE, -f32::MIN_POSITIVE])
            .collect();
        let mut o = vec![0.0f32; x.len()];

        tanh_f32_slice(&x, &mut o);
        for (i, (&got, &inp)) in o.iter().zip(&x).enumerate() {
            assert!(
                (f64::from(got) - f64::from(inp)).abs() <= f64::from(f32::MIN_POSITIVE),
                "lane {i}: tanh({inp:e}) = {got:e} exceeds the documented subnormal bound"
            );
            assert_eq!(
                got.is_sign_negative(),
                inp.is_sign_negative(),
                "lane {i}: tanh({inp:e}) lost the sign"
            );
        }

        sigmoid_f32_slice(&x, &mut o);
        assert!(o.iter().all(|&v| v == 0.5), "sigmoid near zero must be 0.5");
    }

    /// The `< SIMD_MIN_LEN` scalar path and the vector path must agree to
    /// within the sum of their documented bounds, so a caller cannot observe a
    /// discontinuity purely from tensor size.
    #[test]
    fn scalar_and_vector_paths_agree() {
        if !vector_path_available() {
            return;
        }
        let x: Vec<f32> = (0..SIMD_MIN_LEN * 4)
            .map(|i| (i as f32 - 60.0) * 0.21)
            .collect();
        let mut vector = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut vector);
        for (i, (&xi, &v)) in x.iter().zip(&vector).enumerate() {
            let scalar = tanh_scalar(xi);
            let e = scaled_err(v, scalar, xi);
            assert!(
                e <= TANH_BOUND,
                "lane {i}: vector {v} vs scalar {scalar}, err {e:e}"
            );
        }
        sigmoid_f32_slice(&x, &mut vector);
        for (i, (&xi, &v)) in x.iter().zip(&vector).enumerate() {
            let scalar = sigmoid_scalar(xi);
            let e = scaled_err(v, scalar, xi);
            assert!(
                e <= SIGMOID_BOUND,
                "lane {i}: vector {v} vs scalar {scalar}, err {e:e}"
            );
        }
    }

    // ── direct-write plumbing ─────────────────────────────────────────────

    use crate::kernels::testutil::Owned;

    /// `write_mapped` must produce the same bytes whether it took the
    /// direct-write arm or the owned-scratch arm. The interesting case is an
    /// in-place node (`y = tanh(y)`), where the widened input borrows the
    /// output's own storage: writing through would corrupt the tail of the
    /// input mid-kernel, so `output_direct_write_eligible` has to reject it and
    /// send us to the scratch buffer.
    #[test]
    fn write_mapped_agrees_between_direct_and_aliased_outputs() {
        let n = 257; // not a multiple of 8, so the masked tail runs too
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.11).collect();

        let mut expected = vec![0.0f32; n];
        tanh_f32_slice(&src, &mut expected);

        // Disjoint output: takes the direct-write arm.
        let mut disjoint = Owned::f32(&[n], &vec![0.0f32; n]);
        write_mapped("Tanh", &mut disjoint.view_mut(), &src, |x, y| {
            tanh_f32_slice(x, y)
        })
        .unwrap();
        assert_eq!(disjoint.to_f32(), expected);

        // Aliased output: the input slice *is* the output storage.
        let mut aliased = Owned::f32(&[n], &src);
        {
            let mut view = aliased.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32; the slice is only
            // read while `write_mapped` decides which arm to take, which is
            // exactly the aliasing `output_direct_write_eligible` must detect.
            let borrowed: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), n) };
            let borrowed: &[f32] = unsafe { std::mem::transmute(borrowed) };
            write_mapped("Tanh", &mut view, borrowed, tanh_f32_slice).unwrap();
        }
        assert_eq!(aliased.to_f32(), expected);

        // A non-f32 output can never take the direct arm; it must still narrow
        // correctly.
        let mut narrowed = Owned::f16(&[n], &vec![0.0f32; n]);
        write_mapped("Tanh", &mut narrowed.view_mut(), &src, |x, y| {
            tanh_f32_slice(x, y)
        })
        .unwrap();
        let got = narrowed.to_u16_bits();
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(*g, half::f16::from_f32(*e).to_bits());
        }
    }

    /// Fusing FastGelu's bias into the vector kernel must be bit-identical to
    /// materialising `x + bias` and mapping over it — that equivalence is the
    /// whole justification for the fused path.
    #[test]
    fn bias_fusion_is_bit_identical_to_folding_first() {
        for width in [1usize, 7, 8, 31, 32, 33, 64, 129] {
            let rows = 5;
            let n = rows * width;
            let x: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 7.0).collect();
            let bias: Vec<f32> = (0..width).map(|i| (i as f32).cos() * 3.0).collect();

            let mut folded_in = vec![0.0f32; n];
            for (row_in, row_out) in x.chunks(width).zip(folded_in.chunks_mut(width)) {
                for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(&bias) {
                    *o = v + b;
                }
            }
            let mut want = vec![0.0f32; n];
            tanh_gelu_f32_slice(&folded_in, &mut want);

            let mut got = vec![0.0f32; n];
            tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "width={width} i={i}: fused {g} != folded {w}"
                );
            }
        }
    }

    /// Special values must survive the fused bias add: `+inf + finite` stays
    /// `+inf`, `-inf` maps to `0`, and a NaN bias poisons the row.
    #[test]
    fn bias_fusion_handles_special_values() {
        let width = 40;
        let mut x = vec![1.0f32; width * 2];
        x[0] = f32::INFINITY;
        x[1] = f32::NEG_INFINITY;
        x[2] = f32::NAN;
        let mut bias = vec![0.5f32; width];
        bias[3] = f32::NAN;
        bias[4] = f32::INFINITY;

        let mut got = vec![0.0f32; x.len()];
        tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

        assert_eq!(got[0], f32::INFINITY);
        assert_eq!(got[1], 0.0);
        assert!(got[2].is_nan());
        assert!(got[3].is_nan());
        assert_eq!(got[4], f32::INFINITY);
    }

    /// A bias that aliases the output must force the scratch arm. Writing the
    /// first row through would otherwise corrupt the bias that every later row
    /// still needs, so this is a data-corruption regression test.
    #[test]
    fn write_mapped_reading_rejects_an_aliasing_extra_read() {
        let width = 40;
        let n = width * 4;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 8.0).collect();
        let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let mut want = vec![0.0f32; n];
        tanh_gelu_bias_f32_slice(&x, &bias, width, &mut want);

        // Place the bias inside the output buffer itself.
        let mut seeded = vec![0.0f32; n];
        seeded[..width].copy_from_slice(&bias);
        let mut out = Owned::f32(&[n], &seeded);
        {
            let mut view = out.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32, the first `width` of
            // which hold the bias. This is exactly the overlap that the extra
            // read range must make `write_mapped_reading` detect.
            let aliased: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), width) };
            let aliased: &[f32] = unsafe { std::mem::transmute(aliased) };
            write_mapped_reading(
                "FastGelu",
                &mut view,
                &x,
                &[crate::dtype::slice_byte_range(aliased)],
                |x, y| tanh_gelu_bias_f32_slice(x, aliased, width, y),
            )
            .unwrap();
        }
        assert_eq!(out.to_f32(), want);
    }

    /// A tensor whose length is not a whole number of bias rows must still have
    /// every element written, with the bias broadcast as `bias[i % width]`.
    #[test]
    fn bias_fusion_writes_a_trailing_partial_row() {
        let width = 48;
        for n in [1usize, width + 1, width * 2 + 7, width * 3 - 1] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 5.0).collect();
            let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.02 - 0.4).collect();

            let mut want = vec![f32::NAN; n];
            for (row_in, row_out) in x.chunks(width).zip(want.chunks_mut(width)) {
                let folded: Vec<f32> = row_in.iter().zip(&bias).map(|(v, b)| v + b).collect();
                tanh_gelu_f32_slice(&folded, row_out);
            }

            let mut got = vec![f32::NAN; n];
            tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

            for (i, g) in got.iter().enumerate() {
                assert!(!g.is_nan(), "n={n} i={i}: element never written");
            }
            // A short final row can land on the other side of the length
            // threshold from the reference, so bound rather than bit-compare.
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let scale = x[i].abs().max(1.0);
                assert!(
                    f64::from((g - w).abs()) / f64::from(scale) <= GELU_BOUND,
                    "n={n} i={i}: {g} vs {w}"
                );
            }
        }
    }
}

#[cfg(test)]
mod exact_tests {
    use super::*;

    /// Every value a `f32` kernel can be asked about that is interesting to a
    /// sign-bit mask, a division, or a rounding mode.
    fn adversarial() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_1234),
            8_388_608.0,
            -8_388_608.0,
            16_777_216.0,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let f = f32::from_bits(state);
            if f.is_finite() {
                v.push(f);
            }
            v.push((state as i32 as f32) / 1024.0);
        }
        v
    }

    /// Bit-compare a vector kernel against its scalar reference over the
    /// adversarial set, at every length from 0 to `4 * SIMD_MIN_LEN` so the
    /// masked tail, the scalar-fallback seam and the aligned body are all hit.
    fn assert_bit_exact(
        name: &str,
        vector: impl Fn(&[f32], &mut [f32]),
        scalar: impl Fn(f32) -> f32,
    ) {
        let values = adversarial();
        for len in 0..=(4 * SIMD_MIN_LEN) {
            for start in [0usize, 1, 7, 13] {
                if start + len > values.len() {
                    continue;
                }
                let x = &values[start..start + len];
                let mut got = vec![0.0f32; len];
                vector(x, &mut got);
                for (i, (&g, &v)) in got.iter().zip(x).enumerate() {
                    let want = scalar(v);
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{name}: len={len} start={start} i={i} x={v:e} ({:#010x}): \
                         got {g:e} ({:#010x}), want {want:e} ({:#010x})",
                        v.to_bits(),
                        g.to_bits(),
                        want.to_bits()
                    );
                }
            }
        }
        // And once over the whole set, which is far longer than any threshold.
        let mut got = vec![0.0f32; values.len()];
        vector(&values, &mut got);
        for (i, (&g, &v)) in got.iter().zip(&values).enumerate() {
            assert_eq!(
                g.to_bits(),
                scalar(v).to_bits(),
                "{name}: full sweep i={i} x={v:e}"
            );
        }
    }

    #[test]
    fn neg_is_bit_exact() {
        assert_bit_exact("neg", neg_f32_slice, |v| -v);
    }

    #[test]
    fn abs_is_bit_exact() {
        assert_bit_exact("abs", abs_f32_slice, f32::abs);
    }

    #[test]
    fn reciprocal_is_bit_exact() {
        assert_bit_exact("reciprocal", reciprocal_f32_slice, |v| 1.0 / v);
    }

    #[test]
    fn floor_is_bit_exact() {
        assert_bit_exact("floor", floor_f32_slice, f32::floor);
    }

    #[test]
    fn ceil_is_bit_exact() {
        assert_bit_exact("ceil", ceil_f32_slice, f32::ceil);
    }

    #[test]
    fn round_is_bit_exact_and_ties_to_even() {
        assert_bit_exact(
            "round_ties_even",
            round_ties_even_f32_slice,
            f32::round_ties_even,
        );
        // The property that separates ONNX `Round` from `f32::round`: halves
        // go to the even neighbour, not away from zero. Long enough to be on
        // the vector path.
        let x: Vec<f32> = std::iter::repeat_n([-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5], 16)
            .flatten()
            .collect();
        let mut got = vec![0.0f32; x.len()];
        round_ties_even_f32_slice(&x, &mut got);
        for chunk in got.chunks(6) {
            assert_eq!(chunk, &[-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]);
        }
    }

    #[test]
    fn sign_is_bit_exact_and_keeps_nan_payloads() {
        assert_bit_exact("sign", sign_f32_slice, sign_scalar);
        // Explicit: a NaN lane must come back as the *same* NaN, and both
        // zeros must come back as `+0.0`, on the vector path.
        let nan = f32::from_bits(0x7fc0_1234);
        let neg_nan = f32::from_bits(0xffc0_1234);
        let x: Vec<f32> = std::iter::repeat_n([nan, neg_nan, -0.0, 3.0, -3.0], 16)
            .flatten()
            .collect();
        let mut got = vec![0.0f32; x.len()];
        sign_f32_slice(&x, &mut got);
        for chunk in got.chunks(5) {
            // Matches ORT 1.28.0: the NaN comes back with its payload *and*
            // sign bit intact, not canonicalised.
            assert_eq!(chunk[0].to_bits(), nan.to_bits());
            assert_eq!(chunk[1].to_bits(), neg_nan.to_bits());
            assert_eq!(chunk[2].to_bits(), 0.0f32.to_bits());
            assert_eq!(chunk[3], 1.0);
            assert_eq!(chunk[4], -1.0);
        }
    }

    #[test]
    fn softsign_is_bit_exact() {
        assert_bit_exact("softsign", softsign_f32_slice, softsign_scalar);
    }

    /// The exact group must not have the `SIMD_MIN_LEN` accuracy seam the
    /// approximations do: a value's result cannot depend on how many elements
    /// share the tensor with it.
    /// Signalling NaN is the one input class where the scalar and AVX2 paths
    /// are **not** bit-identical, so it is deliberately excluded from
    /// [`adversarial`] and pinned here instead.
    ///
    /// The divergence is structural: `_mm256_round_ps`, `_mm256_div_ps` and
    /// the FMA in `softsign` are IEEE arithmetic and quiet a signalling NaN,
    /// while Rust's `f32::ceil`/`floor`/`round_ties_even` return the operand
    /// untouched. Neither is wrong -- ONNX does not specify NaN payload
    /// propagation -- and ORT is itself inconsistent across exactly the same
    /// split. Measured against ORT 1.28.0 CPU with input `0x7f801234`:
    ///
    /// | op | ORT | our scalar | our AVX2 |
    /// |----|-----|------------|----------|
    /// | `Ceil`       | quiets   | preserves | quiets    |
    /// | `Floor`      | quiets   | preserves | quiets    |
    /// | `Reciprocal` | quiets   | quiets    | quiets    |
    /// | `Softsign`   | quiets   | quiets    | quiets    |
    /// | `Round`      | preserves| preserves | quiets    |
    /// | `Sign`       | preserves| preserves | preserves |
    /// | `Neg`        | preserves| preserves | preserves |
    /// | `Abs`        | preserves| preserves | preserves |
    ///
    /// So on the vector path this EP matches ORT everywhere except `Round`,
    /// and on the scalar path it matches everywhere except `Ceil`/`Floor`.
    /// Closing the gap would need a per-element payload check that would cost
    /// the entire measured win, for an input class that does not occur in
    /// inference. This test exists so the behaviour cannot drift unnoticed.
    #[test]
    fn signalling_nan_behaviour_is_pinned() {
        const SNAN: u32 = 0x7f80_1234;
        const QUIETED: u32 = 0x7fc0_1234;
        let input = vec![f32::from_bits(SNAN); SIMD_MIN_LEN * 2];

        // Sign-bit-only kernels never touch the payload, on either path.
        for (name, kernel) in [
            ("sign", sign_f32_slice as fn(&[f32], &mut [f32])),
            ("neg", neg_f32_slice),
            ("abs", abs_f32_slice),
        ] {
            let mut out = vec![0.0; input.len()];
            kernel(&input, &mut out);
            assert_eq!(
                out[0].to_bits() & 0x7fff_ffff,
                SNAN,
                "{name} must leave a signalling NaN payload untouched",
            );
        }

        // Arithmetic kernels quiet it. Asserting this pins the behaviour that
        // `adversarial` cannot cover.
        for (name, kernel) in [
            ("ceil", ceil_f32_slice as fn(&[f32], &mut [f32])),
            ("floor", floor_f32_slice),
            ("round_ties_even", round_ties_even_f32_slice),
            ("reciprocal", reciprocal_f32_slice),
            ("softsign", softsign_f32_slice),
        ] {
            let mut out = vec![0.0; input.len()];
            kernel(&input, &mut out);
            assert!(
                out[0].is_nan(),
                "{name} of a signalling NaN must still be NaN",
            );
            if vector_path_available() {
                assert_eq!(
                    out[0].to_bits(),
                    QUIETED,
                    "{name} on the vector path is expected to quiet a signalling NaN",
                );
            }
        }
    }

    #[test]
    fn exact_kernels_have_no_length_seam() {
        let values = adversarial();
        type ExactKernel = fn(&[f32], &mut [f32]);
        let kernels: [(&str, ExactKernel); 8] = [
            ("neg", neg_f32_slice),
            ("abs", abs_f32_slice),
            ("reciprocal", reciprocal_f32_slice),
            ("floor", floor_f32_slice),
            ("ceil", ceil_f32_slice),
            ("round", round_ties_even_f32_slice),
            ("sign", sign_f32_slice),
            ("softsign", softsign_f32_slice),
        ];
        for (name, k) in kernels {
            // Below the threshold (scalar) and far above it (vector).
            let short = &values[..SIMD_MIN_LEN - 1];
            let mut got_short = vec![0.0f32; short.len()];
            k(short, &mut got_short);

            let long = &values[..SIMD_MIN_LEN * 8];
            let mut got_long = vec![0.0f32; long.len()];
            k(long, &mut got_long);

            for i in 0..short.len() {
                assert_eq!(
                    got_short[i].to_bits(),
                    got_long[i].to_bits(),
                    "{name}: element {i} (x={:e}) differs between a {}-element and a \
                     {}-element tensor — the exact group must have no length seam",
                    short[i],
                    short.len(),
                    long.len()
                );
            }
        }
    }
}

/// `Exp`'s vector path is a different approximation from `f32::exp`, so its
/// tests are error-bounded and special-value-pinned rather than bit-exact.
#[cfg(test)]
mod exp_tests {
    use super::*;

    /// Long enough to reach the vector path on every host that has one.
    const N: usize = 4096;

    fn vector_exp(values: &[f32]) -> Vec<f32> {
        assert!(
            values.len() >= SIMD_MIN_LEN,
            "input must be long enough to reach the vector path"
        );
        let mut out = vec![0.0f32; values.len()];
        exp_f32_slice(values, &mut out);
        out
    }

    /// Error against an `f64` reference over a dense sweep of the whole
    /// argument range, measured in ulp rather than relative error.
    ///
    /// Relative error is the wrong metric near the bottom of the range: below
    /// about `-87` the result is subnormal and carries only a handful of
    /// significand bits, so a *correctly rounded* answer can already be tens of
    /// percent away from the real value. Comparing representations counts what
    /// actually matters — how many representable `f32` steps separate us from
    /// the best possible answer — and stays meaningful through the subnormal
    /// range and at the overflow boundary.
    #[test]
    fn dense_sweep_stays_within_two_ulp_of_a_f64_reference() {
        let mut x = Vec::with_capacity(1 << 16);
        let (lo, hi) = (-110.0f64, 89.0f64);
        for i in 0..(1 << 16) {
            x.push((lo + (hi - lo) * (i as f64) / ((1 << 16) as f64 - 1.0)) as f32);
        }
        let got = vector_exp(&x);

        let mut worst = 0i64;
        let mut worst_at = 0.0f32;
        for (&v, &g) in x.iter().zip(&got) {
            let want = f64::from(v).exp() as f32;
            // Both are non-negative, so the bit patterns are monotonic in value
            // and their difference is exactly the number of representable steps
            // between them — including across the subnormal/normal boundary.
            let ulp = i64::from(g.to_bits()) - i64::from(want.to_bits());
            if ulp.abs() > worst {
                worst = ulp.abs();
                worst_at = v;
            }
        }
        assert!(
            worst <= 2,
            "worst error {worst} ulp at x={worst_at} (exp = {})",
            f64::from(worst_at).exp()
        );
    }

    /// The seam between the vector path and the correctly-rounded scalar
    /// fallback is allowed to move a result by a bounded amount, never more.
    #[test]
    fn vector_and_scalar_paths_agree_to_two_ulp() {
        let mut x = Vec::with_capacity(N);
        let mut state = 0x1234_5678u32;
        for _ in 0..N {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x.push((state >> 8) as f32 / (1 << 24) as f32 * 60.0 - 30.0);
        }
        let vector = vector_exp(&x);
        for (&v, &g) in x.iter().zip(&vector) {
            let want = v.exp();
            let rel = ((f64::from(g) - f64::from(want)) / f64::from(want)).abs();
            assert!(
                rel <= 2.0 * f64::from(f32::EPSILON),
                "exp({v}): vector {g} vs scalar {want} (rel {rel:e})"
            );
        }
    }

    /// The reconstruction reads the biased exponent as an integer, so every
    /// saturating and non-finite argument has to be pinned explicitly. These
    /// answers were read off ORT 1.28.0's own CPU `Exp` kernel.
    #[test]
    fn special_values_match_ort() {
        let cases: [(f32, f32); 12] = [
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, 0.0),
            (100.0, f32::INFINITY),
            (89.0, f32::INFINITY),
            (88.7762626647950, f32::INFINITY),
            (88.3762626647949, 2.4061436e38),
            (-87.0, 1.6458115e-38),
            (-200.0, 0.0),
            (0.0, 1.0),
            (-0.0, 1.0),
            (1.0, std::f32::consts::E),
            (-1.0, 0.36787945),
        ];
        let mut x = vec![0.0f32; N];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = cases[i % cases.len()].0;
        }
        let got = vector_exp(&x);
        for (i, &g) in got.iter().enumerate() {
            let (arg, want) = cases[i % cases.len()];
            if want.is_infinite() || want == 0.0 || want == 1.0 {
                assert_eq!(g, want, "exp({arg}) = {g}, expected {want}");
            } else {
                let rel = ((f64::from(g) - f64::from(want)) / f64::from(want)).abs();
                assert!(
                    rel <= 2.0 * f64::from(f32::EPSILON),
                    "exp({arg}) = {g}, expected {want}"
                );
            }
        }
    }

    /// `NaN` survives the exponent reconstruction only because of the clamp's
    /// operand order — see [`avx2::exp_full_ps`]. Nothing in the arithmetic
    /// makes that obvious, so it gets its own test, across both signs and both
    /// quiet and signalling payloads.
    #[test]
    fn nan_propagates_instead_of_becoming_finite() {
        let payloads = [
            f32::NAN.to_bits(),
            0x7FC0_1234,
            0xFFC0_1234,
            0x7F80_0001, // signalling
        ];
        let mut x = vec![0.0f32; N];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = f32::from_bits(payloads[i % payloads.len()]);
        }
        let got = vector_exp(&x);
        for (i, &g) in got.iter().enumerate() {
            assert!(
                g.is_nan(),
                "exp(NaN 0x{:08X}) at {i} produced {g}",
                payloads[i % payloads.len()]
            );
        }
    }

    /// The tail is processed through the same kernel via a masked load, so a
    /// non-multiple-of-8 length must not change any answer.
    #[test]
    fn tail_lengths_are_computed_identically() {
        let base: Vec<f32> = (0..(SIMD_MIN_LEN + 8))
            .map(|i| (i as f32) * 0.37 - 6.0)
            .collect();
        let full = vector_exp(&base);
        for n in SIMD_MIN_LEN..base.len() {
            let mut out = vec![0.0f32; n];
            exp_f32_slice(&base[..n], &mut out);
            for (i, (&g, &f)) in out.iter().zip(&full).enumerate() {
                assert_eq!(g.to_bits(), f.to_bits(), "n={n} i={i}");
            }
        }
    }

    /// `y = exp(y)` is a legal graph; the kernel must tolerate one buffer.
    #[test]
    fn aliased_input_and_output_are_supported_by_the_slice_form() {
        let mut buf: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let want: Vec<f32> = {
            let mut o = vec![0.0f32; N];
            exp_f32_slice(&buf, &mut o);
            o
        };
        let copy = buf.clone();
        exp_f32_slice(&copy, &mut buf);
        for (i, (&g, &w)) in buf.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "i={i}");
        }
    }
}

/// Same-binary A/B of the MLAS route against the pure-Rust SIMD route.
///
/// Lives inside the crate because the slice kernels are `pub(crate)`, and
/// in-binary because `benches/activation_bench.rs` documents a uniform
/// cross-build offset on byte-identical kernels. Both sides here are compiled
/// once, into one binary, and interleaved.
///
/// `cargo test -p onnx-runtime-ep-cpu --features mlas --release --lib \
///   mlas_ab -- --ignored --nocapture`
#[cfg(all(test, feature = "mlas"))]
mod mlas_ab {
    use super::*;
    use std::time::Instant;

    /// Time two routes **interleaved**, one iteration each per round, so that
    /// any drift in machine state over the run hits both equally. Timing all of
    /// A and then all of B would let a noisy neighbour land on one side only.
    fn best_ns_pair(n: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
        for _ in 0..3 {
            a();
            b();
        }
        let (mut best_a, mut best_b) = (f64::MAX, f64::MAX);
        for _ in 0..15 {
            let t = Instant::now();
            a();
            best_a = best_a.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            b();
            best_b = best_b.min(t.elapsed().as_secs_f64());
        }
        let scale = 1e9 / n as f64;
        (best_a * scale, best_b * scale)
    }

    fn max_ulp(a: &[f32], b: &[f32]) -> (f64, u32) {
        let mut maxabs = 0.0f64;
        let mut maxulp = 0u32;
        for (x, y) in a.iter().zip(b) {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            maxabs = maxabs.max((f64::from(*x) - f64::from(*y)).abs());
            let u = (i64::from(x.to_bits()) - i64::from(y.to_bits())).unsigned_abs() as u32;
            maxulp = maxulp.max(u);
        }
        (maxabs, maxulp)
    }

    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn mlas_vs_rust_simd() {
        println!("op\tn\tours_ns\tmlas_ns\tspeedup\tmaxabs\tmaxulp");
        for n in [1usize << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i as f32 % 2003.0) - 1000.0) / 128.0)
                .collect();
            let mut ours = vec![0.0f32; n];
            let mut mlas = vec![0.0f32; n];

            #[allow(clippy::type_complexity)]
            let cases: [(&str, fn(&[f32], &mut [f32]), fn(&[f32], &mut [f32])); 4] = [
                ("Tanh", tanh_avx2_shim, tanh_mlas_ref),
                ("Sigmoid", sigmoid_avx2_shim, sigmoid_mlas_ref),
                ("Erf", erf_avx2_shim, mlas_sys::compute_erf),
                ("GeluErf", erf_gelu_avx2_shim, erf_gelu_mlas),
            ];
            for (name, rust, mlas_fn) in cases {
                let (a, b) = best_ns_pair(n, || rust(&x, &mut ours), || mlas_fn(&x, &mut mlas));
                let (mabs, mulp) = max_ulp(&ours, &mlas);
                println!(
                    "{name}\t{n}\t{a:.4}\t{b:.4}\t{:.2}x\t{mabs:.3e}\t{mulp}",
                    a / b
                );
            }
        }
    }

    /// The largest absolute disagreement tolerated between the two routes on a
    /// target where they are *not* the same polynomial.
    ///
    /// On x86-64 the pure-Rust route was written to mirror MLAS's AVX2 kernel
    /// instruction for instruction, so the two are bit-identical and this
    /// constant is unused. On aarch64 MLAS dispatches its own NEON kernels,
    /// which are a different approximation of the same functions; the dense
    /// sweep in `mlas_vs_rust_dense_ulp_sweep` measures the worst
    /// disagreement over the whole domain at 4.8e-7 (Tanh 3.0e-7, Sigmoid
    /// 1.8e-7, Erf 6.0e-8, GeluErf 4.8e-7), all of it last-bit noise in f32.
    ///
    /// Note that a ULP bound is the wrong instrument here: these functions
    /// cross zero, and a bit-difference across the sign boundary reads as
    /// ~8.7e8 ULP for an absolute difference of 1e-7.
    const CROSS_ISA_ABS_TOL: f32 = 1e-6;

    /// MLAS must agree with the pure-Rust route on every special value, not
    /// just on the dense interior sweep. A difference here would be a silent
    /// behaviour change for anyone who builds without the feature.
    ///
    /// The *semantics* — NaN, the infinities, and signed zero — must match
    /// exactly everywhere, the one architectural exception being AdvSIMD's
    /// flush-to-zero on subnormals. Bit-identity of finite results is only
    /// required on x86-64, where both routes are the same polynomial; see
    /// [`CROSS_ISA_ABS_TOL`].
    #[test]
    fn mlas_matches_rust_simd_on_special_values() {
        let specials = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            1.0,
            -1.0,
            88.5,
            -88.5,
            1e-30,
            -1e-30,
        ];
        // Pad past SIMD_MIN_LEN so both routes take their vector path, and
        // repeat the pattern so the values land at every lane offset.
        let mut x = Vec::new();
        while x.len() < SIMD_MIN_LEN * 4 {
            x.extend_from_slice(&specials);
        }
        let n = x.len();
        let mut ours = vec![0.0f32; n];
        let mut mlas = vec![0.0f32; n];

        #[allow(clippy::type_complexity)]
        let cases: [(&str, fn(&[f32], &mut [f32]), fn(&[f32], &mut [f32])); 4] = [
            ("Tanh", tanh_avx2_shim, tanh_mlas_ref),
            ("Sigmoid", sigmoid_avx2_shim, sigmoid_mlas_ref),
            ("Erf", erf_avx2_shim, mlas_sys::compute_erf),
            ("GeluErf", erf_gelu_avx2_shim, erf_gelu_mlas),
        ];
        for (name, rust, mlas_fn) in cases {
            rust(&x, &mut ours);
            mlas_fn(&x, &mut mlas);
            for (i, ((a, b), input)) in ours.iter().zip(&mlas).zip(&x).enumerate() {
                let where_ = format!(
                    "{name}: the MLAS route and the pure-Rust SIMD route \
                     disagree at index {i} for input {input:e} \
                     (rust={a:e} mlas={b:e})"
                );
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                // Same polynomial on x86-64: nothing less than bit-identity
                // would be a real result there.
                if cfg!(target_arch = "x86_64") {
                    assert_eq!(a.to_bits(), b.to_bits(), "{where_}");
                    continue;
                }
                assert_eq!(a.is_nan(), b.is_nan(), "{where_} (NaN-ness differs)");
                assert_eq!(
                    a.is_infinite(),
                    b.is_infinite(),
                    "{where_} (infinite-ness differs)"
                );
                if !a.is_finite() {
                    // The infinities are semantics, not precision.
                    assert_eq!(a.to_bits(), b.to_bits(), "{where_}");
                    continue;
                }
                // AdvSIMD flushes subnormals to zero, so `tanh(1e-45)` is 1e-45
                // on the scalar-tail route and +0 through MLAS's NEON kernel.
                // That is architectural, and 1e-45 is inside the tolerance
                // below anyway; only *normal* zeros carry a sign contract.
                let subnormal = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
                if (*a == 0.0 || *b == 0.0) && !subnormal(*a) && !subnormal(*b) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{where_} (signed zero)");
                    continue;
                }
                assert!(
                    (a - b).abs() <= CROSS_ISA_ABS_TOL,
                    "{where_} by {:e}, over the {CROSS_ISA_ABS_TOL:e} tolerance",
                    (a - b).abs()
                );
            }
        }
    }

    /// Dense ULP sweep of the MLAS route against the pure-Rust route over the
    /// whole interesting domain, printing the worst disagreement per op.
    #[test]
    #[ignore = "diagnostic sweep; run explicitly with --ignored --nocapture"]
    fn mlas_vs_rust_dense_ulp_sweep() {
        let n = 1usize << 20;
        // -20 .. 20 densely: covers both saturation tails and the linear core.
        let x: Vec<f32> = (0..n)
            .map(|i| -20.0 + 40.0 * (i as f32) / (n as f32))
            .collect();
        let mut ours = vec![0.0f32; n];
        let mut mlas = vec![0.0f32; n];

        #[allow(clippy::type_complexity)]
        let cases: [(&str, fn(&[f32], &mut [f32]), fn(&[f32], &mut [f32])); 4] = [
            ("Tanh", tanh_avx2_shim, tanh_mlas_ref),
            ("Sigmoid", sigmoid_avx2_shim, sigmoid_mlas_ref),
            ("Erf", erf_avx2_shim, mlas_sys::compute_erf),
            ("GeluErf", erf_gelu_avx2_shim, erf_gelu_mlas),
        ];
        for (name, rust, mlas_fn) in cases {
            rust(&x, &mut ours);
            mlas_fn(&x, &mut mlas);
            let (mabs, mulp) = max_ulp(&ours, &mlas);
            let over_rust = ours.iter().filter(|v| v.abs() > 1.0).count();
            let over_mlas = mlas.iter().filter(|v| v.abs() > 1.0).count();
            println!(
                "{name}\tmaxabs={mabs:.3e}\tmaxulp={mulp}\t|y|>1: rust={over_rust} mlas={over_mlas}"
            );
        }
    }

    /// What the removed `Tanh` MLAS route did: MLAS's kernel, then the clamp
    /// back into `[-1, 1]` that its non-range-preserving rational requires.
    ///
    /// The route is gone from the dispatcher (the clamp pass costs more than
    /// the polynomial saves — see the note above `dispatch_mlas!`), but it
    /// stays here as the measurement reference `mlas_vs_rust_simd` reports
    /// against, and as the thing the special-value and ULP sweeps pin us to.
    fn tanh_mlas_ref(x: &[f32], y: &mut [f32]) {
        mlas_clamped_ref(x, y, mlas_sys::compute_tanh, -1.0, 1.0);
    }

    /// Ditto for `Sigmoid` and `[0, 1]`.
    fn sigmoid_mlas_ref(x: &[f32], y: &mut [f32]) {
        mlas_clamped_ref(x, y, mlas_sys::compute_logistic, 0.0, 1.0);
    }

    /// Blocked so the clamp re-reads each block while it is still in L1/L2,
    /// exactly as the shipped route used to.
    fn mlas_clamped_ref(
        input: &[f32],
        output: &mut [f32],
        kernel: fn(&[f32], &mut [f32]),
        lo: f32,
        hi: f32,
    ) {
        const BLOCK: usize = 8192;
        for (xs, ys) in input.chunks(BLOCK).zip(output.chunks_mut(BLOCK)) {
            kernel(xs, ys);
            for v in ys.iter_mut() {
                *v = v.clamp(lo, hi);
            }
        }
    }

    // Thin wrappers pinning the pure-Rust route regardless of feature flags.
    fn tanh_avx2_shim(x: &[f32], y: &mut [f32]) {
        dispatch!(x, y, tanh_scalar, tanh_avx2)
    }
    fn sigmoid_avx2_shim(x: &[f32], y: &mut [f32]) {
        dispatch!(x, y, sigmoid_scalar, sigmoid_avx2)
    }
    fn erf_avx2_shim(x: &[f32], y: &mut [f32]) {
        dispatch!(x, y, erf_scalar, erf_avx2)
    }
    fn erf_gelu_avx2_shim(x: &[f32], y: &mut [f32]) {
        dispatch!(x, y, erf_gelu_scalar, erf_gelu_avx2)
    }
}

/// Falsifiers for the saturation removed from `tanh_ps` / `sigmoid_ps`.
///
/// Both kernels used to follow the `[-1, 1]` / `[0, 1]` clamp with a pair of
/// `vcmpps` + `vblendvps` that forced the saturated constant for `|x|` beyond
/// the rational's clamp range. That step is redundant: the input clamp means
/// the rational never sees an argument past `±9` / `±18`, and at those points
/// the result is already outside the output clamp, so the clamp alone
/// saturates. These tests keep the old sequence as a reference and assert the
/// current kernels reproduce it bit for bit.
#[cfg(all(test, target_arch = "x86_64"))]
mod saturation_absorption {
    use super::*;
    use std::arch::x86_64::*;

    /// `tanh_ps` as it was written before the blends were removed.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tanh_ps_with_blend(x: __m256) -> __m256 {
        unsafe {
            let poly = avx2::tanh_ps(x);
            let above = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(-1.0), below)
        }
    }

    /// `sigmoid_ps` as it was written before the blends were removed.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sigmoid_ps_with_blend(x: __m256) -> __m256 {
        unsafe {
            let poly = avx2::sigmoid_ps(x);
            let above = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(0.0), below)
        }
    }

    /// Runs both forms over one vector of inputs and returns the two results.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pair(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        input: &[f32; 8],
    ) -> ([f32; 8], [f32; 8]) {
        unsafe {
            let v = _mm256_loadu_ps(input.as_ptr());
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            _mm256_storeu_ps(a.as_mut_ptr(), lean(v));
            _mm256_storeu_ps(b.as_mut_ptr(), reference(v));
            (a, b)
        }
    }

    fn assert_same(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        input: &[f32; 8],
        what: &str,
    ) {
        let (a, b) = unsafe { pair(lean, reference, input) };
        for i in 0..8 {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{what}({:e}): without blend {:e} ({:08x}), with blend {:e} ({:08x})",
                input[i],
                a[i],
                a[i].to_bits(),
                b[i],
                b[i].to_bits(),
            );
        }
    }

    /// Walks every `f32` in `[lo, hi]` by ULP, eight at a time, plus the
    /// negation of each.
    fn sweep(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        lo: f32,
        hi: f32,
        what: &str,
    ) -> u64 {
        let (mut bits, end) = (lo.to_bits(), hi.to_bits());
        let mut seen = 0u64;
        while bits <= end {
            let mut pos = [0.0f32; 8];
            let mut neg = [0.0f32; 8];
            for slot in 0..8 {
                let b = (bits + slot as u32).min(end);
                pos[slot] = f32::from_bits(b);
                neg[slot] = -pos[slot];
            }
            assert_same(lean, reference, &pos, what);
            assert_same(lean, reference, &neg, what);
            seen += 8;
            bits += 8;
        }
        seen
    }

    fn avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn tanh_saturation_blend_is_redundant_near_the_boundary() {
        if !avx2() {
            return;
        }
        // Dense over the decade above the clamp, where an overshoot would show.
        let n = sweep(avx2::tanh_ps, tanh_ps_with_blend, 9.0, 128.0, "tanh");
        assert!(n > 32_000_000, "sweep covered only {n} values");
    }

    #[test]
    fn sigmoid_saturation_blend_is_redundant_near_the_boundary() {
        if !avx2() {
            return;
        }
        let n = sweep(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            18.0,
            256.0,
            "sigmoid",
        );
        assert!(n > 20_000_000, "sweep covered only {n} values");
    }

    #[test]
    fn saturation_blend_is_redundant_for_special_values() {
        if !avx2() {
            return;
        }
        let specials = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            // A signalling NaN and a non-canonical quiet payload: `blendv`
            // looked at the sign bit and `andnot` looks at every bit, so the
            // substitution is only safe if these pass through untouched too.
            f32::from_bits(0x7F80_0001),
            f32::from_bits(0xFFC0_DEAD),
            f32::MAX,
            f32::MIN,
        ];
        assert_same(avx2::tanh_ps, tanh_ps_with_blend, &specials, "tanh");
        assert_same(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            &specials,
            "sigmoid",
        );

        // Exactly at, and one ULP either side of, both clamp boundaries.
        for &edge in &[tanh_c::LOWER, tanh_c::UPPER] {
            let e = edge.to_bits();
            let around = [
                f32::from_bits(e - 2),
                f32::from_bits(e - 1),
                edge,
                f32::from_bits(e + 1),
                f32::from_bits(e + 2),
                edge,
                edge,
                edge,
            ];
            assert_same(avx2::tanh_ps, tanh_ps_with_blend, &around, "tanh");
        }
        for &edge in &[logistic_c::LOWER, logistic_c::UPPER] {
            let e = edge.to_bits();
            let around = [
                f32::from_bits(e - 2),
                f32::from_bits(e - 1),
                edge,
                f32::from_bits(e + 1),
                f32::from_bits(e + 2),
                edge,
                edge,
                edge,
            ];
            assert_same(avx2::sigmoid_ps, sigmoid_ps_with_blend, &around, "sigmoid");
        }
    }

    /// The complete proof: every finite `f32` past the clamp, both signs.
    /// 1 047 527 424 values for `tanh` and 1 039 138 816 for `sigmoid`, which
    /// is minutes in an unoptimised test build, so it is not run by default.
    #[test]
    #[ignore = "exhaustive; run with --ignored"]
    fn saturation_blend_is_redundant_exhaustively() {
        if !avx2() {
            return;
        }
        sweep(avx2::tanh_ps, tanh_ps_with_blend, 9.0, f32::MAX, "tanh");
        sweep(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            18.0,
            f32::MAX,
            "sigmoid",
        );
    }
}

/// Splitting an elementwise kernel across threads must not change its result.
///
/// This is not a tolerance check. Every kernel here is exactly
/// chunk-independent, so a parallel run has to be **bit-identical** to a serial
/// one — not merely close. The failure this guards against is a chunk that
/// lands on a different code path (short enough to drop out of the vector
/// path) or, for the bias kernels, one cut mid-row, which would rotate
/// `bias[i % width]` for every element after the cut. Either would make output
/// depend on the machine's core count.
///
/// Note on how the parallel side is obtained: `ThreadPool::install` runs its
/// closure *on a pool worker*, which trips the nesting guard in
/// [`run_chunked`] and silently serialises. So the parallel run is a plain
/// direct call — the test binary's global pool is multi-threaded — and only
/// the serial reference goes through a one-thread pool. An earlier version of
/// this test used `install` for both sides; it passed even with the row
/// alignment deliberately broken.
#[cfg(test)]
mod thread_invariance {
    use super::*;

    /// Long enough to be split, and not a multiple of the chunk size or the
    /// lane count. It *is* divisible by 3, so the width-3 case happens to
    /// divide evenly; the adversarial coverage comes from width 4099, which
    /// leaves a real partial final row, and from chunk boundaries that are not
    /// multiples of 8 (262 146 and 262 336 for widths 3 and 11), which is what
    /// would expose a cut made mid-row.
    const N: usize = 5 * PAR_MIN_LEN + 37;

    fn probe(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / 991.0;
                // Spans both saturation bands of every kernel here, plus the
                // near-zero region where the polynomials are least alike.
                (t.sin() * 24.0) + (i % 7) as f32 * 1e-7 - 3.0
            })
            .collect()
    }

    /// Runs `f` with the global pool serialised, i.e. what a one-core host
    /// would compute.
    fn serial<T: Send>(f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool")
            .install(f)
    }

    fn assert_same(name: &str, run: impl Fn(&[f32], &mut [f32]) + Sync) {
        if rayon::current_num_threads() < 2 {
            eprintln!(
                "skipped {name}: global pool is single-threaded, so this test \
                 would compare serial against serial and could not fail"
            );
            return;
        }
        let x = probe(N);
        let mut want = vec![0.0f32; N];
        serial(|| run(&x, &mut want));

        let mut got = vec![0.0f32; N];
        run(&x, &mut got);

        for (i, (&w, &g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "{name}: element {i} differs (serial {w:e}, parallel {g:e}) \
                 — splitting the slice changed the result"
            );
        }
    }

    #[test]
    fn unary_kernels_are_thread_count_invariant() {
        assert_same("tanh", tanh_f32_slice);
        assert_same("sqrt", sqrt_f32_slice);
        assert_same("sigmoid", sigmoid_f32_slice);
        assert_same("exp", exp_f32_slice);
        assert_same("erf", erf_f32_slice);
        assert_same("erf_gelu", erf_gelu_f32_slice);
        assert_same("tanh_gelu", tanh_gelu_f32_slice);
    }

    #[test]
    fn quick_gelu_is_thread_count_invariant() {
        for alpha in [1.702f32, 1.0, -0.5] {
            assert_same("quick_gelu", |x, y| quick_gelu_f32_slice(x, y, alpha));
        }
    }

    /// Widths are chosen to be coprime with the lane count and not to divide
    /// the chunk size, so a mid-row cut cannot go unnoticed.
    #[test]
    fn bias_kernels_are_thread_count_invariant() {
        for width in [1usize, 3, 7, 11, 64, 4096, 4099] {
            let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.013 - 0.4).collect();
            assert_same("tanh_gelu_bias", |x, y| {
                tanh_gelu_bias_f32_slice(x, &bias, width, y)
            });
            assert_same("erf_gelu_bias", |x, y| {
                erf_gelu_bias_f32_slice(x, &bias, width, y)
            });
        }
    }

    /// The execution above can only use the core count this machine happens to
    /// have. The chunk policy is pure, so sweep it directly over thread counts
    /// and lengths the host cannot produce, and assert the two invariants the
    /// kernels depend on.
    #[test]
    fn chunk_policy_holds_across_thread_counts_and_lengths() {
        for threads in [1usize, 2, 3, 4, 7, 16, 64, 256, 4096] {
            for len in [
                0,
                1,
                8,
                PAR_MIN_CHUNK - 1,
                PAR_MIN_CHUNK,
                PAR_MIN_CHUNK + 1,
                PAR_MIN_LEN - 1,
                PAR_MIN_LEN,
                PAR_MIN_LEN + 1,
                N,
                1 << 22,
                (1 << 22) + 13,
            ] {
                if let Some(chunk) = par_chunk_len(len, threads) {
                    assert!(chunk < len, "chunk {chunk} !< len {len}");
                    assert_eq!(chunk % 8, 0, "chunk {chunk} is not a whole vector");
                    assert!(
                        chunk >= PAR_MIN_CHUNK,
                        "chunk {chunk} could drop below the vector threshold"
                    );
                }
                for width in [1usize, 3, 8, 64, 4099, PAR_MIN_LEN] {
                    if let Some(chunk) = par_chunk_len_rows(len, width, threads) {
                        assert!(chunk < len, "rows: chunk {chunk} !< len {len}");
                        assert_eq!(
                            chunk % width,
                            0,
                            "rows: chunk {chunk} cuts row width {width} in half"
                        );
                        assert!(
                            chunk >= PAR_MIN_CHUNK.min(len),
                            "rows: chunk {chunk} too short"
                        );
                    }
                }
            }
        }
    }

    /// `serial_scope` is what keeps the f16/bf16 sandwich off the pool. If it
    /// ever stopped suppressing the split, those paths would silently pick up
    /// the measured 0.59-0.79x regression again with nothing to catch it.
    #[test]
    fn serial_scope_suppresses_the_split() {
        assert!(!force_serial());
        serial_scope(|| {
            assert!(force_serial(), "guard did not take effect");
            assert_eq!(
                par_chunk_len_under_guard(1 << 20, 16),
                None,
                "run_chunked would still split inside serial_scope"
            );
        });
        assert!(!force_serial(), "guard leaked past its scope");
    }

    /// Mirrors the two early-outs `run_chunked` applies before consulting the
    /// policy, so the test above exercises the same condition the kernel does.
    fn par_chunk_len_under_guard(len: usize, threads: usize) -> Option<usize> {
        if len < PAR_MIN_LEN || force_serial() {
            return None;
        }
        par_chunk_len(len, threads)
    }

    /// The guard must be restored even if the closure unwinds, or one panicking
    /// kernel would serialise every later call on that thread for the rest of
    /// the process -- silently, since results stay correct either way.
    #[test]
    fn serial_scope_is_restored_after_a_panic() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(|| serial_scope(|| panic!("boom")));
        std::panic::set_hook(prev);

        assert!(r.is_err(), "the panic should still propagate");
        assert!(
            !force_serial(),
            "guard leaked across an unwind: this thread is now serial forever"
        );
    }

    /// Nesting must restore the outer value, not reset to "parallel".
    #[test]
    fn serial_scope_nests() {
        serial_scope(|| {
            serial_scope(|| assert!(force_serial()));
            assert!(force_serial(), "inner scope cleared the outer guard");
        });
        assert!(!force_serial());
    }

    /// A zero width would divide by zero; a width whose row count overflows
    /// must not panic either.
    #[test]
    fn chunk_policy_rejects_degenerate_widths() {
        assert_eq!(par_chunk_len_rows(N, 0, 8), None);
        assert_eq!(par_chunk_len_rows(N, usize::MAX, 8), None);
        assert_eq!(par_chunk_len(N, 0), None);
    }
}

/// Every activation entry point must reach the pool through [`run_chunked`],
/// including the ones that hand the arithmetic to MLAS.
///
/// The MLAS route used to call its kernel on the whole tensor and return,
/// skipping `run_chunked` entirely. Nothing caught it: the results were right,
/// the thread-invariance tests passed (a route that never splits is trivially
/// split-invariant), and the only symptom was that an `mlas`-on build stayed
/// single-threaded while an `mlas`-off build scaled — worth 3.4-4.3x at 16
/// threads on this host, in the configuration the wheel ships.
///
/// So this asserts the mechanism rather than the output: a tensor over
/// `PAR_MIN_LEN`, submitted from outside the pool, must increment
/// `run_chunked`'s parallel-branch counter.
#[cfg(test)]
mod parallel_reachability {
    use super::*;

    /// Over `PAR_MIN_LEN` so the parallel branch is eligible, and not a
    /// multiple of the chunk size or the lane count.
    const N: usize = PAR_MIN_LEN + 4099;

    fn assert_parallelises(name: &str, run: impl Fn(&[f32], &mut [f32])) {
        if rayon::current_num_threads() < 2 {
            eprintln!(
                "skipped {name}: global pool is single-threaded, so run_chunked \
                 would take its serial branch and this test could not fail"
            );
            return;
        }
        assert!(
            rayon::current_thread_index().is_none(),
            "this test must run outside the pool: run_chunked deliberately \
             stays serial when it is already inside a parallel region"
        );

        let x = vec![0.5f32; N];
        let mut y = vec![0.0f32; N];

        let before = parallel_dispatches();
        run(&x, &mut y);
        let after = parallel_dispatches();

        assert!(
            after > before,
            "{name}: a {N}-element call did not reach run_chunked's parallel \
             branch, so it runs single-threaded no matter how large the pool \
             is. A kernel that calls its backend directly and returns will \
             fail here."
        );
    }

    #[test]
    fn mlas_routed_kernels_still_go_through_run_chunked() {
        // `Erf` and exact `Gelu` are the two that still take the MLAS route
        // when the `mlas` feature is on, which is what the wheel ships.
        assert_parallelises("erf", erf_f32_slice);
        assert_parallelises("erf_gelu", erf_gelu_f32_slice);
    }

    #[test]
    fn pure_rust_kernels_go_through_run_chunked() {
        assert_parallelises("tanh", tanh_f32_slice);
        assert_parallelises("sigmoid", sigmoid_f32_slice);
        assert_parallelises("sqrt", sqrt_f32_slice);
        assert_parallelises("tanh_gelu", tanh_gelu_f32_slice);
        assert_parallelises("exp", exp_f32_slice);
    }
}

/// `run_chunked` is generic, and where it is *instantiated* changes codegen.
///
/// #1130 wrapped `Clip` in `run_chunked` from `selection.rs`. The runtime path
/// for the other unary ops did not change by a single instruction, but moving
/// that one instantiation across a module boundary repartitioned the crate's
/// codegen units and the AVX2 unary kernels in this module stopped being
/// vectorised. Measured at n = 65536, one thread, same machine, same commit
/// except for that instantiation:
///
/// | op | instantiated here | instantiated in `selection.rs` |
/// |---|---|---|
/// | `Sqrt` | 21.5 us | 48.5 us |
/// | `Tanh` | 30.4 us | 54.7 us |
/// | `Sigmoid` | 32.2 us | 57.1 us |
/// | `QuickGelu` | 42.6 us | 64.6 us |
/// | `FastGelu` | 55.7 us | 79.2 us |
///
/// Nothing in the compiler promises that partitioning is stable, so the rule is
/// mechanical rather than clever: `run_chunked` stays private to this module,
/// and callers elsewhere go through `run_chunked_fn` or `clip_chunked`, which
/// are instantiated here. This test enforces it, because the failure mode is a
/// 2.3x slowdown in files the author never opened and no test result changes.
#[cfg(test)]
mod chunking_instantiation_is_local {
    #[test]
    fn no_module_outside_this_one_instantiates_run_chunked() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).expect("read src") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_some_and(|e| e == "rs")
                    && path.file_name().is_some_and(|f| f != "simd_activations.rs")
                {
                    let text = std::fs::read_to_string(&path).expect("read source");
                    for (i, line) in text.lines().enumerate() {
                        // `run_chunked_fn(` and `run_chunked_rows(` are fine: the
                        // first is not generic, the second is only used here.
                        if line.trim_start().starts_with("//") {
                            continue;
                        }
                        // Match the identifier on a word boundary, then require
                        // the next non-space token to open a call or a
                        // turbofish. That accepts `run_chunked(`, `run_chunked
                        // ::<T>(` and `run_chunked ()`, while rejecting
                        // `run_chunked_fn`, `run_chunked_rows`, test names like
                        // `silu_reaches_run_chunked_parallel_branch`, and prose
                        // that does not go on to open a call. It is a
                        // heuristic, not a parser: a call spelled through an
                        // aliased import, split across two lines, or generated
                        // by a macro would slip past it, and the literal text
                        // `run_chunked(` inside a string would trip it. It
                        // catches the accidental case, which is a plain call
                        // added in another module.
                        let is_word = |c: char| c.is_alphanumeric() || c == '_';
                        let calls = line.match_indices("run_chunked").any(|(at, _)| {
                            let before_ok =
                                line[..at].chars().next_back().is_none_or(|c| !is_word(c));
                            let rest = line[at + "run_chunked".len()..].trim_start();
                            before_ok && (rest.starts_with('(') || rest.starts_with("::<"))
                        });
                        if calls {
                            offenders.push(format!("{}:{}", path.display(), i + 1));
                        }
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "run_chunked is generic and must only be instantiated inside \
             simd_activations.rs; instantiating it elsewhere repartitioned the \
             crate's codegen units and cost up to 2.3x (Sqrt; 1.7-1.8x on Tanh and Sigmoid). Use \
             run_chunked_fn or add a wrapper next to clip_chunked instead. \
             Offending call sites: {offenders:?}"
        );
    }

    /// The host branch is cold, and inlining it into `run_chunked` costs 34%.
    ///
    /// `try_host`/`try_host_rows` decide with one relaxed load, and the split
    /// they guard only runs inside an ORT session whose intra-op pool has been
    /// proven parallel. `run_chunked`'s callers, meanwhile, are the hottest
    /// elementwise kernels in the crate. When the branch was written inline,
    /// `Relu` at 1 Mi and one thread went 236 -> 315 us with no runtime path
    /// change at all -- the same codegen-unit repartitioning as above. Adding
    /// an `eprintln!` for diagnosis moved it back to 237 us, which is how the
    /// cause was identified. `#[inline(never)]` pins it.
    #[test]
    fn the_host_branch_stays_outlined() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/kernels/simd_activations.rs");
        let text = std::fs::read_to_string(&path).expect("read source");
        for name in ["fn try_host<F>", "fn try_host_rows<F>"] {
            let at = text.find(name).unwrap_or_else(|| panic!("{name} exists"));
            assert!(
                text[..at]
                    .lines()
                    .next_back()
                    .is_some_and(|l| l.trim() == "#[inline(never)]"),
                "{name} must be preceded by #[inline(never)]: inlining the host \
                 branch into run_chunked repartitioned codegen units and cost \
                 Relu 34% at 1 Mi (236 -> 315 us) with no path change"
            );
        }
    }
}

/// The host-pool split: what happens when ORT lends us its intra-op threads.
///
/// The production host is ORT's `KernelContext_ParallelFor`, which needs a
/// live session to exercise. These tests stand a real thread pool in its
/// place, so everything on our side of the seam — the chunk policy, the
/// disjointness of the ranges, the suppression of the rayon path and of
/// nesting — is covered without one.
#[cfg(test)]
mod host_pool_split {
    use super::*;
    use onnx_runtime_ep_api::HostParallel;
    use onnx_runtime_ep_api::host_parallel;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    thread_local! {
        /// Indices dispatched through the fake host from *this* thread since
        /// the last reset.
        ///
        /// Thread-local rather than a global atomic for the same reason
        /// `PARALLEL_DISPATCHES` is: the test binary runs these tests
        /// concurrently, and a global counter would let them bump each
        /// other's. The increment happens on whichever thread called
        /// `threaded_host`, which is also the thread that reads it back --
        /// including inside a task, which is what makes the nesting test
        /// work.
        static HOST_INDICES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn dispatched() -> usize {
        HOST_INDICES.with(std::cell::Cell::get)
    }

    fn reset_dispatched() {
        HOST_INDICES.with(|c| c.set(0));
    }

    /// Stands in for ORT: runs the indices on four real threads.
    ///
    /// Genuinely concurrent on purpose. A serial stand-in would still prove
    /// the arithmetic but not that the ranges are disjoint, which is the part
    /// that would corrupt an output tensor if it were wrong.
    ///
    /// # Safety
    ///
    /// Ignores `host`, so any pointer is valid for it.
    unsafe fn threaded_host(_host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync)) {
        HOST_INDICES.with(|c| c.set(c.get() + total));
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= total {
                            break;
                        }
                        body(index);
                    }
                });
            }
        });
    }

    /// A host pool that has been seen running our chunks on its own threads.
    static HELPED: std::sync::atomic::AtomicU32 =
        std::sync::atomic::AtomicU32::new(host_parallel::HOST_HELPED);

    /// A host pool that never has, past its opening burst of probes: the
    /// kernels should keep the work on their own pool.
    static NEVER_HELPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(
        (host_parallel::PROBE_MIN << 16) | host_parallel::PROBE_MIN,
    );

    fn fake_host() -> HostParallel {
        // SAFETY: `threaded_host` never dereferences its `host` argument, and
        // the probe cell is a `static`.
        unsafe {
            HostParallel::new(
                core::ptr::null_mut(),
                threaded_host,
                core::ptr::from_ref(&HELPED),
            )
        }
    }

    /// The same host, but one that has never been seen to help.
    fn unhelpful_host() -> HostParallel {
        // SAFETY: as `fake_host`.
        unsafe {
            HostParallel::new(
                core::ptr::null_mut(),
                threaded_host,
                core::ptr::from_ref(&NEVER_HELPED),
            )
        }
    }

    /// Long enough to be split, and deliberately not a multiple of the chunk
    /// size, so the final chunk is short.
    const N: usize = 5 * HOST_MIN_LEN + 37;

    fn probe(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 / 991.0).sin() * 24.0 + (i % 7) as f32 * 1e-7 - 3.0)
            .collect()
    }

    /// Asserts the host split computes exactly what one call would.
    fn assert_same(name: &str, run: impl Fn(&[f32], &mut [f32]) + Sync) {
        let x = probe(N);
        let mut want = vec![0.0f32; N];
        serial_scope(|| run(&x, &mut want));

        reset_dispatched();
        let mut got = vec![0.0f32; N];
        host_parallel::scope(fake_host(), || run(&x, &mut got));
        assert!(
            dispatched() > 1,
            "{name}: the host pool was never asked to split anything"
        );

        for (i, (&w, &g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "{name}: element {i} differs (whole {w:e}, host-split {g:e})"
            );
        }
    }

    #[test]
    fn unary_kernels_match_the_unsplit_result() {
        assert_same("tanh", tanh_f32_slice);
        assert_same("sqrt", sqrt_f32_slice);
        assert_same("sigmoid", sigmoid_f32_slice);
        assert_same("exp", exp_f32_slice);
        assert_same("erf", erf_f32_slice);
        assert_same("erf_gelu", erf_gelu_f32_slice);
        assert_same("tanh_gelu", tanh_gelu_f32_slice);
        assert_same("quick_gelu", |x, y| quick_gelu_f32_slice(x, y, 1.702));
    }

    #[test]
    fn bias_kernels_match_the_unsplit_result() {
        for width in [1usize, 3, 11, 4096, 4099] {
            let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.013 - 0.4).collect();
            assert_same("tanh_gelu_bias", |x, y| {
                tanh_gelu_bias_f32_slice(x, &bias, width, y)
            });
            assert_same("erf_gelu_bias", |x, y| {
                erf_gelu_bias_f32_slice(x, &bias, width, y)
            });
        }
    }

    /// The point of the whole change: with a host pool installed we must not
    /// also start our own.
    #[test]
    fn the_rayon_pool_is_not_used_when_a_host_is_installed() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipped: single-threaded rayon pool cannot show the difference");
            return;
        }
        let x = probe(N);
        let mut y = vec![0.0f32; N];

        reset_dispatched();
        host_parallel::scope(fake_host(), || tanh_f32_slice(&x, &mut y));
        let host_indices = dispatched();

        assert!(host_indices > 1, "the host pool was not used");
        assert_eq!(
            host_indices,
            host_chunk_len(N).expect("N is long enough to split").1,
            "every chunk should have been dispatched exactly once"
        );
    }

    /// A kernel reached from inside a host task is already on a host thread.
    /// Splitting again would nest a pool inside the pool we were handed.
    #[test]
    fn a_nested_split_stays_serial() {
        let x = probe(N);
        let mut y = vec![0.0f32; N];
        host_parallel::scope(fake_host(), || {
            fake_host().run(2, &|_| {
                assert!(host_parallel::in_host_task());
                let before = dispatched();
                let mut inner = vec![0.0f32; N];
                tanh_f32_slice(&x, &mut inner);
                assert_eq!(
                    dispatched(),
                    before,
                    "a kernel inside a host task dispatched again"
                );
            });
            tanh_f32_slice(&x, &mut y);
        });
    }

    /// Below the gate the slice stays whole even with a host installed: the
    /// dispatch would cost more than the work it hands out.
    #[test]
    fn a_short_slice_is_not_dispatched() {
        let n = HOST_MIN_LEN - 8;
        let x = probe(n);
        let mut y = vec![0.0f32; n];
        reset_dispatched();
        host_parallel::scope(fake_host(), || tanh_f32_slice(&x, &mut y));
        assert_eq!(dispatched(), 0, "{n} elements should have stayed whole");

        let mut want = vec![0.0f32; n];
        serial_scope(|| tanh_f32_slice(&x, &mut want));
        assert!(want.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// A host pool that has never been seen doing our work is not using the
    /// machine, so ours may. Borrowing a single ORT thread instead measured
    /// 2-9x slower over 1-4 Mi at `intra_op = 1`, so this is worth asserting.
    #[test]
    fn an_unhelpful_host_is_not_borrowed() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipped: single-threaded rayon pool cannot show the difference");
            return;
        }
        let n = 2 * PAR_MIN_LEN;
        let x = probe(n);
        let mut y = vec![0.0f32; n];
        reset_dispatched();
        let before = parallel_dispatches();
        // Past the opening burst with nothing to show for it, which is the
        // steady state for a session whose host pool has no workers.
        NEVER_HELPED.store(
            (host_parallel::PROBE_MIN << 16) | host_parallel::PROBE_MIN,
            std::sync::atomic::Ordering::Relaxed,
        );
        host_parallel::scope(unhelpful_host(), || tanh_f32_slice(&x, &mut y));
        assert_eq!(
            dispatched(),
            0,
            "an unhelpful host pool was borrowed anyway"
        );
        assert_eq!(
            parallel_dispatches() - before,
            1,
            "the work should have gone to our own pool instead"
        );

        let mut want = vec![0.0f32; n];
        serial_scope(|| tanh_f32_slice(&x, &mut want));
        assert!(want.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// `serial_scope` has to keep suppressing the split on the host path too,
    /// or the f16/bf16 sandwich picks its measured regression back up.
    #[test]
    fn serial_scope_still_suppresses_the_split() {
        let x = probe(N);
        let mut y = vec![0.0f32; N];
        reset_dispatched();
        host_parallel::scope(fake_host(), || {
            serial_scope(|| tanh_f32_slice(&x, &mut y));
        });
        assert_eq!(dispatched(), 0);
    }

    /// The host chunk policy is pure, so sweep it over lengths and widths this
    /// machine's memory could not hold, and assert the invariants the kernels
    /// depend on: whole vectors, never below the vector threshold, and — for
    /// the bias kernels — never a cut through the middle of a row.
    #[test]
    fn host_chunk_policy_holds_across_lengths() {
        for len in [
            0,
            1,
            8,
            HOST_MIN_CHUNK - 1,
            HOST_MIN_CHUNK,
            HOST_MIN_CHUNK + 1,
            HOST_MIN_LEN - 1,
            HOST_MIN_LEN,
            HOST_MIN_LEN + 1,
            PAR_MIN_CHUNK,
            PAR_MIN_LEN,
            N,
            1 << 26,
            (1 << 26) + 13,
            usize::MAX / 2,
        ] {
            if let Some((chunk, count)) = host_chunk_len(len) {
                assert!(chunk < len, "chunk {chunk} !< len {len}");
                assert_eq!(chunk % 8, 0, "chunk {chunk} is not a whole vector");
                assert!(len >= HOST_MIN_LEN, "split below the gate");
                assert!(chunk >= HOST_MIN_CHUNK, "chunk {chunk} is too short");
                assert!(chunk >= SIMD_MIN_LEN, "chunk {chunk} would go scalar");
                assert_eq!(count, len.div_ceil(chunk));
                assert!(
                    (count - 1) * chunk < len,
                    "chunk {chunk} x {count} would dispatch an empty range"
                );
                assert!(count <= MAX_HOST_CHUNKS, "{count} chunks is past the cap");
            }
            for width in [1usize, 3, 8, 64, 4099, PAR_MIN_LEN] {
                if let Some((chunk, count)) = host_chunk_len_rows(len, width) {
                    assert!(chunk < len, "rows: chunk {chunk} !< len {len}");
                    assert_eq!(chunk % width, 0, "rows: chunk {chunk} cuts width {width}");
                    assert!(len >= HOST_MIN_LEN, "rows: split below the gate");
                    assert!(
                        chunk >= HOST_MIN_CHUNK.min(len),
                        "rows: chunk {chunk} short"
                    );
                    assert_eq!(count, len.div_ceil(chunk));
                    assert!((count - 1) * chunk < len, "rows: empty final range");
                }
            }
            assert_eq!(host_chunk_len_rows(len, 0), None, "width 0 must not split");
            if len < HOST_MIN_LEN {
                assert_eq!(host_chunk_len(len), None, "{len} is below the gate");
            }
        }
    }
}

/// Interleaved three-arm micro-measurement for the host-pool split.
///
/// This is the arm that decides PR #1143: everything measured before compared
/// *our rayon split against staying serial*, and serial won every row. Nothing
/// showed that dispatching the split onto the **host's** pool — the change this
/// PR actually makes — beats serial. This harness runs all three arms in one
/// process, interleaved per rep, so between-arm drift on a shared box cannot
/// flip the conclusion.
///
/// # Why a stand-in pool, and why it is trustworthy
///
/// A real ORT session cannot switch arms within one process — its rayon width
/// is a process global and its host install is a code path — so the three arms
/// are driven directly against the real `simd_activations` code paths. The one
/// thing that has to be reproduced faithfully is *why* our rayon split loses
/// under an ORT session: ORT's intra-op pool **spins**, so a second (rayon)
/// pool alongside it oversubscribes the cores. The stand-in models exactly
/// that — a persistent pool of workers that spin while idle (as ORT's do) and
/// claim indices dynamically when dispatched (`num_batch = 0` semantics) — and
/// it latches [`host_parallel::HOST_HELPED`] the honest way, by having a worker
/// (not the dispatcher) run one of the chunks, so `prefer_host` reaches its
/// steady state through the real mechanism rather than by fiat.
///
/// The `rayon split` arm is therefore a **built-in control**: its result is
/// already known from the PR's nine-row table (serial wins at `intra_op = 16`).
/// If this harness reproduces `rayon < serial`, the model is validated and its
/// `host-pool` number can be trusted. If it does not, the run is unusable and
/// we know it before drawing a conclusion.
///
/// Ignored by the normal gate; run with:
/// `EP_BENCH=1 cargo test --release -p onnx-runtime-ep-cpu --lib three_arm_bench -- --nocapture --ignored --test-threads=1`
#[cfg(test)]
mod three_arm_bench {
    use super::*;
    use onnx_runtime_ep_api::HostParallel;
    use onnx_runtime_ep_api::host_parallel;
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// State shared between the dispatcher and the stand-in pool's workers.
    ///
    /// A dispatch publishes the body (as its two raw fat-pointer words) and the
    /// index count, then bumps `generation`; workers pick the new generation up
    /// and claim indices off `cursor` until it is drained, recording progress
    /// in `done`. Everything is lock-free so idle workers can *spin*, which is
    /// the property that makes a coexisting rayon pool oversubscribe.
    struct Shared {
        generation: AtomicUsize,
        total: AtomicUsize,
        cursor: AtomicUsize,
        done: AtomicUsize,
        body_data: AtomicUsize,
        body_vtable: AtomicUsize,
        stop: AtomicBool,
        /// The probe cell `prefer_host` reads: starts at zero (opening burst)
        /// and reaches `HOST_HELPED` the first time a worker runs a chunk.
        probe: AtomicU32,
        /// Set by a worker (never the dispatcher) when it runs a chunk, so the
        /// dispatcher can latch the probe cell only on real positive evidence.
        worker_helped: AtomicBool,
    }

    impl Shared {
        fn new() -> Self {
            Self {
                generation: AtomicUsize::new(0),
                total: AtomicUsize::new(0),
                cursor: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                body_data: AtomicUsize::new(0),
                body_vtable: AtomicUsize::new(0),
                stop: AtomicBool::new(false),
                probe: AtomicU32::new(0),
                worker_helped: AtomicBool::new(false),
            }
        }

        /// Claims and runs indices off `cursor` for the current generation.
        ///
        /// Shared by the workers and the calling thread — ORT's own
        /// `ParallelFor` runs tasks on the caller too, so the calling thread is
        /// one of the `intra_op` threads, not a spectator. A worker flags
        /// `worker_helped` so the dispatcher can latch the probe cell.
        ///
        /// # Safety
        ///
        /// The published body pointer must still be valid, which the blocking
        /// dispatch guarantees for its whole duration.
        unsafe fn drain(&self, total: usize, is_worker: bool) {
            let data = self.body_data.load(Ordering::Relaxed);
            let vtable = self.body_vtable.load(Ordering::Relaxed);
            // SAFETY: `data`/`vtable` are the two words of a `&(dyn Fn(usize) +
            // Sync)` published under the `generation` release/acquire, and the
            // referent outlives the dispatch.
            let body: *const (dyn Fn(usize) + Sync) =
                unsafe { core::mem::transmute([data, vtable]) };
            let body = unsafe { &*body };
            let mut ran = false;
            loop {
                let index = self.cursor.fetch_add(1, Ordering::Relaxed);
                if index >= total {
                    break;
                }
                body(index);
                ran = true;
                self.done.fetch_add(1, Ordering::Relaxed);
            }
            if is_worker && ran {
                self.worker_helped.store(true, Ordering::Relaxed);
            }
        }
    }

    /// A persistent pool of spinning workers standing in for ORT's intra-op
    /// pool. Sized so that `workers + 1` (the calling thread) equals the
    /// modelled `intra_op_num_threads`.
    struct SpinPool {
        shared: Arc<Shared>,
        handles: Vec<std::thread::JoinHandle<()>>,
    }

    impl SpinPool {
        fn new(workers: usize) -> Self {
            let shared = Arc::new(Shared::new());
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let shared = Arc::clone(&shared);
                handles.push(std::thread::spawn(move || {
                    let mut seen = 0usize;
                    loop {
                        if shared.stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let g = shared.generation.load(Ordering::Acquire);
                        if g != seen {
                            seen = g;
                            let total = shared.total.load(Ordering::Relaxed);
                            // SAFETY: the dispatch that bumped `generation` is
                            // blocked until `done == total`, so the body it
                            // published is alive for this whole drain.
                            unsafe { shared.drain(total, true) };
                        } else {
                            // Idle: burn the core, exactly as ORT's workers do
                            // between parallel regions. This is what makes the
                            // rayon arm oversubscribe.
                            std::hint::spin_loop();
                        }
                    }
                }));
            }
            Self { shared, handles }
        }

        /// The `HostParallel` handle our kernels see.
        fn handle(&self) -> HostParallel {
            // SAFETY: `spin_dispatch` reads the host pointer back as
            // `*const Shared`, and the probe pointer is the cell inside the same
            // `Arc`. The `Arc` in `self` keeps both alive for as long as the
            // handle is installed (the handle never escapes this struct).
            unsafe {
                HostParallel::new(
                    Arc::as_ptr(&self.shared).cast::<c_void>().cast_mut(),
                    spin_dispatch,
                    core::ptr::from_ref(&self.shared.probe),
                )
            }
        }
    }

    impl Drop for SpinPool {
        fn drop(&mut self) {
            self.shared.stop.store(true, Ordering::Relaxed);
            for h in self.handles.drain(..) {
                let _ = h.join();
            }
        }
    }

    /// Publishes one dispatch and drives it to completion on the pool + caller.
    ///
    /// # Safety
    ///
    /// `host` must be the `*const Shared` produced by [`SpinPool::handle`].
    unsafe fn spin_dispatch(host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync)) {
        let shared = unsafe { &*host.cast::<Shared>() };
        let raw = body as *const (dyn Fn(usize) + Sync);
        let [data, vtable]: [usize; 2] = unsafe { core::mem::transmute(raw) };
        shared.worker_helped.store(false, Ordering::Relaxed);
        shared.total.store(total, Ordering::Relaxed);
        shared.cursor.store(0, Ordering::Relaxed);
        shared.done.store(0, Ordering::Relaxed);
        shared.body_data.store(data, Ordering::Relaxed);
        shared.body_vtable.store(vtable, Ordering::Relaxed);
        // Release the body/total stores to the workers, then help drain.
        shared.generation.fetch_add(1, Ordering::Release);
        // SAFETY: we published the body above and block below until every index
        // is done, so it stays valid for the whole call.
        unsafe { shared.drain(total, false) };
        while shared.done.load(Ordering::Acquire) < total {
            std::hint::spin_loop();
        }
        // Latch the probe the honest way: only if a worker (not this calling
        // thread) actually ran a chunk. A pool with no workers can never set
        // this, which is exactly the fact `prefer_host` is trying to learn.
        if shared.worker_helped.load(Ordering::Relaxed) {
            shared
                .probe
                .store(host_parallel::HOST_HELPED, Ordering::Relaxed);
        }
    }

    /// A reproducible non-degenerate input in a range that exercises every
    /// branch of the activation polynomials.
    fn probe(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 / 977.0).sin() * 9.0 + (i % 13) as f32 * 1e-6 - 2.0)
            .collect()
    }

    fn pct(sorted: &[Duration], num: usize, den: usize) -> Duration {
        sorted[(sorted.len() * num / den).min(sorted.len() - 1)]
    }

    fn us(d: Duration) -> f64 {
        d.as_secs_f64() * 1e6
    }

    type Kernel = fn(&[f32], &mut [f32]);

    fn kernels() -> [(&'static str, Kernel); 2] {
        [
            ("Sqrt", sqrt_f32_slice as Kernel),
            ("Gelu", erf_gelu_f32_slice as Kernel),
        ]
    }

    /// The three arms at `intra_op = 16`, interleaved per rep.
    #[test]
    #[ignore = "measurement harness; run explicitly with EP_BENCH=1"]
    fn three_arm_intra_op_16() {
        if std::env::var("EP_BENCH").is_err() {
            return;
        }
        const N: usize = 1 << 20; // 1 Mi f32
        const REPS: usize = 201;
        const WARMUP: usize = 25;
        // The modelled intra-op width. Defaults to this box's physical core
        // count (14) rather than the spec's 16, because the stand-in's spinning
        // workers are *real* threads: on a 14-core box, 15 spinners + the caller
        // already oversubscribe, which starves the serial baseline before the
        // rayon arm even runs. Matching the core count gives every arm a clean
        // core and keeps the rayon arm the only one that oversubscribes.
        let intra_op: usize = std::env::var("EP_INTRA_OP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14);

        // (intra_op - 1) spinning workers + the calling thread == intra_op threads.
        let pool = SpinPool::new(intra_op - 1);
        let host = pool.handle();
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(intra_op)
            .build()
            .expect("rayon pool");

        let x = probe(N);
        let mut out = vec![0.0f32; N];

        println!("\n=== three-arm, intra_op = {intra_op}, N = 1 Mi f32, p50 us (p10..p90) ===");
        println!(
            "host stand-in: {} spinning workers + caller; rayon: {intra_op} threads",
            intra_op - 1
        );
        for (name, kernel) in kernels() {
            // Interleave the arms within one process: each rep samples all
            // three back to back, so a busy neighbour hits every arm equally.
            let arms = ["serial", "rayon split", "host-pool split"];
            let mut samples: [Vec<Duration>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for _ in 0..WARMUP + REPS {
                let mut run_arm = |arm: usize| {
                    let t = Instant::now();
                    match arm {
                        0 => serial_scope(|| kernel(&x, &mut out)),
                        1 => rayon_pool.install(|| kernel(&x, &mut out)),
                        _ => host_parallel::scope(host, || kernel(&x, &mut out)),
                    }
                    t.elapsed()
                };
                for (arm, bucket) in samples.iter_mut().enumerate() {
                    let d = run_arm(arm);
                    bucket.push(d);
                }
            }
            // The host arm must have latched, or its numbers are the rayon
            // fall-through, not the host path.
            assert!(
                host.helped(),
                "{name}: the stand-in host was never seen to help; numbers invalid"
            );
            print!("{name:>5}: ");
            let mut p50s = [0.0f64; 3];
            for (i, arm) in arms.iter().enumerate() {
                let s = &mut samples[i];
                s.drain(..WARMUP);
                s.sort_unstable();
                p50s[i] = us(pct(s, 1, 2));
                print!(
                    "{arm} {:.0} ({:.0}..{:.0})  ",
                    us(pct(s, 1, 2)),
                    us(pct(s, 1, 10)),
                    us(pct(s, 9, 10))
                );
            }
            println!();
            println!(
                "       -> rayon/serial {:.2}x  host/serial {:.2}x  (control: rayon must be > 1x)",
                p50s[1] / p50s[0],
                p50s[2] / p50s[0]
            );
        }
    }

    /// The no-host fall-through on a free machine: rayon split must keep its
    /// large win over serial at `intra_op = 1`. No stand-in pool, so nothing
    /// spins and the machine is the native executor's to use.
    #[test]
    #[ignore = "measurement harness; run explicitly with EP_BENCH=1"]
    fn no_host_fall_through_intra_op_1() {
        if std::env::var("EP_BENCH").is_err() {
            return;
        }
        const REPS: usize = 151;
        const WARMUP: usize = 20;
        println!("\n=== no-host fall-through, free machine, p50 us (p10..p90) ===");
        for n_label in ["1 Mi", "4 Mi"] {
            let n = if n_label == "1 Mi" { 1 << 20 } else { 1 << 22 };
            let x = probe(n);
            let mut out = vec![0.0f32; n];
            for (name, kernel) in kernels() {
                let mut ser = Vec::with_capacity(REPS);
                let mut par = Vec::with_capacity(REPS);
                for r in 0..WARMUP + REPS {
                    let t = Instant::now();
                    serial_scope(|| kernel(&x, &mut out));
                    let ds = t.elapsed();
                    let t = Instant::now();
                    // No host installed and not inside rayon: run_chunked takes
                    // the rayon fall-through on the global pool.
                    kernel(&x, &mut out);
                    let dp = t.elapsed();
                    if r >= WARMUP {
                        ser.push(ds);
                        par.push(dp);
                    }
                }
                ser.sort_unstable();
                par.sort_unstable();
                let (s, prl) = (pct(&ser, 1, 2), pct(&par, 1, 2));
                println!(
                    "{n_label} {name:>5}: serial {:.0}  rayon {:.0}  -> {:.2}x faster",
                    us(s),
                    us(prl),
                    us(s) / us(prl)
                );
            }
        }
    }
}
