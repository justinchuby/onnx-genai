//! Correctness-first `com.microsoft::MatMulNBits` for f32/f16/bf16 activations
//! and block-quantized 2-bit, 4-bit, or 8-bit weights.
//!
//! ORT stores `B` as
//! `[N, ceil(K / block_size), block_size * bits / 8]`, least-significant bits
//! first within each byte. For M=1 decode, constant quantized weights are
//! prepacked once and reused by a N-parallel GEMV. For symmetric block-32
//! int4 M=1, `accuracy_level=4` streams the packed weights directly into a VNNI
//! dot product on x86, or an opt-in ARM dot-product int4 GEMV on aarch64. Other
//! int4 accuracy-level-4 shapes keep the weights in int8 and quantize each
//! activation into int8 per K-block (matching ORT/MLAS CompInt8).
//! `weight_prepacked=1` accepts the host-specific buffer produced by
//! `MlasQNBitGemmPackQuantBData`, avoiding the standard-layout-to-MLAS repack.
//! The 2-bit path decodes packed weights directly inside its f32 GEMV/GEMM,
//! while the default int4 path dequantizes to f32. The 8-bit correctness path
//! uses the same affine dequantization with one uint8 weight and optional uint8
//! zero point per block.

use std::borrow::Cow;
use std::cell::Cell;
#[cfg(feature = "mlas")]
use std::collections::HashMap;
#[cfg(feature = "mlas")]
use std::sync::Arc;
#[cfg(feature = "mlas")]
use std::sync::LazyLock;
#[cfg(feature = "mlas")]
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, LazyWeightBoundary, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node};
use rayon::prelude::*;

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

/// Temporary, opt-in per-phase timers for the f16-activation MatMulNBits decode
/// path, gated by `ONNX_GENAI_PROFILE_MM=1`. Zero-cost when unset (one cached
/// bool load, no `Instant::now`). Splits a MatMulNBits call into: activation
/// widen (f16->f32), the MLAS SQNBit GEMV, and output narrow (f32->f16), so the
/// int4 GEMV cost can be separated from the per-op activation conversion.
mod mm_profile {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    static WIDEN_NS: AtomicU64 = AtomicU64::new(0);
    // MLAS-only phase counters are retained for ONNX_GENAI_PROFILE_MM=1.
    #[cfg(feature = "mlas")]
    static GEMV_NS: AtomicU64 = AtomicU64::new(0);
    #[cfg(feature = "mlas")]
    static NARROW_NS: AtomicU64 = AtomicU64::new(0);
    #[cfg(feature = "mlas")]
    static CALLS: AtomicU64 = AtomicU64::new(0);
    // One-time constant-weight repack/dequant done on the first execution of
    // each MatMulNBits node (the `dequantize_weight` int4->f32 expansion for the
    // non-MLAS hand path, or `build_mlas_*` for the MLAS path). This is the
    // lazy O(weight-bytes) work that dominates the native load/init fixed cost;
    // gated by the same `ONNX_GENAI_PROFILE_MM=1` switch.
    static PREPACK_NS: AtomicU64 = AtomicU64::new(0);
    static PREPACK_CALLS: AtomicU64 = AtomicU64::new(0);
    static PREPACK_BYTES: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("ONNX_GENAI_PROFILE_MM").is_ok_and(|v| {
                let v = v.trim();
                !v.is_empty() && v != "0"
            })
        })
    }

    /// Time `f`, adding the elapsed nanoseconds to `bucket` when profiling is on.
    #[inline]
    fn timed<T>(bucket: &AtomicU64, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let start = Instant::now();
        let out = f();
        bucket.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    pub fn time_widen<T>(f: impl FnOnce() -> T) -> T {
        timed(&WIDEN_NS, f)
    }
    #[cfg(feature = "mlas")]
    pub fn time_gemv<T>(f: impl FnOnce() -> T) -> T {
        timed(&GEMV_NS, f)
    }
    #[cfg(feature = "mlas")]
    pub fn time_narrow<T>(f: impl FnOnce() -> T) -> T {
        timed(&NARROW_NS, f)
    }

    /// Time one constant-weight one-time repack/dequant (`build_mlas_*` or
    /// `dequantize_weight`), tagged with a `phase` label and its `weight_bytes`
    /// (standard-layout `B` size). Every call emits the running total to stderr
    /// so a harness can read the last line to get the whole model's one-time
    /// prepack/dequant cost and its share of load time.
    pub fn time_prepack<T>(phase: &'static str, weight_bytes: usize, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let start = Instant::now();
        let out = f();
        let ns = start.elapsed().as_nanos() as u64;
        let total_ns = PREPACK_NS.fetch_add(ns, Ordering::Relaxed) + ns;
        let calls = PREPACK_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let bytes =
            PREPACK_BYTES.fetch_add(weight_bytes as u64, Ordering::Relaxed) + weight_bytes as u64;
        eprintln!(
            "[mm_prepack] phase={phase} calls={calls} prepack_total={total:.1}ms \
             cum_bytes={bytes} this={this:.2}ms this_bytes={weight_bytes}",
            total = total_ns as f64 / 1e6,
            this = ns as f64 / 1e6,
        );
        out
    }

    /// Record one MatMulNBits call and, every 512 calls, emit the running split
    /// to stderr so the harness can tail the final line.
    #[cfg(feature = "mlas")]
    pub fn tick() {
        if !enabled() {
            return;
        }
        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if calls.is_multiple_of(512) {
            let widen = WIDEN_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let gemv = GEMV_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let narrow = NARROW_NS.load(Ordering::Relaxed) as f64 / 1e6;
            let total = widen + gemv + narrow;
            eprintln!(
                "[mm_profile] calls={calls} total={total:.1}ms widen={widen:.1}ms \
                 ({wp:.1}%) gemv={gemv:.1}ms ({gp:.1}%) narrow={narrow:.1}ms ({np:.1}%)",
                wp = 100.0 * widen / total,
                gp = 100.0 * gemv / total,
                np = 100.0 * narrow / total,
            );
        }
    }
}

/// Overrides the bounded M=1 decode pool size; set to `0` to use the global
/// Rayon pool as an escape hatch.
const DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";

/// `mlas-sys` reads the same knob to size its standalone pool. Fail the build
/// rather than let the two names drift apart silently.
#[cfg(feature = "mlas")]
const _: () = {
    const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }
    assert!(
        bytes_eq(
            DECODE_THREADS_ENV.as_bytes(),
            mlas_sys::CPU_DECODE_THREADS_ENV.as_bytes()
        ),
        "DECODE_THREADS_ENV must match mlas_sys::CPU_DECODE_THREADS_ENV"
    );
};
/// Process-local override set by first-class callers such as the CLI. Zero means
/// no override, so the environment variable and automatic default still apply.
static DECODE_THREADS_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Governed opt-out of the resident dequantized-f32 decode cache.
///
/// The `m == 1` generic decode path (`weight_nk`) materializes a full f32
/// expansion of the packed weight and holds it in a per-kernel `OnceLock` for
/// the session -- ~8x the packed int4 bytes. That trade buys ~2.4x decode
/// throughput and is worth it whenever the expansion fits; when it does not, the
/// box pages and decode collapses (#971). When this flag is set the caching
/// branches skip the resident `OnceLock` and dequantize into a transient
/// per-call buffer that is dropped after each GEMV -- byte-identical output,
/// slower per token, but no resident 8x footprint. The memory-strategy plan
/// owns the fit decision and calls [`set_resident_dequant_f32_cache_enabled`]
/// before the native session loads. Default: enabled (unchanged fast path).
static RESIDENT_DEQUANT_F32_CACHE_DISABLED: AtomicBool = AtomicBool::new(false);

/// The `bits == 4, accuracy_level == 0` MLAS SQNBit CompFp32 route packs the
/// constant weight into an MLAS-owned buffer ([`mlas_sys::sqnbit_packed_b_size`],
/// which for CompFp32 is `N * K / 2` -- the same size as the on-disk int4
/// bytes) held for the session *beside* the still-resident memory-mapped weight.
/// Each packed buffer also retains its own `f32` scale (and, when asymmetric,
/// `u8` zero-point) copy, and the shape-keyed kernel cache materializes one such
/// buffer per activation shape -- a prefill (`m > 1`) and a decode (`m == 1`)
/// instance -- so the resident footprint is roughly `2 * 1.25` the int4 bytes
/// (measured on `qwen05b-symzp`: ~604 MB against 367 MB of int4 weights, with
/// the mapped weight still resident). That buffer buys a large decode speedup on
/// x86 (the borrowed int4 path has no SIMD kernel there). Like the resident f32
/// cache above, the memory-strategy plan owns the fit decision: it accounts
/// those bytes ([`matmul_nbits_resident_side_cache_bytes`], which counts the
/// scales and both instantiations) and, when they do not fit the residency
/// budget, calls [`set_mlas_sqnbit_packing_enabled`] with `false`.
/// When disabled, [`MatMulNBitsKernel::mlas_sqnbit_owns_fp32_compute`] declines
/// ownership so the node stays on the borrowed zero-copy int4 path -- the same
/// behaviour as before the dispatch change, holding only the on-disk weights.
/// Default: enabled (the fast MLAS route).
static MLAS_SQNBIT_PACKING_DISABLED: AtomicBool = AtomicBool::new(false);

/// Decode is bandwidth-bound and pays one fork/join per projection. Profiling
/// across the existing 4--96 worker sweep found no gain above eight workers and
/// clear regressions at 16+, so topology scaling is capped here; the environment
/// override remains available for processors whose measurements differ.
const MAX_TOPOLOGY_DECODE_THREADS: usize = 8;
static DECODE_POOL: OnceLock<std::result::Result<Option<rayon::ThreadPool>, String>> =
    OnceLock::new();

/// Upper bound on the bounded pool used for the **dense-f32** decode path (see
/// [`configured_dense_decode_threads`]). Unlike the quantized `MatMulNBits`
/// kernels, a dense-f32 model's decode is serviced by the multi-threaded MLAS
/// GEMM (one cache-aware parallel-for per projection, not a per-row fork/join),
/// so it scales well past the eight-worker flat ceiling. It is nonetheless pure
/// memory-bandwidth-bound GEMV, so throughput plateaus once the memory channels
/// saturate and extra workers only add sync + contention; the cap keeps the
/// default inside that plateau on many-core hosts (measured flat 16--32 workers
/// on a 96-core 2-socket Xeon; regressions when oversubscribing toward all 96).
const MAX_DENSE_DECODE_THREADS: usize = 32;
static DENSE_DECODE_POOL: OnceLock<std::result::Result<Option<rayon::ThreadPool>, String>> =
    OnceLock::new();

/// Env knob for the int4 MatMulNBits hand-decode ↔ MLAS SQNBit crossover (`m`
/// row count). MatMulNBits int4 with `m < NXRT_SQNBIT_DECODE_MIN` uses the
/// specialized hand-written int4/int8 decode path (`int4_matmul_m1` for block-32
/// symmetric M=1, `int8_matmul` otherwise), which ties MLAS on the
/// bandwidth-bound M=1 decode while avoiding int8 activation rounding; `m` at or
/// above the threshold routes to MLAS `MlasQNBitGemmBatch`, whose cache-tiled
/// kernels win prefill by 6--9x.
#[cfg(feature = "mlas")]
const SQNBIT_DECODE_MIN_ENV: &str = "NXRT_SQNBIT_DECODE_MIN";

/// Basis for the topology-derived int4 MatMulNBits hand-decode ↔ MLAS crossover.
/// Measurements on Sapphire Rapids (Xeon 8480C) found:
///
/// * Isolated GEMV microbench (`matmulnbits_mlas_perf`, weights L3-resident)
///   reports MLAS int4 M=1 ~1.7--1.9x faster, but that is a cache artifact.
/// * Cold, DRAM-streamed full-decode-step microbench
///   (`matmulnbits_mlas_decode_step`, one distinct 3.5 GB weight set per token,
///   32 threads) has the hand path and MLAS CompInt8 **tie** at ~90 GB/s
///   (~25 tok/s) for M=1 -- decode is memory-bandwidth bound, so the int4 path
///   choice is a wash and the hand path is preferred (no int8 rounding).
/// * End-to-end Qwen2.5-Coder-7B decode is the same (~8 tok/s) with either M=1
///   route; the 2.3x gap vs ORT/foundry is per-op Rayon fork-join and NUMA
///   locality, not the MatMulNBits kernel (see docs/performance/BENCH_MLAS_INT4_E2E.md).
///
/// The default crossover is twice the topology-derived decode worker count.
/// That preserves `m=16` on the profiled 96-way host while scaling down on
/// smaller machines where MLAS needs fewer rows to occupy the available cores.
/// Override with `NXRT_SQNBIT_DECODE_MIN`.
#[cfg(feature = "mlas")]
static SQNBIT_DECODE_MIN: OnceLock<usize> = OnceLock::new();

/// Escape hatch: set `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` to force the MLAS
/// SQNBit decode GEMV back to a single full-width `multithread=true` call
/// instead of the per-N-shard, per-worker decode-pool dispatch. MLAS's SIMD
/// column-tiling is not bit-stable across N-partition boundaries (results can
/// differ by ~1 ULP), so this restores byte-for-byte the pre-sharding output
/// for A/B parity checks. Off by default (the sharded path is far faster under
/// the persistent decode pool). Parsed once.
#[cfg(feature = "mlas")]
fn mlas_no_shard() -> bool {
    static NO_SHARD: OnceLock<bool> = OnceLock::new();
    *NO_SHARD.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_MM_MLAS_NO_SHARD").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

/// Override for the MLAS QNBit MatMulNBits route. `0`/`off` disables all MLAS
/// QNBit calls; unset or `1`/`on` enables it. On non-Apple ARM64 this makes the
/// KleidiAI-backed MLAS QNBit shard path the default for Qwen-style bits4/bits8
/// block-128 decode; set `0` to A/B against the native KAI fallback.
#[cfg(feature = "mlas")]
fn mlas_qnbit_env_override() -> Option<bool> {
    std::env::var("ONNX_GENAI_CPU_MM_MLAS_QNBIT")
        .ok()
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("off")
        })
}

#[cfg(feature = "mlas")]
fn mlas_qnbit_enabled() -> bool {
    mlas_qnbit_env_override().unwrap_or(true)
}

#[cfg(feature = "mlas")]
fn arm64_mlas_qnbit_decode_opted_in() -> bool {
    mlas_qnbit_env_override().unwrap_or(true)
}

#[cfg(feature = "mlas")]
fn sqnbit_backend_forced_mlas() -> bool {
    std::env::var("NXRT_CPU_GEMM_BACKEND")
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("mlas"))
}

/// Force the prefill (`m > 1`) MLAS SQNBit path back to the pre-fix serial
/// loop (each per-worker shard run with MLAS `multithread=true`, one after
/// another). The default parallel `(shard x m-row-block)` dispatch is
/// bit-identical to this loop (`max_ulp = 0`) and ~15x faster on a busy
/// many-core box, so this exists only as an A/B escape hatch. Off by default.
/// Parsed once.
#[cfg(feature = "mlas")]
fn mlas_prefill_serial() -> bool {
    static PREFILL_SERIAL: OnceLock<bool> = OnceLock::new();
    *PREFILL_SERIAL.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_MM_MLAS_PREFILL_SERIAL").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

/// Smallest `m` (batch·seq row count) that routes int4 MatMulNBits to MLAS
/// SQNBit; smaller `m` uses the hand int4/int8 decode path. Parsed once from
/// `NXRT_SQNBIT_DECODE_MIN`, defaulting to [`default_sqnbit_decode_min`].
#[cfg(feature = "mlas")]
fn sqnbit_decode_min() -> usize {
    *SQNBIT_DECODE_MIN.get_or_init(|| {
        let available = available_parallelism();
        resolve_decode_min(
            std::env::var(SQNBIT_DECODE_MIN_ENV).ok().as_deref(),
            available,
        )
    })
}

/// Parse the SQNBit decode crossover, falling back to
/// [`default_sqnbit_decode_min`] for absent, empty, or malformed values.
#[cfg(feature = "mlas")]
fn resolve_decode_min(raw: Option<&str>, available: usize) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or_else(|| default_sqnbit_decode_min(available))
}

/// Whether this host has a native int8 dot-product instruction behind the hand
/// int4/int8 `accuracy_level = 4` decode kernels.
///
/// The `m < sqnbit_decode_min()` short-circuit in [`Kernel::try_mlas_sqnbit`]
/// exists because the hand decode kernels beat MLAS SQNBit CompInt8 at small
/// `m` while also avoiding MLAS's one-time packing. That is only true where the
/// int8 accumulation is a single instruction:
///
/// * x86_64 needs AVX-VNNI or AVX-512-VNNI (`vpdpbusd`). Without it the AVX2
///   fallback emulates the dot with `vpmaddubsw` + `vpmaddwd` + widening adds,
///   and it loses badly. Measured on an AVX2-only AMD EPYC 9V74 against ORT
///   1.27's CPU EP, both pinned to 8 intra-op threads, int4 block-32 M=1:
///   0.762 ms on the hand path vs 0.079 ms once MLAS SQNBit takes the node --
///   a 9.6x regression, or 16.1x vs ORT instead of 1.9x.
/// * aarch64 is unconditionally true: NEON is baseline on ARM64, so
///   [`DotKernel::Neon`]/[`DotKernel::NeonDot`] always have a real dot product
///   and the previous behaviour is preserved exactly.
/// * Every other architecture runs the scalar `DotKernel`, which has no dot
///   product at all, so MLAS (when present) should take the node.
///
/// This deliberately reads the *selected* kernel rather than probing CPUID
/// directly, so `ONNX_GENAI_CPU_DOT_KERNEL`-style overrides and the test
/// harness stay consistent with what actually executes.
#[cfg(feature = "mlas")]
fn hand_int8_decode_has_native_dot() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        selected_dot_kernel().uses_vnni_int4_direct()
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

pub struct MatMulNBitsKernel {
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    /// Structural FLOPs (`2*rows*N*K`) when the activation's leading dims were
    /// static at build time; `None` otherwise (issue #995 — never fabricated).
    flops: Option<u64>,
    weight_prepacked: bool,
    constant_inputs: [bool; 5],
    weight_nk: OnceLock<Vec<f32>>,
    int8_weight: OnceLock<Int8Weight>,
    packed_int4_weight: OnceLock<PackedInt4Weight>,
    packed_int4_n16_weight: OnceLock<PackedN16SdotWeight>,
    packed_kai_qsi4_weight: OnceLock<PackedKaiSdotWeight>,
    packed_nbits_weight: OnceLock<PackedNBitsWeight>,
    packed_u8_weight: OnceLock<PackedU8Weight>,
    packed_u8_n16_weight: OnceLock<PackedN16SdotWeight>,
    packed_kai_qsi8_weight: OnceLock<PackedKaiSdotWeight>,
    #[cfg(feature = "mlas")]
    mlas_shards: OnceLock<Option<Arc<Vec<Option<MlasShard>>>>>,
    #[cfg(feature = "mlas")]
    mlas_packed: OnceLock<Option<Arc<MlasPreparedPacked>>>,
}

/// One contiguous output-column shard of an MLAS SQNBit-packed weight: columns
/// `start .. start + len` of the `[m, N]` output, prepacked so a single decode
/// worker can compute them independently of the other shards.
#[cfg(feature = "mlas")]
struct MlasShard {
    start: usize,
    len: usize,
    prepared: MlasPreparedPacked,
}

#[cfg(feature = "mlas")]
struct MlasPreparedPacked {
    packed: mlas_sys::SQNBitPackedB,
    workspace: Mutex<mlas_sys::SQNBitGemmWorkspace>,
}

#[cfg(feature = "mlas")]
impl MlasPreparedPacked {
    fn new(packed: mlas_sys::SQNBitPackedB) -> Self {
        let mut workspace = mlas_sys::SQNBitGemmWorkspace::new();
        workspace.reserve_for(&packed, 1);
        Self {
            packed,
            workspace: Mutex::new(workspace),
        }
    }
}

/// Identity of one MLAS SQNBit packed weight, shared across the shape-keyed
/// kernel instances of a single `MatMulNBits` node (#1056 packed dedup).
///
/// The executor's kernel cache is shape-keyed, so an autoregressive decoder
/// compiles two `MatMulNBits` instances for each node -- one for prefill
/// (`m > 1`) and one for decode (`m == 1`). Before this key each instance packed
/// its own full copy of the same weight, so the resident packed footprint was
/// `2x` the single-copy cost (measured: a 169-boundary model packed 169 buffers
/// on a 1-token run and 338 on a 48-token run). Because the packed buffer is a
/// pure function of the source weight bytes and the pack parameters, both
/// instances may share one allocation whenever they agree on every field here:
///
/// * **`addr`** -- the constant weight's mmap address, distinguishing weights;
/// * **`n`, `k`, `bits`, `block_size`** -- the geometry, so a same-address
///   different-shape weight (an allocator recycling a freed address) misses
///   rather than serving the wrong bytes;
/// * **`has_zero_points`** -- symmetric vs asymmetric pack (different contents);
/// * **`comp_int8`** -- the [`mlas_sys::SQNBitComputeType`] discriminant, since
///   `Int8` bakes scale/zero-point into per-block sums while `Fp32` keeps the
///   nibbles only, so the packed layouts differ.
///
/// `addr` alone is **not** an identity: an allocator recycles a freed weight's
/// address for a later same-shaped weight (the #845/#1079 hazard that cost a
/// full debugging cycle). The shape fields close the different-shape case; the
/// remaining same-address, same-shape, across-model-lifetimes window is a
/// lifetime problem, closed by [`clear_mlas_packed_caches`] on `Executor` drop
/// -- the exact boundary at which `weight_transpose::clear_all` runs. The
/// N-shard partition is *not* a field because it is a pure function of the fixed
/// process topology and `n` (see [`MatMulNBitsKernel::mlas_shard_segments`]), so
/// two instances of one node always partition identically.
#[cfg(feature = "mlas")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct MlasPackedKey {
    addr: usize,
    n: usize,
    k: usize,
    bits: usize,
    block_size: usize,
    has_zero_points: bool,
    comp_int8: bool,
}

#[cfg(feature = "mlas")]
impl MlasPackedKey {
    fn new(
        addr: usize,
        n: usize,
        k: usize,
        bits: usize,
        block_size: usize,
        has_zero_points: bool,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Self {
        Self {
            addr,
            n,
            k,
            bits,
            block_size,
            has_zero_points,
            comp_int8: matches!(comp, mlas_sys::SQNBitComputeType::Int8),
        }
    }
}

/// Process-global, weight-identity-keyed store of MLAS SQNBit packed weights,
/// so the prefill and decode instances of one node share a single packed
/// allocation instead of packing one copy each (#1056).
///
/// Two maps because the two MLAS routes hold different value types -- the
/// N-sharded decode layout (`shards`) and the single full-width layout
/// (`packed`, used by `weight_prepacked=1` and the `NO_SHARD` A/B) -- but a
/// given node only ever populates one of them, and both are keyed by the same
/// [`MlasPackedKey`]. Entries are `Arc` so a kernel-local `OnceLock` memo and
/// the store share one allocation; the byte total (`SQNBIT_PACKED_LIVE_BYTES` in
/// `mlas-sys`) counts each packed buffer once, so a shared entry is one copy.
#[cfg(feature = "mlas")]
#[derive(Default)]
struct MlasPackedCaches {
    shards: Mutex<HashMap<MlasPackedKey, Arc<Vec<Option<MlasShard>>>>>,
    packed: Mutex<HashMap<MlasPackedKey, Arc<MlasPreparedPacked>>>,
}

#[cfg(feature = "mlas")]
impl MlasPackedCaches {
    /// Return the shared N-sharded pack for `key`, building it on a miss with
    /// `build`. `build` runs only on a miss, so a decode instance whose prefill
    /// sibling already packed the weight neither re-packs nor re-times it (the
    /// pack-count halving of #1056). `build` returning `Ok(None)` means MLAS has
    /// no kernel for the shape and nothing is cached.
    fn get_or_build_shards(
        &self,
        key: MlasPackedKey,
        build: impl FnOnce() -> Result<Option<Vec<Option<MlasShard>>>>,
    ) -> Result<Option<Arc<Vec<Option<MlasShard>>>>> {
        if let Some(hit) = self
            .shards
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(Some(hit));
        }
        let Some(built) = build()? else {
            return Ok(None);
        };
        let arc = Arc::new(built);
        // A concurrent racer may have inserted an identical entry meanwhile;
        // keep whichever landed first so every reader shares one allocation.
        Ok(Some(
            self.shards
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(key)
                .or_insert(arc)
                .clone(),
        ))
    }

    /// Full-width analogue of [`Self::get_or_build_shards`].
    fn get_or_build_packed(
        &self,
        key: MlasPackedKey,
        build: impl FnOnce() -> Result<Option<MlasPreparedPacked>>,
    ) -> Result<Option<Arc<MlasPreparedPacked>>> {
        if let Some(hit) = self
            .packed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(Some(hit));
        }
        let Some(built) = build()? else {
            return Ok(None);
        };
        let arc = Arc::new(built);
        Ok(Some(
            self.packed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(key)
                .or_insert(arc)
                .clone(),
        ))
    }

    fn clear(&self) {
        self.shards
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.packed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// The process-global packed store, the **only** store production uses.
#[cfg(feature = "mlas")]
static MLAS_PACKED_GLOBAL: LazyLock<MlasPackedCaches> = LazyLock::new(MlasPackedCaches::default);

#[cfg(all(test, feature = "mlas"))]
thread_local! {
    /// Test-only, per-thread packed store. libtest runs each `#[test]` on its
    /// own fresh thread, so a thread-local store is empty at the start of every
    /// test and dropped when the test's thread ends. That gives each test a
    /// private store -- a weight address freed by one test can never serve a
    /// stale pack to another test on a different thread -- while a single test's
    /// prefill and decode kernels (which run on that one test thread) still share
    /// one entry, exactly as production does. Production never reads this; the
    /// process-global store is the only writer of any global (#983/#1033/#1079).
    static MLAS_PACKED_TEST_LOCAL: MlasPackedCaches = MlasPackedCaches::default();
}

/// Run `f` against the active packed store: the per-thread store under test, the
/// process-global store in production.
#[cfg(feature = "mlas")]
#[inline]
fn with_mlas_packed_caches<R>(f: impl FnOnce(&MlasPackedCaches) -> R) -> R {
    #[cfg(test)]
    {
        return MLAS_PACKED_TEST_LOCAL.with(|caches| f(caches));
    }
    #[cfg(not(test))]
    {
        f(&MLAS_PACKED_GLOBAL)
    }
}

/// Evict every shared MLAS SQNBit packed weight.
///
/// **Must** run when an `Executor` drops, for the same reason
/// `weight_transpose::clear_all` does: the store is keyed on `(address, shape,
/// pack params)`, which makes a stale hit impossible for a *different-shaped*
/// weight, but a later model whose mmap places a same-shaped weight at a
/// recycled address would still match. Clearing on `Executor` drop closes that
/// window and bounds the store across model lifetimes. In production only the
/// global store exists; the per-test thread-locals are cleared by their threads
/// ending, so this is a no-op for them.
#[cfg(feature = "mlas")]
pub fn clear_mlas_packed_caches() {
    MLAS_PACKED_GLOBAL.clear();
}

/// No-op stand-in when the `mlas` feature is off, so the executor's drop path
/// can call it unconditionally.
#[cfg(not(feature = "mlas"))]
pub fn clear_mlas_packed_caches() {}

/// N-tile alignment for the persistent-pool MLAS SQNBit decode shards.
///
/// MLAS's SQNBit M=1 GEMV kernels process output columns in fixed-width N-tiles
/// (four columns on the x86 AVX2/AVX-512 and ARM NEON `SQ4BitGemmM1Kernel`
/// paths). A per-worker shard boundary that splits an N-tile makes MLAS reduce
/// that tile's block-sums through its narrower remainder path, so the column
/// lands ~1 ULP off the full-width call -- enough to flip a razor-thin greedy
/// argmax tie (observed on qwen3-0.6b int4 decode). Snapping every interior
/// shard boundary to a multiple of this constant keeps every N-tile whole
/// inside one shard, so each shard reproduces the full-width tiling exactly and
/// the concatenated decode output is bit-identical to the unsharded path.
///
/// `16` is a safe multiple of the 4-wide SQNBit M=1 N-tile on every supported
/// ISA (all such tiles are powers of two <= 16, so they divide 16); the tiny
/// (< 16 column) load imbalance it introduces is negligible for decode's large
/// projection widths.
#[cfg(feature = "mlas")]
const MLAS_SQNBIT_DECODE_SHARD_ALIGN: usize = 16;

struct Int8Weight {
    values: Vec<i8>,
    scales: Vec<f32>,
    block_sums: Vec<i32>,
}

struct PackedInt4Weight {
    values: Vec<u8>,
    scales: Vec<f32>,
}

const N16_SDOT_OUTPUTS: usize = 16;
const N16_SDOT_K_GROUP: usize = 4;

/// Tile-major signed weights for ARM dot-product decode.
///
/// Layout: `[ceil(N/16), k_blocks, block_size/4, 16 outputs, 4 K lanes]`.
/// Each four-output group can be loaded as one 16-byte vector and consumed by
/// one `sdot`, leaving the four int32 lanes as four output columns.
struct PackedN16SdotWeight {
    values: Vec<i8>,
    scales: Vec<f32>,
    zero_point_offsets: Vec<i16>,
}

const KAI_SDOT_OUTPUTS: usize = 16;
const KAI_SDOT_K_GROUP: usize = 4;

/// KleidiAI-inspired ARM decode prepack. Int4 weights stay packed (two
/// nibbles per byte); int8 weights stay one byte per weight centered around
/// 128. Layout: `[ceil(N/16), k_blocks, block_size/4, 16 outputs, payload]`,
/// where payload is 2 bytes for qsi4 and 4 bytes for qsi8.
struct PackedKaiSdotWeight {
    bits: usize,
    values: Vec<u8>,
    scales: Vec<f32>,
    rhs_sums: Vec<i32>,
    zero_point_offsets: Vec<i16>,
}

#[cfg(test)]
static INT4_DIRECT_M1_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static N16_SDOT_M1_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static KAI_SDOT_M1_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Serialises every test that reads a dispatch-probe counter against every
/// test that can increment one.
///
/// The probe counters above are process-global while `cargo test` runs test
/// functions on parallel threads, so a reachability test can otherwise observe
/// *another* test's dispatch between its own before/after reads and conclude
/// its kernel took a route it never took. That is not hypothetical: several
/// tests call `kai_sdot_matmul_m1` directly, bypassing the route gate that
/// makes the counter meaningful, and on lanes where the route is enabled a
/// plain `execute` reaches it too.
///
/// Poisoning is deliberately ignored: a panic in one probe test must surface
/// as that test's own failure, not as a cascade of unrelated lock errors.
#[cfg(test)]
static DISPATCH_PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(all(test, feature = "mlas"))]
static MLAS_SQNBIT_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Positive-proof counters for the zero-copy borrowed int4 decode path (#979).
/// Split by symmetry so a test can assert the *symmetric* branch is the one that
/// executed, not merely that some path avoided the resident `weight_nk` f32
/// cache. Incremented once per `execute` that routes into
/// [`borrowed_affine_int4_matmul`].
#[cfg(test)]
static BORROWED_INT4_SYMMETRIC_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static BORROWED_INT4_ASYMMETRIC_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);
/// Standard ONNX row-major packed NBits weight with affine block metadata.
///
/// This preserves the wire layout instead of expanding it to f32. Direct
/// kernels obtain [`PackedNBitsRow`] and [`PackedNBitsBlock`] views so packed
/// value, scale, and zero-point indexing is shared across bit widths.
struct PackedNBitsWeight {
    values: Vec<u8>,
    scales: Vec<f32>,
    zero_points: Option<Vec<u8>>,
}

#[derive(Clone, Copy)]
enum BorrowedScales<'a> {
    F32(&'a [f32]),
    F16(&'a [half::f16]),
    Bf16(&'a [half::bf16]),
}

impl BorrowedScales<'_> {
    #[inline]
    fn get(&self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::F16(values) => values[index].to_f32(),
            Self::Bf16(values) => values[index].to_f32(),
        }
    }
}

#[derive(Clone, Copy)]
struct NBitsLayout {
    bits: usize,
    block_size: usize,
}

impl NBitsLayout {
    #[inline]
    fn mask(self) -> u8 {
        if self.bits == 8 {
            u8::MAX
        } else {
            (1u8 << self.bits) - 1
        }
    }

    #[inline]
    fn packed_block_size(self) -> usize {
        self.block_size * self.bits / 8
    }

    #[inline]
    fn zero_point_row_size(self, block_count: usize) -> usize {
        (block_count * self.bits).div_ceil(8)
    }

    #[inline]
    fn values_per_byte(self) -> usize {
        8 / self.bits
    }

    #[inline]
    fn unpack_byte(self, packed: u8, index: usize) -> u8 {
        (packed >> (index * self.bits)) & self.mask()
    }

    #[inline]
    fn unpack(self, packed: &[u8], index: usize) -> u8 {
        let values_per_byte = self.values_per_byte();
        self.unpack_byte(packed[index / values_per_byte], index % values_per_byte)
    }

    #[inline]
    fn zero_point(self, zero_points: Option<&[u8]>, block: usize) -> u8 {
        zero_points.map_or(1u8 << (self.bits - 1), |points| self.unpack(points, block))
    }
}

struct PackedNBitsRow<'a> {
    values: &'a [u8],
    scales: &'a [f32],
    zero_points: Option<&'a [u8]>,
    layout: NBitsLayout,
}

impl<'a> PackedNBitsRow<'a> {
    #[inline]
    fn block(&self, block: usize) -> PackedNBitsBlock<'a> {
        let packed_block_size = self.layout.packed_block_size();
        let start = block * packed_block_size;
        PackedNBitsBlock {
            values: &self.values[start..start + packed_block_size],
            scale: self.scales[block],
            zero_point: self.layout.zero_point(self.zero_points, block),
            layout: self.layout,
        }
    }

    #[inline]
    fn dequantized_value(&self, depth: usize) -> f32 {
        let block = depth / self.layout.block_size;
        self.block(block)
            .dequantized_value(depth % self.layout.block_size)
    }
}

struct PackedNBitsBlock<'a> {
    values: &'a [u8],
    scale: f32,
    zero_point: u8,
    layout: NBitsLayout,
}

impl PackedNBitsBlock<'_> {
    #[inline]
    fn dequantized_packed_value(&self, packed: u8, index: usize) -> f32 {
        let quantized = self.layout.unpack_byte(packed, index);
        (quantized as f32 - self.zero_point as f32) * self.scale
    }

    #[inline]
    fn dequantized_value(&self, within_block: usize) -> f32 {
        let values_per_byte = self.layout.values_per_byte();
        self.dequantized_packed_value(
            self.values[within_block / values_per_byte],
            within_block % values_per_byte,
        )
    }
}

impl PackedNBitsWeight {
    fn row(&self, output: usize, block_count: usize, layout: NBitsLayout) -> PackedNBitsRow<'_> {
        let packed_row_size = block_count * layout.packed_block_size();
        let packed_start = output * packed_row_size;
        let scale_start = output * block_count;
        let zero_point_row_size = layout.zero_point_row_size(block_count);
        let zero_points = self.zero_points.as_ref().map(|points| {
            let start = output * zero_point_row_size;
            &points[start..start + zero_point_row_size]
        });
        PackedNBitsRow {
            values: &self.values[packed_start..packed_start + packed_row_size],
            scales: &self.scales[scale_start..scale_start + block_count],
            zero_points,
            layout,
        }
    }
}

/// Dense `u8` weight for the 8-bit `MatMulNBits` decode (`m == 1`) path.
///
/// Unlike [`weight_nk`](MatMulNBitsKernel::weight_nk) — which fully expands each
/// weight to `f32` (4 bytes/elem) — this keeps the quantized weight at one byte
/// per element and dequantizes on the fly inside the GEMV, cutting the weight
/// memory traffic (which dominates decode) ~4x while keeping the activations in
/// `f32` so the result stays full-precision. Rows are `[N, K]` (`values`), with
/// one `scale` and one pre-scaled zero point (`scale * zero_point`) per K block.
struct PackedU8Weight {
    values: Vec<u8>,
    scales: Vec<f32>,
    scaled_zero_points: Vec<f32>,
}

pub struct MatMulNBitsFactory;

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if !matches!(bits, 2 | 4 | 8) {
            return Err(error(format!(
                "MatMulNBits CPU supports bits in {{2, 4, 8}}, got bits={bits}. Why: other packed \
                 widths do not have a validated dequantization path. How to fix: export bits=2, \
                 bits=4, or bits=8, or select another execution provider"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if !matches!(weight_prepacked, 0 | 1) {
            return Err(error(format!(
                "weight_prepacked must be 0 (standard ONNX layout) or 1 (MLAS SQNBit packed layout), got {weight_prepacked}"
            )));
        }
        #[cfg(not(feature = "mlas"))]
        if weight_prepacked == 1 {
            return Err(error(
                "weight_prepacked=1 requires the onnx-runtime-ep-cpu 'mlas' feature",
            ));
        }
        let block_size = required_positive_attr(node, "block_size")?;
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }

        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(|value| value.as_int())
            .unwrap_or(0);

        // Structural FLOPs: the activation `A` is `[..leading.., K]`; the GEMM
        // does `2*rows*N*K` multiply-adds where `rows` is the product of the
        // leading dims. When the shape is dynamic we report `None` rather than
        // guess a token count (issue #995 constraint 2).
        let rows = input_shapes
            .first()
            .and_then(|a| super::flops::leading_rows(a));
        let flops = super::flops::matmul_nbits_flops(rows, n as u64, k as u64);

        Ok(Box::new(MatMulNBitsKernel {
            k,
            n,
            bits: bits as usize,
            block_size,
            accuracy_level,
            flops,
            weight_prepacked: weight_prepacked == 1,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
            int8_weight: OnceLock::new(),
            packed_int4_weight: OnceLock::new(),
            packed_int4_n16_weight: OnceLock::new(),
            packed_kai_qsi4_weight: OnceLock::new(),
            packed_nbits_weight: OnceLock::new(),
            packed_u8_weight: OnceLock::new(),
            packed_u8_n16_weight: OnceLock::new(),
            packed_kai_qsi8_weight: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_shards: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_packed: OnceLock::new(),
        }))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MatMulNBits", inputs, outputs, 3, 6, 1)?;
        require_float_compute_dtype("A", inputs[0].dtype)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_float_compute_dtype("scales", inputs[2].dtype)?;
        require_float_compute_dtype("Y", outputs[0].dtype)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            let zp_blob_size = (k_blocks * self.bits).div_ceil(8);
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_blob_size)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
        }
        if self.weight_prepacked {
            #[cfg(feature = "mlas")]
            {
                if group_indices.is_some() {
                    return Err(error(
                        "weight_prepacked=1 does not support g_idx because MLAS SQNBit packed weights use contiguous K blocks",
                    ));
                }
                let comp = self.mlas_compute_type();
                let expected = mlas_sys::sqnbit_packed_b_size(
                    self.n,
                    self.k,
                    self.bits,
                    self.block_size,
                    zero_points.is_some(),
                    comp,
                )
                .ok_or_else(|| {
                    error(format!(
                        "weight_prepacked=1 is unavailable for bits={}, block_size={}, accuracy_level={} on this CPU",
                        self.bits, self.block_size, self.accuracy_level
                    ))
                })?;
                let actual = numel(inputs[1].shape);
                if actual != expected {
                    return Err(error(format!(
                        "prepacked B must contain exactly {expected} bytes for N={}, K={}, bits={}, block_size={}, accuracy_level={}, got shape {:?} ({actual} bytes)",
                        self.n,
                        self.k,
                        self.bits,
                        self.block_size,
                        self.accuracy_level,
                        inputs[1].shape
                    )));
                }
            }
        } else {
            require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        }

        let bias = if let Some(bias) = optional_input(inputs, 5) {
            require_float_compute_dtype("bias", bias.dtype)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_compute_f32(bias)?)
        } else {
            None
        };

        let can_prepack = self.constant_inputs[1]
            && self.constant_inputs[2]
            && zero_points.is_none_or(|_| self.constant_inputs[3])
            && group_indices.is_none_or(|_| self.constant_inputs[4]);
        let activations = mm_profile::time_widen(|| compute_activations_cow(&inputs[0]))?;
        let m = numel(&a_shape[..a_shape.len() - 1]);
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let mut flops = (m as u64)
                .saturating_mul(self.n as u64)
                .saturating_mul(self.k as u64)
                .saturating_mul(2);
            if bias.is_some() {
                flops = flops.saturating_add((m as u64).saturating_mul(self.n as u64));
            }
            flops
        });
        let result_len = m * self.n;
        let direct_result = outputs[0].dtype == DataType::Float32
            && outputs[0].is_contiguous()
            && outputs[0].device.is_host_accessible();
        let mut owned_result;
        let result: &mut [f32] = if direct_result {
            // SAFETY: the executor provides an exclusive, in-bounds contiguous
            // output buffer whose validated shape contains `result_len` f32s.
            unsafe { std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<f32>(), result_len) }
        } else {
            owned_result = vec![0.0f32; result_len];
            &mut owned_result
        };
        let dot_kernel = selected_dot_kernel();
        if self.bits == 4
            && self.accuracy_level == 0
            && !self.weight_prepacked
            && group_indices.is_none()
            && !self.mlas_sqnbit_owns_fp32_compute(can_prepack, zero_points.is_some())
            && let Some(packed) = contiguous_host_slice::<u8>(&inputs[1])
            && let Some(scales) = borrowed_scales(&inputs[2])
            && let Some(borrowed_zero_points) = borrow_optional_int4_zero_points(zero_points)
        {
            // Gate on *symmetry* explicitly, not on "a zero_points input happens
            // to exist". Symmetric int4 (no zero_points) has the implicit
            // midpoint 8 and is mathematically simpler than the asymmetric case,
            // yet it used to fall past this zero-copy path all the way to the
            // resident f32 `weight_nk` cache (~8x the file size in RAM). It now
            // borrows the packed int4 in place like the asymmetric case, using
            // `None` zero points to mean the implicit midpoint (see #979).
            #[cfg(test)]
            if borrowed_zero_points.is_none() {
                BORROWED_INT4_SYMMETRIC_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            } else {
                BORROWED_INT4_ASYMMETRIC_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            with_decode_pool(|| {
                #[cfg(target_arch = "x86_64")]
                if borrowed_int4_nblock_enabled()
                    && !matches!(dot_kernel, DotKernel::Scalar)
                    && self.block_size.is_multiple_of(32)
                {
                    borrowed_affine_int4_matmul_nblock(
                        &activations,
                        packed,
                        scales,
                        borrowed_zero_points,
                        bias.as_deref(),
                        result,
                        m,
                        self.k,
                        self.n,
                        self.block_size,
                    );
                    return;
                }
                borrowed_affine_int4_matmul(
                    &activations,
                    packed,
                    scales,
                    borrowed_zero_points,
                    bias.as_deref(),
                    result,
                    m,
                    self.k,
                    self.n,
                    self.block_size,
                    dot_kernel,
                );
            })?;
            return if direct_result {
                Ok(())
            } else {
                write_compute_f32(&mut outputs[0], result)
            };
        }
        #[cfg(feature = "mlas")]
        {
            if let Some(()) = self.try_mlas_sqnbit(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                can_prepack,
                &activations,
                m,
                bias.as_deref(),
                result,
            )? {
                let out = if direct_result {
                    Ok(())
                } else {
                    mm_profile::time_narrow(|| write_compute_f32(&mut outputs[0], result))
                };
                mm_profile::tick();
                return out;
            }
        }
        if self.bits == 2 && !self.weight_prepacked && group_indices.is_none() {
            let owned_weight;
            let packed_weight = if can_prepack {
                if let Some(weight) = self.packed_nbits_weight.get() {
                    weight
                } else {
                    let weight = self.prepack_nbits_weight(&inputs[1], &inputs[2], zero_points)?;
                    let weight = numa_place_nbits(weight, self.n);
                    let _ = self.packed_nbits_weight.set(weight);
                    self.packed_nbits_weight
                        .get()
                        .expect("constant MatMulNBits packed weight was just initialized")
                }
            } else {
                let built = self.prepack_nbits_weight(&inputs[1], &inputs[2], zero_points)?;
                owned_weight = numa_place_nbits(built, self.n);
                &owned_weight
            };
            if m == 1 {
                with_decode_pool(|| {
                    packed_nbits_gemv(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.bits,
                        self.block_size,
                    );
                })?;
            } else {
                packed_nbits_gemm(
                    &activations,
                    packed_weight,
                    result,
                    m,
                    self.k,
                    self.n,
                    self.bits,
                    self.block_size,
                );
            }
        } else if self.bits == 4
            && self.accuracy_level == 4
            && m == 1
            && group_indices.is_none()
            && dot_kernel.supports_int4_direct(self.block_size, zero_points.is_some())
        {
            if dot_kernel.uses_kai_sdot_direct(self.bits, self.block_size) {
                let owned_weight;
                let packed_weight = if can_prepack {
                    if let Some(weight) = self.packed_kai_qsi4_weight.get() {
                        weight
                    } else {
                        let weight =
                            self.prepack_kai_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                        let weight = numa_place_kai_sdot(weight, self.n);
                        let _ = self.packed_kai_qsi4_weight.set(weight);
                        self.packed_kai_qsi4_weight
                            .get()
                            .expect("constant MatMulNBits KAI qsi4 weight was just initialized")
                    }
                } else {
                    let built =
                        self.prepack_kai_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                    owned_weight = numa_place_kai_sdot(built, self.n);
                    &owned_weight
                };
                with_decode_pool(|| {
                    kai_sdot_matmul_m1(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            } else if dot_kernel.uses_n16_sdot_direct() {
                let owned_weight;
                let packed_weight = if can_prepack {
                    if let Some(weight) = self.packed_int4_n16_weight.get() {
                        weight
                    } else {
                        let weight =
                            self.prepack_n16_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                        let weight = numa_place_n16_sdot(weight, self.n);
                        let _ = self.packed_int4_n16_weight.set(weight);
                        self.packed_int4_n16_weight
                            .get()
                            .expect("constant MatMulNBits N16 int4 weight was just initialized")
                    }
                } else {
                    let built =
                        self.prepack_n16_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                    owned_weight = numa_place_n16_sdot(built, self.n);
                    &owned_weight
                };
                with_decode_pool(|| {
                    n16_sdot_matmul_m1(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            } else {
                let owned_weight;
                let packed_weight = if can_prepack {
                    if let Some(weight) = self.packed_int4_weight.get() {
                        weight
                    } else {
                        let weight = PackedInt4Weight {
                            values: to_dense_bytes(&inputs[1])?,
                            scales: to_dense_compute_f32(&inputs[2])?,
                        };
                        let weight = numa_place_int4(weight, self.n);
                        let _ = self.packed_int4_weight.set(weight);
                        self.packed_int4_weight
                            .get()
                            .expect("constant MatMulNBits packed int4 weight was just initialized")
                    }
                } else {
                    let built = PackedInt4Weight {
                        values: to_dense_bytes(&inputs[1])?,
                        scales: to_dense_compute_f32(&inputs[2])?,
                    };
                    owned_weight = numa_place_int4(built, self.n);
                    &owned_weight
                };
                with_decode_pool(|| {
                    int4_matmul_m1(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            }
        } else if self.bits == 4 && self.accuracy_level == 4 && group_indices.is_none() {
            let owned_weight;
            let int8_weight = if can_prepack {
                if let Some(weight) = self.int8_weight.get() {
                    weight
                } else {
                    let weight = self.prepack_int8_weight(&inputs[1], &inputs[2], zero_points)?;
                    let weight = numa_place_int8(weight, self.n);
                    let _ = self.int8_weight.set(weight);
                    self.int8_weight
                        .get()
                        .expect("constant MatMulNBits int8 prepack was just initialized")
                }
            } else {
                let built = self.prepack_int8_weight(&inputs[1], &inputs[2], zero_points)?;
                owned_weight = numa_place_int8(built, self.n);
                &owned_weight
            };
            if m == 1 {
                with_decode_pool(|| {
                    int8_matmul(
                        &activations,
                        int8_weight,
                        result,
                        m,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            } else {
                #[cfg(target_arch = "x86_64")]
                {
                    if m >= amx::AMX_PREFILL_MIN_M
                        && amx::amx_block_size_supported(self.block_size)
                        && amx::amx_int8_available()
                    {
                        amx::int8_matmul_amx(
                            &activations,
                            int8_weight,
                            result,
                            m,
                            self.k,
                            self.n,
                            self.block_size,
                        );
                    } else {
                        int8_matmul(
                            &activations,
                            int8_weight,
                            result,
                            m,
                            self.k,
                            self.n,
                            self.block_size,
                            dot_kernel,
                        );
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    int8_matmul(
                        &activations,
                        int8_weight,
                        result,
                        m,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                }
            }
        } else if self.bits == 8 && m == 1 && group_indices.is_none() {
            // 8-bit decode GEMV: keep the weight at one byte per element and
            // dequantize on the fly, instead of pre-expanding it to f32 (4x the
            // memory traffic) as the generic `weight_nk` path below does.
            //
            // Precision gate: the two SDOT routes below are *not* full
            // precision -- `kai_sdot_matmul_m1` quantizes the activations via
            // `quantize_activation_qai8dxp` and `n16_sdot_u8_i16_matmul_m1` via
            // `quantize_activation_signed`, both int8, costing ~1e-3 relative
            // RMSE. That is ONNX CompInt8, so both are gated on
            // `accuracy_level == 4`, matching the 4-bit SDOT routes above and
            // the 0/1 == fp32 convention used throughout this kernel. They
            // previously had no `accuracy_level` gate at all while
            // `arm64_kai_sdot_direct_enabled` defaults *on* for non-Apple
            // aarch64, so `accuracy_level = 0` 8-bit decode silently ran
            // CompInt8 there. The final `else` does not quantize the
            // activations to int8: it keeps them at f32 or, when
            // `eight_bit_int16_activation()` is on *and the model asked for
            // reduced precision*, at int16. int16 activations are far more
            // accurate than int8 but they are NOT fp32, so they are gated too
            // (`reduced_precision_activation_allowed`).
            let int8_compute_allowed = self.accuracy_level == 4;
            if int8_compute_allowed && dot_kernel.uses_kai_sdot_direct(self.bits, self.block_size) {
                let owned_weight;
                let packed_weight = if can_prepack {
                    if let Some(weight) = self.packed_kai_qsi8_weight.get() {
                        weight
                    } else {
                        let weight =
                            self.prepack_kai_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                        let weight = numa_place_kai_sdot(weight, self.n);
                        let _ = self.packed_kai_qsi8_weight.set(weight);
                        self.packed_kai_qsi8_weight
                            .get()
                            .expect("constant MatMulNBits KAI qsi8 weight was just initialized")
                    }
                } else {
                    let built =
                        self.prepack_kai_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                    owned_weight = numa_place_kai_sdot(built, self.n);
                    &owned_weight
                };
                with_decode_pool(|| {
                    kai_sdot_matmul_m1(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            } else if int8_compute_allowed
                && dot_kernel.uses_n16_sdot_direct()
                && self.block_size == 128
                && activation_quant_group() == 32
            {
                let owned_weight;
                let packed_weight = if can_prepack {
                    if let Some(weight) = self.packed_u8_n16_weight.get() {
                        weight
                    } else {
                        let weight =
                            self.prepack_n16_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                        let weight = numa_place_n16_sdot(weight, self.n);
                        let _ = self.packed_u8_n16_weight.set(weight);
                        self.packed_u8_n16_weight
                            .get()
                            .expect("constant MatMulNBits N16 bits8 weight was just initialized")
                    }
                } else {
                    let built =
                        self.prepack_n16_sdot_weight(&inputs[1], &inputs[2], zero_points)?;
                    owned_weight = numa_place_n16_sdot(built, self.n);
                    &owned_weight
                };
                with_decode_pool(|| {
                    n16_sdot_u8_i16_matmul_m1(
                        &activations,
                        packed_weight,
                        result,
                        self.k,
                        self.n,
                        self.block_size,
                        dot_kernel,
                    );
                })?;
            } else {
                let owned_weight;
                let weight_u8 = if can_prepack {
                    if let Some(weight) = self.packed_u8_weight.get() {
                        weight
                    } else {
                        let weight = self.prepack_u8_weight(&inputs[1], &inputs[2], zero_points)?;
                        let weight = numa_place_u8(weight, self.n);
                        let _ = self.packed_u8_weight.set(weight);
                        self.packed_u8_weight
                            .get()
                            .expect("constant MatMulNBits u8 prepack was just initialized")
                    }
                } else {
                    let built = self.prepack_u8_weight(&inputs[1], &inputs[2], zero_points)?;
                    owned_weight = numa_place_u8(built, self.n);
                    &owned_weight
                };
                let int16_activation = eight_bit_int16_activation()
                    && reduced_precision_activation_allowed(self.accuracy_level);
                with_decode_pool(|| {
                    if int16_activation {
                        // int16-activation fast path: quantize the activation to
                        // int16 per K-block and reduce against the u8 weight with a
                        // SIMD int16 dot product, replacing the widening u8->f32
                        // FMA with a denser int16 madd. This *is* a reduced-precision
                        // compute type, so it is gated on `accuracy_level` -- see
                        // `reduced_precision_activation_allowed`.
                        gemv_nk_u8_i16(
                            &activations,
                            &weight_u8.values,
                            &weight_u8.scales,
                            &weight_u8.scaled_zero_points,
                            result,
                            self.k,
                            self.n,
                            self.block_size,
                        );
                    } else {
                        gemv_nk_u8(
                            &activations,
                            &weight_u8.values,
                            &weight_u8.scales,
                            &weight_u8.scaled_zero_points,
                            result,
                            self.k,
                            self.n,
                            self.block_size,
                        );
                    }
                })?;
            }
        } else if m == 1 {
            let owned_weight;
            let weight_nk = if can_prepack && resident_dequant_f32_cache_enabled() {
                if let Some(weight) = self.weight_nk.get() {
                    weight
                } else {
                    let weight =
                        mm_profile::time_prepack("dequant-nk", self.nbits_weight_bytes(), || {
                            self.dequantize_weight(
                                &inputs[1],
                                &inputs[2],
                                zero_points,
                                group_indices,
                                WeightLayout::Nk,
                            )
                        })?;
                    let weight = numa_place_nk(weight, self.n);
                    let _ = self.weight_nk.set(weight);
                    self.weight_nk
                        .get()
                        .expect("constant MatMulNBits prepack was just initialized")
                }
            } else {
                let built = self.dequantize_weight(
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    group_indices,
                    WeightLayout::Nk,
                )?;
                owned_weight = numa_place_nk(built, self.n);
                &owned_weight
            };
            with_decode_pool(|| {
                gemv_nk(&activations, weight_nk, result, self.k, self.n);
            })?;
        } else {
            // Prefill / batched (m > 1) fallback for cases not owned by the
            // 4-bit int8 path above (notably 8-bit weights, and generic
            // accuracy levels or grouped quantization). Dequantizing straight
            // into the transposed `Kn` layout the dense GEMM wants is a
            // strided-scatter transpose (each K step writes at stride N), which
            // thrashes cache and dominated prefill wall time (~95% of the
            // MatMulNBits cost, measured ~45 ms/node on Qwen3-0.6B). Instead,
            // on the MLAS backend dequantize once into the natural, contiguous
            // `Nk` layout -- cached in the same `weight_nk` slot the m==1
            // generic path uses, so constant weights pay the dequant once -- and
            // let MLAS's cache-tiled `sgemm` consume it transposed (`trans_b`,
            // `ldb = k`). MLAS then streams each weight row once and reuses it
            // across all m activation rows, the amortization a per-row GEMV
            // lacks. Non-MLAS hosts keep the previous direct-`Kn` dense path.
            let used_fast_nt = self.try_prefill_mlas_nt(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                can_prepack,
                &activations,
                m,
                result,
            )?;
            if !used_fast_nt {
                let weight_kn =
                    mm_profile::time_prepack("dequant-kn", self.nbits_weight_bytes(), || {
                        self.dequantize_weight(
                            &inputs[1],
                            &inputs[2],
                            zero_points,
                            group_indices,
                            WeightLayout::Kn,
                        )
                    })?;
                gemm(&activations, &weight_kn, result, m, self.k, self.n)?;
            }
        }
        if let Some(bias) = bias {
            for row in result.chunks_exact_mut(self.n) {
                for (value, bias) in row.iter_mut().zip(&bias) {
                    *value += bias;
                }
            }
        }
        if direct_result {
            Ok(())
        } else {
            write_compute_f32(&mut outputs[0], result)
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
    }
}

impl MatMulNBitsKernel {
    fn prepack_nbits_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<PackedNBitsWeight> {
        Ok(PackedNBitsWeight {
            values: to_dense_bytes(packed)?,
            scales: to_dense_compute_f32(scales)?,
            zero_points: zero_points.map(to_dense_bytes).transpose()?,
        })
    }

    /// Route the blockwise-quantized MatMul through MLAS's `MlasQNBitGemmBatch`
    /// when the `mlas` feature is on, the backend resolves to
    /// [`CpuBackend::Mlas`], and the case is one MLAS supports. Returns
    /// `Ok(Some(()))` when it filled `result` (the caller writes output and
    /// returns), or `Ok(None)` to signal a fall back to the hand-written paths.
    ///
    /// Fallback cases (return `Ok(None)`): the decode regime (`m` below the
    /// crossover [`sqnbit_decode_min`]) when the hand path is a *fast*
    /// specialized int4/int8 route (`bits == 4 && accuracy_level == 4`), which
    /// ties MLAS on bandwidth-bound M=1 while avoiding int8 activation rounding;
    /// `accuracy_level == 4` when the resolved [`CpuBackend`] is not MLAS (its
    /// hand int8 path owns MatMulNBits and matches ORT's CompInt8 numerics);
    /// `bits != 4` (2-bit is owned by the direct packed hand kernels); `g_idx`
    /// is present (MLAS SQNBit has no per-row group indices); or MLAS reports no
    /// kernel is available for this shape on the host. A case whose hand path
    /// would instead fall to the slow full-f32-dequant GEMV (any
    /// `accuracy_level != 4`, e.g. the `accuracy_level = 0` "implementation's
    /// choice" that Foundry `cuda-gpu` int4 exports emit) is **not** dropped
    /// here: MLAS SQNBit (CompFp32) beats a dequantize-then-GEMM there, matching
    /// how ORT/onnxruntime-genai run those models. Bias, when present, is added
    /// by MLAS itself, so the caller's post-loop bias add is skipped on this
    /// path.
    /// Whether MLAS SQNBit's CompFp32 kernels own this `accuracy_level = 0`
    /// int4 node, so [`Kernel::execute`] must not short-circuit into the
    /// borrowed hand path first.
    ///
    /// The borrowed zero-copy int4 path (#979) fixed a memory regression (it
    /// avoids the ~8x resident f32 `weight_nk` expansion), but it also became
    /// the *first* branch in `execute`, which silently made the intended
    /// `accuracy_level = 0` -> MLAS CompFp32 route unreachable: `try_mlas_sqnbit`
    /// is only consulted after it. On x86_64 the borrowed path has no SIMD
    /// kernel at all (the vectorized block dot is `cfg(target_arch = "aarch64")`),
    /// so every `bits=4, accuracy_level=0` node ran a scalar nibble-unpack GEMV
    /// -- measured 29x-303x slower than ONNX Runtime's CPU EP on the same graph.
    ///
    /// MLAS keeps ownership only when it can pack the weight **once**:
    /// `can_prepack` means B/scales/zero-points are graph constants, so the
    /// packed buffer is built on the first call and cached for the session. With
    /// non-constant (dynamic) weights MLAS would repack on every call, which the
    /// borrowed path avoids entirely, so ownership stays with the borrowed path
    /// there. `sqnbit_packed_b_size` is MLAS's own "do I have a kernel for this
    /// shape on this CPU" probe; when it says no, the borrowed path remains the
    /// fallback.
    ///
    /// Finally, the packed buffer is a session-lifetime resident allocation
    /// (~2x the int4 bytes) held beside the mapped weight, so the memory-strategy
    /// plan governs it exactly like the resident f32 cache (#987): when the plan
    /// declines it (over budget), [`set_mlas_sqnbit_packing_enabled`] is `false`
    /// and ownership is refused so the borrowed zero-copy path stays reachable.
    #[cfg(feature = "mlas")]
    fn mlas_sqnbit_owns_fp32_compute(&self, can_prepack: bool, has_zero_points: bool) -> bool {
        mlas_qnbit_enabled()
            && mlas_sqnbit_packing_enabled()
            && can_prepack
            && mlas_sys::sqnbit_packed_b_size(
                self.n,
                self.k,
                self.bits,
                self.block_size,
                has_zero_points,
                self.mlas_compute_type(),
            )
            .is_some()
    }

    /// Without the vendored MLAS there is no SQNBit kernel to defer to, so the
    /// borrowed int4 path always owns `accuracy_level = 0`.
    #[cfg(not(feature = "mlas"))]
    fn mlas_sqnbit_owns_fp32_compute(&self, _can_prepack: bool, _has_zero_points: bool) -> bool {
        false
    }

    #[cfg(feature = "mlas")]
    #[allow(clippy::too_many_arguments)]
    fn try_mlas_sqnbit(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        can_prepack: bool,
        activations: &[f32],
        m: usize,
        bias: Option<&[f32]>,
        result: &mut [f32],
    ) -> Result<Option<()>> {
        use crate::backend::CpuBackend;

        if !mlas_qnbit_enabled() {
            return Ok(None);
        }

        let comp = self.mlas_compute_type();

        if self.weight_prepacked {
            if matches!(comp, mlas_sys::SQNBitComputeType::Int8)
                && m == 1
                && zero_points.is_some()
                && !host_supports_mlas_sqnbit_m1_asym_int8()
            {
                return Err(error(
                    "weight_prepacked=1 asymmetric CompInt8 M=1 is unavailable on this CPU because its MLAS kernel is not numerically correct",
                ));
            }

            let owned;
            let packed_weight: &Option<Arc<MlasPreparedPacked>> = if can_prepack {
                if let Some(weight) = self.mlas_packed.get() {
                    weight
                } else {
                    let weight = self.shared_mlas_prepacked(packed, scales, zero_points, comp)?;
                    let _ = self.mlas_packed.set(weight);
                    self.mlas_packed
                        .get()
                        .expect("constant prepacked MatMulNBits weight was just initialized")
                }
            } else {
                owned = self
                    .build_mlas_prepacked(packed, scales, zero_points, comp)?
                    .map(Arc::new);
                &owned
            };
            let packed_weight = packed_weight.as_ref().ok_or_else(|| {
                error(format!(
                    "weight_prepacked=1 is unavailable for bits={}, block_size={}, accuracy_level={} on this CPU",
                    self.bits, self.block_size, self.accuracy_level
                ))
            })?;
            #[cfg(all(test, feature = "mlas"))]
            MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            mm_profile::time_gemv(|| {
                self.run_mlas_prepared(packed_weight, m, activations, bias, result, true)
            })?;
            return Ok(Some(()));
        }

        if can_prepack
            && !mlas_no_shard()
            && let Some(shards) = self.mlas_shards.get()
        {
            let Some(shards) = shards.as_ref() else {
                return Ok(None);
            };
            #[cfg(all(test, feature = "mlas"))]
            MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            mm_profile::time_gemv(|| self.run_mlas_shards(shards, activations, m, bias, result));
            return Ok(Some(()));
        }

        if can_prepack
            && mlas_no_shard()
            && let Some(cached) = self.mlas_packed.get()
        {
            let Some(packed_weight) = cached.as_ref() else {
                return Ok(None);
            };
            #[cfg(all(test, feature = "mlas"))]
            MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            mm_profile::time_gemv(|| {
                self.run_mlas_prepared(packed_weight, m, activations, bias, result, true)
            })?;
            return Ok(Some(()));
        }

        let prefer_arm64_mlas_decode = prefer_arm64_mlas_qnbit_decode(
            self.bits,
            self.block_size,
            self.accuracy_level,
            m,
            group_indices.is_none(),
        );

        // Cheapest gate first: outside the ARM64 QNBit/KleidiAI route, keep the
        // fast hand int4 decode path for small `m` and avoid MLAS packing. When
        // explicitly opted in on non-Apple ARM64, the vendored MLAS QNBit path
        // can serve accuracy-4 qsi4/qsi8 decode with the same KleidiAI kernels
        // ORT uses.
        //
        // "Fast" is a claim about the *host*, not about the source: the hand
        // int4/int8 decode kernels are only competitive where the CPU has a
        // native int8 dot product to dispatch to (see
        // `hand_int8_decode_has_native_dot`). On a host without one they lose to
        // MLAS SQNBit CompInt8 by an order of magnitude, so the short-circuit
        // must not fire there.
        //
        // ...unless there is nothing to amortize MLAS's packing against.
        // Without constant weights (`can_prepack == false`) the packed buffer
        // cannot be cached in `mlas_shards`/`mlas_packed`, so MLAS would repack
        // on *every* call -- measured at 55.8 ms for a 6.4 MB int4 weight
        // (`ONNX_GENAI_PROFILE_MM=1`), against a sub-millisecond decode. Dynamic
        // weights keep the hand path regardless of ISA; a slow kernel still
        // beats repacking the whole weight per token.
        let hand_decode_is_fast = self.bits == 4
            && self.accuracy_level == 4
            && !prefer_arm64_mlas_decode
            && (!can_prepack || hand_int8_decode_has_native_dot());
        if m < sqnbit_decode_min() && hand_decode_is_fast {
            return Ok(None);
        }

        // MLAS SQNBit is a specialized blockwise-quantized kernel, distinct from
        // the dense-f32 GEMM microkernel that `CpuBackend` selects. For
        // `accuracy_level == 4` the fast hand int8/int4 paths own MatMulNBits and
        // match ORT's CompInt8 numerics, so only defer to MLAS (CompInt8) when the
        // whole GEMM backend was explicitly forced to MLAS. For every other
        // accuracy level the hand fallback is a slow full-f32-dequant GEMV, so
        // prefer MLAS SQNBit (CompFp32) whenever MLAS actually has a kernel --
        // this matches ORT/onnxruntime-genai, which treat `accuracy_level` 0/1 as
        // CompFp32 rather than dequantizing the whole weight.
        let backend_is_mlas =
            CpuBackend::auto_detect() == CpuBackend::Mlas || sqnbit_backend_forced_mlas();
        let use_mlas = backend_is_mlas || self.accuracy_level != 4 || prefer_arm64_mlas_decode;
        let supports_bits = self.bits == 4 || (prefer_arm64_mlas_decode && self.bits == 8);
        if !supports_bits || group_indices.is_some() || !use_mlas {
            return Ok(None);
        }

        // Cross-CPU correctness guard: MLAS's AVX2 M=1 CompInt8 SQNBit kernel
        // with a zero point (`SQ4BitGemmM1Kernel_CompInt8_avx2`, all block sizes)
        // is numerically broken -- it disagrees with the reference by ~46% on
        // asymmetric int4 (verified under Intel SDE `-hsw`; the AVX-512 M=1 and
        // every AVX2 M>1 kernel are correct). The `sqnbit_decode_min() >= 2`
        // crossover above already keeps the default M=1 decode on the hand int8
        // kernel, but an operator lowering `NXRT_SQNBIT_DECODE_MIN` to <= 1 would
        // otherwise expose the broken kernel, so refuse it explicitly here and
        // fall back to the (correct) hand int8 path. Asymmetric = zero points
        // present; only M=1 and only non-AVX-512 hosts are affected.
        if matches!(comp, mlas_sys::SQNBitComputeType::Int8)
            && m == 1
            && zero_points.is_some()
            && !host_supports_mlas_sqnbit_m1_asym_int8()
        {
            return Ok(None);
        }

        // Constant weights (the decode hot path) normally use the historical
        // static-SPMD shards. `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` opts into
        // ORT's full-width `MlasQNBitGemmBatch(..., multithread=true)` design,
        // where MLAS dynamically partitions N tiles across its persistent
        // work-stealing intra-op pool, for A/B profiling without changing the
        // default until correctness and throughput are proven.
        let use_static_shards = can_prepack && !mlas_no_shard();
        if use_static_shards {
            let shards = if let Some(shards) = self.mlas_shards.get() {
                shards
            } else {
                let built = self.shared_mlas_shards(packed, scales, zero_points, comp)?;
                let _ = self.mlas_shards.set(built);
                self.mlas_shards
                    .get()
                    .expect("constant MatMulNBits MLAS shards were just initialized")
            };
            let Some(shards) = shards.as_ref() else {
                return Ok(None);
            };
            #[cfg(all(test, feature = "mlas"))]
            MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            mm_profile::time_gemv(|| self.run_mlas_shards(shards, activations, m, bias, result));
            return Ok(Some(()));
        }

        if can_prepack {
            let cached = if let Some(cached) = self.mlas_packed.get() {
                cached
            } else {
                let built = self.shared_mlas_packed(packed, scales, zero_points, comp)?;
                let _ = self.mlas_packed.set(built);
                self.mlas_packed
                    .get()
                    .expect("constant MatMulNBits MLAS full-width weight was just initialized")
            };
            let Some(packed_weight) = cached.as_ref() else {
                return Ok(None);
            };
            #[cfg(all(test, feature = "mlas"))]
            MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
            mm_profile::time_gemv(|| {
                self.run_mlas_prepared(packed_weight, m, activations, bias, result, true)
            })?;
            return Ok(Some(()));
        }

        let owned = self.build_mlas_packed(packed, scales, zero_points, comp)?;
        let Some(packed_weight) = owned.as_ref() else {
            return Ok(None);
        };
        #[cfg(all(test, feature = "mlas"))]
        MLAS_SQNBIT_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
        mm_profile::time_gemv(|| {
            self.run_mlas_prepared(packed_weight, m, activations, bias, result, true)
        })?;
        Ok(Some(()))
    }

    /// Prefill / batched (`m > 1`) fast path for the dense-dequantize fallback:
    /// dequantize the constant weight once into the natural, contiguous `Nk`
    /// (`[n, k]`) layout, cache it, and run MLAS's cache-tiled `sgemm` with
    /// `trans_b` so each weight row is streamed once and reused across all `m`
    /// activation rows. Returns `Ok(true)` when it handled the GEMM, `Ok(false)`
    /// when the host GEMM backend is not MLAS (the caller then uses the previous
    /// direct-`Kn` dense path). Bit-identical to a no-transpose dense GEMM: MLAS
    /// with `trans_b` computes `C = A * B_nk^T` where `B_nk[n, k]` is the same
    /// weight the `Kn` path stores transposed.
    #[cfg(feature = "mlas")]
    #[allow(clippy::too_many_arguments)]
    fn try_prefill_mlas_nt(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        can_prepack: bool,
        activations: &[f32],
        m: usize,
        result: &mut [f32],
    ) -> Result<bool> {
        use crate::backend::CpuBackend;

        if CpuBackend::auto_detect() != CpuBackend::Mlas {
            return Ok(false);
        }

        let owned_weight;
        let weight_nk: &[f32] = if can_prepack && resident_dequant_f32_cache_enabled() {
            if let Some(weight) = self.weight_nk.get() {
                weight
            } else {
                let weight = self.dequantize_weight(
                    packed,
                    scales,
                    zero_points,
                    group_indices,
                    WeightLayout::Nk,
                )?;
                let weight = numa_place_nk(weight, self.n);
                let _ = self.weight_nk.set(weight);
                self.weight_nk
                    .get()
                    .expect("constant MatMulNBits Nk prepack was just initialized")
            }
        } else {
            let built = self.dequantize_weight(
                packed,
                scales,
                zero_points,
                group_indices,
                WeightLayout::Nk,
            )?;
            owned_weight = numa_place_nk(built, self.n);
            &owned_weight
        };

        // C[m, n] = A[m, k] * B_nk[n, k]^T. `trans_b` with `ldb = k` reads the
        // Nk weight as the transposed operand without materializing a Kn copy.
        mm_profile::time_gemv(|| {
            mlas_sys::sgemm(
                false,
                true,
                m,
                self.n,
                self.k,
                1.0,
                activations,
                self.k,
                weight_nk,
                self.k,
                0.0,
                result,
                self.n,
            );
        });
        Ok(true)
    }

    /// Non-MLAS builds have no cache-tiled dense GEMM here, so the batched
    /// fallback always uses the direct-`Kn` dense path.
    #[cfg(not(feature = "mlas"))]
    #[allow(clippy::too_many_arguments)]
    fn try_prefill_mlas_nt(
        &self,
        _packed: &TensorView,
        _scales: &TensorView,
        _zero_points: Option<&TensorView>,
        _group_indices: Option<&TensorView>,
        _can_prepack: bool,
        _activations: &[f32],
        _m: usize,
        _result: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Run a pre-partitioned MLAS SQNBit GEMV (`self.n` split into contiguous
    /// output-column shards, one per decode worker).
    ///
    /// * `m == 1` under an active persistent SPMD decode scope: broadcast the
    ///   shards to the resident workers under one barrier -- each worker runs its
    ///   own shard's GEMV serially (`multithread=false`), so the whole projection
    ///   stays on the hot decode pool with no global-Rayon fork.
    /// * Otherwise (prefill `m > 1`, or decode with no SPMD scope): run each shard
    ///   with MLAS's own tile parallelism (`multithread=true`) writing its columns
    ///   into the shared `[m, n]` output at stride `self.n`. With a single
    ///   full-width shard (no SPMD pool) this is exactly the previous one-call
    ///   `multithread=true` behaviour.
    ///
    /// Row-major N-partitioning computes each output column independently of the
    /// others, so the concatenated result is bit-identical to a single full-width
    /// GEMV regardless of how N is split or how many workers run it.
    #[cfg(feature = "mlas")]
    fn run_mlas_shards(
        &self,
        shards: &[Option<MlasShard>],
        activations: &[f32],
        m: usize,
        bias: Option<&[f32]>,
        result: &mut [f32],
    ) {
        if let Some(spmd) = (m == 1).then(spmd_decode_active).flatten() {
            spmd.dispatch_output_rows_indexed(
                result,
                MLAS_SQNBIT_DECODE_SHARD_ALIGN,
                &|global_index, start, outputs| {
                    let Some(shard) = shards.get(global_index).and_then(Option::as_ref) else {
                        return;
                    };
                    debug_assert_eq!(shard.start, start);
                    debug_assert_eq!(shard.len, outputs.len());
                    let bias = bias.map(|bias| &bias[start..start + outputs.len()]);
                    // m == 1: `outputs` is this shard's contiguous output row.
                    self.run_mlas_prepared(&shard.prepared, 1, activations, bias, outputs, false)
                        .expect("MLAS SQNBit shard GEMV must not fail");
                },
            );
            return;
        }
        let base = result.as_mut_ptr();
        let active = shards.iter().filter(|shard| shard.is_some()).count();

        // Prefill (`m > 1`) with per-worker shards: the persistent SPMD decode
        // pool splits N into one shard per worker so decode can hand each
        // resident worker a column range. That partition is *wrong* for prefill
        // if run the way decode's flat path would -- looping the shards serially
        // and letting each one fan across the pool with MLAS's own
        // `multithread=true` fires one global-Rayon fork/join per *narrow* shard
        // (N / workers columns). A prefill with `W` shards then pays `W`
        // sequential fork/joins, each spreading a sliver of work across the whole
        // machine with poor per-tile efficiency; measured ~15x slower than a
        // single dispatch (155 vs ~2300 GFLOP/s, Xeon 8480C, 48 shards). This is
        // the prefill analogue of the decode fork/join cost the persistent pool
        // removes -- so remove it the same way: issue ONE Rayon parallel pass
        // over `(shard x m-row-block)` tiles, each running `multithread=false` on
        // its own disjoint `[rows) x [cols)` output window. Row-major
        // N-partitioning (with the shipped [`MLAS_SQNBIT_DECODE_SHARD_ALIGN`]
        // boundaries) and independent M rows make every tile's outputs
        // independent, so the concatenated result is *bit-identical* to the
        // serial `multithread=true` loop and to a single full-width call
        // (verified `max_ulp = 0`). Tiling M as well as N keeps every core busy
        // when the shard count is below the host thread count, without needing a
        // second full-width packed weight. Gated on `m > 1 && active > 1`:
        // decode (`m == 1`) with no SPMD scope and the single-shard no-pool case
        // keep the original one-call path below (a single full-width
        // `multithread=true` call is already optimal there).
        if m > 1 && active > 1 && !mlas_prefill_serial() {
            let n = self.n;
            let k = self.k;
            let threads = rayon::current_num_threads().max(1);
            // Aim for roughly one tile per hardware thread: split each shard's M
            // rows into enough blocks to reach `threads` tiles total.
            let row_blocks = (threads / active).clamp(1, m);
            let rows_per_block = m.div_ceil(row_blocks);
            let live: Vec<&MlasShard> = shards.iter().flatten().collect();
            let mut tiles: Vec<(usize, usize, usize)> = Vec::with_capacity(live.len() * row_blocks);
            for (shard_index, _) in live.iter().enumerate() {
                let mut row = 0;
                while row < m {
                    let rows = rows_per_block.min(m - row);
                    tiles.push((shard_index, row, rows));
                    row += rows;
                }
            }
            struct OutputBase(*mut f32);
            // SAFETY: every tile writes a disjoint `[row_start, row_start+rows)`
            // x `[shard.start, shard.start+shard.len)` window of the single
            // `[m, n]` output, so the workers never alias.
            unsafe impl Sync for OutputBase {}
            let out = OutputBase(base);
            let out = &out;
            let live = &live;
            tiles
                .par_iter()
                .for_each(|&(shard_index, row_start, rows)| {
                    let shard = live[shard_index];
                    let bias = bias.map(|bias| &bias[shard.start..shard.start + shard.len]);
                    let activations = &activations[row_start * k..(row_start + rows) * k];
                    // SAFETY: `out.0.add(row_start * n + shard.start)` is the first
                    // element of this tile's window; the kernel writes `rows` rows at
                    // leading dimension `n`, each covering `shard.len` columns, all
                    // within `[m, n]` (`row_start + rows <= m`,
                    // `shard.start + shard.len <= n`).
                    let dst = unsafe { out.0.add(row_start * n + shard.start) };
                    unsafe {
                        mlas_sys::sqnbit_gemm_into(
                            &shard.prepared.packed,
                            rows,
                            activations,
                            bias,
                            dst,
                            n,
                            false,
                        );
                    }
                });
            return;
        }

        // Single full-width shard (no persistent pool), or `m == 1` with no
        // active decode scope: each shard writes its columns into the shared
        // `[m, n]` output at leading dimension `self.n`, using MLAS's own tile
        // parallelism. With a single full-width shard this is exactly the
        // previous one-call `multithread=true` behaviour.
        for shard in shards.iter().flatten() {
            let bias = bias.map(|bias| &bias[shard.start..shard.start + shard.len]);
            // SAFETY: shards own disjoint contiguous column ranges of a single
            // [m, self.n] row-major output; `base.add(shard.start)` is the first
            // element of this shard's columns and `(m - 1) * self.n + shard.len`
            // stays within `result` (shard.start + shard.len <= self.n).
            unsafe {
                self.run_mlas_prepared_into(
                    &shard.prepared,
                    m,
                    activations,
                    bias,
                    base.add(shard.start),
                    self.n,
                    true,
                )
                .expect("MLAS SQNBit shard GEMM must not fail");
            }
        }
    }

    #[cfg(feature = "mlas")]
    fn run_mlas_prepared(
        &self,
        prepared: &MlasPreparedPacked,
        m: usize,
        activations: &[f32],
        bias: Option<&[f32]>,
        result: &mut [f32],
        multithread: bool,
    ) -> Result<()> {
        let mut workspace = prepared
            .workspace
            .lock()
            .expect("MLAS SQNBit workspace lock poisoned");
        mlas_sys::sqnbit_gemm_with_workspace(
            &prepared.packed,
            m,
            activations,
            bias,
            result,
            &mut workspace,
            multithread,
        );
        Ok(())
    }

    #[cfg(feature = "mlas")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_mlas_prepared_into(
        &self,
        prepared: &MlasPreparedPacked,
        m: usize,
        activations: &[f32],
        bias: Option<&[f32]>,
        result: *mut f32,
        ldc: usize,
        multithread: bool,
    ) -> Result<()> {
        let mut workspace = prepared
            .workspace
            .lock()
            .expect("MLAS SQNBit workspace lock poisoned");
        unsafe {
            mlas_sys::sqnbit_gemm_into_with_workspace(
                &prepared.packed,
                m,
                activations,
                bias,
                result,
                ldc,
                &mut workspace,
                multithread,
            );
        }
        Ok(())
    }

    /// The contiguous output-column shards `self.n` is split into for the MLAS
    /// SQNBit decode path: one per persistent SPMD decode worker (so decode can
    /// dispatch a shard to each resident worker), or a single full-width shard
    /// when no persistent pool exists (preserving the one-call behaviour).
    #[cfg(feature = "mlas")]
    fn mlas_shard_segments(&self) -> Vec<(usize, usize)> {
        match crate::decode_spmd::pools() {
            Some(spmd) => spmd.output_column_segments(self.n, MLAS_SQNBIT_DECODE_SHARD_ALIGN),
            None => vec![(0, self.n)],
        }
    }

    /// Standard-layout `B` size in bytes for this node
    /// (`[N, ceil(K / block_size), block_size * bits / 8]`), i.e. the number of
    /// packed weight bytes a prepack/dequant reads. Used only to size the
    /// one-time prepack/dequant cost under `ONNX_GENAI_PROFILE_MM=1`.
    fn nbits_weight_bytes(&self) -> usize {
        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        self.n * k_blocks * blob_size
    }

    /// The shared-store identity of this node's packed weight, or `None` when the
    /// weight is not a contiguous host buffer (so it has no stable address to key
    /// on and cannot be shared). Only ever `Some` on the constant-weight
    /// (`can_prepack`) route the callers already guard, where the initializer is
    /// a contiguous mmap slice whose address is stable for the session.
    #[cfg(feature = "mlas")]
    fn mlas_packed_key(
        &self,
        packed: &TensorView,
        has_zero_points: bool,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Option<MlasPackedKey> {
        let addr = contiguous_host_slice::<u8>(packed)?.as_ptr() as usize;
        Some(MlasPackedKey::new(
            addr,
            self.n,
            self.k,
            self.bits,
            self.block_size,
            has_zero_points,
            comp,
        ))
    }

    /// The N-sharded pack for this node, shared across its prefill and decode
    /// kernel instances via the weight-identity store (#1056). The one-time pack
    /// (and its `ONNX_GENAI_PROFILE_MM` timing) runs only on the *first*
    /// instance to reach it; the sibling instance takes the cached `Arc`, so the
    /// runtime holds one packed copy per weight instead of two. Falls back to an
    /// unshared owned pack only when the weight has no stable address to key on
    /// (never on the constant route).
    #[cfg(feature = "mlas")]
    fn shared_mlas_shards(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<Arc<Vec<Option<MlasShard>>>>> {
        let build = || {
            mm_profile::time_prepack("mlas-shards", self.nbits_weight_bytes(), || {
                self.build_mlas_shards(packed, scales, zero_points, comp)
            })
        };
        match self.mlas_packed_key(packed, zero_points.is_some(), comp) {
            Some(key) => with_mlas_packed_caches(|caches| caches.get_or_build_shards(key, build)),
            None => Ok(build()?.map(Arc::new)),
        }
    }

    /// Full-width analogue of [`Self::shared_mlas_shards`] for the `NO_SHARD`
    /// A/B route.
    #[cfg(feature = "mlas")]
    fn shared_mlas_packed(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<Arc<MlasPreparedPacked>>> {
        let build = || {
            mm_profile::time_prepack("mlas-packed", self.nbits_weight_bytes(), || {
                self.build_mlas_packed(packed, scales, zero_points, comp)
            })
        };
        match self.mlas_packed_key(packed, zero_points.is_some(), comp) {
            Some(key) => with_mlas_packed_caches(|caches| caches.get_or_build_packed(key, build)),
            None => Ok(build()?.map(Arc::new)),
        }
    }

    /// Full-width analogue of [`Self::shared_mlas_shards`] for the
    /// `weight_prepacked=1` route (reconstructs an already-packed buffer).
    #[cfg(feature = "mlas")]
    fn shared_mlas_prepacked(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<Arc<MlasPreparedPacked>>> {
        let build = || self.build_mlas_prepacked(packed, scales, zero_points, comp);
        match self.mlas_packed_key(packed, zero_points.is_some(), comp) {
            Some(key) => with_mlas_packed_caches(|caches| caches.get_or_build_packed(key, build)),
            None => Ok(build()?.map(Arc::new)),
        }
    }

    /// Pack the constant int4 weight into one MLAS SQNBit shard per entry of
    /// [`Self::mlas_shard_segments`]. Returns `Ok(None)` when MLAS has no kernel
    /// for this `(bits, block_size, compute_type)` on the host (any shard failing
    /// to pack), so the caller falls back to a hand path. Empty segments (when
    /// `N` is smaller than the worker count) map to `None` entries.
    #[cfg(feature = "mlas")]
    fn build_mlas_shards(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<Vec<Option<MlasShard>>>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let zero_points = zero_points.map(to_dense_bytes).transpose()?;

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        let zp_blob_size = (k_blocks * self.bits).div_ceil(8);

        let mut shards = Vec::new();
        for (start, len) in self.mlas_shard_segments() {
            if len == 0 {
                shards.push(None);
                continue;
            }
            let packed_shard =
                &packed[start * k_blocks * blob_size..(start + len) * k_blocks * blob_size];
            let scales_shard = &scales[start * k_blocks..(start + len) * k_blocks];
            let zero_points_shard = zero_points
                .as_ref()
                .map(|zp| &zp[start * zp_blob_size..(start + len) * zp_blob_size]);
            match mlas_sys::SQNBitPackedB::new(
                len,
                self.k,
                self.bits,
                self.block_size,
                comp,
                packed_shard,
                scales_shard,
                zero_points_shard,
            ) {
                Some(packed) => shards.push(Some(MlasShard {
                    start,
                    len,
                    prepared: MlasPreparedPacked::new(packed),
                })),
                None => return Ok(None),
            }
        }
        if mm_profile::enabled() {
            eprintln!(
                "[mlas_pack] kind=shards n={} k={} block_size={} shards={} \
                 packed_b_size={:?} live_total={}",
                self.n,
                self.k,
                self.block_size,
                shards.iter().filter(|s| s.is_some()).count(),
                mlas_sys::sqnbit_packed_b_size(
                    self.n,
                    self.k,
                    self.bits,
                    self.block_size,
                    zero_points.is_some(),
                    comp,
                ),
                mlas_sys::sqnbit_packed_live_bytes(),
            );
        }
        Ok(Some(shards))
    }

    /// Pack the constant int4 weight into MLAS's SQNBit layout, or `None` when
    /// MLAS has no kernel for this `(bits, block_size, compute_type)` on the
    /// host. The ONNX `B`/scales/zero-point bytes map directly onto MLAS's pack
    /// inputs; an absent zero point defaults to the shared int4 midpoint (8).
    #[cfg(feature = "mlas")]
    fn build_mlas_packed(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<MlasPreparedPacked>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let zero_points = zero_points.map(to_dense_bytes).transpose()?;
        Ok(mlas_sys::SQNBitPackedB::new(
            self.n,
            self.k,
            self.bits,
            self.block_size,
            comp,
            &packed,
            &scales,
            zero_points.as_deref(),
        )
        .map(MlasPreparedPacked::new))
    }

    #[cfg(feature = "mlas")]
    fn mlas_compute_type(&self) -> mlas_sys::SQNBitComputeType {
        if self.accuracy_level == 4 {
            mlas_sys::SQNBitComputeType::Int8
        } else {
            // accuracy_level is a minimum compute-precision hint. MLAS exposes
            // fp32 and int8 SQNBit paths here, so fp16/bf16 requests are safely
            // upgraded to fp32.
            mlas_sys::SQNBitComputeType::Fp32
        }
    }

    #[cfg(feature = "mlas")]
    fn build_mlas_prepacked(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<MlasPreparedPacked>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let zero_points = zero_points.map(to_dense_bytes).transpose()?;
        Ok(mlas_sys::SQNBitPackedB::from_prepacked(
            self.n,
            self.k,
            self.bits,
            self.block_size,
            comp,
            &packed,
            &scales,
            zero_points.as_deref(),
        )
        .map(MlasPreparedPacked::new))
    }

    fn prepack_int8_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<Int8Weight> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        debug_assert_eq!(self.bits, 4);
        let blob_size = self.block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let padded_k = k_blocks * self.block_size;
        let mut values = vec![0i8; self.n * padded_k];
        let mut block_sums = vec![0i32; self.n * k_blocks];

        for output in 0..self.n {
            for block in 0..k_blocks {
                let zero_point = packed_zero_points.as_ref().map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + block / 2];
                    if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                let block_start = block * self.block_size;
                let valid = self.k.saturating_sub(block_start).min(self.block_size);
                let packed_start = (output * k_blocks + block) * blob_size;
                let values_start = output * padded_k + block_start;
                let mut sum = 0i32;
                for offset in 0..valid {
                    let byte = packed[packed_start + offset / 2];
                    let quantized = if offset.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    let value = quantized as i8 - zero_point as i8;
                    values[values_start + offset] = value;
                    sum += value as i32;
                }
                block_sums[output * k_blocks + block] = sum;
            }
        }

        Ok(Int8Weight {
            values,
            scales,
            block_sums,
        })
    }

    fn prepack_n16_sdot_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<PackedN16SdotWeight> {
        debug_assert!(matches!(self.bits, 4 | 8));
        debug_assert!(self.block_size.is_multiple_of(N16_SDOT_K_GROUP));
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        Ok(prepack_n16_sdot_from_bytes(
            &packed,
            scales,
            packed_zero_points.as_deref(),
            self.n,
            self.k,
            self.bits,
            self.block_size,
        ))
    }

    fn prepack_kai_sdot_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<PackedKaiSdotWeight> {
        debug_assert!(matches!(self.bits, 4 | 8));
        debug_assert!(self.block_size.is_multiple_of(KAI_SDOT_K_GROUP));
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        Ok(prepack_kai_sdot_from_bytes(
            &packed,
            scales,
            packed_zero_points.as_deref(),
            self.n,
            self.k,
            self.bits,
            self.block_size,
        ))
    }

    /// Prepack an 8-bit `MatMulNBits` weight into a dense `[N, K]` `u8` buffer
    /// with per-block `scale` and pre-scaled zero point (`scale * zero_point`),
    /// for the on-the-fly-dequant decode GEMV ([`gemv_nk_u8`]).
    ///
    /// Keeps one byte per weight (vs. 4 for a fully dequantized `f32` weight),
    /// which is the memory that dominates 8-bit CPU decode.
    fn prepack_u8_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<PackedU8Weight> {
        debug_assert_eq!(self.bits, 8);
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        // 8-bit blob is one byte per weight; zero points are one byte per block.
        let blob_size = self.block_size;
        let zp_row_bytes = k_blocks;
        let mut values = vec![0u8; self.n * self.k];
        let mut scaled_zero_points = vec![0.0f32; self.n * k_blocks];
        for output in 0..self.n {
            for block in 0..k_blocks {
                let zero_point = packed_zero_points
                    .as_ref()
                    .map_or(128u8, |points| points[output * zp_row_bytes + block]);
                let scale = scales[output * k_blocks + block];
                scaled_zero_points[output * k_blocks + block] = scale * zero_point as f32;
                let block_start = block * self.block_size;
                let valid = self.k.saturating_sub(block_start).min(self.block_size);
                let packed_start = (output * k_blocks + block) * blob_size;
                let values_start = output * self.k + block_start;
                values[values_start..values_start + valid]
                    .copy_from_slice(&packed[packed_start..packed_start + valid]);
            }
        }
        Ok(PackedU8Weight {
            values,
            scales,
            scaled_zero_points,
        })
    }

    fn dequantize_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        layout: WeightLayout,
    ) -> Result<Vec<f32>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_compute_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let group_indices = group_indices.map(to_dense_i64).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        if let Some(indices) = &group_indices {
            for (index, &group) in indices.iter().enumerate() {
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let blob_size = self.block_size * self.bits / 8;
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        let quantized_mask = if self.bits == 8 {
            u8::MAX
        } else {
            (1u8 << self.bits) - 1
        };
        let default_zero_point = 1u8 << (self.bits - 1);
        let mut weight_kn = vec![0.0f32; self.k * self.n];

        // Fast path: the natural `Nk` ([n, k]) layout with no per-column group
        // indices writes each output row contiguously and independently, so it
        // parallelizes cleanly across the pool. This is the layout the batched
        // prefill GEMM consumes (via `trans_b`), and dequantizing the whole
        // weight was previously a single-threaded scalar loop that dominated the
        // first (cache-cold) prefill. Each `par_chunks_mut(k)` chunk is one
        // output row; the row-major `Kn` transpose and the grouped path keep the
        // original serial scatter below.
        if matches!(layout, WeightLayout::Nk) && group_indices.is_none() {
            let bits = self.bits;
            let block_size = self.block_size;
            weight_kn
                .par_chunks_mut(self.k)
                .enumerate()
                .for_each(|(output, row)| {
                    let packed_start = output * k_blocks * blob_size;
                    let scale_start = output * k_blocks;
                    let zero_point_start = output * zp_row_bytes;
                    let packed_row = &packed[packed_start..packed_start + k_blocks * blob_size];
                    let scale_row = &scales[scale_start..scale_start + k_blocks];
                    let zero_point_row = packed_zero_points
                        .as_ref()
                        .map(|points| &points[zero_point_start..zero_point_start + zp_row_bytes]);
                    dequantize_nbits_row(
                        packed_row,
                        scale_row,
                        zero_point_row,
                        row,
                        bits,
                        block_size,
                    );
                });
            return Ok(weight_kn);
        }

        // Parallel `Kn` fast path: the transposed ([k, n]) layout the dense
        // prefill GEMM consumes. Element `weight_kn[depth * n + output]` means
        // one depth-row `[depth * n .. (depth + 1) * n)` is contiguous, so we can
        // parallelize across depth-rows with `par_chunks_mut(n)` and fill every
        // output within a row. This is byte-identical to the serial scatter
        // below (same `dequantize_nbits_value`, same indices) but distributes the
        // f32 materialization across the pool. Previously this `Kn` transpose was
        // the single-threaded phase that dominated the cache-cold first prefill
        // (~5 s on qwen0.5b-q4, ~43 s/GB), the load-time gap versus ORT (#959).
        // The grouped path keeps the original serial scatter below.
        if matches!(layout, WeightLayout::Kn) && group_indices.is_none() {
            let bits = self.bits;
            let block_size = self.block_size;
            let n = self.n;
            weight_kn
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(depth, row)| {
                    for (output, slot) in row.iter_mut().enumerate() {
                        let packed_start = output * k_blocks * blob_size;
                        let scale_start = output * k_blocks;
                        let zero_point_start = output * zp_row_bytes;
                        let packed_row = &packed[packed_start..packed_start + k_blocks * blob_size];
                        let scale_row = &scales[scale_start..scale_start + k_blocks];
                        let zero_point_row = packed_zero_points.as_ref().map(|points| {
                            &points[zero_point_start..zero_point_start + zp_row_bytes]
                        });
                        *slot = dequantize_nbits_value(
                            packed_row,
                            scale_row,
                            zero_point_row,
                            depth,
                            bits,
                            block_size,
                        );
                    }
                });
            return Ok(weight_kn);
        }

        for output in 0..self.n {
            if group_indices.is_none() {
                let packed_start = output * k_blocks * blob_size;
                let scale_start = output * k_blocks;
                let zero_point_start = output * zp_row_bytes;
                let packed_row = &packed[packed_start..packed_start + k_blocks * blob_size];
                let scale_row = &scales[scale_start..scale_start + k_blocks];
                let zero_point_row = packed_zero_points
                    .as_ref()
                    .map(|points| &points[zero_point_start..zero_point_start + zp_row_bytes]);
                for depth in 0..self.k {
                    let index = match layout {
                        WeightLayout::Kn => depth * self.n + output,
                        WeightLayout::Nk => output * self.k + depth,
                    };
                    weight_kn[index] = dequantize_nbits_value(
                        packed_row,
                        scale_row,
                        zero_point_row,
                        depth,
                        self.bits,
                        self.block_size,
                    );
                }
                continue;
            }
            for depth in 0..self.k {
                let block = depth / self.block_size;
                let within_block = depth % self.block_size;
                let bit_offset = within_block * self.bits;
                let byte = packed[(output * k_blocks + block) * blob_size + bit_offset / 8];
                let quantized = (byte >> (bit_offset % 8)) & quantized_mask;
                let group = group_indices
                    .as_ref()
                    .map_or(block, |indices| indices[depth] as usize);
                let zero_point = packed_zero_points
                    .as_ref()
                    .map_or(default_zero_point, |points| {
                        let bit_offset = group * self.bits;
                        let byte = points[output * zp_row_bytes + bit_offset / 8];
                        (byte >> (bit_offset % 8)) & quantized_mask
                    });
                let index = match layout {
                    WeightLayout::Kn => depth * self.n + output,
                    WeightLayout::Nk => output * self.k + depth,
                };
                weight_kn[index] =
                    (quantized as f32 - zero_point as f32) * scales[output * k_blocks + group];
            }
        }
        Ok(weight_kn)
    }
}

/// Dequantize one packed output row using ORT's LSB-first affine NBits layout.
///
/// `scales` contains one value per K block. `zero_points`, when present,
/// contains those block zero points packed with the same bit width.
pub(super) fn dequantize_nbits_row(
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    output: &mut [f32],
    bits: usize,
    block_size: usize,
) {
    for (depth, value) in output.iter_mut().enumerate() {
        *value = dequantize_nbits_value(packed, scales, zero_points, depth, bits, block_size);
    }
}

#[inline]
fn dequantize_nbits_value(
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    depth: usize,
    bits: usize,
    block_size: usize,
) -> f32 {
    PackedNBitsRow {
        values: packed,
        scales,
        zero_points,
        layout: NBitsLayout { bits, block_size },
    }
    .dequantized_value(depth)
}

fn configured_decode_threads() -> Option<usize> {
    let value = std::env::var(DECODE_THREADS_ENV).ok();
    let available = available_parallelism();
    resolve_decode_threads_with_override(decode_threads_override(), value.as_deref(), available)
}

/// The worker count for the persistent SPMD decode pool ([`crate::decode_spmd`]).
///
/// It honors `ONNX_GENAI_CPU_DECODE_THREADS` when set (`0` opts out), but when
/// the variable is unset it uses a *different, higher* default than the flat
/// pool: [`default_persistent_threads`] (about half the logical CPUs) instead of
/// the flat pool's eight-worker ceiling. The flat Rayon pool caps at eight
/// because its per-op fork/join regresses beyond that; the persistent pool
/// replaces that fork/join with one hot broadcast barrier, so it keeps scaling
/// with cores until it hits the memory-bandwidth knee (measured plateau ~half
/// the logical CPUs on a 2-socket Xeon 8480C). Sizing it from the flat default
/// would leave the out-of-box path at the flat pool's throughput and defeat the
/// point of making it the default.
pub fn configured_persistent_decode_threads() -> Option<usize> {
    let value = std::env::var(DECODE_THREADS_ENV).ok();
    let available = available_parallelism();
    let threads = resolve_persistent_decode_threads_with_override(
        decode_threads_override(),
        value.as_deref(),
        available,
    );
    // Snapdragon/X Elite style ARM64 hosts have measured their decode roofline
    // at 6--8 workers, and the KAI-style packed SDOT path still scales from the
    // generic `available/2` default (6 on a 12-way Oryon) to 8 workers. Keep the
    // generic half-machine rule for other platforms, but use the measured ARM64
    // topology ceiling when no explicit budget was provided.
    #[cfg(all(
        target_arch = "aarch64",
        not(any(target_os = "macos", target_os = "ios"))
    ))]
    if decode_threads_override().is_none()
        && value
            .as_deref()
            .is_none_or(|v| v.is_empty() || v.parse::<usize>().is_err())
        && available >= MAX_TOPOLOGY_DECODE_THREADS
    {
        return Some(MAX_TOPOLOGY_DECODE_THREADS.min(available));
    }
    // On Apple Silicon, when no explicit thread count was set, override the
    // generic `available/2` default with `P_cores - 1`. The SPMD dispatcher
    // thread spin-waits on completion counters, occupying one P-core; using
    // P_cores - 1 workers fills the performance cluster exactly. This is
    // derived at runtime from `hw.perflevel0.physicalcpu` and generalizes
    // across all Apple Silicon tiers. Falls back silently on Intel Macs.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if decode_threads_override().is_none()
        && value
            .as_deref()
            .is_none_or(|v| v.is_empty() || v.parse::<usize>().is_err())
        && let Some(perf_cores) = performance_core_count()
    {
        let override_threads = perf_cores.saturating_sub(1).max(1).min(available);
        return Some(override_threads);
    }
    threads
}

/// Set or clear a process-local CPU decode worker budget.
///
/// This is the programmatic equivalent of `ONNX_GENAI_CPU_DECODE_THREADS`, with
/// higher precedence. Call it before constructing a native decode session; pools
/// are initialized lazily and retain their initial size for the process lifetime.
///
/// When an explicit budget is set, native CPU EP initialization also bounds the
/// global (prefill/MLAS) Rayon pool and, on Linux, the process CPU affinity to
/// the budget (see [`bound_process_to_decode_budget`]), so the budget governs
/// the whole engine -- not just the steady-decode SPMD pool.
pub fn set_decode_thread_budget(threads: Option<usize>) -> std::result::Result<(), &'static str> {
    if threads == Some(0) {
        return Err("CPU decode thread budget must be greater than zero");
    }
    DECODE_THREADS_OVERRIDE.store(threads.unwrap_or(0), Ordering::Release);
    // The standalone MLAS pool sits below this crate and cannot read
    // `DECODE_THREADS_OVERRIDE`, so forward the budget explicitly. Without this
    // the budget would only reach dense `MlasGemmBatch` work on Linux, and only
    // indirectly, via the affinity mask shrinking `available_parallelism`.
    //
    // Forwarded after the store because the two layers reject exactly the same
    // input (`Some(0)`, already rejected above), so this call cannot fail and
    // the two budgets cannot diverge. Keep that invariant if either guard grows.
    #[cfg(feature = "mlas")]
    mlas_sys::set_pool_thread_budget(threads)?;
    Ok(())
}

fn decode_threads_override() -> Option<usize> {
    std::num::NonZeroUsize::new(DECODE_THREADS_OVERRIDE.load(Ordering::Acquire))
        .map(std::num::NonZeroUsize::get)
}

/// Enable or disable the resident dequantized-f32 decode cache for the process.
///
/// The memory-strategy plan calls this before the native decode session is
/// built. Passing `false` makes every `MatMulNBits` decode kernel dequantize its
/// weight into a transient per-call buffer instead of materializing the resident
/// ~8x f32 `weight_nk` expansion, trading decode throughput for footprint when
/// the expansion would not fit the budget (#971). The result is byte-identical
/// (the same `dequantize_weight` + `gemv_nk` math, just not retained). Default
/// is enabled, so an unset process keeps the unchanged fast path.
pub fn set_resident_dequant_f32_cache_enabled(enabled: bool) {
    RESIDENT_DEQUANT_F32_CACHE_DISABLED.store(!enabled, Ordering::Release);
}

/// Whether the resident dequantized-f32 decode cache is currently enabled.
fn resident_dequant_f32_cache_enabled() -> bool {
    !RESIDENT_DEQUANT_F32_CACHE_DISABLED.load(Ordering::Acquire)
}

/// Enable or disable the MLAS SQNBit CompFp32 packed-weight route for the
/// process (the `bits == 4, accuracy_level == 0` dispatch).
///
/// The memory-strategy plan calls this before the native decode session is
/// built, with the same admission verdict it uses for the resident f32 cache.
/// Passing `false` makes [`MatMulNBitsKernel::mlas_sqnbit_owns_fp32_compute`]
/// decline ownership, so those nodes stay on the borrowed zero-copy int4 path
/// and the runtime holds only the on-disk weights instead of an extra
/// session-lifetime MLAS packed buffer (~2x the int4 bytes) beside them. The
/// borrowed fallback is byte-identical, only slower per token on x86. Default is
/// enabled, so an unset process keeps the fast MLAS route.
pub fn set_mlas_sqnbit_packing_enabled(enabled: bool) {
    MLAS_SQNBIT_PACKING_DISABLED.store(!enabled, Ordering::Release);
}

/// Whether the MLAS SQNBit CompFp32 packed-weight route is currently admitted.
#[cfg(feature = "mlas")]
fn mlas_sqnbit_packing_enabled() -> bool {
    !MLAS_SQNBIT_PACKING_DISABLED.load(Ordering::Acquire)
}

/// Predict whether an `m == 1` (decode) `MatMulNBits` node with **constant**
/// weights will materialize the resident dequantized-f32 `weight_nk` cache
/// (the ~8x expansion of #971), given its static attributes and the host's
/// selected decode dot-kernel.
///
/// This is the single authority the engine's memory plan consults so its
/// footprint accounting cannot silently drift from the kernel's own dispatch
/// (the #947 failure mode). It mirrors the branch order of
/// [`MatMulNBitsKernel::execute`] exactly: any earlier packed / on-the-fly path
/// that claims the node returns `false`; only the terminal generic `m == 1`
/// branch that populates `weight_nk` returns `true`. A control test
/// (`predictor_matches_actual_resident_cache`) runs real kernels and asserts
/// this function agrees with the weight actually cached, so a change to the
/// dispatch that this predictor does not track fails the build.
///
/// `n`/`k` are the canonical output/reduction dims and are only consulted on the
/// MLAS path (whose SQNBit interception depends on the shape).
#[allow(clippy::too_many_arguments)]
pub fn matmul_nbits_decode_caches_dequant_f32(
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    n: usize,
    k: usize,
    has_zero_points: bool,
    has_g_idx: bool,
    weight_prepacked: bool,
) -> bool {
    // Prepacked MLAS SQNBit weights are never expanded to f32; they take the
    // dedicated packed path (or are rejected). Not the f32 cache.
    if weight_prepacked {
        return false;
    }
    let dot_kernel = selected_dot_kernel();
    // (A) borrowed_affine_int4 on-the-fly path: bits==4, accuracy_level==0,
    // no g_idx, symmetric OR asymmetric. Dequantizes per call, no resident
    // cache. Constant initializers are contiguous host slices, so the runtime
    // slice guards this path also checks always hold here.
    //
    // Symmetry is deliberately NOT part of this condition. #979 extended the
    // borrowed path to symmetric int4 (implicit midpoint 8, no zero_points
    // input); before that, symmetric fell through to the resident f32 cache and
    // paid ~8x the file size in RAM. This predictor must track the kernel's gate
    // exactly — when it did not, it over-reported `resident_f32_cache_bytes` for
    // symmetric models and the governed cache decision (#987) would have
    // declined a cache that was never going to be built.
    if bits == 4 && accuracy_level == 0 && !has_g_idx {
        return false;
    }
    // (B) MLAS SQNBit path (feature-gated). When present and the shape is
    // supported it consumes the packed weight directly, never the f32 cache.
    #[cfg(feature = "mlas")]
    {
        if !has_g_idx && mlas_qnbit_enabled() {
            let comp = if accuracy_level == 4 {
                mlas_sys::SQNBitComputeType::Int8
            } else {
                mlas_sys::SQNBitComputeType::Fp32
            };
            if mlas_sys::sqnbit_packed_b_size(n, k, bits, block_size, has_zero_points, comp)
                .is_some()
            {
                return false;
            }
        }
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (n, k);
    }
    // (C) 2-bit packed path.
    if bits == 2 && !has_g_idx {
        return false;
    }
    // (D) int4-direct decode (accuracy_level==4, host supports it).
    if bits == 4
        && accuracy_level == 4
        && !has_g_idx
        && dot_kernel.supports_int4_direct(block_size, has_zero_points)
    {
        return false;
    }
    // (E) int8-repacked decode (accuracy_level==4).
    if bits == 4 && accuracy_level == 4 && !has_g_idx {
        return false;
    }
    // (F) 8-bit on-the-fly decode (one byte per element, no f32 expansion).
    if bits == 8 && !has_g_idx {
        return false;
    }
    // (G) Terminal generic m==1 path: materializes the resident f32 weight_nk.
    true
}

/// How many resident copies of each MLAS SQNBit packed buffer the CPU decode
/// runtime holds over a session.
///
/// The executor's kernel cache is **shape-keyed** (`KernelKey { node, shapes }`
/// in `onnx-runtime-session/src/executor/kernel_cache.rs`): a `MatMulNBits`
/// kernel is compiled and cached *per distinct resolved activation shape*. An
/// autoregressive decoder presents exactly two activation shapes to each such
/// node -- the prefill shape (`m == prompt_len`) and the decode shape
/// (`m == 1`) -- so it compiles **two** kernel instances.
///
/// Before #1056 each instance packed its own full copy of the weight into its
/// own `mlas_shards` `OnceLock`, so a session held **two** copies (measured on
/// `qwen05b-symzp`, 169 weight boundaries: a 1-token run packed 169 buffers, a
/// multi-token run 338 = 2x169). #1056 makes the packed buffer **shared per
/// weight**: both instances look it up in the weight-identity store
/// ([`MlasPackedCaches`]) keyed on `(address, N, K, bits, block_size,
/// has_zero_points, compute_type)`, so the first instance to reach it packs
/// once and the sibling takes the same `Arc`. The session now holds **one**
/// copy per weight, and `sqnbit_packed_live_bytes` reflects that (the
/// multi-token run packs the *same* count as the 1-token run).
///
/// This multiplier is therefore `1`: the accounting states the true single
/// resident copy so the memory plan's prediction equals what the kernels
/// actually retain. Keeping it as a named constant (rather than dropping the
/// factor) documents that the shape-keyed duplication is intentionally
/// deduplicated to one, and keeps the accounting and the allocation in lockstep
/// -- change the sharing and this must change with it.
#[cfg(feature = "mlas")]
const MLAS_PACKED_DECODE_INSTANTIATIONS: u64 = 1;

/// Bytes the owned scale (and optional zero-point) copies retained by a single
/// [`mlas_sys::SQNBitPackedB`] add on top of the packed buffer.
///
/// MLAS's `Fp32` pack keeps its own `f32` scale copy and, for an asymmetric
/// weight, a `u8` zero-point copy, so the durable heap of one packed buffer is
/// `packed + scales (+ zero_points)`, not `packed` alone. The layout matches the
/// ONNX initializer shapes the kernel feeds `SQNBitPackedB::new`: scales are
/// `[N, ceil(K / block_size)]` `f32`, and int4 zero points are
/// `[N, ceil(ceil(K / block_size) / 2)]` `u8`. Summed over the N-sharded packs
/// this is exact (the shards split on whole N rows), so it equals
/// [`mlas_sys::SQNBitPackedB::owned_heap_bytes`] minus the packed size.
#[cfg(feature = "mlas")]
fn mlas_sqnbit_scale_zp_bytes(block_size: usize, n: usize, k: usize, has_zero_points: bool) -> u64 {
    if block_size == 0 {
        return 0;
    }
    let blocks = k.div_ceil(block_size) as u64;
    let n = n as u64;
    let scale_bytes = n
        .saturating_mul(blocks)
        .saturating_mul(std::mem::size_of::<f32>() as u64);
    let zp_bytes = if has_zero_points {
        n.saturating_mul(blocks.div_ceil(2))
    } else {
        0
    };
    scale_bytes.saturating_add(zp_bytes)
}

/// The durable heap bytes a **constant-weight** `bits == 4, accuracy_level == 0`
/// `MatMulNBits` node holds for the session on the MLAS SQNBit CompFp32 route --
/// **one packed copy** -- or `None` when that node does not take the MLAS route.
///
/// This mirrors [`MatMulNBitsKernel::mlas_sqnbit_owns_fp32_compute`]'s gate on
/// static attributes: MLAS owns the node only for constant int4
/// `accuracy_level == 0` weights without `g_idx`, when the `mlas` build has a
/// SQNBit kernel for the shape on this host and the route is enabled.
///
/// The value is MLAS's own [`mlas_sys::sqnbit_packed_b_size`] -- the same probe
/// the route decision uses -- **plus** the scale and zero-point copies a
/// [`mlas_sys::SQNBitPackedB`] retains ([`mlas_sqnbit_scale_zp_bytes`]), so it
/// equals the buffer the kernel actually allocates for one instance and cannot
/// drift from it. It is the *per-copy* figure; the session holds
/// [`MLAS_PACKED_DECODE_INSTANTIATIONS`] of these (see
/// [`matmul_nbits_resident_side_cache_bytes`]).
///
/// It deliberately does **not** consult [`mlas_sqnbit_packing_enabled`]: the
/// accounting states what admitting the route would cost, and the plan then
/// decides admission from that cost. Gating it on the admission flag would make
/// the cost vanish exactly when the plan needs it to make the decision.
#[cfg(feature = "mlas")]
fn mlas_sqnbit_packed_b_cache_bytes(
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    n: usize,
    k: usize,
    has_zero_points: bool,
    has_g_idx: bool,
    weight_prepacked: bool,
) -> Option<u64> {
    if weight_prepacked || has_g_idx || bits != 4 || accuracy_level != 0 || !mlas_qnbit_enabled() {
        return None;
    }
    mlas_sys::sqnbit_packed_b_size(
        n,
        k,
        bits,
        block_size,
        has_zero_points,
        mlas_sys::SQNBitComputeType::Fp32,
    )
    .map(|packed| {
        (packed as u64).saturating_add(mlas_sqnbit_scale_zp_bytes(
            block_size,
            n,
            k,
            has_zero_points,
        ))
    })
}

/// Resident side-buffer bytes a single **constant-weight** `MatMulNBits` node
/// holds for the session *beside* the on-disk weights, given its static
/// attributes: either the MLAS SQNBit CompFp32 packed buffer (the packed
/// weight plus its retained scale/zero-point copies, shared across the node's
/// compiled kernel instances) when the node takes that route, the fully-expanded
/// f32 `weight_nk` cache (`N * K * 4`, #971) when it takes the generic decode
/// path, or zero for a truly zero-copy path (borrowed int4, 2-bit, int8,
/// accuracy-4 direct).
///
/// For the MLAS route the figure is [`MLAS_PACKED_DECODE_INSTANTIATIONS`] times
/// the per-copy cost. That multiplier is `1` since #1056: although the
/// shape-keyed kernel cache still compiles a separate `MatMulNBits` instance for
/// the prefill (`m > 1`) and decode (`m == 1`) activation shapes, both instances
/// now share one packed buffer through the weight-identity store (see the
/// constant's docs), so the session holds a single resident copy. The generic
/// f32 `weight_nk` cache is likewise a decode-only (`m == 1`) materialization
/// held once, so no multiplier applies there either.
///
/// This is the single per-node authority [`resident_dequant_f32_cache_bytes`]
/// sums over the graph so the memory plan's footprint accounting tracks what the
/// kernel actually retains. The MLAS packed buffer is checked first because its
/// route pre-empts the f32 cache in [`MatMulNBitsKernel::execute`].
#[allow(clippy::too_many_arguments)]
pub fn matmul_nbits_resident_side_cache_bytes(
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    n: usize,
    k: usize,
    has_zero_points: bool,
    has_g_idx: bool,
    weight_prepacked: bool,
) -> u64 {
    #[cfg(feature = "mlas")]
    if let Some(per_copy) = mlas_sqnbit_packed_b_cache_bytes(
        bits,
        block_size,
        accuracy_level,
        n,
        k,
        has_zero_points,
        has_g_idx,
        weight_prepacked,
    ) {
        return per_copy.saturating_mul(MLAS_PACKED_DECODE_INSTANTIATIONS);
    }
    if matmul_nbits_decode_caches_dequant_f32(
        bits,
        block_size,
        accuracy_level,
        n,
        k,
        has_zero_points,
        has_g_idx,
        weight_prepacked,
    ) {
        (n as u64).saturating_mul(k as u64).saturating_mul(4)
    } else {
        0
    }
}

/// Total resident side-buffer bytes this CPU EP will hold for `graph` beside the
/// on-disk weights, i.e. the extra footprint the memory plan must budget for.
///
/// For every constant-weight `MatMulNBits` node this sums
/// [`matmul_nbits_resident_side_cache_bytes`]: the fully-expanded f32 `weight_nk`
/// cache (`N * K * 4`, #971) when the node takes the generic decode path, or the
/// MLAS SQNBit CompFp32 packed buffer when it takes the `accuracy_level == 0`
/// MLAS route -- the packed weight plus its retained scale/zero-point copies,
/// counted [`MLAS_PACKED_DECODE_INSTANTIATIONS`] (`1`) time: the prefill and
/// decode kernel instances share one packed buffer via the weight-identity
/// store (#1056), so the weight is resident once.
/// Nodes on a truly zero-copy path (borrowed int4, 2-bit, int8) contribute
/// nothing, and non-constant-weight nodes never retain a session buffer (they
/// rebuild a transient one per call) and are excluded.
///
/// The name is kept for API stability; the total now covers the MLAS packed
/// buffer too, so a model whose int4 `accuracy_level == 0` nodes route to MLAS
/// is no longer accounted as zero (the ledger blind spot that made the packed
/// weight invisible).
pub fn resident_dequant_f32_cache_bytes(graph: &Graph) -> u64 {
    let mut total = 0_u64;
    for node in graph.nodes.values() {
        if !LazyWeightBoundary::MatMulNBits.matches(&node.domain, &node.op_type) {
            continue;
        }
        // Only constant weights are cached; a non-constant B input rebuilds a
        // transient dequant per call, so it never holds a resident expansion.
        let Some(Some(weight_value)) = node.inputs.get(1) else {
            continue;
        };
        if !graph.initializers.contains_key(weight_value) {
            continue;
        }
        let bits = node
            .attr("bits")
            .and_then(Attribute::as_int)
            .unwrap_or(4)
            .max(0) as usize;
        let block_size = node
            .attr("block_size")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            .max(0) as usize;
        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(Attribute::as_int)
            .unwrap_or(0);
        let Some(n) = node.attr("N").and_then(Attribute::as_int) else {
            continue;
        };
        let Some(k) = node.attr("K").and_then(Attribute::as_int) else {
            continue;
        };
        if n <= 0 || k <= 0 {
            continue;
        }
        let (n, k) = (n as u64, k as u64);
        let has_zero_points = matches!(node.inputs.get(3), Some(Some(_)));
        let has_g_idx = matches!(node.inputs.get(4), Some(Some(_)));
        total = total.saturating_add(matmul_nbits_resident_side_cache_bytes(
            bits,
            block_size,
            accuracy_level,
            n as usize,
            k as usize,
            has_zero_points,
            has_g_idx,
            false,
        ));
    }
    total
}

/// The heap bytes all live MLAS SQNBit packed weights currently retain: the
/// packed buffers plus their owned scale and zero-point copies, summed across
/// every distinct packed allocation. Since #1056 a node's prefill and decode
/// kernel instances share one packed buffer, so each weight is counted once --
/// matching what [`resident_dequant_f32_cache_bytes`] now *predicts*. This is
/// the *actual* runtime figure that prediction must equal; the `matmul_nbits`
/// accounting tests compare the two directly. Zero on a build without the
/// `mlas` feature.
pub fn mlas_sqnbit_packed_live_bytes() -> u64 {
    #[cfg(feature = "mlas")]
    {
        mlas_sys::sqnbit_packed_live_bytes() as u64
    }
    #[cfg(not(feature = "mlas"))]
    {
        0
    }
}

/// should be built with, given the explicit decode budget.
///
/// Only an *explicit* budget bounds the global pool: the process-local override
/// (from [`set_decode_thread_budget`]) or a positive `ONNX_GENAI_CPU_DECODE_THREADS`.
/// An unset budget returns `None` so the default `available_parallelism()`
/// sizing is left untouched (no regression to the out-of-box path); the opt-out
/// `ONNX_GENAI_CPU_DECODE_THREADS=0` and any unparseable value likewise return
/// `None`. The result is clamped to `available` so a budget larger than the host
/// never over-subscribes.
fn resolve_rayon_global_threads(
    override_threads: Option<usize>,
    raw: Option<&str>,
    available: usize,
) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let requested = match override_threads {
        Some(threads) if threads > 0 => threads,
        Some(_) => return None,
        None => match raw?.trim().parse::<usize>() {
            Ok(0) => return None,
            Ok(threads) => threads,
            Err(_) => return None,
        },
    };
    Some(requested.min(available))
}

/// Guards the one-shot process-wide "good CPU citizen" bounding so it runs at
/// most once and only its first attempt logs.
static PROCESS_BUDGET_BOUND: OnceLock<()> = OnceLock::new();

/// Confine the whole process to the explicit decode budget so a user who caps
/// cores (via `--cpu-cores N`, `ONNX_GENAI_CPU_DECODE_THREADS=N`, or
/// [`set_decode_thread_budget`]) disturbs at most `N` CPUs -- covering prefill
/// and every MLAS GEMM, not just the steady-decode SPMD pool.
///
/// Two mechanisms, applied once, early (at CPU EP initialization, before any
/// GEMM builds the lazily-initialized global Rayon pool):
///
/// 1. **Rayon global-pool size.** The pool is built with `N` threads instead of
///    `available_parallelism()`, so `mlas-sys`'s `rayon_max_threads` reports `N`
///    and MLAS partitions each GEMM into `N` tiles. The pool is fixed for the
///    process lifetime and `build_global` fails if it already exists, so this
///    must run before the first Rayon use; if the pool was already built we log
///    once and leave it (a no-op with warning).
/// 2. **(Linux) process CPU affinity.** The calling (main) thread is confined to
///    `N` CPUs packed on a single NUMA node where possible; threads spawned
///    afterwards (the Rayon pool, the SPMD decode pool) inherit the mask, so the
///    process stays on `N` CPUs without an external `taskset`. This composes
///    with the existing decode-affinity control: if the user set an explicit
///    `ONNX_GENAI_CPU_DECODE_AFFINITY`, their choice wins and the auto-mask
///    stands down. Non-Linux hosts skip affinity (the Rayon-count bound still
///    applies).
///
/// A no-op when no explicit budget is set, so the default sizing is unchanged.
pub fn bound_process_to_decode_budget() {
    if PROCESS_BUDGET_BOUND.set(()).is_err() {
        return;
    }
    let raw = std::env::var(DECODE_THREADS_ENV).ok();
    let available = available_parallelism();
    let Some(threads) =
        resolve_rayon_global_threads(decode_threads_override(), raw.as_deref(), available)
    else {
        return;
    };

    // Apply the process CPU affinity mask *before* building the Rayon global
    // pool so its worker threads inherit the restricted CPU set.
    #[cfg(target_os = "linux")]
    if crate::decode_affinity::explicit_decode_affinity_requested() {
        eprintln!(
            "onnx-genai: CPU decode budget {threads} bounds the prefill/MLAS Rayon pool; \
             process CPU affinity is left to the explicit ONNX_GENAI_CPU_DECODE_AFFINITY setting"
        );
    } else if let Some(cpus) = crate::decode_affinity::select_budget_cpus(threads) {
        match crate::decode_affinity::set_current_thread_affinity(&cpus) {
            Ok(()) => eprintln!(
                "onnx-genai: CPU decode budget {threads} confined the process to {count} CPUs \
                 {cpus:?} (prefill/MLAS + decode stay off the rest of the machine)",
                count = cpus.len()
            ),
            Err(message) => eprintln!(
                "onnx-genai: CPU decode budget {threads}: could not apply process CPU affinity \
                 ({message}); the Rayon thread-count bound still applies"
            ),
        }
    }

    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        Ok(()) => eprintln!(
            "onnx-genai: CPU decode budget {threads} bounded the global Rayon pool \
             (prefill/MLAS parallelism capped at {threads} workers)"
        ),
        Err(err) => eprintln!(
            "onnx-genai: CPU decode budget {threads}: the global Rayon pool was already built \
             and cannot be resized ({err}); set the budget before the first inference to bound \
             prefill/MLAS parallelism"
        ),
    }
}

/// Default persistent-pool worker count for `available` logical CPUs: half of
/// them (at least one), derived purely from topology (Rule 2).
///
/// Half leaves a full set of hardware threads free for the dispatcher (which
/// runs the forward inline and spins on the completion counters), the prefill
/// global Rayon pool, and co-tenants on a shared box. Because the SPMD workers
/// *spin* before parking, a fully-subscribed pool starves the dispatcher and
/// collapses throughput (measured 1.4 tok/s at 96 workers vs 28.7 at 48 on a
/// 96-logical-CPU host); half sits at the measured plateau while avoiding that
/// cliff, and on SMT hosts it maps to roughly the physical-core count.
fn default_persistent_threads(available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    Some((available / 2).max(1))
}

/// Query the performance-core count on Apple Silicon via sysctl.
/// Returns `None` on Intel Macs or if the query fails.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn performance_core_count() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.perflevel0.physicalcpu").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: sysctlbyname reads a kernel sysctl into a correctly-sized buffer.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &raw mut value as *mut std::ffi::c_void,
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

/// Resolve the persistent-pool worker count from the raw `ONNX_GENAI_CPU_DECODE_THREADS`
/// value and the host's logical CPU count. `Some("0")` opts out (`None`); an
/// explicit positive count is honored (clamped to `available`); an unset or
/// unparseable value falls back to [`default_persistent_threads`].
#[cfg(test)]
fn resolve_persistent_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    resolve_persistent_decode_threads_with_override(None, raw, available)
}

fn resolve_persistent_decode_threads_with_override(
    override_threads: Option<usize>,
    raw: Option<&str>,
    available: usize,
) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let default = default_persistent_threads(available)?;
    let threads = match override_threads {
        Some(threads) => threads,
        None => match raw {
            Some("0") => return None,
            Some(raw) => raw
                .parse::<usize>()
                .ok()
                .filter(|threads| *threads > 0)
                .unwrap_or(default),
            None => default,
        },
    };
    Some(threads.min(available))
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// Choose a bounded decode pool from the host's logical CPU count.
///
/// Decode projections are small and bandwidth-bound, so worker demand grows
/// much more slowly than core count: `1 + ceil(log2(logical_cpus))` gives 3
/// workers on 4-way hosts, 4 on 8-way hosts, and the profiled 8 workers on the
/// 96-way Xeon. The measured eight-worker ceiling limits fork/join overhead, and
/// the result never exceeds the CPUs available to a cgroup.
fn default_decode_threads(available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let ceil_log2 = usize::BITS as usize - (available - 1).leading_zeros() as usize;
    Some(
        (ceil_log2 + 1)
            .min(MAX_TOPOLOGY_DECODE_THREADS)
            .min(available),
    )
}

#[cfg(test)]
fn resolve_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    resolve_decode_threads_with_override(None, raw, available)
}

fn resolve_decode_threads_with_override(
    override_threads: Option<usize>,
    raw: Option<&str>,
    available: usize,
) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let default = default_decode_threads(available)?;
    let threads = match override_threads {
        Some(threads) => threads,
        None => match raw {
            Some("0") => return None,
            Some(raw) => raw.parse::<usize>().unwrap_or(default),
            None => default,
        },
    };
    (threads > 0).then(|| threads.min(available))
}

/// Default worker count for the **dense-f32** decode pool ([`DENSE_DECODE_POOL`]).
///
/// The dense path runs the multi-threaded MLAS GEMM, which scales past the flat
/// pool's eight-worker ceiling but is memory-bandwidth-bound, so it plateaus
/// once the memory controllers saturate. `available / 4`, clamped to
/// `[8, MAX_DENSE_DECODE_THREADS]`, lands inside the measured plateau on large
/// hosts (24 workers on a 96-way 2-socket Xeon) while still using every core on
/// small hosts. Derived purely from the logical CPU count (Rule 2, no per-model
/// tuning); `ONNX_GENAI_CPU_DECODE_THREADS` overrides it.
fn default_dense_decode_threads(available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let scaled = (available / 4).max(1);
    Some(
        scaled
            .clamp(8.min(available), MAX_DENSE_DECODE_THREADS)
            .min(available),
    )
}

/// Resolve the dense-f32 decode pool worker count from the raw
/// `ONNX_GENAI_CPU_DECODE_THREADS` value and the host's logical CPU count.
/// `Some("0")` opts out (`None` → run on the global Rayon pool); an explicit
/// positive count is honored (clamped to `available`); unset/unparseable falls
/// back to [`default_dense_decode_threads`].
#[cfg(test)]
fn resolve_dense_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    resolve_dense_decode_threads_with_override(None, raw, available)
}

fn resolve_dense_decode_threads_with_override(
    override_threads: Option<usize>,
    raw: Option<&str>,
    available: usize,
) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let default = default_dense_decode_threads(available)?;
    let threads = match override_threads {
        Some(threads) => threads,
        None => match raw {
            Some("0") => return None,
            Some(raw) => raw
                .parse::<usize>()
                .ok()
                .filter(|threads| *threads > 0)
                .unwrap_or(default),
            None => default,
        },
    };
    Some(threads.min(available))
}

fn configured_dense_decode_threads() -> Option<usize> {
    let value = std::env::var(DECODE_THREADS_ENV).ok();
    let available = available_parallelism();
    resolve_dense_decode_threads_with_override(
        decode_threads_override(),
        value.as_deref(),
        available,
    )
}

#[cfg(feature = "mlas")]
fn default_sqnbit_decode_min(available: usize) -> usize {
    default_decode_threads(available)
        .unwrap_or(1)
        .saturating_mul(2)
}

fn build_decode_pool(
    threads: Option<usize>,
) -> std::result::Result<Option<rayon::ThreadPool>, String> {
    threads
        .map(|threads| {
            let affinity_cpus = decode_affinity_cpus(threads)?;
            let mut builder = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("onnx-genai-decode-{index}"));
            if let Some(cpus) = affinity_cpus {
                builder = builder.start_handler(move |worker_index| {
                    // Pin each worker to a distinct CPU of the selected NUMA node
                    // so the per-op fork-join barrier and the streamed weights
                    // stay node-local. Best-effort: a restricted cgroup that
                    // rejects the request is logged once, not fatal, so decode
                    // still runs (unpinned) rather than failing outright.
                    let cpu = cpus[worker_index % cpus.len()];
                    if let Err(message) = crate::decode_affinity::pin_current_thread_to_cpu(cpu) {
                        report_decode_affinity_failure(&message);
                    }
                });
            }
            builder
                .build()
                .map_err(|err| format!("failed to build {DECODE_THREADS_ENV} pool: {err}"))
        })
        .transpose()
}

/// Resolve the CPU set the decode pool should pin `threads` workers to, honoring
/// the explicit [`crate::decode_affinity::DECODE_AFFINITY_ENV`] switch, the
/// auto-enable policy, and the process's allowed CPU set (cpuset/taskset). The
/// chosen auto-policy is logged once at info. Returns `Ok(None)` when pinning is
/// off, unsupported, or declined; propagates malformed configuration as a clear
/// error.
fn decode_affinity_cpus(threads: usize) -> std::result::Result<Option<Vec<usize>>, String> {
    let plan = crate::decode_affinity::plan_decode_affinity(threads)?;
    if let Some(message) = plan.log {
        report_decode_affinity_policy(&message);
    }
    Ok(plan.cpus)
}

/// Log the decode-affinity auto-policy decision once (info): whether pinning was
/// auto-enabled, declined (cpuset/single-node/unsupported OS), and why.
fn report_decode_affinity_policy(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: decode affinity policy: {message}");
    }
}

/// Log the first decode-affinity pinning failure once so a restricted
/// environment surfaces the reason without spamming every worker.
fn report_decode_affinity_failure(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!(
            "onnx-genai: decode-pool CPU affinity unavailable; \
             continuing without pinning ({message})"
        );
    }
}

fn with_decode_pool<T: Send>(operation: impl FnOnce() -> T + Send) -> Result<T> {
    // If we are already resident inside a `with_decode_pool_scope` installation
    // on this worker thread, run inline: the enclosing `pool.install(...)` already
    // put us on a decode-pool worker, so a fresh `install` here would only add a
    // redundant external-thread-to-pool crossing (task publication + wakeup +
    // join) per projection -- exactly the per-op fork-join fragmentation the
    // whole-forward residency scope eliminates. Inline `operation()` still
    // fans out via rayon's work-stealing on the current (decode) pool.
    if IN_DECODE_POOL.with(Cell::get) {
        return Ok(operation());
    }
    match DECODE_POOL.get_or_init(|| build_decode_pool(configured_decode_threads())) {
        Ok(Some(pool)) => Ok(pool.install(operation)),
        Ok(None) => Ok(operation()),
        Err(message) => Err(error(message.clone())),
    }
}

thread_local! {
    /// Per-worker-thread flag marking that the current thread is executing
    /// inside a [`with_decode_pool_scope`] installation. Set on the decode-pool
    /// worker that runs the wrapped forward pass so the inner [`with_decode_pool`]
    /// calls run inline instead of re-installing.
    static IN_DECODE_POOL: Cell<bool> = const { Cell::new(false) };

    /// Per-thread flag marking that the current thread is running the forward
    /// pass inside a `numa-split` [`with_decode_pool_scope`] installation. Set on
    /// the dispatcher worker that runs the forward so each M=1 projection fans
    /// its output rows out across the per-node sub-pools (see
    /// [`parallel_output_rows`] and [`crate::decode_numa`]).
    static IN_NUMA_SCOPE: Cell<bool> = const { Cell::new(false) };

    /// Per-thread flag marking that the current thread is running the forward
    /// pass inside a persistent SPMD-pool ([`crate::decode_spmd`])
    /// [`with_decode_pool_scope`] installation, so each M=1 projection fans its
    /// output rows out across the persistent worker set instead of a per-op
    /// Rayon region.
    static IN_SPMD_SCOPE: Cell<bool> = const { Cell::new(false) };
}

/// The lazily built `numa-split` decode layout, or `None` when the mode is not
/// requested or the host cannot be split (fallback, logged once).
///
/// It is sized from [`configured_persistent_decode_threads`] (about half the
/// logical CPUs), *not* the flat pool's eight-worker ceiling. `numa-split` is
/// the two-level, node-pinned mirror of the persistent SPMD pool (see
/// [`crate::decode_spmd`]) and its whole purpose is to reach *both* sockets'
/// memory bandwidth; the eight-worker flat ceiling would leave only ~four
/// row-sharded workers per node, far too few to saturate either memory
/// controller, so it could never realize the bandwidth win the layout exists
/// for. Half the logical CPUs, split across the nodes, lands each per-node
/// sub-pool at the measured bandwidth knee while leaving cores for the
/// dispatcher and co-tenants (a *fully*-subscribed split oversubscribes the
/// cores and collapses throughput). `ONNX_GENAI_CPU_DECODE_THREADS` still
/// overrides the count (and `0` opts out).
fn numa_pools() -> Option<&'static crate::decode_numa::NumaDecodePools> {
    static NUMA_POOLS: OnceLock<Option<crate::decode_numa::NumaDecodePools>> = OnceLock::new();
    NUMA_POOLS
        .get_or_init(|| crate::decode_numa::build_from_env(configured_persistent_decode_threads()))
        .as_ref()
}

/// The active `numa-split` layout when the current thread is running a
/// `numa-split` decode forward; `None` otherwise (so prefill, non-decode work,
/// and the flat single-node modes keep their existing behaviour).
fn numa_decode_active() -> Option<&'static crate::decode_numa::NumaDecodePools> {
    if IN_NUMA_SCOPE.with(Cell::get) {
        numa_pools()
    } else {
        None
    }
}

/// The active persistent SPMD layout when the current thread is running a
/// persistent-pool decode forward; `None` otherwise.
pub(crate) fn spmd_decode_active() -> Option<&'static crate::decode_spmd::SpmdDecodePools> {
    if IN_SPMD_SCOPE.with(Cell::get) {
        crate::decode_spmd::pools()
    } else {
        None
    }
}

#[cfg(test)]
static SPMD_TEST_DISPATCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Fan a projection's output rows out across the decode workers.
///
/// With `numa-split` active, the rows are sharded across the per-node sub-pools
/// (node-local weights, single cross-node join). Otherwise the flat single-node
/// pool chunks them as before. `compute(output_start, outputs)` fills the rows
/// `output_start .. output_start + outputs.len()`, so the math is identical
/// regardless of how the rows are partitioned (row-sharding a GEMV is exactly
/// associative -- no cross-row reduction -- so results are bit-identical).
fn parallel_output_rows<F>(result: &mut [f32], k: usize, compute: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(numa) = numa_decode_active() {
        numa.dispatch_output_rows(result, k, &compute);
        return;
    }
    if let Some(spmd) = spmd_decode_active() {
        #[cfg(test)]
        SPMD_TEST_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        spmd.dispatch_output_rows(result, k, &compute);
        return;
    }
    let chunk = output_chunk_len(result.len(), k);
    if chunk < result.len() {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

/// Fan `num_rows` fixed-width output rows (each `row_len` elements of `result`)
/// out across the active decode workers, running `compute(row_index, row_slice)`
/// on each whole row.
///
/// This is the row-block analogue of [`parallel_output_rows`] for decode kernels
/// (e.g. `GroupQueryAttention`) whose parallel unit is a full contiguous row
/// rather than a GEMV scalar output. It exists so those kernels use the *same*
/// decode pool as the `MatMulNBits` projections instead of a second thread pool:
///
/// * When a persistent SPMD decode scope is active the forward runs on the
///   engine thread (not a Rayon worker), so a bare `par_chunks_mut` here would
///   fall to the *global* Rayon pool and contend with the SPMD pool's resident,
///   pinned, spinning workers. Routing through the SPMD pool removes that
///   contention (measured to dominate 7B CPU decode).
/// * The `numa-split` and flat decode scopes install the forward onto a bounded
///   Rayon pool, so `par_chunks_mut` already runs on that pool (no global-pool
///   contention); they keep the existing behaviour.
///
/// Each row is independent, so sharding it across workers reproduces the
/// single-threaded result bit-for-bit. Generality: the routing keys off which
/// decode scope is active, never off op or model identity (RULES.md §2).
pub fn decode_parallel_output_row_blocks<F>(
    result: &mut [f32],
    row_len: usize,
    num_rows: usize,
    compute: F,
) where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(spmd) = spmd_decode_active() {
        spmd.dispatch_output_row_blocks(result, row_len, num_rows, &compute);
        return;
    }
    // numa-split and flat decode scopes run the forward on a bounded Rayon pool,
    // so this `par_chunks_mut` uses that pool rather than the global one.
    //
    // `for_each_init` rather than `for_each` so the trace span is opened once
    // per Rayon job — that is, once per worker per contiguous slice of rows —
    // instead of once per row. A row is far too small to carry a span.
    result.par_chunks_mut(row_len).enumerate().for_each_init(
        || crate::trace::worker_span("MatMulNBits.decode_rows"),
        |_span, (row_index, row)| compute(row_index, row),
    );
}

/// Fan `num_tasks` independent decode subtasks out across the active decode
/// workers, running `compute(task_index)` on each.
///
/// Tasks are handed out in contiguous index ranges, mirroring
/// [`decode_parallel_output_row_blocks`] but *without* partitioning a shared
/// result buffer — each `compute` writes into its own disjoint scratch region,
/// an invariant the caller owns. This is the scheduling primitive
/// `GroupQueryAttention` flash-decoding uses to spread its
/// `attention_rows × split_count` KV-chunk partials across otherwise-idle cores
/// (when `attention_rows` alone cannot fill the pool). Routing keys off which
/// decode scope is active, never off op or model identity, exactly like
/// [`decode_parallel_output_row_blocks`].
pub fn decode_parallel_index_tasks<F>(num_tasks: usize, compute: F)
where
    F: Fn(usize) + Sync + Send,
{
    if let Some(spmd) = spmd_decode_active() {
        spmd.dispatch_index_tasks(num_tasks, &compute);
        return;
    }
    // numa-split and flat decode scopes run the forward on a bounded Rayon pool,
    // so this parallel iterator uses that pool rather than the global one.
    (0..num_tasks).into_par_iter().for_each(compute);
}

/// Effective number of decode workers on the active decode scope.
///
/// Used to decide whether attention has idle cores to exploit for flash-decoding
/// KV splitting: when the worker count exceeds `attention_rows` (16 query heads
/// in decode), the surplus cores would otherwise sit idle during the attention
/// reduction. Returns the persistent SPMD or `numa-split` worker count when one
/// of those scopes is active, otherwise the current Rayon pool width (the flat
/// decode pool the forward is installed on).
pub fn active_decode_worker_count() -> usize {
    if let Some(numa) = numa_decode_active() {
        return numa.total_workers();
    }
    if let Some(spmd) = spmd_decode_active() {
        return spmd.total_workers();
    }
    rayon::current_num_threads()
}

/// First-touch each row-major weight component on the NUMA node that will read
/// it under `numa-split` or the persistent SPMD pool, so each node's workers
/// stream node-local memory. A no-op (returns the input) when neither node-aware
/// decode mode is active.
fn numa_place_int4(weight: PackedInt4Weight, n: usize) -> PackedInt4Weight {
    if let Some(numa) = numa_decode_active() {
        return PackedInt4Weight {
            values: numa.place_rows(&weight.values, n),
            scales: numa.place_rows(&weight.scales, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return PackedInt4Weight {
            values: spmd.place_rows(&weight.values, n),
            scales: spmd.place_rows(&weight.scales, n),
        };
    }
    weight
}

fn numa_place_n16_sdot(weight: PackedN16SdotWeight, n: usize) -> PackedN16SdotWeight {
    if let Some(numa) = numa_decode_active() {
        return PackedN16SdotWeight {
            values: numa.place_rows(&weight.values, n.div_ceil(N16_SDOT_OUTPUTS)),
            scales: numa.place_rows(&weight.scales, n),
            zero_point_offsets: numa.place_rows(&weight.zero_point_offsets, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return PackedN16SdotWeight {
            values: spmd.place_rows(&weight.values, n.div_ceil(N16_SDOT_OUTPUTS)),
            scales: spmd.place_rows(&weight.scales, n),
            zero_point_offsets: spmd.place_rows(&weight.zero_point_offsets, n),
        };
    }
    weight
}

fn numa_place_kai_sdot(weight: PackedKaiSdotWeight, n: usize) -> PackedKaiSdotWeight {
    let tiles = n.div_ceil(KAI_SDOT_OUTPUTS);
    if let Some(numa) = numa_decode_active() {
        return PackedKaiSdotWeight {
            bits: weight.bits,
            values: numa.place_rows(&weight.values, tiles),
            scales: numa.place_rows(&weight.scales, tiles),
            rhs_sums: numa.place_rows(&weight.rhs_sums, tiles),
            zero_point_offsets: numa.place_rows(&weight.zero_point_offsets, tiles),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return PackedKaiSdotWeight {
            bits: weight.bits,
            values: spmd.place_rows(&weight.values, tiles),
            scales: spmd.place_rows(&weight.scales, tiles),
            rhs_sums: spmd.place_rows(&weight.rhs_sums, tiles),
            zero_point_offsets: spmd.place_rows(&weight.zero_point_offsets, tiles),
        };
    }
    weight
}

/// Node-local first-touch for a standard packed NBits weight.
fn numa_place_nbits(weight: PackedNBitsWeight, n: usize) -> PackedNBitsWeight {
    let place =
        |values: Vec<u8>, scales: Vec<f32>, zero_points: Option<Vec<u8>>| PackedNBitsWeight {
            values,
            scales,
            zero_points,
        };
    if let Some(numa) = numa_decode_active() {
        return place(
            numa.place_rows(&weight.values, n),
            numa.place_rows(&weight.scales, n),
            weight
                .zero_points
                .as_ref()
                .map(|points| numa.place_rows(points, n)),
        );
    }
    if let Some(spmd) = spmd_decode_active() {
        return place(
            spmd.place_rows(&weight.values, n),
            spmd.place_rows(&weight.scales, n),
            weight
                .zero_points
                .as_ref()
                .map(|points| spmd.place_rows(points, n)),
        );
    }
    weight
}

/// Node-local first-touch for the prepacked int8 weight (see [`numa_place_int4`]).
fn numa_place_int8(weight: Int8Weight, n: usize) -> Int8Weight {
    if let Some(numa) = numa_decode_active() {
        return Int8Weight {
            values: numa.place_rows(&weight.values, n),
            scales: numa.place_rows(&weight.scales, n),
            block_sums: numa.place_rows(&weight.block_sums, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return Int8Weight {
            values: spmd.place_rows(&weight.values, n),
            scales: spmd.place_rows(&weight.scales, n),
            block_sums: spmd.place_rows(&weight.block_sums, n),
        };
    }
    weight
}

/// Node-local first-touch for the prepacked 8-bit `u8` weight (see
/// [`numa_place_int4`]).
fn numa_place_u8(weight: PackedU8Weight, n: usize) -> PackedU8Weight {
    if let Some(numa) = numa_decode_active() {
        return PackedU8Weight {
            values: numa.place_rows(&weight.values, n),
            scales: numa.place_rows(&weight.scales, n),
            scaled_zero_points: numa.place_rows(&weight.scaled_zero_points, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return PackedU8Weight {
            values: spmd.place_rows(&weight.values, n),
            scales: spmd.place_rows(&weight.scales, n),
            scaled_zero_points: spmd.place_rows(&weight.scaled_zero_points, n),
        };
    }
    weight
}

/// Node-local first-touch for the dequantized `[N, K]` weight (see
/// [`numa_place_int4`]).
fn numa_place_nk(weight: Vec<f32>, n: usize) -> Vec<f32> {
    if let Some(numa) = numa_decode_active() {
        return numa.place_rows(&weight, n);
    }
    if let Some(spmd) = spmd_decode_active() {
        return spmd.place_rows(&weight, n);
    }
    weight
}

/// RAII guard that marks the current thread as running a `numa-split` decode
/// forward and restores the previous state on drop (including on panic).
struct NumaScopeGuard {
    previous: bool,
}

impl NumaScopeGuard {
    fn enter() -> Self {
        let previous = IN_NUMA_SCOPE.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for NumaScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_NUMA_SCOPE.with(|flag| flag.set(previous));
    }
}

/// RAII guard that marks the current thread as running a persistent SPMD-pool
/// decode forward and restores the previous state on drop (including on panic).
struct SpmdScopeGuard {
    previous: bool,
}

impl SpmdScopeGuard {
    fn enter() -> Self {
        let previous = IN_SPMD_SCOPE.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for SpmdScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_SPMD_SCOPE.with(|flag| flag.set(previous));
    }
}

/// RAII guard that marks the current thread as resident inside the decode pool
/// and restores the previous state on drop -- including during panic unwinding,
/// so a panicking forward pass cannot leak a stale `true` onto a pooled worker.
struct DecodeResidencyGuard {
    previous: bool,
}

impl DecodeResidencyGuard {
    fn enter() -> Self {
        let previous = IN_DECODE_POOL.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for DecodeResidencyGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_DECODE_POOL.with(|flag| flag.set(previous));
    }
}

/// Run `f` with the whole call tree resident inside the bounded M=1 decode pool.
///
/// Wrapping an entire single-token CPU decode forward in one installation lets
/// the many inner `MatMulNBits` projections execute inline on already-woken
/// decode-pool workers (see [`with_decode_pool`]), eliminating the per-op
/// external-thread-to-pool crossing that fragments end-to-end decode throughput.
///
/// `model_uses_spmd_pool` says whether this model's decode actually dispatches
/// work *through* the shared decode pool -- i.e. it contains quantized
/// `MatMulNBits` (or quantized MoE) projections whose row-sharding runs on the
/// persistent SPMD / numa-split workers. When it is `false` the decode is a
/// **dense-f32** graph whose dominant `MatMul`s are serviced by the
/// multi-threaded MLAS GEMM on the pool's Rayon workers; the persistent SPMD
/// pool's pinned, *spinning* workers then provide no benefit and actively steal
/// cores from (and contend with) that GEMM. Such models therefore skip the
/// spinning SPMD/numa pools (unless the pool was explicitly forced on) and use
/// the bounded, non-spinning [`DENSE_DECODE_POOL`] instead. This keys off a
/// structural graph property (quantized vs dense), never off a specific model,
/// so it generalizes across every f32 and quantized model (RULES §2).
///
/// Behaviour by pool state:
/// * `Ok(Some(pool))` -- install `f` on the decode pool; the residency flag is
///   set *inside* the installed closure (on the worker thread that actually runs
///   `f`, not the caller) and cleared by the RAII guard on exit or panic.
/// * `Ok(None)` -- decode pool opted out (`ONNX_GENAI_CPU_DECODE_THREADS=0`); run
///   `f` inline on the global rayon pool with the flag left `false`, so inner
///   `with_decode_pool` calls keep their existing global-pool behaviour.
/// * `Err(_)` -- pool construction failed; run `f` inline with the flag `false`.
///   The inner `with_decode_pool` calls surface the same error and the forward
///   fails identically to the un-scoped path.
///
/// Callers should enter this scope only for the M=1 CPU decode case; prefill
/// (M>1) and non-CPU paths must keep using the global pool.
pub fn with_decode_pool_scope<R: Send>(
    model_uses_spmd_pool: bool,
    f: impl FnOnce() -> R + Send,
) -> R {
    // The persistent SPMD pool benefits quantized models whose decode kernels
    // dispatch output shards through it, including the MLAS QNBit shard route.
    // Keep the existing default/forced SPMD policy for all other callers.
    #[cfg(not(feature = "mlas"))]
    let spmd_pool_eligible = model_uses_spmd_pool
        || crate::decode_spmd::is_forced()
        || crate::decode_spmd::pools().is_some(); // default or adaptive
    #[cfg(feature = "mlas")]
    let spmd_pool_eligible = model_uses_spmd_pool || crate::decode_spmd::is_forced();
    if !spmd_pool_eligible {
        return with_dense_decode_pool_scope(f);
    }
    // The persistent SPMD pool is the default (unset or `=1`). Precedence:
    // explicit numa-split env > persistent SPMD > flat + auto-compact. The
    // "mutually exclusive" diagnostic below is scoped to users who set both.
    let both_requested = crate::decode_spmd::is_forced()
        && std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV)
            .is_ok_and(|value| value.trim() == "numa-split");
    // `numa-split`: run the forward on the dispatcher pool and let each M=1
    // projection fan its output rows out across the per-node sub-pools. The
    // decode-residency flag is set too, so the inner `with_decode_pool` calls
    // run inline on the dispatcher worker (they must not re-install the flat
    // single-node pool); the numa-scope flag makes `parallel_output_rows`
    // choose the two-level per-node dispatch.
    if let Some(numa) = numa_pools() {
        if both_requested {
            report_decode_strategy_precedence(
                "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the persistent \
                 SPMD decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL) are mutually \
                 exclusive; numa-split is active because it has precedence and its two-level \
                 NUMA layout was built successfully. Set \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=0 to silence this if intentional",
            );
        }
        return numa.install_scope(move || {
            let _numa_guard = NumaScopeGuard::enter();
            let _decode_guard = DecodeResidencyGuard::enter();
            f()
        });
    }
    // Persistent SPMD pool: run the forward inline on this (dispatcher) thread
    // and let each M=1 projection broadcast its output-row shards to the hot
    // persistent workers under one lightweight barrier. The decode-residency
    // flag makes inner `with_decode_pool` calls run inline (they must not
    // re-install the flat pool); the SPMD-scope flag routes `parallel_output_rows`
    // through the persistent pool.
    //
    // The pool is built for both `On` (default/`=1`) and `Adaptive` (`=auto`).
    // `On` always dispatches to it. `Adaptive` lets the calibrator time the same
    // token-exact step both ways and commit the faster path (defaulting to flat,
    // the safe choice under load) -- see `crate::decode_spmd::Calibrator`.
    if crate::decode_spmd::pools().is_some() {
        if both_requested {
            report_decode_strategy_precedence(
                "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the persistent \
                 SPMD decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL) are mutually \
                 exclusive; persistent SPMD is active because the higher-precedence \
                 numa-split layout was unavailable",
            );
        }
        if crate::decode_spmd::is_forced() {
            return with_spmd_decode_scope(f);
        }
        // `Adaptive` (`=auto`): measure the live decode step on the chosen path
        // and feed the timing back so the pool is adopted only when genuinely faster.
        return with_auto_calibrated_decode_scope(f);
    }
    if both_requested {
        report_decode_strategy_precedence(
            "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the persistent SPMD \
             decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL) are mutually exclusive; \
             neither strategy is active because no bounded decode worker count or usable \
             numa-split layout is available",
        );
    }
    with_flat_decode_pool_scope(f)
}

/// Run the forward inline under the persistent SPMD decode scope: the SPMD-scope
/// flag routes each projection's row-sharding through the persistent pool, and
/// the decode-residency flag makes inner `with_decode_pool` calls run inline.
fn with_spmd_decode_scope<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let _spmd_guard = SpmdScopeGuard::enter();
    let _decode_guard = DecodeResidencyGuard::enter();
    f()
}

/// Install the forward on the flat, bounded [`DECODE_POOL`] (the legacy decode
/// path, used by `=0` or when the SPMD pool cannot be built). When the pool opts
/// out (`ONNX_GENAI_CPU_DECODE_THREADS=0`) or fails to build, `f` runs on the
/// global Rayon pool, preserving correctness.
fn with_flat_decode_pool_scope<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match DECODE_POOL.get_or_init(|| build_decode_pool(configured_decode_threads())) {
        Ok(Some(pool)) => pool.install(move || {
            let _guard = DecodeResidencyGuard::enter();
            f()
        }),
        _ => f(),
    }
}

/// Adaptive-mode calibrate-and-pick (`=auto`): ask the calibrator which path this
/// decode step should take, time the *real* step on it, and feed the wall time
/// back. Both paths are token-exact, so the choice never changes the emitted
/// tokens -- only how fast the step runs. The calibrator keeps the flat path
/// committed by default and adopts the pool only when it measures faster, so a
/// loaded host stays on the flat path (no regression); see
/// `crate::decode_spmd::Calibrator`.
fn with_auto_calibrated_decode_scope<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    use crate::decode_spmd::AutoPath;
    let path = crate::decode_spmd::auto_choose_path();
    let start = std::time::Instant::now();
    let result = match path {
        AutoPath::Pool => with_spmd_decode_scope(f),
        AutoPath::Flat => with_flat_decode_pool_scope(f),
    };
    crate::decode_spmd::auto_record_sample(path, start.elapsed());
    result
}

/// Install `f` on the bounded, non-spinning [`DENSE_DECODE_POOL`] used by the
/// dense-f32 decode path. The multi-threaded MLAS GEMM behind each dense
/// `MatMul` tiles its work across this pool's Rayon workers; the decode-residency
/// flag makes any inner [`with_decode_pool`] calls run inline. When the pool
/// opts out (`ONNX_GENAI_CPU_DECODE_THREADS=0`) or fails to build, `f` runs on
/// the global Rayon pool, preserving correctness.
fn with_dense_decode_pool_scope<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match DENSE_DECODE_POOL.get_or_init(|| build_decode_pool(configured_dense_decode_threads())) {
        Ok(Some(pool)) => pool.install(move || {
            let _guard = DecodeResidencyGuard::enter();
            f()
        }),
        _ => f(),
    }
}

fn report_decode_strategy_precedence(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: decode strategy selection: {message}");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DotKernel {
    // On aarch64 the SIMD `Neon` kernel is baseline and always selected, so the
    // runtime `Scalar` variant is never constructed in library code there (it
    // still backs the parity tests and the dispatcher reference). Allow it to
    // avoid a dead-code error under `-D warnings` on aarch64.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    Scalar,
    /// AVX2, 256-bit, **no** VNNI. Exact `u8 x i8` int8 dot for the huge
    /// installed base of AVX2 CPUs without VNNI (Haswell/Broadwell/Skylake/
    /// Cascade-Lake client, all AMD Zen1/Zen2, most pre-Ice-Lake cloud
    /// instances). Consumes the natural (non-deinterleaved) activation layout —
    /// it MUST NOT be routed into the VNNI int4-direct path.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// ARM NEON / AdvSIMD exact `u8 x i8` int8 dot for aarch64. NEON is the
    /// baseline ISA on aarch64, so this is selected unconditionally on ARM
    /// (Apple Silicon, AWS Graviton, Ampere, Windows-on-ARM) to replace the slow
    /// scalar fallback. It consumes the natural (non-deinterleaved) activation
    /// layout and takes the int8 route — it MUST NOT enter the VNNI int4-direct
    /// path. The kernel uses a widen-`vmlal` baseline (stable, always-available
    /// AdvSIMD) that is bit-exact vs scalar.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// ARMv8.2-A dot-product int4 decode kernel. This consumes signed int8
    /// activations and signed int4 weights (`q - 8`), so it uses signed `sdot`
    /// via inline asm rather than the unavailable mixed unsigned/signed
    /// dot-product intrinsic.
    #[cfg(target_arch = "aarch64")]
    NeonDot,
}

/// The best `DotKernel` this host can actually execute, resolved once.
///
/// [`selected_dot_kernel`] does the CPUID probing; this caches it so
/// [`DotKernel::clamped_to_host`] costs one relaxed load per dot instead of a
/// feature-detection call per dot.
fn host_dot_kernel() -> DotKernel {
    static HOST: OnceLock<DotKernel> = OnceLock::new();
    *HOST.get_or_init(selected_dot_kernel)
}

/// Bitmask of the `DotKernel`s whose ISA this host implements, resolved once.
fn host_dot_kernel_mask() -> u8 {
    static MASK: OnceLock<u8> = OnceLock::new();
    *MASK.get_or_init(|| {
        let mut mask = DotKernel::Scalar.bit();
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                mask |= DotKernel::Avx2.bit();
                if std::arch::is_x86_feature_detected!("avxvnni") {
                    mask |= DotKernel::AvxVnni.bit();
                }
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                {
                    mask |= DotKernel::Avx512Vnni.bit();
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // AdvSIMD is baseline on aarch64 and `dot_u8_i8_neon` uses only
            // baseline intrinsics, so both NEON variants are always runnable;
            // `dotprod` selects between them for throughput, not for safety.
            mask |= DotKernel::Neon.bit() | DotKernel::NeonDot.bit();
        }
        mask
    })
}

impl DotKernel {
    /// One-hot tag used by [`host_dot_kernel_mask`].
    fn bit(self) -> u8 {
        match self {
            DotKernel::Scalar => 1 << 0,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2 => 1 << 1,
            #[cfg(target_arch = "x86_64")]
            DotKernel::AvxVnni => 1 << 2,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512Vnni => 1 << 3,
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon => 1 << 1,
            #[cfg(target_arch = "aarch64")]
            DotKernel::NeonDot => 1 << 2,
        }
    }

    /// Whether this host implements every instruction this kernel issues.
    ///
    /// Membership is tested per kernel rather than by ranking the ladder,
    /// because the x86 ISA ladder is **not** a total order: Ice Lake and Cooper
    /// Lake server parts ship AVX512-VNNI (EVEX `vpdpbusd`) with **no**
    /// AVX-VNNI (the VEX encoding), so "supports the stronger kernel" does not
    /// imply "supports the weaker one". A rank comparison would keep an
    /// `AvxVnni` request on such a host and fault.
    fn is_runnable_here(self) -> bool {
        host_dot_kernel_mask() & self.bit() != 0
    }

    /// Degrade a requested kernel to one this host can actually run.
    ///
    /// The `unsafe fn`s behind [`dot_u8_i8`] carry `#[target_feature]` for
    /// AVX2/AVX-VNNI/AVX-512-VNNI and are called on the strength of the
    /// `DotKernel` value alone. Producing that value from anywhere other than
    /// [`selected_dot_kernel`] -- a test, a future env override, a plumbed-through
    /// field that outlives a process migration -- would execute an instruction
    /// the CPU may not implement, i.e. `SIGILL`, from safe code. Rather than rely
    /// on every present and future caller having probed CPUID first, the
    /// dispatcher clamps: a request this host cannot run is answered by the
    /// host's own best kernel.
    ///
    /// Every kernel in the ladder is bit-exact against [`dot_u8_i8_scalar`]
    /// (asserted for all of them by `every_dot_kernel_is_bit_exact_on_this_host`),
    /// so the substitution costs throughput and never accuracy.
    ///
    /// This is deliberately *not* an assertion. Clamping is total, allocation
    /// free and bit-exact, so it is safe to leave enabled in release builds,
    /// where a `debug_assert!` would be compiled out -- restoring the `SIGILL`
    /// exactly where it is least debuggable.
    fn clamped_to_host(self) -> DotKernel {
        if self.is_runnable_here() {
            self
        } else {
            host_dot_kernel()
        }
    }
}

impl DotKernel {
    /// Whether this kernel consumes the VNNI-only *deinterleaved* int4
    /// activation layout and may take the int4-direct decode path. `Scalar`
    /// and `Avx2` use the natural layout / int8 route, so they must NOT enter
    /// that path (a wrong classification silently corrupts decode).
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    fn uses_vnni_int4_direct(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // `is_runnable_here` rather than `clamped_to_host`: the int4-direct
            // route is layout-coupled (the VNNI kernels consume the
            // deinterleaved activation of `deinterleave_activation_int4`, the
            // others natural order), so substituting a different kernel *inside*
            // `int4_dot_row` would silently mis-decode. Refusing the route
            // instead sends an unrunnable request down the int8 path, which
            // builds its own layout and is bit-exact -- a degrade that is
            // correct rather than merely safe.
            matches!(self, DotKernel::AvxVnni | DotKernel::Avx512Vnni) && self.is_runnable_here()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Whether this host kernel can consume the packed int4 weight directly for
    /// M=1 decode. x86 VNNI currently supports only block-32 because its
    /// deinterleaved activation layout is hard-coded at that granularity; the
    /// aarch64 dot-product kernel handles any quantization block that is a
    /// multiple of 32, including the Foundry Qwen3 block-128 graph.
    fn supports_int4_direct(self, block_size: usize, _has_zero_points: bool) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            !_has_zero_points && self.uses_vnni_int4_direct() && block_size == 32
        }
        #[cfg(target_arch = "aarch64")]
        {
            matches!(self, DotKernel::NeonDot)
                && block_size.is_multiple_of(32)
                && Self::arm64_kai_sdot_direct_enabled()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = block_size;
            false
        }
    }

    fn uses_n16_sdot_direct(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            matches!(self, DotKernel::NeonDot) && Self::arm64_int4_direct_enabled()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    fn uses_kai_sdot_direct(self, bits: usize, block_size: usize) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            matches!(self, DotKernel::NeonDot)
                && matches!(bits, 4 | 8)
                && block_size.is_multiple_of(32)
                && Self::arm64_kai_sdot_direct_enabled()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = bits;
            let _ = block_size;
            false
        }
    }

    /// Whether the aarch64 N16 SDOT direct int4 kernels are enabled.
    ///
    /// These kernels quantize the f32 activations to int8
    /// ([`quantize_activation_signed`]), so they are *not* numerically
    /// equivalent to the fp32 reference and must stay opt-in. The body used to
    /// be `#[cfg(test)] { true }` in front of a `#[cfg(not(test))]` production
    /// policy, which meant every test silently ran this opt-in,
    /// precision-reducing kernel while the shipped binary did not. Tests now
    /// observe the same default the shipped binary uses.
    #[cfg(target_arch = "aarch64")]
    fn arm64_int4_direct_enabled() -> bool {
        // The asymmetric N16 SDOT kernels are correctness-locked, but the
        // first full-model Qwen3 measurement regressed decode throughput.
        // Keep them opt-in while the microkernel is tightened.
        //
        // Resolved once: this gate is read from the per-token decode path,
        // where `std::env::var` would take the process-wide environment
        // lock and allocate on every generated token.
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("ONNX_GENAI_CPU_ARM64_INT4_DIRECT").is_ok_and(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0"
            })
        })
    }

    /// Whether the aarch64 KleidiAI SDOT direct kernels are enabled.
    ///
    /// Same test-fidelity fix as [`DotKernel::arm64_int4_direct_enabled`]: the
    /// previous `#[cfg(test)] { true }` forced this on under test even on
    /// macOS/iOS, where production defaults it off.
    #[cfg(target_arch = "aarch64")]
    fn arm64_kai_sdot_direct_enabled() -> bool {
        // Resolved once: see the note in `arm64_int4_direct_enabled`.
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            if let Ok(value) = std::env::var("ONNX_GENAI_CPU_ARM64_INT4_DIRECT") {
                let value = value.trim();
                return !value.is_empty() && value != "0";
            }
            // Validated against the f32 reference for the asymmetric block128
            // shapes Qwen3 actually emits, so this path is on by default.
            // Apple silicon keeps the existing route pending its own
            // measurement pass.
            !cfg!(any(target_os = "macos", target_os = "ios"))
        })
    }
}

fn selected_dot_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            return DotKernel::Avx512Vnni;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avxvnni")
        {
            return DotKernel::AvxVnni;
        }
        // AVX2 without any VNNI: still far faster than scalar for the int8
        // decode dot. Covers the large pre-VNNI installed base instead of
        // silently falling back to `Scalar`.
        if std::arch::is_x86_feature_detected!("avx2") {
            return DotKernel::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return DotKernel::NeonDot;
        }
        // NEON/AdvSIMD is baseline on aarch64, so the SIMD int8 dot is always
        // available — never fall back to the slow scalar path on ARM. The
        // kernel uses the widen-`vmlal` baseline, bit-exact vs scalar.
        DotKernel::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        DotKernel::Scalar
    }
}

/// Whether the running host selects MLAS's *AVX-512* SQNBit dispatch
/// (`MlasSQNBitGemmDispatchAvx512`). MLAS only installs it when the CPU has
/// AVX512F **and** the AVX-512 core trio BW+DQ+VL (vendored MLAS
/// `platform.cpp:572`, nested under the AVX512F check at `:547`); AVX512F alone
/// leaves `QNBitGemmDispatch` at the Avx2/Avx2vnni path (`:504`/`:537`), i.e. the
/// broken AVX2 M=1 asymmetric CompInt8 kernel. So we must mirror MLAS's exact
/// gate (F+BW+DQ+VL) here, not just AVX512F, or the guard leaks on AVX512F-only
/// hosts (Xeon Phi KNL/KNM). On non-x86-64 targets no such kernel exists so this
/// is always `false`.
#[cfg(feature = "mlas")]
fn prefer_arm64_mlas_qnbit_decode(
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    m: usize,
    no_group_indices: bool,
) -> bool {
    cfg!(all(
        target_arch = "aarch64",
        not(any(target_os = "macos", target_os = "ios"))
    )) && arm64_mlas_qnbit_decode_opted_in()
        && matches!(bits, 4 | 8)
        && block_size == 128
        && accuracy_level == 4
        && m == 1
        && no_group_indices
}

#[cfg(feature = "mlas")]
fn host_supports_mlas_sqnbit_m1_asym_int8() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        host_has_mlas_sqnbit_avx512()
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg(feature = "mlas")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn host_has_mlas_sqnbit_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Symmetric int8 activation quantization, per K-block, matching ORT/MLAS's
/// `QuantizeARow_CompInt8`: each `block_size`-wide block gets its own
/// `scale = max_abs_block / 127` and round-to-nearest int8 codes. A single
/// per-row scale (the previous scheme) let one outlier block inflate the scale
/// for the whole row, roughly doubling the CompInt8 error versus ORT; per-block
/// scaling closes that gap. Returns the padded int8 activations and one scale
/// per block (`padded_k / block_size` entries).
fn quantize_activation_signed(
    activation: &[f32],
    padded_k: usize,
    block_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    let k_blocks = padded_k / block_size;
    let mut quantized = vec![0i8; padded_k];
    let mut scales = vec![0.0f32; k_blocks];
    for (block, (out_block, scale)) in quantized
        .chunks_mut(block_size)
        .zip(scales.iter_mut())
        .enumerate()
    {
        let start = block * block_size;
        let real_end = (start + block_size).min(activation.len());
        if real_end <= start {
            continue;
        }
        let src = &activation[start..real_end];
        *scale = crate::kernels::simd_quant::quantize_block_i8(src, &mut out_block[..src.len()]);
    }
    (quantized, scales)
}

fn int4_matmul_m1(
    activation: &[f32],
    weight: &PackedInt4Weight,
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    debug_assert!(block_size.is_multiple_of(32));
    #[cfg(target_arch = "x86_64")]
    debug_assert!(!dot_kernel.uses_vnni_int4_direct() || block_size == 32);
    let packed_block_size = block_size / 2;
    let k_blocks = k.div_ceil(block_size);
    let padded_k = k_blocks * block_size;
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight.values.len(), n * k_blocks * packed_block_size);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(result.len(), n);

    let (activation, activation_scales) =
        quantize_activation_signed(activation, padded_k, block_size);
    // The SIMD int4 kernels consume a deinterleaved activation layout (evens
    // then odds per 32-block) so they can skip the per-block nibble
    // deinterleave; the scalar reference keeps natural order. Deinterleave once
    // here, amortized over all N output rows.
    #[cfg(target_arch = "x86_64")]
    let use_simd = dot_kernel.uses_vnni_int4_direct();
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    let deinterleaved;
    let activation: &[i8] = if use_simd {
        deinterleaved = deinterleave_activation_int4(&activation);
        &deinterleaved
    } else {
        &activation
    };
    // The int4 zero-point correction is `8 * sum(activation)` per K-block, which
    // depends only on the (deinterleaved) activation — it is identical for every
    // one of the N output columns. The AVX-512 VNNI kernel used to recompute it
    // per column via a second `vpdpbusd` against all-ones (doubling the VNNI-port
    // work in the hot loop). Precompute it once here (already `<< 3`) in the exact
    // per-lane `vpdpbusd(ones, act)` layout so the kernel just loads and subtracts,
    // keeping the integer result bit-identical while halving the hot-loop dpbusd
    // count. Only the AVX-512 kernel consumes it; other kernels ignore the slice.
    #[cfg(target_arch = "x86_64")]
    let precompute_act_sums = matches!(dot_kernel, DotKernel::Avx512Vnni);
    #[cfg(not(target_arch = "x86_64"))]
    let precompute_act_sums = false;
    let act_sum8: Vec<i32> = if precompute_act_sums {
        activation_block_sums8(activation, k_blocks)
    } else {
        Vec::new()
    };
    #[cfg(test)]
    INT4_DIRECT_M1_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (offset, output) in outputs.iter_mut().enumerate() {
            let output_index = output_start + offset;
            let packed_start = output_index * k_blocks * packed_block_size;
            let packed_end = packed_start + k_blocks * packed_block_size;
            let scale_start = output_index * k_blocks;
            let scale_end = scale_start + k_blocks;
            *output = int4_dot_row(
                activation,
                &weight.values[packed_start..packed_end],
                &weight.scales[scale_start..scale_end],
                &activation_scales,
                &act_sum8,
                block_size,
                dot_kernel,
            );
        }
    };

    parallel_output_rows(result, padded_k, compute);
}

fn prepack_kai_sdot_from_bytes(
    packed: &[u8],
    scales: Vec<f32>,
    zero_points: Option<&[u8]>,
    n: usize,
    k: usize,
    bits: usize,
    block_size: usize,
) -> PackedKaiSdotWeight {
    debug_assert!(matches!(bits, 4 | 8));
    debug_assert!(block_size.is_multiple_of(KAI_SDOT_K_GROUP));
    let layout = NBitsLayout { bits, block_size };
    let k_blocks = k.div_ceil(block_size);
    let groups_per_block = block_size / KAI_SDOT_K_GROUP;
    let payload = if bits == 4 { 2 } else { KAI_SDOT_K_GROUP };
    let tile_count = n.div_ceil(KAI_SDOT_OUTPUTS);
    let group_stride = KAI_SDOT_OUTPUTS * payload;
    let tile_stride = k_blocks * groups_per_block * group_stride;
    let packed_block_size = layout.packed_block_size();
    let zero_point_row_size = layout.zero_point_row_size(k_blocks);
    let midpoint = 1i16 << (bits - 1);
    let mut values = vec![0u8; tile_count * tile_stride];
    let meta_stride = k_blocks * KAI_SDOT_OUTPUTS;
    let mut packed_scales = vec![0.0f32; tile_count * meta_stride];
    let mut rhs_sums = vec![0i32; tile_count * meta_stride];
    let mut zero_point_offsets = vec![0i16; tile_count * meta_stride];

    for output in 0..n {
        let tile = output / KAI_SDOT_OUTPUTS;
        let lane = output % KAI_SDOT_OUTPUTS;
        for block in 0..k_blocks {
            let meta = tile * meta_stride + block * KAI_SDOT_OUTPUTS + lane;
            let zero_point = zero_points.map_or(midpoint as u8, |points| {
                layout.zero_point(
                    Some(
                        &points[output * zero_point_row_size
                            ..output * zero_point_row_size + zero_point_row_size],
                    ),
                    block,
                )
            });
            packed_scales[meta] = scales[output * k_blocks + block];
            zero_point_offsets[meta] = midpoint - zero_point as i16;
            let mut sum = 0i32;
            for group in 0..groups_per_block {
                let base = tile * tile_stride
                    + (block * groups_per_block + group) * group_stride
                    + lane * payload;
                let mut centered_group = [0i8; KAI_SDOT_K_GROUP];
                for (kk, centered_slot) in centered_group.iter_mut().enumerate() {
                    let within_block = group * KAI_SDOT_K_GROUP + kk;
                    let depth = block * block_size + within_block;
                    let centered = if depth < k {
                        let block_start = (output * k_blocks + block) * packed_block_size;
                        let q = layout.unpack(
                            &packed[block_start..block_start + packed_block_size],
                            within_block,
                        );
                        q as i16 - midpoint
                    } else {
                        0
                    };
                    *centered_slot = centered as i8;
                    sum += centered as i32;
                }
                if bits == 4 {
                    values[base] = ((centered_group[0] + 8) as u8 & 0x0f)
                        | (((centered_group[1] + 8) as u8 & 0x0f) << 4);
                    values[base + 1] = ((centered_group[2] + 8) as u8 & 0x0f)
                        | (((centered_group[3] + 8) as u8 & 0x0f) << 4);
                } else {
                    for (kk, centered) in centered_group.iter().enumerate() {
                        values[base + kk] = *centered as u8;
                    }
                }
            }
            rhs_sums[meta] = sum;
        }
    }

    PackedKaiSdotWeight {
        bits,
        values,
        scales: packed_scales,
        rhs_sums,
        zero_point_offsets,
    }
}

struct Qai8dxpActivation {
    values: Vec<i8>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    group_words: Vec<u32>,
    scale: f32,
    zero_point: i32,
    block_sums: Vec<i32>,
    block_counts: Vec<i32>,
}

fn quantize_activation_qai8dxp(
    activation: &[f32],
    padded_k: usize,
    block_size: usize,
) -> Qai8dxpActivation {
    let k_blocks = padded_k / block_size;
    let mut min = 0.0f32;
    let mut max = 0.0f32;
    for &value in activation {
        min = min.min(value);
        max = max.max(value);
    }
    let qmin = i8::MIN as f32;
    let qmax = i8::MAX as f32;
    let qscale = if min == max {
        1.0
    } else {
        (qmax - qmin) / (max - min)
    };
    let scale = if qscale == 0.0 { 1.0 } else { 1.0 / qscale };
    let zero_from_min = qmin - min * qscale;
    let zero_from_max = qmax - max * qscale;
    let zero_point = if zero_from_min + zero_from_max > 0.0 {
        zero_from_min
    } else {
        zero_from_max
    }
    .round()
    .clamp(qmin, qmax) as i32;
    let mut values = vec![zero_point as i8; padded_k];
    let mut block_sums = vec![0i32; k_blocks];
    let mut block_counts = vec![0i32; k_blocks];
    for (idx, &value) in activation.iter().enumerate() {
        let q = (value * qscale).round() as i32 + zero_point;
        let q = q.clamp(i8::MIN as i32, i8::MAX as i32);
        values[idx] = q as i8;
        let block = idx / block_size;
        block_sums[block] += q;
        block_counts[block] += 1;
    }
    let group_words = values
        .chunks_exact(KAI_SDOT_K_GROUP)
        .map(|group| {
            u32::from_le_bytes([
                group[0] as u8,
                group[1] as u8,
                group[2] as u8,
                group[3] as u8,
            ])
        })
        .collect();
    Qai8dxpActivation {
        values,
        group_words,
        scale,
        zero_point,
        block_sums,
        block_counts,
    }
}

fn kai_sdot_matmul_m1(
    activation: &[f32],
    weight: &PackedKaiSdotWeight,
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    debug_assert!(matches!(weight.bits, 4 | 8));
    debug_assert!(block_size.is_multiple_of(KAI_SDOT_K_GROUP));
    let k_blocks = k.div_ceil(block_size);
    let padded_k = k_blocks * block_size;
    let packed_meta_len = n.div_ceil(KAI_SDOT_OUTPUTS) * k_blocks * KAI_SDOT_OUTPUTS;
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight.scales.len(), packed_meta_len);
    debug_assert_eq!(weight.rhs_sums.len(), packed_meta_len);
    debug_assert_eq!(weight.zero_point_offsets.len(), packed_meta_len);
    debug_assert_eq!(result.len(), n);

    let activation = quantize_activation_qai8dxp(activation, padded_k, block_size);
    #[cfg(test)]
    KAI_SDOT_M1_TEST_CALLS.fetch_add(1, Ordering::Relaxed);

    let compute = |output_start: usize, outputs: &mut [f32]| {
        #[cfg(target_arch = "aarch64")]
        if matches!(dot_kernel, DotKernel::NeonDot) {
            // SAFETY: `DotKernel::NeonDot` is selected only after runtime dotprod
            // detection; the prepack layout is validated by debug assertions.
            unsafe {
                kai_sdot_matmul_m1_neon_dot(
                    &activation,
                    weight,
                    output_start,
                    outputs,
                    n,
                    k_blocks,
                    block_size,
                );
            }
            return;
        }

        let _ = dot_kernel;
        kai_sdot_matmul_m1_scalar(
            &activation,
            weight,
            output_start,
            outputs,
            k_blocks,
            block_size,
        );
    };
    parallel_kai_output_rows(result, padded_k, compute);
}

fn parallel_kai_output_rows<F>(result: &mut [f32], k: usize, compute: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(spmd) = spmd_decode_active() {
        spmd.dispatch_output_rows_indexed(result, KAI_SDOT_OUTPUTS, &|_, start, outputs| {
            compute(start, outputs)
        });
        return;
    }
    if let Some(numa) = numa_decode_active() {
        numa.dispatch_output_rows(result, k, &compute);
        return;
    }
    let chunk = output_chunk_len(result.len(), k);
    let chunk = if chunk < result.len() {
        chunk.div_ceil(KAI_SDOT_OUTPUTS) * KAI_SDOT_OUTPUTS
    } else {
        chunk
    };
    if chunk < result.len() {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

fn kai_group_payload(bits: usize) -> usize {
    if bits == 4 { 2 } else { KAI_SDOT_K_GROUP }
}

#[inline]
fn kai_meta_index(output: usize, block: usize, k_blocks: usize) -> usize {
    let tile = output / KAI_SDOT_OUTPUTS;
    let lane = output % KAI_SDOT_OUTPUTS;
    tile * k_blocks * KAI_SDOT_OUTPUTS + block * KAI_SDOT_OUTPUTS + lane
}

fn kai_centered_weight(
    weight: &PackedKaiSdotWeight,
    output: usize,
    block: usize,
    group: usize,
    kk: usize,
    k_blocks: usize,
    groups_per_block: usize,
) -> i32 {
    let payload = kai_group_payload(weight.bits);
    let group_stride = KAI_SDOT_OUTPUTS * payload;
    let tile_stride = k_blocks * groups_per_block * group_stride;
    let tile = output / KAI_SDOT_OUTPUTS;
    let lane = output % KAI_SDOT_OUTPUTS;
    let base =
        tile * tile_stride + (block * groups_per_block + group) * group_stride + lane * payload;
    if weight.bits == 4 {
        let byte = weight.values[base + kk / 2];
        let q = if kk.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        q as i32 - 8
    } else {
        weight.values[base + kk] as i8 as i32
    }
}

fn kai_sdot_matmul_m1_scalar(
    activation: &Qai8dxpActivation,
    weight: &PackedKaiSdotWeight,
    output_start: usize,
    result: &mut [f32],
    k_blocks: usize,
    block_size: usize,
) {
    let groups_per_block = block_size / KAI_SDOT_K_GROUP;
    for (offset, output_value) in result.iter_mut().enumerate() {
        let output = output_start + offset;
        let mut total = 0.0f32;
        for block in 0..k_blocks {
            let mut acc = 0i32;
            for group in 0..groups_per_block {
                let activation_base = block * block_size + group * KAI_SDOT_K_GROUP;
                for kk in 0..KAI_SDOT_K_GROUP {
                    let a = activation.values[activation_base + kk] as i32;
                    let w = kai_centered_weight(
                        weight,
                        output,
                        block,
                        group,
                        kk,
                        k_blocks,
                        groups_per_block,
                    );
                    acc += a * w;
                }
            }
            let idx = kai_meta_index(output, block, k_blocks);
            let correction = kai_qai8_correction(
                activation,
                weight.rhs_sums[idx],
                weight.zero_point_offsets[idx] as i32,
                block,
            );
            total += (acc + correction) as f32 * (activation.scale * weight.scales[idx]);
        }
        *output_value = total;
    }
}

#[inline]
fn kai_qai8_correction(
    activation: &Qai8dxpActivation,
    rhs_sum: i32,
    zp_offset: i32,
    block: usize,
) -> i32 {
    let a_zp = activation.zero_point;
    -a_zp * rhs_sum
        + zp_offset * (activation.block_sums[block] - a_zp * activation.block_counts[block])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn kai_sdot_matmul_m1_neon_dot(
    activation: &Qai8dxpActivation,
    weight: &PackedKaiSdotWeight,
    output_start: usize,
    result: &mut [f32],
    n: usize,
    k_blocks: usize,
    block_size: usize,
) {
    let mut offset = 0usize;
    while offset < result.len() {
        let output = output_start + offset;
        if output.is_multiple_of(KAI_SDOT_OUTPUTS)
            && output + KAI_SDOT_OUTPUTS <= n
            && offset + KAI_SDOT_OUTPUTS <= result.len()
        {
            let out =
                unsafe { kai_sdot_tile_neon(activation, weight, output, k_blocks, block_size) };
            result[offset..offset + KAI_SDOT_OUTPUTS].copy_from_slice(&out);
            offset += KAI_SDOT_OUTPUTS;
        } else {
            kai_sdot_matmul_m1_scalar(
                activation,
                weight,
                output,
                &mut result[offset..offset + 1],
                k_blocks,
                block_size,
            );
            offset += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn kai_sdot_tile_neon(
    activation: &Qai8dxpActivation,
    weight: &PackedKaiSdotWeight,
    output_start: usize,
    k_blocks: usize,
    block_size: usize,
) -> [f32; KAI_SDOT_OUTPUTS] {
    use std::arch::aarch64::*;

    let groups_per_block = block_size / KAI_SDOT_K_GROUP;
    let payload = kai_group_payload(weight.bits);
    let group_stride = KAI_SDOT_OUTPUTS * payload;
    let tile_stride = k_blocks * groups_per_block * group_stride;
    let tile = output_start / KAI_SDOT_OUTPUTS;
    let meta_tile_base = tile * k_blocks * KAI_SDOT_OUTPUTS;
    let mut total0 = vdupq_n_f32(0.0);
    let mut total1 = vdupq_n_f32(0.0);
    let mut total2 = vdupq_n_f32(0.0);
    let mut total3 = vdupq_n_f32(0.0);
    for block in 0..k_blocks {
        let mut acc0_even = vdupq_n_s32(0);
        let mut acc1_even = vdupq_n_s32(0);
        let mut acc2_even = vdupq_n_s32(0);
        let mut acc3_even = vdupq_n_s32(0);
        let mut acc0_odd = vdupq_n_s32(0);
        let mut acc1_odd = vdupq_n_s32(0);
        let mut acc2_odd = vdupq_n_s32(0);
        let mut acc3_odd = vdupq_n_s32(0);
        for group in 0..groups_per_block {
            let word = activation.group_words[block * groups_per_block + group];
            let act = vreinterpretq_s8_u32(vdupq_n_u32(word));
            let weight_base =
                tile * tile_stride + (block * groups_per_block + group) * group_stride;
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                let ahead = if weight.bits == 4 { 16 } else { 8 };
                if group + ahead < groups_per_block {
                    let prefetch_base = tile * tile_stride
                        + (block * groups_per_block + group + ahead) * group_stride;
                    unsafe {
                        kai_prefetch_l1(weight.values.as_ptr().wrapping_add(prefetch_base));
                    }
                }
            }
            let (acc0, acc1, acc2, acc3) = if group.is_multiple_of(2) {
                (
                    &mut acc0_even,
                    &mut acc1_even,
                    &mut acc2_even,
                    &mut acc3_even,
                )
            } else {
                (&mut acc0_odd, &mut acc1_odd, &mut acc2_odd, &mut acc3_odd)
            };
            if weight.bits == 4 {
                let packed0 = unsafe { vld1q_u8(weight.values.as_ptr().add(weight_base)) };
                let low0 = vandq_u8(packed0, vdupq_n_u8(0x0f));
                let high0 = vshrq_n_u8::<4>(packed0);
                let w0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(low0, high0)), vdupq_n_s8(8));
                let w1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(low0, high0)), vdupq_n_s8(8));
                let packed1 = unsafe { vld1q_u8(weight.values.as_ptr().add(weight_base + 16)) };
                let low1 = vandq_u8(packed1, vdupq_n_u8(0x0f));
                let high1 = vshrq_n_u8::<4>(packed1);
                let w2 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(low1, high1)), vdupq_n_s8(8));
                let w3 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(low1, high1)), vdupq_n_s8(8));
                *acc0 = unsafe { sdot_i8x16(*acc0, act, w0) };
                *acc1 = unsafe { sdot_i8x16(*acc1, act, w1) };
                *acc2 = unsafe { sdot_i8x16(*acc2, act, w2) };
                *acc3 = unsafe { sdot_i8x16(*acc3, act, w3) };
            } else {
                let ptr = unsafe { weight.values.as_ptr().add(weight_base) as *const i8 };
                let w0 = unsafe { vld1q_s8(ptr) };
                let w1 = unsafe { vld1q_s8(ptr.add(16)) };
                let w2 = unsafe { vld1q_s8(ptr.add(32)) };
                let w3 = unsafe { vld1q_s8(ptr.add(48)) };
                *acc0 = unsafe { sdot_i8x16(*acc0, act, w0) };
                *acc1 = unsafe { sdot_i8x16(*acc1, act, w1) };
                *acc2 = unsafe { sdot_i8x16(*acc2, act, w2) };
                *acc3 = unsafe { sdot_i8x16(*acc3, act, w3) };
            }
        }
        let acc0 = vaddq_s32(acc0_even, acc0_odd);
        let acc1 = vaddq_s32(acc1_even, acc1_odd);
        let acc2 = vaddq_s32(acc2_even, acc2_odd);
        let acc3 = vaddq_s32(acc3_even, acc3_odd);
        let meta = meta_tile_base + block * KAI_SDOT_OUTPUTS;
        let rhs = unsafe { weight.rhs_sums.as_ptr().add(meta) };
        let zp = unsafe { weight.zero_point_offsets.as_ptr().add(meta) };
        let scale = unsafe { weight.scales.as_ptr().add(meta) };
        let centered_activation_sum =
            activation.block_sums[block] - activation.zero_point * activation.block_counts[block];
        let corr0 = unsafe {
            kai_qai8_correction_vec_neon(
                activation.zero_point,
                centered_activation_sum,
                vld1q_s32(rhs),
                vmovl_s16(vld1_s16(zp)),
            )
        };
        let corr1 = unsafe {
            kai_qai8_correction_vec_neon(
                activation.zero_point,
                centered_activation_sum,
                vld1q_s32(rhs.add(4)),
                vmovl_s16(vld1_s16(zp.add(4))),
            )
        };
        let corr2 = unsafe {
            kai_qai8_correction_vec_neon(
                activation.zero_point,
                centered_activation_sum,
                vld1q_s32(rhs.add(8)),
                vmovl_s16(vld1_s16(zp.add(8))),
            )
        };
        let corr3 = unsafe {
            kai_qai8_correction_vec_neon(
                activation.zero_point,
                centered_activation_sum,
                vld1q_s32(rhs.add(12)),
                vmovl_s16(vld1_s16(zp.add(12))),
            )
        };
        total0 = unsafe { kai_accumulate_block_neon(acc0, corr0, scale, activation.scale, total0) };
        total1 = unsafe {
            kai_accumulate_block_neon(acc1, corr1, scale.add(4), activation.scale, total1)
        };
        total2 = unsafe {
            kai_accumulate_block_neon(acc2, corr2, scale.add(8), activation.scale, total2)
        };
        total3 = unsafe {
            kai_accumulate_block_neon(acc3, corr3, scale.add(12), activation.scale, total3)
        };
    }
    let mut out = [0.0f32; KAI_SDOT_OUTPUTS];
    unsafe {
        vst1q_f32(out.as_mut_ptr(), total0);
        vst1q_f32(out.as_mut_ptr().add(4), total1);
        vst1q_f32(out.as_mut_ptr().add(8), total2);
        vst1q_f32(out.as_mut_ptr().add(12), total3);
    };
    out
}

#[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "macos", target_os = "ios"))
))]
#[inline(always)]
unsafe fn kai_prefetch_l1(ptr: *const u8) {
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly, preserves_flags)
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn kai_qai8_correction_vec_neon(
    activation_zero_point: i32,
    centered_activation_sum: i32,
    rhs_sum: std::arch::aarch64::int32x4_t,
    zero_point_offset: std::arch::aarch64::int32x4_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    vaddq_s32(
        vmulq_n_s32(rhs_sum, -activation_zero_point),
        vmulq_n_s32(zero_point_offset, centered_activation_sum),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn kai_accumulate_block_neon(
    acc: std::arch::aarch64::int32x4_t,
    correction: std::arch::aarch64::int32x4_t,
    scale: *const f32,
    activation_scale: f32,
    total: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let block_i32 = vaddq_s32(acc, correction);
    let block_f32 = vcvtq_f32_s32(block_i32);
    let scale = vmulq_n_f32(unsafe { vld1q_f32(scale) }, activation_scale);
    vaddq_f32(total, vmulq_f32(block_f32, scale))
}

fn prepack_n16_sdot_from_bytes(
    packed: &[u8],
    scales: Vec<f32>,
    zero_points: Option<&[u8]>,
    n: usize,
    k: usize,
    bits: usize,
    block_size: usize,
) -> PackedN16SdotWeight {
    debug_assert!(matches!(bits, 4 | 8));
    debug_assert!(block_size.is_multiple_of(N16_SDOT_K_GROUP));
    let k_blocks = k.div_ceil(block_size);
    let packed_block_size = block_size * bits / 8;
    let groups_per_block = block_size / N16_SDOT_K_GROUP;
    let tile_count = n.div_ceil(N16_SDOT_OUTPUTS);
    let tile_stride = k_blocks * groups_per_block * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP;
    let zero_point_row_size = (k_blocks * bits).div_ceil(8);
    let midpoint = 1i16 << (bits - 1);
    let mut values = vec![0i8; tile_count * tile_stride];
    let mut zero_point_offsets = vec![0i16; n * k_blocks];

    for output in 0..n {
        let tile = output / N16_SDOT_OUTPUTS;
        let lane = output % N16_SDOT_OUTPUTS;
        for block in 0..k_blocks {
            let zero_point = zero_points.map_or(midpoint as u8, |points| {
                unpack_nbits(
                    &points[output * zero_point_row_size
                        ..output * zero_point_row_size + zero_point_row_size],
                    block,
                    bits,
                )
            });
            zero_point_offsets[output * k_blocks + block] = midpoint - zero_point as i16;
            for group in 0..groups_per_block {
                for kk in 0..N16_SDOT_K_GROUP {
                    let within_block = group * N16_SDOT_K_GROUP + kk;
                    let depth = block * block_size + within_block;
                    let centered = if depth < k {
                        let block_start = (output * k_blocks + block) * packed_block_size;
                        let q = unpack_nbits(
                            &packed[block_start..block_start + packed_block_size],
                            within_block,
                            bits,
                        );
                        q as i16 - midpoint
                    } else {
                        0
                    };
                    let index = tile * tile_stride
                        + (block * groups_per_block + group) * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP
                        + lane * N16_SDOT_K_GROUP
                        + kk;
                    values[index] = centered as i8;
                }
            }
        }
    }

    PackedN16SdotWeight {
        values,
        scales,
        zero_point_offsets,
    }
}

#[inline]
fn unpack_nbits(packed: &[u8], index: usize, bits: usize) -> u8 {
    if bits == 8 {
        packed[index]
    } else {
        let values_per_byte = 8 / bits;
        let mask = (1u8 << bits) - 1;
        (packed[index / values_per_byte] >> ((index % values_per_byte) * bits)) & mask
    }
}

#[allow(clippy::too_many_arguments)]
fn n16_sdot_matmul_m1(
    activation: &[f32],
    weight: &PackedN16SdotWeight,
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    debug_assert!(block_size.is_multiple_of(N16_SDOT_K_GROUP));
    let k_blocks = k.div_ceil(block_size);
    let padded_k = k_blocks * block_size;
    let groups_per_block = block_size / N16_SDOT_K_GROUP;
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(weight.zero_point_offsets.len(), n * k_blocks);
    debug_assert_eq!(result.len(), n);

    let (activation, activation_scales) =
        quantize_activation_signed(activation, padded_k, block_size);
    let activation_sums = activation_signed_block_sums(&activation, k_blocks, block_size);
    #[cfg(test)]
    N16_SDOT_M1_TEST_CALLS.fetch_add(1, Ordering::Relaxed);

    let compute = |output_start: usize, outputs: &mut [f32]| {
        #[cfg(target_arch = "aarch64")]
        if matches!(dot_kernel, DotKernel::NeonDot) {
            // SAFETY: `DotKernel::NeonDot` is selected only after runtime dotprod
            // detection; the prepack layout is validated by the debug assertions.
            unsafe {
                n16_sdot_matmul_m1_neon_dot(
                    &activation,
                    &activation_scales,
                    &activation_sums,
                    weight,
                    output_start,
                    outputs,
                    n,
                    k_blocks,
                    groups_per_block,
                );
            }
            return;
        }

        let _ = dot_kernel;
        n16_sdot_matmul_m1_scalar(
            &activation,
            &activation_scales,
            &activation_sums,
            weight,
            output_start,
            outputs,
            k_blocks,
            groups_per_block,
        );
    };
    parallel_n16_output_rows(result, padded_k, compute);
}

fn parallel_n16_output_rows<F>(result: &mut [f32], k: usize, compute: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(spmd) = spmd_decode_active() {
        spmd.dispatch_output_rows_indexed(result, N16_SDOT_OUTPUTS, &|_, start, outputs| {
            compute(start, outputs)
        });
        return;
    }
    if let Some(numa) = numa_decode_active() {
        numa.dispatch_output_rows(result, k, &compute);
        return;
    }
    let chunk = output_chunk_len(result.len(), k);
    let chunk = if chunk < result.len() {
        chunk.div_ceil(N16_SDOT_OUTPUTS) * N16_SDOT_OUTPUTS
    } else {
        chunk
    };
    if chunk < result.len() {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

fn activation_signed_block_sums(activation: &[i8], k_blocks: usize, block_size: usize) -> Vec<i32> {
    let mut sums = vec![0i32; k_blocks];
    for (block, sum) in sums.iter_mut().enumerate() {
        let start = block * block_size;
        let end = start + block_size;
        *sum = activation[start..end]
            .iter()
            .map(|&value| value as i32)
            .sum();
    }
    sums
}

#[allow(clippy::too_many_arguments)]
fn n16_sdot_matmul_m1_scalar(
    activation: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    result: &mut [f32],
    k_blocks: usize,
    groups_per_block: usize,
) {
    for (offset, output_value) in result.iter_mut().enumerate() {
        let output = output_start + offset;
        *output_value = n16_sdot_int4_output_scalar(
            activation,
            activation_scales,
            activation_sums,
            weight,
            output,
            k_blocks,
            groups_per_block,
        );
    }
}

fn n16_sdot_int4_output_scalar(
    activation: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i32],
    weight: &PackedN16SdotWeight,
    output: usize,
    k_blocks: usize,
    groups_per_block: usize,
) -> f32 {
    let tile_stride = k_blocks * groups_per_block * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP;
    let tile = output / N16_SDOT_OUTPUTS;
    let lane = output % N16_SDOT_OUTPUTS;
    let mut total = 0.0f32;
    for block in 0..k_blocks {
        let mut dot =
            weight.zero_point_offsets[output * k_blocks + block] as i32 * activation_sums[block];
        for group in 0..groups_per_block {
            let base = tile * tile_stride
                + (block * groups_per_block + group) * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP
                + lane * N16_SDOT_K_GROUP;
            let activation_base =
                block * groups_per_block * N16_SDOT_K_GROUP + group * N16_SDOT_K_GROUP;
            for kk in 0..N16_SDOT_K_GROUP {
                dot += weight.values[base + kk] as i32 * activation[activation_base + kk] as i32;
            }
        }
        total += dot as f32 * (weight.scales[output * k_blocks + block] * activation_scales[block]);
    }
    total
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn n16_sdot_u8_i16_matmul_m1(
    activation: &[f32],
    weight: &PackedN16SdotWeight,
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    debug_assert_eq!(block_size, 128);
    debug_assert_eq!(activation_quant_group(), 32);
    let k_blocks = k.div_ceil(block_size);
    let k_groups_32 = k.div_ceil(32);
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(weight.zero_point_offsets.len(), n * k_blocks);
    debug_assert_eq!(result.len(), n);

    let mut quantized = vec![0i16; k];
    let mut group_scales = vec![0.0f32; k_groups_32];
    let mut group_sums = vec![0i32; k_groups_32];
    let mut block_activation_sums = vec![0.0f32; k_blocks];
    for group in 0..k_groups_32 {
        let start = group * 32;
        let end = (start + 32).min(k);
        group_scales[group] =
            quantize_block_i16(&activation[start..end], &mut quantized[start..end]);
        group_sums[group] = quantized[start..end]
            .iter()
            .map(|&value| value as i32)
            .sum();
    }
    for block in 0..k_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(k);
        block_activation_sums[block] = activation[start..end].iter().sum();
    }
    #[cfg(test)]
    N16_SDOT_M1_TEST_CALLS.fetch_add(1, Ordering::Relaxed);

    let compute = |output_start: usize, outputs: &mut [f32]| {
        #[cfg(target_arch = "aarch64")]
        if matches!(dot_kernel, DotKernel::NeonDot) {
            // SAFETY: `DotKernel::NeonDot` is selected only after runtime dotprod
            // detection; block/group sizes are fixed by the guards above.
            unsafe {
                n16_sdot_u8_i16_matmul_m1_neon_dot(
                    &quantized,
                    &group_scales,
                    &group_sums,
                    &block_activation_sums,
                    weight,
                    output_start,
                    outputs,
                    n,
                    k_blocks,
                );
            }
            return;
        }

        let _ = dot_kernel;
        n16_sdot_u8_i16_matmul_m1_scalar(
            &quantized,
            &group_scales,
            &group_sums,
            &block_activation_sums,
            weight,
            output_start,
            outputs,
            k_blocks,
        );
    };
    parallel_n16_output_rows(result, k, compute);
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn n16_sdot_u8_i16_matmul_m1_scalar(
    activation: &[i16],
    group_scales: &[f32],
    group_sums: &[i32],
    block_activation_sums: &[f32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    result: &mut [f32],
    k_blocks: usize,
) {
    for (offset, output_value) in result.iter_mut().enumerate() {
        let output = output_start + offset;
        *output_value = n16_sdot_bits8_output_scalar(
            activation,
            group_scales,
            group_sums,
            block_activation_sums,
            weight,
            output,
            k_blocks,
        );
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn n16_sdot_bits8_output_scalar(
    activation: &[i16],
    group_scales: &[f32],
    group_sums: &[i32],
    block_activation_sums: &[f32],
    weight: &PackedN16SdotWeight,
    output: usize,
    k_blocks: usize,
) -> f32 {
    let groups_per_block = 4usize;
    let groups_per_block_4 = 32 / N16_SDOT_K_GROUP;
    let tile_stride = k_blocks * 128 / N16_SDOT_K_GROUP * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP;
    let tile = output / N16_SDOT_OUTPUTS;
    let lane = output % N16_SDOT_OUTPUTS;
    let mut total = 0.0f32;
    for (block, &block_sum) in block_activation_sums.iter().enumerate().take(k_blocks) {
        let mut product = 0.0f32;
        for group_in_block in 0..groups_per_block {
            let group = block * groups_per_block + group_in_block;
            let mut centered_dot = 0i32;
            for sub in 0..groups_per_block_4 {
                let group4 = group_in_block * groups_per_block_4 + sub;
                let base = tile * tile_stride
                    + (block * 128 / N16_SDOT_K_GROUP + group4)
                        * N16_SDOT_OUTPUTS
                        * N16_SDOT_K_GROUP
                    + lane * N16_SDOT_K_GROUP;
                let activation_base = group * 32 + sub * N16_SDOT_K_GROUP;
                for kk in 0..N16_SDOT_K_GROUP {
                    if activation_base + kk < activation.len() {
                        centered_dot += weight.values[base + kk] as i32
                            * activation[activation_base + kk] as i32;
                    }
                }
            }
            let q_dot = centered_dot + 128 * group_sums[group];
            product += group_scales[group] * q_dot as f32;
        }
        let scale = weight.scales[output * k_blocks + block];
        let zero_point = 128i32 - weight.zero_point_offsets[output * k_blocks + block] as i32;
        total += scale * product - scale * zero_point as f32 * block_sum;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn n16_sdot_matmul_m1_neon_dot(
    activation: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    result: &mut [f32],
    n: usize,
    k_blocks: usize,
    groups_per_block: usize,
) {
    let mut offset = 0usize;
    while offset < result.len() {
        let output = output_start + offset;
        if output.is_multiple_of(4) && output + 4 <= n && offset + 4 <= result.len() {
            let out = unsafe {
                n16_sdot_int4_quad_neon(
                    activation,
                    activation_scales,
                    activation_sums,
                    weight,
                    output,
                    k_blocks,
                    groups_per_block,
                )
            };
            result[offset..offset + 4].copy_from_slice(&out);
            offset += 4;
        } else {
            result[offset] = n16_sdot_int4_output_scalar(
                activation,
                activation_scales,
                activation_sums,
                weight,
                output,
                k_blocks,
                groups_per_block,
            );
            offset += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn n16_sdot_int4_quad_neon(
    activation: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    k_blocks: usize,
    groups_per_block: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    debug_assert!(output_start.is_multiple_of(4));
    let tile_stride = k_blocks * groups_per_block * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP;
    let tile = output_start / N16_SDOT_OUTPUTS;
    let quad = (output_start % N16_SDOT_OUTPUTS) / 4;
    let mut acc_f32 = vdupq_n_f32(0.0);
    for block in 0..k_blocks {
        let mut acc_i32 = vdupq_n_s32(0);
        for group in 0..groups_per_block {
            let activation_base =
                block * groups_per_block * N16_SDOT_K_GROUP + group * N16_SDOT_K_GROUP;
            let word = u32::from_le_bytes([
                activation[activation_base] as u8,
                activation[activation_base + 1] as u8,
                activation[activation_base + 2] as u8,
                activation[activation_base + 3] as u8,
            ]);
            let act = vreinterpretq_s8_u32(vdupq_n_u32(word));
            let weight_base = tile * tile_stride
                + (block * groups_per_block + group) * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP
                + quad * 4 * N16_SDOT_K_GROUP;
            let weights = unsafe { vld1q_s8(weight.values.as_ptr().add(weight_base)) };
            acc_i32 = unsafe { sdot_i8x16(acc_i32, act, weights) };
        }
        let mut correction = [0i32; 4];
        let mut scale = [0.0f32; 4];
        for lane in 0..4 {
            let output = output_start + lane;
            correction[lane] = weight.zero_point_offsets[output * k_blocks + block] as i32
                * activation_sums[block];
            scale[lane] = weight.scales[output * k_blocks + block];
        }
        let corr = unsafe { vld1q_s32(correction.as_ptr()) };
        let block_i32 = vaddq_s32(acc_i32, corr);
        let block_f32 = vcvtq_f32_s32(block_i32);
        let scales = vmulq_n_f32(
            unsafe { vld1q_f32(scale.as_ptr()) },
            activation_scales[block],
        );
        acc_f32 = vaddq_f32(acc_f32, vmulq_f32(block_f32, scales));
    }
    let mut out = [0.0f32; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), acc_f32) };
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn n16_sdot_u8_i16_matmul_m1_neon_dot(
    activation: &[i16],
    group_scales: &[f32],
    group_sums: &[i32],
    block_activation_sums: &[f32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    result: &mut [f32],
    n: usize,
    k_blocks: usize,
) {
    let mut offset = 0usize;
    while offset < result.len() {
        let output = output_start + offset;
        if output.is_multiple_of(4) && output + 4 <= n && offset + 4 <= result.len() {
            let out = unsafe {
                n16_sdot_bits8_quad_neon(
                    activation,
                    group_scales,
                    group_sums,
                    block_activation_sums,
                    weight,
                    output,
                    k_blocks,
                )
            };
            result[offset..offset + 4].copy_from_slice(&out);
            offset += 4;
        } else {
            result[offset] = n16_sdot_bits8_output_scalar(
                activation,
                group_scales,
                group_sums,
                block_activation_sums,
                weight,
                output,
                k_blocks,
            );
            offset += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn n16_sdot_bits8_quad_neon(
    activation: &[i16],
    group_scales: &[f32],
    group_sums: &[i32],
    block_activation_sums: &[f32],
    weight: &PackedN16SdotWeight,
    output_start: usize,
    k_blocks: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    debug_assert!(output_start.is_multiple_of(4));
    let groups_per_block = 4usize;
    let groups_per_block_4 = 32 / N16_SDOT_K_GROUP;
    let tile_stride = k_blocks * 128 / N16_SDOT_K_GROUP * N16_SDOT_OUTPUTS * N16_SDOT_K_GROUP;
    let tile = output_start / N16_SDOT_OUTPUTS;
    let quad = (output_start % N16_SDOT_OUTPUTS) / 4;
    let mut total = vdupq_n_f32(0.0);
    for block in 0..k_blocks {
        let mut product = vdupq_n_f32(0.0);
        for group_in_block in 0..groups_per_block {
            let group = block * groups_per_block + group_in_block;
            let mut acc_hi = vdupq_n_s32(0);
            let mut acc_lo = vdupq_n_s32(0);
            for sub in 0..groups_per_block_4 {
                let group4 = group_in_block * groups_per_block_4 + sub;
                let activation_base = group * 32 + sub * N16_SDOT_K_GROUP;
                let mut hi_bytes = [0i8; 4];
                let mut lo_bytes = [0i8; 4];
                for kk in 0..N16_SDOT_K_GROUP {
                    let value = activation.get(activation_base + kk).copied().unwrap_or(0);
                    let hi = value.div_euclid(256) as i8;
                    let lo_center = (value - i16::from(hi) * 256 - 128) as i8;
                    hi_bytes[kk] = hi;
                    lo_bytes[kk] = lo_center;
                }
                let hi_word = u32::from_le_bytes([
                    hi_bytes[0] as u8,
                    hi_bytes[1] as u8,
                    hi_bytes[2] as u8,
                    hi_bytes[3] as u8,
                ]);
                let lo_word = u32::from_le_bytes([
                    lo_bytes[0] as u8,
                    lo_bytes[1] as u8,
                    lo_bytes[2] as u8,
                    lo_bytes[3] as u8,
                ]);
                let hi_vec = vreinterpretq_s8_u32(vdupq_n_u32(hi_word));
                let lo_vec = vreinterpretq_s8_u32(vdupq_n_u32(lo_word));
                let weight_base = tile * tile_stride
                    + (block * 128 / N16_SDOT_K_GROUP + group4)
                        * N16_SDOT_OUTPUTS
                        * N16_SDOT_K_GROUP
                    + quad * 4 * N16_SDOT_K_GROUP;
                let weights = unsafe { vld1q_s8(weight.values.as_ptr().add(weight_base)) };
                acc_hi = unsafe { sdot_i8x16(acc_hi, hi_vec, weights) };
                acc_lo = unsafe { sdot_i8x16(acc_lo, lo_vec, weights) };
            }
            let q_dot = vaddq_s32(
                vaddq_s32(acc_lo, vmulq_n_s32(acc_hi, 256)),
                vdupq_n_s32(128 * group_sums[group]),
            );
            product = vaddq_f32(
                product,
                vmulq_n_f32(vcvtq_f32_s32(q_dot), group_scales[group]),
            );
        }
        let mut scales = [0.0f32; 4];
        let mut zero_points = [0.0f32; 4];
        for lane in 0..4 {
            let output = output_start + lane;
            scales[lane] = weight.scales[output * k_blocks + block];
            zero_points[lane] =
                (128i32 - weight.zero_point_offsets[output * k_blocks + block] as i32) as f32;
        }
        let scale_vec = unsafe { vld1q_f32(scales.as_ptr()) };
        let zp_vec = unsafe { vld1q_f32(zero_points.as_ptr()) };
        let block_sum = vdupq_n_f32(block_activation_sums[block]);
        let contribution = vsubq_f32(product, vmulq_f32(zp_vec, block_sum));
        total = vaddq_f32(total, vmulq_f32(scale_vec, contribution));
    }
    let mut out = [0.0f32; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), total) };
    out
}

#[allow(clippy::too_many_arguments)]
fn packed_nbits_gemv(
    activation: &[f32],
    weight: &PackedNBitsWeight,
    result: &mut [f32],
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(result.len(), n);
    packed_nbits_output_row(activation, weight, result, k, n, bits, block_size, true);
}

#[allow(clippy::too_many_arguments)]
fn packed_nbits_gemm(
    activations: &[f32],
    weight: &PackedNBitsWeight,
    result: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
) {
    debug_assert_eq!(activations.len(), m * k);
    debug_assert_eq!(result.len(), m * n);
    let parallel_columns = m < rayon::current_num_threads() && output_chunk_len(n, k) < n;
    result
        .par_chunks_mut(n)
        .zip(activations.par_chunks_exact(k))
        .for_each(|(output, activation)| {
            packed_nbits_output_row(
                activation,
                weight,
                output,
                k,
                n,
                bits,
                block_size,
                parallel_columns,
            );
        });
}

#[allow(clippy::too_many_arguments)]
fn packed_nbits_output_row(
    activation: &[f32],
    weight: &PackedNBitsWeight,
    result: &mut [f32],
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    parallel: bool,
) {
    let layout = NBitsLayout { bits, block_size };
    let block_count = k.div_ceil(block_size);
    debug_assert_eq!(
        weight.values.len(),
        n * block_count * layout.packed_block_size()
    );
    debug_assert_eq!(weight.scales.len(), n * block_count);
    debug_assert_eq!(result.len(), n);
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (offset, output) in outputs.iter_mut().enumerate() {
            let weight_row = weight.row(output_start + offset, block_count, layout);
            let mut sum = 0.0f32;
            for block_index in 0..block_count {
                let block = weight_row.block(block_index);
                let depth_start = block_index * block_size;
                let valid = k.saturating_sub(depth_start).min(block_size);
                let values_per_byte = layout.values_per_byte();
                for (byte_index, &packed_values) in block.values.iter().enumerate() {
                    let within_start = byte_index * values_per_byte;
                    let packed_count = valid.saturating_sub(within_start).min(values_per_byte);
                    for packed_index in 0..packed_count {
                        sum += activation[depth_start + within_start + packed_index]
                            * block.dequantized_packed_value(packed_values, packed_index);
                    }
                }
            }
            *output = sum;
        }
    };
    if parallel {
        parallel_output_rows(result, k, compute);
    } else {
        compute(0, result);
    }
}

/// Borrow the optional int4 zero-point tensor for the zero-copy decode path.
///
/// Returns `Some(None)` for a symmetric model (no zero_points input) — the
/// borrowed kernel then uses the implicit midpoint 8. Returns `Some(Some(zp))`
/// when an asymmetric uint8 zero_points input is present and host-contiguous.
/// Returns `None` only when a zero_points input exists but cannot be borrowed
/// in place, so the caller must fall through to another path. Gating on this
/// (rather than on "a zero_points input happens to exist") is what lets
/// symmetric int4 take the borrowed path instead of the resident f32 cache.
fn borrow_optional_int4_zero_points<'a>(
    zero_points: Option<&TensorView<'a>>,
) -> Option<Option<&'a [u8]>> {
    match zero_points {
        None => Some(None),
        Some(view) => contiguous_host_slice::<u8>(view).map(Some),
    }
}

fn contiguous_host_slice<'a, T>(view: &TensorView<'a>) -> Option<&'a [T]> {
    if !view.device.is_host_accessible() || !view.is_contiguous() {
        return None;
    }
    // SAFETY: the executor guarantees the TensorView backing is valid for `'a`; callers validate
    // the dtype before selecting T, and contiguous views contain exactly `numel` elements.
    Some(unsafe { std::slice::from_raw_parts(view.data_ptr::<T>(), view.numel()) })
}

fn borrowed_scales<'a>(view: &TensorView<'a>) -> Option<BorrowedScales<'a>> {
    match view.dtype {
        DataType::Float32 => contiguous_host_slice(view).map(BorrowedScales::F32),
        DataType::Float16 => contiguous_host_slice(view).map(BorrowedScales::F16),
        DataType::BFloat16 => contiguous_host_slice(view).map(BorrowedScales::Bf16),
        _ => None,
    }
}

/// A/B toggle for the register-blocked ("N-blocked") borrowed int4 decode
/// kernel that absorbs MLAS SQNBit CompFp32's activation-reuse locality without
/// a resident packed copy. `1`/`on` enables it; unset or `0`/`off` keeps the
/// per-column path. Default off so the shipped borrowed path is unchanged until
/// the win is measured, exactly like the `ONNX_GENAI_CPU_MM_MLAS_QNBIT` and
/// `#994` toggles that preceded it. Production is the only writer of process
/// state here: this is a read-only env probe, never mutated at runtime.
#[cfg(target_arch = "x86_64")]
fn borrowed_int4_nblock_enabled() -> bool {
    std::env::var("ONNX_GENAI_CPU_MM_INT4_NBLK")
        .ok()
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("off")
        })
        .unwrap_or(false)
}

/// Register-blocked borrowed int4 matmul. Reads the *same* zero-copy mmap int4
/// layout as [`borrowed_affine_int4_matmul`] (no repack, no resident copy), but
/// processes up to four output columns together with four independent f32
/// accumulators, loading each activation vector once and reusing it across the
/// group. This mirrors MLAS SQNBit CompFp32's M==1 `NCols4` register blocking
/// (`sqnbitgemm_kernel_avx2.cpp`), whose activation reuse and single
/// end-of-column horizontal reduction are the two locality wins that do not
/// depend on its prepacked buffer.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
fn borrowed_affine_int4_matmul_nblock(
    activations: &[f32],
    packed: &[u8],
    scales: BorrowedScales<'_>,
    zero_points: Option<&[u8]>,
    bias: Option<&[f32]>,
    result: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) {
    debug_assert_eq!(activations.len(), m * k);
    debug_assert_eq!(result.len(), m * n);
    let bits = 4usize;
    let layout = NBitsLayout { bits, block_size };
    let block_count = k.div_ceil(block_size);
    let packed_row_size = block_count * layout.packed_block_size();
    let zero_point_row_size = layout.zero_point_row_size(block_count);
    for (activation, output_row) in activations.chunks_exact(k).zip(result.chunks_exact_mut(n)) {
        let activation_sums = activation
            .chunks(block_size)
            .map(|block| block.iter().sum::<f32>())
            .collect::<Vec<_>>();
        let compute = |output_start: usize, outputs: &mut [f32]| {
            let mut group_start = 0usize;
            while group_start < outputs.len() {
                let group = (outputs.len() - group_start).min(4);
                let mut packed_rows: [&[u8]; 4] = [&[] as &[u8]; 4];
                let mut scale_bases = [0usize; 4];
                let mut zp_rows: [Option<&[u8]>; 4] = [None; 4];
                let mut biases = [0.0f32; 4];
                for j in 0..group {
                    let output_index = output_start + group_start + j;
                    packed_rows[j] = &packed
                        [output_index * packed_row_size..(output_index + 1) * packed_row_size];
                    scale_bases[j] = output_index * block_count;
                    zp_rows[j] = zero_points.map(|zp| {
                        &zp[output_index * zero_point_row_size
                            ..(output_index + 1) * zero_point_row_size]
                    });
                    biases[j] = bias.map_or(0.0, |values| values[output_index]);
                }
                let mut out_buf = [0.0f32; 4];
                // SAFETY: this function is only reached when `selected_dot_kernel`
                // reported an AVX2-capable host (see the route in `compute`), so
                // AVX2+FMA are available; every slice passed is bounds-checked
                // above and the block loop never reads past `k`.
                unsafe {
                    borrowed_int4_nblock4_avx2(
                        activation,
                        &activation_sums,
                        &packed_rows[..group],
                        &scales,
                        &scale_bases[..group],
                        &zp_rows[..group],
                        &biases[..group],
                        layout,
                        block_count,
                        block_size,
                        k,
                        &mut out_buf[..group],
                    );
                }
                outputs[group_start..group_start + group].copy_from_slice(&out_buf[..group]);
                group_start += group;
            }
        };
        parallel_output_rows(output_row, k, compute);
    }
}

/// AVX2 + FMA register-blocked int4 kernel for up to four output columns.
///
/// For each K block it loads the block's activation vectors once and reuses
/// them across every column in the group (MLAS's activation-reuse trick), folds
/// the per-block scale into a running f32 accumulator via a single FMA, and
/// carries the zero-point affine correction in scalar. A single horizontal
/// reduction per column at the very end replaces the per-block reduction the
/// per-column path pays. The nibble unpack is byte-for-byte the same as
/// [`borrowed_int4_block_dot_avx2`], so results differ from that path only by
/// f32 summation reassociation (a few ULP), never in nibble decode.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn borrowed_int4_nblock4_avx2(
    activation: &[f32],
    activation_sums: &[f32],
    packed_rows: &[&[u8]],
    scales: &BorrowedScales<'_>,
    scale_bases: &[usize],
    zp_rows: &[Option<&[u8]>],
    biases: &[f32],
    layout: NBitsLayout,
    block_count: usize,
    block_size: usize,
    k: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let group = packed_rows.len();
    let mask = _mm_set1_epi8(0x0f);
    let packed_block_size = layout.packed_block_size();
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut correction = [0.0f32; 4];
    let mut extra = [0.0f32; 4];
    for block in 0..block_count {
        let depth_start = block * block_size;
        let valid = k.saturating_sub(depth_start).min(block_size);
        if valid != block_size || !block_size.is_multiple_of(32) {
            // Ragged or non-32-multiple tail block: scalar, matching the
            // per-column path's scalar fallback exactly.
            for c in 0..group {
                let block_values =
                    &packed_rows[c][block * packed_block_size..(block + 1) * packed_block_size];
                let scale = scales.get(scale_bases[c] + block);
                let zero_point = layout.zero_point(zp_rows[c], block) as f32;
                let mut dot = 0.0f32;
                for (byte_index, &byte) in block_values.iter().enumerate() {
                    let within = byte_index * 2;
                    if within < valid {
                        dot += activation[depth_start + within] * (byte & 0x0f) as f32;
                    }
                    if within + 1 < valid {
                        dot += activation[depth_start + within + 1] * (byte >> 4) as f32;
                    }
                }
                extra[c] += dot * scale;
                correction[c] += scale * zero_point * activation_sums[block];
            }
            continue;
        }
        let chunks = block_size / 32;
        let mut blk = [_mm256_setzero_ps(); 4];
        for chunk in 0..chunks {
            let base = depth_start + chunk * 32;
            // SAFETY: `valid == block_size` and `base + 32 <= k`, so all four
            // 8-lane loads stay within the activation row.
            let a0 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base)) };
            let a1 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 8)) };
            let a2 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 16)) };
            let a3 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 24)) };
            for c in 0..group {
                // SAFETY: 16 packed bytes per 32-lane chunk are in bounds for
                // this column's block slice; loadu permits unaligned pointers.
                let bytes = unsafe {
                    _mm_loadu_si128(
                        packed_rows[c]
                            .as_ptr()
                            .add(block * packed_block_size + chunk * 16)
                            .cast(),
                    )
                };
                let low = _mm_and_si128(bytes, mask);
                let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
                let inter_lo = _mm_unpacklo_epi8(low, high);
                let inter_hi = _mm_unpackhi_epi8(low, high);
                let w0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(inter_lo));
                let w1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(inter_lo, 8)));
                let w2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(inter_hi));
                let w3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(inter_hi, 8)));
                blk[c] = _mm256_fmadd_ps(a0, w0, blk[c]);
                blk[c] = _mm256_fmadd_ps(a1, w1, blk[c]);
                blk[c] = _mm256_fmadd_ps(a2, w2, blk[c]);
                blk[c] = _mm256_fmadd_ps(a3, w3, blk[c]);
            }
        }
        for c in 0..group {
            let scale = scales.get(scale_bases[c] + block);
            let zero_point = layout.zero_point(zp_rows[c], block) as f32;
            acc[c] = _mm256_fmadd_ps(blk[c], _mm256_set1_ps(scale), acc[c]);
            correction[c] += scale * zero_point * activation_sums[block];
        }
    }
    for c in 0..group {
        let vector = acc[c];
        let lo = _mm256_castps256_ps128(vector);
        let hi = _mm256_extractf128_ps(vector, 1);
        let sum4 = _mm_add_ps(lo, hi);
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x55));
        out[c] = biases[c] + _mm_cvtss_f32(sum1) + extra[c] - correction[c];
    }
}

#[allow(clippy::too_many_arguments)]
fn borrowed_affine_int4_matmul(
    activations: &[f32],
    packed: &[u8],
    scales: BorrowedScales<'_>,
    zero_points: Option<&[u8]>,
    bias: Option<&[f32]>,
    result: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    debug_assert_eq!(activations.len(), m * k);
    debug_assert_eq!(result.len(), m * n);
    // Precision contract: this helper is reached only from the
    // `accuracy_level == 0` borrowed int4 route, i.e. ONNX CompFp32. Every
    // path below therefore has to keep the activations in f32. It must never
    // dispatch an int8-activation kernel (`quantize_activation_signed` /
    // `quantize_activation_qai8dxp`) -- that is CompInt8, and delivering it
    // where CompFp32 was requested costs ~1e-3 relative error. An `aarch64`
    // `m == 1, block_size == 32` NEON-SDOT diversion used to sit here and did
    // exactly that; it was removed rather than gated, because acc0 is its only
    // caller and so it had no semantically valid use.
    //
    // `dot_kernel` drives the `x86_64` AVX2/AVX-512 f32 fast path below, which
    // keeps full precision (`_mm512_fmadd_ps`). On other targets it is unused,
    // so discard it there to avoid an unused-variable error under `-D warnings`.
    #[cfg(not(target_arch = "x86_64"))]
    let _ = dot_kernel;
    let bits = 4usize;
    let layout = NBitsLayout { bits, block_size };
    let block_count = k.div_ceil(block_size);
    let packed_row_size = block_count * layout.packed_block_size();
    let zero_point_row_size = layout.zero_point_row_size(block_count);
    for (activation, output_row) in activations.chunks_exact(k).zip(result.chunks_exact_mut(n)) {
        let activation_sums = activation
            .chunks(block_size)
            .map(|block| block.iter().sum::<f32>())
            .collect::<Vec<_>>();
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, output) in outputs.iter_mut().enumerate() {
                let output_index = output_start + offset;
                let packed_row =
                    &packed[output_index * packed_row_size..(output_index + 1) * packed_row_size];
                let zp_row = zero_points.map(|zp| {
                    &zp[output_index * zero_point_row_size
                        ..(output_index + 1) * zero_point_row_size]
                });
                let mut sum = bias.map_or(0.0, |values| values[output_index]);
                for block in 0..block_count {
                    let depth_start = block * block_size;
                    let valid = k.saturating_sub(depth_start).min(block_size);
                    let block_values = &packed_row[block * layout.packed_block_size()
                        ..(block + 1) * layout.packed_block_size()];
                    let scale = scales.get(output_index * block_count + block);
                    let zero_point = layout.zero_point(zp_row, block) as f32;
                    let mut dot;
                    #[cfg(target_arch = "aarch64")]
                    if valid == 32 && block_size == 32 {
                        // SAFETY: AArch64 guarantees NEON, and both slices contain one full block.
                        dot = unsafe {
                            affine_int4_block32_dot_neon(
                                &activation[depth_start..depth_start + 32],
                                block_values,
                            )
                        };
                        sum += (dot - activation_sums[block] * zero_point) * scale;
                        continue;
                    }
                    #[cfg(target_arch = "x86_64")]
                    if valid == block_size && valid.is_multiple_of(32) {
                        // Vectorised int4 unpack + f32 FMA for the x86 borrowed
                        // path (issue #994). Falls through to the scalar loop
                        // when the host has no AVX2 (`DotKernel::Scalar`), so
                        // correctness never depends on the fast path existing.
                        if let Some(vec_dot) = borrowed_int4_block_dot_x86(
                            &activation[depth_start..depth_start + valid],
                            block_values,
                            dot_kernel,
                        ) {
                            sum += (vec_dot - activation_sums[block] * zero_point) * scale;
                            continue;
                        }
                    }
                    dot = 0.0;
                    for (byte_index, &byte) in block_values.iter().enumerate() {
                        let within = byte_index * 2;
                        if within < valid {
                            dot += activation[depth_start + within] * (byte & 0x0f) as f32;
                        }
                        if within + 1 < valid {
                            dot += activation[depth_start + within + 1] * (byte >> 4) as f32;
                        }
                    }
                    sum += (dot - activation_sums[block] * zero_point) * scale;
                }
                *output = sum;
            }
        };
        parallel_output_rows(output_row, k, compute);
    }
}

/// Whether the x86 SIMD borrowed int4 path (issue #994) is enabled. On by
/// default; set `ONNX_GENAI_CPU_DISABLE_INT4_SIMD=1` to fall back to the scalar
/// unpack loop. This is an A/B escape hatch so the vectorised kernel can be
/// diffed against the scalar reference in the *same* binary (same-run A/B is
/// more trustworthy than a cross-build comparison). Resolved once via a
/// `OnceLock`, mirroring `arm64_int4_direct_enabled`, so the per-token decode
/// path never takes the process-wide env lock or allocates.
#[cfg(target_arch = "x86_64")]
fn int4_borrowed_simd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_DISABLE_INT4_SIMD").map_or(true, |value| {
            let value = value.trim();
            value.is_empty() || value == "0"
        })
    })
}

/// Dispatch one int4 borrowed-path block dot (`sum(activation[j] * nibble[j])`,
/// nibbles as raw `0..=15`) to the widest available x86 SIMD f32 kernel.
///
/// `activation.len()` must equal the (full) block length and be a multiple of
/// 32; `packed` holds `activation.len() / 2` bytes in the same low-then-high
/// nibble interleave the scalar loop reads. Returns `None` for
/// [`DotKernel::Scalar`] (no AVX2) so the caller keeps the scalar reference.
///
/// This is the f32 analogue of the aarch64 `affine_int4_block32_dot_neon`
/// route: the per-block scale and the zero-point correction stay in the caller,
/// so symmetric (`zero_points = None`, implicit midpoint 8) and asymmetric
/// weights both work unchanged. The nibble unpack and the multiply-accumulate
/// are vectorised; nothing dequantised outlives the call, so the #979
/// zero-copy footprint is preserved.
#[cfg(target_arch = "x86_64")]
#[inline]
fn borrowed_int4_block_dot_x86(
    activation: &[f32],
    packed: &[u8],
    dot_kernel: DotKernel,
) -> Option<f32> {
    debug_assert_eq!(activation.len() % 32, 0);
    debug_assert_eq!(packed.len(), activation.len() / 2);
    if !int4_borrowed_simd_enabled() {
        // A/B escape hatch: fall back to the scalar reference loop so the SIMD
        // path can be diffed against it in the same binary (issue #994).
        return None;
    }
    // Both arms consume the *same* natural-order activation, so unlike
    // `int4_dot_row` this dispatcher is layout-agnostic and a clamp is directly
    // correct: an unrunnable request is answered by the host's own kernel (or
    // by `None`, the scalar reference) instead of faulting.
    match dot_kernel.clamped_to_host() {
        // AVX-512F is a superset of AVX2; `Avx512Vnni` is selected only after
        // `avx512f` was runtime-detected. Use the 512-bit f32 kernel.
        DotKernel::Avx512Vnni => {
            // SAFETY: selected_dot_kernel confirmed AVX2 + AVX-512F for this
            // host, and the length invariants above hold.
            Some(unsafe { borrowed_int4_block_dot_avx512(activation, packed) })
        }
        // Both plain AVX2 and AVX-VNNI hosts have AVX2 + FMA3 (FMA3 shipped
        // with the first AVX2 cores); the borrowed path is f32, so VNNI's
        // integer dot does not apply — the AVX2 f32 kernel serves both.
        DotKernel::Avx2 | DotKernel::AvxVnni => {
            // SAFETY: selected_dot_kernel confirmed AVX2 for this host, which
            // implies FMA3, and the length invariants above hold.
            Some(unsafe { borrowed_int4_block_dot_avx2(activation, packed) })
        }
        DotKernel::Scalar => None,
    }
}

/// AVX2 + FMA int4 block dot for the borrowed path. Unpacks the packed nibbles
/// into the natural `w[2i]=lo(byte i)`, `w[2i+1]=hi(byte i)` order, widens them
/// to f32, and accumulates `activation[j] * w[j]` over the block. Processes the
/// block in 32-lane chunks (16 packed bytes each) into a single 256-bit
/// accumulator, so blocks larger than 32 (e.g. block-128) also vectorise.
///
/// The horizontal reduction reorders the additions relative to the scalar
/// loop, so the f32 result can differ from the scalar reference by a few ULP;
/// it is not a bit-identical transform. Callers rely on argmax stability, not
/// on bit-identity (see the borrowed-path parity tests).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn borrowed_int4_block_dot_avx2(activation: &[f32], packed: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    let chunks = activation.len() / 32;
    for chunk in 0..chunks {
        // SAFETY: `packed.len() == activation.len() / 2`, so 16 bytes per chunk
        // are in bounds; loadu permits unaligned pointers.
        let bytes = unsafe { _mm_loadu_si128(packed.as_ptr().add(chunk * 16).cast()) };
        let low = _mm_and_si128(bytes, mask);
        let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        // Interleave low/high nibbles byte-wise to reproduce the scalar order
        // w0=lo(b0), w1=hi(b0), w2=lo(b1), ...
        let inter_lo = _mm_unpacklo_epi8(low, high); // weights 0..=15
        let inter_hi = _mm_unpackhi_epi8(low, high); // weights 16..=31
        let base = chunk * 32;
        // Four groups of 8 weights: zero-extend u8 -> i32 -> f32, fmadd with the
        // matching 8 activations.
        let w0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(inter_lo));
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a0 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base)) };
        acc = _mm256_fmadd_ps(a0, w0, acc);

        let w1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(inter_lo, 8)));
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a1 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 8)) };
        acc = _mm256_fmadd_ps(a1, w1, acc);

        let w2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(inter_hi));
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a2 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 16)) };
        acc = _mm256_fmadd_ps(a2, w2, acc);

        let w3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(inter_hi, 8)));
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a3 = unsafe { _mm256_loadu_ps(activation.as_ptr().add(base + 24)) };
        acc = _mm256_fmadd_ps(a3, w3, acc);
    }
    // Horizontal sum of the 8 f32 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x55));
    _mm_cvtss_f32(sum1)
}

/// AVX-512F int4 block dot for the borrowed path. Same math and unpack as
/// [`borrowed_int4_block_dot_avx2`], but widens 16 nibbles at a time into a
/// 512-bit f32 accumulator. Selected only on hosts where `selected_dot_kernel`
/// detected AVX-512F. Cannot be exercised on AVX2-only hardware; kept behind the
/// runtime `Avx512Vnni` dispatch so AVX2-only builds never call it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f")]
unsafe fn borrowed_int4_block_dot_avx512(activation: &[f32], packed: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_ps();
    let chunks = activation.len() / 32;
    for chunk in 0..chunks {
        // SAFETY: `packed.len() == activation.len() / 2`, so 16 bytes per chunk
        // are in bounds; loadu permits unaligned pointers.
        let bytes = unsafe { _mm_loadu_si128(packed.as_ptr().add(chunk * 16).cast()) };
        let low = _mm_and_si128(bytes, mask);
        let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let inter_lo = _mm_unpacklo_epi8(low, high); // weights 0..=15
        let inter_hi = _mm_unpackhi_epi8(low, high); // weights 16..=31
        let base = chunk * 32;
        // 16 nibbles per half -> 16 i32 -> 16 f32.
        let w_lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(inter_lo));
        let w_hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(inter_hi));
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a_lo = unsafe { _mm512_loadu_ps(activation.as_ptr().add(base)) };
        // SAFETY: base + 32 <= activation.len(); loadu permits unaligned.
        let a_hi = unsafe { _mm512_loadu_ps(activation.as_ptr().add(base + 16)) };
        acc = _mm512_fmadd_ps(a_lo, w_lo, acc);
        acc = _mm512_fmadd_ps(a_hi, w_hi, acc);
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn affine_int4_block32_dot_neon(activation: &[f32], packed: &[u8]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(activation.len(), 32);
    debug_assert_eq!(packed.len(), 16);
    let bytes = unsafe { vld1q_u8(packed.as_ptr()) };
    let low = vandq_u8(bytes, vdupq_n_u8(0x0f));
    let high = vshrq_n_u8::<4>(bytes);
    let values = [vzip1q_u8(low, high), vzip2q_u8(low, high)];
    let mut acc = vdupq_n_f32(0.0);
    for (half_index, values) in values.into_iter().enumerate() {
        let lo16 = vmovl_u8(vget_low_u8(values));
        let hi16 = vmovl_high_u8(values);
        let groups = [
            vmovl_u16(vget_low_u16(lo16)),
            vmovl_high_u16(lo16),
            vmovl_u16(vget_low_u16(hi16)),
            vmovl_high_u16(hi16),
        ];
        for (group_index, values) in groups.into_iter().enumerate() {
            let weights = vcvtq_f32_u32(values);
            let activation_offset = half_index * 16 + group_index * 4;
            let acts = unsafe { vld1q_f32(activation.as_ptr().add(activation_offset)) };
            acc = vmlaq_f32(acc, acts, weights);
        }
    }
    vaddvq_f32(acc)
}

/// Deinterleave int8 activations for the SIMD int4 kernels. Within each
/// 32-wide K-block, emit the 16 even-index activations (`act[2i]`, which pair
/// with the packed low nibbles / natural weights `2i`) followed by the 16
/// odd-index activations (`act[2i+1]`, pairing with the high nibbles / weights
/// `2i+1`). Done once per matmul, this lets [`int4_dot_row_avxvnni`] and
/// [`int4_dot_row_avx512vnni`] skip the per-block `unpacklo/unpackhi` nibble
/// deinterleave (the int4-decode bottleneck) — the weights stay in
/// low-then-high order and the matching activation permutation is amortized
/// over every N output row. The scalar reference keeps the natural layout.
fn deinterleave_activation_int4(activation: &[i8]) -> Vec<i8> {
    debug_assert_eq!(activation.len() % 32, 0);
    let mut out = vec![0i8; activation.len()];
    for (block_in, block_out) in activation.chunks_exact(32).zip(out.chunks_exact_mut(32)) {
        for i in 0..16 {
            block_out[i] = block_in[2 * i];
            block_out[16 + i] = block_in[2 * i + 1];
        }
    }
    out
}

/// Precompute the int4 zero-point correction `8 * sum(activation)` per K-block
/// in the exact per-lane layout of `_mm512_dpbusd_epi32(0, ones, act)`: each
/// 32-byte deinterleaved block yields 8 int32 lanes, lane `j` being the sum of
/// the four activation bytes `act[block*32 + 4j .. +4]`, left-shifted by 3 (the
/// `* 8` zero-point factor). Because it depends only on the activation, it is
/// identical across every N output column, so [`int4_dot_row_avx512vnni`] loads
/// it instead of recomputing a second `vpdpbusd` per column. The scalar sum
/// matches the VNNI integer arithmetic exactly (four `i8` addends per lane), so
/// the decode stays bit-identical.
fn activation_block_sums8(activation: &[i8], k_blocks: usize) -> Vec<i32> {
    debug_assert!(activation.len() >= k_blocks * 32);
    let mut sums = vec![0i32; k_blocks * 8];
    for block in 0..k_blocks {
        let base = block * 32;
        for lane in 0..8 {
            let o = base + lane * 4;
            let sum = activation[o] as i32
                + activation[o + 1] as i32
                + activation[o + 2] as i32
                + activation[o + 3] as i32;
            sums[block * 8 + lane] = sum << 3;
        }
    }
    sums
}

/// Compute one int4 output row (`m=1` decode). For the SIMD kernels
/// (`AvxVnni`/`Avx512Vnni`) `activation` MUST be in the deinterleaved layout of
/// [`deinterleave_activation_int4`]; the `Scalar` kernel takes natural order.
/// `act_sum8` is the precomputed per-block activation zero-point correction
/// (see [`activation_block_sums8`]); it is consumed only by the AVX-512 kernel
/// and may be empty for the others. [`int4_matmul_m1`] selects the right layout
/// and precomputation for the chosen kernel.
fn int4_dot_row(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
    act_sum8: &[i32],
    block_size: usize,
    _kernel: DotKernel,
) -> f32 {
    // This dispatcher cannot clamp: the caller has already laid the activation
    // out for `_kernel` specifically, so answering with a different kernel would
    // decode the wrong layout. `uses_vnni_int4_direct` -- the only gate that
    // routes anything here -- already requires `is_runnable_here`, so an
    // unrunnable kernel never reaches this function; assert that rather than
    // leave it to provenance. This is a per-row check on a K-length dot: a
    // cached relaxed load and a bit test.
    debug_assert!(
        _kernel.is_runnable_here(),
        "int4_dot_row reached with a kernel this host cannot execute"
    );
    #[cfg(target_arch = "x86_64")]
    {
        match _kernel {
            // Avx2 never reaches int4_matmul_m1 (gated to the int8 route by
            // `uses_vnni_int4_direct`); if it ever did, `use_simd` is false so
            // the activation is in natural order and the scalar reference below
            // is the correct decode.
            DotKernel::Avx2 => {}
            DotKernel::AvxVnni => {
                // SAFETY: selected_dot_kernel checked AVX2 and AVX-VNNI.
                return unsafe {
                    int4_dot_row_avxvnni(activation, packed_weight, scales, activation_scales)
                };
            }
            DotKernel::Avx512Vnni => {
                // SAFETY: selected_dot_kernel checked AVX2, AVX512-VNNI, and AVX512VL.
                return unsafe {
                    int4_dot_row_avx512vnni(
                        activation,
                        packed_weight,
                        scales,
                        activation_scales,
                        act_sum8,
                    )
                };
            }
            DotKernel::Scalar => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(_kernel, DotKernel::NeonDot) {
            // SAFETY: selected_dot_kernel checked FEAT_DotProd and the caller
            // provides block_size as a multiple of 32.
            return unsafe {
                int4_dot_row_neon_dot(
                    activation,
                    packed_weight,
                    scales,
                    activation_scales,
                    block_size,
                )
            };
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = act_sum8;
    int4_dot_row_scalar_block(
        activation,
        packed_weight,
        scales,
        activation_scales,
        block_size,
    )
}

#[allow(dead_code)]
fn int4_dot_row_scalar(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
) -> f32 {
    int4_dot_row_scalar_block(activation, packed_weight, scales, activation_scales, 32)
}

fn int4_dot_row_scalar_block(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
    block_size: usize,
) -> f32 {
    debug_assert_eq!(activation.len(), scales.len() * block_size);
    debug_assert_eq!(packed_weight.len(), scales.len() * (block_size / 2));
    debug_assert_eq!(activation_scales.len(), scales.len());
    let mut value = 0.0f32;
    for (block, &scale) in scales.iter().enumerate() {
        let activation = &activation[block * block_size..(block + 1) * block_size];
        let packed = &packed_weight[block * (block_size / 2)..(block + 1) * (block_size / 2)];
        let mut dot = 0i32;
        for (pair, &byte) in packed.iter().enumerate() {
            dot += activation[pair * 2] as i32 * (i32::from(byte & 0x0f) - 8);
            dot += activation[pair * 2 + 1] as i32 * (i32::from(byte >> 4) - 8);
        }
        value += dot as f32 * (scale * activation_scales[block]);
    }
    value
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn int4_dot_row_neon_dot(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
    block_size: usize,
) -> f32 {
    use std::arch::aarch64::*;

    debug_assert!(block_size.is_multiple_of(32));
    debug_assert_eq!(activation.len(), scales.len() * block_size);
    debug_assert_eq!(packed_weight.len(), scales.len() * (block_size / 2));
    debug_assert_eq!(activation_scales.len(), scales.len());

    let low_mask = vdupq_n_u8(0x0f);
    let zp = vdupq_n_s8(8);
    let mut value = 0.0f32;
    for (block, &scale) in scales.iter().enumerate() {
        let mut block_acc = vdupq_n_s32(0);
        let activation_base = block * block_size;
        let packed_base = block * (block_size / 2);
        for sub in 0..(block_size / 32) {
            // SAFETY: slice lengths are validated above; each sub-block owns
            // 16 packed bytes and 32 activation bytes.
            let packed_ptr = unsafe { packed_weight.as_ptr().add(packed_base + sub * 16) };
            let act_ptr = unsafe { activation.as_ptr().add(activation_base + sub * 32) };

            // SAFETY: pointers above are in bounds for one vector load.
            let packed = unsafe { vld1q_u8(packed_ptr) };
            let low = vandq_u8(packed, low_mask);
            let high = vshrq_n_u8::<4>(packed);
            let w0 = vsubq_s8(vreinterpretq_s8_u8(vzip1q_u8(low, high)), zp);
            let w1 = vsubq_s8(vreinterpretq_s8_u8(vzip2q_u8(low, high)), zp);
            // SAFETY: pointers above are in bounds for two vector loads.
            let a0 = unsafe { vld1q_s8(act_ptr) };
            let a1 = unsafe { vld1q_s8(act_ptr.add(16)) };
            block_acc = unsafe { sdot_i8x16(block_acc, a0, w0) };
            block_acc = unsafe { sdot_i8x16(block_acc, a1, w1) };
        }
        let block_dot = vaddvq_s32(block_acc);
        value += block_dot as f32 * (scale * activation_scales[block]);
    }
    value
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_i8x16(
    acc: std::arch::aarch64::int32x4_t,
    lhs: std::arch::aarch64::int8x16_t,
    rhs: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    // SAFETY: `sdot` is available because callers are `target_feature =
    // "dotprod"` and runtime-gated by `is_aarch64_feature_detected!("dotprod")`.
    unsafe {
        std::arch::asm!(
            "sdot {out:v}.4s, {lhs:v}.16b, {rhs:v}.16b",
            out = inout(vreg) out,
            lhs = in(vreg) lhs,
            rhs = in(vreg) rhs,
            options(nostack, nomem, preserves_flags)
        );
    }
    out
}

/// 256-bit VNNI int4 block dot. `activation` MUST be in the deinterleaved
/// layout produced by [`deinterleave_activation_int4`] (per 32-wide block: the
/// 16 even-index activations, then the 16 odd-index ones). That lets the weight
/// unpack keep the low nibbles (natural weights `2i`) in lanes 0..16 and the
/// high nibbles (`2i+1`) in lanes 16..32 without an `unpacklo/unpackhi`
/// deinterleave per block: the matching activation permutation is done once per
/// matmul instead of once per output row. The numeric result is bit-identical
/// to the natural-order kernel (same integer products, same reduction order).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int4_dot_row_avxvnni(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    // Two independent f32 accumulators break the loop-carried `add_ps` latency
    // chain (mirrors the 512-bit kernel); the two partials are summed at the end.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let low_mask = _mm_set1_epi8(0x0f);
    let zero_point = _mm256_set1_epi8(8);
    let block_scaled = |block: usize| -> __m256 {
        // SAFETY: each scale corresponds to 32 activation bytes and 16 packed bytes.
        let packed = unsafe { _mm_loadu_si128(packed_weight.as_ptr().add(block * 16).cast()) };
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        // Low nibbles (weights 2i) stay in lanes 0..16, high nibbles (2i+1) in
        // lanes 16..32; the deinterleaved activation load below matches this.
        let weight = _mm256_sub_epi8(_mm256_set_m128i(high, low), zero_point);
        // SAFETY: each block has 32 activation bytes, including zero padding.
        let activation = unsafe { _mm256_loadu_si256(activation.as_ptr().add(block * 32).cast()) };
        let absolute_weight = _mm256_sign_epi8(weight, weight);
        let signed_activation = _mm256_sign_epi8(activation, weight);
        let dot =
            _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), absolute_weight, signed_activation);
        let block_scale = scales[block] * activation_scales[block];
        _mm256_mul_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(block_scale))
    };
    let block_count = scales.len();
    let mut block = 0usize;
    while block + 2 <= block_count {
        acc0 = _mm256_add_ps(acc0, block_scaled(block));
        acc1 = _mm256_add_ps(acc1, block_scaled(block + 1));
        block += 2;
    }
    if block < block_count {
        acc0 = _mm256_add_ps(acc0, block_scaled(block));
    }
    horizontal_sum_f32_256(_mm256_add_ps(acc0, acc1))
}

/// 512-bit VNNI int4 block dot. Each int4 block is 32 int8 activations / 16
/// packed nibbles, so two blocks (64 weights) are fused into one 512-bit
/// `_mm512_dpbusd_epi32`. Rather than the 256-bit path's `sign_epi8` trick
/// (unavailable at 512-bit), the raw *unsigned* nibbles (0..15) drive `dpbusd`
/// directly and the `-8` zero-point is corrected once per pair via a second
/// `dpbusd` against all-ones (the exact per-lane activation sum) shifted left by
/// 3: `sum((nibble-8)*a) = sum(nibble*a) - 8*sum(a)`.
///
/// `activation` MUST be in the deinterleaved layout produced by
/// [`deinterleave_activation_int4`]: per 32-wide block the 16 even-index
/// activations then the 16 odd-index ones. This removes the per-block
/// `unpacklo/unpackhi` nibble deinterleave that dominated the (unpack-bound)
/// int4 decode. Instead the weights stay in low-then-high nibble order and the
/// two blocks of a pair are unpacked from a single 32-byte `_mm256_loadu_si256`
/// with one `_mm512_permutex2var_epi64` that assembles the four 128-bit halves
/// `[b0_low, b0_high, b1_low, b1_high]` — matching the deinterleaved activation
/// load — replacing four `unpack` + two cross-lane inserts per pair. The
/// activation permutation is amortized once per matmul over all N output rows.
///
/// The 16 int32 lanes split into the even block (0..8) and odd block (8..16); a
/// per-lane scale vector folds both into one f32x16 accumulator with a single
/// horizontal reduction. An odd trailing block uses the same unsigned-nibble
/// scheme at 256-bit. Each block's dot is an exact integer inside f32's 2^24
/// range; the result is bit-identical to the natural-order kernel (same integer
/// products and reduction order) and matches the scalar reference to a few ULP
/// (only the cross-block f32 accumulation order differs, as before).
/// Scaled f32x16 contribution of one *pair* of int4 blocks (`b0`, `b0+1`) for
/// the 512-bit VNNI kernel: the exact integer dot of each block (unsigned nibble
/// `dpbusd` minus the `8*sum(act)` zero-point correction) converted to f32 and
/// multiplied by its combined `weight_scale * activation_scale`. Lanes 0..8 hold
/// block `b0`, lanes 8..16 hold block `b0+1`. Factored out so [`int4_dot_row_avx512vnni`]
/// can run several of these into independent accumulators, breaking the
/// loop-carried `add_ps` latency chain (the reduction was latency-bound, not
/// throughput-bound). The integer products are identical to the inlined form; only
/// the cross-pair f32 accumulation order differs, exactly as before.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni,avx512vl")]
#[inline]
unsafe fn int4_pair_scaled_avx512(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
    act_sum8: &[i32],
    b0: usize,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;

    let low_mask256 = _mm256_set1_epi8(0x0f);
    // Assemble weight512 128-bit lanes [b0_low, b0_high, b1_low, b1_high] from
    // low256=[b0_low,b1_low] (64-bit words 0,1,2,3) and high256=[b0_high,b1_high]
    // (words 0,1,2,3 selected as 8..11). Index order is lane0..lane7.
    let perm_idx = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    let b1 = b0 + 1;
    // SAFETY: two contiguous blocks own 32 packed bytes.
    let packed = unsafe { _mm256_loadu_si256(packed_weight.as_ptr().add(b0 * 16).cast()) };
    let low = _mm256_and_si256(packed, low_mask256);
    let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask256);
    let weight = _mm512_permutex2var_epi64(
        _mm512_castsi256_si512(low),
        perm_idx,
        _mm512_castsi256_si512(high),
    );
    // SAFETY: two contiguous deinterleaved blocks own 64 activation bytes
    // (including zero padding).
    let act = unsafe { _mm512_loadu_si512(activation.as_ptr().add(b0 * 32).cast()) };
    let wdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), weight, act);
    // The `8*sum(act)` zero-point correction is activation-only and identical for
    // every N column, so it was precomputed once (already `<< 3`) in the exact
    // `dpbusd(ones, act)` lane layout — load it instead of issuing a second
    // `dpbusd` per column. Integer result is unchanged. Lanes 0..8 = block b0,
    // lanes 8..16 = block b1 (16 contiguous i32 starting at `b0 * 8`).
    // SAFETY: `act_sum8` has 8 lanes per block; the pair owns 16 in range.
    let asum8 = unsafe { _mm512_loadu_si512(act_sum8.as_ptr().add(b0 * 8).cast()) };
    let dot = _mm512_sub_epi32(wdot, asum8);
    let s0 = scales[b0] * activation_scales[b0];
    let s1 = scales[b1] * activation_scales[b1];
    // Lanes 0..8 carry block b0's scale, lanes 8..16 carry block b1's.
    let scale_vec = _mm512_set_ps(
        s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
    );
    _mm512_mul_ps(_mm512_cvtepi32_ps(dot), scale_vec)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn int4_dot_row_avx512vnni(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scales: &[f32],
    act_sum8: &[i32],
) -> f32 {
    use std::arch::x86_64::*;

    let low_mask = _mm_set1_epi8(0x0f);

    // Unsigned nibble weights (0..15) for one block in low-then-high layout
    // (`lanes 0..16 = low(byte b)`, `lanes 16..32 = high(byte b)`), matching the
    // deinterleaved activation load. Used for the odd trailing block; the main
    // loop unpacks two blocks at once. The `-8` zero point is applied afterwards.
    let block_weight = |block: usize| -> __m256i {
        // SAFETY: each block owns 16 packed bytes.
        let packed = unsafe { _mm_loadu_si128(packed_weight.as_ptr().add(block * 16).cast()) };
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        _mm256_set_m128i(high, low)
    };

    let block_count = scales.len();

    // Fuse two blocks per 512-bit `dpbusd`; defer reduction to one final pass.
    // Four independent f32 accumulators break the loop-carried `add_ps`
    // dependency chain: with a single accumulator each block's `add_ps` waited on
    // the previous one (~4-cycle latency x block_count), leaving the `dpbusd`
    // per pair — the actual work — stalled on the reduction. Rotating four chains
    // (unroll-by-4 over pairs) lets the out-of-order engine keep the VNNI ports
    // busy; the four partials are summed once at the end. Software-prefetch the
    // pair four iterations ahead so the streamed weight bytes are resident. The
    // per-column zero-point `dpbusd` is gone — its `8*sum(act)` correction is
    // precomputed once per matmul in `act_sum8` — so each pair now issues a
    // single `dpbusd` (weight·act), halving the hot-loop VNNI-port pressure.
    let pairs = block_count / 2;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut pair = 0usize;
    while pair + 4 <= pairs {
        // SAFETY: prefetch is a hint; an out-of-bounds address is harmless.
        unsafe {
            _mm_prefetch(
                packed_weight.as_ptr().add((pair + 4) * 2 * 16).cast(),
                _MM_HINT_T0,
            );
        }
        // SAFETY: each pair index is < block_count/2, so its blocks are in range.
        acc0 = _mm512_add_ps(acc0, unsafe {
            int4_pair_scaled_avx512(
                activation,
                packed_weight,
                scales,
                activation_scales,
                act_sum8,
                pair * 2,
            )
        });
        acc1 = _mm512_add_ps(acc1, unsafe {
            int4_pair_scaled_avx512(
                activation,
                packed_weight,
                scales,
                activation_scales,
                act_sum8,
                (pair + 1) * 2,
            )
        });
        acc2 = _mm512_add_ps(acc2, unsafe {
            int4_pair_scaled_avx512(
                activation,
                packed_weight,
                scales,
                activation_scales,
                act_sum8,
                (pair + 2) * 2,
            )
        });
        acc3 = _mm512_add_ps(acc3, unsafe {
            int4_pair_scaled_avx512(
                activation,
                packed_weight,
                scales,
                activation_scales,
                act_sum8,
                (pair + 3) * 2,
            )
        });
        pair += 4;
    }
    while pair < pairs {
        // SAFETY: pair < block_count/2.
        acc0 = _mm512_add_ps(acc0, unsafe {
            int4_pair_scaled_avx512(
                activation,
                packed_weight,
                scales,
                activation_scales,
                act_sum8,
                pair * 2,
            )
        });
        pair += 1;
    }
    let accumulator = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));

    let mut value = _mm512_reduce_add_ps(accumulator);

    // Odd trailing block via the same unsigned-nibble scheme at 256-bit. Its
    // zero-point correction is likewise the precomputed `act_sum8` (8 lanes for
    // this block), so no second `dpbusd` is issued here either.
    if block_count % 2 == 1 {
        let block = block_count - 1;
        let weight = block_weight(block);
        // SAFETY: the final block owns 32 activation bytes (incl. padding).
        let act = unsafe { _mm256_loadu_si256(activation.as_ptr().add(block * 32).cast()) };
        let wdot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), weight, act);
        // SAFETY: the final block owns 8 precomputed sum lanes at `block * 8`.
        let asum8 = unsafe { _mm256_loadu_si256(act_sum8.as_ptr().add(block * 8).cast()) };
        let dot = _mm256_sub_epi32(wdot, asum8);
        let block_scale = scales[block] * activation_scales[block];
        let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(block_scale));
        value += horizontal_sum_f32_256(scaled);
    }

    value
}

#[cfg(target_arch = "x86_64")]
fn horizontal_sum_f32_256(value: std::arch::x86_64::__m256) -> f32 {
    // SAFETY: __m256 and [f32; 8] are both 32-byte plain-data values.
    let lanes: [f32; 8] = unsafe { std::mem::transmute(value) };
    lanes.into_iter().sum()
}

/// Symmetric int8 activation quantization for the int4→int8 dequantized weight
/// path, stored offset by 128 (unsigned) so a VNNI `u8×i8` dot can run, with one
/// `scale = max_abs_block / 127` per K-block (see [`quantize_activation_signed`]
/// for why per-block scaling matches ORT's CompInt8 accuracy). Returns the
/// padded u8 activations and one scale per block (`padded_k / block_size`).
fn quantize_activation(
    activation: &[f32],
    padded_k: usize,
    block_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let k_blocks = padded_k / block_size;
    let mut quantized = vec![128u8; padded_k];
    let mut scales = vec![0.0f32; k_blocks];
    for (block, (out_block, scale)) in quantized
        .chunks_mut(block_size)
        .zip(scales.iter_mut())
        .enumerate()
    {
        let start = block * block_size;
        let real_end = (start + block_size).min(activation.len());
        if real_end <= start {
            continue;
        }
        let src = &activation[start..real_end];
        *scale =
            crate::kernels::simd_quant::quantize_block_u8_offset(src, &mut out_block[..src.len()]);
    }
    (quantized, scales)
}

fn int8_matmul(
    activations: &[f32],
    weight: &Int8Weight,
    result: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    let k_blocks = k.div_ceil(block_size);
    let padded_k = k_blocks * block_size;
    debug_assert_eq!(weight.values.len(), n * padded_k);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(weight.block_sums.len(), n * k_blocks);

    if m == 1 {
        let (activation, activation_scales) =
            quantize_activation(activations, padded_k, block_size);
        int8_row(
            &activation,
            &activation_scales,
            weight,
            result,
            k_blocks,
            padded_k,
            block_size,
            dot_kernel,
            true,
        );
    } else {
        let parallel_columns =
            m < rayon::current_num_threads() && output_chunk_len(n, padded_k) < n;
        result
            .par_chunks_mut(n)
            .zip(activations.par_chunks_exact(k))
            .for_each(|(output, activation)| {
                let (activation, activation_scales) =
                    quantize_activation(activation, padded_k, block_size);
                int8_row(
                    &activation,
                    &activation_scales,
                    weight,
                    output,
                    k_blocks,
                    padded_k,
                    block_size,
                    dot_kernel,
                    parallel_columns,
                );
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn int8_row(
    activation: &[u8],
    activation_scales: &[f32],
    weight: &Int8Weight,
    result: &mut [f32],
    k_blocks: usize,
    padded_k: usize,
    block_size: usize,
    dot_kernel: DotKernel,
    parallel: bool,
) {
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (offset, output) in outputs.iter_mut().enumerate() {
            let output_index = output_start + offset;
            let mut value = 0.0f32;
            let weight_row = &weight.values[output_index * padded_k..(output_index + 1) * padded_k];
            let block_sums =
                &weight.block_sums[output_index * k_blocks..(output_index + 1) * k_blocks];
            let weight_scales =
                &weight.scales[output_index * k_blocks..(output_index + 1) * k_blocks];
            for (block, &activation_scale) in activation_scales.iter().enumerate() {
                let start = block * block_size;
                let end = start + block_size;
                let unsigned_dot =
                    dot_u8_i8(&activation[start..end], &weight_row[start..end], dot_kernel);
                let signed_dot = unsigned_dot - 128 * block_sums[block];
                value += signed_dot as f32 * (activation_scale * weight_scales[block]);
            }
            *output = value;
        }
    };

    let chunk = output_chunk_len(result.len(), padded_k);
    if parallel && chunk < result.len() {
        parallel_output_rows(result, padded_k, compute);
    } else {
        compute(0, result);
    }
}

/// Intel AMX INT8 tile GEMM fast path for int4 `MatMulNBits` **prefill** (M > 1).
///
/// Decode (`M == 1`) is a GEMV and AMX (which multiplies 16-row A tiles) cannot
/// help it, so this is scoped to prefill. It reproduces the exact `int8_row`
/// arithmetic — per-K-block `u8 x i8` dot, the `- 128 * block_sum` unsigned->
/// signed correction, and the per-block `f32` scale accumulation — but computes
/// each 16xN x block int8 dot with an AMX `tdpbusd` (exact `u8 x i8 -> i32`
/// tile MAC, no saturating intermediate) instead of a scalar/VNNI reduction.
/// Because `tdpbusd` accumulates in i32 with no rounding and the f32 scaling is
/// applied in the same block order as the scalar reference, the result is
/// **bit-identical** to `int8_matmul`'s `DotKernel::Scalar` path.
///
/// Everything is gated on runtime AMX detection ([`amx_int8_available`]) and a
/// worthwhile `M` ([`AMX_PREFILL_MIN_M`]); non-AMX hosts, decode, and other
/// accuracy levels never reach here and are byte-for-byte unaffected.
#[cfg(target_arch = "x86_64")]
mod amx {
    use std::arch::asm;
    use std::sync::OnceLock;

    use rayon::prelude::*;

    use super::{Int8Weight, quantize_activation};

    /// Minimum prefill `M` before the AMX tile GEMM is worth its fixed setup
    /// (tile config + one-time VNNI4 weight repack). AMX consumes A in 16-row
    /// tiles, so below one full tile there is no tile to fill; the existing
    /// VNNI/scalar prefill path handles `M < AMX_PREFILL_MIN_M`. Tunable.
    pub(super) const AMX_PREFILL_MIN_M: usize = 16;

    /// 64-byte `TILECFG` operand for `ldtilecfg` (palette 1). Only tiles 0..=2
    /// are used: tmm0 = C (i32 accumulator), tmm1 = A (u8 activations), tmm2 =
    /// B (i8 weights, VNNI4-packed).
    #[repr(C, align(64))]
    struct TileConfig {
        palette: u8,
        start_row: u8,
        reserved: [u8; 14],
        colsb: [u16; 16],
        rows: [u8; 16],
    }

    impl TileConfig {
        /// Build the config for a `ksub`-wide K sub-tile (`ksub` is the bytes of
        /// K consumed per `tdpbusd`, `<= 64` and a multiple of 4).
        fn new(ksub: usize) -> Self {
            let mut cfg = TileConfig {
                palette: 1,
                start_row: 0,
                reserved: [0; 14],
                colsb: [0; 16],
                rows: [0; 16],
            };
            // tmm0 = C: 16 rows x 16 i32 columns (64 bytes/row).
            cfg.rows[0] = 16;
            cfg.colsb[0] = 64;
            // tmm1 = A: 16 rows x `ksub` u8 columns.
            cfg.rows[1] = 16;
            cfg.colsb[1] = ksub as u16;
            // tmm2 = B (VNNI4): `ksub/4` rows x 16 columns x 4 bytes = 64 bytes.
            cfg.rows[2] = (ksub / 4) as u8;
            cfg.colsb[2] = 64;
            cfg
        }
    }

    /// Request permission to use AMX tile data (`XFEATURE_XTILEDATA`) from the
    /// Linux kernel via `arch_prctl(ARCH_REQ_XCOMP_PERM, ...)`. Without this the
    /// first tile instruction `#GP`-faults. The grant is process-wide, so a
    /// single successful call enables every current and future thread. Returns
    /// `true` on success (`rax == 0`).
    fn request_amx_tile_permission() -> bool {
        const SYS_ARCH_PRCTL: i64 = 158;
        const ARCH_REQ_XCOMP_PERM: i64 = 0x1023;
        const XFEATURE_XTILEDATA: i64 = 18;
        let ret: i64;
        // SAFETY: a plain `arch_prctl` syscall with constant, valid arguments;
        // it only toggles this process's AMX permission and clobbers the
        // syscall-clobbered `rcx`/`r11`.
        unsafe {
            asm!(
                "syscall",
                inlateout("rax") SYS_ARCH_PRCTL => ret,
                in("rdi") ARCH_REQ_XCOMP_PERM,
                in("rsi") XFEATURE_XTILEDATA,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        ret == 0
    }

    /// Runtime AMX-INT8 capability, detected once and cached.
    ///
    /// Requires CPUID leaf 7 sub-leaf 0 `EDX[24]` (AMX-TILE) and `EDX[25]`
    /// (AMX-INT8) **and** a successful tile-data permission request. The
    /// unstable `is_x86_feature_detected!("amx-int8")` macro is unavailable on
    /// stable Rust, so this reads CPUID directly (the same bits it would check).
    pub(super) fn amx_int8_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // CPUID leaf 7 is present on every x86_64 CPU that reaches this
            // crate; `__cpuid_count` only reads processor feature flags.
            let leaf7 = std::arch::x86_64::__cpuid_count(7, 0);
            let has_amx_tile = (leaf7.edx >> 24) & 1 == 1;
            let has_amx_int8 = (leaf7.edx >> 25) & 1 == 1;
            has_amx_tile && has_amx_int8 && request_amx_tile_permission()
        })
    }

    /// Whether a `block_size` maps cleanly onto AMX K sub-tiles: K per
    /// `tdpbusd` is `min(block_size, 64)` and must be a positive multiple of 4,
    /// and a `> 64` block must be a whole number of 64-wide sub-tiles. Every
    /// spec-legal power-of-two `MatMulNBits` block size (16/32/64/128/256/...)
    /// qualifies; the guard keeps any exotic size on the scalar/VNNI path.
    pub(super) fn amx_block_size_supported(block_size: usize) -> bool {
        let ksub = block_size.min(64);
        ksub >= 4 && ksub.is_multiple_of(4) && block_size.is_multiple_of(ksub)
    }

    /// One K-block of the tile GEMM for a fixed 16-row A tile and 16-column B
    /// tile: zero the C accumulator, run `steps` `tdpbusd`s over the block's K
    /// (advancing the A and B pointers between sub-tiles), and store the i32
    /// result into `c_buf` (16x16, row stride 64 bytes).
    ///
    /// # Safety
    /// - AMX must be available and `ldtilecfg` already loaded for `(16, ksub)`
    ///   tiles on this thread.
    /// - `a_ptr` must address at least 16 rows at stride `a_stride`, each with
    ///   `steps * ksub` valid bytes from `a_ptr`; `b_ptr` likewise `steps`
    ///   contiguous VNNI4 sub-tiles; `c_buf` must hold 256 `i32`s.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn amx_block(
        a_ptr: *const u8,
        a_stride: u64,
        a_advance: u64,
        b_ptr: *const i8,
        b_advance: u64,
        steps: u64,
        c_buf: *mut i32,
    ) {
        // SAFETY: caller guarantees the tile config and buffer extents; the tmm
        // registers are the only AMX state touched and are marked clobbered.
        unsafe {
            asm!(
                "tilezero tmm0",
                "2:",
                "tileloadd tmm1, [{a} + {a_stride}]",
                "tileloadd tmm2, [{b} + {b_stride}]",
                "tdpbusd tmm0, tmm1, tmm2",
                "add {a}, {a_advance}",
                "add {b}, {b_advance}",
                "dec {steps}",
                "jnz 2b",
                "tilestored [{c} + {c_stride}], tmm0",
                a = inout(reg) a_ptr => _,
                b = inout(reg) b_ptr => _,
                steps = inout(reg) steps => _,
                a_stride = in(reg) a_stride,
                b_stride = in(reg) 64u64,
                a_advance = in(reg) a_advance,
                b_advance = in(reg) b_advance,
                c = in(reg) c_buf,
                c_stride = in(reg) 64u64,
                out("tmm0") _,
                out("tmm1") _,
                out("tmm2") _,
                options(nostack),
            );
        }
    }

    /// Repack the whole int8 weight `[N, padded_k]` into the AMX B-tile "VNNI4"
    /// layout, padding `N` up to a multiple of 16 with zero columns.
    ///
    /// For output column `n` and K index `k`, `weight.values[n * padded_k + k]`
    /// lands at `packed[n_tile * n_tile_stride + (k/4) * 64 + (n%16) * 4 + k%4]`,
    /// where `n_tile = n / 16`. A B sub-tile for `(n_tile, k0)` is then the
    /// contiguous 64-byte-per-row block at `n_tile * n_tile_stride + (k0/4)*64`.
    fn pack_b_vnni4(weight: &Int8Weight, n: usize, padded_k: usize) -> Vec<i8> {
        let n_tiles = n.div_ceil(16);
        let n_tile_stride = (padded_k / 4) * 64;
        let mut packed = vec![0i8; n_tiles * n_tile_stride];
        for col in 0..n {
            let n_tile = col / 16;
            let nn = col % 16;
            let src = &weight.values[col * padded_k..(col + 1) * padded_k];
            let base = n_tile * n_tile_stride + nn * 4;
            for (k, &value) in src.iter().enumerate() {
                packed[base + (k / 4) * 64 + (k % 4)] = value;
            }
        }
        packed
    }

    /// AMX INT8 tile GEMM for int4 `MatMulNBits` prefill; bit-identical to
    /// [`super::int8_matmul`]'s scalar path (see the module doc comment).
    ///
    /// `result` is the row-major `[M, N]` output; `activations` is `[M, K]`
    /// f32. Parallelism is over 16-row M-tiles (each writes a disjoint,
    /// contiguous slice of `result`), which keeps the shared VNNI4 weight repack
    /// one-time and avoids any cross-thread output aliasing.
    pub(super) fn int8_matmul_amx(
        activations: &[f32],
        weight: &Int8Weight,
        result: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        block_size: usize,
    ) {
        let k_blocks = k.div_ceil(block_size);
        let padded_k = k_blocks * block_size;
        debug_assert_eq!(result.len(), m * n);
        debug_assert_eq!(weight.values.len(), n * padded_k);
        debug_assert_eq!(weight.scales.len(), n * k_blocks);
        debug_assert_eq!(weight.block_sums.len(), n * k_blocks);

        let ksub = block_size.min(64);
        let steps = (block_size / ksub) as u64;
        let a_advance = ksub as u64;
        let b_advance = (ksub / 4 * 64) as u64;
        let n_tiles = n.div_ceil(16);
        let n_tile_stride = (padded_k / 4) * 64;

        // One-time VNNI4 repack of the weight, shared read-only by every worker.
        let b_packed = pack_b_vnni4(weight, n, padded_k);

        // Process 16 output rows per parallel task. `par_chunks_mut(16 * n)`
        // yields contiguous, disjoint row bands (last band may be short), so no
        // unsafe aliasing is needed and each task owns its output rows.
        result.par_chunks_mut(16 * n).enumerate().for_each_init(
            || crate::trace::worker_span("MatMulNBits.prefill_tiles"),
            |_span, (m_tile, out_rows)| {
                let m0 = m_tile * 16;
                let mr = out_rows.len() / n; // valid rows in this band (<= 16)

                // Quantize this band's activations into a 16-row-padded buffer so
                // A tiles always load 16 whole rows (padding rows read as 128 =
                // signed 0 and are never written back).
                let mut act = vec![128u8; 16 * padded_k];
                let mut act_scales = vec![0.0f32; 16 * k_blocks];
                for row in 0..mr {
                    let (q, s) = quantize_activation(
                        &activations[(m0 + row) * k..(m0 + row + 1) * k],
                        padded_k,
                        block_size,
                    );
                    act[row * padded_k..(row + 1) * padded_k].copy_from_slice(&q);
                    act_scales[row * k_blocks..(row + 1) * k_blocks].copy_from_slice(&s);
                }

                let cfg = TileConfig::new(ksub);
                // SAFETY: AMX availability was verified before dispatch; the
                // config describes only tmm0..=2 with in-range rows/colsb.
                unsafe { asm!("ldtilecfg [{0}]", in(reg) &cfg, options(nostack, readonly)) };

                let mut c_buf = [0i32; 16 * 16];
                for n_tile in 0..n_tiles {
                    let n0 = n_tile * 16;
                    let nr = (n - n0).min(16);
                    let b_tile_base = n_tile * n_tile_stride;
                    let mut acc = [0.0f32; 16 * 16];
                    for block in 0..k_blocks {
                        let k0 = block * block_size;
                        // SAFETY: `act` holds 16 rows of `padded_k`, `b_packed`
                        // holds this tile's `steps` sub-tiles, `c_buf` is 256
                        // i32; pointers/extents are all in bounds.
                        unsafe {
                            amx_block(
                                act.as_ptr().add(k0),
                                padded_k as u64,
                                a_advance,
                                b_packed.as_ptr().add(b_tile_base + (k0 / 4) * 64),
                                b_advance,
                                steps,
                                c_buf.as_mut_ptr(),
                            );
                        }
                        for i in 0..mr {
                            let activation_scale = act_scales[i * k_blocks + block];
                            for j in 0..nr {
                                let weight_index = (n0 + j) * k_blocks + block;
                                let signed_dot =
                                    c_buf[i * 16 + j] - 128 * weight.block_sums[weight_index];
                                acc[i * 16 + j] += signed_dot as f32
                                    * (activation_scale * weight.scales[weight_index]);
                            }
                        }
                    }
                    for i in 0..mr {
                        for j in 0..nr {
                            out_rows[i * n + n0 + j] = acc[i * 16 + j];
                        }
                    }
                }

                // SAFETY: releases this thread's tile state after all tile ops.
                unsafe { asm!("tilerelease", options(nostack)) };
            },
        );
    }
}

fn dot_u8_i8(activation: &[u8], weight: &[i8], _kernel: DotKernel) -> i32 {
    debug_assert_eq!(activation.len(), weight.len());
    // Never issue an instruction this CPU does not implement, whatever the
    // caller asked for. See `DotKernel::clamped_to_host`.
    let _kernel = _kernel.clamped_to_host();
    #[cfg(target_arch = "x86_64")]
    {
        match _kernel {
            DotKernel::Avx2 => {
                // SAFETY: `clamped_to_host` confirmed AVX2 via CPUID.
                return unsafe { dot_u8_i8_avx2(activation, weight) };
            }
            DotKernel::AvxVnni => {
                // SAFETY: `clamped_to_host` confirmed AVX2+AVX-VNNI via CPUID.
                return unsafe { dot_u8_i8_avxvnni(activation, weight) };
            }
            DotKernel::Avx512Vnni => {
                // SAFETY: `clamped_to_host` confirmed AVX512F/BW/VNNI/VL via CPUID.
                return unsafe { dot_u8_i8_avx512vnni(activation, weight) };
            }
            DotKernel::Scalar => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match _kernel {
            DotKernel::Neon | DotKernel::NeonDot => {
                // SAFETY: NEON/AdvSIMD is baseline on aarch64; the `i8mm` fast
                // path inside is runtime-gated by `is_aarch64_feature_detected!`.
                return unsafe { dot_u8_i8_neon(activation, weight) };
            }
            DotKernel::Scalar => {}
        }
    }
    dot_u8_i8_scalar(activation, weight)
}

fn dot_u8_i8_scalar(activation: &[u8], weight: &[i8]) -> i32 {
    activation
        .iter()
        .zip(weight)
        .map(|(&activation, &weight)| activation as i32 * weight as i32)
        .sum()
}

/// AVX2 (256-bit, **no** VNNI) exact `u8 x i8` dot for the pre-VNNI installed
/// base. Correctness is the hard part: `_mm256_maddubs_epi16` forms
/// `a0*b0 + a1*b1` per i16 lane with **signed saturation** to i16, and with
/// `a in [0,255]`, `b in [-128,127]` a two-product partial sum spans
/// `[-65280, +64770]`, which overflows i16 and saturates — diverging from the
/// scalar reference. To stay bit-exact we instead widen each 8-bit lane to
/// 16-bit (u8 zero-extends via `cvtepu8`, i8 sign-extends via `cvtepi8`) and use
/// `_mm256_madd_epi16`, which forms the same adjacent-pair products directly in
/// **32-bit** lanes with no saturating intermediate. Each product magnitude is
/// <= 32640 and a lane sum <= 65280, so the i32 accumulation never overflows and
/// equals the scalar i32 reduction exactly. Throughput is 16 elements/iter
/// (half of VNNI's 32) but still several times faster than scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_avx2(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let len = activation.len();
    let vector_len = len / 16 * 16;
    let mut accumulator = _mm256_setzero_si256();
    for index in (0..vector_len).step_by(16) {
        // SAFETY: index + 16 <= len over equal-length slices; loadu allows unaligned.
        let a8 = unsafe { _mm_loadu_si128(activation.as_ptr().add(index).cast()) };
        // SAFETY: index + 16 <= len over equal-length slices; loadu allows unaligned.
        let b8 = unsafe { _mm_loadu_si128(weight.as_ptr().add(index).cast()) };
        // Widen to 16-bit lanes so products stay exact (no saturating i16
        // maddubs intermediate): u8 zero-extends, i8 sign-extends.
        let a16 = _mm256_cvtepu8_epi16(a8);
        let b16 = _mm256_cvtepi8_epi16(b8);
        // madd_epi16 forms exact adjacent-pair products in i32 lanes.
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(a16, b16));
    }
    horizontal_sum_256(accumulator)
        + dot_u8_i8_scalar(&activation[vector_len..], &weight[vector_len..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_u8_i8_avxvnni(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let vector_len = activation.len() / 32 * 32;
    let mut accumulator = _mm256_setzero_si256();
    for index in (0..vector_len).step_by(32) {
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let a = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index).cast()) };
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let b = unsafe { _mm256_loadu_si256(weight.as_ptr().add(index).cast()) };
        accumulator = _mm256_dpbusd_avx_epi32(accumulator, a, b);
    }
    horizontal_sum_256(accumulator)
        + dot_u8_i8_scalar(&activation[vector_len..], &weight[vector_len..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
unsafe fn dot_u8_i8_avx512vnni(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let len = activation.len();
    let wide_len = len / 64 * 64;
    let mut wide = _mm512_setzero_si512();
    for index in (0..wide_len).step_by(64) {
        // SAFETY: index + 64 <= len over equal-length slices; loadu allows unaligned.
        let a = unsafe { _mm512_loadu_si512(activation.as_ptr().add(index).cast()) };
        // SAFETY: index + 64 <= len over equal-length slices; loadu allows unaligned.
        let b = unsafe { _mm512_loadu_si512(weight.as_ptr().add(index).cast()) };
        wide = _mm512_dpbusd_epi32(wide, a, b);
    }
    let mut sum = _mm512_reduce_add_epi32(wide);

    // 256-bit VNNI remainder for a trailing 32-byte chunk, then scalar tail.
    let mut index = wide_len;
    if index + 32 <= len {
        // SAFETY: index + 32 <= len over equal-length slices; loadu allows unaligned.
        let a = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index).cast()) };
        // SAFETY: index + 32 <= len over equal-length slices; loadu allows unaligned.
        let b = unsafe { _mm256_loadu_si256(weight.as_ptr().add(index).cast()) };
        sum += horizontal_sum_256(_mm256_dpbusd_epi32(_mm256_setzero_si256(), a, b));
        index += 32;
    }
    sum + dot_u8_i8_scalar(&activation[index..], &weight[index..])
}

#[cfg(target_arch = "x86_64")]
fn horizontal_sum_256(value: std::arch::x86_64::__m256i) -> i32 {
    // SAFETY: __m256i and [i32; 8] are both 32-byte plain-data values.
    let lanes: [i32; 8] = unsafe { std::mem::transmute(value) };
    lanes.into_iter().sum()
}

/// ARM NEON exact `u8 x i8` dot for aarch64. Uses the widen-`vmlal` baseline,
/// which relies only on stable baseline AdvSIMD intrinsics (present on every
/// aarch64 CPU) and produces the same i32 sum as `dot_u8_i8_scalar` bit-for-bit.
///
/// A `dotprod`/`i8mm` fast path (`vusdotq_s32`) is intentionally NOT used: the
/// exact unsigned x signed primitive `vusdotq_s32` is still gated behind the
/// unstable `stdarch_neon_i8mm` feature on the stable toolchain, so it cannot be
/// called without nightly. The widen-`vmlal` baseline already replaces the slow
/// scalar fallback with wide SIMD on all ARM hardware; a `dotprod`/`i8mm` fast
/// path can be layered on later once the intrinsic stabilizes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_u8_i8_neon(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let len = activation.len();
    let vector_len = len / 16 * 16;
    let mut accumulator = vdupq_n_s32(0);
    for index in (0..vector_len).step_by(16) {
        // SAFETY: index + 16 <= len over equal-length slices; loads are unaligned.
        let a = unsafe { vld1q_u8(activation.as_ptr().add(index)) };
        // SAFETY: index + 16 <= len over equal-length slices; loads are unaligned.
        let b = unsafe { vld1q_s8(weight.as_ptr().add(index)) };
        // Widen u8 -> i16 (zero-extend) and i8 -> i16 (sign-extend) for both
        // halves. Reinterpret the widened-unsigned lanes as signed i16: values
        // are <= 255 so they stay non-negative and the products are exact.
        let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
        let a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        // vmlal_s16 forms exact i32 products from the i16 halves and adds them
        // into the i32 accumulator (no saturating intermediate). With
        // `a in [0,255]`, `b in [-128,127]` each product magnitude is <= 32640,
        // so the i32 accumulation matches the scalar reduction exactly.
        accumulator = vmlal_s16(accumulator, vget_low_s16(a_lo), vget_low_s16(b_lo));
        accumulator = vmlal_s16(accumulator, vget_high_s16(a_lo), vget_high_s16(b_lo));
        accumulator = vmlal_s16(accumulator, vget_low_s16(a_hi), vget_low_s16(b_hi));
        accumulator = vmlal_s16(accumulator, vget_high_s16(a_hi), vget_high_s16(b_hi));
    }
    vaddvq_s32(accumulator) + dot_u8_i8_scalar(&activation[vector_len..], &weight[vector_len..])
}

#[derive(Clone, Copy)]
enum WeightLayout {
    Kn,
    Nk,
}

fn gemv_nk(activation: &[f32], weight_nk: &[f32], result: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight_nk.len(), n * k);
    debug_assert_eq!(result.len(), n);
    let compute = |output_start: usize, outputs: &mut [f32]| {
        let weights = &weight_nk[output_start * k..(output_start + outputs.len()) * k];
        for (output, weight) in outputs.iter_mut().zip(weights.chunks_exact(k)) {
            *output = activation.iter().zip(weight).map(|(&a, &b)| a * b).sum();
        }
    };
    let chunk = output_chunk_len(n, k);
    if chunk < n {
        parallel_output_rows(result, k, compute);
    } else {
        compute(0, result);
    }
}

/// Multiply-accumulate a `u8` weight slice against an `f32` activation slice.
///
/// Uses sixteen independent accumulators so LLVM can vectorize the widening
/// `u8 -> f32` FMA (a plain `iter().map().sum()` keeps a single serial `f32`
/// reduction chain and stays scalar, which dominates 8-bit decode).
#[inline]
fn dot_u8_f32(weight: &[u8], activation: &[f32]) -> f32 {
    debug_assert_eq!(weight.len(), activation.len());
    const LANES: usize = 16;
    let mut acc = [0.0f32; LANES];
    let mut weight_chunks = weight.chunks_exact(LANES);
    let mut activation_chunks = activation.chunks_exact(LANES);
    for (w, a) in weight_chunks.by_ref().zip(activation_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += w[lane] as f32 * a[lane];
        }
    }
    let mut tail = 0.0f32;
    for (w, a) in weight_chunks
        .remainder()
        .iter()
        .zip(activation_chunks.remainder())
    {
        tail += *w as f32 * *a;
    }
    tail + acc.iter().sum::<f32>()
}

/// 8-bit decode GEMV that dequantizes a dense `u8` `[N, K]` weight on the fly.
///
/// Computes, for each output row, `sum_block scale * (w . a) - (scale*zp) *
/// sum(a)`, which is algebraically the dequantized `sum((w - zp) * scale * a)`
/// but keeps the weight at one byte per element (4x less memory traffic than a
/// fully expanded `f32` weight) and the activations in `f32` (full precision).
#[allow(clippy::too_many_arguments)]
fn gemv_nk_u8(
    activation: &[f32],
    values: &[u8],
    scales: &[f32],
    scaled_zero_points: &[f32],
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(values.len(), n * k);
    debug_assert_eq!(result.len(), n);
    let k_blocks = k.div_ceil(block_size);
    debug_assert_eq!(scales.len(), n * k_blocks);
    debug_assert_eq!(scaled_zero_points.len(), n * k_blocks);
    // Per-block activation sums are shared across every output row, so compute
    // them once rather than N times inside the row loop.
    let mut block_activation_sums = vec![0.0f32; k_blocks];
    for (block, sum) in block_activation_sums.iter_mut().enumerate() {
        let start = block * block_size;
        let end = (start + block_size).min(k);
        *sum = activation[start..end].iter().sum();
    }
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (index, output) in outputs.iter_mut().enumerate() {
            let row = output_start + index;
            let weights = &values[row * k..row * k + k];
            let scale_row = &scales[row * k_blocks..row * k_blocks + k_blocks];
            let zp_row = &scaled_zero_points[row * k_blocks..row * k_blocks + k_blocks];
            let mut acc = 0.0f32;
            for block in 0..k_blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let dot = dot_u8_f32(&weights[start..end], &activation[start..end]);
                acc += scale_row[block] * dot - zp_row[block] * block_activation_sums[block];
            }
            *output = acc;
        }
    };
    let chunk = output_chunk_len(n, k);
    if chunk < n {
        parallel_output_rows(result, k, compute);
    } else {
        compute(0, result);
    }
}

/// Whether the model has opted out of full-precision activations.
///
/// ONNX's `accuracy_level` on `MatMulNBits` names the *compute* type:
/// `0` (unset) and `1` mean fp32, `2` fp16, `3` bf16, `4` int8. Anything that
/// quantizes the activation before reducing is therefore only legal at `>= 2`;
/// at `0`/`1` the kernel owes the caller an fp32 reduction.
///
/// The int8 SDOT routes above already gate on `accuracy_level == 4`. The
/// int16-activation GEMV is more accurate than those but still short of fp32,
/// so it lands at `>= 2` rather than being unconditional.
///
/// To be precise about *why* `>= 2` and not `== 4`: the justification is not
/// "int16 is at least as accurate as the fp16 that level 2 names". It is not,
/// necessarily -- int16 here is per-block *fixed point* whose scale is set by
/// the block maximum, so for a block with one outlier it can resolve the small
/// values worse than fp16's floating point would. The justification is that
/// levels 2/3/4 are the levels at which the model has explicitly given up fp32,
/// and int16 is strictly more accurate than the int8 that level 4 already
/// permits. Level 0/1 is where the guarantee actually matters, and there the
/// answer is unambiguous: fp32.
///
/// # Why this is not a free 12%
///
/// Measured on this repo's A/B harness, `MatMulNBits` bits=8, block_size=32,
/// M=1, 8 threads, versus ONNX Runtime 1.27 on the same host (max absolute
/// deviation from a float64 oracle over the whole output row):
///
/// | K=N  | int16 act | fp32 act | ORT   | int16 speed |
/// |------|-----------|----------|-------|-------------|
/// | 1024 | 1.9e-3    | 2.3e-5   | --    | +6.5%       |
/// | 3584 | 6.0e-3    | 1.1e-4   | 1.2e-4| +12%        |
/// | 4096 | 6.4e-3    | 9.2e-5   | --    | +18%        |
///
/// The fp32-activation path tracks ORT's own error almost exactly; the int16
/// path is ~55x worse and fails the harness's default f32 parity gate
/// (`1e-4 + 1e-3 * |y|`) on every one of those shapes. Buying 6-18% with 55x
/// of the output's accuracy is not a trade this kernel may make on a model's
/// behalf when the model asked for fp32 -- especially at M=1 decode, where the
/// consumer is an argmax over near-ties (the same failure mode that makes
/// ORT's int8 path flip qwen3's token 1479 -> 3988).
///
/// The fp32 path is still 4.3x faster than ORT at K=N=3584, so gating this
/// costs no *honest* win.
fn reduced_precision_activation_allowed(accuracy_level: i64) -> bool {
    accuracy_level >= 2
}

/// Whether the 8-bit decode GEMV may use the int16-activation fast path at all.
///
/// Default **on**, but subject to [`reduced_precision_activation_allowed`]:
/// this is the measurement/kill switch, not the precision policy. Opt out
/// (restore the fp32-activation [`gemv_nk_u8`]) with
/// `ONNX_GENAI_CPU_8BIT_ACT=fp32` (also accepts `f32`, `0`, `off`), used for
/// A/B before/after measurement. This never enables int8-activation, which is
/// ORT's fast-but-wrong path (it flips qwen3's near-tie token 1479 -> 3988).
fn eight_bit_int16_activation() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("ONNX_GENAI_CPU_8BIT_ACT") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "fp32" | "f32" | "0" | "off" | "false")
        }
        Err(_) => true,
    })
}

/// Quantize one K-block of `f32` activations to symmetric int16 with a single
/// per-block scale, returning that scale.
///
/// int16 (15 usable magnitude bits per block) preserves qwen3's massive
/// activation channels far better than int8 (7 bits), which is what keeps the
/// correct fp32 token. The scale is `amax / 32767`; a block of all zeros maps to
/// scale 0 and all-zero codes.
fn quantize_block_i16(activation: &[f32], out: &mut [i16]) -> f32 {
    debug_assert_eq!(activation.len(), out.len());
    crate::kernels::simd_quant::quantize_block_i16(activation, out)
}

/// Activation quantization granularity (elements per int16 scale) for the 8-bit
/// int16-activation decode path.
///
/// A *massive activation channel* forces a large per-group scale that coarsens
/// its group-mates; a finer group confines that loss to fewer neighbors, so a
/// razor-thin logit stays on the fp32 side. 32 (finer than the typical 128
/// weight block) keeps qwen3-0.6b byte-identical to the fp32 oracle. Overridable
/// with `ONNX_GENAI_CPU_8BIT_ACT_QGROUP` (rounded up to a multiple of 16, min
/// 16) for tuning. Groups nest inside the weight block (a divisor when both are
/// powers of two), so each group carries exactly one weight scale.
fn activation_quant_group() -> usize {
    static GROUP: OnceLock<usize> = OnceLock::new();
    *GROUP.get_or_init(|| {
        let requested = std::env::var("ONNX_GENAI_CPU_8BIT_ACT_QGROUP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(32);
        requested.max(16).div_ceil(16) * 16
    })
}

/// 8-bit decode GEMV with an int16-quantized activation (fast path).
///
/// Quantizes the activation to int16 in groups of [`activation_quant_group`]
/// (nested inside each weight block), then for each output row/block computes
/// `weight_scale * sum_group act_scale * (w_i16 . a_i16) - (weight_scale*zp) *
/// sum(a)`. The `w . a` term uses a SIMD grouped block dot ([`block_dot_u8_i16`],
/// one horizontal reduction per weight block); the zero-point term keeps the
/// exact `f32` block activation sum, so only the weight*activation product
/// carries int16 rounding (well within the fp32 token's margin). Algebraically
/// the same affine dequant as [`gemv_nk_u8`], with int16 activations.
#[allow(clippy::too_many_arguments)]
fn gemv_nk_u8_i16(
    activation: &[f32],
    values: &[u8],
    scales: &[f32],
    scaled_zero_points: &[f32],
    result: &mut [f32],
    k: usize,
    n: usize,
    block_size: usize,
) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(values.len(), n * k);
    debug_assert_eq!(result.len(), n);
    let k_blocks = k.div_ceil(block_size);
    debug_assert_eq!(scales.len(), n * k_blocks);
    debug_assert_eq!(scaled_zero_points.len(), n * k_blocks);
    let group = activation_quant_group().min(block_size.max(1));
    let k_groups = k.div_ceil(group);

    // Quantize the activation once (shared across every output row): per-group
    // int16 codes + scales for the product term, plus the exact f32 per-block
    // sum for the zero-point term.
    let mut quantized = vec![0i16; k];
    let mut group_scales = vec![0.0f32; k_groups];
    let mut block_activation_sums = vec![0.0f32; k_blocks];
    // These index parallel activation and output arrays by the shared group.
    #[allow(clippy::needless_range_loop)]
    for grp in 0..k_groups {
        let start = grp * group;
        let end = (start + group).min(k);
        group_scales[grp] = quantize_block_i16(&activation[start..end], &mut quantized[start..end]);
    }
    // These index parallel activation and output arrays by the shared block.
    #[allow(clippy::needless_range_loop)]
    for block in 0..k_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(k);
        block_activation_sums[block] = activation[start..end].iter().sum();
    }

    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (index, output) in outputs.iter_mut().enumerate() {
            let row = output_start + index;
            let weights = &values[row * k..row * k + k];
            let scale_row = &scales[row * k_blocks..row * k_blocks + k_blocks];
            let zp_row = &scaled_zero_points[row * k_blocks..row * k_blocks + k_blocks];
            let mut acc = 0.0f32;
            for block in 0..k_blocks {
                let block_start = block * block_size;
                let block_end = (block_start + block_size).min(k);
                let first_group = block_start / group;
                let last_group = (block_end - 1) / group;
                let block_partial = block_dot_u8_i16(
                    &weights[block_start..block_end],
                    &quantized[block_start..block_end],
                    &group_scales[first_group..=last_group],
                    group,
                );
                acc +=
                    scale_row[block] * block_partial - zp_row[block] * block_activation_sums[block];
            }
            *output = acc;
        }
    };
    let chunk = output_chunk_len(n, k);
    if chunk < n {
        parallel_output_rows(result, k, compute);
    } else {
        compute(0, result);
    }
}

/// Weighted sum over activation-quant groups of a single weight block:
/// `sum_group group_scale * (w_u8 . a_i16)`.
///
/// The caller multiplies the result by the block's weight scale (constant across
/// the block) and applies the zero-point term. Each group carries its own int16
/// activation scale (finer than the weight block to confine massive-activation
/// coarsening), but the products accumulate into a single running sum with **one**
/// horizontal reduction per block -- so a fine group costs only an extra cheap
/// vector scale-add, not an extra reduction.
#[inline]
fn block_dot_u8_i16(weights: &[u8], activation: &[i16], group_scales: &[f32], group: usize) -> f32 {
    debug_assert_eq!(weights.len(), activation.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx512bw() {
            // SAFETY: have_avx512bw confirmed AVX-512BW/F support at runtime.
            return unsafe { block_dot_u8_i16_avx512bw(weights, activation, group_scales, group) };
        }
        if have_avx2() {
            // SAFETY: have_avx2 confirmed AVX2 support at runtime.
            return unsafe { block_dot_u8_i16_avx2(weights, activation, group_scales, group) };
        }
    }
    block_dot_u8_i16_scalar(weights, activation, group_scales, group)
}

fn block_dot_u8_i16_scalar(
    weights: &[u8],
    activation: &[i16],
    group_scales: &[f32],
    group: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (group_index, chunk) in weights.chunks(group).enumerate() {
        let start = group_index * group;
        let dot: i32 = chunk
            .iter()
            .zip(&activation[start..start + chunk.len()])
            .map(|(&w, &a)| w as i32 * a as i32)
            .sum();
        acc += group_scales[group_index] * dot as f32;
    }
    acc
}

/// AVX2 grouped block dot: `u8 x i16 -> i32` via `_mm256_madd_epi16`, converted
/// to f32 and scaled per group into one f32x8 accumulator (single reduction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_dot_u8_i16_avx2(
    weights: &[u8],
    activation: &[i16],
    group_scales: &[f32],
    group: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let len = weights.len();
    let mut acc = _mm256_setzero_ps();
    let mut scalar_tail = 0.0f32;
    let mut position = 0usize;
    let mut group_index = 0usize;
    while position < len {
        let group_end = (position + group).min(len);
        let group_scale = group_scales[group_index];
        let mut group_acc = _mm256_setzero_si256();
        let mut inner = position;
        while inner + 16 <= group_end {
            // SAFETY: inner + 16 <= group_end <= len; loadu permits unaligned loads.
            let weight_bytes = unsafe { _mm_loadu_si128(weights.as_ptr().add(inner).cast()) };
            let weight_i16 = _mm256_cvtepu8_epi16(weight_bytes);
            // SAFETY: 16 i16 = 32 bytes within the equal-length activation slice.
            let activation_i16 =
                unsafe { _mm256_loadu_si256(activation.as_ptr().add(inner).cast()) };
            group_acc = _mm256_add_epi32(group_acc, _mm256_madd_epi16(weight_i16, activation_i16));
            inner += 16;
        }
        // Scale this group's partial into the block f32 accumulator (mul+add,
        // mirroring the int4 path -- no FMA feature dependency).
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_cvtepi32_ps(group_acc), _mm256_set1_ps(group_scale)),
        );
        // Non-multiple-of-16 remainder (only the final partial K block/group).
        if inner < group_end {
            let dot = dot_u8_i16_scalar(&weights[inner..group_end], &activation[inner..group_end]);
            scalar_tail += group_scale * dot as f32;
        }
        position = group_end;
        group_index += 1;
    }
    horizontal_sum_f32_256(acc) + scalar_tail
}

/// AVX-512BW grouped block dot: `u8 x i16 -> i32` via `_mm512_madd_epi16` over
/// 32 int16 lanes per step, converted to f32 and scaled per group into one
/// f32x16 accumulator (single reduction per block). Mirrors the AVX2 path's
/// structure exactly -- one running f32 accumulator, one horizontal reduction,
/// same group=32 int16 quantization -- so it keeps the fp32-correct argmax the
/// int16 activation path exists to protect, at double the vector width. A
/// 16-wide (`_mm256_madd_epi16`) step then scalar handle non-multiple-of-32
/// group tails.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn block_dot_u8_i16_avx512bw(
    weights: &[u8],
    activation: &[i16],
    group_scales: &[f32],
    group: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let len = weights.len();
    let mut acc = _mm512_setzero_ps();
    let mut scalar_tail = 0.0f32;
    let mut position = 0usize;
    let mut group_index = 0usize;
    while position < len {
        let group_end = (position + group).min(len);
        let group_scale = group_scales[group_index];
        let mut group_acc = _mm512_setzero_si512();
        let mut inner = position;
        while inner + 32 <= group_end {
            // SAFETY: inner + 32 <= group_end <= len; loadu permits unaligned loads.
            let weight_bytes = unsafe { _mm256_loadu_si256(weights.as_ptr().add(inner).cast()) };
            let weight_i16 = _mm512_cvtepu8_epi16(weight_bytes);
            // SAFETY: 32 i16 = 64 bytes within the equal-length activation slice.
            let activation_i16 =
                unsafe { _mm512_loadu_si512(activation.as_ptr().add(inner).cast()) };
            group_acc = _mm512_add_epi32(group_acc, _mm512_madd_epi16(weight_i16, activation_i16));
            inner += 32;
        }
        // Fold this group's partial into the block f32 accumulator (mul+add,
        // matching the AVX2/scalar structure -- no FMA feature dependency).
        acc = _mm512_add_ps(
            acc,
            _mm512_mul_ps(_mm512_cvtepi32_ps(group_acc), _mm512_set1_ps(group_scale)),
        );
        // 16-wide AVX2 step for a 16..31 int16 remainder, then a scalar tail.
        if inner + 16 <= group_end {
            // SAFETY: inner + 16 <= group_end <= len; loadu permits unaligned loads.
            let weight_bytes = unsafe { _mm_loadu_si128(weights.as_ptr().add(inner).cast()) };
            let weight_i16 = _mm256_cvtepu8_epi16(weight_bytes);
            // SAFETY: 16 i16 = 32 bytes within the equal-length activation slice.
            let activation_i16 =
                unsafe { _mm256_loadu_si256(activation.as_ptr().add(inner).cast()) };
            let partial = _mm256_madd_epi16(weight_i16, activation_i16);
            let dot = horizontal_sum_256(partial);
            scalar_tail += group_scale * dot as f32;
            inner += 16;
        }
        if inner < group_end {
            let dot = dot_u8_i16_scalar(&weights[inner..group_end], &activation[inner..group_end]);
            scalar_tail += group_scale * dot as f32;
        }
        position = group_end;
        group_index += 1;
    }
    horizontal_sum_f32_512(acc) + scalar_tail
}

/// Horizontal sum of a 16-lane f32 vector.
#[cfg(target_arch = "x86_64")]
fn horizontal_sum_f32_512(value: std::arch::x86_64::__m512) -> f32 {
    // SAFETY: __m512 and [f32; 16] are both 64-byte plain-data values.
    let lanes: [f32; 16] = unsafe { std::mem::transmute(value) };
    lanes.into_iter().sum()
}
/// accumulating in `i32`. Backs the scalar remainder of the grouped block dot.
///
/// A block is at most `block_size` (<= a few hundred) elements with weights in
/// `0..=255` and activations in `-32767..=32767`, so the widest partial sum
/// (`block_size * 255 * 32767`) stays well inside `i32`.
#[cfg(target_arch = "x86_64")]
fn dot_u8_i16_scalar(weight: &[u8], activation: &[i16]) -> i32 {
    debug_assert_eq!(weight.len(), activation.len());
    weight
        .iter()
        .zip(activation)
        .map(|(&w, &a)| w as i32 * a as i32)
        .sum()
}

/// Runtime AVX2 detection cached for the int16 decode dot product.
#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Runtime AVX-512BW/F detection cached for the 512-bit int16 decode dot.
#[cfg(target_arch = "x86_64")]
fn have_avx512bw() -> bool {
    static AVX512BW: OnceLock<bool> = OnceLock::new();
    *AVX512BW.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
    })
}

const MIN_PARALLEL_DOT_PRODUCTS_PER_TASK: usize = 32 * 1024;
const MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD: usize = 8 * 1024;
const MANY_THREAD_DOT_PRODUCTS_PER_THREAD: usize = 64 * 1024;
const MIN_OUTPUTS_PER_TASK: usize = 16;
const MANY_THREAD_CUTOFF: usize = 48;

pub(crate) fn output_chunk_len(n: usize, k: usize) -> usize {
    let threads = rayon::current_num_threads();
    let total_work = n.saturating_mul(k);
    // Small projections amortize Rayon well on one socket, but dispatching each
    // one across a larger pool costs more than its GEMV on the dual-socket host.
    let work_per_thread = if threads <= MANY_THREAD_CUTOFF {
        MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD
    } else {
        MANY_THREAD_DOT_PRODUCTS_PER_THREAD
    };
    if threads <= 1 || total_work < threads.saturating_mul(work_per_thread) {
        return n.max(1);
    }
    let max_tasks = if threads <= MANY_THREAD_CUTOFF {
        threads.saturating_mul(2)
    } else {
        threads
    };
    let tasks = total_work
        .div_ceil(MIN_PARALLEL_DOT_PRODUCTS_PER_TASK)
        .min(max_tasks)
        .min(n);
    if tasks < 2 {
        return n.max(1);
    }
    n.div_ceil(tasks).max(MIN_OUTPUTS_PER_TASK).min(n)
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_float_compute_dtype(name: &str, got: DataType) -> Result<()> {
    if !matches!(
        got,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(error(format!(
            "{name} must have dtype Float32, Float16, or BFloat16, got {got:?}"
        )));
    }
    Ok(())
}

/// Preserve the original f32 materialization path exactly; lower-precision
/// tensors reuse the shared scalar, cross-architecture widening machinery.
fn to_dense_compute_f32(view: &TensorView) -> Result<Vec<f32>> {
    match view.dtype {
        DataType::Float32 => to_dense_f32(view),
        DataType::Float16 | DataType::BFloat16 => {
            Ok(to_dense_f32_widen("MatMulNBits", view)?.into_owned())
        }
        other => Err(error(format!(
            "compute input must have dtype Float32, Float16, or BFloat16, got {other:?}"
        ))),
    }
}

/// Materialize the activation operand as a contiguous `f32` slice for the GEMV /
/// MLAS SQNBit kernels, **borrowing** it in place when it is already a
/// contiguous, host-resident `f32` tensor (the common case: these int4 decoder
/// graphs carry `float32` activations). All downstream kernels only *read* the
/// activations, so borrowing skips a redundant per-call `m*k` copy that
/// `to_dense_compute_f32` would otherwise allocate on every `MatMulNBits`
/// invocation. Strided or lower-precision (`f16`/`bf16`) inputs still widen into
/// an owned buffer, preserving the previous behaviour bit-for-bit.
fn compute_activations_cow<'a>(view: &'a TensorView<'_>) -> Result<Cow<'a, [f32]>> {
    if view.dtype == DataType::Float32 && view.is_contiguous() && view.device.is_host_accessible() {
        view.validate()?;
        let n = numel(view.shape);
        if n == 0 {
            return Ok(Cow::Borrowed(&[]));
        }
        // SAFETY: the validated, host-accessible, contiguous `f32` view describes
        // `n` consecutive readable `f32` from its element origin, bounds-checked
        // against the backing allocation by the owning EP (ep-api safety
        // invariant #1). `f32` has no invalid bit patterns and the borrow is tied
        // to the view's lifetime, which outlives this kernel call.
        let slice = unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), n) };
        return Ok(Cow::Borrowed(slice));
    }
    Ok(Cow::Owned(to_dense_compute_f32(view)?))
}

/// Preserve the original f32 writer exactly; f16/bf16 outputs reuse the shared
/// narrowing path, which has portable scalar conversion on every processor.
fn write_compute_f32(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    match out.dtype {
        DataType::Float32 => write_dense_f32(out, data),
        DataType::Float16 | DataType::BFloat16 => write_dense_f32_narrow("MatMulNBits", out, data),
        other => Err(error(format!(
            "Y must have dtype Float32, Float16, or BFloat16, got {other:?}"
        ))),
    }
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_flat_or_matrix_shape(
    name: &str,
    got: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if got != [rows * columns] && got != [rows, columns] {
        return Err(error(format!(
            "{name} must have shape [{}] or [{rows}, {columns}], got {got:?}",
            rows * columns
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MatMulNBits: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Take the dispatch-probe lock for the duration of a test that reads or
    /// perturbs a `*_TEST_CALLS` counter. See `DISPATCH_PROBE_LOCK`.
    fn lock_dispatch_probe() -> std::sync::MutexGuard<'static, ()> {
        DISPATCH_PROBE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Keeps the lock complete as tests are added.
    ///
    /// A `Mutex` only serialises threads that take it, so locking the
    /// *observers* is worthless unless every concurrent *perturber* locks too.
    /// Auditing that by hand is exactly the kind of thing that rots: the
    /// original failure was on a lane where `arm64_kai_sdot_direct_enabled()`
    /// is on, so tests that merely call `execute` reach a probed route without
    /// naming it.
    ///
    /// This walks this module's own source and requires every `#[test]` that
    /// can move a probe counter -- directly, or through a helper that does --
    /// to take the lock. It is pure text analysis, so it holds on every target
    /// including the aarch64 lanes this host cannot execute.
    #[test]
    fn every_probe_perturbing_test_takes_the_dispatch_lock() {
        const SOURCE: &str = include_str!("matmul_nbits.rs");
        const PERTURBS: [&str; 3] = [".execute(", "kai_sdot_matmul_m1(", "try_mlas_sqnbit("];

        // (name, is_test, body) for every fn declared in the tests module.
        let tests_module = SOURCE
            .split_once("\nmod tests {")
            .expect("this module defines `mod tests`")
            .1;
        let mut functions: Vec<(String, bool, String)> = Vec::new();
        let mut pending_test_attr = false;
        let mut current: Option<(String, bool, String)> = None;
        for line in tests_module.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("fn ")
                && line.starts_with("    fn ")
            {
                if let Some(done) = current.take() {
                    functions.push(done);
                }
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                current = Some((name, pending_test_attr, String::new()));
                pending_test_attr = false;
                continue;
            }
            if line.trim() == "#[test]" {
                pending_test_attr = true;
            }
            if let Some((_, _, body)) = current.as_mut() {
                body.push_str(line);
                body.push('\n');
            }
        }
        if let Some(done) = current.take() {
            functions.push(done);
        }
        assert!(
            functions.len() > 100,
            "source walk found only {} functions, so the parser is broken and this test proves \
             nothing",
            functions.len()
        );

        // Match on code only: a body that merely *mentions* `.execute(` in a
        // comment moves no counter, and forcing it to lock would make this
        // check obstruct unrelated work.
        let code_of = |body: &str| -> String {
            body.lines()
                .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let bodies: Vec<(String, bool, String)> = functions
            .iter()
            .map(|(name, is_test, body)| (name.clone(), *is_test, code_of(body)))
            .collect();

        let perturbs = |body: &str| PERTURBS.iter().any(|pattern| body.contains(pattern));
        // Helpers reach counters through helpers too, so close the set rather
        // than looking one level deep -- a two-level chain would otherwise slip
        // through exactly the check that is supposed to prevent it.
        let mut perturbing_helpers: Vec<&str> = bodies
            .iter()
            .filter(|(_, is_test, body)| !is_test && perturbs(body))
            .map(|(name, _, _)| name.as_str())
            .collect();
        loop {
            let grown: Vec<&str> = bodies
                .iter()
                .filter(|(name, is_test, body)| {
                    !is_test
                        && !perturbing_helpers.contains(&name.as_str())
                        && perturbing_helpers
                            .iter()
                            .any(|helper| body.contains(&format!("{helper}(")))
                })
                .map(|(name, _, _)| name.as_str())
                .collect();
            if grown.is_empty() {
                break;
            }
            perturbing_helpers.extend(grown);
        }
        assert!(
            !perturbing_helpers.is_empty(),
            "no perturbing helpers found, so the helper arm of this check is vacuous"
        );

        let mut unlocked: Vec<&str> = Vec::new();
        let mut checked = 0usize;
        for (name, is_test, body) in &bodies {
            if !is_test {
                continue;
            }
            let reaches = perturbs(body)
                || perturbing_helpers
                    .iter()
                    .any(|helper| body.contains(&format!("{helper}(")));
            if !reaches {
                continue;
            }
            checked += 1;
            if !body.contains("lock_dispatch_probe()") {
                unlocked.push(name);
            }
        }
        assert!(
            checked > 20,
            "only {checked} perturbing tests found; the detector has stopped matching"
        );
        assert!(
            unlocked.is_empty(),
            "these tests can move a dispatch-probe counter without holding \
             DISPATCH_PROBE_LOCK, so a concurrent reachability test can be handed their \
             dispatch: {unlocked:?}"
        );
    }

    /// The bug this lock exists for, reproduced without needing the hardware
    /// that exposed it.
    ///
    /// `matmulnbits_arm64_kai_qsi8_asymmetric_qwen_shape_is_reachable` reads a
    /// process-global counter before and after its own call and concludes from
    /// the delta which route its kernel took. Several *other* tests call
    /// `kai_sdot_matmul_m1` directly, so on a parallel test runner the delta
    /// could include a dispatch the observing test never made -- which is
    /// exactly how it reported that an `accuracy_level = 0` kernel had reached
    /// the activation-quantizing KAI route it is gated out of.
    ///
    /// Observers that hold the lock must see only their own dispatches. This
    /// fails without the lock and is deterministic with it: every thread
    /// asserts an exact delta, so any interleaving at all is caught.
    #[test]
    fn dispatch_probe_lock_hides_other_threads_dispatches() {
        const OBSERVERS: usize = 8;
        const DISPATCHES: usize = 32;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..OBSERVERS)
                .map(|_| {
                    scope.spawn(|| {
                        let _probe = lock_dispatch_probe();
                        let before = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed);
                        for _ in 0..DISPATCHES {
                            KAI_SDOT_M1_TEST_CALLS.fetch_add(1, Ordering::Relaxed);
                            std::thread::yield_now();
                        }
                        let observed = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed) - before;
                        assert_eq!(
                            observed, DISPATCHES,
                            "an observer saw {observed} dispatches but \
                             made {DISPATCHES}: another thread's route was attributed to it"
                        );
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("probe observer must not panic");
            }
        });
    }
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    // Only the MLAS-gated packed-buffer accounting tests build a graph with an
    // inline initializer, so these are unused (and `-D warnings` fatal) in the
    // portable configuration.
    #[cfg(feature = "mlas")]
    use onnx_runtime_ir::{TensorData, WeightRef};
    use onnx_runtime_loader::{Model, encode_model_proto};

    fn model_node(
        a_shape: &[usize],
        b_shape: &[usize],
        scales_shape: &[usize],
        zero_points_shape: Option<&[usize]>,
        output_shape: &[usize],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let mut inputs = Vec::new();
        for (name, dtype, shape) in [
            ("A", DataType::Float32, a_shape),
            ("B", DataType::Uint8, b_shape),
            ("scales", DataType::Float32, scales_shape),
        ] {
            let value = graph.create_named_value(name, dtype, static_shape(shape.iter().copied()));
            graph.add_input(value);
            inputs.push(Some(value));
        }
        if let Some(shape) = zero_points_shape {
            let value = graph.create_named_value(
                "zero_points",
                DataType::Uint8,
                static_shape(shape.iter().copied()),
            );
            graph.add_input(value);
            inputs.push(Some(value));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(block_size as i64));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn test_kernel(k: usize, n: usize, block_size: usize) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 0,
            flops: None,
            weight_prepacked: false,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
            int8_weight: OnceLock::new(),
            packed_u8_weight: OnceLock::new(),
            packed_int4_weight: OnceLock::new(),
            packed_int4_n16_weight: OnceLock::new(),
            packed_kai_qsi4_weight: OnceLock::new(),
            packed_nbits_weight: OnceLock::new(),
            packed_u8_n16_weight: OnceLock::new(),
            packed_kai_qsi8_weight: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_shards: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_packed: OnceLock::new(),
        }
    }

    fn accuracy4_kernel(k: usize, n: usize, block_size: usize) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            accuracy_level: 4,
            ..test_kernel(k, n, block_size)
        }
    }

    /// Serializes the tests that toggle a process-global dispatch flag
    /// (`set_resident_dequant_f32_cache_enabled`,
    /// `set_mlas_sqnbit_packing_enabled`) or that positively assert a node
    /// reached MLAS SQNBit, so they never observe each other's setting under
    /// Rust's parallel test harness. Route-agnostic correctness tests do not need
    /// it: both the borrowed and MLAS routes are byte-identical, and every
    /// residency assertion checks `is_none`, which a disabled flag preserves.
    static CACHE_FLAG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Build the constant `MatMulNBits` decode inputs for a bits=4 weight.
    fn cache_probe_inputs(
        weights: &[f32],
        a: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Owned, Owned, Owned, Option<Owned>) {
        let blocks = k.div_ceil(block_size);
        let (packed, scales, zps, _) = quantize(weights, n, k, block_size, asymmetric);
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales_t = Owned::f32(&[n, blocks], &scales);
        let a_t = Owned::f32(&[1, k], a);
        let zp_t = zps.as_ref().map(|z| Owned::u8(&[n, blocks.div_ceil(2)], z));
        (b, scales_t, a_t, zp_t)
    }

    /// Control that could falsify the #971 footprint predictor: for a matrix of
    /// bits=4 decode configurations, run a real constant-weight kernel and assert
    /// [`matmul_nbits_decode_caches_dequant_f32`] agrees with whether the kernel
    /// actually materialized the resident f32 `weight_nk`. If the dispatch in
    /// `execute` changes which path a config takes and the predictor is not
    /// updated to match, this test fails — the engine's accounting cannot silently
    /// drift from the kernel (#947).
    #[test]
    fn predictor_matches_actual_resident_cache() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        set_resident_dequant_f32_cache_enabled(true);
        let (k, n, block_size) = (128usize, 16usize, 32usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let a: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();

        for &accuracy_level in &[0i64, 4] {
            for &asymmetric in &[false, true] {
                for &has_g_idx in &[false, true] {
                    let (b, scales_t, a_t, zp_t) =
                        cache_probe_inputs(&weights, &a, n, k, block_size, asymmetric);
                    let mut kernel = MatMulNBitsKernel {
                        accuracy_level,
                        ..test_kernel(k, n, block_size)
                    };
                    kernel.set_constant_inputs(&[true; 5]);
                    // Natural per-block group indices: in range and identity, but
                    // present, which forces the resident `weight_nk` cache — the
                    // one config family that still caches after #979.
                    let g_idx_t = has_g_idx.then(|| {
                        let indices: Vec<i32> =
                            (0..k).map(|depth| (depth / block_size) as i32).collect();
                        Owned::i32(&[k], &indices)
                    });
                    let mut inputs = vec![a_t.view(), b.view(), scales_t.view()];
                    match (&zp_t, &g_idx_t) {
                        (Some(zp), _) => inputs.push(zp.view()),
                        // g_idx sits at input 4, so a symmetric config must fill
                        // input 3 with an explicit absent placeholder.
                        (None, Some(_)) => inputs.push(TensorView::absent(DataType::Uint8)),
                        (None, None) => {}
                    }
                    if let Some(g_idx) = &g_idx_t {
                        inputs.push(g_idx.view());
                    }
                    let mut y = Owned::zeros_f32(&[1, n]);
                    kernel.execute(&inputs, &mut [y.view_mut()]).unwrap();

                    let predicted = matmul_nbits_decode_caches_dequant_f32(
                        4,
                        block_size,
                        accuracy_level,
                        n,
                        k,
                        asymmetric,
                        has_g_idx,
                        false,
                    );
                    assert_eq!(
                        predicted,
                        kernel.weight_nk.get().is_some(),
                        "predictor disagreed with actual weight_nk residency \
                         (accuracy_level={accuracy_level}, asymmetric={asymmetric}, \
                         has_g_idx={has_g_idx})",
                    );
                }
            }
        }
    }

    /// Declining the resident cache (#971 governed decision) must hold no f32
    /// expansion and yet produce byte-identical decode output: the transient
    /// per-call dequant runs the same math, just without retaining it.
    #[test]
    fn declining_resident_cache_is_byte_identical_and_holds_no_expansion() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (k, n, block_size) = (128usize, 16usize, 32usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let a: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();
        // bits=4, accuracy_level=0 with a constant `g_idx` reaches the generic
        // `weight_nk` path. Symmetric int4 with no g_idx no longer caches after
        // #979 (the borrowed midpoint-8 path), so `has_g_idx` is what now forces
        // the resident cache: every earlier branch in `execute` (borrowed int4,
        // 2-bit, accuracy_level==4, 8-bit) is gated on `group_indices.is_none()`,
        // so a present g_idx falls through to the `m == 1` `weight_nk` cache.
        assert!(
            matmul_nbits_decode_caches_dequant_f32(4, block_size, 0, n, k, false, true, false),
            "test premise: this config must be a caching config",
        );
        // Natural per-block group indices ([0,0,..,1,1,..]): in range 0..k_blocks
        // and semantically identity, but their mere presence forces the cache.
        let g_idx: Vec<i32> = (0..k).map(|depth| (depth / block_size) as i32).collect();

        let run = |enabled: bool| {
            set_resident_dequant_f32_cache_enabled(enabled);
            let (b, scales_t, a_t, _) = cache_probe_inputs(&weights, &a, n, k, block_size, false);
            let g_idx_t = Owned::i32(&[k], &g_idx);
            let mut kernel = test_kernel(k, n, block_size);
            kernel.set_constant_inputs(&[true; 5]);
            let mut y = Owned::zeros_f32(&[1, n]);
            kernel
                .execute(
                    &[
                        a_t.view(),
                        b.view(),
                        scales_t.view(),
                        // Symmetric: no zero_points, but g_idx sits at input 4,
                        // so input 3 must be an explicit absent placeholder.
                        TensorView::absent(DataType::Uint8),
                        g_idx_t.view(),
                    ],
                    &mut [y.view_mut()],
                )
                .unwrap();
            let cached = kernel.weight_nk.get().is_some();
            (y.bytes, cached)
        };

        let (bytes_enabled, cached_enabled) = run(true);
        let (bytes_disabled, cached_disabled) = run(false);
        set_resident_dequant_f32_cache_enabled(true);

        assert!(
            cached_enabled,
            "enabled arm must materialize the resident cache"
        );
        assert!(
            !cached_disabled,
            "declined arm must hold no resident f32 expansion",
        );
        assert_eq!(
            bytes_enabled, bytes_disabled,
            "declining the resident cache changed decode output",
        );
    }

    /// `None` if none is populated yet. Which cache is filled depends on the
    /// route: MLAS SQNBit (`mlas_shards` or serialized `mlas_packed`) when
    /// available, otherwise the hand GEMV/int8 caches. Returning a raw address
    /// lets tests assert the cache is *reused* (stable) across calls, not merely
    /// populated. The address is stable because every cache is a `OnceLock` that
    /// stores its value in place.
    fn prepack_cache_ptr(kernel: &MatMulNBitsKernel) -> Option<*const ()> {
        if let Some(w) = kernel.weight_nk.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.int8_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_int4_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_int4_n16_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_kai_qsi4_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_nbits_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_u8_n16_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_kai_qsi8_weight.get() {
            return Some(w as *const _ as *const ());
        }
        #[cfg(feature = "mlas")]
        if let Some(w) = kernel.mlas_shards.get() {
            return Some(w as *const _ as *const ());
        }
        #[cfg(feature = "mlas")]
        if let Some(w) = kernel.mlas_packed.get() {
            return Some(w as *const _ as *const ());
        }
        None
    }

    /// True when a constant `MatMulNBits` weight has been prepacked into any of
    /// the reuse caches (see [`prepack_cache_ptr`]).
    fn prepack_cache_populated(kernel: &MatMulNBitsKernel) -> bool {
        prepack_cache_ptr(kernel).is_some()
    }

    /// Route probe for constant-weight `bits = 4, accuracy_level = 0` nodes.
    ///
    /// Two invariants are asserted together because they are the two halves of
    /// the same policy:
    ///
    /// * **#979 (memory):** such a node must never expand its weight into the
    ///   resident f32 `weight_nk` cache (~8x the file size in RAM).
    /// * **Dispatch:** it must land on the *fastest available* zero/low-copy
    ///   route for this build -- MLAS SQNBit CompFp32 when the vendored MLAS
    ///   has a kernel for the shape on this host, otherwise the borrowed
    ///   zero-copy int4 path.
    ///
    /// The expected route is derived from the same predicate production code
    /// uses ([`MatMulNBitsKernel::mlas_sqnbit_owns_fp32_compute`]), so the
    /// assertion stays exact on hosts and shapes where MLAS declines instead of
    /// silently degrading into a tautology.
    #[derive(Clone, Copy)]
    struct Int4Acc0RouteProbe {
        symmetric: usize,
        asymmetric: usize,
        #[cfg(feature = "mlas")]
        mlas: usize,
    }

    impl Int4Acc0RouteProbe {
        fn start() -> Self {
            Self {
                symmetric: BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed),
                asymmetric: BORROWED_INT4_ASYMMETRIC_TEST_CALLS.load(Ordering::Relaxed),
                #[cfg(feature = "mlas")]
                mlas: MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed),
            }
        }

        /// `symmetric` selects which borrowed counter proves the branch; the
        /// counters are global and shared with tests running in parallel, hence
        /// the strict-increase (`>`) comparisons rather than exact deltas.
        fn assert_fast_route(self, kernel: &MatMulNBitsKernel, symmetric: bool) {
            assert!(
                kernel.weight_nk.get().is_none(),
                "constant-weight int4 accuracy_level=0 must not expand the weight to a resident f32 cache (#979)"
            );
            // Footprint, not field-emptiness (#1027). Asserting only that
            // `weight_nk` stays empty passes straight through the accounting
            // bug: the MLAS SQNBit route also leaves `weight_nk` empty while
            // holding a packed buffer ~2x the int4 bytes beside the mapped
            // weight. The memory plan budgets exactly the byte total this
            // function returns, so assert the route's real resident footprint.
            let accounted = matmul_nbits_resident_side_cache_bytes(
                kernel.bits,
                kernel.block_size,
                kernel.accuracy_level,
                kernel.n,
                kernel.k,
                !symmetric,
                false,
                false,
            );
            // Never the ~8x f32 expansion on either route (#979).
            assert_ne!(
                accounted,
                (kernel.n as u64) * (kernel.k as u64) * 4,
                "int4 accuracy_level=0 must never be accounted as the ~8x resident f32 expansion (#979)"
            );
            #[cfg_attr(feature = "mlas", allow(unused_variables))]
            let borrowed_ran = if symmetric {
                BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed) > self.symmetric
            } else {
                BORROWED_INT4_ASYMMETRIC_TEST_CALLS.load(Ordering::Relaxed) > self.asymmetric
            };
            #[cfg(feature = "mlas")]
            if kernel.mlas_sqnbit_owns_fp32_compute(true, !symmetric) {
                // Per-kernel state, not the global counters: this is race-free
                // proof that *this* node's weight was packed by MLAS SQNBit.
                let mlas_packed_this_node = kernel
                    .mlas_shards
                    .get()
                    .is_some_and(|shards| shards.is_some())
                    || kernel
                        .mlas_packed
                        .get()
                        .is_some_and(|packed| packed.is_some());
                assert!(
                    mlas_packed_this_node,
                    "constant-weight int4 accuracy_level=0 must reach MLAS SQNBit CompFp32 when MLAS has a kernel for the shape"
                );
                assert!(
                    MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed) > self.mlas,
                    "the MLAS SQNBit GEMM must actually have run for this node"
                );
                // The packed buffer is a real resident allocation. The memory
                // plan budgets the byte total this function returns, which must
                // equal what the kernels actually hold across the session: the
                // per-copy packed + scale/zp bytes. Since #1056 the prefill and
                // decode kernel instances share one packed buffer, so this is a
                // single copy -- not zero (the pre-#1027 blind spot), and not the
                // 2x the #1051 accounting reported before the buffer was shared.
                let packed_bytes = mlas_sys::sqnbit_packed_b_size(
                    kernel.n,
                    kernel.k,
                    kernel.bits,
                    kernel.block_size,
                    !symmetric,
                    mlas_sys::SQNBitComputeType::Fp32,
                )
                .expect("MLAS reported a kernel for this shape but no packed size");
                assert!(packed_bytes > 0);
                let per_copy = mlas_sqnbit_packed_b_cache_bytes(
                    kernel.bits,
                    kernel.block_size,
                    kernel.accuracy_level,
                    kernel.n,
                    kernel.k,
                    !symmetric,
                    false,
                    false,
                )
                .expect("MLAS route was asserted available for this shape");
                assert_eq!(
                    accounted,
                    per_copy.saturating_mul(MLAS_PACKED_DECODE_INSTANTIATIONS),
                    "the MLAS SQNBit packed buffer must be accounted for the memory plan (#1027) as the single shared per-copy footprint (#1056)"
                );
                assert!(
                    accounted > packed_bytes as u64,
                    "the accounted footprint must exceed a bare packed buffer (it includes the retained scales/zero-points)"
                );
                return;
            }
            assert!(
                borrowed_ran,
                "int4 accuracy_level=0 must route into the borrowed zero-copy path when MLAS has no kernel for it"
            );
            assert!(
                !prepack_cache_populated(kernel),
                "the borrowed path must borrow the packed inputs instead of building a weight cache (#979)"
            );
            // The borrowed zero-copy path holds no resident side buffer at all.
            assert_eq!(
                accounted, 0,
                "the borrowed zero-copy int4 path must account for no resident side buffer (#979)"
            );
        }
    }

    fn quantize(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let mut packed = vec![0u8; n * blocks * blob];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![0u8; n * blocks.div_ceil(2)];
        let mut dequantized = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let (scale, zp) = if asymmetric {
                    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((max - min) / 15.0).max(1e-6);
                    (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
                } else {
                    let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    ((max_abs / 7.0).max(1e-6), 8)
                };
                scales[row * blocks + block] = scale;
                if asymmetric {
                    let byte = &mut zps[row * blocks.div_ceil(2) + block / 2];
                    *byte |= zp << (4 * (block % 2));
                }
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                    packed[(row * blocks + block) * blob + offset / 2] |= q << (4 * (offset % 2));
                    dequantized[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, asymmetric.then_some(zps), dequantized)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn nblock_matches_per_column_borrowed_path() {
        // The register-blocked ("N-blocked") borrowed int4 kernel must produce
        // the same result as the per-column borrowed path (up to f32
        // reassociation), for symmetric and asymmetric int4, single- and
        // multi-row activations, and an N that is not a multiple of 4 (tail
        // group). Guarded to x86_64 because the kernel is x86-only.
        if matches!(selected_dot_kernel(), DotKernel::Scalar) {
            // No AVX2 on this host; the N-blocked path is never selected.
            return;
        }
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        };
        for &(m, k, n, block_size) in &[
            (1usize, 96usize, 13usize, 32usize),
            (1, 128, 8, 32),
            (2, 64, 7, 32),
            (1, 256, 16, 128),
        ] {
            for &asymmetric in &[false, true] {
                let weights_nk: Vec<f32> = (0..n * k).map(|_| next()).collect();
                let activations: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let (packed, scales, zps, _dequant) =
                    quantize(&weights_nk, n, k, block_size, asymmetric);
                let zp_slice = zps.as_deref();
                let bias: Vec<f32> = (0..n).map(|_| next()).collect();

                let mut expected = vec![0.0f32; m * n];
                borrowed_affine_int4_matmul(
                    &activations,
                    &packed,
                    BorrowedScales::F32(&scales),
                    zp_slice,
                    Some(&bias),
                    &mut expected,
                    m,
                    k,
                    n,
                    block_size,
                    selected_dot_kernel(),
                );

                let mut got = vec![0.0f32; m * n];
                borrowed_affine_int4_matmul_nblock(
                    &activations,
                    &packed,
                    BorrowedScales::F32(&scales),
                    zp_slice,
                    Some(&bias),
                    &mut got,
                    m,
                    k,
                    n,
                    block_size,
                );

                for (index, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                    let tolerance = 1e-3 * a.abs().max(1.0);
                    assert!(
                        (a - b).abs() <= tolerance,
                        "m={m} k={k} n={n} block={block_size} asym={asymmetric} \
                         index {index}: per-column {a} vs n-blocked {b}"
                    );
                }
            }
        }
    }

    fn reference(a: &[f32], weights_nk: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                for depth in 0..k {
                    output[row * n + column] += a[row * k + depth] * weights_nk[column * k + depth];
                }
            }
        }
        output
    }

    fn dequantize_reference(
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 2;
        let zp_row_bytes = blocks.div_ceil(2);
        let mut weights = vec![0.0; n * k];
        for output in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let within_block = depth % block_size;
                let byte = packed[(output * blocks + block) * blob_size + within_block / 2];
                let q = if within_block.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let zero_point = zero_points.map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + block / 2];
                    if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                weights[output * k + depth] =
                    (q as f32 - zero_point as f32) * scales[output * blocks + block];
            }
        }
        weights
    }

    fn quantize_2bit(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>) {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 4;
        let mut packed = vec![0u8; n * blocks * blob_size];
        let mut scales = vec![0.0f32; n * blocks];
        let zero_point_row_size = blocks.div_ceil(4);
        let mut zero_points = vec![0u8; n * zero_point_row_size];
        for output in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[output * k + start..output * k + end];
                let (scale, zero_point) = if asymmetric {
                    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((maximum - minimum) / 3.0).max(1e-6);
                    (scale, (-minimum / scale).round().clamp(0.0, 3.0) as u8)
                } else {
                    let maximum_absolute =
                        values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    (maximum_absolute.max(1e-6), 2)
                };
                scales[output * blocks + block] = scale;
                if asymmetric {
                    zero_points[output * zero_point_row_size + block / 4] |=
                        zero_point << (2 * (block % 4));
                }
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zero_point as f32).round().clamp(0.0, 3.0) as u8;
                    packed[(output * blocks + block) * blob_size + offset / 4] |=
                        q << (2 * (offset % 4));
                }
            }
        }
        (packed, scales, asymmetric.then_some(zero_points))
    }

    fn dequantize_2bit_reference(
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 4;
        let zero_point_row_size = blocks.div_ceil(4);
        let mut dequantized = vec![0.0f32; n * k];
        for output in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let within_block = depth % block_size;
                let byte = packed[(output * blocks + block) * blob_size + within_block / 4];
                let q = (byte >> (2 * (within_block % 4))) & 0x03;
                let zero_point = zero_points.map_or(2, |points| {
                    (points[output * zero_point_row_size + block / 4] >> (2 * (block % 4))) & 0x03
                });
                dequantized[output * k + depth] =
                    (q as f32 - zero_point as f32) * scales[output * blocks + block];
            }
        }
        dequantized
    }

    fn run_direct_2bit_parity_case(
        m: usize,
        k: usize,
        n: usize,
        block_size: usize,
        asymmetric: bool,
        scale_dtype: DataType,
    ) {
        let activations: Vec<f32> = (0..m * k)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|index| ((index * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales, zero_points) = quantize_2bit(&weights, n, k, block_size, asymmetric);
        let reference_scales: Vec<f32> = match scale_dtype {
            DataType::Float32 => scales.clone(),
            DataType::Float16 => scales
                .iter()
                .map(|&scale| half::f16::from_f32(scale).to_f32())
                .collect(),
            _ => unreachable!(),
        };
        let dequantized = dequantize_2bit_reference(
            &packed,
            &reference_scales,
            zero_points.as_deref(),
            n,
            k,
            block_size,
        );
        let expected = reference(&activations, &dequantized, m, k, n);
        let blocks = k.div_ceil(block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.bits = 2;
        kernel.set_constant_inputs(&[false, true, true, true, false]);

        let activation = Owned::f32(&[m, k], &activations);
        let packed = Owned::u8(&[n, blocks, block_size / 4], &packed);
        let scales = match scale_dtype {
            DataType::Float32 => Owned::f32(&[n, blocks], &scales),
            DataType::Float16 => Owned::f16(&[n, blocks], &scales),
            _ => unreachable!(),
        };
        let zero_points =
            zero_points.map(|points| Owned::u8(&[n, (blocks * 2).div_ceil(8)], &points));
        let mut output = Owned::zeros_f32(&[m, n]);
        if let Some(zero_points) = &zero_points {
            kernel
                .execute(
                    &[
                        activation.view(),
                        packed.view(),
                        scales.view(),
                        zero_points.view(),
                    ],
                    &mut [output.view_mut()],
                )
                .unwrap();
        } else {
            kernel
                .execute(
                    &[activation.view(), packed.view(), scales.view()],
                    &mut [output.view_mut()],
                )
                .unwrap();
        }

        assert_close(&output.to_f32(), &expected);
        assert!(
            kernel.packed_nbits_weight.get().is_some(),
            "bits=2 must cache the packed direct-decode weight"
        );
        assert!(
            kernel.weight_nk.get().is_none(),
            "bits=2 direct decode must not materialize an f32 weight matrix"
        );
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn assert_int8_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.05 + 0.05 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn assert_qai8dxp_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.12 + 0.10 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn run_decode_case(
        activations: &[f32],
        packed: &[u8],
        scales: &[f32],
        n: usize,
        block_size: usize,
        accuracy_level: Option<i64>,
    ) -> Vec<f32> {
        let k = activations.len();
        let blocks = k.div_ceil(block_size);
        let (mut graph, node) = model_node(
            &[1, k],
            &[n, blocks, block_size / 2],
            &[n, blocks],
            None,
            &[1, n],
            k,
            n,
            block_size,
        );
        if let Some(level) = accuracy_level {
            graph
                .node_mut(node)
                .attributes
                .insert("accuracy_level".into(), Attribute::Int(level));
        }
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[1, k], activations);
        let b = Owned::u8(&[n, blocks, block_size / 2], packed);
        let scales = Owned::f32(&[n, blocks], scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        y.to_f32()
    }

    fn accuracy4_model(m: usize, k: usize, n: usize, block_size: usize) -> (Graph, NodeId) {
        let blocks = k.div_ceil(block_size);
        let (mut graph, node) = model_node(
            &[m, k],
            &[n, blocks, block_size / 2],
            &[n, blocks],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        graph
            .node_mut(node)
            .attributes
            .insert("accuracy_level".into(), Attribute::Int(4));
        let proto = encode_model_proto(&Model::new(&graph)).expect("IR model must encode to ONNX");
        let attribute = &proto.graph.as_ref().unwrap().node[0].attribute;
        assert!(
            attribute
                .iter()
                .any(|attr| attr.name == "accuracy_level" && attr.i == 4)
        );
        (graph, node)
    }

    fn run_accuracy4_case(m: usize, k: usize, n: usize, block_size: usize) {
        let a_values: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let expected = reference(&a_values, &dequantized, m, k, n);
        let (graph, node) = accuracy4_model(m, k, n, block_size);
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::u8(&[n, k.div_ceil(block_size), block_size / 2], &packed);
        let scales = Owned::f32(&[n, k.div_ceil(block_size)], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_int8_close(&y.to_f32(), &expected);
    }

    #[test]
    fn matmulnbits_accuracy4_block32_partial_k_m1_matches_fp32_reference() {
        let _probe = lock_dispatch_probe();
        run_accuracy4_case(1, 45, 9, 32);
    }

    #[test]
    fn matmulnbits_accuracy4_block128_partial_k_batched_matches_fp32_reference() {
        let _probe = lock_dispatch_probe();
        run_accuracy4_case(3, 141, 7, 128);
    }

    /// Regression for the zero-copy activation borrow (`compute_activations_cow`):
    /// a contiguous host `f32` activation is passed straight through to the GEMV /
    /// MLAS kernels without the old per-call `to_dense_compute_f32` copy. A
    /// strided (non-contiguous) `f32` view of the *same* logical values still
    /// takes the owned-materialization path. The kernel output MUST be
    /// byte-for-byte identical between the borrowed and copied activation for both
    /// the `m == 1` decode and `m > 1` batched/prefill routes; otherwise the
    /// borrow shortcut changed numerics.
    #[test]
    fn matmulnbits_activation_borrow_matches_strided_copy_bit_exact() {
        let _probe = lock_dispatch_probe();
        for &(m, k, n, block_size) in &[(1usize, 128usize, 16usize, 32usize), (4, 96, 8, 32)] {
            let a_values: Vec<f32> = (0..m * k)
                .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
                .collect();
            let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
            let (graph, node) = accuracy4_model(m, k, n, block_size);
            let model = Model::new(&graph);
            let blocks = k.div_ceil(block_size);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
            let scales_t = Owned::f32(&[n, blocks], &scales);

            // Contiguous f32 activation -> zero-copy borrow path.
            let a_contig = Owned::f32(&[m, k], &a_values);
            let kernel = CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .unwrap();
            let mut y_borrow = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(
                    &[a_contig.view(), b.view(), scales_t.view()],
                    &mut [y_borrow.view_mut()],
                )
                .unwrap();

            // Same values exposed as a NON-contiguous view (padded row stride) ->
            // forces the owned `to_dense_compute_f32` materialization path.
            let pad = 3usize;
            let mut padded = vec![0.0f32; m * (k + pad)];
            for r in 0..m {
                padded[r * (k + pad)..r * (k + pad) + k]
                    .copy_from_slice(&a_values[r * k..r * k + k]);
            }
            let a_strided =
                Owned::f32(&[m, k + pad], &padded).with_view(&[m, k], &[(k + pad) as i64, 1]);
            assert!(
                !a_strided.view().is_contiguous(),
                "strided activation must be non-contiguous to exercise the copy path"
            );
            let kernel = CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .unwrap();
            let mut y_copy = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(
                    &[a_strided.view(), b.view(), scales_t.view()],
                    &mut [y_copy.view_mut()],
                )
                .unwrap();

            assert_eq!(
                y_borrow.bytes, y_copy.bytes,
                "borrowed vs copied activation diverged (m={m}, k={k}, n={n}, block={block_size})"
            );
        }
    }

    /// Cross-CPU regression: an m=1 **asymmetric** int4 `accuracy_level=4`
    /// MatMulNBits must match its dequantized-f32 reference on every host. MLAS's
    /// AVX2 M=1 CompInt8 SQNBit kernel with a zero point is numerically broken
    /// (~46% error; see `try_mlas_sqnbit`'s guard and
    /// .squad/decisions/inbox/ripley-mlas-cross-cpu.md), so the routing must keep
    /// this case on the correct hand int8 decode path. Without the decode-min
    /// crossover *and* the explicit `try_mlas_sqnbit` guard this would produce
    /// garbage on AVX2 CI runners while passing on our AVX-512 dev hosts.
    #[test]
    fn matmulnbits_accuracy4_m1_asymmetric_matches_fp32_reference() {
        let _probe = lock_dispatch_probe();
        for &(k, n, block_size) in &[(256usize, 96usize, 32usize), (256, 64, 64), (128, 8, 128)] {
            let m = 1;
            let a_values: Vec<f32> = (0..m * k)
                .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
                .collect();
            let (packed, scales, zps, _) = quantize(&weights, n, k, block_size, true);
            let zps = zps.expect("asymmetric quantization must emit zero points");
            let dequantized = dequantize_reference(&packed, &scales, Some(&zps), n, k, block_size);
            let expected = reference(&a_values, &dequantized, m, k, n);

            let blocks = k.div_ceil(block_size);
            let (mut graph, node) = model_node(
                &[m, k],
                &[n, blocks, block_size / 2],
                &[n, blocks],
                Some(&[n, blocks.div_ceil(2)]),
                &[m, n],
                k,
                n,
                block_size,
            );
            graph
                .node_mut(node)
                .attributes
                .insert("accuracy_level".into(), Attribute::Int(4));
            let model = Model::new(&graph);
            let kernel = CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .unwrap();
            let a = Owned::f32(&[m, k], &a_values);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
            let scales_owned = Owned::f32(&[n, blocks], &scales);
            let zero_points = Owned::u8(&[n, blocks.div_ceil(2)], &zps);
            let mut y = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(
                    &[a.view(), b.view(), scales_owned.view(), zero_points.view()],
                    &mut [y.view_mut()],
                )
                .unwrap();
            assert_int8_close(&y.to_f32(), &expected);
        }
    }

    #[test]
    fn matmulnbits_fp32_activation_is_more_accurate_than_accuracy4() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (16, 2, 16);
        let mut weights_nk = vec![0i8; n * k];
        weights_nk[0] = 1;
        weights_nk[1] = 7;
        weights_nk[2] = -6;
        weights_nk[k] = 1;
        let mut packed = vec![0x88u8; n * block_size / 2];
        for (output, weights) in weights_nk.chunks_exact(k).enumerate() {
            for (depth, &weight) in weights.iter().enumerate() {
                let q = (weight + 8) as u8;
                let byte = &mut packed[output * block_size / 2 + depth / 2];
                if depth.is_multiple_of(2) {
                    *byte = (*byte & 0xf0) | q;
                } else {
                    *byte = (*byte & 0x0f) | (q << 4);
                }
            }
        }
        let scales = vec![1.0; n];
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);

        // max_abs=127 makes the accuracy-4 activation scale exactly 1.0. Thus
        // 0.49 rounds to 0 and 0.51 rounds to 1: the exact margin
        // 7*0.49 - 6*0.51 = +0.37 becomes -6 after activation quantization.
        let mut activations = vec![0.0; k];
        activations[..3].copy_from_slice(&[127.0, 0.49, 0.51]);
        let expected = reference(&activations, &dequantized, 1, k, n);
        let absent = run_decode_case(&activations, &packed, &scales, n, block_size, None);
        let level1 = run_decode_case(&activations, &packed, &scales, n, block_size, Some(1));
        let level4 = run_decode_case(&activations, &packed, &scales, n, block_size, Some(4));

        assert_close(&absent, &expected);
        assert_close(&level1, &expected);
        assert!(absent[0].total_cmp(&absent[1]).is_gt());
        assert!(level1[0].total_cmp(&level1[1]).is_gt());
        assert!(level4[1].total_cmp(&level4[0]).is_gt());
        let fp32_error = absent
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        let accuracy4_error = level4
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(
            accuracy4_error > fp32_error + 1.0,
            "accuracy-4 error {accuracy4_error} must materially exceed fp32 error {fp32_error}"
        );

        let mut on_grid = vec![0.0; k];
        on_grid[..3].copy_from_slice(&[127.0, 1.0, 1.0]);
        let on_grid_expected = reference(&on_grid, &dequantized, 1, k, n);
        assert_close(
            &run_decode_case(&on_grid, &packed, &scales, n, block_size, Some(4)),
            &on_grid_expected,
        );
    }

    #[test]
    fn matmulnbits_accuracy4_prepack_reuses_selected_weight_format() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (45, 5, 32);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 9.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 41) as f32 - 20.0) / 11.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let mut kernel = accuracy4_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);
        let a = Owned::f32(&[1, k], &activations);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        let direct_int4 = selected_dot_kernel().supports_int4_direct(block_size, false);
        let cached =
            prepack_cache_ptr(&kernel).expect("selected accuracy-4 weight cache must be populated");
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        let reused =
            prepack_cache_ptr(&kernel).expect("selected accuracy-4 weight cache must be reused");
        assert_eq!(reused, cached);
        assert!(kernel.weight_nk.get().is_none());
        // On a host whose hand int8 decode has no native dot product, MLAS
        // SQNBit CompInt8 owns accuracy-4 decode instead (see
        // `hand_int8_decode_has_native_dot`). The invariant under test -- one
        // weight format, chosen once and reused, never the f32 expansion --
        // holds either way; only *which* cache holds it differs.
        #[cfg(feature = "mlas")]
        if !hand_int8_decode_has_native_dot() {
            assert!(
                kernel
                    .mlas_shards
                    .get()
                    .is_some_and(|shards| shards.is_some())
                    || kernel
                        .mlas_packed
                        .get()
                        .is_some_and(|packed| packed.is_some()),
                "accuracy-4 decode must reach MLAS SQNBit where the hand int8 kernel has no native dot product"
            );
            assert!(
                kernel.int8_weight.get().is_none(),
                "MLAS-owned accuracy-4 decode must not also build the hand int8 weight"
            );
            return;
        }
        assert_eq!(
            kernel.packed_int4_weight.get().is_some()
                || kernel.packed_int4_n16_weight.get().is_some()
                || kernel.packed_kai_qsi4_weight.get().is_some()
                || {
                    #[cfg(feature = "mlas")]
                    {
                        kernel.mlas_shards.get().is_some()
                    }
                    #[cfg(not(feature = "mlas"))]
                    {
                        false
                    }
                },
            direct_int4
        );
        assert_eq!(kernel.int8_weight.get().is_some(), !direct_int4);
    }

    #[cfg(all(
        feature = "mlas",
        target_arch = "aarch64",
        not(any(target_os = "macos", target_os = "ios"))
    ))]
    #[test]
    fn matmulnbits_arm64_mlas_qnbit_reaches_qwen_decode_bits4_and_bits8() {
        let _probe = lock_dispatch_probe();
        let (k, n) = (128usize, 256usize);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 127) as f32 - 63.0) / 50.0)
            .collect();
        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("ONNX_GENAI_CPU_MM_MLAS_QNBIT").ok();
        for (label, override_value) in [("default", None), ("explicit", Some("1"))] {
            // SAFETY: the backend env lock serializes readers/writers of this var in tests.
            unsafe {
                match override_value {
                    Some(value) => std::env::set_var("ONNX_GENAI_CPU_MM_MLAS_QNBIT", value),
                    None => std::env::remove_var("ONNX_GENAI_CPU_MM_MLAS_QNBIT"),
                }
            }
            for (bits, block_size) in [(4usize, 128usize), (8, 128)] {
                let weights: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 31 % 251) as f32 - 125.0) / 50.0)
                    .collect();
                let (packed, scales, zps, _) = if bits == 4 {
                    quantize(&weights, n, k, block_size, true)
                } else {
                    quantize_8bit(&weights, n, k, block_size, true)
                };
                let zps = zps.expect("asymmetric quantization must emit qzeros");
                if mlas_sys::SQNBitPackedB::new(
                    n,
                    k,
                    bits,
                    block_size,
                    mlas_sys::SQNBitComputeType::Int8,
                    &packed,
                    &scales,
                    Some(&zps),
                )
                .is_none()
                {
                    eprintln!("MLAS QNBit bits={bits} CompInt8 unavailable; skipping reachability");
                    continue;
                }

                let mut kernel = accuracy4_kernel(k, n, block_size);
                kernel.bits = bits;
                kernel.set_constant_inputs(&[false, true, true, true]);
                let blocks = k.div_ceil(block_size);
                let blob = block_size * bits / 8;
                let zp_blob = (blocks * bits).div_ceil(8);
                let a = Owned::f32(&[1, k], &activations);
                let b = Owned::u8(&[n, blocks, blob], &packed);
                let scales = Owned::f32(&[n, blocks], &scales);
                let zero_points = Owned::u8(&[n, zp_blob], &zps);
                let mut y = Owned::zeros_f32(&[1, n]);

                let before = MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed);
                kernel
                    .execute(
                        &[a.view(), b.view(), scales.view(), zero_points.view()],
                        &mut [y.view_mut()],
                    )
                    .unwrap();
                assert!(
                    MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed) > before,
                    "{label}: bits={bits} block{block_size} asymmetric M=1 decode must route through MLAS QNBit",
                );
                assert!(
                    kernel.mlas_shards.get().and_then(Option::as_ref).is_some(),
                    "{label}: bits={bits} block{block_size} must prepack MLAS QNBit shards for decode by default",
                );
                assert!(
                    kernel.mlas_packed.get().is_none(),
                    "{label}: bits={bits} block{block_size} should only use full-width MLAS when ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1",
                );
                assert!(
                    kernel.packed_kai_qsi4_weight.get().is_none()
                        && kernel.packed_kai_qsi8_weight.get().is_none(),
                    "{label}: bits={bits} block{block_size} must prefer MLAS over the native KAI fallback",
                );
            }
        }
        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match previous {
                Some(value) => std::env::set_var("ONNX_GENAI_CPU_MM_MLAS_QNBIT", value),
                None => std::env::remove_var("ONNX_GENAI_CPU_MM_MLAS_QNBIT"),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmulnbits_arm64_kai_qsi4_block128_qwen_shapes_match_reference() {
        let _probe = lock_dispatch_probe();
        #[cfg(feature = "mlas")]
        let _guard = backend_env_lock().lock().unwrap();
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let block_size = 128usize;
        // Dispatch selects the KAI SDOT route only where production enables it:
        // `arm64_kai_sdot_direct_enabled` is off on macOS/iOS. Assert the
        // documented policy rather than unconditional support. The kernel
        // numerics below are exercised by *direct* calls, not by dispatch, so
        // they still run everywhere `dotprod` is present.
        assert_eq!(
            selected_dot_kernel().supports_int4_direct(block_size, false),
            DotKernel::arm64_kai_sdot_direct_enabled(),
            "int4-direct dispatch support must follow the documented per-OS KAI policy"
        );
        for &(k, n) in &[
            (1000usize, 257usize), // K and N tails.
            (1024, 1024),          // attention/output projection.
            (1024, 3072),          // gate/up projection.
            (2048, 1024),          // grouped attention projection.
            (3072, 1024),          // FFN down projection.
        ] {
            let activations: Vec<f32> = (0..k)
                .map(|i| ((i * 17 % 127) as f32 - 63.0) / 50.0)
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| ((i * 31 % 251) as f32 - 125.0) / 50.0)
                .collect();
            let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
            let kai_weight =
                prepack_kai_sdot_from_bytes(&packed, scales.clone(), None, n, k, 4, block_size);
            let mut scalar = vec![0.0f32; n];
            let mut dot = vec![0.0f32; n];
            kai_sdot_matmul_m1(
                &activations,
                &kai_weight,
                &mut scalar,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );
            kai_sdot_matmul_m1(
                &activations,
                &kai_weight,
                &mut dot,
                k,
                n,
                block_size,
                selected_dot_kernel(),
            );
            assert_close(&dot, &scalar);
            let expected = reference(&activations, &dequantized, 1, k, n);
            assert_qai8dxp_close(&dot, &expected);

            let blocks = k.div_ceil(block_size);
            let mut kernel = accuracy4_kernel(k, n, block_size);
            kernel.set_constant_inputs(&[false, true, true]);
            let a = Owned::f32(&[1, k], &activations);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
            let scales = Owned::f32(&[n, blocks], &scales);
            let mut y = Owned::zeros_f32(&[1, n]);
            kernel
                .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
                .unwrap();
            let used_kai = kernel.packed_kai_qsi4_weight.get().is_some();
            #[cfg(feature = "mlas")]
            let used_mlas = kernel.mlas_shards.get().is_some();
            #[cfg(not(feature = "mlas"))]
            let used_mlas = false;
            // Whether a fast int4 route exists at all is per-OS policy:
            // production disables KAI SDOT on macOS/iOS
            // (`arm64_kai_sdot_direct_enabled`). Only demand one where the
            // policy provides one; the kernel numerics above are direct calls
            // and ran regardless.
            let kai_dispatch_enabled = DotKernel::arm64_kai_sdot_direct_enabled();
            assert!(
                used_kai || used_mlas || !kai_dispatch_enabled,
                "aarch64 bits=4/block128 decode reached neither MLAS QNBit nor KAI SDOT for K={k} N={n}",
            );
            assert!(
                kernel.int8_weight.get().is_none() || !kai_dispatch_enabled,
                "block-128 direct int4 path must bypass the int8 prepack fallback"
            );
            if used_kai {
                assert_close(&y.to_f32(), &dot);
            } else if used_mlas {
                mlas_close(&y.to_f32(), &dot, 2e-1, "aarch64 MLAS qsi4 decode");
            } else {
                // No fast route by policy (macOS/iOS). Whatever generic
                // accuracy_level = 4 path runs must still track the f32 oracle
                // to CompInt8 tolerance.
                assert_qai8dxp_close(&y.to_f32(), &expected);
            }
        }
    }

    #[test]
    fn n16_sdot_int4_block128_asymmetric_matches_int8_and_f32_reference() {
        for &(k, n) in &[
            (1024usize, 1024usize),
            (1024, 3072),
            (1000, 257), // K and N tails.
        ] {
            let block_size = 128usize;
            let blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let mut packed = vec![0u8; n * blocks * blob];
            let mut scales = vec![0.0f32; n * blocks];
            let mut zero_points = vec![0u8; n * blocks.div_ceil(2)];
            for output in 0..n {
                for packed_block in 0..blocks.div_ceil(2) {
                    zero_points[output * blocks.div_ceil(2) + packed_block] =
                        if (output + packed_block).is_multiple_of(2) {
                            39
                        } else {
                            217
                        };
                }
                for block in 0..blocks {
                    scales[output * blocks + block] =
                        0.003 + ((output * 17 + block * 13) % 19) as f32 * 0.0007;
                    for offset in 0..block_size {
                        let byte = &mut packed[(output * blocks + block) * blob + offset / 2];
                        let q = ((output * 31 + block * 17 + offset * 7) & 0x0f) as u8;
                        if offset.is_multiple_of(2) {
                            *byte = (*byte & 0xf0) | q;
                        } else {
                            *byte = (*byte & 0x0f) | (q << 4);
                        }
                    }
                }
            }
            assert!(
                zero_points.contains(&39) && zero_points.contains(&217),
                "fixture must cover real Qwen-style packed qzero bytes 39..217"
            );
            let activations: Vec<f32> = (0..k)
                .map(|i| ((i * 19 % 251) as f32 - 125.0) / 80.0)
                .collect();
            let n16 = prepack_n16_sdot_from_bytes(
                &packed,
                scales.clone(),
                Some(&zero_points),
                n,
                k,
                4,
                block_size,
            );
            let mut n16_out = vec![0.0f32; n];
            n16_sdot_matmul_m1(
                &activations,
                &n16,
                &mut n16_out,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );

            let kernel = accuracy4_kernel(k, n, block_size);
            let b = Owned::u8(&[n, blocks, blob], &packed);
            let scales_t = Owned::f32(&[n, blocks], &scales);
            let zps_t = Owned::u8(&[n, blocks.div_ceil(2)], &zero_points);
            let int8_weight = kernel
                .prepack_int8_weight(&b.view(), &scales_t.view(), Some(&zps_t.view()))
                .unwrap();
            let mut int8_out = vec![0.0f32; n];
            int8_matmul(
                &activations,
                &int8_weight,
                &mut int8_out,
                1,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );
            assert_close(&n16_out, &int8_out);

            let dequantized =
                dequantize_reference(&packed, &scales, Some(&zero_points), n, k, block_size);
            let f32_reference = reference(&activations, &dequantized, 1, k, n);
            assert_int8_close(&n16_out, &f32_reference);
        }
    }

    #[test]
    fn kai_sdot_qsi4_block128_asymmetric_qwen_shapes_match_reference() {
        let _probe = lock_dispatch_probe();
        for &(k, n) in &[
            (1024usize, 1024usize),
            (1024, 3072),
            (1000, 257), // K and N tails.
        ] {
            let block_size = 128usize;
            let blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let mut packed = vec![0u8; n * blocks * blob];
            let mut scales = vec![0.0f32; n * blocks];
            let mut zero_points = vec![0u8; n * blocks.div_ceil(2)];
            let mut dequantized = vec![0.0f32; n * k];
            for output in 0..n {
                for packed_block in 0..blocks.div_ceil(2) {
                    zero_points[output * blocks.div_ceil(2) + packed_block] =
                        if (output + packed_block).is_multiple_of(2) {
                            39
                        } else {
                            217
                        };
                }
                for block in 0..blocks {
                    scales[output * blocks + block] =
                        0.003 + ((output * 17 + block * 13) % 19) as f32 * 0.0007;
                    let zp_byte = zero_points[output * blocks.div_ceil(2) + block / 2];
                    let zp = if block.is_multiple_of(2) {
                        zp_byte & 0x0f
                    } else {
                        zp_byte >> 4
                    };
                    for offset in 0..block_size {
                        let byte = &mut packed[(output * blocks + block) * blob + offset / 2];
                        let q = ((output * 31 + block * 17 + offset * 7) & 0x0f) as u8;
                        if offset.is_multiple_of(2) {
                            *byte = (*byte & 0xf0) | q;
                        } else {
                            *byte = (*byte & 0x0f) | (q << 4);
                        }
                        let depth = block * block_size + offset;
                        if depth < k {
                            dequantized[output * k + depth] =
                                (q as f32 - zp as f32) * scales[output * blocks + block];
                        }
                    }
                }
            }
            assert!(zero_points.contains(&39) && zero_points.contains(&217));
            let activations: Vec<f32> = (0..k)
                .map(|i| ((i * 19 % 251) as f32 - 125.0) / 80.0)
                .collect();
            let kai = prepack_kai_sdot_from_bytes(
                &packed,
                scales.clone(),
                Some(&zero_points),
                n,
                k,
                4,
                block_size,
            );
            let mut scalar = vec![0.0f32; n];
            let mut selected = vec![0.0f32; n];
            kai_sdot_matmul_m1(
                &activations,
                &kai,
                &mut scalar,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );
            kai_sdot_matmul_m1(
                &activations,
                &kai,
                &mut selected,
                k,
                n,
                block_size,
                selected_dot_kernel(),
            );
            assert_close(&selected, &scalar);
            let expected = reference(&activations, &dequantized, 1, k, n);
            assert_qai8dxp_close(&selected, &expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmulnbits_arm64_kai_qsi4_asymmetric_qwen_shape_is_reachable() {
        let _probe = lock_dispatch_probe();
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (k, n, block_size) = (1024usize, 1024usize, 128usize);
        // "Reachable" is a per-OS policy statement, not a universal one:
        // production disables the KAI SDOT route on macOS/iOS
        // (`arm64_kai_sdot_direct_enabled`), so on Apple silicon without MLAS
        // there is no fast int4 route for this shape to reach. The KAI kernel's
        // numerics stay covered there by
        // `matmulnbits_arm64_kai_qsi4_block128_qwen_shapes_match_reference`,
        // which calls it directly instead of through dispatch.
        assert_eq!(
            selected_dot_kernel().supports_int4_direct(block_size, true),
            DotKernel::arm64_kai_sdot_direct_enabled(),
            "int4-direct dispatch support must follow the documented per-OS KAI policy"
        );
        if !DotKernel::arm64_kai_sdot_direct_enabled() && !cfg!(feature = "mlas") {
            return;
        }
        let blocks = k.div_ceil(block_size);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 127) as f32 - 63.0) / 50.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 31 % 251) as f32 - 125.0) / 50.0)
            .collect();
        let (packed, scales, zero_points, dequantized) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.expect("asymmetric quantization emits qzeros");
        let expected = reference(&activations, &dequantized, 1, k, n);
        let mut kernel = accuracy4_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true, true]);
        let a = Owned::f32(&[1, k], &activations);
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales = Owned::f32(&[n, blocks], &scales);
        let zps = Owned::u8(&[n, blocks.div_ceil(2)], &zero_points);
        let mut y = Owned::zeros_f32(&[1, n]);
        let before = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed);
        #[cfg(feature = "mlas")]
        let before_mlas = MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zps.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        let reached_kai = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed) > before;
        #[cfg(feature = "mlas")]
        let reached_mlas = MLAS_SQNBIT_TEST_CALLS.load(Ordering::Relaxed) > before_mlas;
        #[cfg(not(feature = "mlas"))]
        let reached_mlas = false;
        assert!(
            reached_kai || reached_mlas,
            "real Qwen bits=4/block128/asymmetric shape reached neither MLAS QNBit nor KAI SDOT",
        );
        assert!(
            kernel.packed_kai_qsi4_weight.get().is_some() || {
                #[cfg(feature = "mlas")]
                {
                    kernel.mlas_shards.get().is_some()
                }
                #[cfg(not(feature = "mlas"))]
                {
                    false
                }
            }
        );
        assert!(kernel.int8_weight.get().is_none());
        assert_qai8dxp_close(&y.to_f32(), &expected);
    }

    #[test]
    fn matmulnbits_accuracy4_vnni_matches_scalar_when_available() {
        let activation: Vec<u8> = (0..128).map(|i| ((i * 29 + 7) % 255) as u8).collect();
        let weight: Vec<i8> = (0..128).map(|i| ((i * 17 % 31) as i8) - 15).collect();
        let scalar = dot_u8_i8(&activation, &weight, DotKernel::Scalar);
        let selected = selected_dot_kernel();
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avxvnni")
            || (std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            assert_ne!(
                selected,
                DotKernel::Scalar,
                "a VNNI CPU must select the VNNI path"
            );
        }
        assert_eq!(dot_u8_i8(&activation, &weight, selected), scalar);

        // Any AVX2 host (VNNI or not) must select a real SIMD kernel, never
        // Scalar, and the forced Avx2 dot must stay bit-exact vs Scalar.
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_ne!(
                selected,
                DotKernel::Scalar,
                "an AVX2 CPU must not select Scalar"
            );
            assert_eq!(dot_u8_i8(&activation, &weight, DotKernel::Avx2), scalar);
        }

        let activations: Vec<f32> = (0..256)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let values: Vec<i8> = (0..384).map(|i| ((i * 11 % 16) as i8) - 8).collect();
        let block_sums = values
            .chunks_exact(128)
            .map(|block| block.iter().map(|&value| value as i32).sum())
            .collect();
        let prepacked = Int8Weight {
            values,
            scales: vec![0.01, 0.02, 0.03],
            block_sums,
        };
        let mut scalar_output = vec![0.0; 6];
        let mut selected_output = vec![0.0; 6];
        int8_matmul(
            &activations,
            &prepacked,
            &mut scalar_output,
            2,
            128,
            3,
            128,
            DotKernel::Scalar,
        );
        int8_matmul(
            &activations,
            &prepacked,
            &mut selected_output,
            2,
            128,
            3,
            128,
            selected,
        );
        assert_close(&selected_output, &scalar_output);
    }

    /// End-to-end forced-AMX parity: drive the int4 `accuracy_level=4` CompInt8
    /// prefill (`M > 1`) through the AMX INT8 tile GEMM and confirm it is
    /// **bit-identical** to the `DotKernel::Scalar` reference across block sizes
    /// (32/64/128) and tile-unaligned `M`/`N`/`K` tails. AMX `tdpbusd` performs
    /// exact `u8 x i8 -> i32` tile MACs and the per-block `f32` scaling is
    /// applied in the same order as the scalar path, so the outputs must match
    /// exactly (not merely `assert_close`). Mirrors the forced-Avx2/Neon dot
    /// parity tests. Skipped on hosts without AMX (e.g. pre-Sapphire-Rapids CI).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int8_matmul_amx_matches_scalar_prefill() {
        if !amx::amx_int8_available() {
            eprintln!("skipping: AMX-INT8 not available on this host");
            return;
        }

        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // (m, k, n, block_size): full tiles, M/N tails, K tails within a block,
        // multi-sub-tile blocks (128), and a large multi-tile shape.
        let cases = [
            (16usize, 64usize, 16usize, 32usize),
            (17, 128, 20, 32),
            (33, 96, 7, 32),
            (64, 256, 64, 64),
            (128, 512, 100, 128),
            (200, 100, 48, 32),
            (16, 33, 16, 32),
            (48, 130, 40, 64),
        ];

        for &(m, k, n, block_size) in &cases {
            assert!(amx::amx_block_size_supported(block_size));
            let k_blocks = k.div_ceil(block_size);
            let padded_k = k_blocks * block_size;

            let activations: Vec<f32> = (0..m * k)
                .map(|_| ((next() % 4000) as f32 - 2000.0) / 173.0)
                .collect();
            // Weight values: random int8 in the valid K range, zero in the
            // per-row padded tail (exactly how `prepack_int8_weight` fills it).
            let values: Vec<i8> = (0..n * padded_k)
                .map(|index| {
                    let col = index % padded_k;
                    if col >= k {
                        0
                    } else {
                        ((next() % 15) as i8) - 7
                    }
                })
                .collect();
            let scales: Vec<f32> = (0..n * k_blocks)
                .map(|_| (next() % 100) as f32 / 5000.0 + 0.001)
                .collect();
            let block_sums: Vec<i32> = (0..n * k_blocks)
                .map(|index| {
                    let start = index * block_size;
                    values[start..start + block_size]
                        .iter()
                        .map(|&w| w as i32)
                        .sum()
                })
                .collect();
            let weight = Int8Weight {
                values,
                scales,
                block_sums,
            };

            let mut scalar_out = vec![0.0f32; m * n];
            int8_matmul(
                &activations,
                &weight,
                &mut scalar_out,
                m,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );

            let mut amx_out = vec![0.0f32; m * n];
            amx::int8_matmul_amx(&activations, &weight, &mut amx_out, m, k, n, block_size);

            assert_eq!(
                amx_out, scalar_out,
                "AMX prefill diverged from scalar for m={m} k={k} n={n} block_size={block_size}"
            );
        }
    }

    #[test]
    fn matmulnbits_direct_int4_gemv_matches_int8_reference() {
        let (k, n, block_size) = (77usize, 9usize, 32usize);
        let blocks = k.div_ceil(block_size);
        let padded_k = blocks * block_size;
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let packed_weight = PackedInt4Weight {
            values: packed.clone(),
            scales: scales.clone(),
        };
        let kernel = accuracy4_kernel(k, n, block_size);
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales_tensor = Owned::f32(&[n, blocks], &scales);
        let int8_weight = kernel
            .prepack_int8_weight(&b.view(), &scales_tensor.view(), None)
            .unwrap();
        let mut expected = vec![0.0; n];
        let mut scalar = vec![0.0; n];
        let mut actual = vec![0.0; n];
        int8_matmul(
            &activations,
            &int8_weight,
            &mut expected,
            1,
            k,
            n,
            block_size,
            DotKernel::Scalar,
        );
        int4_matmul_m1(
            &activations,
            &packed_weight,
            &mut scalar,
            k,
            n,
            block_size,
            DotKernel::Scalar,
        );
        int4_matmul_m1(
            &activations,
            &packed_weight,
            &mut actual,
            k,
            n,
            block_size,
            selected_dot_kernel(),
        );
        assert_eq!(
            padded_k,
            activations.len().div_ceil(block_size) * block_size
        );
        for (index, ((&actual, &scalar), &expected)) in
            actual.iter().zip(&scalar).zip(&expected).enumerate()
        {
            let tolerance = 1e-4 + 1e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: direct int4={actual}, int8 reference={expected}, tolerance={tolerance}"
            );
            assert!(
                (scalar - expected).abs() <= tolerance,
                "index {index}: scalar int4={scalar}, int8 reference={expected}, tolerance={tolerance}"
            );
        }
    }

    fn rmse(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        let sum: f32 = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e) * (a - e))
            .sum();
        (sum / actual.len() as f32).sqrt()
    }

    /// Fake-quantize `activations` to int8 the way the CompInt8 path does, then
    /// dequantize, so the resulting matmul against the f32-dequantized weights
    /// isolates the *activation* quantization error. `per_block == false`
    /// reproduces the old single-per-row scale (one outlier inflates the whole
    /// row); `per_block == true` is the ORT/MLAS scheme this fix adopts.
    fn activation_quant_oracle(
        activations: &[f32],
        dequantized: &[f32],
        k: usize,
        n: usize,
        block_size: usize,
        per_block: bool,
    ) -> Vec<f32> {
        let mut hat = activations.to_vec();
        if per_block {
            for block in 0..k.div_ceil(block_size) {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let max_abs = activations[start..end]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0, f32::max);
                if max_abs == 0.0 {
                    continue;
                }
                let scale = max_abs / 127.0;
                for i in start..end {
                    hat[i] = (activations[i] / scale).round().clamp(-127.0, 127.0) * scale;
                }
            }
        } else {
            let max_abs = activations.iter().map(|v| v.abs()).fold(0.0, f32::max);
            let scale = max_abs / 127.0;
            for value in hat.iter_mut() {
                *value = (*value / scale).round().clamp(-127.0, 127.0) * scale;
            }
        }
        reference(&hat, dequantized, 1, k, n)
    }

    /// Root-cause regression: cross-K-block magnitude spread is exactly what
    /// broke the old single-scale (per-row) CompInt8 quant -- the largest block
    /// set one row scale that crushed every smaller block's int8 codes, leaving
    /// native ~2.6x less accurate than ORT. Per-block activation scaling
    /// (ORT/MLAS `QuantizeARow_CompInt8`) must keep BOTH accuracy-level-4 hand
    /// paths (int4 GEMV and the int4->int8 GEMM) within ORT-class error of the
    /// dequantized-f32 oracle, and must be materially better than per-row.
    #[test]
    fn matmulnbits_compint8_per_block_activation_tracks_dequant_f32_oracle() {
        let (k, n, block_size) = (256usize, 8usize, 32usize);
        let blocks = k.div_ceil(block_size);

        // Anti-correlated magnitudes: block 0 has a large activation amplitude but
        // tiny weights, while every other block has a small activation amplitude
        // paired with large weights. A per-row scale is pinned by block 0 and
        // crushes the small-activation blocks to a handful of int8 levels -- yet
        // those blocks carry most of the output signal (large weights), so the
        // per-row error is large. Per-block scaling keeps each block at full
        // int8 resolution. This mirrors the real Qwen/Phi CompInt8 divergence.
        let amp_a = |block: usize| if block == 0 { 2.5 } else { 0.06 };
        let amp_w = |block: usize| if block == 0 { 0.05 } else { 1.5 };
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i as f32 * 0.37).sin()) * amp_a(i / block_size))
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.11).cos()) * amp_w((i % k) / block_size))
            .collect();
        let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
        let oracle = reference(&activations, &dequantized, 1, k, n);
        let oracle_rms = rmse(&oracle, &vec![0.0; n]);

        let packed_weight = PackedInt4Weight {
            values: packed.clone(),
            scales: scales.clone(),
        };
        let mut int4_out = vec![0.0; n];
        int4_matmul_m1(
            &activations,
            &packed_weight,
            &mut int4_out,
            k,
            n,
            block_size,
            DotKernel::Scalar,
        );

        let kernel = accuracy4_kernel(k, n, block_size);
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales_tensor = Owned::f32(&[n, blocks], &scales);
        let int8_weight = kernel
            .prepack_int8_weight(&b.view(), &scales_tensor.view(), None)
            .unwrap();
        let mut int8_out = vec![0.0; n];
        int8_matmul(
            &activations,
            &int8_weight,
            &mut int8_out,
            1,
            k,
            n,
            block_size,
            DotKernel::Scalar,
        );

        // ORT-class accuracy is a small *relative* error vs the f32 oracle.
        let int4_rel = rmse(&int4_out, &oracle) / oracle_rms;
        let int8_rel = rmse(&int8_out, &oracle) / oracle_rms;
        assert!(
            int4_rel <= 5e-3,
            "per-block int4 CompInt8 relative RMSE {int4_rel} exceeds ORT-class 5e-3",
        );
        assert!(
            int8_rel <= 5e-3,
            "per-block int8 CompInt8 relative RMSE {int8_rel} exceeds ORT-class 5e-3",
        );

        let per_row_rel = rmse(
            &activation_quant_oracle(&activations, &dequantized, k, n, block_size, false),
            &oracle,
        ) / oracle_rms;
        let per_block_rel = rmse(
            &activation_quant_oracle(&activations, &dequantized, k, n, block_size, true),
            &oracle,
        ) / oracle_rms;
        assert!(
            per_block_rel < per_row_rel * 0.25,
            "per-block activation quant ({per_block_rel}) must be far better than per-row ({per_row_rel})",
        );
    }

    /// Fixed-form of the CompInt8 argmax-reversal characterization (the bug seen
    /// at Qwen2.5-Coder-7B decode index 23 and Phi-3.5-mini index 2): at a near
    /// tie the native accuracy-level-4 path must pick the SAME greedy winner as
    /// the dequantized-f32 oracle. Activations carry cross-block magnitude spread
    /// so the old per-row scale is exercised. The near-tie window is kept
    /// comfortably above 2x the ORT-class per-output error (measured relative to
    /// the output magnitude) so a passing argmax is a real accuracy result.
    #[test]
    fn matmulnbits_compint8_argmax_matches_dequant_f32_oracle_at_near_tie() {
        let (k, n, block_size) = (128usize, 2usize, 32usize);
        let mut checked = 0usize;
        for seed in 1..=400u32 {
            let s = seed as f32;
            let activations: Vec<f32> = (0..k)
                .map(|i| {
                    let block = (i / block_size) as f32;
                    ((i as f32 * 0.017 + s * 0.013).sin()) * (0.05 + 0.45 * block)
                })
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 * 0.011 + s * 0.019).cos())
                .collect();
            let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
            let oracle = reference(&activations, &dequantized, 1, k, n);
            let oracle_rms = rmse(&oracle, &vec![0.0; n]).max(1e-6);
            let margin_rel = (oracle[1] - oracle[0]).abs() / oracle_rms;
            // Genuine relative near-ties only, but wide enough that ORT-class
            // error (< 0.5% relative) cannot legitimately flip the winner.
            if !(0.02..=0.08).contains(&margin_rel) {
                continue;
            }
            checked += 1;
            let packed_weight = PackedInt4Weight {
                values: packed.clone(),
                scales: scales.clone(),
            };
            let mut native = vec![0.0; n];
            int4_matmul_m1(
                &activations,
                &packed_weight,
                &mut native,
                k,
                n,
                block_size,
                selected_dot_kernel(),
            );
            let case_rel = rmse(&native, &oracle) / oracle_rms;
            assert!(
                case_rel <= 5e-3,
                "seed {seed}: native CompInt8 relative RMSE {case_rel} exceeds ORT-class 5e-3",
            );
            assert_eq!(
                usize::from(native[1] > native[0]),
                usize::from(oracle[1] > oracle[0]),
                "seed {seed}: native CompInt8 argmax != f32 oracle (margin_rel {margin_rel}, native {native:?}, oracle {oracle:?})",
            );
        }
        assert!(
            checked >= 5,
            "deterministic search must exercise several near-tie cases (got {checked})",
        );
    }

    /// Divergence-class guard for the Phi-3.5-mini int4 (block-32, acc-level-4)
    /// native-vs-ORT greedy split at decode index 65 (native picks token 263, ORT
    /// picks 6455). An independent fp32/fp16/bf16 oracle (the same `model.onnx`
    /// run through ONNX Runtime with every `MatMulNBits` `accuracy_level`
    /// rewritten to 1/2/3) selects 263 by a +0.0128 logit margin (~0.02% of the
    /// ~59.7 logit); ONLY `accuracy_level=4` (int8 *activation* quantization)
    /// flips the winner to 6455. Native matches the high-precision oracle, so per
    /// project policy native is KEPT as the more-accurate backend.
    ///
    /// The end-to-end flip is a whole-graph accumulation effect, but its root
    /// mechanism is int8 activation quantization tipping a razor-thin logit race
    /// -- the same class Ridley proved on qwen3-0.6b. This pins that mechanism at
    /// the kernel level, model-independently: on genuine near-ties, the loose
    /// per-ROW int8 activation scale (one cross-block outlier crushes the smaller
    /// blocks' int8 codes) reverses the fp32-reference argmax, while native's
    /// decode kernel -- which quantizes activations per BLOCK -- must preserve the
    /// fp32-reference winner on EVERY near-tie. A regression that reverted native
    /// to a per-row (or otherwise looser) activation scale would reintroduce
    /// exactly this greedy-token divergence, and this test would catch it.
    #[test]
    fn int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips() {
        let (k, n, block_size) = (128usize, 2usize, 32usize);
        let mut near_ties = 0usize;
        let mut per_row_flips = 0usize;
        for seed in 1..=4000u32 {
            let s = seed as f32;
            // Anti-correlated magnitude spread: block 0 pairs a large activation
            // outlier with tiny weights (it pins any per-row activation scale and
            // crushes every other block to a few int8 codes) while the remaining
            // blocks pair small activations with large weights (they carry the
            // output signal). This is the CompInt8 failure geometry from the real
            // Qwen/Phi divergences.
            let activations: Vec<f32> = (0..k)
                .map(|i| {
                    let amp = if i / block_size == 0 { 3.0 } else { 0.05 };
                    (i as f32 * 0.017 + s * 0.013).sin() * amp
                })
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| {
                    let amp = if (i % k) / block_size == 0 { 0.03 } else { 1.5 };
                    (i as f32 * 0.011 + s * 0.019).cos() * amp
                })
                .collect();
            let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
            let oracle = reference(&activations, &dequantized, 1, k, n);
            let oracle_rms = rmse(&oracle, &vec![0.0; n]).max(1e-6);
            let margin_rel = (oracle[1] - oracle[0]).abs() / oracle_rms;
            // Genuine near-ties only. The upper bound keeps the race tight; the
            // lower bound stays above native's per-block int8 error so a passing
            // argmax is a real accuracy result, not luck.
            if !(0.005..=0.08).contains(&margin_rel) {
                continue;
            }
            near_ties += 1;
            let oracle_argmax = usize::from(oracle[1] > oracle[0]);

            // The loose per-row int8 activation scale (the failure mode): it may
            // flip the greedy winner. We only *witness* that it happens on this
            // family; native must never do the same.
            let per_row =
                activation_quant_oracle(&activations, &dequantized, k, n, block_size, false);
            if usize::from(per_row[1] > per_row[0]) != oracle_argmax {
                per_row_flips += 1;
            }

            // Native's real int4 block-32 decode kernel (per-block int8
            // activation). It must keep the fp32-reference argmax on every
            // near-tie -- both scalar and the host-selected SIMD kernel.
            let packed_weight = PackedInt4Weight {
                values: packed.clone(),
                scales: scales.clone(),
            };
            #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
            let mut kernels = vec![DotKernel::Scalar, selected_dot_kernel()];
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                kernels.push(DotKernel::Avx2);
            }
            for kernel in kernels {
                let mut native = vec![0.0; n];
                int4_matmul_m1(
                    &activations,
                    &packed_weight,
                    &mut native,
                    k,
                    n,
                    block_size,
                    kernel,
                );
                assert_eq!(
                    usize::from(native[1] > native[0]),
                    oracle_argmax,
                    "seed {seed} ({kernel:?}): native per-block int4 decode flipped the fp32 \
                     argmax at a near-tie (margin_rel {margin_rel}, native {native:?}, \
                     oracle {oracle:?}) -- this is the Phi-3.5 index-65 divergence class",
                );
            }
        }
        assert!(
            near_ties >= 20,
            "deterministic search must exercise many near-ties (got {near_ties})",
        );
        assert!(
            per_row_flips >= 3,
            "the per-row int8 activation failure mode must actually flip some near-ties \
             (got {per_row_flips}); otherwise this test does not witness the divergence class",
        );
    }

    /// Round-trip quantize `weights_nk` ([N, K], row-major) to 8-bit blocks the
    /// way ORT's `MatMulNBits` stores them (LSB-first, one byte per weight, one
    /// scale per K block, an optional one-byte-per-block uint8 zero point).
    /// Returns the packed bytes, scales, optional zero points, and the exact f32
    /// dequantization (`(q - zp) * scale`) so a test can build the same oracle
    /// the kernel's `dequantize_weight` reconstructs.
    fn quantize_8bit(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob = block_size; // 8 bits -> one byte per weight
        let mut packed = vec![0u8; n * blocks * blob];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![128u8; n * blocks];
        let mut dequantized = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let (scale, zp) = if asymmetric {
                    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((max - min) / 255.0).max(1e-6);
                    (scale, (-min / scale).round().clamp(0.0, 255.0) as u8)
                } else {
                    let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    ((max_abs / 127.0).max(1e-6), 128u8)
                };
                scales[row * blocks + block] = scale;
                zps[row * blocks + block] = zp;
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 255.0) as u8;
                    packed[(row * blocks + block) * blob + offset] = q;
                    dequantized[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, asymmetric.then_some(zps), dequantized)
    }

    #[allow(clippy::needless_range_loop)]
    fn bits8_n16_i16_reference(
        activation: &[f32],
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let group = 32usize;
        let groups = k.div_ceil(group);
        let mut quantized = vec![0i16; k];
        let mut group_scales = vec![0.0f32; groups];
        for g in 0..groups {
            let start = g * group;
            let end = (start + group).min(k);
            group_scales[g] =
                quantize_block_i16(&activation[start..end], &mut quantized[start..end]);
        }
        let mut output = vec![0.0f32; n];
        for column in 0..n {
            for block in 0..blocks {
                let block_start = block * block_size;
                let block_end = (block_start + block_size).min(k);
                let zp = zero_points.map_or(128u8, |points| points[column * blocks + block]);
                let scale = scales[column * blocks + block];
                let block_sum: f32 = activation[block_start..block_end].iter().sum();
                let mut product = 0.0f32;
                for g in block_start / group..block_end.div_ceil(group) {
                    let start = (g * group).max(block_start);
                    let end = ((g + 1) * group).min(block_end);
                    let dot: i32 = (start..end)
                        .map(|depth| {
                            let q = packed
                                [(column * blocks + block) * block_size + (depth - block_start)];
                            q as i32 * quantized[depth] as i32
                        })
                        .sum();
                    product += group_scales[g] * dot as f32;
                }
                output[column] += scale * product - scale * zp as f32 * block_sum;
            }
        }
        output
    }

    /// Whether the executed 8-bit `m == 1` route quantizes the activations to
    /// int8, and therefore cannot be held to the fp32 oracle's tolerance.
    ///
    /// Both the KleidiAI SDOT route (`kai_sdot_matmul_m1` ->
    /// `quantize_activation_qai8dxp`) and the N16 SDOT route
    /// (`n16_sdot_u8_i16_matmul_m1` -> `quantize_activation_signed`) do this,
    /// and the KAI route is tried *first*. The predicate used to name only the
    /// N16 route; that was latent because the dispatch gates were hard-wired
    /// `true` under `cfg(test)`, so whenever KAI was active N16 was too. With
    /// the gates now inheriting the production default the two can differ, so
    /// the predicate has to mirror the real dispatch order.
    ///
    /// `accuracy_level` mirrors the production gate: int8 activation compute is
    /// ONNX CompInt8 and is only reachable at `accuracy_level == 4`, so at
    /// every other level this returns `false` and the fp32 tolerance applies.
    fn bits8_int8_activation_active_for_test(accuracy_level: i64, block_size: usize) -> bool {
        if accuracy_level != 4 {
            return false;
        }
        let dot_kernel = selected_dot_kernel();
        dot_kernel.uses_kai_sdot_direct(8, block_size)
            || (dot_kernel.uses_n16_sdot_direct()
                && block_size == 128
                && activation_quant_group() == 32)
    }

    #[test]
    fn n16_sdot_bits8_block128_asymmetric_matches_i16_and_f32_reference() {
        for &(k, n) in &[
            (1024usize, 1024usize),
            (1024, 3072),
            (1000, 257), // K and N tails.
        ] {
            let block_size = 128usize;
            let weights_nk: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 * 0.011).sin() * 0.9 + (i as f32 * 0.0003).cos() * 0.25)
                .collect();
            let activations: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.019 + 0.4).cos() * 0.7)
                .collect();
            let (mut packed, scales, mut zero_points, mut dequantized) =
                quantize_8bit(&weights_nk, n, k, block_size, true);
            let zero_points = zero_points.as_mut().expect("asymmetric bits8 emits qzeros");
            let blocks = k.div_ceil(block_size);
            for output in 0..n {
                for block in 0..blocks {
                    zero_points[output * blocks + block] = if (output + block).is_multiple_of(2) {
                        39
                    } else {
                        217
                    };
                    for offset in 0..block_size {
                        let q = ((output * 37 + block * 19 + offset * 11) & 0xff) as u8;
                        packed[(output * blocks + block) * block_size + offset] = q;
                        let depth = block * block_size + offset;
                        if depth < k {
                            dequantized[output * k + depth] = (q as f32
                                - zero_points[output * blocks + block] as f32)
                                * scales[output * blocks + block];
                        }
                    }
                }
            }
            assert!(zero_points.contains(&39) && zero_points.contains(&217));
            let n16 = prepack_n16_sdot_from_bytes(
                &packed,
                scales.clone(),
                Some(zero_points),
                n,
                k,
                8,
                block_size,
            );
            let mut n16_out = vec![0.0f32; n];
            n16_sdot_u8_i16_matmul_m1(
                &activations,
                &n16,
                &mut n16_out,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );
            let i16_reference = bits8_n16_i16_reference(
                &activations,
                &packed,
                &scales,
                Some(zero_points),
                n,
                k,
                block_size,
            );
            assert_close(&n16_out, &i16_reference);

            let f32_reference = reference(&activations, &dequantized, 1, k, n);
            assert_int8_close(&n16_out, &f32_reference);
        }
    }

    #[test]
    fn kai_sdot_qsi8_block128_asymmetric_qwen_shapes_match_reference() {
        let _probe = lock_dispatch_probe();
        for &(k, n) in &[
            (1024usize, 1024usize),
            (1024, 3072),
            (1000, 257), // K and N tails.
        ] {
            let block_size = 128usize;
            let weights_nk: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 * 0.011).sin() * 0.9 + (i as f32 * 0.0003).cos() * 0.25)
                .collect();
            let activations: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.019 + 0.4).cos() * 0.7)
                .collect();
            let (mut packed, scales, mut zero_points, mut dequantized) =
                quantize_8bit(&weights_nk, n, k, block_size, true);
            let zero_points = zero_points.as_mut().expect("asymmetric bits8 emits qzeros");
            let blocks = k.div_ceil(block_size);
            for output in 0..n {
                for block in 0..blocks {
                    zero_points[output * blocks + block] = if (output + block).is_multiple_of(2) {
                        39
                    } else {
                        217
                    };
                    for offset in 0..block_size {
                        let q = ((output * 37 + block * 19 + offset * 11) & 0xff) as u8;
                        packed[(output * blocks + block) * block_size + offset] = q;
                        let depth = block * block_size + offset;
                        if depth < k {
                            dequantized[output * k + depth] = (q as f32
                                - zero_points[output * blocks + block] as f32)
                                * scales[output * blocks + block];
                        }
                    }
                }
            }
            assert!(zero_points.contains(&39) && zero_points.contains(&217));
            let kai = prepack_kai_sdot_from_bytes(
                &packed,
                scales.clone(),
                Some(zero_points),
                n,
                k,
                8,
                block_size,
            );
            let mut scalar = vec![0.0f32; n];
            let mut selected = vec![0.0f32; n];
            kai_sdot_matmul_m1(
                &activations,
                &kai,
                &mut scalar,
                k,
                n,
                block_size,
                DotKernel::Scalar,
            );
            kai_sdot_matmul_m1(
                &activations,
                &kai,
                &mut selected,
                k,
                n,
                block_size,
                selected_dot_kernel(),
            );
            assert_close(&selected, &scalar);
            let expected = reference(&activations, &dequantized, 1, k, n);
            assert_qai8dxp_close(&selected, &expected);
        }
    }

    /// The KleidiAI SDOT int8 route must be reachable for the real Qwen
    /// bits=8/block128/asymmetric decode shape -- but only at
    /// `accuracy_level = 4`.
    ///
    /// `kai_sdot_matmul_m1` quantizes the activations via
    /// `quantize_activation_qai8dxp`, i.e. it is ONNX CompInt8. This test used
    /// to build the node with no `accuracy_level` attribute (so 0 == CompFp32)
    /// and assert the int8 route was taken anyway, with the loose
    /// `assert_qai8dxp_close` tolerance -- encoding the precision bug as an
    /// expectation. It now asserts both halves of the real contract: acc4
    /// reaches KAI, and acc0 does *not*.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmulnbits_arm64_kai_qsi8_asymmetric_qwen_shape_is_reachable() {
        let _probe = lock_dispatch_probe();
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (k, n, block_size) = (1024usize, 1024usize, 128usize);
        let blocks = k.div_ceil(block_size);
        let activations: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.017 + 0.2).sin() * 0.8)
            .collect();
        let weights_nk: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.013).cos() * 1.1).collect();
        let (packed, scales, zero_points, dequantized) =
            quantize_8bit(&weights_nk, n, k, block_size, true);
        let zero_points = zero_points.expect("asymmetric bits8 emits qzeros");
        let expected = reference(&activations, &dequantized, 1, k, n);

        let run = |accuracy_level: i64| -> (Vec<f32>, bool) {
            let (mut graph, node) = model_node(
                &[1, k],
                &[n, blocks, block_size],
                &[n, blocks],
                Some(&[n, blocks]),
                &[1, n],
                k,
                n,
                block_size,
            );
            graph
                .node_mut(node)
                .attributes
                .insert("bits".into(), Attribute::Int(8));
            graph
                .node_mut(node)
                .attributes
                .insert("accuracy_level".into(), Attribute::Int(accuracy_level));
            let model = Model::new(&graph);
            let kernel = CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .unwrap();
            let a = Owned::f32(&[1, k], &activations);
            let b = Owned::u8(&[n, blocks, block_size], &packed);
            let scales = Owned::f32(&[n, blocks], &scales);
            let zps = Owned::u8(&[n, blocks], &zero_points);
            let mut y = Owned::zeros_f32(&[1, n]);
            let before = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed);
            kernel
                .execute(
                    &[a.view(), b.view(), scales.view(), zps.view()],
                    &mut [y.view_mut()],
                )
                .unwrap();
            let reached_kai = KAI_SDOT_M1_TEST_CALLS.load(Ordering::Relaxed) > before;
            (y.to_f32(), reached_kai)
        };

        // accuracy_level = 4 (CompInt8): the int8 route is licensed, so it must
        // be reached wherever production enables it. `arm64_kai_sdot_direct_enabled`
        // is off on macOS/iOS, so reachability is asserted against that policy
        // rather than unconditionally.
        let kai_enabled = DotKernel::arm64_kai_sdot_direct_enabled();
        let (acc4, acc4_reached_kai) = run(4);
        assert_eq!(
            acc4_reached_kai, kai_enabled,
            "accuracy_level=4 KAI SDOT reachability must follow the documented per-OS policy \
             (enabled={kai_enabled})"
        );
        if kai_enabled {
            // Downcast to the concrete kernel type is not available through the
            // EP trait object; the counter above is the dispatch proof.
            assert_qai8dxp_close(&acc4, &expected);
        }

        // accuracy_level = 0 (CompFp32): the int8 route must be declined, and
        // the fp32 fallback must hold the tight oracle tolerance. This half is
        // the precision regression guard and runs on *every* aarch64 host,
        // including Apple silicon. Before the gate this shape ran CompInt8 on
        // non-Apple aarch64, where `arm64_kai_sdot_direct_enabled` defaults on.
        let (acc0, acc0_reached_kai) = run(0);
        assert!(
            !acc0_reached_kai,
            "accuracy_level=0 is CompFp32 and must not reach the activation-quantizing KAI SDOT route"
        );
        // `assert_close`'s absolute 1e-5 does not scale with K, and this shape
        // accumulates K=1024 products, so compare relative RMSE against the
        // oracle instead. fp32 reassociation lands around 1e-6 relative; the
        // int8 route lands around 1e-3, so 1e-5 cleanly separates them.
        let oracle_rms = rmse(&expected, &vec![0.0; expected.len()]).max(1e-6);
        let acc0_rel = rmse(&acc0, &expected) / oracle_rms;
        let acc4_rel = rmse(&acc4, &expected) / oracle_rms;
        assert!(
            acc0_rel <= 1e-5,
            "accuracy_level=0 must reconstruct the fp32 oracle: relative RMSE {acc0_rel} exceeds 1e-5"
        );
        if kai_enabled {
            // Self-calibrating: whatever this host's absolute error happens to
            // be, the CompFp32 route must be strictly more accurate than the
            // CompInt8 one. Only meaningful where acc4 actually took CompInt8.
            assert!(
                acc0_rel < acc4_rel,
                "accuracy_level=0 (relative RMSE {acc0_rel}) must be more accurate than accuracy_level=4 (relative RMSE {acc4_rel})"
            );
        }
    }

    /// Run the real end-to-end `MatMulNBits` kernel (`execute`) for an 8-bit,
    /// block-128 weight -- the exact path Qwen3-0.6B CPU int8 decode and prefill
    /// take (`dequantize_weight` -> `gemv_nk` for M=1, `-> gemm` for M>1). The
    /// output must track an independent dequantize-to-f32 oracle to near-float
    /// precision AND, crucially for greedy token selection, pick the SAME argmax
    /// in every row. This pins the 8-bit block-128 route that previously had no
    /// end-to-end execute-level oracle coverage.
    fn run_8bit_execute(
        n: usize,
        k: usize,
        block_size: usize,
        m: usize,
        asymmetric: bool,
        activations: &[f32],
        weights_nk: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let (packed, scales, zps, dequantized) =
            quantize_8bit(weights_nk, n, k, block_size, asymmetric);

        let zp_shape = zps.as_ref().map(|_| vec![n, blocks]);
        let (graph, node) = model_node(
            &[m, k],
            &[n, blocks, block_size],
            &[n, blocks],
            zp_shape.as_deref(),
            &[m, n],
            k,
            n,
            block_size,
        );
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("bits=8 block-128 kernel must build");

        let a = Owned::f32(&[m, k], activations);
        let b = Owned::u8(&[n, blocks, block_size], &packed);
        let scales_tensor = Owned::f32(&[n, blocks], &scales);
        let zp_owned = zps.as_ref().map(|z| Owned::u8(&[n, blocks], z));
        let mut inputs = vec![a.view(), b.view(), scales_tensor.view()];
        if let Some(zp) = zp_owned.as_ref() {
            inputs.push(zp.view());
        }
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&inputs, &mut [y.view_mut()])
            .expect("8-bit block-128 execute must succeed");

        let oracle = reference(activations, &dequantized, m, k, n);
        (y.to_f32(), oracle)
    }

    /// Run the 8-bit `execute` path at an explicit `accuracy_level` and return
    /// the largest absolute deviation from a float64 dequantize-and-reduce
    /// oracle.
    ///
    /// float64, not the usual f32 `reference`, because the quantity under test
    /// here *is* the reduction's precision: an f32 oracle carries its own
    /// ~1e-4 of rounding at this K and would mask the effect entirely.
    fn bits8_m1_max_abs_error_at(
        accuracy_level: i64,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> f32 {
        // One large value per block and many small ones: a per-block int16
        // activation scale is set by the large value, so every small value is
        // quantized coarsely. This is the shape of real decode activations
        // (a few outlier channels) and it is where a 16-bit activation loses
        // most, which is what makes this test sensitive rather than lucky.
        let activations: Vec<f32> = (0..k)
            .map(|i| {
                if i % block_size == 0 {
                    18.0
                } else {
                    (i as f32 * 0.031 + 0.7).sin() * 0.05
                }
            })
            .collect();
        let weights_nk: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.0091).sin() * 1.2 + (i as f32 * 0.0003).cos() * 0.5)
            .collect();
        let blocks = k.div_ceil(block_size);
        let (packed, scales, _, dequantized) = quantize_8bit(&weights_nk, n, k, block_size, false);
        let (graph, node) = model_node(
            &[1, k],
            &[n, blocks, block_size],
            &[n, blocks],
            None,
            &[1, n],
            k,
            n,
            block_size,
        );
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        graph
            .node_mut(node)
            .attributes
            .insert("accuracy_level".into(), Attribute::Int(accuracy_level));
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("bits=8 kernel must build");
        let a = Owned::f32(&[1, k], &activations);
        let b = Owned::u8(&[n, blocks, block_size], &packed);
        let scales_tensor = Owned::f32(&[n, blocks], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales_tensor.view()],
                &mut [y.view_mut()],
            )
            .expect("8-bit execute must succeed");
        let out = y.to_f32();
        (0..n)
            .map(|column| {
                let oracle: f64 = (0..k)
                    .map(|i| activations[i] as f64 * dequantized[column * k + i] as f64)
                    .sum();
                (out[column] as f64 - oracle).abs() as f32
            })
            .fold(0.0f32, f32::max)
    }

    /// `accuracy_level` 0/1 mean fp32 compute, so the 8-bit M=1 decode GEMV may
    /// not quantize the activation -- not to int8, and not to int16 either.
    ///
    /// This is a *precision* regression test, and it is built to fail if the
    /// gate is removed: the same input at `accuracy_level = 2` (fp16 compute
    /// permitted) is asserted to be at least 20x less accurate, which both
    /// proves the int16 path is genuinely reachable for this shape -- so the
    /// fp32 assertion is not passing vacuously on a shape that never had a fast
    /// path -- and pins the size of what the gate is holding back.
    ///
    /// Measured against ONNX Runtime 1.27 on the A/B harness at K=N=3584, the
    /// ungated int16 path deviated by 6.0e-3 where ORT itself deviates 1.2e-4
    /// from a float64 oracle, i.e. ~50x worse, while buying 12%.
    #[test]
    fn eight_bit_decode_keeps_fp32_activations_unless_the_model_allows_less() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (32usize, 2048usize, 128usize);
        let fp32_error = bits8_m1_max_abs_error_at(0, n, k, block_size);
        let level1_error = bits8_m1_max_abs_error_at(1, n, k, block_size);
        let reduced_error = bits8_m1_max_abs_error_at(2, n, k, block_size);

        // The weights are 8-bit quantized, so even an exact reduction carries
        // the weight quantization error; what must NOT appear on top of it is
        // activation quantization. Scale the bound to the operand magnitudes
        // rather than hard-coding a constant that a future shape change would
        // silently loosen.
        // Tight enough to catch the int16 path itself, not just the falsifier
        // below: at this shape fp32 lands ~5e-4 and int16 ~9.4e-3, so a bound
        // between them makes the end-to-end assertion the real guard. Scaled by
        // sqrt(k) because the reduction's own f32 error grows that way, so a
        // future K change loosens it honestly rather than silently.
        let bound = 6e-5 * (k as f32).sqrt();
        assert!(
            fp32_error <= bound,
            "accuracy_level=0 must reduce in fp32: max abs error {fp32_error} > {bound}"
        );
        assert!(
            level1_error <= bound,
            "accuracy_level=1 must reduce in fp32: max abs error {level1_error} > {bound}"
        );
        assert!(
            !reduced_precision_activation_allowed(0),
            "accuracy_level 0 (unset) means fp32 compute"
        );
        assert!(
            !reduced_precision_activation_allowed(1),
            "accuracy_level 1 means fp32 compute"
        );
        for level in [2, 3, 4] {
            assert!(
                reduced_precision_activation_allowed(level),
                "accuracy_level {level} permits a reduced-precision compute type"
            );
        }
        // Falsifier: if this ever stops holding, the int16 path is no longer
        // reachable here and the assertions above have gone vacuous.
        if eight_bit_int16_activation() {
            assert!(
                reduced_error >= fp32_error * 20.0,
                "int16 activation path is not being exercised at accuracy_level=2 \
                 (fp32 error {fp32_error}, reduced-precision error {reduced_error}): \
                 the fp32 assertions above are now vacuous"
            );
        }
    }

    /// Qwen3-0.6B CPU int8/block-128 regression: the production `execute` path
    /// for 8-bit weights (symmetric default zero point and explicit asymmetric
    /// uint8 zero points; decode M=1 and prefill M=5) must reconstruct the same
    /// values as a from-scratch dequantize-to-f32 GEMM and, for every output
    /// row, select the same greedy argmax. Divergence here is exactly the
    /// native-vs-ORT token mismatch this path is meant to prevent.
    #[test]
    fn matmulnbits_8bit_block128_execute_matches_dequant_f32_oracle() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (48usize, 256usize, 128usize);
        let weights_nk: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.013).sin() * 1.3 + (i as f32 * 0.0007).cos() * 0.4)
            .collect();
        for &asymmetric in &[false, true] {
            for &m in &[1usize, 5] {
                let activations: Vec<f32> = (0..m * k)
                    .map(|i| (i as f32 * 0.021 + 0.3).cos() * 0.9)
                    .collect();
                let (out, oracle) =
                    run_8bit_execute(n, k, block_size, m, asymmetric, &activations, &weights_nk);
                let oracle_rms = rmse(&oracle, &vec![0.0; oracle.len()]).max(1e-6);
                let rel = rmse(&out, &oracle) / oracle_rms;
                // `model_node` sets no `accuracy_level` attribute, so this runs
                // at the default 0 == CompFp32. The int8-activation SDOT routes
                // are gated off there, so the fp32 tolerance must hold on every
                // architecture -- including aarch64, where
                // `arm64_kai_sdot_direct_enabled` is on by default and used to
                // drag this case to ~1e-3.
                let tolerance = if m == 1 && bits8_int8_activation_active_for_test(0, block_size) {
                    1e-3
                } else {
                    1e-5
                };
                assert!(
                    rel <= tolerance,
                    "asymmetric={asymmetric} m={m}: 8-bit execute relative RMSE {rel} exceeds tolerance {tolerance}",
                );
                for row in 0..m {
                    let winner = |v: &[f32]| {
                        (0..n)
                            .max_by(|&a, &b| v[row * n + a].total_cmp(&v[row * n + b]))
                            .unwrap()
                    };
                    assert_eq!(
                        winner(&out),
                        winner(&oracle),
                        "asymmetric={asymmetric} m={m} row={row}: 8-bit execute argmax != f32 oracle",
                    );
                }
            }
        }
    }

    /// Batched prefill regression for the 8-bit `execute` path across realistic
    /// prompt lengths (`m` = 16/32/100). On the MLAS backend these `m > 1` calls
    /// take the cache-tiled `Nk` + `sgemm(trans_b)` prefill route added to close
    /// the ~10x native-vs-ORT prefill gap (previously a strided-`Kn` transpose
    /// dequant plus dense GEMM). It must reconstruct the same values as an
    /// independent dequantize-to-f32 GEMM to float precision AND pick the same
    /// greedy argmax in every row, so a future kernel/layout regression that
    /// silently changes prefill outputs is caught here.
    #[test]
    fn matmulnbits_8bit_prefill_batched_matches_dequant_f32_oracle() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (96usize, 384usize, 128usize);
        let weights_nk: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.011).sin() * 1.1 + (i as f32 * 0.0005).cos() * 0.5)
            .collect();
        for &asymmetric in &[false, true] {
            for &m in &[16usize, 32, 100] {
                let activations: Vec<f32> = (0..m * k)
                    .map(|i| (i as f32 * 0.017 + 0.2).cos() * 0.8)
                    .collect();
                let (out, oracle) =
                    run_8bit_execute(n, k, block_size, m, asymmetric, &activations, &weights_nk);
                let oracle_rms = rmse(&oracle, &vec![0.0; oracle.len()]).max(1e-6);
                let rel = rmse(&out, &oracle) / oracle_rms;
                assert!(
                    rel <= 1e-5,
                    "asymmetric={asymmetric} m={m}: 8-bit prefill relative RMSE {rel} exceeds 1e-5",
                );
                for row in 0..m {
                    let winner = |v: &[f32]| {
                        (0..n)
                            .max_by(|&a, &b| v[row * n + a].total_cmp(&v[row * n + b]))
                            .unwrap()
                    };
                    assert_eq!(
                        winner(&out),
                        winner(&oracle),
                        "asymmetric={asymmetric} m={m} row={row}: 8-bit prefill argmax != f32 oracle",
                    );
                }
            }
        }
    }

    /// The 8-bit block-128 `execute` path must pick the SAME greedy winner as the
    /// dequantized-f32 oracle even when two output columns are a genuine near tie
    /// -- the regime where a token id actually flips. A deterministic seed sweep
    /// finds real near-ties (relative margin in a narrow window) and asserts the
    /// production kernel never reverses the oracle's argmax.
    #[test]
    fn matmulnbits_8bit_block128_argmax_matches_dequant_f32_oracle_at_near_tie() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (2usize, 128usize, 128usize);
        let mut checked = 0usize;
        for seed in 1..=400u32 {
            let s = seed as f32;
            let activations: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.017 + s * 0.013).sin() * 0.8)
                .collect();
            let weights_nk: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 * 0.011 + s * 0.019).cos())
                .collect();
            let (out, oracle) =
                run_8bit_execute(n, k, block_size, 1, false, &activations, &weights_nk);
            let oracle_rms = rmse(&oracle, &vec![0.0; n]).max(1e-6);
            let margin_rel = (oracle[1] - oracle[0]).abs() / oracle_rms;
            if !(0.001..=0.02).contains(&margin_rel) {
                continue;
            }
            checked += 1;
            assert_eq!(
                usize::from(out[1] > out[0]),
                usize::from(oracle[1] > oracle[0]),
                "seed {seed}: 8-bit execute argmax != f32 oracle (margin_rel {margin_rel}, out {out:?}, oracle {oracle:?})",
            );
        }
        assert!(
            checked >= 5,
            "deterministic search must exercise several 8-bit near-tie cases (got {checked})",
        );
    }

    /// `dot_u8_f32` (the vectorized `u8`-weight x `f32`-activation multiply-add
    /// that backs the 8-bit decode GEMV) must equal a plain serial `f32`
    /// reduction, including a non-multiple-of-16 tail.
    #[test]
    fn dot_u8_f32_matches_serial_reference() {
        for len in [0usize, 1, 7, 16, 17, 31, 128, 129] {
            let weight: Vec<u8> = (0..len).map(|i| ((i * 37 + 5) % 256) as u8).collect();
            let activation: Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.031 - 0.4).sin() * 1.7)
                .collect();
            let expected: f32 = weight
                .iter()
                .zip(&activation)
                .map(|(&w, &a)| w as f32 * a)
                .sum();
            let got = dot_u8_f32(&weight, &activation);
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "len={len}: dot_u8_f32={got} != serial reference {expected}",
            );
        }
    }

    /// The 8-bit decode GEMV ([`gemv_nk_u8`]) must reconstruct the same result as
    /// a from-scratch dequantize-to-f32 GEMV for both symmetric and asymmetric
    /// zero points and for a partial trailing K block. This pins the on-the-fly
    /// u8 dequant path independently of the `execute` dispatch harness.
    #[test]
    fn gemv_nk_u8_matches_dequant_f32_reference() {
        let (n, k, block_size) = (40usize, 200usize, 128usize);
        let k_blocks = k.div_ceil(block_size);
        let weights_nk: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.017).sin() * 1.1 + (i as f32 * 0.0009).cos() * 0.5)
            .collect();
        let activation: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.023 + 0.2).cos() * 0.8)
            .collect();
        for &asymmetric in &[false, true] {
            let (packed, scales, zps, dequantized) =
                quantize_8bit(&weights_nk, n, k, block_size, asymmetric);
            // Prepack into the dense [N, K] u8 layout gemv_nk_u8 consumes.
            let mut values = vec![0u8; n * k];
            let mut scaled_zero_points = vec![0.0f32; n * k_blocks];
            for output in 0..n {
                for block in 0..k_blocks {
                    let start = block * block_size;
                    let valid = k.saturating_sub(start).min(block_size);
                    let zp = zps.as_ref().map_or(128u8, |z| z[output * k_blocks + block]);
                    scaled_zero_points[output * k_blocks + block] =
                        scales[output * k_blocks + block] * zp as f32;
                    for offset in 0..valid {
                        values[output * k + start + offset] =
                            packed[(output * k_blocks + block) * block_size + offset];
                    }
                }
            }
            let mut out = vec![0.0f32; n];
            gemv_nk_u8(
                &activation,
                &values,
                &scales,
                &scaled_zero_points,
                &mut out,
                k,
                n,
                block_size,
            );
            let oracle = reference(&activation, &dequantized, 1, k, n);
            let oracle_rms = rmse(&oracle, &vec![0.0; n]).max(1e-6);
            let rel = rmse(&out, &oracle) / oracle_rms;
            assert!(
                rel <= 1e-5,
                "asymmetric={asymmetric}: gemv_nk_u8 relative RMSE {rel} exceeds 1e-5",
            );
        }
    }

    /// The grouped `block_dot_u8_i16` and, on x86_64, its scalar SIMD-tail helper
    /// must agree with an independent reference, including a non-multiple-of-16 tail.
    #[test]
    fn dot_u8_i16_matches_serial_reference() {
        for len in [0usize, 1, 7, 15, 16, 17, 31, 128, 130] {
            let weight: Vec<u8> = (0..len).map(|i| ((i * 53 + 11) % 256) as u8).collect();
            let activation: Vec<i16> = (0..len)
                .map(|i| (((i * 1103) % 65535) as i32 - 32767) as i16)
                .collect();
            let expected: i32 = weight
                .iter()
                .zip(&activation)
                .map(|(&w, &a)| w as i32 * a as i32)
                .sum();
            #[cfg(target_arch = "x86_64")]
            assert_eq!(
                dot_u8_i16_scalar(&weight, &activation),
                expected,
                "len={len}: dot_u8_i16_scalar mismatch",
            );
            // Single group covering the whole slice, scale 1 -> block_dot == dot.
            let got = block_dot_u8_i16(&weight, &activation, &[1.0], len.max(1));
            assert!(
                (got - expected as f32).abs() <= 1.0 + expected.unsigned_abs() as f32 * 1e-6,
                "len={len}: block_dot_u8_i16={got} != reference {expected}",
            );
        }
    }

    /// The grouped block dot must apply a distinct per-group activation scale and
    /// still reduce once, matching a scalar per-group reference.
    #[test]
    fn block_dot_u8_i16_applies_per_group_scales() {
        let group = 16usize;
        let len = 3 * group + 5; // three full groups + a partial tail group
        let weight: Vec<u8> = (0..len).map(|i| ((i * 17 + 3) % 256) as u8).collect();
        let activation: Vec<i16> = (0..len)
            .map(|i| ((i as i32 * 211) % 4001 - 2000) as i16)
            .collect();
        let n_groups = len.div_ceil(group);
        let group_scales: Vec<f32> = (0..n_groups).map(|g| 0.01 * (g as f32 + 1.0)).collect();
        let expected: f32 = (0..n_groups)
            .map(|g| {
                let start = g * group;
                let end = (start + group).min(len);
                let dot: i32 = (start..end)
                    .map(|i| weight[i] as i32 * activation[i] as i32)
                    .sum();
                group_scales[g] * dot as f32
            })
            .sum();
        let got = block_dot_u8_i16(&weight, &activation, &group_scales, group);
        assert!(
            (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
            "block_dot_u8_i16={got} != per-group reference {expected}",
        );
    }

    /// The int16-activation 8-bit GEMV ([`gemv_nk_u8_i16`]) must reconstruct the
    /// dequantize-to-f32 result for symmetric and asymmetric zero points and a
    /// partial trailing K block. int16 activations keep the product accurate to
    /// well under a greedy token's margin, so a tight relative RMSE bound holds.
    #[test]
    fn gemv_nk_u8_i16_matches_dequant_f32_reference() {
        let (n, k, block_size) = (40usize, 200usize, 128usize);
        let k_blocks = k.div_ceil(block_size);
        let weights_nk: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.017).sin() * 1.1 + (i as f32 * 0.0009).cos() * 0.5)
            .collect();
        let activation: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.023 + 0.2).cos() * 0.8)
            .collect();
        for &asymmetric in &[false, true] {
            let (packed, scales, zps, dequantized) =
                quantize_8bit(&weights_nk, n, k, block_size, asymmetric);
            let mut values = vec![0u8; n * k];
            let mut scaled_zero_points = vec![0.0f32; n * k_blocks];
            for output in 0..n {
                for block in 0..k_blocks {
                    let start = block * block_size;
                    let valid = k.saturating_sub(start).min(block_size);
                    let zp = zps.as_ref().map_or(128u8, |z| z[output * k_blocks + block]);
                    scaled_zero_points[output * k_blocks + block] =
                        scales[output * k_blocks + block] * zp as f32;
                    for offset in 0..valid {
                        values[output * k + start + offset] =
                            packed[(output * k_blocks + block) * block_size + offset];
                    }
                }
            }
            let mut out = vec![0.0f32; n];
            gemv_nk_u8_i16(
                &activation,
                &values,
                &scales,
                &scaled_zero_points,
                &mut out,
                k,
                n,
                block_size,
            );
            let oracle = reference(&activation, &dequantized, 1, k, n);
            let oracle_rms = rmse(&oracle, &vec![0.0; n]).max(1e-6);
            let rel = rmse(&out, &oracle) / oracle_rms;
            assert!(
                rel <= 1e-3,
                "asymmetric={asymmetric}: gemv_nk_u8_i16 relative RMSE {rel} exceeds 1e-3",
            );
        }
    }

    /// Regression for the qwen3 "massive activation channel" failure mode: a
    /// near-tie between two output rows decided by many small activation
    /// channels, alongside one huge channel. int8-activation quantization
    /// (ORT's `accuracy_level=4`) crushes the small channels to zero and FLIPS
    /// the argmax; the int16-activation path must keep the fp32 winner.
    #[test]
    fn gemv_nk_u8_i16_preserves_argmax_on_massive_activation_channel() {
        let (n, k, block_size) = (2usize, 128usize, 128usize);
        // Symmetric weights, scale 1, zero-point 128 -> effective w = value - 128.
        // Row0: massive-channel weight 0, small channels +4. Row1: massive 1, 0.
        let mut values = vec![128u8; n * k];
        for value in values.iter_mut().take(k).skip(1) {
            *value = 132; // row0 small channels -> +4
        }
        values[k] = 129; // row1 massive channel -> +1
        let scales = vec![1.0f32; n]; // one block per row
        let scaled_zero_points = vec![128.0f32; n]; // scale * zp
        // Massive channel 0 = 300, all others = 1.
        let mut activation = vec![1.0f32; k];
        activation[0] = 300.0;

        // fp32 oracle from the same effective (dequantized) weights.
        let dequantized: Vec<f32> = values.iter().map(|&v| v as f32 - 128.0).collect();
        let oracle = reference(&activation, &dequantized, 1, k, n);
        let oracle_argmax = argmax(&oracle);

        // int8-activation simulation (per-block symmetric) MUST flip the argmax,
        // proving this case actually exercises the real failure mode.
        let amax = activation.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let i8_scale = amax / 127.0;
        let a_int8: Vec<f32> = activation
            .iter()
            .map(|&v| (v / i8_scale).round().clamp(-127.0, 127.0) * i8_scale)
            .collect();
        let int8_out = reference(&a_int8, &dequantized, 1, k, n);
        assert_ne!(
            argmax(&int8_out),
            oracle_argmax,
            "test is vacuous: int8-activation did not flip the argmax",
        );

        // int16-activation path must match the fp32 oracle argmax.
        let mut out = vec![0.0f32; n];
        gemv_nk_u8_i16(
            &activation,
            &values,
            &scales,
            &scaled_zero_points,
            &mut out,
            k,
            n,
            block_size,
        );
        assert_eq!(
            argmax(&out),
            oracle_argmax,
            "int16-activation flipped the argmax (oracle={oracle:?}, got={out:?})",
        );
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Microbench (ignored): median-of-3 wall time for the int4 and int16
    /// decode dots, 256-bit vs 512-bit, on this host. Run with
    /// `cargo test -p onnx-runtime-ep-cpu --features mlas --release -- --ignored --nocapture avx512_microbench`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn avx512_microbench() {
        use std::time::Instant;
        if !have_avx512bw() {
            eprintln!("avx512 not available; skipping");
            return;
        }
        let median3 = |mut f: Box<dyn FnMut() -> u64>| -> u64 {
            let mut runs = [f(), f(), f()];
            runs.sort_unstable();
            runs[1]
        };

        // int4: 4096 blocks (K=131072), many rows worth of work.
        let blocks = 4096usize;
        let activation: Vec<i8> = (0..blocks * 32)
            .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
            .collect();
        let packed: Vec<u8> = (0..blocks * 16)
            .map(|i| ((i * 53 + 7) % 256) as u8)
            .collect();
        // SIMD int4 kernels consume the deinterleaved activation layout.
        let activation_deint = deinterleave_activation_int4(&activation);
        let act_sum8 = activation_block_sums8(&activation_deint, blocks);
        let scales: Vec<f32> = (0..blocks).map(|i| ((i % 17) + 1) as f32 / 100.0).collect();
        let ascales: Vec<f32> = (0..blocks).map(|i| ((i % 11) + 1) as f32 / 50.0).collect();
        let iters = 2000u32;
        let int4_256 = median3(Box::new(|| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..iters {
                // SAFETY: avxvnni present (implied by avx512vnni box) or 256 path.
                acc +=
                    unsafe { int4_dot_row_avxvnni(&activation_deint, &packed, &scales, &ascales) };
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as u64
        }));
        let int4_512 = median3(Box::new(|| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..iters {
                // SAFETY: avx512 features confirmed above.
                acc += unsafe {
                    int4_dot_row_avx512vnni(
                        &activation_deint,
                        &packed,
                        &scales,
                        &ascales,
                        &act_sum8,
                    )
                };
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as u64
        }));

        // int16: one big block, group=32 (activation_quant_group), K=65536.
        let k16 = 65536usize;
        let group = 32usize;
        let w16: Vec<u8> = (0..k16).map(|i| ((i * 53 + 11) % 256) as u8).collect();
        let a16: Vec<i16> = (0..k16)
            .map(|i| (((i * 1103) % 65535) as i32 - 32767) as i16)
            .collect();
        let gs: Vec<f32> = (0..k16.div_ceil(group))
            .map(|g| 0.001 * (g % 7 + 1) as f32)
            .collect();
        let iters16 = 2000u32;
        let int16_256 = median3(Box::new(|| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..iters16 {
                // SAFETY: avx2 present.
                acc += unsafe { block_dot_u8_i16_avx2(&w16, &a16, &gs, group) };
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as u64
        }));
        let int16_512 = median3(Box::new(|| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..iters16 {
                // SAFETY: avx512bw confirmed above.
                acc += unsafe { block_dot_u8_i16_avx512bw(&w16, &a16, &gs, group) };
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as u64
        }));

        eprintln!(
            "MICROBENCH (median-of-3, {iters} iters):\n\
             int4  256-bit avxvnni : {:>10.3} ms\n\
             int4  512-bit avx512  : {:>10.3} ms  ({:.2}x)\n\
             int16 256-bit avx2    : {:>10.3} ms\n\
             int16 512-bit avx512bw: {:>10.3} ms  ({:.2}x)",
            int4_256 as f64 / 1e6,
            int4_512 as f64 / 1e6,
            int4_256 as f64 / int4_512 as f64,
            int16_256 as f64 / 1e6,
            int16_512 as f64 / 1e6,
            int16_256 as f64 / int16_512 as f64,
        );
    }

    /// Reference copy of the PREVIOUS 512-bit int4 unpack (natural-order
    /// activation, per-block `unpacklo/unpackhi` deinterleave + `inserti64x4`),
    /// kept only in the test module to time the before/after delta of the
    /// deinterleaved-activation unpack. Numerically identical to the current
    /// kernel; used solely as the "before" baseline in
    /// `int4_unpack_before_after_bench`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni,avx512vl")]
    unsafe fn int4_dot_row_avx512vnni_old(
        activation: &[i8],
        packed_weight: &[u8],
        scales: &[f32],
        activation_scales: &[f32],
    ) -> f32 {
        use std::arch::x86_64::*;
        let low_mask = _mm_set1_epi8(0x0f);
        let block_weight = |block: usize| -> __m256i {
            // SAFETY: each block owns 16 packed bytes.
            let packed = unsafe { _mm_loadu_si128(packed_weight.as_ptr().add(block * 16).cast()) };
            let low = _mm_and_si128(packed, low_mask);
            let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
            _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high))
        };
        let block_count = scales.len();
        let ones = _mm512_set1_epi8(1);
        let ones256 = _mm256_set1_epi8(1);
        let mut accumulator = _mm512_setzero_ps();
        for pair in 0..block_count / 2 {
            let b0 = pair * 2;
            let b1 = b0 + 1;
            let weight = _mm512_inserti64x4(
                _mm512_castsi256_si512(block_weight(b0)),
                block_weight(b1),
                1,
            );
            // SAFETY: two contiguous blocks own 64 activation bytes.
            let act = unsafe { _mm512_loadu_si512(activation.as_ptr().add(b0 * 32).cast()) };
            let wdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), weight, act);
            let asum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, act);
            let dot = _mm512_sub_epi32(wdot, _mm512_slli_epi32(asum, 3));
            let s0 = scales[b0] * activation_scales[b0];
            let s1 = scales[b1] * activation_scales[b1];
            let scale_vec = _mm512_set_ps(
                s1, s1, s1, s1, s1, s1, s1, s1, s0, s0, s0, s0, s0, s0, s0, s0,
            );
            accumulator = _mm512_add_ps(
                accumulator,
                _mm512_mul_ps(_mm512_cvtepi32_ps(dot), scale_vec),
            );
        }
        let mut value = _mm512_reduce_add_ps(accumulator);
        if block_count % 2 == 1 {
            let block = block_count - 1;
            let weight = block_weight(block);
            // SAFETY: the final block owns 32 activation bytes.
            let act = unsafe { _mm256_loadu_si256(activation.as_ptr().add(block * 32).cast()) };
            let wdot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), weight, act);
            let asum = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones256, act);
            let dot = _mm256_sub_epi32(wdot, _mm256_slli_epi32(asum, 3));
            let block_scale = scales[block] * activation_scales[block];
            let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(block_scale));
            value += horizontal_sum_f32_256(scaled);
        }
        value
    }

    /// Before/after timing of the int4 512-bit unpack over a realistic decode
    /// GEMV (K=2048, N=2048, block_size=32, m=1). "Before" is the previous
    /// natural-order `unpacklo/unpackhi + inserti64x4` unpack; "after" is the
    /// current deinterleaved-activation + `permutex2var` unpack (the activation
    /// deinterleave is done once, amortized over all N rows). Prints median of 5
    /// runs. Run with `--ignored --nocapture`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn int4_unpack_before_after_bench() {
        use std::time::Instant;
        assert!(
            std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("avx512bw"),
            "bench requires avx512vnni/vl/bw",
        );
        let k = 2048usize;
        let n = 2048usize;
        let blocks = k / 32;
        // One shared activation row (natural order) + its deinterleaved form.
        let activation: Vec<i8> = (0..k)
            .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
            .collect();
        let activation_deint = deinterleave_activation_int4(&activation);
        let ascales: Vec<f32> = (0..blocks).map(|i| ((i % 11) + 1) as f32 / 50.0).collect();
        // N independent weight rows.
        let packed: Vec<u8> = (0..n * blocks * 16)
            .map(|i| ((i * 53 + 7) % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..n * blocks)
            .map(|i| ((i % 17) + 1) as f32 / 100.0)
            .collect();

        let reps = 7u32;
        let median = |mut runs: Vec<u64>| -> u64 {
            runs.sort_unstable();
            runs[runs.len() / 2]
        };
        let mut old_runs = Vec::new();
        let mut new_runs = Vec::new();
        // Interleave old/new reps to share any thermal/load drift.
        for _ in 0..reps {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for row in 0..n {
                let ps = &packed[row * blocks * 16..(row + 1) * blocks * 16];
                let ss = &scales[row * blocks..(row + 1) * blocks];
                // SAFETY: features asserted above; slices sized per row.
                acc += unsafe { int4_dot_row_avx512vnni_old(&activation, ps, ss, &ascales) };
            }
            std::hint::black_box(acc);
            old_runs.push(t.elapsed().as_nanos() as u64);

            let t = Instant::now();
            let mut acc = 0.0f32;
            // Amortized: deinterleave once per matmul (as production does).
            let act = deinterleave_activation_int4(&activation);
            let act_sum8 = activation_block_sums8(&act, blocks);
            for row in 0..n {
                let ps = &packed[row * blocks * 16..(row + 1) * blocks * 16];
                let ss = &scales[row * blocks..(row + 1) * blocks];
                // SAFETY: features asserted above; slices sized per row.
                acc += unsafe { int4_dot_row_avx512vnni(&act, ps, ss, &ascales, &act_sum8) };
            }
            std::hint::black_box(acc);
            new_runs.push(t.elapsed().as_nanos() as u64);
        }
        // Correctness spot-check: new (deinterleaved) == old (natural) per row.
        let ps = &packed[0..blocks * 16];
        let ss = &scales[0..blocks];
        let act_sum8 = activation_block_sums8(&activation_deint, blocks);
        // SAFETY: features asserted above.
        let old0 = unsafe { int4_dot_row_avx512vnni_old(&activation, ps, ss, &ascales) };
        let new0 =
            unsafe { int4_dot_row_avx512vnni(&activation_deint, ps, ss, &ascales, &act_sum8) };
        assert!(
            (old0 - new0).abs() <= 1e-4 * old0.abs().max(1.0),
            "old {old0} new {new0}"
        );

        let old_ms = median(old_runs) as f64 / 1e6;
        let new_ms = median(new_runs) as f64 / 1e6;
        eprintln!(
            "INT4 UNPACK BEFORE/AFTER (K={k} N={n} block=32, median of {reps}):\n\
             before (unpacklo/unpackhi + inserti64x4): {old_ms:>8.3} ms\n\
             after  (deinterleave-once + permutex2var): {new_ms:>8.3} ms\n\
             speedup: {:.3}x  ({:+.1}%)",
            old_ms / new_ms,
            (old_ms / new_ms - 1.0) * 100.0,
        );
    }

    /// The 512-bit VNNI int4 block dot must match the scalar reference to a few
    /// ULP for any block count, including an odd trailing block routed through
    /// the 256-bit remainder. (Only the cross-block f32 accumulation order
    /// differs from scalar, exactly as the existing 256-bit kernel does; each
    /// block's integer dot is exact.) Skips (no-op passes) where the feature is
    /// absent so the same test binary is valid on AVX2-only hosts.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_dot_row_avx512vnni_matches_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        // Cover every 4-pair unroll boundary and remainder (pairs % 4 ∈
        // {0,1,2,3}) plus an odd trailing block: block counts 10..=18 exercise
        // the multi-accumulator tail that the [1,2,3,4,5,8,9] set alone misses.
        for blocks in [1usize, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 17, 18] {
            let activation: Vec<i8> = (0..blocks * 32)
                .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
                .collect();
            let packed: Vec<u8> = (0..blocks * 16)
                .map(|i| ((i * 53 + 7) % 256) as u8)
                .collect();
            let scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 13 % 17) + 1) as f32 / 100.0)
                .collect();
            let activation_scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 7 % 11) + 1) as f32 / 50.0)
                .collect();
            let scalar = int4_dot_row_scalar(&activation, &packed, &scales, &activation_scales);
            // The SIMD kernel consumes the deinterleaved activation layout; the
            // scalar oracle stays natural-order.
            let activation_deint = deinterleave_activation_int4(&activation);
            let act_sum8 = activation_block_sums8(&activation_deint, blocks);
            // SAFETY: feature support confirmed above.
            let wide = unsafe {
                int4_dot_row_avx512vnni(
                    &activation_deint,
                    &packed,
                    &scales,
                    &activation_scales,
                    &act_sum8,
                )
            };
            assert!(
                (wide - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
                "blocks={blocks}: avx512 int4 dot {wide} != scalar {scalar}",
            );
        }
    }

    /// Multi-accumulator regression: the 4-way-unrolled AVX-512 int4 GEMV must
    /// pick the SAME winning output row as the scalar reference on a near-tie
    /// across N rows. The four independent f32 accumulators reorder the
    /// cross-block reduction; this guards that the reorder never flips the argmax
    /// (token divergence) even when two rows are within a few ULP, across every
    /// unroll remainder (block counts 6..=18) and an odd trailing block.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_dot_row_avx512vnni_multiaccumulator_preserves_argmax_vs_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        for blocks in [6usize, 7, 10, 11, 12, 13, 16, 17, 18] {
            // Two output rows differing only in the last block's weights, so
            // their dots land in a near-tie and any accumulation-order drift
            // would be maximally able to flip the winner.
            let n = 8usize;
            let activation: Vec<i8> = (0..blocks * 32)
                .map(|i| (((i * 29 + 3) % 255) as i32 - 127) as i8)
                .collect();
            let activation_deint = deinterleave_activation_int4(&activation);
            let act_sum8 = activation_block_sums8(&activation_deint, blocks);
            let activation_scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 5 % 13) + 1) as f32 / 40.0)
                .collect();
            let base_packed: Vec<u8> = (0..blocks * 16)
                .map(|i| ((i * 53 + 7) % 256) as u8)
                .collect();
            let base_scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 13 % 17) + 1) as f32 / 100.0)
                .collect();

            let mut scalar_out = vec![0.0f32; n];
            let mut wide_out = vec![0.0f32; n];
            for row in 0..n {
                // Perturb one nibble per row by a single quantum to create a
                // tight spread of dot products across rows.
                let mut packed = base_packed.clone();
                let idx = row % packed.len();
                packed[idx] = packed[idx].wrapping_add(1);
                scalar_out[row] =
                    int4_dot_row_scalar(&activation, &packed, &base_scales, &activation_scales);
                // SAFETY: features asserted above.
                wide_out[row] = unsafe {
                    int4_dot_row_avx512vnni(
                        &activation_deint,
                        &packed,
                        &base_scales,
                        &activation_scales,
                        &act_sum8,
                    )
                };
            }
            assert_eq!(
                argmax(&scalar_out),
                argmax(&wide_out),
                "blocks={blocks}: multi-accumulator avx512 flipped argmax vs scalar\n\
                 scalar={scalar_out:?}\nwide={wide_out:?}",
            );
            for row in 0..n {
                assert!(
                    (wide_out[row] - scalar_out[row]).abs()
                        <= 1e-4 * scalar_out[row].abs().max(1.0),
                    "blocks={blocks} row={row}: {} != {}",
                    wide_out[row],
                    scalar_out[row],
                );
            }
        }
    }

    /// The 256-bit VNNI int4 block dot (`AvxVnni` path) must match the scalar
    /// reference for any block count, including an odd trailing block. On this
    /// AVX-512 host the selected kernel is `Avx512Vnni`, so this path is not
    /// otherwise exercised; the direct call keeps it non-vacuously validated
    /// wherever `avxvnni` is present.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_dot_row_avxvnni_matches_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avxvnni"))
        {
            return;
        }
        for blocks in [1usize, 2, 3, 4, 5, 8, 9] {
            let activation: Vec<i8> = (0..blocks * 32)
                .map(|i| (((i * 41 + 5) % 255) as i32 - 127) as i8)
                .collect();
            let packed: Vec<u8> = (0..blocks * 16)
                .map(|i| ((i * 47 + 3) % 256) as u8)
                .collect();
            let scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 13 % 17) + 1) as f32 / 100.0)
                .collect();
            let activation_scales: Vec<f32> = (0..blocks)
                .map(|i| ((i * 7 % 11) + 1) as f32 / 50.0)
                .collect();
            let scalar = int4_dot_row_scalar(&activation, &packed, &scales, &activation_scales);
            let activation_deint = deinterleave_activation_int4(&activation);
            // SAFETY: avx2 + avxvnni confirmed above.
            let wide = unsafe {
                int4_dot_row_avxvnni(&activation_deint, &packed, &scales, &activation_scales)
            };
            assert!(
                (wide - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
                "blocks={blocks}: avxvnni int4 dot {wide} != scalar {scalar}",
            );
        }
    }

    /// The deinterleave transform used by the SIMD int4 kernels is a pure
    /// per-block permutation: recombining evens/odds reproduces natural order,
    /// so it cannot change any dot result.
    #[test]
    fn deinterleave_activation_int4_is_a_block_permutation() {
        let activation: Vec<i8> = (0..96i32).map(|i| (i - 48) as i8).collect();
        let deint = deinterleave_activation_int4(&activation);
        assert_eq!(deint.len(), activation.len());
        for block in 0..activation.len() / 32 {
            for i in 0..16 {
                assert_eq!(deint[block * 32 + i], activation[block * 32 + 2 * i]);
                assert_eq!(
                    deint[block * 32 + 16 + i],
                    activation[block * 32 + 2 * i + 1]
                );
            }
        }
    }

    /// The 512-bit VNNI `u8 x i8` dot must equal the scalar reduction exactly
    /// (pure integer) for lengths that exercise the 64-byte body, the 32-byte
    /// 256-bit remainder, and the scalar tail.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_u8_i8_avx512vnni_matches_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        for len in [0usize, 1, 7, 31, 32, 33, 63, 64, 65, 96, 129, 200] {
            let activation: Vec<u8> = (0..len).map(|i| ((i * 29 + 7) % 255) as u8).collect();
            let weight: Vec<i8> = (0..len).map(|i| ((i * 17 % 31) as i8) - 15).collect();
            let scalar = dot_u8_i8_scalar(&activation, &weight);
            // SAFETY: feature support confirmed above.
            let wide = unsafe { dot_u8_i8_avx512vnni(&activation, &weight) };
            assert_eq!(wide, scalar, "len={len}: avx512 u8xi8 dot mismatch");
        }
    }

    /// The AVX2 (non-VNNI) `u8 x i8` dot MUST equal the scalar reduction
    /// bit-exactly. This is the correctness proof that the saturation-safe
    /// widen + `madd_epi16` sequence avoids the `maddubs` i16-saturation error.
    /// Adversarial inputs (0/255 activations, ±128/±127 weights) drive the
    /// two-product partial sums to the values that WOULD saturate a naive
    /// `maddubs`, across lengths exercising the 16-wide body and the scalar tail.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_u8_i8_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Deterministic pseudo-random + adversarial extremes over many blocks.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [
            0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 200,
            255, 256, 257, 512, 1000,
        ] {
            // Case A: worst-case saturation drivers (all-max u8, weights that
            // maximize |a0*b0 + a1*b1| both positive and negative).
            let a_max = vec![255u8; len];
            let w_pos = vec![127i8; len];
            let w_neg = vec![-128i8; len];
            let a_zero = vec![0u8; len];
            for (act, wgt) in [
                (&a_max, &w_pos),
                (&a_max, &w_neg),
                (&a_zero, &w_neg),
                (&a_max, &a_max.iter().map(|_| -1i8).collect::<Vec<_>>()),
            ] {
                let scalar = dot_u8_i8_scalar(act, wgt);
                // SAFETY: avx2 confirmed above.
                let simd = unsafe { dot_u8_i8_avx2(act, wgt) };
                assert_eq!(simd, scalar, "len={len}: adversarial avx2 dot mismatch");
            }
            // Case B: random full-range inputs.
            let activation: Vec<u8> = (0..len).map(|_| (next() & 0xff) as u8).collect();
            let weight: Vec<i8> = (0..len).map(|_| (next() & 0xff) as u8 as i8).collect();
            let scalar = dot_u8_i8_scalar(&activation, &weight);
            // SAFETY: avx2 confirmed above.
            let simd = unsafe { dot_u8_i8_avx2(&activation, &weight) };
            assert_eq!(simd, scalar, "len={len}: random avx2 dot mismatch");
        }
    }

    /// End-to-end: force the `Avx2` kernel through the full int4 accuracy_level=4
    /// CompInt8 decode (`int8_row` via `dot_u8_i8`) and confirm it is token-exact
    /// vs the `Scalar` reference. `selected_dot_kernel()` picks `Avx512Vnni` on
    /// this host, so we force `Avx2` explicitly to cover the path on VNNI hosts
    /// too (it is the only way CI on this box exercises the AVX2 dot).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int8_row_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let (k, n, block_size) = (131usize, 7usize, 32usize);
        let padded_k = k.div_ceil(block_size) * block_size;
        let k_blocks = padded_k / block_size;

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let activation: Vec<u8> = (0..padded_k).map(|_| (next() & 0xff) as u8).collect();
        let activation_scales: Vec<f32> = (0..k_blocks)
            .map(|_| (next() & 0xff) as f32 / 512.0 + 0.01)
            .collect();
        let values: Vec<i8> = (0..n * padded_k)
            .map(|_| (next() & 0xff) as u8 as i8)
            .collect();
        let scales: Vec<f32> = (0..n * k_blocks)
            .map(|_| (next() & 0xff) as f32 / 512.0 + 0.01)
            .collect();
        let block_sums: Vec<i32> = values
            .chunks_exact(block_size)
            .map(|b| b.iter().map(|&w| w as i32).sum())
            .collect();
        let weight = Int8Weight {
            values,
            scales,
            block_sums,
        };

        let mut scalar_out = vec![0.0f32; n];
        let mut avx2_out = vec![0.0f32; n];
        int8_row(
            &activation,
            &activation_scales,
            &weight,
            &mut scalar_out,
            k_blocks,
            padded_k,
            block_size,
            DotKernel::Scalar,
            false,
        );
        int8_row(
            &activation,
            &activation_scales,
            &weight,
            &mut avx2_out,
            k_blocks,
            padded_k,
            block_size,
            DotKernel::Avx2,
            false,
        );
        // dot_u8_i8 is bit-exact across kernels, so the accumulated f32 outputs
        // must be identical (not merely close) — token-exact decode.
        assert_eq!(avx2_out, scalar_out, "Avx2 int8_row diverged from Scalar");
    }

    /// The NEON int8 dot (widen-`vmlal` baseline) must equal the scalar
    /// reference bit-for-bit, including on adversarial extremes (all-max u8,
    /// ±128/±127 weights) that would saturate a naive i16 intermediate, across
    /// lengths exercising the 16-wide body and the scalar tail. Runs on aarch64
    /// CI.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_u8_i8_neon_matches_scalar() {
        // Deterministic pseudo-random + adversarial extremes over many blocks.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [
            0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 200,
            255, 256, 257, 512, 1000,
        ] {
            let a_max = vec![255u8; len];
            let w_pos = vec![127i8; len];
            let w_neg = vec![-128i8; len];
            let a_zero = vec![0u8; len];
            for (act, wgt) in [
                (&a_max, &w_pos),
                (&a_max, &w_neg),
                (&a_zero, &w_neg),
                (&a_max, &a_max.iter().map(|_| -1i8).collect::<Vec<_>>()),
            ] {
                let scalar = dot_u8_i8_scalar(act, wgt);
                // SAFETY: NEON is baseline on aarch64.
                let simd = unsafe { dot_u8_i8_neon(act, wgt) };
                assert_eq!(simd, scalar, "len={len}: adversarial neon dot mismatch");
            }
            // Random full-range inputs.
            let activation: Vec<u8> = (0..len).map(|_| (next() & 0xff) as u8).collect();
            let weight: Vec<i8> = (0..len).map(|_| (next() & 0xff) as u8 as i8).collect();
            let scalar = dot_u8_i8_scalar(&activation, &weight);
            // SAFETY: NEON is baseline on aarch64.
            let simd = unsafe { dot_u8_i8_neon(&activation, &weight) };
            assert_eq!(simd, scalar, "len={len}: random neon dot mismatch");
        }
    }

    /// Every `DotKernel` variant, forced through the public dispatcher, on any
    /// host, at every length that separates the kernels' vector bodies from
    /// their remainder and scalar tails.
    ///
    /// Two independent properties, neither of which needs the exotic hardware:
    ///
    /// 1. **No `SIGILL`.** The `#[target_feature]` kernels are reached from safe
    ///    code purely on the strength of a `DotKernel` value. Forcing an
    ///    AVX-512-VNNI request on an AVX2-only host must be answered, not
    ///    faulted. A test that merely called `selected_dot_kernel()` would prove
    ///    nothing here -- it can only ever name a kernel the host already runs.
    /// 2. **Bit-exactness.** On a host that *does* implement the ISA -- the
    ///    AVX-512 CI lane, an ARM lane -- the same assertion stops being a clamp
    ///    check and becomes the only correctness oracle the VNNI kernels have.
    ///    So this test grows teeth exactly where it is otherwise untested.
    ///
    /// Lengths cover 0 (empty), sub-vector, each vector width and each width
    /// plus/minus one, so the AVX-512 path's 64-wide body, its 32-byte VNNI
    /// remainder and its scalar tail are all entered.
    #[test]
    fn every_dot_kernel_is_bit_exact_on_this_host() {
        let kernels = [
            DotKernel::Scalar,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2,
            #[cfg(target_arch = "x86_64")]
            DotKernel::AvxVnni,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512Vnni,
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon,
            #[cfg(target_arch = "aarch64")]
            DotKernel::NeonDot,
        ];
        for len in [
            0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129,
        ] {
            // Extremes, not a mild distribution: 255 x -128 is the product that
            // saturates a 16-bit intermediate, which is how a kernel that used
            // `vpmaddubsw`-style pairwise arithmetic would be caught.
            let activation: Vec<u8> = (0..len)
                .map(|i| {
                    if i % 3 == 0 {
                        255
                    } else {
                        (i * 37 % 256) as u8
                    }
                })
                .collect();
            let weight: Vec<i8> = (0..len)
                .map(|i| {
                    if i % 3 == 0 {
                        -128
                    } else {
                        ((i * 53 % 256) as i32 - 128) as i8
                    }
                })
                .collect();
            let expected = dot_u8_i8_scalar(&activation, &weight);
            for kernel in kernels {
                assert_eq!(
                    dot_u8_i8(&activation, &weight, kernel),
                    expected,
                    "{kernel:?} diverged from scalar at len {len}"
                );
            }
        }
    }

    /// The clamp must be a real filter, not a constant.
    ///
    /// Falsifies both degenerate implementations: "always keep the request"
    /// (which would fault on the unsupported kernels) and "always return the
    /// host kernel" (which would silently discard a legitimately supported
    /// request and make the ladder unreachable).
    #[test]
    fn dot_kernel_clamp_keeps_supported_requests_and_rewrites_only_the_rest() {
        let host = host_dot_kernel();
        assert!(
            host.is_runnable_here(),
            "the host's own selection must be runnable on the host"
        );
        assert_eq!(
            host.clamped_to_host(),
            host,
            "clamping must not rewrite a kernel the host supports"
        );
        assert_eq!(
            DotKernel::Scalar.clamped_to_host(),
            DotKernel::Scalar,
            "scalar has no ISA requirement and must survive clamping everywhere"
        );
        assert_eq!(
            host.clamped_to_host().clamped_to_host(),
            host.clamped_to_host(),
            "clamping must be idempotent"
        );
        // Whatever this host is, an unsupported request lands on something it
        // can run -- that is the entire postcondition.
        for kernel in [
            DotKernel::Scalar,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2,
            #[cfg(target_arch = "x86_64")]
            DotKernel::AvxVnni,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512Vnni,
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon,
            #[cfg(target_arch = "aarch64")]
            DotKernel::NeonDot,
        ] {
            assert!(
                kernel.clamped_to_host().is_runnable_here(),
                "{kernel:?} clamped to a kernel this host cannot run"
            );
        }
    }

    /// The one-hot tags must actually be one-hot and distinct, or the mask
    /// aliases two kernels and `is_runnable_here` answers for the wrong one.
    #[test]
    fn dot_kernel_bits_are_distinct() {
        let kernels = [
            DotKernel::Scalar,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx2,
            #[cfg(target_arch = "x86_64")]
            DotKernel::AvxVnni,
            #[cfg(target_arch = "x86_64")]
            DotKernel::Avx512Vnni,
            #[cfg(target_arch = "aarch64")]
            DotKernel::Neon,
            #[cfg(target_arch = "aarch64")]
            DotKernel::NeonDot,
        ];
        let mut seen = 0u8;
        for kernel in kernels {
            let bit = kernel.bit();
            assert_eq!(bit.count_ones(), 1, "{kernel:?} tag is not one-hot");
            assert_eq!(seen & bit, 0, "{kernel:?} tag collides with an earlier one");
            seen |= bit;
        }
    }

    /// The AVX-512-VNNI gate must demand every feature its `#[target_feature]`
    /// list enables. `vpdpbusd` on `zmm` needs AVX512F+VNNI; the 256-bit
    /// remainder in the same function needs AVX512VL; and MLAS's own SQNBit
    /// dispatch additionally requires BW, so an AVX512F-only part (Xeon Phi
    /// KNL/KNM) must not reach either. Hardware-independent: it reads the gate,
    /// it does not need the CPU.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_vnni_selection_requires_the_full_feature_set() {
        let has_all = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl");
        assert_eq!(
            selected_dot_kernel() == DotKernel::Avx512Vnni,
            has_all,
            "AVX-512-VNNI must be selected exactly when F+BW+VNNI+VL are all present"
        );
        assert_eq!(
            host_dot_kernel_mask() & DotKernel::Avx512Vnni.bit() != 0,
            has_all,
            "the runnable-kernel mask must agree with the selection gate"
        );
        // AVX-VNNI is the VEX encoding and is *independent* of the AVX-512
        // family: Ice Lake / Cooper Lake server parts have AVX512-VNNI without
        // it. The mask must track it separately, never infer it.
        assert_eq!(
            host_dot_kernel_mask() & DotKernel::AvxVnni.bit() != 0,
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avxvnni"),
            "AVX-VNNI availability must be probed, not implied by AVX-512-VNNI"
        );
    }

    /// aarch64 must never fall back to the scalar dot: `selected_dot_kernel`
    /// picks a NEON-family kernel, and the forced `Neon` int8 dot stays
    /// bit-exact vs scalar.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn selected_dot_kernel_is_neon_on_aarch64() {
        assert!(
            matches!(selected_dot_kernel(), DotKernel::Neon | DotKernel::NeonDot),
            "aarch64 must select a NEON-family dot, never Scalar"
        );
        let activation: Vec<u8> = (0..128).map(|i| ((i * 29 + 7) % 255) as u8).collect();
        let weight: Vec<i8> = (0..128).map(|i| ((i * 17 % 31) as i8) - 15).collect();
        let scalar = dot_u8_i8(&activation, &weight, DotKernel::Scalar);
        assert_eq!(dot_u8_i8(&activation, &weight, DotKernel::Neon), scalar);
    }

    /// End-to-end: force the `Neon` kernel through the full int4 accuracy_level=4
    /// CompInt8 decode (`int8_row` via `dot_u8_i8`) and confirm it is token-exact
    /// vs the `Scalar` reference. Runs on aarch64 CI.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn int8_row_neon_matches_scalar() {
        let (k, n, block_size) = (131usize, 7usize, 32usize);
        let padded_k = k.div_ceil(block_size) * block_size;
        let k_blocks = padded_k / block_size;

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let activation: Vec<u8> = (0..padded_k).map(|_| (next() & 0xff) as u8).collect();
        let activation_scales: Vec<f32> = (0..k_blocks)
            .map(|_| (next() & 0xff) as f32 / 512.0 + 0.01)
            .collect();
        let values: Vec<i8> = (0..n * padded_k)
            .map(|_| (next() & 0xff) as u8 as i8)
            .collect();
        let scales: Vec<f32> = (0..n * k_blocks)
            .map(|_| (next() & 0xff) as f32 / 512.0 + 0.01)
            .collect();
        let block_sums: Vec<i32> = values
            .chunks_exact(block_size)
            .map(|b| b.iter().map(|&w| w as i32).sum())
            .collect();
        let weight = Int8Weight {
            values,
            scales,
            block_sums,
        };

        let mut scalar_out = vec![0.0f32; n];
        let mut neon_out = vec![0.0f32; n];
        int8_row(
            &activation,
            &activation_scales,
            &weight,
            &mut scalar_out,
            k_blocks,
            padded_k,
            block_size,
            DotKernel::Scalar,
            false,
        );
        int8_row(
            &activation,
            &activation_scales,
            &weight,
            &mut neon_out,
            k_blocks,
            padded_k,
            block_size,
            DotKernel::Neon,
            false,
        );
        // dot_u8_i8 is bit-exact across kernels, so the accumulated f32 outputs
        // must be identical (not merely close) — token-exact decode.
        assert_eq!(neon_out, scalar_out, "Neon int8_row diverged from Scalar");
    }

    /// The 512-bit AVX-512BW grouped `u8 x i16` block dot must agree with an
    /// independent serial reference (and the scalar/AVX2 paths) within the same
    /// tight tolerance as the AVX2 path, across group and tail sizes that
    /// exercise the 32-wide body, the 16-wide remainder, and the scalar tail.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_u8_i16_avx512_matches_serial_reference() {
        if !have_avx512bw() {
            return;
        }
        for &(group, len) in &[
            (32usize, 32usize),
            (32, 33),
            (32, 48),
            (32, 96),
            (32, 130),
            (16, 48),
            (64, 200),
            (128, 128),
        ] {
            let weight: Vec<u8> = (0..len).map(|i| ((i * 53 + 11) % 256) as u8).collect();
            let activation: Vec<i16> = (0..len)
                .map(|i| (((i * 1103) % 65535) as i32 - 32767) as i16)
                .collect();
            let n_groups = len.div_ceil(group);
            let group_scales: Vec<f32> = (0..n_groups).map(|g| 0.01 * (g as f32 + 1.0)).collect();
            let expected: f32 = (0..n_groups)
                .map(|g| {
                    let start = g * group;
                    let end = (start + group).min(len);
                    let dot: i32 = (start..end)
                        .map(|i| weight[i] as i32 * activation[i] as i32)
                        .sum();
                    group_scales[g] * dot as f32
                })
                .sum();
            // SAFETY: have_avx512bw confirmed support above.
            let got =
                unsafe { block_dot_u8_i16_avx512bw(&weight, &activation, &group_scales, group) };
            assert!(
                (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "group={group} len={len}: avx512bw block dot {got} != reference {expected}",
            );
        }
    }

    /// 512-bit analog of the massive-activation-channel argmax regression: on a
    /// box with AVX-512BW the int16 GEMV routes through `block_dot_u8_i16_avx512bw`.
    /// int8-activation quantization FLIPS the argmax; the 512-bit int16 path must
    /// keep the fp32-reference winner.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gemv_nk_u8_i16_avx512_preserves_argmax_on_massive_activation_channel() {
        if !have_avx512bw() {
            return;
        }
        let (n, k, block_size) = (2usize, 128usize, 128usize);
        let mut values = vec![128u8; n * k];
        for value in values.iter_mut().take(k).skip(1) {
            *value = 132; // row0 small channels -> +4
        }
        values[k] = 129; // row1 massive channel -> +1
        let scales = vec![1.0f32; n];
        let scaled_zero_points = vec![128.0f32; n];
        let mut activation = vec![1.0f32; k];
        activation[0] = 300.0;

        let dequantized: Vec<f32> = values.iter().map(|&v| v as f32 - 128.0).collect();
        let oracle = reference(&activation, &dequantized, 1, k, n);
        let oracle_argmax = argmax(&oracle);

        // int8-activation simulation MUST flip the argmax (proves the case bites).
        let amax = activation.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let i8_scale = amax / 127.0;
        let a_int8: Vec<f32> = activation
            .iter()
            .map(|&v| (v / i8_scale).round().clamp(-127.0, 127.0) * i8_scale)
            .collect();
        let int8_out = reference(&a_int8, &dequantized, 1, k, n);
        assert_ne!(
            argmax(&int8_out),
            oracle_argmax,
            "test is vacuous: int8-activation did not flip the argmax",
        );

        // 512-bit int16 path (via block_dot_u8_i16_avx512bw) must keep the winner.
        let mut out = vec![0.0f32; n];
        gemv_nk_u8_i16(
            &activation,
            &values,
            &scales,
            &scaled_zero_points,
            &mut out,
            k,
            n,
            block_size,
        );
        assert_eq!(
            argmax(&out),
            oracle_argmax,
            "avx512bw int16 flipped the argmax (oracle={oracle:?}, got={out:?})",
        );
    }

    #[test]
    fn matmulnbits_direct_int4_parallel_partial_k_matches_serial() {
        let (k, n, block_size) = (77usize, 1025usize, 32usize);
        let blocks = k.div_ceil(block_size);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let packed_weight = PackedInt4Weight {
            values: (0..n * blocks * block_size / 2)
                .map(|i| ((i * 29 + 7) % 256) as u8)
                .collect(),
            scales: (0..n * blocks)
                .map(|i| ((i * 13 % 17) + 1) as f32 / 100.0)
                .collect(),
        };
        let mut serial = vec![0.0; n];
        let mut parallel = vec![0.0; n];
        let dot_kernel = selected_dot_kernel();
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                int4_matmul_m1(
                    &activations,
                    &packed_weight,
                    &mut serial,
                    k,
                    n,
                    block_size,
                    dot_kernel,
                );
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                int4_matmul_m1(
                    &activations,
                    &packed_weight,
                    &mut parallel,
                    k,
                    n,
                    block_size,
                    dot_kernel,
                );
            });
        assert_eq!(parallel, serial);
    }

    #[test]
    fn matmulnbits_parallel_n_partition_matches_serial() {
        let (k, n, block_size) = (1025usize, 1025usize, 32usize);
        let padded_k = k.div_ceil(block_size) * block_size;
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let values: Vec<i8> = (0..n * padded_k)
            .map(|i| ((i * 11 % 16) as i8) - 8)
            .collect();
        let block_sums = values
            .chunks_exact(block_size)
            .map(|block| block.iter().map(|&value| value as i32).sum())
            .collect();
        let weight = Int8Weight {
            values,
            scales: vec![0.01; n * k.div_ceil(block_size)],
            block_sums,
        };
        let mut serial = vec![0.0; n];
        let mut parallel = vec![0.0; n];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                int8_matmul(
                    &activations,
                    &weight,
                    &mut serial,
                    1,
                    k,
                    n,
                    block_size,
                    DotKernel::Scalar,
                );
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                int8_matmul(
                    &activations,
                    &weight,
                    &mut parallel,
                    1,
                    k,
                    n,
                    block_size,
                    DotKernel::Scalar,
                );
            });
        assert_eq!(parallel, serial);
    }

    #[test]
    fn matmulnbits_partition_scales_with_pool_size_and_work() {
        let chunk = |threads, n, k| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| output_chunk_len(n, k))
        };

        assert_eq!(chunk(1, 4864, 896), 4864);
        assert_eq!(chunk(24, 16, 32), 16);
        assert_eq!(chunk(24, 896, 896), 36);
        assert_eq!(chunk(48, 896, 896), 36);
        assert_eq!(chunk(96, 896, 896), 896);
        assert_eq!(chunk(96, 4864, 896), 4864);
        assert_eq!(chunk(96, 151_936, 896), 1583);
    }

    #[test]
    fn decode_thread_count_defaults_invalid_values_and_clamps() {
        assert_eq!(resolve_decode_threads(None, 96), Some(8));
        assert_eq!(resolve_decode_threads(None, 4), Some(3));
        assert_eq!(resolve_decode_threads(None, 8), Some(4));
        assert_eq!(resolve_decode_threads(None, 1), Some(1));
        assert_eq!(resolve_decode_threads(Some(""), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("0"), 8), None);
        assert_eq!(resolve_decode_threads(Some("4"), 96), Some(4));
        assert_eq!(resolve_decode_threads(Some("1000"), 96), Some(96));
        assert_eq!(resolve_decode_threads(Some("abc"), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("-4"), 4), Some(3));
        assert_eq!(resolve_decode_threads(Some("4"), 0), None);
    }

    #[test]
    fn persistent_decode_thread_default_is_half_the_logical_cpus() {
        // The persistent pool scales past the flat pool's 8-worker ceiling: unset
        // -> half the logical CPUs (topology-derived, rule 2), not the flat cap.
        assert_eq!(default_persistent_threads(96), Some(48));
        assert_eq!(default_persistent_threads(8), Some(4));
        assert_eq!(default_persistent_threads(4), Some(2));
        assert_eq!(default_persistent_threads(2), Some(1));
        assert_eq!(default_persistent_threads(1), Some(1));
        assert_eq!(default_persistent_threads(0), None);
        // Distinct from the flat default on a big host (48 vs 8) -- proving the
        // persistent path does not inherit the fork/join-bound cap.
        assert_ne!(default_persistent_threads(96), default_decode_threads(96));
    }

    #[test]
    fn persistent_decode_threads_honor_env_and_opt_out() {
        // Unset -> the persistent default (half cores), not the flat cap.
        assert_eq!(resolve_persistent_decode_threads(None, 96), Some(48));
        assert_eq!(resolve_persistent_decode_threads(Some(""), 96), Some(48));
        // Explicit `0` opts out of the bounded pool (flat legacy path).
        assert_eq!(resolve_persistent_decode_threads(Some("0"), 96), None);
        // An explicit positive count is honored and clamped to the host.
        assert_eq!(resolve_persistent_decode_threads(Some("32"), 96), Some(32));
        assert_eq!(resolve_persistent_decode_threads(Some("1"), 96), Some(1));
        assert_eq!(
            resolve_persistent_decode_threads(Some("1000"), 96),
            Some(96)
        );
        // Unparseable/negative values fall back to the persistent default.
        assert_eq!(resolve_persistent_decode_threads(Some("abc"), 96), Some(48));
        assert_eq!(resolve_persistent_decode_threads(Some("-4"), 8), Some(4));
        assert_eq!(resolve_persistent_decode_threads(Some("8"), 0), None);
    }

    #[test]
    fn rayon_global_threads_bound_only_by_an_explicit_budget() {
        // Unset budget -> None: the default `available_parallelism()` sizing of
        // the global (prefill/MLAS) Rayon pool is left untouched (no regression).
        assert_eq!(resolve_rayon_global_threads(None, None, 96), None);
        assert_eq!(resolve_rayon_global_threads(None, Some(""), 96), None);
        // The programmatic override (`--cpu-cores N`) bounds the pool to N,
        // clamped to the host, and takes precedence over the environment.
        assert_eq!(resolve_rayon_global_threads(Some(8), None, 96), Some(8));
        assert_eq!(
            resolve_rayon_global_threads(Some(8), Some("2"), 96),
            Some(8)
        );
        assert_eq!(resolve_rayon_global_threads(Some(1000), None, 96), Some(96));
        // A positive env value bounds the pool when no override is set.
        assert_eq!(resolve_rayon_global_threads(None, Some("8"), 96), Some(8));
        assert_eq!(resolve_rayon_global_threads(None, Some(" 8 "), 96), Some(8));
        // The `=0` opt-out and unparseable values leave the default sizing.
        assert_eq!(resolve_rayon_global_threads(None, Some("0"), 96), None);
        assert_eq!(resolve_rayon_global_threads(None, Some("abc"), 96), None);
        // A zero override is not an explicit budget (the setter rejects 0).
        assert_eq!(resolve_rayon_global_threads(Some(0), Some("8"), 96), None);
        // A degenerate host reports no parallelism to bound.
        assert_eq!(resolve_rayon_global_threads(Some(8), None, 0), None);
    }

    #[test]
    fn explicit_budget_precedes_env_for_every_decode_pool() {
        assert_eq!(
            resolve_decode_threads_with_override(Some(6), Some("2"), 96),
            Some(6)
        );
        assert_eq!(
            resolve_persistent_decode_threads_with_override(Some(8), Some("32"), 96),
            Some(8)
        );
        assert_eq!(
            resolve_dense_decode_threads_with_override(Some(12), Some("0"), 96),
            Some(12)
        );
        assert_eq!(
            resolve_persistent_decode_threads_with_override(None, None, 96),
            Some(48),
            "the uncapped automatic default must remain unchanged"
        );
    }

    #[test]
    fn dense_decode_thread_default_scales_and_clamps() {
        // The dense-f32 MLAS path scales past the flat 8-cap but plateaus at the
        // memory-bandwidth knee, so the default is `available/4` clamped to
        // `[8, MAX_DENSE_DECODE_THREADS]` (topology-derived, rule 2).
        assert_eq!(default_dense_decode_threads(96), Some(24));
        assert_eq!(default_dense_decode_threads(128), Some(32)); // clamped to the cap
        assert_eq!(default_dense_decode_threads(64), Some(16));
        assert_eq!(default_dense_decode_threads(48), Some(12));
        assert_eq!(default_dense_decode_threads(16), Some(8)); // clamped up to the floor
        assert_eq!(default_dense_decode_threads(8), Some(8)); // floor never exceeds host
        assert_eq!(default_dense_decode_threads(4), Some(4)); // tiny host: use all cores
        assert_eq!(default_dense_decode_threads(1), Some(1));
        assert_eq!(default_dense_decode_threads(0), None);
        // Distinct from both the flat cap (8) and the persistent default (48) on
        // a big host: the dense path has its own bandwidth-tuned sizing.
        assert_ne!(default_dense_decode_threads(96), default_decode_threads(96));
        assert_ne!(
            default_dense_decode_threads(96),
            default_persistent_threads(96)
        );
    }

    #[test]
    fn dense_decode_threads_honor_env_and_opt_out() {
        // Unset -> the dense default (available/4, clamped).
        assert_eq!(resolve_dense_decode_threads(None, 96), Some(24));
        assert_eq!(resolve_dense_decode_threads(Some(""), 96), Some(24));
        // Explicit `0` opts out -> run on the global Rayon pool.
        assert_eq!(resolve_dense_decode_threads(Some("0"), 96), None);
        // An explicit positive count is honored and clamped to the host.
        assert_eq!(resolve_dense_decode_threads(Some("20"), 96), Some(20));
        assert_eq!(resolve_dense_decode_threads(Some("1000"), 96), Some(96));
        assert_eq!(resolve_dense_decode_threads(Some("1"), 96), Some(1));
        // Unparseable/negative values fall back to the dense default.
        assert_eq!(resolve_dense_decode_threads(Some("abc"), 96), Some(24));
        assert_eq!(resolve_dense_decode_threads(Some("8"), 0), None);
    }

    #[test]
    fn dense_decode_pool_scope_runs_and_clears_residency() {
        // The dense scope (model_uses_spmd_pool = false) must run `f` to
        // completion and never leave the residency flag set on the caller
        // thread, regardless of whether the bounded pool was built.
        let sum = with_decode_pool_scope(false, || (0..1_000u64).sum::<u64>());
        assert_eq!(sum, 499_500);
        assert!(
            !IN_DECODE_POOL.with(Cell::get),
            "dense scope must not leak the residency flag to the caller"
        );
        // A dense scope must never route through the SPMD/numa persistent pools
        // (those are for quantized decode), so the SPMD-active probe is false in
        // the caller's frame afterwards.
        assert!(spmd_decode_active().is_none());
    }

    #[test]
    fn decode_thread_pool_supports_global_pool_opt_out() {
        assert!(build_decode_pool(None).unwrap().is_none());
        let pool = build_decode_pool(Some(3)).unwrap().unwrap();
        assert_eq!(pool.install(rayon::current_num_threads), 3);
    }

    #[test]
    fn decode_residency_guard_sets_and_restores_flag() {
        assert!(!IN_DECODE_POOL.with(Cell::get));
        {
            let _outer = DecodeResidencyGuard::enter();
            assert!(IN_DECODE_POOL.with(Cell::get));
            {
                let _inner = DecodeResidencyGuard::enter();
                assert!(IN_DECODE_POOL.with(Cell::get));
            }
            // Nested drop restores the previous (still-resident) state.
            assert!(IN_DECODE_POOL.with(Cell::get));
        }
        assert!(!IN_DECODE_POOL.with(Cell::get));
    }

    #[test]
    fn decode_residency_guard_clears_on_panic() {
        assert!(!IN_DECODE_POOL.with(Cell::get));
        let result = std::panic::catch_unwind(|| {
            let _guard = DecodeResidencyGuard::enter();
            assert!(IN_DECODE_POOL.with(Cell::get));
            panic!("decode forward panicked");
        });
        assert!(result.is_err());
        assert!(
            !IN_DECODE_POOL.with(Cell::get),
            "residency flag must be cleared after a panicking forward unwinds"
        );
    }

    #[test]
    fn with_decode_pool_runs_inline_when_resident() {
        // With the residency flag set, `with_decode_pool` must NOT re-install the
        // pool: it runs `operation` inline on the current thread. Observing the
        // running thread id proves no external-thread-to-pool crossing happened.
        let _guard = DecodeResidencyGuard::enter();
        let caller = std::thread::current().id();
        let ran_on = with_decode_pool(|| std::thread::current().id()).unwrap();
        assert_eq!(
            ran_on, caller,
            "resident with_decode_pool must run inline on the caller thread"
        );
    }

    #[test]
    fn with_decode_pool_scope_marks_residency_when_pool_active() {
        // When a bounded decode pool exists, the scope must set the residency
        // flag on the worker thread that runs the closure, and inner
        // `with_decode_pool` calls must then run inline on that same worker.
        let pool_active = DECODE_POOL
            .get_or_init(|| build_decode_pool(configured_decode_threads()))
            .as_ref()
            .ok()
            .and_then(Option::as_ref)
            .is_some();
        let (flag_inside, inline_same_thread) = with_decode_pool_scope(true, || {
            let worker = std::thread::current().id();
            let inner = with_decode_pool(|| std::thread::current().id()).unwrap();
            (IN_DECODE_POOL.with(Cell::get), inner == worker)
        });
        if pool_active {
            assert!(flag_inside, "scope must set residency flag inside the pool");
            assert!(
                inline_same_thread,
                "inner with_decode_pool must run inline on the scope worker"
            );
        }
        // The calling thread never observes the flag set (it is set on the worker).
        assert!(!IN_DECODE_POOL.with(Cell::get));
    }

    #[test]
    fn matmulnbits_symmetric_block32_matches_independent_dequantization() {
        let _probe = lock_dispatch_probe();
        let (m, k, n, block_size) = (3, 64, 8, 32);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) / 11.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 9.0)
            .collect();
        let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
        let (graph, node) = model_node(
            &[m, k],
            &[n, 2, 16],
            &[n, 2],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[m, k], &a);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_f16_bf16_inputs_match_widened_f32_for_decode_and_prefill() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (64usize, 9usize, 32usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 9.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let bias_values: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) / 17.0).collect();

        for dtype in [DataType::Float16, DataType::BFloat16] {
            for m in [1usize, 3usize] {
                let a_values: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
                    .collect();
                let low_a = match dtype {
                    DataType::Float16 => Owned::f16(&[m, k], &a_values),
                    DataType::BFloat16 => Owned::bf16(&[m, k], &a_values),
                    _ => unreachable!(),
                };
                let low_scales = match dtype {
                    DataType::Float16 => Owned::f16(&[n, 2], &scales),
                    DataType::BFloat16 => Owned::bf16(&[n, 2], &scales),
                    _ => unreachable!(),
                };
                let low_bias = match dtype {
                    DataType::Float16 => Owned::f16(&[n], &bias_values),
                    DataType::BFloat16 => Owned::bf16(&[n], &bias_values),
                    _ => unreachable!(),
                };
                let widened = |owned: &Owned| match dtype {
                    DataType::Float16 => owned.to_f16_as_f32(),
                    DataType::BFloat16 => owned.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                let f32_a = Owned::f32(&[m, k], &widened(&low_a));
                let f32_scales = Owned::f32(&[n, 2], &widened(&low_scales));
                let f32_bias = Owned::f32(&[n], &widened(&low_bias));
                let b = Owned::u8(&[n, 2, 16], &packed);
                let absent_zp = TensorView::absent(DataType::Uint8);
                let absent_gidx = TensorView::absent(DataType::Int32);

                let mut low_y = Owned::zeros(dtype, &[m, n]);
                accuracy4_kernel(k, n, block_size)
                    .execute(
                        &[
                            low_a.view(),
                            b.view(),
                            low_scales.view(),
                            absent_zp,
                            absent_gidx,
                            low_bias.view(),
                        ],
                        &mut [low_y.view_mut()],
                    )
                    .unwrap();

                let mut f32_y = Owned::zeros_f32(&[m, n]);
                accuracy4_kernel(k, n, block_size)
                    .execute(
                        &[
                            f32_a.view(),
                            b.view(),
                            f32_scales.view(),
                            absent_zp,
                            absent_gidx,
                            f32_bias.view(),
                        ],
                        &mut [f32_y.view_mut()],
                    )
                    .unwrap();

                let actual = widened(&low_y);
                let reference = f32_y.to_f32();
                let narrowed_reference: Vec<f32> = reference
                    .iter()
                    .map(|&value| match dtype {
                        DataType::Float16 => half::f16::from_f32(value).to_f32(),
                        DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
                        _ => unreachable!(),
                    })
                    .collect();
                assert_eq!(
                    actual, narrowed_reference,
                    "{dtype:?} M={m} must compute in f32 and narrow only at output"
                );

                let tolerance: f32 = match dtype {
                    DataType::Float16 => 2e-2,
                    DataType::BFloat16 => 1.5e-1,
                    _ => unreachable!(),
                };
                for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                    assert!(
                        (actual - reference).abs() <= tolerance.max(tolerance * reference.abs()),
                        "{dtype:?} M={m} index {index}: actual={actual}, widened f32={reference}"
                    );
                }
            }
        }
    }

    #[test]
    fn matmulnbits_direct_2bit_gemv_matches_dequantized_reference() {
        let _probe = lock_dispatch_probe();
        run_direct_2bit_parity_case(1, 32, 5, 16, false, DataType::Float32);
        run_direct_2bit_parity_case(1, 81, 4, 16, true, DataType::Float16);
    }

    #[test]
    fn matmulnbits_direct_2bit_gemm_matches_dequantized_reference() {
        let _probe = lock_dispatch_probe();
        run_direct_2bit_parity_case(4, 45, 7, 32, true, DataType::Float32);
        run_direct_2bit_parity_case(3, 64, 6, 32, false, DataType::Float16);
    }

    #[test]
    fn matmulnbits_2bit_unpacks_low_bits_first() {
        let _probe = lock_dispatch_probe();
        let k = 32;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 32);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(2));
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![0.0; k];
        activation[..4].copy_from_slice(&[1.0, 10.0, 100.0, 1000.0]);
        let mut packed = vec![0xaa; 8];
        packed[0] = 0b11_10_01_00;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![988.0]); // -2*1 + -1*10 + 0*100 + 1*1000
    }

    #[test]
    fn matmulnbits_asymmetric_block16_batched_non_square() {
        let _probe = lock_dispatch_probe();
        let (m, k, n, block_size) = (6, 48, 5, 16);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 - 5.0) / 8.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 37) as f32 - 9.0) / 10.0)
            .collect();
        let (packed, scales, zero_points, dequantized) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let kernel = test_kernel(k, n, block_size);
        let a = Owned::f32(&[2, 3, k], &a);
        let b = Owned::u8(&[n, 3, 8], &packed);
        let scales = Owned::f32(&[n * 3], &scales);
        let zero_points = Owned::u8(&[n, 2], &zero_points);
        let mut y = Owned::zeros_f32(&[2, 3, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
        assert!(
            kernel.weight_nk.get().is_none(),
            "batched asymmetric INT4 must not expand the weight to f32"
        );
    }

    #[test]
    fn matmulnbits_bf16_asymmetric_decode_keeps_int4_packed() {
        let _probe = lock_dispatch_probe();
        let (m, k, n, block_size) = (1, 64, 7, 32);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 17 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 37) as f32 - 18.0) / 11.0)
            .collect();
        let (packed, scales, zero_points, _) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let rounded_scales: Vec<f32> = scales
            .iter()
            .map(|&value| half::bf16::from_f32(value).to_f32())
            .collect();
        let dequantized = dequantize_reference(
            &packed,
            &rounded_scales,
            Some(&zero_points),
            n,
            k,
            block_size,
        );
        let kernel = test_kernel(k, n, block_size);
        let a = Owned::bf16(&[m, k], &a_values);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::bf16(&[n, 2], &rounded_scales);
        let zero_points = Owned::u8(&[n, 1], &zero_points);
        let mut y = Owned::zeros(DataType::BFloat16, &[m, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert!(
            kernel.weight_nk.get().is_none(),
            "Muse-style decode must not expand the INT4 initializer to f32"
        );
        let rounded_a: Vec<f32> = a_values
            .iter()
            .map(|&value| half::bf16::from_f32(value).to_f32())
            .collect();
        let expected = reference(&rounded_a, &dequantized, m, k, n);
        let actual = y.to_bf16_as_f32();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.02_f32.max(0.02 * expected.abs()),
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn matmulnbits_m1_block32_symmetric_borrows_weight_for_new_activations() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Post-#979: symmetric int4 (constant B) never expands the weight into
        // the resident f32 cache. Whichever low-copy route serves it (borrowed
        // zero-copy, or MLAS SQNBit CompFp32 with its int4-sized packed buffer),
        // the per-activation correctness invariant must hold: a second call with
        // different activations must recompute rather than reuse the first
        // result.
        let (k, n, block_size) = (35, 7, 32);
        let a1_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let a2_values: Vec<f32> = a1_values
            .iter()
            .enumerate()
            .map(|(i, &value)| value * -0.5 + i as f32 / 17.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);

        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let a1 = Owned::f32(&[1, k], &a1_values);
        let mut y1 = Owned::zeros_f32(&[1, n]);
        let probe = Int4Acc0RouteProbe::start();
        kernel
            .execute(&[a1.view(), b.view(), scales.view()], &mut [y1.view_mut()])
            .unwrap();
        assert_close(&y1.to_f32(), &reference(&a1_values, &dequantized, 1, k, n));
        probe.assert_fast_route(&kernel, true);

        let cached_ptr = prepack_cache_ptr(&kernel);
        let a2 = Owned::f32(&[1, k], &a2_values);
        let mut y2 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a2.view(), b.view(), scales.view()], &mut [y2.view_mut()])
            .unwrap();
        assert_eq!(
            prepack_cache_ptr(&kernel),
            cached_ptr,
            "the second activation must reuse the first call's routing decision and cache identity, not build a new one"
        );
        assert!(
            kernel.weight_nk.get().is_none(),
            "symmetric decode must never expand the weight to f32 across activations (#979)"
        );
        assert_close(&y2.to_f32(), &reference(&a2_values, &dequantized, 1, k, n));
        assert_ne!(y1.to_f32(), y2.to_f32());
    }

    #[test]
    fn matmulnbits_borrowed_m1_block128_explicit_zp_partial_block_matches_reference() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (k, n, block_size) = (141, 7, 128);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, zero_points, _) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let dequantized =
            dequantize_reference(&packed, &scales, Some(&zero_points), n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 64], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let zero_points = Owned::u8(&[n, 1], &zero_points);
        let mut y = Owned::zeros_f32(&[1, n]);
        let probe = Int4Acc0RouteProbe::start();
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        probe.assert_fast_route(&kernel, false);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn borrowed_int4_block_dot_x86_matches_scalar_reference() {
        // The x86 SIMD borrowed int4 block dot (#994) must reproduce the scalar
        // `sum(activation[j] * nibble[j])` reference within f32 rounding for the
        // block sizes the borrowed path feeds it (32 and multiples). Single-chunk
        // (32) and multi-chunk (64/128) both exercise the accumulator loop.
        fn scalar_dot(activation: &[f32], packed: &[u8]) -> f32 {
            let mut dot = 0.0f32;
            for (byte_index, &byte) in packed.iter().enumerate() {
                dot += activation[2 * byte_index] * (byte & 0x0f) as f32;
                dot += activation[2 * byte_index + 1] * (byte >> 4) as f32;
            }
            dot
        }
        let kernel = selected_dot_kernel();
        if matches!(kernel, DotKernel::Scalar) {
            return; // Host has no AVX2; the borrowed path uses the scalar loop.
        }
        for &len in &[32usize, 64, 128] {
            let activation: Vec<f32> = (0..len)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0)
                .collect();
            let packed: Vec<u8> = (0..len / 2)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(5))
                .collect();
            let simd = borrowed_int4_block_dot_x86(&activation, &packed, kernel)
                .expect("non-scalar kernel returns Some");
            let scalar = scalar_dot(&activation, &packed);
            let tol = scalar.abs() * 1e-5 + 1e-4;
            assert!(
                (simd - scalar).abs() <= tol,
                "len={len} kernel={kernel:?} simd={simd} scalar={scalar}"
            );
        }
        // Scalar dispatch declines so the caller keeps the scalar reference.
        assert!(
            borrowed_int4_block_dot_x86(&[0.0f32; 32], &[0u8; 16], DotKernel::Scalar).is_none()
        );
    }

    #[test]
    fn matmulnbits_symmetric_m1_borrows_instead_of_building_f32_cache() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Regression for #979: symmetric int4 (no zero_points input) must take
        // the zero-copy borrowed decode path using the implicit midpoint 8, and
        // must NOT fall through to the resident f32 `weight_nk` cache (~8x the
        // file size in RAM). With constant B/scales, the pre-#979 code populated
        // `weight_nk`; the borrowed path returns before any cache is built, so
        // `!prepack_cache_populated` is a *positive* proof the branch changed.
        let (k, n, block_size) = (128, 7, 32);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, zero_points, _) = quantize(&weights, n, k, block_size, false);
        assert!(
            zero_points.is_none(),
            "symmetric quantize must omit the zero_points tensor"
        );
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 4, 16], &packed);
        let scales = Owned::f32(&[n, 4], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        let probe = Int4Acc0RouteProbe::start();
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        // Positive proof the intended fast branch ran (borrowed zero-copy, or
        // MLAS SQNBit CompFp32 when the vendored MLAS has a kernel), plus
        // negative proof no resident f32 `weight_nk` expansion was ever built.
        probe.assert_fast_route(&kernel, true);
    }

    #[test]
    fn matmulnbits_m1_dynamic_b_falls_back_without_populating_prepack_cache() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (35, 5, 32);
        let a_values: Vec<f32> = (0..k).map(|i| ((i * 5 % 29) as f32 - 14.0) / 9.0).collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 10.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, false, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel.weight_nk.get().is_none(),
            "dynamic B must use the fallback rather than populate the prepack cache"
        );
    }

    /// Dynamic (non-constant) `B` must stay on the borrowed zero-copy path even
    /// when the vendored MLAS *does* have a SQNBit kernel for the shape: MLAS
    /// would have to repack the weight on every single call, and there is no
    /// session-lifetime cache to amortize it against. This is the explicit
    /// decline half of the dispatch policy; the win half is covered by
    /// [`Int4Acc0RouteProbe::assert_fast_route`] on constant weights.
    #[test]
    fn matmulnbits_int4_acc0_dynamic_weight_keeps_borrowed_path() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (128, 16, 32);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        // B is *not* a graph constant: `can_prepack` is false.
        kernel.set_constant_inputs(&[false, false, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 4, 16], &packed);
        let scales = Owned::f32(&[n, 4], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        let sym_before = BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed) > sym_before,
            "dynamic-weight int4 accuracy_level=0 must keep the borrowed zero-copy path (MLAS would repack every call)"
        );
        assert!(
            !prepack_cache_populated(&kernel),
            "dynamic-weight int4 must not build any per-call weight cache"
        );
    }

    /// `accuracy_level = 0` means fp32 compute, on every architecture.
    ///
    /// Regression guard for the aarch64 dispatch bug: with `dotprod` present,
    /// `borrowed_affine_int4_matmul` diverted `m == 1, block_size == 32` into an
    /// `m1_neon_dot` kernel that quantized the activations to int8 via
    /// `quantize_activation_signed`. That is CompInt8 accuracy delivered where
    /// CompFp32 was requested: it moved results by ~1e-3 relative, far outside
    /// f32 reassociation. It only reached CI because
    /// `DotKernel::arm64_int4_direct_enabled` was hard-wired `true` under
    /// `cfg(test)` while production defaults it *off*, so every test ran an
    /// opt-in, precision-reducing kernel that no shipped binary would pick.
    ///
    /// The fix is structural, not a tolerance bump: the diversion was deleted
    /// (acc0 was its only caller, so it had no semantically valid use), so
    /// `ONNX_GENAI_CPU_ARM64_INT4_DIRECT=1` can no longer reduce acc0 precision.
    /// The 8-bit sibling of this invariant is covered by
    /// [`matmulnbits_8bit_block128_execute_matches_dequant_f32_oracle`], whose
    /// SDOT routes are now gated on `accuracy_level == 4`.
    ///
    /// This test is deliberately architecture-independent: it states the
    /// invariant ("acc0 reconstructs the f32 dequantize-then-GEMM oracle") and
    /// so fails on any host whose acc0 route silently quantizes, rather than
    /// encoding which kernel happens to be selected here.
    #[test]
    fn matmulnbits_acc0_m1_block32_matches_fp32_oracle_on_every_arch() {
        let _probe = lock_dispatch_probe();
        for &(k, n) in &[(32usize, 8usize), (35, 7), (256, 48), (129, 33)] {
            let block_size = 32usize;
            let activations: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.019 + 0.11).cos() * 1.7)
                .collect();
            let weights: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 * 0.0131).sin() * 1.3 + (i as f32 * 0.0007).cos() * 0.4)
                .collect();
            let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
            let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
            let expected = reference(&activations, &dequantized, 1, k, n);

            let mut kernel = test_kernel(k, n, block_size);
            kernel.set_constant_inputs(&[false, true, true]);
            let block_count = k.div_ceil(block_size);
            let b = Owned::u8(&[n, block_count, block_size / 2], &packed);
            let scale_tensor = Owned::f32(&[n, block_count], &scales);
            let a = Owned::f32(&[1, k], &activations);
            let mut y = Owned::zeros_f32(&[1, n]);
            kernel
                .execute(
                    &[a.view(), b.view(), scale_tensor.view()],
                    &mut [y.view_mut()],
                )
                .unwrap();

            // Bound the f32 dot product itself (gamma_k ~ k * EPSILON) instead of
            // a flat epsilon, so the assertion stays valid for any legitimate
            // SIMD reassociation while still rejecting int8 activation
            // quantization, which is ~1e-3 relative -- orders of magnitude larger.
            for (index, (&actual, &want)) in y.to_f32().iter().zip(&expected).enumerate() {
                let magnitude: f32 = (0..k)
                    .map(|j| (activations[j] * dequantized[index * k + j]).abs())
                    .sum();
                let tolerance = (k as f32) * f32::EPSILON * magnitude.max(1.0);
                assert!(
                    (actual - want).abs() <= tolerance,
                    "k={k} n={n} index={index}: acc0 must be fp32-exact, \
                     actual={actual} want={want} tolerance={tolerance}"
                );
            }
        }
    }

    /// The dispatch gates must report the *production* policy under `cfg(test)`.
    ///
    /// Pins the defect class directly: a `#[cfg(test)] { true }` in front of a
    /// `#[cfg(not(test))]` policy body makes the whole suite exercise a
    /// configuration no shipped binary uses, and silently disables any test of
    /// the default. Re-introducing that pattern flips these assertions.
    #[test]
    fn dispatch_gates_report_production_policy_under_cfg_test() {
        #[cfg(target_arch = "aarch64")]
        {
            // Opt-in via ONNX_GENAI_CPU_ARM64_INT4_DIRECT; unset in CI. These
            // kernels quantize activations to int8, so defaulting them on would
            // silently reduce accuracy_level=0 precision.
            if std::env::var_os("ONNX_GENAI_CPU_ARM64_INT4_DIRECT").is_none() {
                assert!(
                    !DotKernel::arm64_int4_direct_enabled(),
                    "the aarch64 N16 SDOT direct kernels are opt-in and must stay off by default under test"
                );
                assert_eq!(
                    DotKernel::arm64_kai_sdot_direct_enabled(),
                    !cfg!(any(target_os = "macos", target_os = "ios")),
                    "the KleidiAI SDOT gate must match its documented per-OS production default"
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // Mirror image: on by default, disabled via the escape hatch.
            if std::env::var_os("ONNX_GENAI_CPU_DISABLE_INT4_SIMD").is_none() {
                assert!(
                    int4_borrowed_simd_enabled(),
                    "the x86 borrowed int4 SIMD path is on by default"
                );
            }
        }
    }

    ///
    /// [`Int4Acc0RouteProbe::assert_fast_route`] asks the production predicate
    /// what should have happened, which catches "the code disagrees with its own
    /// policy" but cannot catch "the policy itself regressed to never choosing
    /// MLAS". This pins the concrete case the fix exists for: on an x86_64 host
    /// with the vendored MLAS, a constant-weight symmetric int4
    /// `accuracy_level = 0` node with the block size every real export uses
    /// *must* end up on MLAS SQNBit. If someone reintroduces the branch order
    /// that made that route dead code, this fails even if the predicate is
    /// changed to agree with it.
    #[cfg(all(feature = "mlas", target_arch = "x86_64"))]
    #[test]
    fn int4_acc0_constant_weight_reaches_mlas_on_x86_64() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (k, n, block_size) = (256, 32, 32);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 8, 16], &packed);
        let scales_tensor = Owned::f32(&[n, 8], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales_tensor.view()],
                &mut [y.view_mut()],
            )
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel
                .mlas_shards
                .get()
                .is_some_and(|shards| shards.is_some())
                || kernel
                    .mlas_packed
                    .get()
                    .is_some_and(|packed| packed.is_some()),
            "int4 accuracy_level=0 with constant block-32 weights must reach MLAS SQNBit on x86_64; \
             the borrowed path's block dot is aarch64-only, so pre-empting MLAS here means a scalar GEMV"
        );
        assert!(
            kernel.weight_nk.get().is_none(),
            "and it must still never expand the weight to f32 (#979)"
        );
    }

    /// The ownership predicate must be a pure function of the node's static
    /// shape/quantization plus `can_prepack`, so `execute`'s branch order and
    /// the route assertions in tests cannot disagree.
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_sqnbit_ownership_requires_constant_weights() {
        let kernel = test_kernel(128, 16, 32);
        assert!(
            !kernel.mlas_sqnbit_owns_fp32_compute(false, false),
            "dynamic weights must never be handed to MLAS SQNBit on the accuracy_level=0 route"
        );
        assert!(
            !kernel.mlas_sqnbit_owns_fp32_compute(false, true),
            "dynamic asymmetric weights must never be handed to MLAS SQNBit either"
        );
        // A block size MLAS has no SQNBit kernel for must be declined even with
        // constant weights, so the borrowed path stays reachable as the fallback.
        let unsupported = test_kernel(128, 16, 8);
        assert!(
            !unsupported.mlas_sqnbit_owns_fp32_compute(true, false),
            "block_size=8 has no MLAS SQNBit kernel and must fall back to the borrowed path"
        );
    }

    /// Sum the heap bytes the MLAS SQNBit shards a kernel packed actually hold
    /// (packed buffer + retained scale/zero-point copies), reading each shard's
    /// real [`mlas_sys::SQNBitPackedB::owned_heap_bytes`]. This is the *actual*
    /// footprint the shared packed buffer retains, read back deterministically
    /// from the kernel (not the process-global counter, so parallel tests cannot
    /// perturb it).
    #[cfg(feature = "mlas")]
    fn mlas_shards_owned_bytes(kernel: &MatMulNBitsKernel) -> u64 {
        kernel
            .mlas_shards
            .get()
            .and_then(Option::as_ref)
            .map(|shards| {
                shards
                    .iter()
                    .flatten()
                    .map(|s| s.prepared.packed.owned_heap_bytes() as u64)
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Identity of the shared `mlas_shards` allocation a kernel holds, so a test
    /// can prove two kernel instances point at the *same* packed buffer (#1056
    /// dedup) rather than two equal-but-distinct copies.
    #[cfg(feature = "mlas")]
    fn mlas_shards_arc_ptr(kernel: &MatMulNBitsKernel) -> Option<*const Vec<Option<MlasShard>>> {
        kernel
            .mlas_shards
            .get()
            .and_then(Option::as_ref)
            .map(Arc::as_ptr)
    }

    /// #1027/#1051/#1056: the accounting the memory plan reads must equal the
    /// MLAS packed bytes the kernels *actually* allocate -- predicted == actual,
    /// so an admission gate is never fed an underestimate.
    ///
    /// This asserts the tie against the real allocated bytes rather than
    /// re-deriving `sqnbit_packed_b_size` (which would only prove the predictor
    /// calls the same function it did before, the tautology that let the earlier
    /// under-report through):
    ///
    ///   1. the per-copy predictor equals the packed + scale/zp bytes one
    ///      executed kernel instance retains;
    ///   2. the prefill (`m > 1`) and decode (`m == 1`) instances -- the two the
    ///      shape-keyed kernel cache compiles for one node from the *same*
    ///      constant weight -- **share one packed allocation** (#1056), so the
    ///      per-node predictor equals a **single** copy, not two;
    ///   3. the graph walker the plan calls
    ///      ([`resident_dequant_f32_cache_bytes`]) equals that same actual
    ///      session footprint.
    ///
    /// It also guards the #979 direction: the total must be non-zero and never
    /// the ~8x f32 expansion.
    #[cfg(feature = "mlas")]
    #[test]
    fn int4_acc0_mlas_packed_accounting_equals_actual_allocated() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (k, n, block_size) = (256usize, 32usize, 32usize);
        let blocks = k.div_ceil(block_size);

        if !test_kernel(k, n, block_size).mlas_sqnbit_owns_fp32_compute(true, false) {
            // This host has no MLAS SQNBit kernel for the shape, so the node
            // stays on the borrowed zero-copy path and holds no side buffer.
            assert_eq!(
                matmul_nbits_resident_side_cache_bytes(4, block_size, 0, n, k, false, false, false),
                0
            );
            return;
        }

        set_mlas_sqnbit_packing_enabled(true);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);

        // The constant weight (`b`) and its scales are built **once** and shared
        // by both runs, exactly as the executor hands the same mmapped
        // initializer to a node's prefill and decode kernel instances. That
        // shared address is what lets the two instances rendezvous on one packed
        // buffer; building a fresh copy per run (distinct addresses) would defeat
        // the dedup the same way distinct weights do.
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales_t = Owned::f32(&[n, blocks], &scales);

        // Execute one kernel instance at row count `m` so it packs (or shares)
        // its MLAS shards exactly as the decode runtime does, then return the
        // kernel so its actual retained bytes can be read back.
        let run = |m: usize| -> MatMulNBitsKernel {
            let a_values: Vec<f32> = (0..m * k)
                .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
                .collect();
            let mut kernel = test_kernel(k, n, block_size);
            kernel.set_constant_inputs(&[false, true, true]);
            let a = Owned::f32(&[m, k], &a_values);
            let mut y = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(&[a.view(), b.view(), scales_t.view()], &mut [y.view_mut()])
                .unwrap();
            kernel
        };

        // Two distinct activation shapes -> two compiled kernel instances, the
        // exact duplication the shape-keyed kernel cache produces in a real
        // autoregressive session (prefill `m > 1`, decode `m == 1`).
        let prefill = run(3);
        let decode = run(1);

        let per_copy_predicted =
            mlas_sqnbit_packed_b_cache_bytes(4, block_size, 0, n, k, false, false, false)
                .expect("MLAS route was asserted available for this shape");
        let prefill_actual = mlas_shards_owned_bytes(&prefill);
        let decode_actual = mlas_shards_owned_bytes(&decode);
        assert!(
            prefill_actual > 0,
            "the prefill instance must have packed MLAS shards"
        );
        assert_eq!(
            prefill_actual, per_copy_predicted,
            "per-copy predictor must equal the packed + scale/zp bytes one instance actually holds"
        );
        assert_eq!(
            decode_actual, per_copy_predicted,
            "the decode instance sees the same per-copy bytes as the prefill instance"
        );

        // #1056: the prefill and decode instances must share **one** packed
        // allocation, not hold two equal-but-distinct copies. Proven by pointer
        // identity of the shared `mlas_shards` `Arc`, so the equality above is
        // one buffer read twice, not two buffers that happen to match.
        assert_eq!(
            mlas_shards_arc_ptr(&prefill),
            mlas_shards_arc_ptr(&decode),
            "prefill and decode instances of one node must share a single packed buffer (#1056)"
        );

        // Per-node accounting must equal the bytes the session actually retains:
        // a single shared copy, because the two instances point at one buffer.
        let session_actual = per_copy_predicted;
        let node_predicted =
            matmul_nbits_resident_side_cache_bytes(4, block_size, 0, n, k, false, false, false);
        assert_eq!(
            node_predicted, session_actual,
            "per-node accounting must equal the one shared packed buffer's actual bytes (#1056)"
        );

        // The graph walker the memory plan actually calls must equal the same
        // actual session footprint, for a graph whose weight is a constant
        // initializer (the only case that caches).
        let (mut graph, node_id) = model_node(
            &[1, k],
            &[n, blocks, block_size / 2],
            &[n, blocks],
            None,
            &[1, n],
            k,
            n,
            block_size,
        );
        let b_value = graph
            .node(node_id)
            .inputs
            .get(1)
            .and_then(|v| *v)
            .expect("MatMulNBits B input");
        graph.set_initializer(
            b_value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![n, blocks, block_size / 2],
                vec![0u8; n * blocks * (block_size / 2)],
            )),
        );
        assert_eq!(
            resident_dequant_f32_cache_bytes(&graph),
            session_actual,
            "the memory plan's graph accounting must equal the actual session footprint"
        );

        // #979 direction: non-zero, and never the ~8x f32 expansion.
        assert!(
            node_predicted > 0,
            "the packed buffer must not be accounted as zero (#1027)"
        );
        assert_ne!(
            node_predicted,
            (n as u64) * (k as u64) * 4,
            "the int4 packed buffer must not be accounted as the ~8x resident f32 expansion (#979)"
        );

        drop(prefill);
        drop(decode);
    }

    /// #1027 item 2: admitting the packed buffer through the memory strategy has
    /// a real decline path. When the plan declines it (over budget) it calls
    /// [`set_mlas_sqnbit_packing_enabled`] with `false`; the node must then keep
    /// the borrowed zero-copy int4 path -- correct output via the borrowed path,
    /// no MLAS packed buffer built. This flips the process-global admission flag,
    /// so it shares [`CACHE_FLAG_TEST_LOCK`] with the other route-asserting tests.
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_sqnbit_packing_decline_keeps_borrowed_path() {
        let _probe = lock_dispatch_probe();
        let _guard = CACHE_FLAG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (k, n, block_size) = (256, 32, 32);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);

        let run = || {
            let mut kernel = test_kernel(k, n, block_size);
            kernel.set_constant_inputs(&[false, true, true]);
            let a = Owned::f32(&[1, k], &a_values);
            let b = Owned::u8(&[n, 8, 16], &packed);
            let scales_t = Owned::f32(&[n, 8], &scales);
            let mut y = Owned::zeros_f32(&[1, n]);
            kernel
                .execute(&[a.view(), b.view(), scales_t.view()], &mut [y.view_mut()])
                .unwrap();
            (kernel, y.to_f32())
        };

        set_mlas_sqnbit_packing_enabled(true);
        let (admitted_kernel, admitted_y) = run();
        let mlas_had_a_kernel = admitted_kernel.mlas_sqnbit_owns_fp32_compute(true, false);

        set_mlas_sqnbit_packing_enabled(false);
        let sym_before = BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed);
        let (declined_kernel, declined_y) = run();
        // Capture the route decision while the admission flag is still false;
        // `mlas_sqnbit_owns_fp32_compute` reads the process-global flag live.
        let declined_owns = declined_kernel.mlas_sqnbit_owns_fp32_compute(true, false);
        set_mlas_sqnbit_packing_enabled(true);

        assert_close(&admitted_y, &reference(&a_values, &dequantized, 1, k, n));
        assert_close(&declined_y, &reference(&a_values, &dequantized, 1, k, n));
        // Admitting takes MLAS SQNBit CompFp32; declining takes the borrowed
        // dequant path. Both are numerically the same GEMV, but MLAS's CompFp32
        // rounding differs from the scalar dequant by a few ULPs, so this is a
        // near-equality (the MLAS-vs-scalar tolerance used elsewhere in this
        // file), not bit-identity. The behavioural no-op we assert on is the
        // *route*: declining must not claim ownership and must not build a
        // packed buffer (checked below), leaving exactly the pre-existing
        // borrowed zero-copy path that other tests already exercise.
        mlas_close(&admitted_y, &declined_y, 2e-3, "admitted-vs-declined route");
        assert!(
            !declined_owns,
            "a declined MLAS route must not claim ownership"
        );
        assert!(
            declined_kernel.mlas_shards.get().is_none()
                && declined_kernel.mlas_packed.get().is_none(),
            "the declined route must not build any MLAS packed buffer"
        );
        assert!(
            BORROWED_INT4_SYMMETRIC_TEST_CALLS.load(Ordering::Relaxed) > sym_before,
            "the declined int4 accuracy_level=0 node must fall back to the borrowed zero-copy path"
        );
        // If MLAS had no kernel for this shape the fallback is trivially the
        // borrowed path in both arms; the correctness check still holds.
        let _ = mlas_had_a_kernel;
    }

    #[test]
    fn matmulnbits_unpacks_low_nibble_before_high_nibble() {
        let _probe = lock_dispatch_probe();
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]); // (1-8)*1 + (14-8)*10
    }

    #[test]
    fn matmulnbits_honors_non_contiguous_group_indices() {
        let _probe = lock_dispatch_probe();
        let k = 32;
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let a_value = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b_value = graph.create_named_value("B", DataType::Uint8, static_shape([1, 2, 8]));
        let scales_value =
            graph.create_named_value("scales", DataType::Float32, static_shape([1, 2]));
        let g_idx_value = graph.create_named_value("g_idx", DataType::Int32, static_shape([k]));
        for value in [a_value, b_value, scales_value, g_idx_value] {
            graph.add_input(value);
        }
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, 1]));
        let mut node = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![
                Some(a_value),
                Some(b_value),
                Some(scales_value),
                None,
                Some(g_idx_value),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(1));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(16));
        let node = graph.insert_node(node);
        graph.add_output(output);

        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![1.0; k];
        activation[16..].fill(2.0);
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 2, 8], &[0x99; 16]);
        let scales = Owned::f32(&[1, 2], &[1.0, 2.0]);
        let groups: Vec<i32> = (0..k).map(|i| if i < 16 { 1 } else { 0 }).collect();
        let groups = Owned::i32(&[k], &groups);
        let absent_zp = TensorView::absent(DataType::Uint8);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), absent_zp, groups.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_eq!(y.to_f32(), vec![64.0]);
    }

    #[test]
    fn matmulnbits_rejects_unsupported_bit_width() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(3));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("bits=3 must be rejected");
        assert!(format!("{error}").contains("supports bits in {2, 4, 8}"));
    }

    #[test]
    fn matmulnbits_factory_accepts_bits8() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 16], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("bits=8 must be accepted");
    }

    #[test]
    fn matmulnbits_defaults_missing_bits_to_int4() {
        let _probe = lock_dispatch_probe();
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let mut graph = graph;
        graph.node_mut(node).attributes.remove("bits");
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("missing bits must default to 4");
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_factory_accepts_mlas_prepacked_weight_layout() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("weight_prepacked".into(), Attribute::Int(1));
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("MLAS prepacked weights must be accepted");
    }

    #[test]
    fn matmulnbits_factory_rejects_unknown_prepacked_weight_layout() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("weight_prepacked".into(), Attribute::Int(2));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("unknown prepacked layouts must be rejected");
        assert!(format!("{error}").contains("must be 0"));
    }

    // Test helper used only by MLAS/aarch64-gated tests; unused in the default
    // (non-mlas) build, so suppress dead_code there rather than duplicating the
    // callers' cfg matrix on the helper.
    #[allow(dead_code)]
    fn mlas_close(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (a - e).abs();
            let rel = diff / e.abs().max(1.0);
            assert!(
                diff <= tol || rel <= tol,
                "{ctx}: index {i} mlas={a} ref={e} diff={diff}"
            );
        }
    }

    #[cfg(feature = "mlas")]
    fn pseudo(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.017 + seed).sin()) * 1.5)
            .collect()
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_mlas_prepacked_matches_standard_layout() {
        let _probe = lock_dispatch_probe();
        let mut tested = 0;
        for &(m, n, k, block_size) in &[(3usize, 32usize, 96usize, 32usize), (5, 48, 192, 64)] {
            let k_blocks = k.div_ceil(block_size);
            let blob_size = block_size / 2;
            let weights_nk = pseudo(n * k, 0.3 + block_size as f32 * 0.01);
            let (packed, scales, _zero_points, _dequantized) =
                quantize(&weights_nk, n, k, block_size, false);
            let Some(mlas_packed) = mlas_sys::SQNBitPackedB::new(
                n,
                k,
                4,
                block_size,
                mlas_sys::SQNBitComputeType::Fp32,
                &packed,
                &scales,
                None,
            ) else {
                eprintln!(
                    "MLAS SQNBit int4 block_size={block_size} unavailable; skipping prepacked parity case"
                );
                continue;
            };
            tested += 1;

            let activations = Owned::f32(&[m, k], &pseudo(m * k, 0.8));
            let standard_b = Owned::u8(&[n, k_blocks, blob_size], &packed);
            let prepacked_bytes = mlas_packed.as_bytes().to_vec();
            let prepacked_b = Owned::u8(&[prepacked_bytes.len()], &prepacked_bytes);
            let scales = Owned::f32(&[n, k_blocks], &scales);
            let mut standard_output = Owned::zeros_f32(&[m, n]);
            let mut prepacked_output = Owned::zeros_f32(&[m, n]);

            let mut standard_kernel = test_kernel(k, n, block_size);
            standard_kernel.set_constant_inputs(&[false, true, true]);
            standard_kernel
                .execute(
                    &[activations.view(), standard_b.view(), scales.view()],
                    &mut [standard_output.view_mut()],
                )
                .unwrap();

            let mut prepacked_kernel = MatMulNBitsKernel {
                weight_prepacked: true,
                ..test_kernel(k, n, block_size)
            };
            prepacked_kernel.set_constant_inputs(&[false, true, true]);
            prepacked_kernel
                .execute(
                    &[activations.view(), prepacked_b.view(), scales.view()],
                    &mut [prepacked_output.view_mut()],
                )
                .unwrap();
            let cached_prepacked = prepacked_kernel
                .mlas_packed
                .get()
                .and_then(Option::as_ref)
                .expect("constant MLAS prepacked weight must be cached")
                as *const _;
            let mut reused_output = Owned::zeros_f32(&[m, n]);
            prepacked_kernel
                .execute(
                    &[activations.view(), prepacked_b.view(), scales.view()],
                    &mut [reused_output.view_mut()],
                )
                .unwrap();
            assert_eq!(
                cached_prepacked,
                prepacked_kernel
                    .mlas_packed
                    .get()
                    .and_then(Option::as_ref)
                    .expect("MLAS prepacked cache must remain populated")
                    as *const _,
                "constant MLAS prepacked weight must be reused"
            );

            let standard_output = standard_output.to_f32();
            let prepacked_output = prepacked_output.to_f32();
            let max_abs_diff = standard_output
                .iter()
                .zip(&prepacked_output)
                .map(|(standard, prepacked)| (standard - prepacked).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "MLAS prepacked parity m={m} n={n} k={k} block_size={block_size}: max_abs_diff={max_abs_diff}"
            );
            assert!(
                max_abs_diff <= 2e-3,
                "m={m} n={n} k={k} block_size={block_size}: max_abs_diff={max_abs_diff}"
            );
        }
        if tested == 0 {
            eprintln!("MLAS SQNBit unavailable on this host; prepacked parity cases skipped");
        }
    }

    /// The MLAS SQNBit path (`build_mlas_packed` + `mlas_sys::sqnbit_gemm`, the
    /// exact code `execute` runs when the backend is MLAS) must match the
    /// existing dequantize-then-GEMM oracle across block sizes, symmetric and
    /// asymmetric zero points, decode (M=1) and prefill (M>1), both compute
    /// types (`accuracy_level` 0 → CompFp32, 4 → CompInt8), and bias.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_mlas_matches_dequant_reference() {
        let (n, k) = (96usize, 256usize);
        for &block_size in &[32usize, 64, 128] {
            let k_blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let weights_nk = pseudo(n * k, 0.3);
            for &asymmetric in &[false, true] {
                let (packed, scales, zps, _dq) =
                    quantize(&weights_nk, n, k, block_size, asymmetric);
                let ref_weights =
                    dequantize_reference(&packed, &scales, zps.as_deref(), n, k, block_size);
                let b = Owned::u8(&[n, k_blocks, blob], &packed);
                let scales_t = Owned::f32(&[n, k_blocks], &scales);
                let zp_owned = zps
                    .as_ref()
                    .map(|z| Owned::u8(&[n, k_blocks.div_ceil(2)], z));

                for &accuracy_level in &[0i64, 4] {
                    let comp = if accuracy_level == 4 {
                        mlas_sys::SQNBitComputeType::Int8
                    } else {
                        mlas_sys::SQNBitComputeType::Fp32
                    };
                    let kernel = MatMulNBitsKernel {
                        accuracy_level,
                        ..test_kernel(k, n, block_size)
                    };
                    let zp_view = zp_owned.as_ref().map(|z| z.view());
                    let Some(packed_weight) = kernel
                        .build_mlas_packed(&b.view(), &scales_t.view(), zp_view.as_ref(), comp)
                        .unwrap()
                    else {
                        eprintln!(
                            "MLAS SQNBit int4 blk={block_size} {comp:?} unavailable; skipping"
                        );
                        continue;
                    };
                    for &m in &[1usize, 5] {
                        let a = pseudo(m * k, 0.8);
                        for bias in [None, Some(pseudo(n, 0.1))] {
                            let mut out = vec![0.0f32; m * n];
                            mlas_sys::sqnbit_gemm(
                                &packed_weight.packed,
                                m,
                                &a,
                                bias.as_deref(),
                                &mut out,
                                true,
                            );
                            let mut expected = reference(&a, &ref_weights, m, k, n);
                            if let Some(bias) = &bias {
                                for row in expected.chunks_exact_mut(n) {
                                    for (v, b) in row.iter_mut().zip(bias) {
                                        *v += b;
                                    }
                                }
                            }
                            // CompInt8 quantizes A to int8, so it needs a looser
                            // tolerance than the near-exact CompFp32 dequant.
                            let tol = if accuracy_level == 4 {
                                #[cfg(target_arch = "aarch64")]
                                {
                                    2e-1
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                {
                                    6e-2
                                }
                            } else {
                                2e-3
                            };
                            mlas_close(
                                &out,
                                &expected,
                                tol,
                                &format!(
                                    "blk{block_size} asym{asymmetric} acc{accuracy_level} m{m} bias{}",
                                    bias.is_some()
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    /// `try_mlas_sqnbit` must fall back (return `Ok(None)`) for cases MLAS
    /// SQNBit cannot serve: `g_idx` present (no per-row group indices) and
    /// `bits == 2` (left to the correctness path). These guards short-circuit
    /// ahead of backend detection, so the decision is deterministic.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_falls_back_for_gidx_and_bits2() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (2usize, 32usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let a = vec![0.5f32; k];
        let mut result = vec![0.0f32; n];

        // int4 with g_idx present → fall back.
        let kernel = test_kernel(k, n, block_size);
        let b = Owned::u8(
            &[n, k_blocks, block_size / 2],
            &vec![0x88; n * k_blocks * block_size / 2],
        );
        let scales = Owned::f32(&[n, k_blocks], &vec![1.0; n * k_blocks]);
        let g_idx: Vec<i32> = (0..k).map(|i| (i / block_size) as i32).collect();
        let g_idx = Owned::i32(&[k], &g_idx);
        assert_eq!(
            kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales.view(),
                    None,
                    Some(&g_idx.view()),
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap(),
            None,
            "g_idx present must fall back",
        );

        // bits == 2 → fall back.
        let blob2 = block_size / 4;
        let kernel2 = MatMulNBitsKernel {
            bits: 2,
            ..test_kernel(k, n, block_size)
        };
        let b2 = Owned::u8(&[n, k_blocks, blob2], &vec![0x55; n * k_blocks * blob2]);
        assert_eq!(
            kernel2
                .try_mlas_sqnbit(
                    &b2.view(),
                    &scales.view(),
                    None,
                    None,
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap(),
            None,
            "bits==2 must fall back",
        );
    }

    /// The SQNBit decode crossover parses `NXRT_SQNBIT_DECODE_MIN`, falling
    /// back to the topology-derived default for absent, empty, or malformed values.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_resolve_decode_min_parses_or_defaults() {
        assert_eq!(default_sqnbit_decode_min(96), 16);
        assert_eq!(default_sqnbit_decode_min(4), 6);
        assert_eq!(default_sqnbit_decode_min(8), 8);
        assert_eq!(resolve_decode_min(None, 96), 16);
        assert_eq!(resolve_decode_min(Some(""), 96), 16);
        assert_eq!(resolve_decode_min(Some("abc"), 96), 16);
        assert_eq!(resolve_decode_min(Some("32"), 96), 32);
        assert_eq!(resolve_decode_min(Some("  8 "), 96), 8);
        assert_eq!(resolve_decode_min(Some("1"), 96), 1);
    }

    /// The standalone MLAS pool cannot see `DECODE_THREADS_OVERRIDE`, so
    /// `set_decode_thread_budget` has to forward explicitly. Without the
    /// forwarding, `--cpu-cores N` bounded dense `MlasGemmBatch` work only on
    /// Linux, and only indirectly via the affinity mask.
    #[cfg(feature = "mlas")]
    #[test]
    fn set_decode_thread_budget_forwards_the_budget_to_the_standalone_mlas_pool() {
        let _guard = backend_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let restore = decode_threads_override();

        set_decode_thread_budget(Some(6)).expect("6 is a legal budget");
        assert_eq!(
            mlas_sys::configured_pool_thread_budget(),
            Some(6),
            "the MLAS pool must observe the EP's programmatic budget"
        );

        set_decode_thread_budget(None).expect("clearing is legal");
        assert_eq!(
            mlas_sys::configured_pool_thread_budget(),
            None,
            "clearing the EP budget must clear the MLAS pool budget too"
        );

        assert!(
            set_decode_thread_budget(Some(0)).is_err(),
            "zero stays the opt-out sentinel, not a legal pool size"
        );
        assert_eq!(
            mlas_sys::configured_pool_thread_budget(),
            None,
            "a rejected budget must not reach the MLAS pool"
        );

        set_decode_thread_budget(restore).expect("restoring the prior budget");
    }

    /// Serialize the few tests that mutate `NXRT_CPU_GEMM_BACKEND` so the global
    /// backend override does not race concurrent test threads.
    #[cfg(feature = "mlas")]
    fn backend_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    const SPMD_PARITY_CHILD_ENV: &str = "NXRT_SPMD_PARITY_CHILD";
    const SPMD_PARITY_MARKER: &str = "NXRT_SPMD_PARITY_BYTES=";

    fn real_int4_decode_fixture_bytes() -> Vec<u8> {
        let (n, k, block_size) = (1024usize, 1024usize, 32usize);
        let blocks = k / block_size;
        let packed = PackedInt4Weight {
            values: (0..n * blocks * (block_size / 2))
                .map(|index| {
                    let low = ((index * 13 + 3) & 0xf) as u8;
                    let high = ((index * 7 + 11) & 0xf) as u8;
                    low | (high << 4)
                })
                .collect(),
            scales: (0..n * blocks)
                .map(|index| 0.000_5 + (index % 29) as f32 * 0.000_031_25)
                .collect(),
        };
        let mut activation: Vec<f32> = (0..k)
            .map(|index| ((index * 37 % 257) as f32 - 128.0) * 0.007_812_5)
            .collect();
        let dot_kernel = selected_dot_kernel();
        let mut bytes = Vec::with_capacity(6 * n * std::mem::size_of::<f32>());

        with_decode_pool_scope(true, || {
            for op in 0..6usize {
                let mut output = vec![0.0f32; n];
                int4_matmul_m1(
                    &activation,
                    &packed,
                    &mut output,
                    k,
                    n,
                    block_size,
                    dot_kernel,
                );
                for value in &output {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                for (index, value) in activation.iter_mut().enumerate() {
                    *value = output[index] * 0.125
                        + ((op * 17 + index * 5) % 31) as f32 * 0.000_976_562_5;
                }
            }
        });
        bytes
    }

    /// Portable, deterministic decode-worker count for the SPMD parity children.
    ///
    /// The invariant this test exercises is that persistent-SPMD row-sharding is
    /// byte-identical to the flat path across an **odd** worker count (an uneven
    /// remainder split), for every sequential int4 op. The worker count itself is
    /// otherwise semantically irrelevant, so it must be chosen to keep that
    /// invariant true on *every* platform while never oversubscribing the host:
    ///
    /// * **Odd** and `>= 3` on any host with `>= 3` logical CPUs, so the uneven
    ///   sharding path (`worker_row_segments` remainder) is always covered.
    /// * **Never greater than the host's logical CPU count**, so the children do
    ///   not spawn more busy-waiting decode/rayon threads than there are cores.
    ///   The previous hard-coded `31` grossly oversubscribed constrained CI
    ///   runners; on the native Windows ARM64 runner that intermittently faulted
    ///   the whole test binary with `STATUS_ACCESS_VIOLATION` (0xC0000005) — a
    ///   flaky, environment-level crash (ThreadSanitizer finds no data race, and
    ///   macOS arm64 / Windows x64 / Linux never reproduce it).
    /// * **Capped at 15** so even a many-core host keeps the thread pressure
    ///   bounded and the sub-second parity run fast.
    ///
    /// Because the value never exceeds `available_parallelism()`, the pool's
    /// worker count is *not* clamped down (unlike the old `min(31, cores)`, whose
    /// effective parity depended on the runner's core-count parity), so the
    /// odd-worker coverage is deterministic per host instead of luck-of-the-draw.
    fn parity_worker_count() -> usize {
        let available = available_parallelism().max(1);
        // Largest odd count that does not exceed the host core count.
        let host_odd = if available.is_multiple_of(2) {
            available.saturating_sub(1)
        } else {
            available
        };
        host_odd.clamp(1, 15)
    }

    fn parity_child_output(persistent: bool) -> Vec<u8> {
        parity_child_output_mode(if persistent {
            SpmdParityMode::On
        } else {
            SpmdParityMode::Off
        })
    }

    /// Which persistence policy the SPMD parity child runs under.
    #[derive(Clone, Copy)]
    enum SpmdParityMode {
        /// `=0`: the flat legacy path (baseline).
        Off,
        /// `=1` or unset: the persistent pool (the default).
        On,
        /// `=auto`: adaptive calibrated mode (warmup routes the fixture through
        /// the pool, so its bytes must still match the flat baseline).
        Adaptive,
    }

    impl SpmdParityMode {
        fn child_tag(self) -> &'static str {
            match self {
                Self::Off => "off",
                Self::On => "on",
                Self::Adaptive => "auto",
            }
        }
    }

    fn parity_child_output_mode(mode: SpmdParityMode) -> Vec<u8> {
        let workers = parity_worker_count().to_string();
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::spmd_real_int4_parity_subprocess")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(SPMD_PARITY_CHILD_ENV, mode.child_tag())
            .env(DECODE_THREADS_ENV, &workers)
            .env("RAYON_NUM_THREADS", &workers)
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        match mode {
            SpmdParityMode::On => {
                command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1");
            }
            SpmdParityMode::Off => {
                // The OFF child sets `=0` to exercise the flat legacy path against
                // the pool children.
                command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "0");
            }
            SpmdParityMode::Adaptive => {
                // `=auto`: the adaptive calibrated mode. The pool is built and the
                // warmup step routes the fixture through it, so this exercises the
                // real adaptive entry point end-to-end.
                command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "auto");
            }
        }
        let persistent = matches!(mode, SpmdParityMode::On);
        let output = command.output().expect("run SPMD parity child process");
        assert!(
            output.status.success(),
            "SPMD parity child failed (persistent={persistent}):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("child output is UTF-8");
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.find(SPMD_PARITY_MARKER)
                    .map(|index| &line[index + SPMD_PARITY_MARKER.len()..])
            })
            .expect("child emitted parity bytes");
        assert_eq!(encoded.len() % 2, 0);
        encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).unwrap();
                u8::from_str_radix(pair, 16).unwrap()
            })
            .collect()
    }

    #[test]
    fn spmd_real_int4_parity_subprocess() {
        let Ok(mode) = std::env::var(SPMD_PARITY_CHILD_ENV) else {
            return;
        };
        let persistent = mode == "on";
        let auto = mode == "auto";
        // The `on` (forced) and `auto` children both build the pool; only `off`
        // (`=0`) leaves it unbuilt.
        assert_eq!(
            crate::decode_spmd::pools().is_some(),
            persistent || auto,
            "the forced/auto children must build the persistent pool and the off child must not"
        );
        SPMD_TEST_DISPATCHES.store(0, std::sync::atomic::Ordering::Relaxed);
        let bytes = real_int4_decode_fixture_bytes();
        if persistent || auto {
            // Forced always dispatches; `auto`'s warmup step routes the fixture's
            // single decode scope through the pool. Either way every real int4 op
            // must have been sharded across the persistent workers.
            assert!(
                SPMD_TEST_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) >= 6,
                "persistent/auto parity child did not route every real int4 op through SPMD"
            );
        }
        if persistent {
            // Self-verify the odd-worker coverage. On a single-node host the
            // requested (odd) worker count is used verbatim, so the uneven
            // remainder sharding path is genuinely exercised; assert that
            // explicitly to guard against a future clamp silently degrading it.
            // A multi-node host may rebalance/cap the split per node
            // (`split_workers`), so there we rely on the SPMD-routing assert
            // above plus the outer byte-parity check rather than a fixed count.
            let pool = crate::decode_spmd::pools().expect("ON child built the persistent pool");
            if pool.node_count() == 1 {
                let workers = pool.total_workers();
                assert_eq!(
                    workers,
                    parity_worker_count(),
                    "single-node persistent pool must use the requested (unclamped) worker count"
                );
                if workers > 1 {
                    assert!(
                        !workers.is_multiple_of(2),
                        "parity must run at an odd worker count to cover uneven row sharding"
                    );
                }
            }
        }
        let encoded: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        println!("{SPMD_PARITY_MARKER}{encoded}");
    }

    #[test]
    fn spmd_real_multi_op_int4_is_bit_identical_at_odd_worker_count() {
        let baseline = parity_child_output(false);
        let persistent = parity_child_output(true);
        assert_eq!(
            persistent, baseline,
            "odd-worker persistent SPMD output must be byte-identical to flag-OFF \
             across every sequential packed-int4 MatMulNBits op"
        );
    }

    #[test]
    fn spmd_adaptive_calibrated_decode_is_bit_identical_to_flat() {
        // Token-exactness of the adaptive path: with
        // `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto`, the calibrator may route
        // a decode step through the persistent pool. Because both paths are
        // token-exact (N-tile aligned, PR #110), the adaptive child's bytes must
        // be identical to the flat (`=0`) baseline. This guards the constraint
        // that adaptive can never route a non-exact config: it only ever selects
        // between the exact pool and the exact flat path.
        let baseline = parity_child_output_mode(SpmdParityMode::Off);
        let adaptive = parity_child_output_mode(SpmdParityMode::Adaptive);
        assert_eq!(
            adaptive, baseline,
            "adaptive-calibrated decode output must be byte-identical to the flat baseline"
        );
    }

    #[cfg(feature = "mlas")]
    const MLAS_SHARD_PARITY_CHILD_ENV: &str = "NXRT_MLAS_SHARD_PARITY_CHILD";
    #[cfg(feature = "mlas")]
    const MLAS_SHARD_PARITY_MARKER: &str = "NXRT_MLAS_SHARD_PARITY_BYTES=";

    #[cfg(feature = "mlas")]
    fn mlas_shard_parity_child_output(no_shard: bool, work_stealing: bool) -> Vec<f32> {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::mlas_sharded_decode_parity_subprocess")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(MLAS_SHARD_PARITY_CHILD_ENV, "1")
            .env(DECODE_THREADS_ENV, "3")
            .env("RAYON_NUM_THREADS", "3")
            .env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1")
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        if work_stealing {
            command.env(crate::decode_spmd::DECODE_SCHEDULE_ENV, "steal");
            command.env("ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER", "2");
        } else {
            command.env_remove(crate::decode_spmd::DECODE_SCHEDULE_ENV);
            command.env_remove("ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER");
        }
        if no_shard {
            command.env("ONNX_GENAI_CPU_MM_MLAS_NO_SHARD", "1");
        } else {
            command.env_remove("ONNX_GENAI_CPU_MM_MLAS_NO_SHARD");
        }
        let output = command
            .output()
            .expect("run MLAS-shard parity child process");
        assert!(
            output.status.success(),
            "MLAS-shard parity child failed (no_shard={no_shard}):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("child output is UTF-8");
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.find(MLAS_SHARD_PARITY_MARKER)
                    .map(|index| &line[index + MLAS_SHARD_PARITY_MARKER.len()..])
            })
            .expect("child emitted MLAS-shard parity bytes");
        assert_eq!(encoded.len() % 8, 0);
        encoded
            .as_bytes()
            .chunks_exact(8)
            .map(|hex| {
                let hex = std::str::from_utf8(hex).unwrap();
                f32::from_bits(u32::from_str_radix(hex, 16).unwrap())
            })
            .collect()
    }

    /// Output width shared by every parity-child config. N=176 with three
    /// persistent workers cuts the unaligned (align=1) worker boundaries at
    /// columns 59 and 118 -- both *mid-N-tile* (59 = 4*14+3, 118 = 4*29+2). The
    /// shipped [`MLAS_SQNBIT_DECODE_SHARD_ALIGN`] = 16 snaps them to 64 and 112
    /// (both whole N-tile boundaries), so every SQNBit 4-wide N-tile stays
    /// inside one shard and the sharded decode is bit-identical to full-width.
    /// Disabling the fix (const -> 1) restores the mid-tile cut, which drifts by
    /// 1+ ULP on this host (verified across every config below), so this guard
    /// is *non-vacuous*: see `.squad/decisions/inbox/chew-spmd-test-harden.md`.
    #[cfg(feature = "mlas")]
    const MLAS_SHARD_PARITY_N: usize = 176;

    /// `(k, block_size)` configs the parity child sweeps. Each one was verified
    /// (drift search harness `search_drift_configs`) to differ from full-width
    /// under an unaligned mid-tile cut at N=176/3-workers, so the aligned guard
    /// cannot pass vacuously for any of them. All symmetric (int4, zp=8) so the
    /// child needs no zero-point plumbing.
    #[cfg(feature = "mlas")]
    const MLAS_SHARD_PARITY_CONFIGS: &[(usize, usize)] =
        &[(256, 128), (512, 32), (256, 64), (256, 32), (128, 32)];

    /// Isolated child so the persistent-pool and `NO_SHARD` one-time env gates
    /// are initialized independently. This exercises the actual cached
    /// `mlas_shards` + SPMD-scope route, not a manually assembled shard proxy.
    ///
    /// Sweeps every [`MLAS_SHARD_PARITY_CONFIGS`] entry at
    /// [`MLAS_SHARD_PARITY_N`] and concatenates all decode outputs into one
    /// bit-stream, so the parent bit-compares the *entire* drift matrix (the
    /// shard route vs the `NO_SHARD` full-width route) in one shot.
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_sharded_decode_parity_subprocess() {
        let _probe = lock_dispatch_probe();
        if std::env::var(MLAS_SHARD_PARITY_CHILD_ENV).is_err() {
            return;
        }

        let n = MLAS_SHARD_PARITY_N;
        let pool = crate::decode_spmd::pools().expect("forced persistent SPMD pool");
        let segments = pool.output_column_segments(n, MLAS_SQNBIT_DECODE_SHARD_ALIGN);
        let work_stealing = std::env::var(crate::decode_spmd::DECODE_SCHEDULE_ENV)
            .is_ok_and(|value| value.trim() == "steal");
        if work_stealing {
            assert!(
                segments.len() > 3,
                "work-stealing child should create extra dynamic MLAS tiles"
            );
        } else {
            assert_eq!(
                segments.len(),
                3,
                "child requested three persistent workers"
            );
            assert!(
                segments.windows(2).any(|pair| pair[0].1 != pair[1].1),
                "N={n} must create uneven worker output-column segments: {segments:?}"
            );
        }
        assert_eq!(segments.iter().map(|&(_, len)| len).sum::<usize>(), n);

        let activation = pseudo(
            MLAS_SHARD_PARITY_CONFIGS
                .iter()
                .map(|&(k, _)| k)
                .max()
                .unwrap(),
            0.8,
        );
        let mut encoded = String::new();
        for &(k, block_size) in MLAS_SHARD_PARITY_CONFIGS {
            let blocks = k.div_ceil(block_size);
            let weights_nk = pseudo(n * k, 0.3);
            let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed_bytes);
            let scales_t = Owned::f32(&[n, blocks], &scales);
            let kernel = test_kernel(k, n, block_size);
            let mut output = vec![0.0f32; n];

            let served = with_decode_pool_scope(true, || {
                assert!(
                    spmd_decode_active().is_some(),
                    "the actual MLAS call must run inside the persistent SPMD scope"
                );
                kernel
                    .try_mlas_sqnbit(
                        &b.view(),
                        &scales_t.view(),
                        None,
                        None,
                        true,
                        &activation[..k],
                        1,
                        None,
                        &mut output,
                    )
                    .unwrap()
            });
            assert_eq!(
                served,
                Some(()),
                "MLAS CompFp32 must serve this decode (k={k} blk={block_size})"
            );

            if mlas_no_shard() {
                assert!(
                    kernel.mlas_shards.get().is_none(),
                    "NO_SHARD must select the full-width MLAS call (k={k} blk={block_size})"
                );
            } else {
                let shards = kernel
                    .mlas_shards
                    .get()
                    .expect("the cached sharded MLAS route must be populated")
                    .as_ref()
                    .expect("MLAS packed every worker shard");
                assert_eq!(shards.len(), segments.len());
                assert!(
                    shards.iter().flatten().count() > 1,
                    "the cached route must contain multiple real MLAS shards \
                     (k={k} blk={block_size})"
                );
            }

            for value in &output {
                encoded.push_str(&format!("{:08x}", value.to_bits()));
            }
        }

        println!("{MLAS_SHARD_PARITY_MARKER}{encoded}");
    }

    /// Chew #2: the real cached SPMD MLAS-shard decode route must agree with
    /// the `NO_SHARD=1` full-width route. The per-worker shard boundaries are
    /// snapped to [`MLAS_SQNBIT_DECODE_SHARD_ALIGN`], keeping every MLAS N-tile
    /// whole inside one shard, so the sharded decode is now *bit-identical* to
    /// the full-width call (no ~1 ULP N-tile-boundary drift): assert byte-for-
    /// byte equality of the raw f32 bit patterns across the entire
    /// [`MLAS_SHARD_PARITY_CONFIGS`] drift matrix.
    ///
    /// Non-vacuity is enforced by construction: every config is a *known
    /// drifter* at N=176/3-workers when alignment is off (see
    /// `search_drift_configs` and the decision note). Flipping
    /// `MLAS_SQNBIT_DECODE_SHARD_ALIGN` from 16 to 1 makes this test fail with
    /// `max_ulp >= 1`; with 16 it passes bit-exact.
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_sharded_decode_matches_no_shard_full_width() {
        let sharded = mlas_shard_parity_child_output(false, false);
        let full_width = mlas_shard_parity_child_output(true, false);
        assert_eq!(
            sharded.len(),
            full_width.len(),
            "cached SPMD MLAS shards vs NO_SHARD full-width decode: length mismatch"
        );
        let mismatches: Vec<_> = sharded
            .iter()
            .zip(&full_width)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i, a.to_bits(), b.to_bits()))
            .collect();
        let max_ulp = sharded
            .iter()
            .zip(&full_width)
            .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            mismatches.is_empty(),
            "aligned SPMD MLAS-shard decode must be bit-identical to NO_SHARD full-width \
             (max_ulp={max_ulp}); if this fails, MLAS_SQNBIT_DECODE_SHARD_ALIGN is not \
             keeping N-tiles whole. mismatching (index, sharded_bits, full_bits): {mismatches:?}"
        );
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_work_stealing_decode_matches_no_shard_full_width() {
        let sharded = mlas_shard_parity_child_output(false, true);
        let full_width = mlas_shard_parity_child_output(true, true);
        assert_eq!(sharded.len(), full_width.len());
        let mismatches: Vec<_> = sharded
            .iter()
            .zip(&full_width)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i, a.to_bits(), b.to_bits()))
            .collect();
        assert!(
            mismatches.is_empty(),
            "work-stealing MLAS decode shards must stay bit-identical to full-width; \
             mismatching (index, sharded_bits, full_bits): {mismatches:?}"
        );
    }

    #[cfg(feature = "mlas")]
    const MLAS_PREFILL_PARITY_CHILD_ENV: &str = "NXRT_MLAS_PREFILL_PARITY_CHILD";
    #[cfg(feature = "mlas")]
    const MLAS_PREFILL_PARITY_MARKER: &str = "NXRT_MLAS_PREFILL_PARITY_BYTES=";

    /// Batched rows the prefill parity child runs (`m > 1`, odd so the
    /// `(shard x m-row-block)` tiling produces uneven row blocks). With the child's
    /// `RAYON_NUM_THREADS` above its persistent-worker count, the parallel prefill
    /// dispatch splits each shard's rows into multiple blocks, so this exercises
    /// both the N-shard and the M-row-block tiling axes.
    #[cfg(feature = "mlas")]
    const MLAS_PREFILL_PARITY_M: usize = 7;

    /// Run the prefill (`m > 1`) MLAS SQNBit path in a child process so the
    /// one-time `ONNX_GENAI_CPU_MM_MLAS_PREFILL_SERIAL` gate initializes cleanly.
    /// With `serial = true` the child forces the pre-fix serial per-shard
    /// `multithread=true` loop; with `serial = false` it uses the default parallel
    /// `(shard x m-row-block)` dispatch. Both run through the real cached
    /// `mlas_shards` route under a forced multi-worker persistent pool, so a
    /// bit-comparison of the two outputs proves the parallel dispatch is
    /// token-exact to the serial baseline. Returns the concatenated `[m, n]`
    /// output bits across the drift-matrix configs.
    #[cfg(feature = "mlas")]
    fn mlas_prefill_parity_child_output(serial: bool) -> Vec<f32> {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::mlas_prefill_dispatch_parity_subprocess")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(MLAS_PREFILL_PARITY_CHILD_ENV, "1")
            .env(DECODE_THREADS_ENV, "3")
            // More Rayon threads than persistent workers so the parallel prefill
            // dispatch also splits M into row blocks (row_blocks = threads/shards).
            .env("RAYON_NUM_THREADS", "6")
            .env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1")
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        if serial {
            command.env("ONNX_GENAI_CPU_MM_MLAS_PREFILL_SERIAL", "1");
        } else {
            command.env_remove("ONNX_GENAI_CPU_MM_MLAS_PREFILL_SERIAL");
        }
        let output = command
            .output()
            .expect("run MLAS prefill parity child process");
        assert!(
            output.status.success(),
            "MLAS prefill parity child failed (serial={serial}):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("child output is UTF-8");
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.find(MLAS_PREFILL_PARITY_MARKER)
                    .map(|index| &line[index + MLAS_PREFILL_PARITY_MARKER.len()..])
            })
            .expect("child emitted MLAS prefill parity bytes");
        assert_eq!(encoded.len() % 8, 0);
        encoded
            .as_bytes()
            .chunks_exact(8)
            .map(|hex| {
                let hex = std::str::from_utf8(hex).unwrap();
                f32::from_bits(u32::from_str_radix(hex, 16).unwrap())
            })
            .collect()
    }

    /// Isolated child: run a batched (`m > 1`) prefill MatMulNBits through the
    /// real cached SPMD MLAS-shard route across the [`MLAS_SHARD_PARITY_CONFIGS`]
    /// drift matrix and emit every output's f32 bits. The parent runs this twice
    /// (serial vs parallel dispatch) and bit-compares. Prefill runs *outside* a
    /// decode scope, so `run_mlas_shards` takes its `m > 1` branch.
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_prefill_dispatch_parity_subprocess() {
        let _probe = lock_dispatch_probe();
        if std::env::var(MLAS_PREFILL_PARITY_CHILD_ENV).is_err() {
            return;
        }

        let n = MLAS_SHARD_PARITY_N;
        let m = MLAS_PREFILL_PARITY_M;
        let pool = crate::decode_spmd::pools().expect("forced persistent SPMD pool");
        let segments = pool.output_column_segments(n, MLAS_SQNBIT_DECODE_SHARD_ALIGN);
        assert!(
            segments.iter().filter(|&&(_, len)| len > 0).count() > 1,
            "prefill parity needs multiple real N shards to exercise the parallel \
             dispatch: {segments:?}"
        );

        let max_k = MLAS_SHARD_PARITY_CONFIGS
            .iter()
            .map(|&(k, _)| k)
            .max()
            .unwrap();
        let activation = pseudo(m * max_k, 0.8);
        let mut encoded = String::new();
        for &(k, block_size) in MLAS_SHARD_PARITY_CONFIGS {
            let blocks = k.div_ceil(block_size);
            let weights_nk = pseudo(n * k, 0.3);
            let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed_bytes);
            let scales_t = Owned::f32(&[n, blocks], &scales);
            let kernel = test_kernel(k, n, block_size);
            let mut output = vec![0.0f32; m * n];

            // Batched activations: the first `m * k` entries reshaped as [m, k].
            let mut activations = vec![0.0f32; m * k];
            for row in 0..m {
                activations[row * k..(row + 1) * k]
                    .copy_from_slice(&activation[row * max_k..row * max_k + k]);
            }

            let served = kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales_t.view(),
                    None,
                    None,
                    true,
                    &activations,
                    m,
                    None,
                    &mut output,
                )
                .unwrap();
            assert_eq!(
                served,
                Some(()),
                "MLAS CompFp32 must serve this prefill (k={k} blk={block_size})"
            );
            let shards = kernel
                .mlas_shards
                .get()
                .expect("the cached sharded MLAS route must be populated")
                .as_ref()
                .expect("MLAS packed every worker shard");
            assert!(
                shards.iter().flatten().count() > 1,
                "the cached route must contain multiple real MLAS shards \
                 (k={k} blk={block_size})"
            );

            for value in &output {
                encoded.push_str(&format!("{:08x}", value.to_bits()));
            }
        }

        println!("{MLAS_PREFILL_PARITY_MARKER}{encoded}");
    }

    /// The default parallel `(shard x m-row-block)` prefill dispatch must be
    /// *bit-identical* to the pre-fix serial per-shard `multithread=true` loop.
    /// Both write the same N-tile-aligned shards into disjoint output windows, so
    /// the concatenated `[m, n]` result cannot differ by even one ULP. This locks
    /// the token-exactness of the prefill-dispatch fix: reusing the warm parked
    /// pool via a single parallel pass changes only *how* the shards are
    /// dispatched, never the arithmetic. (A companion mlas-sys experiment,
    /// `perf_prefill_shard_dispatch`, shows the same output also equals a single
    /// full-width call, `max_ulp = 0`, and is ~15x faster than the serial loop.)
    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_prefill_parallel_dispatch_matches_serial() {
        let parallel = mlas_prefill_parity_child_output(false);
        let serial = mlas_prefill_parity_child_output(true);
        assert_eq!(
            parallel.len(),
            serial.len(),
            "parallel vs serial prefill dispatch: length mismatch"
        );
        let mismatches: Vec<_> = parallel
            .iter()
            .zip(&serial)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i, a.to_bits(), b.to_bits()))
            .collect();
        let max_ulp = parallel
            .iter()
            .zip(&serial)
            .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            mismatches.is_empty(),
            "parallel prefill dispatch must be bit-identical to the serial \
             multithread=true loop (max_ulp={max_ulp}); mismatching \
             (index, parallel_bits, serial_bits): {mismatches:?}"
        );
    }

    /// TEMP search harness: for a sweep of (n, worker_count, k, block_size),
    /// build unaligned (align=1) contiguous N shards exactly the way the SPMD
    /// route would, run each shard's SQNBit GEMV, concat, and report which
    /// configs drift from the full-width call. Run with:
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas -- --ignored --nocapture search_drift_configs
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore]
    fn search_drift_configs() {
        fn split(n: usize, w: usize) -> Vec<(usize, usize)> {
            let base = n / w;
            let rem = n % w;
            let mut segs = Vec::new();
            let mut off = 0;
            for worker in 0..w {
                let len = base + usize::from(worker < rem);
                segs.push((off, len));
                off += len;
            }
            segs
        }
        for &n in &[97usize, 129, 130, 131, 176, 177, 191, 193, 200, 255, 257] {
            for &w in &[2usize, 3, 4, 5, 6, 7, 8] {
                if w > n {
                    continue;
                }
                for &(k, block_size) in &[
                    (256usize, 128usize),
                    (512, 32),
                    (256, 64),
                    (256, 32),
                    (128, 32),
                ] {
                    for &asym in &[false, true] {
                        let blocks = k.div_ceil(block_size);
                        let blob = block_size / 2;
                        let weights_nk = pseudo(n * k, 0.3);
                        let (packed, scales, zps, _dq) =
                            quantize(&weights_nk, n, k, block_size, asym);
                        let activation = pseudo(k, 0.8);

                        let make = |start: usize, len: usize| {
                            let pb = &packed[start * blocks * blob..(start + len) * blocks * blob];
                            let sc = &scales[start * blocks..(start + len) * blocks];
                            let zp = zps.as_deref().map(|z| {
                                let row = blocks.div_ceil(2);
                                &z[start * row..(start + len) * row]
                            });
                            mlas_sys::SQNBitPackedB::new(
                                len,
                                k,
                                4,
                                block_size,
                                mlas_sys::SQNBitComputeType::Fp32,
                                pb,
                                sc,
                                zp,
                            )
                        };
                        let Some(full) = make(0, n) else { continue };
                        let mut c_full = vec![0.0f32; n];
                        mlas_sys::sqnbit_gemm(&full, 1, &activation, None, &mut c_full, false);

                        let segs = split(n, w);
                        let mut c_shard = vec![0.0f32; n];
                        for &(start, len) in &segs {
                            if len == 0 {
                                continue;
                            }
                            let packed_shard = make(start, len).unwrap();
                            let mut out = vec![0.0f32; len];
                            mlas_sys::sqnbit_gemm(
                                &packed_shard,
                                1,
                                &activation,
                                None,
                                &mut out,
                                false,
                            );
                            c_shard[start..start + len].copy_from_slice(&out);
                        }
                        let max_ulp = c_full
                            .iter()
                            .zip(&c_shard)
                            .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
                            .max()
                            .unwrap_or(0);
                        if max_ulp > 0 {
                            let interior: Vec<usize> =
                                segs.iter().map(|&(s, _)| s).skip(1).collect();
                            println!(
                                "DRIFT n={n} w={w} k={k} blk={block_size} asym={asym} \
                                 max_ulp={max_ulp} boundaries={interior:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    const AFFINITY_DEFER_CHILD_ENV: &str = "NXRT_AFFINITY_DEFER_CHILD";
    const AFFINITY_DEFER_MARKER: &str = "NXRT_AFFINITY_DEFER=";

    /// Child process for the affinity-defer routing tests (Chew #1). Dispatches on
    /// the scenario in `AFFINITY_DEFER_CHILD_ENV`; a plain run (var unset) is a
    /// no-op so the test is inert in the normal suite. Env is set by the parent
    /// *before* the process starts, so `pools()`/`plan_decode_affinity` observe it
    /// on their first (and only) evaluation.
    #[test]
    fn affinity_defer_routing_child() {
        let Ok(scenario) = std::env::var(AFFINITY_DEFER_CHILD_ENV) else {
            return;
        };
        match scenario.as_str() {
            // (a) Adaptive + explicit non-numa-split affinity -> defer to the
            // flat path: the persistent SPMD pool must NOT be built.
            "auto_off" | "auto_node" | "auto_compact" => {
                assert!(
                    crate::decode_spmd::pools().is_none(),
                    "Adaptive + explicit affinity ({scenario}) must defer to the flat \
                     path and build no persistent SPMD pool"
                );
            }
            // (b) On (default or `=1`) + affinity set -> SPMD still wins.
            "forced_off" => {
                assert!(
                    crate::decode_spmd::pools().is_some(),
                    "On (default) persistent pool must ignore the affinity defer and build SPMD"
                );
            }
            // (c) Adaptive + malformed affinity -> deferred to flat AND the flat path
            // still surfaces the malformed-value error.
            "auto_malformed" => {
                assert!(
                    crate::decode_spmd::pools().is_none(),
                    "Adaptive + malformed affinity must defer to the flat path"
                );
                assert!(
                    crate::decode_affinity::plan_decode_affinity(4).is_err(),
                    "malformed affinity must still surface an error on the deferred flat path"
                );
            }
            other => panic!("unknown affinity-defer scenario `{other}`"),
        }
        // Deterministically stop and join the persistent pool's workers before
        // this child process exits. The pool lives in a module-level `static`,
        // which Rust never `Drop`s at exit, so without this join the forced
        // scenario's hot worker threads would still be spinning/parked on
        // `Arc<SharedState>` while the process tears down its runtime -- the
        // race that intermittently faulted this child with an empty-stderr
        // `STATUS_ACCESS_VIOLATION` (0xC0000005) on native Windows ARM64. It is a
        // no-op for the auto scenarios (they build no pool).
        crate::decode_spmd::shutdown_pools();
        println!("{AFFINITY_DEFER_MARKER}ok");
    }

    /// The NTSTATUS code Windows reports for `STATUS_ACCESS_VIOLATION`
    /// (`0xC0000005`) surfaced through `ExitStatus::code()` as a signed `i32`.
    /// `ExitStatus::code()` is cross-platform, so this constant compiles on every
    /// target; the crash it names only ever occurs on native Windows ARM64.
    const STATUS_ACCESS_VIOLATION: i32 = -1_073_741_819;

    /// Total attempts allowed for a single affinity-defer child run before we give
    /// up and surface a failure. One nominal attempt plus two retries: the
    /// environmental crash is rare, so a small bound reliably rides through it
    /// without masking a persistent problem.
    const AFFINITY_DEFER_CHILD_MAX_ATTEMPTS: u32 = 3;

    /// Classify an unsuccessful affinity-defer child exit as the *known
    /// environmental* `STATUS_ACCESS_VIOLATION` crash (retryable) versus a real
    /// test failure (not retryable). Extracted as a pure function so the exact
    /// signature is unit-testable and the retry stays narrowly scoped.
    ///
    /// All four conditions must hold to treat the exit as the environmental flake:
    ///   1. the child exited unsuccessfully (`success` is `false`), AND
    ///   2. it emitted no success marker (`{AFFINITY_DEFER_MARKER}ok`), AND
    ///   3. its stderr shows no Rust panic (no `panicked at` / `assertion`
    ///      text) — a genuine assertion failure must fail fast, never retry, AND
    ///   4. the exit code is exactly the Windows `STATUS_ACCESS_VIOLATION`
    ///      NTSTATUS. Matching that specific code keeps the retry Windows-only in
    ///      practice while the code stays portable.
    fn is_environmental_access_violation_crash(
        success: bool,
        exit_code: Option<i32>,
        stdout: &str,
        stderr: &str,
    ) -> bool {
        if success {
            return false;
        }
        if stdout.contains(&format!("{AFFINITY_DEFER_MARKER}ok")) {
            return false;
        }
        if stderr.contains("panicked at") || stderr.contains("assertion") {
            return false;
        }
        exit_code == Some(STATUS_ACCESS_VIOLATION)
    }

    fn run_affinity_defer_child(scenario: &str, affinity: &str, forced: bool) {
        // Worker/Rayon thread count for the child. The `forced` scenario builds
        // the persistent SPMD pool, spinning up this many busy-waiting workers
        // *inside* an already-spawned child test binary; a hard-coded value
        // (previously `8`) oversubscribes the constrained native Windows ARM64 CI
        // runner and intermittently faults the whole child with an empty-stderr
        // `STATUS_ACCESS_VIOLATION` (0xC0000005) — the same environment-level
        // flaky class PR #111 fixed for `parity_worker_count`. Reuse that helper
        // (largest odd count <= `available_parallelism()`, capped at 15) so the
        // pressure is bounded on small runners while every host still forces a
        // real (>= 1 worker) pool, keeping the affinity-defer assertion meaningful.
        let workers = parity_worker_count().to_string();
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::affinity_defer_routing_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(AFFINITY_DEFER_CHILD_ENV, scenario)
            .env(crate::decode_affinity::DECODE_AFFINITY_ENV, affinity)
            .env(DECODE_THREADS_ENV, &workers)
            .env("RAYON_NUM_THREADS", &workers);
        if forced {
            command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1");
        } else {
            // Adaptive (`=auto`): with an explicit decode-affinity set (as these
            // scenarios do), Adaptive defers to that request and builds no persistent
            // SPMD pool, routing decode through the flat/affinity path.
            command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "auto");
        }

        // Bounded retry loop scoped to *exactly* the known environmental crash
        // signature. On weakly-ordered native Windows ARM64 the child process
        // occasionally faults the whole process with `STATUS_ACCESS_VIOLATION`
        // (0xC0000005) and empty stderr during SPMD pool build/teardown — not a
        // Rust panic, not our assertion. We retry only that signature; any real
        // assertion failure (a Rust panic in stderr) still fails fast on the
        // first attempt. On Linux this signature never occurs, so behavior there
        // is unchanged.
        for attempt in 1..=AFFINITY_DEFER_CHILD_MAX_ATTEMPTS {
            let output = command.output().expect("run affinity-defer child process");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if is_environmental_access_violation_crash(
                output.status.success(),
                output.status.code(),
                &stdout,
                &stderr,
            ) && attempt < AFFINITY_DEFER_CHILD_MAX_ATTEMPTS
            {
                eprintln!(
                    "note: retrying affinity-defer child (scenario={scenario}) after \
                     environmental STATUS_ACCESS_VIOLATION crash, attempt {attempt}"
                );
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }

            // Either the child succeeded, it failed for a real (non-signature)
            // reason, or retries are exhausted -- in every case, assert with the
            // same detailed diagnostics as before so a persistent real failure is
            // still surfaced with full scenario/stdout/stderr context.
            assert!(
                output.status.success(),
                "affinity-defer child failed (scenario={scenario}):\nstdout:\n{stdout}\n\
                 stderr:\n{stderr}"
            );
            assert!(
                stdout.contains(&format!("{AFFINITY_DEFER_MARKER}ok")),
                "affinity-defer child did not confirm scenario {scenario}:\n{stdout}"
            );
            return;
        }
    }

    /// (a) Adaptive (`=auto`) with an explicit non-numa-split
    /// affinity defers to the flat path and builds no persistent SPMD pool.
    #[test]
    fn auto_default_with_explicit_affinity_defers_to_flat() {
        run_affinity_defer_child("auto_off", "off", false);
        run_affinity_defer_child("auto_node", "node:0", false);
        run_affinity_defer_child("auto_compact", "compact", false);
    }

    /// (b) On (default or `=1`) keeps the persistent SPMD pool even when an
    /// explicit affinity is set -- the affinity defer must not apply.
    ///
    /// The spawned child (`run_affinity_defer_child`) is wrapped in a bounded
    /// retry that fires *only* on the known native Windows ARM64 environmental
    /// `STATUS_ACCESS_VIOLATION` (0xC0000005) crash during SPMD pool
    /// build/teardown; a real assertion failure still fails fast on the first
    /// attempt, and Linux behavior is unchanged.
    #[test]
    fn forced_persistent_pool_ignores_explicit_affinity() {
        run_affinity_defer_child("forced_off", "off", true);
    }

    /// (c) A malformed affinity value in the Auto-defer path still errors (the flat
    /// path's `plan_decode_affinity` validates it exactly as before).
    #[test]
    fn auto_default_malformed_affinity_still_errors_on_flat_path() {
        run_affinity_defer_child("auto_malformed", "not-a-real-mode", false);
    }

    /// Unit-locks the crash-signature classifier so the affinity-defer retry
    /// stays scoped to *exactly* the known environmental Windows ARM64
    /// `STATUS_ACCESS_VIOLATION`: a success, a real Rust panic, an emitted success
    /// marker, or any other exit code must all be treated as non-retryable.
    #[test]
    fn access_violation_crash_signature_classifier() {
        let marker_ok = format!("{AFFINITY_DEFER_MARKER}ok");

        // The exact environmental crash: unsuccessful, no marker, no panic,
        // access-violation exit code -> retryable.
        assert!(is_environmental_access_violation_crash(
            false,
            Some(STATUS_ACCESS_VIOLATION),
            "",
            "",
        ));

        // A successful run is never retryable, regardless of exit code.
        assert!(!is_environmental_access_violation_crash(
            true,
            Some(STATUS_ACCESS_VIOLATION),
            &marker_ok,
            "",
        ));

        // A real assertion failure (Rust panic in stderr) must fail fast.
        assert!(!is_environmental_access_violation_crash(
            false,
            Some(STATUS_ACCESS_VIOLATION),
            "",
            "thread 'main' panicked at src/foo.rs:1:1:\nassertion failed",
        ));

        // Any other non-success exit code is not the known signature.
        assert!(!is_environmental_access_violation_crash(
            false,
            Some(1),
            "",
            ""
        ));
        assert!(!is_environmental_access_violation_crash(
            false, None, "", ""
        ));

        // Even with the access-violation code, an emitted success marker means the
        // child actually completed its work -> not the environmental crash.
        assert!(!is_environmental_access_violation_crash(
            false,
            Some(STATUS_ACCESS_VIOLATION),
            &marker_ok,
            "",
        ));
    }

    /// M-based hybrid routing gate: with an otherwise-eligible int4 case
    /// (`accuracy_level == 4`) and the MLAS backend selected, `try_mlas_sqnbit`
    /// must fall back (`Ok(None)`) for `m` below the decode crossover -- decode
    /// keeps the hand path -- and serve MLAS (`Ok(Some(()))`) for `m` at/above
    /// it (prefill). This regression-locks the decode/prefill split. Uses the
    /// topology-derived default threshold; the host must have an MLAS SQNBit
    /// int4 kernel or the assertions are skipped.
    ///
    /// The decode half of the split is conditional on the host actually having a
    /// native int8 dot product behind the hand kernels
    /// (`hand_int8_decode_has_native_dot`). Where it does not, keeping decode on
    /// the hand path is the *wrong* answer -- measured 9.6x slower than letting
    /// MLAS take it -- so `Some(())` below the crossover is the expected result
    /// there, and this test asserts that instead.
    ///
    /// Weights are declared constant (`can_prepack = true`) because that is the
    /// only case where the ISA question is live: dynamic weights always keep the
    /// hand path (see
    /// `matmulnbits_accuracy4_dynamic_weight_decode_keeps_hand_path`).
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_gates_decode_by_m_threshold() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);

        // Skip when the host has no MLAS SQNBit int4 kernel for this shape.
        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Int8,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!("MLAS SQNBit int4 kernel unavailable; skipping M-gate test");
            return;
        }

        let kernel = accuracy4_kernel(k, n, block_size);
        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);

        let at = default_sqnbit_decode_min(available_parallelism());
        let below = at - 1;

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "mlas") };

        let call = |m: usize| {
            let a = pseudo(m * k, 0.8);
            let mut result = vec![0.0f32; m * n];
            kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales_t.view(),
                    None,
                    None,
                    true,
                    &a,
                    m,
                    None,
                    &mut result,
                )
                .unwrap()
        };

        let decode = call(below);
        let prefill = call(at);

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        if hand_int8_decode_has_native_dot() {
            assert_eq!(
                decode, None,
                "m={below} (< {at}) must fall back to the hand int4 path when the host has a native int8 dot product",
            );
        } else {
            assert_eq!(
                decode,
                Some(()),
                "m={below} (< {at}) must route to MLAS SQNBit when the hand int8 kernel has no native dot product to dispatch to",
            );
        }
        assert_eq!(prefill, Some(()), "m={at} must route to MLAS SQNBit",);
    }

    /// The decode-crossover short-circuit must be a claim about the host, not a
    /// constant: it may only fire where the hand int8 kernels have a real int8
    /// dot product to dispatch to.
    ///
    /// The x86_64 arm cross-checks against **CPUID directly** rather than
    /// against `selected_dot_kernel().uses_vnni_int4_direct()`, which is the
    /// implementation itself and would be a tautology. Restricted to the default
    /// selection: an explicit `ONNX_GENAI_CPU_DOT_KERNEL` override may
    /// legitimately choose a weaker kernel than the hardware supports, and the
    /// predicate must follow what actually executes, not what CPUID advertises.
    #[cfg(feature = "mlas")]
    #[test]
    fn hand_int8_decode_native_dot_matches_host_capability() {
        let has_native_dot = hand_int8_decode_has_native_dot();
        #[cfg(target_arch = "x86_64")]
        if std::env::var_os("ONNX_GENAI_CPU_DOT_KERNEL").is_none() {
            let host_has_vnni = std::arch::is_x86_feature_detected!("avx512vnni")
                || std::arch::is_x86_feature_detected!("avxvnni");
            assert_eq!(
                has_native_dot, host_has_vnni,
                "x86_64 needs AVX-VNNI/AVX-512-VNNI (vpdpbusd); the AVX2 vpmaddubsw/vpmaddwd \
                 emulation is not competitive with MLAS SQNBit CompInt8"
            );
        }
        #[cfg(target_arch = "aarch64")]
        assert!(
            has_native_dot,
            "NEON is baseline on aarch64, so the hand decode path always has a dot product"
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(
            !has_native_dot,
            "the scalar DotKernel has no dot product, so MLAS should take the node"
        );
    }

    /// The dynamic-weight decline: with non-constant `B` there is no
    /// session-lifetime cache to amortize MLAS's packing against, so decode must
    /// stay on the hand path even on a host with no native int8 dot product.
    /// Repacking a multi-megabyte weight per token is worse than any kernel.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_accuracy4_dynamic_weight_decode_keeps_hand_path() {
        let _probe = lock_dispatch_probe();
        let (k, n, block_size) = (128, 16, 32);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let scales_t = Owned::f32(&[n, 4], &scales);
        let b = Owned::u8(&[n, 4, 16], &packed);
        let kernel = accuracy4_kernel(k, n, block_size);

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "mlas") };

        let a = pseudo(k, 0.8);
        let mut result = vec![0.0f32; n];
        let routed = kernel
            .try_mlas_sqnbit(
                &b.view(),
                &scales_t.view(),
                None,
                None,
                false, // can_prepack = false: dynamic weights
                &a,
                1,
                None,
                &mut result,
            )
            .unwrap();

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            routed, None,
            "dynamic-weight accuracy-4 decode must keep the hand path: MLAS would repack the \
             whole weight on every call with no cache to amortize it against"
        );
    }

    /// Slow-hand-path decode routing: for `m == 1` with `bits == 4` but
    /// `accuracy_level != 4`, the hand path would dequantize the whole weight to
    /// f32 and run a dense GEMV. MLAS SQNBit (CompFp32) beats that, so
    /// `try_mlas_sqnbit` must route this small-`m` case to MLAS
    /// (`Ok(Some(()))`), unlike the fast `accuracy_level == 4` decode case which
    /// stays on the hand path. Skipped when the host lacks the MLAS kernel.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_serves_slow_dequant_decode() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, dq) = quantize(&weights_nk, n, k, block_size, false);

        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Fp32,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!("MLAS SQNBit int4 CompFp32 kernel unavailable; skipping slow-decode test");
            return;
        }

        // accuracy_level 0 => hand path would use the slow f32 dequant GEMV.
        let kernel = test_kernel(k, n, block_size);
        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "mlas") };

        let a = pseudo(k, 0.8);
        let mut result = vec![0.0f32; n];
        let served = kernel
            .try_mlas_sqnbit(
                &b.view(),
                &scales_t.view(),
                None,
                None,
                false,
                &a,
                1,
                None,
                &mut result,
            )
            .unwrap();

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            served,
            Some(()),
            "m=1 bits=4 accuracy_level=0 (slow hand dequant GEMV) must route to MLAS SQNBit",
        );
        // CompFp32 dequant is near-exact, so it must match the f32 reference.
        let expected = reference(&a, &dq, 1, k, n);
        mlas_close(&result, &expected, 2e-3, "slow-dequant m1 CompFp32");
    }

    /// Regression for the `accuracy_level = 0` slow-path bug: MLAS SQNBit is a
    /// specialized quantized kernel independent of the dense-f32 [`CpuBackend`]
    /// microkernel, so an `accuracy_level != 4` MatMulNBits must route to MLAS
    /// (CompFp32) even when the resolved backend is *not* MLAS -- the real
    /// default on an AVX2 host, where [`CpuBackend::auto_detect`] returns
    /// `SimdX86`. Before the fix the `auto_detect() != Mlas` gate dropped this
    /// case to the slow full-f32-dequant GEMV. `accuracy_level = 4` must be
    /// unaffected: its fast hand int8/int4 path stays selected (returns `None`)
    /// unless the whole backend is explicitly forced to MLAS. Skipped when the
    /// host lacks the MLAS kernel.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_routes_acclevel0_without_mlas_backend() {
        let _probe = lock_dispatch_probe();
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, dq) = quantize(&weights_nk, n, k, block_size, false);

        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Fp32,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!(
                "MLAS SQNBit int4 CompFp32 kernel unavailable; skipping acc0-default-backend test"
            );
            return;
        }

        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);
        let a = pseudo(k, 0.8);

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        // Force a non-MLAS backend to model the real-world default: MLAS SQNBit
        // routing for accuracy_level != 4 must not depend on the dense-GEMM
        // backend being MLAS.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "generic") };
        assert_ne!(
            crate::backend::CpuBackend::auto_detect(),
            crate::backend::CpuBackend::Mlas,
            "test precondition: backend must not resolve to MLAS",
        );

        let call = |kernel: &MatMulNBitsKernel| {
            let mut result = vec![0.0f32; n];
            let served = kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales_t.view(),
                    None,
                    None,
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap();
            (served, result)
        };

        let (acc0_served, acc0_result) = call(&test_kernel(k, n, block_size));
        let (acc4_served, _) = call(&accuracy4_kernel(k, n, block_size));

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            acc0_served,
            Some(()),
            "accuracy_level=0 must route to MLAS SQNBit even when the backend is not MLAS",
        );
        // CompFp32 dequant is near-exact, so it must match the f32 reference.
        let expected = reference(&a, &dq, 1, k, n);
        mlas_close(
            &acc0_result,
            &expected,
            2e-3,
            "acc0 default-backend CompFp32",
        );

        assert_eq!(
            acc4_served, None,
            "accuracy_level=4 decode must stay on the fast hand path when the backend is not MLAS",
        );
    }

    /// Before/after perf for int4 MatMulNBits: the existing hand-written VNNI
    /// path (`int4_matmul_m1` for M=1 decode, `int8_matmul` for M>1 prefill,
    /// both `accuracy_level=4`) vs the MLAS SQNBit CompInt8 path, at 1 and 8
    /// threads, for representative LLM shapes. Ignored by default; run with:
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas --release \
    ///     matmulnbits_mlas_perf -- --ignored --nocapture
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn matmulnbits_mlas_perf() {
        use std::time::Instant;

        fn time<F: FnMut() + Send>(threads: usize, mut run: F) -> f64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for _ in 0..20 {
                    run();
                }
                let iters = 200u32;
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                start.elapsed().as_secs_f64() * 1e6 / iters as f64
            })
        }

        let block_size = 32usize;
        let dot_kernel = selected_dot_kernel();
        for &(k, n) in &[(2048usize, 2048usize), (4096, 11008)] {
            let k_blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let weights_nk = pseudo(n * k, 0.3);
            let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);

            let kernel = accuracy4_kernel(k, n, block_size);
            let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
            let scales_t = Owned::f32(&[n, k_blocks], &scales);
            let int8_weight = kernel
                .prepack_int8_weight(&b.view(), &scales_t.view(), None)
                .unwrap();
            let int4_weight = PackedInt4Weight {
                values: packed_bytes.clone(),
                scales: scales.clone(),
            };
            let mlas_packed = mlas_sys::SQNBitPackedB::new(
                n,
                k,
                4,
                block_size,
                mlas_sys::SQNBitComputeType::Int8,
                &packed_bytes,
                &scales,
                None,
            )
            .expect("MLAS SQNBit int4 must be available for the perf probe");

            for &m in &[1usize, 32] {
                let a = pseudo(m * k, 0.8);
                for threads in [1usize, 8] {
                    let hand_us = if m == 1 {
                        time(threads, || {
                            let mut out = vec![0.0f32; n];
                            int4_matmul_m1(
                                &a,
                                &int4_weight,
                                &mut out,
                                k,
                                n,
                                block_size,
                                dot_kernel,
                            );
                        })
                    } else {
                        time(threads, || {
                            let mut out = vec![0.0f32; m * n];
                            int8_matmul(
                                &a,
                                &int8_weight,
                                &mut out,
                                m,
                                k,
                                n,
                                block_size,
                                dot_kernel,
                            );
                        })
                    };
                    let mlas_us = time(threads, || {
                        let mut out = vec![0.0f32; m * n];
                        mlas_sys::sqnbit_gemm(&mlas_packed, m, &a, None, &mut out, true);
                    });
                    eprintln!(
                        "int4 K={k} N={n} M={m} {threads}t: hand={hand_us:.1}us mlas={mlas_us:.1}us \
                         speedup={:.2}x",
                        hand_us / mlas_us
                    );
                }
            }
        }
    }

    /// Focused int4 M=1 GEMV micro-bench at Foundry Qwen3-0.6B block-128 decode
    /// shapes, reporting ns/call, GB/s (int4 weight bytes streamed = N*K/2) and
    /// GFLOP/s (2*N*K) for the direct int4 GEMV, the old native int8-prepack
    /// fallback, and MLAS SQNBit CompInt8. This is the Phase-1 gating probe for
    /// the ARM64 int4 GEMV decode kernel: direct/mlas >1 means slower than MLAS.
    /// Shapes are a probe fixture; production never hardcodes them. Run with:
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas --release \
    ///     int4_gemv_decode_microbench -- --ignored --nocapture
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn int4_gemv_decode_microbench() {
        use std::time::Instant;

        // Median-of-5 ns/call for one warm, L3-resident M=1 GEMV.
        fn median_ns<F: FnMut() + Send>(threads: usize, mut run: F) -> f64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for _ in 0..30 {
                    run();
                }
                let mut samples = [0.0f64; 5];
                for sample in samples.iter_mut() {
                    let iters = 300u32;
                    let start = Instant::now();
                    for _ in 0..iters {
                        run();
                    }
                    *sample = start.elapsed().as_secs_f64() * 1e9 / iters as f64;
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                samples[2]
            })
        }

        let block_size = 128usize;
        let dot_kernel = selected_dot_kernel();
        // (label, K, N) for exact int4 MatMulNBits shapes observed in the
        // Foundry qwen3-0.6b-generic-cpu-4/v4 graph (hidden=1024,
        // intermediate=3072, grouped q/k/v widths).
        let shapes: &[(&str, usize, usize)] = &[
            ("q_proj", 1024, 2048),
            ("kv/o_proj", 1024, 1024),
            ("o_proj", 2048, 1024),
            ("gate_proj", 1024, 3072),
            ("up_proj", 1024, 3072),
            ("down_proj", 3072, 1024),
        ];

        eprintln!(
            "int4 GEMV M=1 microbench (Foundry Qwen3-0.6B block-128 shapes), dot_kernel={dot_kernel:?}, median-of-5"
        );
        for &(label, k, n) in shapes {
            let weights_nk = pseudo(n * k, 0.3);
            let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);
            let int4_weight = PackedInt4Weight {
                values: packed_bytes.clone(),
                scales: scales.clone(),
            };
            let blocks = k.div_ceil(block_size);
            let kernel = accuracy4_kernel(k, n, block_size);
            let b = Owned::u8(&[n, blocks, block_size / 2], &packed_bytes);
            let scales_t = Owned::f32(&[n, blocks], &scales);
            let int8_weight = kernel
                .prepack_int8_weight(&b.view(), &scales_t.view(), None)
                .unwrap();
            let mlas_packed = mlas_sys::SQNBitPackedB::new(
                n,
                k,
                4,
                block_size,
                mlas_sys::SQNBitComputeType::Int8,
                &packed_bytes,
                &scales,
                None,
            )
            .expect("MLAS SQNBit int4 must be available for the perf probe");
            let a = pseudo(k, 0.8);
            let weight_bytes = (n as f64) * (k as f64) / 2.0;
            let flops = 2.0 * (n as f64) * (k as f64);

            for threads in [1usize, 32] {
                let direct_ns = median_ns(threads, || {
                    let mut out = vec![0.0f32; n];
                    int4_matmul_m1(&a, &int4_weight, &mut out, k, n, block_size, dot_kernel);
                });
                let old_native_ns = median_ns(threads, || {
                    let mut out = vec![0.0f32; n];
                    int8_matmul(&a, &int8_weight, &mut out, 1, k, n, block_size, dot_kernel);
                });
                let mlas_ns = median_ns(threads, || {
                    let mut out = vec![0.0f32; n];
                    mlas_sys::sqnbit_gemm(&mlas_packed, 1, &a, None, &mut out, true);
                });
                let direct_gbs = weight_bytes / direct_ns;
                let old_native_gbs = weight_bytes / old_native_ns;
                let mlas_gbs = weight_bytes / mlas_ns;
                let direct_gflops = flops / direct_ns;
                let old_native_gflops = flops / old_native_ns;
                let mlas_gflops = flops / mlas_ns;
                eprintln!(
                    "{label:10} K={k:6} N={n:6} {threads:2}t: \
                     direct {direct_ns:8.0}ns {direct_gbs:6.1}GB/s {direct_gflops:6.1}GF | \
                     old {old_native_ns:8.0}ns {old_native_gbs:6.1}GB/s {old_native_gflops:6.1}GF | \
                     mlas {mlas_ns:8.0}ns {mlas_gbs:6.1}GB/s {mlas_gflops:6.1}GF | \
                     ratio(direct/old)={:.2}x ratio(direct/mlas)={:.2}x",
                    direct_ns / old_native_ns,
                    direct_ns / mlas_ns
                );
            }
        }
    }

    /// Full M=1 decode-step probe at real 7B (Qwen2.5-Coder-7B) projection
    /// shapes: replays the exact per-token MatMulNBits op sequence (qkv, o,
    /// gate, up, down per layer, plus the lm_head) back-to-back inside one
    /// decode-pool residency, so it captures the *sequential per-op dispatch*
    /// overhead the isolated `matmulnbits_mlas_perf` probe misses. Compares the
    /// hand int4 GEMV path against MLAS SQNBit CompInt8 at the real decode
    /// thread count. Shapes come from the model (read once, listed here only as
    /// a probe fixture); production routing never hardcodes them.
    ///
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas --release \
    ///     matmulnbits_mlas_decode_step -- --ignored --nocapture
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn matmulnbits_mlas_decode_step() {
        use std::time::Instant;

        // (K, N, count-per-token) for one Qwen2.5-Coder-7B decode step.
        let layers = 28usize;
        let ops: &[(usize, usize, usize)] = &[
            (3584, 4608, layers),  // qkv_proj
            (3584, 3584, layers),  // o_proj
            (3584, 18944, layers), // gate_proj
            (3584, 18944, layers), // up_proj
            (18944, 3584, layers), // down_proj
            (3584, 152064, 1),     // lm_head
        ];
        let block_size = 32usize;
        let dot_kernel = selected_dot_kernel();

        struct Weights {
            k: usize,
            n: usize,
            int4: PackedInt4Weight,
            mlas_int8: mlas_sys::SQNBitPackedB,
            mlas_fp32: mlas_sys::SQNBitPackedB,
        }

        // Build one *distinct* weight per op instance so the step streams the
        // full ~3.5 GB of cold int4 weights from DRAM, exactly like the model
        // (reusing a handful of buffers would keep them L3-resident and report
        // fantasy bandwidth). Distinct scale seeds also defeat page dedup.
        let mut built: Vec<Weights> = Vec::new();
        let mut weight_bytes = 0u64;
        for (shape_index, &(k, n, count)) in ops.iter().enumerate() {
            for instance in 0..count {
                let seed = 0.3 + shape_index as f32 * 0.11 + instance as f32 * 0.001;
                let weights_nk = pseudo(n * k, seed);
                let (packed_bytes, scales, _zps, _dq) =
                    quantize(&weights_nk, n, k, block_size, false);
                let make = |comp| {
                    mlas_sys::SQNBitPackedB::new(
                        n,
                        k,
                        4,
                        block_size,
                        comp,
                        &packed_bytes,
                        &scales,
                        None,
                    )
                };
                let (Some(mlas_int8), Some(mlas_fp32)) = (
                    make(mlas_sys::SQNBitComputeType::Int8),
                    make(mlas_sys::SQNBitComputeType::Fp32),
                ) else {
                    eprintln!("MLAS SQNBit int4 kernel unavailable; skipping decode-step probe");
                    return;
                };
                weight_bytes += (n as u64) * (k as u64) / 2;
                built.push(Weights {
                    k,
                    n,
                    int4: PackedInt4Weight {
                        values: packed_bytes,
                        scales,
                    },
                    mlas_int8,
                    mlas_fp32,
                });
            }
        }

        let threads = configured_decode_threads()
            .or_else(|| default_decode_threads(available_parallelism()))
            .unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        let run_hand = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                int4_matmul_m1(&a, &w.int4, &mut out, w.k, w.n, block_size, dot_kernel);
            }
        };
        let run_mlas_int8 = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                mlas_sys::sqnbit_gemm(&w.mlas_int8, 1, &a, None, &mut out, true);
            }
        };
        let run_mlas_fp32 = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                mlas_sys::sqnbit_gemm(&w.mlas_fp32, 1, &a, None, &mut out, true);
            }
        };

        let step = |label: &str, run: &(dyn Fn() + Sync)| {
            pool.install(|| {
                for _ in 0..3 {
                    run();
                }
                let iters = 20u32;
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                let per_step = start.elapsed().as_secs_f64() / iters as f64;
                let gbs = weight_bytes as f64 / per_step / 1e9;
                eprintln!(
                    "decode-step {label}: {:.2} ms/step  {:.2} tok/s  {:.1} GB/s ({threads}t)",
                    per_step * 1e3,
                    1.0 / per_step,
                    gbs,
                );
            });
        };

        eprintln!("decode-step probe: {weight_bytes} weight bytes/token, {threads} decode threads");
        step("hand", &run_hand);
        step("mlas-int8", &run_mlas_int8);
        step("mlas-fp32", &run_mlas_fp32);
    }
}
